"""
Spatial tiling for DiT upscaling - processes video in smaller spatial tiles
to reduce peak VRAM usage during the forward pass.

Each tile is processed independently through the full DiT forward pass,
then stitched back together with smooth overlap blending.

Architecture:
- Video input is 2D (L, C) flattened sequence, with vid_shape (batch, 3) = [T, H, W]
  in patch-coordinate units
- Tiling splits H and W dimensions into overlapping tiles
- Each tile runs through the complete DiT forward (patch embed -> blocks -> patch un-embed)
- Tile outputs are blended using quadratic weight maps at overlap boundaries
- Text embeddings are shared across all tiles (same prompt context)

Quality preservation:
- Overlap region is blended between adjacent tiles (quadratic fade)
- Central region of each tile sees the same windowed attention context as full-resolution
- Blending at boundaries prevents visible seams

Usage:
    from src.core.spatial_tiling import SpatialTilingConfig
    config = SpatialTilingConfig(tile_size=64, overlap=32)
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import torch


@dataclass
class SpatialTilingConfig:
    """Configuration for spatial tiling.

    Tile sizes are in *patch-coordinate units* (the same units as vid_shape).
    For a model with patch_size=2, a latent of 180x180 pixels has vid_shape
    of ~90x90 in patch coords. So tile_size=64 means ~128px tiles in pixel space.
    """
    tile_size: int = 0           # Tile size in patch coords (0 = disabled)
    overlap: int = 0             # Overlap in patch coords between adjacent tiles
    min_tile_size: int = 16      # Minimum tile size (below this, tiling is disabled)
    auto_tile_size: bool = False # Auto-select tile size per-batch based on VRAM

    @property
    def enabled(self) -> bool:
        return self.tile_size > 0 and self.tile_size >= self.min_tile_size

    @property
    def effective_stride(self) -> int:
        """Stride between tile starts (tile_size - overlap)."""
        return max(self.tile_size - self.overlap, 1)


class SpatialTilingDiTWrapper(torch.nn.Module):
    """
    Wraps any DiT model and adds spatial tiling to the forward pass.

    Splits the video into overlapping spatial tiles, processes each tile
    through the full DiT forward pass independently, then stitches the
    outputs back together with quadratic overlap blending.

    This is the non-FSDP equivalent of the spatial tiling logic in
    PipelineDiTWrapper.forward(). Used when running on a single GPU
    without pipeline parallelism.

    Usage:
        wrapper = SpatialTilingDiTWrapper(dit_model, device, tile_size=64, overlap=32)
        output = wrapper(vid, txt, vid_shape, txt_shape, timestep)
    """

    def __init__(
        self,
        dit_model: torch.nn.Module,
        device: torch.device,
        debug: Optional[Any] = None,
        spatial_tile_size: int = 0,
        spatial_tile_overlap: int = 0,
        auto_tile_size: bool = False,
    ):
        super().__init__()
        self.dit_model = dit_model
        self.device = device
        self.debug = debug

        self.spatial_config = SpatialTilingConfig(
            tile_size=spatial_tile_size,
            overlap=spatial_tile_overlap,
            auto_tile_size=auto_tile_size,
        )

        if debug:
            if self.spatial_config.enabled:
                debug.log(f"SpatialTilingDiTWrapper: Spatial tiling enabled "
                          f"(tile={self.spatial_config.tile_size}px, "
                          f"overlap={self.spatial_config.overlap}px, "
                          f"stride={self.spatial_config.effective_stride}px)",
                          category="tiling", force=True)
            elif auto_tile_size:
                debug.log(f"SpatialTilingDiTWrapper: Auto tile size enabled "
                          f"(will probe VRAM per-batch)",
                          category="tiling", force=True)

    def forward(
        self,
        vid: torch.Tensor,
        txt: torch.Tensor,
        vid_shape: torch.Tensor,
        txt_shape: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs,
    ):
        """
        Forward pass with optional spatial tiling.

        If spatial tiling is enabled, the video is split into overlapping spatial
        tiles. Each tile is processed through the full DiT forward, then stitched
        back together with overlap blending.

        If spatial tiling is disabled, delegates directly to the wrapped DiT model.
        """
        # Auto tile size: probe VRAM and compute optimal tile size for this batch
        effective_tile_size = self.spatial_config.tile_size
        if self.spatial_config.auto_tile_size and self.spatial_config.tile_size == 0:
            t_val = int(vid_shape[0, 0].item())
            h_val = int(vid_shape[0, 1].item())
            w_val = int(vid_shape[0, 2].item())

            free_vram, _ = torch.cuda.mem_get_info(self.device)

            vid_dim = self.dit_model.vid_in.proj.out_features
            dtype_bytes = vid.element_size()
            patch_size = self.dit_model.vid_in.patch_size  # (pt, ph, pw)

            effective_tile_size = estimate_optimal_tile_size(
                vid_dim=vid_dim,
                t=t_val,
                h=h_val,
                w=w_val,
                dtype_bytes=dtype_bytes,
                free_vram_bytes=free_vram,
                min_tile_size=self.spatial_config.min_tile_size,
                patch_size=tuple(patch_size),
            )

            if self.debug:
                free_gb = free_vram / (1024**3)
                if effective_tile_size == 0:
                    self.debug.log(
                        f"Auto tile: disabled (full {t_val}x{h_val}x{w_val} fits in {free_gb:.1f}GB free VRAM)",
                        category="tiling", force=True,
                    )
                else:
                    self.debug.log(
                        f"Auto tile: {effective_tile_size} (batch T={t_val}, H={h_val}, W={w_val}, "
                        f"free={free_gb:.1f}GB, vid_dim={vid_dim})",
                        category="tiling", force=True,
                    )

        if self.spatial_config.enabled or (self.spatial_config.auto_tile_size and effective_tile_size > 0):
            return self._forward_tiled(
                vid, txt, vid_shape, txt_shape, timestep,
                effective_tile_size, **kwargs,
            )
        else:
            # No tiling - run single forward
            return self.dit_model(
                vid=vid,
                txt=txt,
                vid_shape=vid_shape,
                txt_shape=txt_shape,
                timestep=timestep,
                **kwargs,
            )

    def _forward_tiled(
        self,
        vid: torch.Tensor,
        txt: torch.Tensor,
        vid_shape: torch.Tensor,
        txt_shape: torch.Tensor,
        timestep: torch.Tensor,
        effective_tile_size: int,
        **kwargs,
    ):
        """Forward pass with spatial tiling applied."""
        from src.core.spatial_tiling import (
            generate_tile_grid,
            extract_tile_from_flat,
            stitch_tiles_to_flat,
        )

        device = vid.device
        dtype = vid.dtype
        c_val = vid.shape[1]

        t_val = int(vid_shape[0, 0].item())
        h_val = int(vid_shape[0, 1].item())
        w_val = int(vid_shape[0, 2].item())

        active_tile_size = effective_tile_size if effective_tile_size > 0 else self.spatial_config.tile_size
        active_overlap = self.spatial_config.overlap
        active_stride = max(active_tile_size - active_overlap, 1)

        tile_grid = generate_tile_grid(
            h_val, w_val,
            active_tile_size,
            active_overlap,
        )

        if self.debug:
            self.debug.log(
                f"Spatial tiling: {h_val}x{w_val} (patch coords) -> {len(tile_grid)} tiles "
                f"(tile={active_tile_size}, overlap={active_overlap}, stride={active_stride})",
                category="tiling", force=True,
            )

        tile_outputs = []
        tile_slices = []

        # Per-tile cleanup: free intermediates after each tile to prevent VRAM accumulation
        # Empty cache every N tiles to reclaim fragmented memory
        _empty_cache_interval = max(4, len(tile_grid) // 4) if len(tile_grid) > 4 else 4

        for i, (h_slice, w_slice) in enumerate(tile_grid):
            tile_vid, tile_vid_shape = extract_tile_from_flat(
                vid, vid_shape, h_slice, w_slice,
            )

            tile_out = self.dit_model(
                vid=tile_vid,
                txt=txt,
                vid_shape=tile_vid_shape,
                txt_shape=txt_shape,
                timestep=timestep,
                **kwargs,
            )

            if hasattr(tile_out, 'vid_sample'):
                tile_out = tile_out.vid_sample

            tile_outputs.append(tile_out)
            tile_slices.append((h_slice, w_slice))

            # Free tile intermediates immediately after each tile
            del tile_vid, tile_vid_shape, tile_out

            # Periodic cache cleanup to prevent VRAM fragmentation
            if (i + 1) % _empty_cache_interval == 0 or (i + 1) == len(tile_grid):
                torch.cuda.empty_cache()

            if self.debug:
                self.debug.log(f"  Tile {i + 1}/{len(tile_grid)} complete",
                              category="tiling", force=True)

        # Final cleanup after all tiles processed
        import gc
        gc.collect()
        torch.cuda.synchronize(device)

        patch_size = self.dit_model.vid_in.patch_size  # (pt, ph, pw)

        result = stitch_tiles_to_flat(
            tile_outputs, tile_slices,
            t_val, h_val, w_val, c_val,
            active_overlap,
            patch_size,
            device, dtype,
        )

        if self.debug:
            self.debug.log(
                f"Spatial tiling: stitched {len(tile_outputs)} tiles -> {tuple(result.shape)}",
                category="tiling", force=True,
            )

        try:
            from src.models.dit_7b.nadit import NaDiTOutput
        except ImportError:
            from src.models.dit_3b.nadit import NaDiTOutput
        return NaDiTOutput(vid_sample=result)


def estimate_optimal_tile_size(
    vid_dim: int,
    t: int,
    h: int,
    w: int,
    dtype_bytes: int,
    free_vram_bytes: int,
    min_tile_size: int = 32,
    patch_size: Tuple[int, int, int] = (1, 2, 2),
) -> int:
    """
    Estimate the optimal spatial tile size based on available VRAM and batch dimensions.

    Uses an analytical VRAM cost model to find the largest tile size that fits
    within available VRAM. Per-tile peak VRAM scales linearly with token count
    (T x tile_h x tile_w), so larger batches need smaller tiles.

    Args:
        vid_dim: Model hidden dimension (e.g., 3072 for 7B, 2560 for 3B)
        t: Temporal dimension (number of frames in batch) in patch coords
        h: Height in patch coords
        w: Width in patch coords
        dtype_bytes: Bytes per element (2 for fp16/bf16, 4 for fp32)
        free_vram_bytes: Free VRAM on the target GPU in bytes
        min_tile_size: Minimum tile size (below this, tiling is forced off)
        patch_size: Model patch size (pt, ph, pw), e.g. (1, 2, 2)

    Returns:
        Optimal tile size in patch coords, or 0 if full image fits without tiling.
    """
    # Safety margin - use 80% of free VRAM to avoid edge-case OOM
    safety_margin = 0.8
    usable_vram = free_vram_bytes * safety_margin

    # Estimate per-block weight size (one block loaded to GPU at a time in pipeline mode)
    # A transformer block has ~2 linear layers (QKV+out) + MLP (gate+up+down)
    # Each linear: vid_dim * vid_dim * dtype_bytes
    # Rough estimate: 4-5 linear layers per block
    per_block_weights = vid_dim * vid_dim * dtype_bytes * 5

    # Fixed overhead: I/O layers (vid_in, txt_in, emb_in, vid_out) + text embeddings
    # vid_in: ~vid_in_channels * patch_size * vid_dim
    # txt_in + emb_in + vid_out: ~3 * vid_dim^2
    # Text embeddings: ~77 * txt_dim (small)
    fixed_overhead = vid_dim * vid_dim * dtype_bytes * 4 + 1e9  # ~4 linear + 1GB buffer

    # Available VRAM for activations after weights and fixed overhead
    available_for_activations = usable_vram - per_block_weights - fixed_overhead

    if available_for_activations <= 0:
        # Very constrained - use minimum tile size
        return min_tile_size

    # Activation bytes per token: vid_dim * dtype_bytes * activation_multiplier
    # Multiplier ~5 accounts for: attention intermediates (Q,K,V,O), MLP (gate,up,down),
    # residual connections, and normalization buffers
    activation_multiplier = 5
    activation_bytes_per_token = vid_dim * dtype_bytes * activation_multiplier

    # Check if full image fits without tiling
    total_tokens = t * h * w
    full_image_activation_vram = total_tokens * activation_bytes_per_token

    if full_image_activation_vram <= available_for_activations:
        return 0  # No tiling needed

    # Binary search for max tile_size where per-tile VRAM fits
    # Tokens per tile = t * tile_h * tile_w
    # We want: t * tile_size^2 * activation_bytes_per_token <= available_for_activations
    # tile_size <= sqrt(available / (t * activation_bytes_per_token))

    tokens_per_spatial_pixel = t
    vram_per_spatial_pixel = tokens_per_spatial_pixel * activation_bytes_per_token

    if vram_per_spatial_pixel <= 0:
        return min_tile_size

    max_tile_size = int((available_for_activations / vram_per_spatial_pixel) ** 0.5)

    # Clamp to valid range
    max_tile_size = max(min_tile_size, max_tile_size)
    max_tile_size = min(max_tile_size, max(h, w))  # Can't be larger than latent dims

    # Round down to a multiple of patch_size for spatial dimensions.
    # The model's patch embedding requires tile dimensions to be divisible
    # by the patch size (e.g., patch_size=[1,2,2] means spatial dims must be even).
    ph, pw = patch_size[1], patch_size[2]
    import math
    lcm_val = ph  # Assume ph == pw for simplicity (common case)
    if ph != pw:
        lcm_val = ph * pw // math.gcd(ph, pw)
    if lcm_val > 1:
        max_tile_size = (max_tile_size // lcm_val) * lcm_val

    # If rounding makes tile too small, return 0 (no tiling) rather than a broken tile
    if max_tile_size < min_tile_size:
        return 0

    return max_tile_size


def generate_tile_slices(
    size: int,
    tile_size: int,
    overlap: int,
) -> List[slice]:
    """
    Generate slice objects for tiling a single dimension.

    Args:
        size: Total size of the dimension (in patch coords)
        tile_size: Size of each tile (in patch coords)
        overlap: Overlap between adjacent tiles (in patch coords)

    Returns:
        List of slice objects covering the entire dimension
    """
    if size <= tile_size:
        return [slice(0, size)]

    stride = max(tile_size - overlap, 1)
    slices = []
    start = 0

    while start < size:
        end = min(start + tile_size, size)
        slices.append(slice(start, end))
        start += stride

    # Ensure the last tile covers the end
    if slices[-1].stop < size:
        slices.append(slice(max(0, size - tile_size), size))

    # Ensure the last tile has full tile_size width for model context.
    # Narrow last tiles produce degraded quality because the model's
    # attention windows don't have enough spatial context.
    if len(slices) > 1:
        last = slices[-1]
        last_width = last.stop - last.start
        if last_width < tile_size:
            new_start = max(0, last.stop - tile_size)
            slices[-1] = slice(new_start, last.stop)

    # Remove tiles that are completely contained within another tile
    # (can happen when last tile is extended and swallows a predecessor)
    if len(slices) > 1:
        filtered = [slices[0]]
        for s in slices[1:]:
            prev = filtered[-1]
            # Skip if s completely contains prev (replace prev with s)
            if s.start <= prev.start and s.stop >= prev.stop:
                filtered[-1] = s
            # Skip if s is completely contained by prev
            elif s.start >= prev.start and s.stop <= prev.stop:
                continue
            else:
                filtered.append(s)
        slices = filtered

    return slices


def generate_tile_grid(
    h: int,
    w: int,
    tile_size: int,
    overlap: int,
) -> List[Tuple[slice, slice]]:
    """
    Generate a grid of (h_slice, w_slice) pairs covering the spatial dimensions.

    Args:
        h: Height in patch coords
        w: Width in patch coords
        tile_size: Tile size in patch coords
        overlap: Overlap in patch coords

    Returns:
        List of (h_slice, w_slice) tuples
    """
    h_slices = generate_tile_slices(h, tile_size, overlap)
    w_slices = generate_tile_slices(w, tile_size, overlap)

    grid = []
    for hs in h_slices:
        for ws in w_slices:
            grid.append((hs, ws))

    return grid


def extract_tile_from_flat(
    vid: torch.Tensor,
    vid_shape: torch.Tensor,
    h_slice: slice,
    w_slice: slice,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract a spatial tile from a flattened (L, C) video tensor.

    vid is (L, C) where L = T * H * W (patch coords). Tokens are laid out
    in row-major order: T varies slowest, then H, then W fastest.

    vid_shape is (batch, 3) = [T, H, W] in patch coords.

    Args:
        vid: Flattened video tensor (L, C)
        vid_shape: Shape metadata (batch, 3) = [T, H, W] in patch coords
        h_slice: Height slice in patch coords
        w_slice: Width slice in patch coords

    Returns:
        Tuple of (tiled_vid, tiled_vid_shape)
    """
    t_val = int(vid_shape[0, 0].item())
    h_val = int(vid_shape[0, 1].item())
    w_val = int(vid_shape[0, 2].item())
    c_val = vid.shape[1]

    # Unflatten to (T, H, W, C)
    vid_4d = vid.view(t_val, h_val, w_val, c_val)

    # Slice spatial dims
    tiled_4d = vid_4d[:, h_slice, w_slice, :]

    # Re-flatten to (L_tile, C)
    tiled_vid = tiled_4d.flatten(0, 2)

    # Update vid_shape
    h_new = h_slice.stop - h_slice.start
    w_new = w_slice.stop - w_slice.start
    tiled_vid_shape = torch.tensor(
        [[t_val, h_new, w_new]],
        dtype=vid_shape.dtype,
        device=vid_shape.device,
    )

    return tiled_vid, tiled_vid_shape


def stitch_tiles_to_flat(
    tile_outputs: List[torch.Tensor],
    tile_slices: List[Tuple[slice, slice]],
    t_val: int,
    h_val: int,
    w_val: int,
    c_val: int,
    overlap: int,
    patch_size: Tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Stitch tiled outputs back together with overlap blending, returning flat (L, C).

    Each tile output is (L_tile_out, C_out) where L_tile_out = T*pt * tile_h*ph * tile_w*pw
    (after NaPatchOut rearrange expands spatial dims by patch_size).

    We unflatten to (T*pt, tile_h*ph, tile_w*pw, C_out), blend in expanded space,
    then re-flatten.

    Args:
        tile_outputs: List of output tensors from each tile (L_tile_out, C_out)
        tile_slices: List of (h_slice, w_slice) for each tile (in patch coords)
        t_val: Temporal dimension (patch coords)
        h_val: Full height (patch coords)
        w_val: Full width (patch coords)
        c_val: Input channel dimension (hidden dim, NOT output channels)
        overlap: Overlap size in patch coords
        patch_size: (pt, ph, pw) patch size from the model
        device: CUDA device
        dtype: Tensor dtype

    Returns:
        Stitched output tensor (L_out, C_out) where L_out = T*pt * H*ph * W*pw
    """
    pt, ph, pw = patch_size

    # Key insight: vid_shape passed to the DiT forward is already in *expanded* coordinates
    # (NaPatchIn expands T,H,W by patch_size internally, updating vid_shape).
    # NaPatchOut reverses the patch rearrangement but the output shape equals the input shape.
    # So output has the SAME spatial dims as input — no additional expansion needed.
    # t_val, h_val, w_val are already the output dimensions.

    # Determine output channels from first tile
    c_out = tile_outputs[0].shape[1]

    result = torch.zeros((t_val, h_val, w_val, c_out), device=device, dtype=dtype)
    weight_accum = torch.zeros((t_val, h_val, w_val, 1), device=device, dtype=dtype)

    for tile_out, (h_slice, w_slice) in zip(tile_outputs, tile_slices):
        h_start = h_slice.start
        h_end = h_slice.stop
        w_start = w_slice.start
        w_end = w_slice.stop

        tile_h = h_end - h_start
        tile_w = w_end - w_start

        # Unflatten tile output to (T, tile_H, tile_W, C_out)
        # Output shape matches input shape (no expansion, already in expanded coords)
        tile_4d = tile_out.view(t_val, tile_h, tile_w, c_out)

        # Create weight map for this tile - only blend edges that overlap
        # with adjacent tiles. Outer boundaries (first/last tile in each dim)
        # should NOT be faded, otherwise normalization by tiny weights creates artifacts.
        weight_h = torch.ones(tile_h, device=device, dtype=dtype)
        weight_w = torch.ones(tile_w, device=device, dtype=dtype)

        if overlap > 0:
            fade_len_h = min(overlap, tile_h)
            fade_len_w = min(overlap, tile_w)

            # Vectorized quadratic fade: (i/overlap)^2 for top/left, ((i+1)/overlap)^2 for bottom/right
            # Replaces Python for-loops that iterated overlap times per edge.
            fade_up = (torch.arange(fade_len_h, device=device, dtype=dtype) / overlap) ** 2
            fade_down = (torch.arange(1, fade_len_h + 1, device=device, dtype=dtype) / overlap) ** 2
            fade_left = (torch.arange(fade_len_w, device=device, dtype=dtype) / overlap) ** 2
            fade_right = (torch.arange(1, fade_len_w + 1, device=device, dtype=dtype) / overlap) ** 2

            # Fade top edge only if this is NOT the first tile in H (h_start > 0)
            if h_start > 0:
                weight_h[:fade_len_h] = fade_up
            # Fade bottom edge only if this is NOT the last tile in H (h_end < h_val)
            if h_end < h_val:
                weight_h[-fade_len_h:] = fade_down.flip(0)
            # Fade left edge only if this is NOT the first tile in W (w_start > 0)
            if w_start > 0:
                weight_w[:fade_len_w] = fade_left
            # Fade right edge only if this is NOT the last tile in W (w_end < w_val)
            if w_end < w_val:
                weight_w[-fade_len_w:] = fade_right.flip(0)

        weight_map = weight_h.unsqueeze(1) * weight_w.unsqueeze(0)
        weight_map = weight_map.unsqueeze(-1)

        # Accumulate weighted output
        result[:, h_start:h_end, w_start:w_end, :] += tile_4d * weight_map
        weight_accum[:, h_start:h_end, w_start:w_end, :] += weight_map

    # Normalize by accumulated weights
    weight_accum.clamp_(min=1e-6)
    result = result / weight_accum

    # Re-flatten to (L_out, C_out)
    return result.flatten(0, 2)
