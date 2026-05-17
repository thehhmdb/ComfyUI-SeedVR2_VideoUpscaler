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
from typing import List, Optional, Tuple
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

    @property
    def enabled(self) -> bool:
        return self.tile_size > 0 and self.tile_size >= self.min_tile_size

    @property
    def effective_stride(self) -> int:
        """Stride between tile starts (tile_size - overlap)."""
        return max(self.tile_size - self.overlap, 1)


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
            # Fade top edge only if this is NOT the first tile in H (h_start > 0)
            if h_start > 0:
                for i in range(min(overlap, tile_h)):
                    t_norm = i / overlap
                    weight_h[i] = t_norm * t_norm
            # Fade bottom edge only if this is NOT the last tile in H (h_end < h_val)
            if h_end < h_val:
                for i in range(min(overlap, tile_h)):
                    t_norm = (i + 1) / overlap
                    weight_h[tile_h - 1 - i] = t_norm * t_norm
            # Fade left edge only if this is NOT the first tile in W (w_start > 0)
            if w_start > 0:
                for i in range(min(overlap, tile_w)):
                    t_norm = i / overlap
                    weight_w[i] = t_norm * t_norm
            # Fade right edge only if this is NOT the last tile in W (w_end < w_val)
            if w_end < w_val:
                for i in range(min(overlap, tile_w)):
                    t_norm = (i + 1) / overlap
                    weight_w[tile_w - 1 - i] = t_norm * t_norm

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
