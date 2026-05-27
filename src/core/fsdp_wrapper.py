"""
Pipeline parallel wrapper for SeedVR2 multi-GPU VRAM pooling.

In pipeline mode, model layers are split across GPUs. Activations stream
between GPUs as they pass through each stage. This pools VRAM for larger
resolutions by ensuring activations only live on one GPU at a time.

Architecture:
- DiT transformer blocks are partitioned across N GPUs
- GPU 0 processes blocks 0..K/N, sends activations to GPU 1
- GPU 1 processes blocks K/N..2K/N, sends activations to GPU 2
- etc.
- Each GPU holds ~1/N of model weights + activations for its stage only

Compared to FSDP:
- FSDP: Shards params, but activations are full-size on every GPU (bad for inference)
- Pipeline: Splits layers, activations stream between GPUs (good for inference)

Usage:
    python inference_cli.py video.mp4 --cuda_device 0,1 --fsdp --resolution 1440

Main Functions:
- partition_model_layers: Split model layers across GPUs
- get_stage_layers: Get layer indices for a given GPU stage
- run_pipeline_forward: Execute pipeline-parallel forward pass through DiT
"""

from typing import Any, Dict, List, Optional, Tuple

import torch


def is_fsdp_available() -> bool:
    """Check if pipeline parallel mode is available (always True)."""
    return True


class PipelineDiTWrapper(torch.nn.Module):
    """
    Wraps a NaDiT model and replaces the forward pass with pipeline-parallel execution.

    Instead of loading all blocks onto one GPU, this wrapper splits the transformer
    blocks across multiple GPUs. Activations stream between GPUs via .to(device).

    Usage:
        wrapper = PipelineDiTWrapper(dit_model, device_list=[cuda:0, cuda:1])
        output = wrapper(vid, txt, vid_shape, txt_shape, timestep, disable_cache=True)
    """

    def __init__(
        self,
        dit_model: torch.nn.Module,
        device_list: List[torch.device],
        debug: Optional[Any] = None,
        spatial_tile_size: int = 0,
        spatial_tile_overlap: int = 0,
        auto_tile_size: bool = False,
    ):
        super().__init__()
        self.dit_model = dit_model
        self.device_list = device_list
        self.debug = debug

        # Spatial tiling config
        from src.core.spatial_tiling import SpatialTilingConfig
        self.spatial_config = SpatialTilingConfig(
            tile_size=spatial_tile_size,
            overlap=spatial_tile_overlap,
            auto_tile_size=auto_tile_size,
        )

        num_gpus = len(device_list)
        total_blocks = len(dit_model.blocks)
        self.partitions = partition_model_layers(total_blocks, num_gpus)

        # Unwrap BlockSwap's forward wrappers - pipeline mode manages
        # device placement itself (blocks and I/O go to different GPUs, not a
        # single main_device). BlockSwap's wrapper would move everything to
        # model.main_device which breaks pipeline parallelism.
        # BlockSwap's initial CPU offloading of blocks is still useful - it
        # moves swapped blocks to CPU at startup, and pipeline will move them
        # to the right GPU when needed via .to(device).
        blocks_unwrapped = 0
        io_unwrapped = 0
        for block in dit_model.blocks:
            if hasattr(block, '_original_forward'):
                block.forward = block._original_forward
                delattr(block, '_original_forward')
                blocks_unwrapped += 1

        # Also unwrap I/O components (vid_in, txt_in, emb_in, vid_out, etc.)
        for name, module in dit_model.named_children():
            if name != "blocks" and hasattr(module, '_original_forward'):
                module.forward = module._original_forward
                delattr(module, '_original_forward')
                io_unwrapped += 1

        if debug:
            debug.log(f"PipelineDiTWrapper: {total_blocks} blocks across {num_gpus} GPUs",
                      category="pipeline", force=True)
            for i, parts in enumerate(self.partitions):
                debug.log(f"  GPU {i} ({device_list[i]}): blocks {parts}",
                          category="pipeline", force=True)
            if blocks_unwrapped > 0:
                debug.log(f"PipelineDiTWrapper: Unwrapped {blocks_unwrapped} BlockSwap block forwards "
                          f"(pipeline manages device placement)",
                          category="pipeline", force=True)
            if io_unwrapped > 0:
                debug.log(f"PipelineDiTWrapper: Unwrapped {io_unwrapped} BlockSwap I/O forwards "
                          f"(pipeline manages device placement)",
                          category="pipeline", force=True)
            if self.spatial_config.enabled:
                debug.log(f"PipelineDiTWrapper: Spatial tiling enabled "
                          f"(tile={self.spatial_config.tile_size}px, "
                          f"overlap={self.spatial_config.overlap}px, "
                          f"stride={self.spatial_config.effective_stride}px)",
                          category="tiling", force=True)
            bs_config = getattr(dit_model, '_block_swap_config', None)
            if bs_config:
                debug.log(f"PipelineDiTWrapper: BlockSwap CPU offloading active "
                          f"({bs_config.get('blocks_swapped', 0)} blocks offloaded to "
                          f"{bs_config.get('offload_device', 'cpu')} at startup)",
                          category="pipeline", force=True)

    def _pipeline_forward_single(
        self,
        vid: torch.Tensor,
        txt: torch.Tensor,
        vid_shape: torch.Tensor,
        txt_shape: torch.Tensor,
        timestep: torch.Tensor,
        disable_cache: bool = True,
    ):
        """
        Pipeline-parallel forward pass for a single tile.

        Splits DiT transformer blocks across GPUs. Activations stream between GPUs
        via .to(device) transfers. Each GPU holds only its partition of blocks.
        """
        from src.common.cache import Cache

        dit = self.dit_model
        device_list = self.device_list
        partitions = self.partitions
        num_gpus = len(device_list)
        total_blocks = len(dit.blocks)

        # --- Bypass BlockSwap protection and move ALL blocks to CPU ---
        # BlockSwap's .to() protection blocks device changes when active.
        # Pipeline needs full control over block placement, so we bypass it.
        # Find the innermost model where BlockSwap stores the bypass flag.
        _model = dit
        while hasattr(_model, "dit_model"):
            _model = _model.dit_model
        _had_bypass = getattr(_model, "_blockswap_bypass_protection", False)
        _model._blockswap_bypass_protection = True

        # Move all blocks to CPU — pipeline loads one at a time.
        for block_idx in range(total_blocks):
            dit.blocks[block_idx] = dit.blocks[block_idx].to("cpu")

        # Restore BlockSwap protection state (pipeline doesn't use BlockSwap's
        # device management, so protection is irrelevant during forward pass).
        if not _had_bypass:
            _model._blockswap_bypass_protection = False

        # Force VRAM release after moving blocks to CPU
        import gc
        for d in device_list:
            torch.cuda.synchronize(d)
        gc.collect()
        for d in device_list:
            torch.cuda.empty_cache()

        # Log VRAM state after block CPU offload
        if self.debug:
            for d in device_list:
                allocated = torch.cuda.memory_allocated(d) / (1024**3)
                free = torch.cuda.memory_reserved(d) / (1024**3)
                total = torch.cuda.get_device_properties(d).total_memory / (1024**3)
                self.debug.log(
                    f"  After block CPU offload - GPU {d.index}: {allocated:.1f}GB allocated / {free:.1f}GB reserved / {total:.1f}GB total",
                    category="pipeline", force=True)

        # --- Stage 0: Input layers + first partition of blocks on GPU 0 ---
        current_device = device_list[0]

        # --- Move input tensors through CPU to prevent GPU 1 → GPU 0 double-allocation ---
        # Input tensors (vid, txt) come from GPU 1 (where latent_1 was loaded).
        # Direct GPU 1 → GPU 0 transfer keeps both copies alive, fragmenting memory.
        # Moving through CPU forces the GPU 1 copy to release before loading GPU 0.
        vid = vid.to("cpu", non_blocking=True)
        txt = txt.to("cpu", non_blocking=True)
        timestep = timestep.to("cpu", non_blocking=True)
        vid_shape = vid_shape.to("cpu", non_blocking=True)
        txt_shape = txt_shape.to("cpu", non_blocking=True)

        # Free GPU 1 memory before loading GPU 0
        if len(device_list) > 1:
            torch.cuda.synchronize(device_list[1])
            torch.cuda.empty_cache()

        # Move input layers to GPU 0
        dit.vid_in = dit.vid_in.to(current_device)
        dit.txt_in = dit.txt_in.to(current_device)
        dit.emb_in = dit.emb_in.to(current_device)

        vid = vid.to(current_device)
        txt = txt.to(current_device)
        timestep = timestep.to(current_device)
        vid_shape = vid_shape.to(current_device)
        txt_shape = txt_shape.to(current_device)

        # VRAM log after input tensor transfer
        if self.debug:
            allocated = torch.cuda.memory_allocated(current_device) / (1024**3)
            self.debug.log(f"  After input tensor transfer - GPU {current_device.index}: {allocated:.1f}GB allocated",
                          category="pipeline", force=True)

        # Text input (repeat if needed, like the original forward)
        if txt_shape.size(-1) == 1 and getattr(dit, "need_txt_repeat", False):
            from src.models.dit_7b import na
            txt, txt_shape = na.repeat(txt, txt_shape, "l c -> t l c", t=vid_shape[:, 0])
        txt = dit.txt_in(txt)

        # Video patch embedding
        vid, vid_shape = dit.vid_in(vid, vid_shape)

        # Timestep embedding
        emb = dit.emb_in(timestep, device=vid.device, dtype=vid.dtype)

        # VRAM log after input layer processing
        if self.debug:
            allocated = torch.cuda.memory_allocated(current_device) / (1024**3)
            self.debug.log(f"  After input layer processing - GPU {current_device.index}: {allocated:.1f}GB allocated",
                          category="pipeline", force=True)

        # --- Move input layers to CPU immediately after use ---
        # Input layers (vid_in, txt_in, emb_in) are ~1-2 GB each and linger on GPU 0,
        # fragmenting memory alongside block 0's forward pass. Moving to CPU frees
        # ~3-4 GB before the block loop starts.
        dit.vid_in = dit.vid_in.to("cpu")
        dit.txt_in = dit.txt_in.to("cpu")
        dit.emb_in = dit.emb_in.to("cpu")
        timestep = timestep.to("cpu")

        # Create cache
        cache = Cache(disable=disable_cache)

        # --- Force VRAM cleanup before block loop starts ---
        # After moving input layers to CPU, synchronize and free orphaned allocations
        # to ensure maximum contiguous VRAM is available for block 0's forward pass.
        import gc
        torch.cuda.synchronize(device_list[0])
        gc.collect()
        torch.cuda.empty_cache()

        # --- True pipeline parallelism: 1 block per stage, alternating GPUs ---
        # Each block gets its own stage, alternating between GPUs.
        # GPU 0 processes blocks 0, 2, 4, 6, ... (18 blocks total, but 1 at a time)
        # GPU 1 processes blocks 1, 3, 5, 7, ... (18 blocks total, but 1 at a time)
        # After each block, activations are offloaded to CPU to force VRAM release.
        # This ensures GPU 0 only ever has ~1 block's activations at a time,
        # preventing the 16+ GB accumulation that causes fragmentation OOMs.
        num_stages = total_blocks  # 36 stages, 1 block each
        partitions_fine = partition_model_layers(total_blocks, num_stages)
        device_list_extended = [device_list[i % num_gpus] for i in range(num_stages)]

        if self.debug:
            self.debug.log(f"Pipeline: {total_blocks} blocks split into {num_stages} stages "
                          f"({len(partitions_fine[0])} blocks/stage), alternating across {num_gpus} GPUs",
                          category="pipeline", force=True)
            for i, parts in enumerate(partitions_fine):
                gpu_idx = i % num_gpus
                self.debug.log(f"  Stage {i} ({device_list_extended[i]} / GPU {gpu_idx}): blocks {parts}",
                              category="pipeline")

        block_count = 0
        activations_on_cpu = False  # Track whether activations are on CPU

        for stage_idx in range(num_stages):
            current_device = device_list_extended[stage_idx]

            for block_idx in partitions_fine[stage_idx]:
                # --- Reload activations to current GPU (if on CPU) ---
                # Activations are kept on CPU between blocks to prevent fragmentation.
                # Reload to the correct GPU before each block's forward pass.
                if activations_on_cpu:
                    vid = vid.to(current_device, non_blocking=True)
                    txt = txt.to(current_device, non_blocking=True)
                    emb = emb.to(current_device, non_blocking=True)
                    vid_shape = vid_shape.to(current_device, non_blocking=True)
                    txt_shape = txt_shape.to(current_device, non_blocking=True)
                    activations_on_cpu = False

                # Load block to GPU
                dit.blocks[block_idx] = dit.blocks[block_idx].to(current_device)

                # Forward pass
                vid, txt, vid_shape, txt_shape = dit.blocks[block_idx](vid, txt, vid_shape, txt_shape, emb, cache)

                # Move block to CPU immediately
                dit.blocks[block_idx] = dit.blocks[block_idx].to("cpu")

                # --- Per-block CPU offload: forces immediate VRAM release ---
                vid = vid.detach().to("cpu", non_blocking=True)
                txt = txt.detach().to("cpu", non_blocking=True)
                emb = emb.detach().to("cpu", non_blocking=True)
                vid_shape = vid_shape.to("cpu", non_blocking=True)
                txt_shape = txt_shape.to("cpu", non_blocking=True)
                activations_on_cpu = True

                # Batched VRAM cleanup every 4 blocks instead of every block.
                # gc.collect() + synchronize + empty_cache are expensive CPU-side
                # operations (full GC sweep, GPU sync, allocator walk). Running
                # them every block adds ~1-2s overhead across 36 blocks.
                # Per-block CPU offload already frees VRAM; cleanup is just defragmentation.
                block_count += 1
                if block_count % 4 == 0:
                    torch.cuda.synchronize(current_device)
                    gc.collect()
                    torch.cuda.empty_cache()
                    cache.cache.clear()
                else:
                    cache.cache.clear()

        # Final cleanup after all blocks processed
        if block_count % 4 != 0:
            torch.cuda.synchronize(current_device)
            gc.collect()
            torch.cuda.empty_cache()
            cache.cache.clear()

        # --- Final stage: Output layers on last GPU ---
        current_device = device_list[-1]

        # Move activations back to GPU from CPU (last block offloaded to CPU)
        vid = vid.to(current_device, non_blocking=True)
        txt = txt.to(current_device, non_blocking=True)
        emb = emb.to(current_device, non_blocking=True)
        vid_shape = vid_shape.to(current_device, non_blocking=True)
        txt_shape = txt_shape.to(current_device, non_blocking=True)

        # Output normalization (if exists - 3B model has it, 7B may not)
        vid_out_norm = getattr(dit, "vid_out_norm", None)
        vid_out_ada = getattr(dit, "vid_out_ada", None)

        if vid_out_norm is not None:
            vid_out_norm = vid_out_norm.to(current_device)
            vid = vid_out_norm(vid)
            if vid_out_ada is not None:
                vid_out_ada = vid_out_ada.to(current_device)
                vid = vid_out_ada(vid, emb=emb, layer="out", mode="in",
                                  vid_shape=vid_shape, txt_shape=txt_shape, cache=cache)

        # Video patch un-embedding
        dit.vid_out = dit.vid_out.to(current_device)
        vid, vid_shape = dit.vid_out(vid, vid_shape, cache=cache)

        # --- Cleanup: Move blocks back to CPU after forward pass ---
        # Blocks stay on GPU during forward, but need to be freed between diffusion steps
        # to avoid OOM when activations accumulate
        for block_idx in range(total_blocks):
            dit.blocks[block_idx] = dit.blocks[block_idx].to("cpu")
        dit.vid_in = dit.vid_in.to("cpu")
        dit.txt_in = dit.txt_in.to("cpu")
        dit.emb_in = dit.emb_in.to("cpu")
        dit.vid_out = dit.vid_out.to("cpu")
        vid_out_norm_m = getattr(dit, "vid_out_norm", None)
        if vid_out_norm_m is not None:
            dit.vid_out_norm = vid_out_norm_m.to("cpu")
        vid_out_ada_m = getattr(dit, "vid_out_ada", None)
        if vid_out_ada_m is not None:
            dit.vid_out_ada = vid_out_ada_m.to("cpu")
        torch.cuda.empty_cache()

        # Return same output type as the original NaDiT (works for both 3B and 7B)
        try:
            from src.models.dit_7b.nadit import NaDiTOutput
        except ImportError:
            from src.models.dit_3b.nadit import NaDiTOutput
        return NaDiTOutput(vid_sample=vid)

    def forward(
        self,
        vid: torch.Tensor,
        txt: torch.Tensor,
        vid_shape: torch.Tensor,
        txt_shape: torch.Tensor,
        timestep: torch.Tensor,
        disable_cache: bool = True,
    ):
        """
        Forward pass with optional spatial tiling + pipeline parallelism.

        If spatial tiling is enabled, the video is split into overlapping spatial
        tiles. Each tile is processed through the full pipeline-parallel forward,
        then stitched back together with overlap blending.

        If spatial tiling is disabled, delegates directly to _pipeline_forward_single.
        """
        # Auto tile size: probe VRAM and compute optimal tile size for this batch
        effective_tile_size = self.spatial_config.tile_size
        if self.spatial_config.auto_tile_size and self.spatial_config.tile_size == 0:
            from src.core.spatial_tiling import estimate_optimal_tile_size

            t_val = int(vid_shape[0, 0].item())
            h_val = int(vid_shape[0, 1].item())
            w_val = int(vid_shape[0, 2].item())

            # Probe free VRAM on the primary GPU
            free_vram, _ = torch.cuda.mem_get_info(self.device_list[0])

            # Extract vid_dim from model (vid_in.proj is Linear(in, vid_dim))
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
            from src.core.spatial_tiling import (
                generate_tile_grid,
                extract_tile_from_flat,
                stitch_tiles_to_flat,
            )

            # vid is (L, C) flattened, vid_shape is (batch, 3) = [T, H, W] in patch coords
            device = vid.device
            dtype = vid.dtype
            c_val = vid.shape[1]

            t_val = int(vid_shape[0, 0].item())
            h_val = int(vid_shape[0, 1].item())
            w_val = int(vid_shape[0, 2].item())

            # Use effective tile size (auto-computed or user-specified)
            active_tile_size = effective_tile_size if effective_tile_size > 0 else self.spatial_config.tile_size
            active_overlap = self.spatial_config.overlap
            active_stride = max(active_tile_size - active_overlap, 1)

            # Generate tile grid (in patch coords)
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

            # Process each tile through pipeline forward
            tile_outputs = []
            tile_slices = []

            for i, (h_slice, w_slice) in enumerate(tile_grid):
                # Extract tile from flattened video tensor
                tile_vid, tile_vid_shape = extract_tile_from_flat(
                    vid, vid_shape, h_slice, w_slice,
                )

                # Run pipeline forward on this tile
                tile_out = self._pipeline_forward_single(
                    tile_vid, txt, tile_vid_shape, txt_shape,
                    timestep, disable_cache=disable_cache,
                )

                # Extract vid_sample from output
                if hasattr(tile_out, 'vid_sample'):
                    tile_out = tile_out.vid_sample

                tile_outputs.append(tile_out)
                tile_slices.append((h_slice, w_slice))

                # Free intermediate tile tensors (lightweight - just Python ref drops)
                del tile_vid, tile_vid_shape, tile_out

                # Per-tile cache cleanup: free VRAM every N tiles to prevent accumulation
                # across GPUs during tiling loop
                _interval = max(4, len(tile_grid) // 4) if len(tile_grid) > 4 else 4
                if (i + 1) % _interval == 0 or (i + 1) == len(tile_grid):
                    for d in self.device_list:
                        torch.cuda.empty_cache()

                if self.debug:
                    self.debug.log(f"  Tile {i + 1}/{len(tile_grid)} complete",
                                  category="tiling", force=True)

            # Final cleanup after all tiles processed
            import gc
            gc.collect()
            for d in self.device_list:
                torch.cuda.synchronize(d)
                torch.cuda.empty_cache()

            # Get patch size from model for output expansion
            patch_size = self.dit_model.vid_in.patch_size  # (pt, ph, pw)

            # Stitch tiles back together with overlap blending
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
        else:
            # No tiling - run single pipeline forward
            return self._pipeline_forward_single(
                vid, txt, vid_shape, txt_shape, timestep, disable_cache=disable_cache,
            )


def partition_model_layers(total_layers: int, num_gpus: int) -> List[List[int]]:
    """
    Partition model layer indices across GPUs as evenly as possible.

    Args:
        total_layers: Total number of layers in the model
        num_gpus: Number of GPUs to partition across

    Returns:
        List of lists, where each inner list contains layer indices for that GPU stage.
        E.g., partition_model_layers(28, 2) -> [[0..13], [14..27]]
    """
    base_per_gpu = total_layers // num_gpus
    remainder = total_layers % num_gpus

    partitions: List[List[int]] = []
    start = 0

    for gpu_idx in range(num_gpus):
        count = base_per_gpu + (1 if gpu_idx < remainder else 0)
        partitions.append(list(range(start, start + count)))
        start += count

    return partitions


def get_stage_layers(stage: int, total_layers: int, num_gpus: int) -> List[int]:
    """
    Get the layer indices assigned to a specific GPU stage.

    Args:
        stage: GPU stage index (0 to num_gpus-1)
        total_layers: Total number of layers in the model
        num_gpus: Number of GPUs in the pipeline

    Returns:
        List of layer indices for this stage
    """
    partitions = partition_model_layers(total_layers, num_gpus)
    return partitions[stage] if stage < len(partitions) else []


def get_stage_info(stage: int, num_gpus: int, total_layers: int) -> Dict[str, Any]:
    """
    Get human-readable info about a pipeline stage.

    Args:
        stage: GPU stage index
        num_gpus: Total number of GPUs
        total_layers: Total number of layers

    Returns:
        Dict with stage, layers, layer_range string, and fraction
    """
    layers = get_stage_layers(stage, num_gpus, total_layers)
    return {
        "stage": stage,
        "layers": layers,
        "layer_range": f"{layers[0]}-{layers[-1]}" if layers else "none",
        "num_layers": len(layers),
        "fraction": f"{len(layers) / total_layers * 100:.0f}%" if total_layers else "0%",
    }


def run_pipeline_forward(
    dit_model: torch.nn.Module,
    vid: torch.Tensor,
    txt: torch.Tensor,
    vid_shape: Tuple[int, ...],
    txt_shape: Tuple[int, ...],
    timestep: torch.Tensor,
    device_list: List[torch.device],
    debug: Optional[Any] = None,
) -> torch.Tensor:
    """
    Execute a pipeline-parallel forward pass through the DiT model.

    Splits DiT transformer blocks across GPUs. Activations stream between GPUs
    via .to(device) transfers. Single process, no torch.distributed needed.

    Pipeline stages:
    1. GPU 0: vid_in + txt_in + emb_in + blocks[0:mid]
    2. GPU 1: blocks[mid:] + vid_out_norm + vid_out_ada + vid_out

    Args:
        dit_model: NaDiT model (must have vid_in, txt_in, emb_in, blocks, vid_out)
        vid: Input video latent tensor
        txt: Text embedding tensor
        vid_shape: Video shape tuple for patch in/out
        txt_shape: Text shape tuple for patch in/out
        timestep: Timestep tensor
        device_list: List of GPU devices for pipeline stages
        debug: Optional debug logger

    Returns:
        Output video tensor from the DiT model
    """
    num_gpus = len(device_list)
    total_blocks = len(dit_model.blocks)
    partitions = partition_model_layers(total_blocks, num_gpus)

    if debug:
        debug.log(f"Pipeline forward: {total_blocks} blocks across {num_gpus} GPUs",
                  category="pipeline", force=True)
        for i, parts in enumerate(partitions):
            debug.log(f"  GPU {i} ({device_list[i]}): blocks {parts}",
                      category="pipeline", force=True)

    # --- Stage 0: Input layers + first partition of blocks ---
    current_device = device_list[0]

    # Move input layers to GPU 0
    dit_model.vid_in = dit_model.vid_in.to(current_device)
    dit_model.txt_in = dit_model.txt_in.to(current_device)
    dit_model.emb_in = dit_model.emb_in.to(current_device)

    vid = vid.to(current_device)
    txt = txt.to(current_device)
    timestep = timestep.to(current_device)

    # Create cache for this forward pass
    from src.common.cache import Cache
    cache = Cache(disable=False)

    # Text embedding
    if isinstance(dit_model.txt_in, torch.nn.ModuleList):
        for layer in dit_model.txt_in:
            txt = layer(txt)
    else:
        txt = dit_model.txt_in(txt)

    # Video patch embedding
    vid, vid_shape = dit_model.vid_in(vid, vid_shape, cache=cache)

    # Timestep embedding
    emb = dit_model.emb_in(timestep, device=vid.device, dtype=vid.dtype)

    # Process blocks for stage 0
    for block_idx in partitions[0]:
        block = dit_model.blocks[block_idx].to(current_device)
        vid, txt, vid_shape, txt_shape = block(vid, txt, vid_shape, txt_shape, emb, cache)

    # --- Intermediate stages: Process blocks on each GPU ---
    for stage_idx in range(1, num_gpus):
        current_device = device_list[stage_idx]

        # Move activations to next GPU
        vid = vid.to(current_device)
        txt = txt.to(current_device)
        emb = emb.to(current_device)

        for block_idx in partitions[stage_idx]:
            block = dit_model.blocks[block_idx].to(current_device)
            vid, txt, vid_shape, txt_shape = block(vid, txt, vid_shape, txt_shape, emb, cache)

    # --- Final stage: Output layers on last GPU ---
    current_device = device_list[-1]

    # Output normalization (if exists - 3B model has it, 7B may not)
    vid_out_norm = getattr(dit_model, "vid_out_norm", None)
    vid_out_ada = getattr(dit_model, "vid_out_ada", None)

    if vid_out_norm is not None:
        vid_out_norm = vid_out_norm.to(current_device)
        vid = vid_out_norm(vid)
        if vid_out_ada is not None:
            vid_out_ada = vid_out_ada.to(current_device)
            vid = vid_out_ada(vid, emb=emb, layer="out", mode="in",
                              vid_shape=vid_shape, txt_shape=txt_shape, cache=cache)

    # Video patch un-embedding
    dit_model.vid_out = dit_model.vid_out.to(current_device)
    vid, vid_shape = dit_model.vid_out(vid, vid_shape, cache=cache)

    return vid

