"""
Motion Analysis Module for SeedVR2

Computes optical flow between consecutive frames and classifies each frame
boundary to enable motion-aware temporal blending. Distinguishes between:
- STATIC: minimal motion, normal blending is fine
- MOTION: consistent motion (pan/zoom/object movement), needs motion compensation
- SCENE_CUT: chaotic/inconsistent flow, skip temporal blending

Uses Farneback optical flow (cv2) on low-res input frames for fast CPU-based
analysis. No GPU or extra model downloads required.
"""

import cv2
import numpy as np
import torch
from enum import Enum
from typing import List, Tuple, Optional


class MotionType(Enum):
    """Classification of motion between two consecutive frames."""
    STATIC = "static"
    MOTION = "motion"
    SCENE_CUT = "scene_cut"


class MotionBoundary:
    """Motion classification for a single frame boundary."""
    __slots__ = ['type', 'magnitude', 'consistency', 'flow']

    def __init__(self, type: MotionType, magnitude: float, consistency: float,
                 flow: Optional[np.ndarray] = None):
        self.type = type
        self.magnitude = magnitude
        self.consistency = consistency
        self.flow = flow  # (H, W, 2) flow field from frame i to frame i+1


def compute_motion_analysis(
    frames: torch.Tensor,
    sensitivity: float = 0.5,
    downscale: int = 96,
    return_flow: bool = False,
) -> List[MotionBoundary]:
    """
    Compute optical flow between consecutive frames and classify each boundary.

    Args:
        frames: Video tensor [T, H, W, C] in range [0, 1], RGB or RGBA.
        sensitivity: Float [0.0, 1.0] controlling classification thresholds.
            Lower = more conservative (fewer MOTION/SCENE_CUT classifications).
            Higher = more aggressive (more classifications).
        downscale: Max dimension for flow computation (smaller = faster).
        return_flow: If True, keep flow fields in MotionBoundary for warping.

    Returns:
        List of MotionBoundary with length T-1 (one per frame pair).
    """
    t = frames.shape[0]
    if t < 2:
        return []

    # Convert to grayscale numpy for optical flow
    # Use only RGB channels, ignore alpha
    rgb = frames[:, :, :, :3]  # [T, H, W, 3]

    # Downscale for speed - motion patterns are resolution-independent
    h, w = rgb.shape[1], rgb.shape[2]
    scale = downscale / max(h, w) if max(h, w) > downscale else 1.0
    if scale < 1.0:
        rgb = torch.nn.functional.interpolate(
            rgb.permute(0, 3, 1, 2),  # [T, 3, H, W]
            scale_factor=scale,
            mode='bilinear',
            align_corners=False,
        ).permute(0, 2, 3, 1)  # [T, H', W', 3]

    # Convert to grayscale [T, H', W']
    # Standard luminance weights
    gray = (rgb[:, :, :, 0] * 0.299 + rgb[:, :, :, 1] * 0.587 + rgb[:, :, :, 2] * 0.114)
    # cv2.calcOpticalFlowFarneback expects float32 in [0, 255] range (or uint8)
    # Our tensor is in [0, 1], so scale to [0, 255]
    gray_np = (gray.cpu().numpy() * 255.0).astype(np.float32)  # [T, H', W'] float32 in [0, 255]

    # Thresholds scaled by sensitivity (inverted: higher sensitivity = lower thresholds = more detection)
    # magnitude_threshold: mean pixel displacement to be considered "motion" (in original resolution pixels)
    # consistency_threshold: std/mean ratio to distinguish pan from scene cut
    # At sensitivity=0.0: magnitude=2.0, consistency=0.9 (conservative, few detections)
    # At sensitivity=1.0: magnitude=0.3, consistency=0.3 (aggressive, many detections)
    # At sensitivity=0.5 (default): magnitude=1.15, consistency=0.6 (catches most real motion)
    magnitude_threshold = 2.0 - sensitivity * 1.7  # range: 2.0 (low sens) → 0.3 (high sens)
    consistency_threshold = 0.9 - sensitivity * 0.6  # range: 0.9 (low sens) → 0.3 (high sens)

    boundaries: List[MotionBoundary] = []
    prev_gray = gray_np[0]

    for i in range(1, t):
        curr_gray = gray_np[i]

        # Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=2,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )  # [H', W', 2]

        # Compute flow magnitude per pixel
        flow_mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

        # Normalize by downscale factor to get pixel displacement in original resolution
        if scale < 1.0:
            flow_mag_orig = flow_mag / scale
        else:
            flow_mag_orig = flow_mag

        # Statistics
        mean_mag = float(np.mean(flow_mag_orig))
        std_mag = float(np.std(flow_mag_orig))

        # Consistency: ratio of std to mean (high = chaotic, low = uniform)
        # Clamp to avoid division by zero
        consistency = std_mag / max(mean_mag, 0.1)

        # Classify
        if mean_mag < magnitude_threshold:
            motion_type = MotionType.STATIC
        elif consistency > consistency_threshold:
            motion_type = MotionType.SCENE_CUT
        else:
            motion_type = MotionType.MOTION

        flow_to_keep = flow if return_flow else None
        boundaries.append(MotionBoundary(
            type=motion_type,
            magnitude=mean_mag,
            consistency=consistency,
            flow=flow_to_keep,
        ))

        prev_gray = curr_gray

    return boundaries


def get_flow_for_frame_range(
    frames: torch.Tensor,
    from_idx: int,
    to_idx: int,
    downscale: int = 96,
) -> Optional[np.ndarray]:
    """
    Compute cumulative optical flow from frame at from_idx to frame at to_idx.

    Used for warping frames during motion-compensated blending.
    Returns flow field that maps pixels from frame[to_idx] back to frame[from_idx].

    Args:
        frames: Video tensor [T, H, W, C] in range [0, 1].
        from_idx: Source frame index.
        to_idx: Target frame index (to_idx > from_idx).
        downscale: Max dimension for flow computation.

    Returns:
        Flow field [H_upscaled, W_upscaled, 2] or None if computation fails.
    """
    if from_idx >= to_idx:
        return None

    segment = frames[from_idx:to_idx + 1]  # [N, H, W, C]
    rgb = segment[:, :, :, :3]

    h, w = rgb.shape[1], rgb.shape[2]
    scale = downscale / max(h, w) if max(h, w) > downscale else 1.0
    if scale < 1.0:
        rgb = torch.nn.functional.interpolate(
            rgb.permute(0, 3, 1, 2),
            scale_factor=scale,
            mode='bilinear',
            align_corners=False,
        ).permute(0, 2, 3, 1)

    gray = (rgb[:, :, :, 0] * 0.299 + rgb[:, :, :, 1] * 0.587 + rgb[:, :, :, 2] * 0.114)
    # cv2.calcOpticalFlowFarneback expects float32 in [0, 255] range
    gray_np = (gray.cpu().numpy() * 255.0).astype(np.float32)

    # Accumulate flow across frames
    h_flow, w_flow = gray_np[0].shape
    cumulative_flow = np.zeros((h_flow, w_flow, 2), dtype=np.float32)

    for i in range(len(gray_np) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray_np[i], gray_np[i + 1], None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=2, poly_n=5, poly_sigma=1.2, flags=0,
        )
        cumulative_flow += flow

    # Upscale flow to match original frame size
    if scale < 1.0:
        # Interpolate flow and scale by 1/scale
        cumulative_flow = cv2.resize(
            cumulative_flow, (w, h), interpolation=cv2.INTER_LINEAR
        ) * (1.0 / scale)

    return cumulative_flow


def warp_frame_with_flow(
    frame: torch.Tensor,
    flow: np.ndarray,
) -> torch.Tensor:
    """
    Warp a single frame using an optical flow field.

    The flow field maps each pixel in the warped frame to a position in the
    original frame (forward flow: where each pixel came from).

    Args:
        frame: Single frame [H, W, C] in range [0, 1].
        flow: Flow field [H, W, 2] where flow[y, x] = (dx, dy) displacement.

    Returns:
        Warped frame [H, W, C] in range [0, 1].
    """
    h, w = frame.shape[0], frame.shape[1]

    # Create identity coordinate grid
    y_coords, x_coords = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing='ij',
    )

    # Add flow to get source coordinates
    # flow[y, x] = (dx, dy) means pixel at (x, y) came from (x+dx, y+dy)
    src_x = x_coords + flow[:, :, 0]
    src_y = y_coords + flow[:, :, 1]

    # Clamp to valid range
    src_x = np.clip(src_x, 0, w - 1)
    src_y = np.clip(src_y, 0, h - 1)

    # Convert to torch for grid_sample
    frame_np = frame.cpu().numpy()  # [H, W, C]

    # Use cv2.remap for each channel
    warped_channels = []
    for c in range(frame.shape[2]):
        warped_ch = cv2.remap(
            frame_np[:, :, c],
            src_x, src_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        warped_channels.append(warped_ch)

    warped_np = np.stack(warped_channels, axis=-1)  # [H, W, C]
    result = torch.from_numpy(warped_np).to(frame.device, dtype=frame.dtype)

    return result


def warp_frames_with_flow(
    frames: torch.Tensor,
    flow: np.ndarray,
) -> torch.Tensor:
    """
    Warp multiple frames using the same optical flow field.

    Args:
        frames: Frame batch [N, H, W, C] in range [0, 1].
        flow: Flow field [H, W, 2].

    Returns:
        Warped frames [N, H, W, C] in range [0, 1].
    """
    warped = []
    for i in range(frames.shape[0]):
        warped.append(warp_frame_with_flow(frames[i], flow))
    return torch.stack(warped, dim=0)


def should_skip_blending(
    boundaries: List[MotionBoundary],
    overlap_start: int,
    overlap: int,
) -> bool:
    """
    Check if blending should be skipped for an overlap region.

    Args:
        boundaries: List of MotionBoundary from compute_motion_analysis.
        overlap_start: Frame index where overlap starts (0-indexed in original video).
        overlap: Number of overlapping frames.

    Returns:
        True if any boundary in the overlap region is a scene cut.
    """
    if not boundaries:
        return False

    for i in range(overlap_start, min(overlap_start + overlap, len(boundaries))):
        if boundaries[i].type == MotionType.SCENE_CUT:
            return True
    return False


def has_motion_in_range(
    boundaries: List[MotionBoundary],
    start: int,
    end: int,
) -> bool:
    """
    Check if any MOTION or SCENE_CUT boundaries exist in a frame range.

    Args:
        boundaries: List of MotionBoundary.
        start: Start index (inclusive).
        end: End index (exclusive).

    Returns:
        True if any non-STATIC boundary exists in range.
    """
    if not boundaries:
        return False
    for i in range(start, min(end, len(boundaries))):
        if boundaries[i].type != MotionType.STATIC:
            return True
    return False


def get_scene_cut_indices(
    boundaries: List[MotionBoundary],
) -> List[int]:
    """
    Get frame indices where scene cuts occur.

    Args:
        boundaries: List of MotionBoundary.

    Returns:
        List of frame indices where a scene cut starts.
        E.g. [5] means frame 5 is the start of a new scene (cut between 4 and 5).
    """
    return [i + 1 for i, b in enumerate(boundaries) if b.type == MotionType.SCENE_CUT]


def get_motion_magnitudes_for_range(
    boundaries: List[MotionBoundary],
    start: int,
    end: int,
) -> List[float]:
    """
    Get motion magnitudes for frame boundaries in a range.

    Args:
        boundaries: List of MotionBoundary.
        start: Start index (inclusive).
        end: End index (exclusive).

    Returns:
        List of mean pixel displacement values for each boundary in range.
    """
    if not boundaries:
        return []
    result = []
    for i in range(start, min(end, len(boundaries))):
        result.append(boundaries[i].magnitude)
    return result


def compute_motion_adaptive_batches(
    boundaries: List[MotionBoundary],
    total_frames: int,
    min_batch_size: int = 5,
    max_batch_size: int = 49,
    motion_threshold: float = 0.3,
) -> List[Tuple[int, int]]:
    """
    Compute variable-sized batches based on motion intensity between frames.

    Creates smaller batches for fast motion/scene cuts (to reduce VAE temporal
    mixing and ghosting) and larger batches for static content (to maintain
    temporal consistency and processing efficiency).

    The VAE has a temporal receptive field of ~30 frames (encode) and ~40 frames
    (decode). With large batches, every output frame blends information from nearly
    the entire sequence, causing visible ghosting on scene cuts and fast motion.
    Variable batch sizing limits this mixing to semantically coherent regions.

    Args:
        boundaries: List of MotionBoundary from compute_motion_analysis.
        total_frames: Total number of frames in the video.
        min_batch_size: Minimum frames per batch (default: 5).
        max_batch_size: Maximum frames per batch (default: 49).
        motion_threshold: Motion intensity threshold [0.0-1.0]. Controls how
            aggressively to split on high-motion boundaries (non-scene-cut).
            0.0 = only split on scene cuts, 1.0 = split on any motion above
            the average. Default: 0.3 (split on scene cuts + very high motion).

    Returns:
        List of (start, end) tuples defining batch ranges. Each batch contains
        frames [start:end] (exclusive end). Batches are guaranteed to:
        - Cover all frames exactly once (no gaps, no overlaps)
        - Have size >= min_batch_size (except possibly the last batch)
        - Have size <= max_batch_size
        - Split on scene cuts and high-motion boundaries
    """
    if total_frames <= 0:
        return []

    if total_frames <= max_batch_size:
        return [(0, total_frames)]

    if not boundaries:
        # No motion data - fall back to uniform batching
        batches = []
        start = 0
        while start < total_frames:
            end = min(start + max_batch_size, total_frames)
            batches.append((start, end))
            start = end
        return batches

    # Compute per-segment motion intensity for adaptive batch sizing
    # Each candidate batch gets its own max size based on local motion, not global average.
    # This preserves large batches for static scenes while capping high-motion scenes.
    #
    # Exponential decay with max_batch_size as the base (static ceiling).
    # The decay rate is fixed — increasing max_batch_size raises the static ceiling
    # but does NOT make high-motion batches larger (they floor at min_batch_size).
    #
    # Helper: compute adaptive max batch for a frame range based on local motion
    def _segment_max_batch(seg_start: int, seg_end: int) -> int:
        """Compute max batch size for a segment based on its local motion intensity.

        Uses exponential decay with max_batch_size as the base (static ceiling).
        The decay rate (5.0) is fixed — independent of max_batch_size.

        Magnitudes are mean pixel displacement (original resolution).
        Scale (with max_batch_size=49, decay_rate=5.0):
          0px  → 49 (static)
          5px  → 18 (moderate motion)
          10px → 6  (fast motion)
          15px → 2  (very fast motion)
          20px → min_batch_size (extreme motion, floored)
        """
        seg_mags = []
        for i, b in enumerate(boundaries):
            # Boundary i is between frame i and frame i+1
            if seg_start <= i < seg_end - 1:
                seg_mags.append(b.magnitude)
        if not seg_mags:
            return max_batch_size  # No motion data → safe to use full size
        seg_avg = sum(seg_mags) / len(seg_mags)
        # Exponential decay: max_batch_size * exp(-seg_avg / 5)
        # Decay rate (5.0) is fixed — independent of max_batch_size.
        # Fast motion floors at min_batch_size.
        import math
        return max(min_batch_size, int(max_batch_size * math.exp(-seg_avg / 5.0)))

    magnitudes = [b.magnitude for b in boundaries]
    max_mag = max(magnitudes) if magnitudes else 1.0
    avg_mag = sum(magnitudes) / len(magnitudes) if magnitudes else 0.0

    if max_mag < 0.01:
        # All frames are essentially static - use maximum batch size
        batches = []
        start = 0
        while start < total_frames:
            end = min(start + max_batch_size, total_frames)
            batches.append((start, end))
            start = end
        return batches

    # Determine which boundaries should force a batch split using LOCAL SPIKE DETECTION.
    #
    # A real scene cut is a magnitude SPIKE compared to its immediate neighbors,
    # not just above a global threshold. This is far more robust than global
    # median+std (which gets inflated by the spikes themselves) and ignores
    # classifier noise (SCENE_CUT labels are unreliable at high sensitivity).
    #
    # For each boundary, we compare its magnitude to the local median of
    # surrounding boundaries (window of ±5 frames). If the ratio exceeds a
    # threshold, it's a scene cut.
    #
    # motion_threshold controls the spike ratio:
    #   threshold=0.0 → spike_ratio=5.0 (very conservative, only huge jumps)
    #   threshold=0.3 → spike_ratio=3.0 (default, good balance)
    #   threshold=0.5 → spike_ratio=2.0 (moderate, catches more transitions)
    #   threshold=1.0 → spike_ratio=1.2 (aggressive, almost any increase)
    #
    # Example: boundary at 55px with local median of 0.8px → ratio=68x → spike at any setting
    #          boundary at 7px with local median of 3px → ratio=2.3x → spike only at threshold>=0.5
    split_frames = set()  # Frame indices WHERE a new batch should START
    spike_ratio = max(1.2, 5.0 - motion_threshold * 6.0)  # range: 5.0 (conservative) → 1.2 (aggressive)
    local_window = 5  # Number of neighbor boundaries to consider on each side

    def _is_local_spike(idx: int) -> bool:
        """Check if boundary at idx is a local magnitude spike."""
        local_start = max(0, idx - local_window)
        local_end = min(len(magnitudes), idx + local_window + 1)
        # Get local magnitudes excluding the current boundary
        local_mags = []
        for j in range(local_start, local_end):
            if j != idx:
                local_mags.append(magnitudes[j])
        if not local_mags:
            return False
        local_median = sorted(local_mags)[len(local_mags) // 2]
        # For static regions (local_median near 0), use absolute threshold
        if local_median < 0.5:
            return magnitudes[idx] > spike_ratio
        # For motion regions, use ratio-based threshold
        return magnitudes[idx] > local_median * spike_ratio

    for i in range(len(boundaries)):
        if _is_local_spike(i):
            split_frames.add(i + 1)

    if not split_frames:
        # No scene-cut splits needed - use per-segment motion-adaptive batch sizes
        batches = []
        start = 0
        while start < total_frames:
            seg_max = _segment_max_batch(start, min(start + max_batch_size, total_frames))
            end = min(start + seg_max, total_frames)
            batches.append((start, end))
            start = end
        return batches

    # Build candidate batch boundaries from split points
    # Sort split points and build raw batches
    split_points = sorted(split_frames)
    raw_batches = []
    batch_start = 0

    for split_frame in split_points:
        if split_frame > batch_start:
            raw_batches.append((batch_start, split_frame))
        batch_start = split_frame

    # Add remaining frames
    if batch_start < total_frames:
        raw_batches.append((batch_start, total_frames))

    # Merge pass: ensure ALL batches meet minimum size
    # Greedy forward merge - undersized batches absorb their neighbors
    merged = []
    for batch in raw_batches:
        batch_size = batch[1] - batch[0]

        if batch_size < min_batch_size and merged:
            # Merge with previous batch
            prev = merged[-1]
            merged[-1] = (prev[0], batch[1])
        else:
            merged.append(batch)

    # If first batch is undersized, it's already handled (no previous to merge with)
    # Cap oversized batches at per-segment motion-adaptive max batch size
    result = []
    for start, end in merged:
        seg_max = _segment_max_batch(start, end)
        while end - start > seg_max:
            result.append((start, start + seg_max))
            start = start + seg_max
            # Recompute for remaining segment
            seg_max = _segment_max_batch(start, end)
        result.append((start, end))

    return result



