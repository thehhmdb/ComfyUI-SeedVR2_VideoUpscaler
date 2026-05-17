# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from math import ceil
from typing import Tuple
import math

# Global flag to disable temporal shifting in shifted-window layers.
# When True, shifted windows still shift spatially (H, W) but NOT temporally (T).
# This prevents cross-window temporal propagation (ghosting) while preserving
# spatial quality from the shifted-window method.
# Set via: from src.models.dit_7b.window import set_temporal_window_isolation; set_temporal_window_isolation(True)
_temporal_window_isolation = False

# Global cap on temporal window size (in frames).
# When > 0, limits the number of frames that attend to each other in each temporal window.
# Smaller values reduce ghosting on fast motion/scene cuts but also reduce temporal
# consistency on static regions. Default: 0 (no cap, model uses full temporal window).
# Recommended: 1-4 for fast motion, 5-8 for moderate motion, 0 for maximum temporal consistency.
_temporal_window_size_cap = 0


def set_temporal_window_isolation(enabled: bool):
    """Enable/disable temporal window isolation.

    When enabled, shifted-window layers use st=0 (no temporal shift), which
    prevents information from propagating across temporal windows through the
    shifted-window mechanism. This eliminates long-range temporal mixing
    (ghosting) at the cost of reduced temporal receptive field.

    Args:
        enabled: True to isolate temporal windows, False for normal behavior.
    """
    global _temporal_window_isolation
    _temporal_window_isolation = enabled


def get_temporal_window_isolation() -> bool:
    """Get current temporal window isolation state."""
    return _temporal_window_isolation


def set_temporal_window_size_cap(cap: int):
    """Set maximum temporal window size (in frames).

    Limits how many frames can attend to each other within a single temporal window.
    This directly controls the temporal receptive field of the model.

    Args:
        cap: Maximum frames per temporal window. 0 = no limit (model default).
             Values 1-4: minimal ghosting, reduced temporal consistency.
             Values 5-8: balanced ghosting prevention and temporal consistency.
             Values 9+: more temporal consistency, potential ghosting on fast motion.
    """
    global _temporal_window_size_cap
    _temporal_window_size_cap = max(0, cap)


def get_temporal_window_size_cap() -> int:
    """Get current temporal window size cap."""
    return _temporal_window_size_cap


def get_window_op(name: str):
    if name == "720pwin_by_size_bysize":
        return make_720Pwindows_bysize
    if name == "720pswin_by_size_bysize":
        return make_shifted_720Pwindows_bysize
    raise ValueError(f"Unknown windowing method: {name}")


# -------------------------------- Windowing -------------------------------- #
def make_720Pwindows_bysize(size: Tuple[int, int, int], num_windows: Tuple[int, int, int]):
    t, h, w = size
    resized_nt, resized_nh, resized_nw = num_windows
    #cal windows under 720p
    scale = math.sqrt((45 * 80) / (h * w))
    resized_h, resized_w = round(h * scale), round(w * scale)
    wh, ww = ceil(resized_h / resized_nh), ceil(resized_w / resized_nw)  # window size.
    wt = ceil(min(t, 30) / resized_nt)  # window size.
    # Apply temporal window size cap to limit temporal receptive field
    if _temporal_window_size_cap > 0:
        wt = min(wt, _temporal_window_size_cap)
    nt, nh, nw = ceil(t / wt), ceil(h / wh), ceil(w / ww)  # window size.
    return [
        (
            slice(it * wt, min((it + 1) * wt, t)),
            slice(ih * wh, min((ih + 1) * wh, h)),
            slice(iw * ww, min((iw + 1) * ww, w)),
        )
        for iw in range(nw)
        if min((iw + 1) * ww, w) > iw * ww
        for ih in range(nh)
        if min((ih + 1) * wh, h) > ih * wh
        for it in range(nt)
        if min((it + 1) * wt, t) > it * wt
    ]

def make_shifted_720Pwindows_bysize(size: Tuple[int, int, int], num_windows: Tuple[int, int, int]):
    t, h, w = size
    resized_nt, resized_nh, resized_nw = num_windows
    #cal windows under 720p
    scale = math.sqrt((45 * 80) / (h * w))
    resized_h, resized_w = round(h * scale), round(w * scale)
    wh, ww = ceil(resized_h / resized_nh), ceil(resized_w / resized_nw)  # window size.
    wt = ceil(min(t, 30) / resized_nt)  # window size.

    # Temporal shift: disabled when isolation is enabled to prevent cross-window
    # temporal propagation (ghosting). Spatial shifts remain active for quality.
    st, sh, sw = (  # shift size.
        0.0 if _temporal_window_isolation else (0.5 if wt < t else 0),
        0.5 if wh < h else 0,
        0.5 if ww < w else 0,
    )
    nt, nh, nw = ceil((t - st) / wt), ceil((h - sh) / wh), ceil((w - sw) / ww)  # window size.
    nt, nh, nw = (  # number of window.
        nt + 1 if st > 0 else ceil(t / wt),
        nh + 1 if sh > 0 else ceil(h / wh),
        nw + 1 if sw > 0 else ceil(w / ww),
    )
    return [
        (
            slice(max(int((it - st) * wt), 0), min(int((it - st + 1) * wt), t)),
            slice(max(int((ih - sh) * wh), 0), min(int((ih - sh + 1) * wh), h)),
            slice(max(int((iw - sw) * ww), 0), min(int((iw - sw + 1) * ww), w)),
        )
        for iw in range(nw)
        if min(int((iw - sw + 1) * ww), w) > max(int((iw - sw) * ww), 0)
        for ih in range(nh)
        if min(int((ih - sh + 1) * wh), h) > max(int((ih - sh) * wh), 0)
        for it in range(nt)
        if min(int((it - st + 1) * wt), t) > max(int((it - st) * wt), 0)
    ]
