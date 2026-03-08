"""
Optical Flow Utilities
=======================
Lightweight dense optical flow computation and frame warping
using OpenCV's Farneback algorithm.

This module adds ZERO learnable parameters — pure OpenCV math.
Used by the Temporal Stabilizer for motion-compensated EMA blending.
"""

import cv2
import numpy as np


def compute_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
) -> np.ndarray:
    """Compute dense optical flow between two grayscale frames.

    Uses Farneback's algorithm via OpenCV. Tuned for CCTV footage
    with slow-to-moderate motion.

    Args:
        prev_gray: Previous frame (H, W), uint8 grayscale.
        curr_gray: Current frame (H, W), uint8 grayscale.
        pyr_scale: Pyramid scale factor.
        levels: Number of pyramid levels.
        winsize: Averaging window size (larger = smoother).
        iterations: Algo iterations per level.
        poly_n: Pixel neighbourhood for polynomial expansion.
        poly_sigma: Gaussian std for polynomial smoothing.

    Returns:
        Flow field (H, W, 2) — dx, dy per pixel in float32.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        flow=None,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=0,
    )
    return flow


def warp_frame(
    frame: np.ndarray,
    flow: np.ndarray,
) -> np.ndarray:
    """Warp a frame using a dense optical flow field.

    Maps pixels from *frame* according to *flow* to produce a
    motion-compensated version aligned to the target frame.

    Args:
        frame: BGR uint8 image (H, W, 3) to warp.
        flow: Dense flow (H, W, 2) from ``compute_flow``.

    Returns:
        Warped BGR uint8 image (H, W, 3).
    """
    h, w = flow.shape[:2]

    # Build remap coordinates: dst(x,y) = src(x + flow_x, y + flow_y)
    map_x = np.arange(w, dtype=np.float32)[None, :] + flow[:, :, 0]
    map_y = np.arange(h, dtype=np.float32)[:, None] + flow[:, :, 1]

    warped = cv2.remap(
        frame,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return warped
