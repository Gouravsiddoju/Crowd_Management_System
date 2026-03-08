"""
Temporal Stabilizer — Stage 0
===============================
EMA frame fusion with optional optical-flow warping to remove
temporal flicker before the enhancement pipeline runs.

This stage is applied BEFORE any neural network processing.
It smooths raw pixel values across consecutive frames so that
the downstream models see stable input, reducing brightness
flickering and noise jitter.

Key properties:
* Zero learnable parameters
* Works on BGR uint8 numpy arrays
* Optional flow-based alignment for moving cameras / PTZ
"""

import cv2
import numpy as np

from vision_enhance_v2_5.temporal.ema_buffer import EMABuffer
from vision_enhance_v2_5.temporal.optical_flow import compute_flow, warp_frame


class TemporalStabilizer:
    """Pre-enhancement temporal frame stabilizer.

    Smooths raw frames using exponential moving average (EMA).
    Optionally aligns previous frames via dense optical flow
    before blending, which improves quality for PTZ cameras.

    Args:
        alpha: EMA blending weight for current frame (0.1–0.5).
            Higher = less smoothing, faster response.
        use_flow: If True, warp previous EMA to current geometry
            before blending (better for PTZ / panning cameras).
    """

    def __init__(self, alpha: float = 0.2, use_flow: bool = False):
        self.alpha = alpha
        self.use_flow = use_flow
        self._ema_buffer = EMABuffer(alpha=alpha)
        self._prev_gray: np.ndarray = None

    def stabilize(self, frame: np.ndarray) -> np.ndarray:
        """Stabilize a single frame.

        Args:
            frame: BGR uint8 numpy array (H, W, 3).

        Returns:
            Temporally smoothed BGR uint8 numpy array (H, W, 3).
        """
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.use_flow and self._prev_gray is not None and self._ema_buffer.initialized:
            # Compute flow from previous EMA frame to current
            flow = compute_flow(self._prev_gray, curr_gray)
            # Warp the current EMA estimate to align with current frame
            prev_ema = self._ema_buffer.get().astype(np.uint8)
            warped_ema = warp_frame(prev_ema, flow)
            # Blend: use warped EMA as the "previous" and current as new
            blended = (
                self.alpha * frame.astype(np.float64)
                + (1.0 - self.alpha) * warped_ema.astype(np.float64)
            )
            result = np.clip(blended, 0, 255).astype(np.uint8)
            # Update EMA buffer with the result (not the raw blend)
            self._ema_buffer._ema = result.astype(np.float64)
        else:
            # Standard EMA without flow
            result = self._ema_buffer.update(frame)

        self._prev_gray = curr_gray
        return result

    def reset(self):
        """Clear temporal state (e.g. on scene cut)."""
        self._ema_buffer.reset()
        self._prev_gray = None
