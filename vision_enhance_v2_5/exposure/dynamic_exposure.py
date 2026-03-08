"""
Dynamic Exposure Compensation — Stage 1.5
==========================================
Analyzes per-frame luminance and applies adaptive exposure correction
with temporal smoothing to prevent brightness flicker.

This runs AFTER HDRNet tone mapping and BEFORE denoising.
It acts as a safety net: if HDRNet over-brightens or under-brightens,
this module normalizes the result to a consistent exposure level.

Key properties:
* Zero learnable parameters — pure math
* EMA-smoothed exposure prevents frame-to-frame flicker
* Operates in LAB color space (luminance-only adjustment)
"""

import cv2
import numpy as np

from vision_enhance_v2_5.temporal.ema_buffer import EMABuffer
from vision_enhance_v2_5.exposure.histogram_utils import (
    compute_luminance_stats,
    apply_gamma,
)


class DynamicExposure:
    """Adaptive exposure compensation with temporal smoothing.

    Analyzes each frame's luminance, computes an ideal gamma correction,
    and smooths the correction across frames to avoid jitter.

    Args:
        target_mean: Target mean luminance (0–255). 120 is a good default
            for CCTV that balances visibility and naturalness.
        ema_alpha: Smoothing factor for exposure parameters (0.1–0.5).
            Lower = smoother transitions, higher = faster adaptation.
        min_gamma: Minimum allowed gamma (prevents extreme brightening).
        max_gamma: Maximum allowed gamma.
    """

    def __init__(
        self,
        target_mean: float = 120.0,
        ema_alpha: float = 0.15,
        min_gamma: float = 0.4,
        max_gamma: float = 3.0,
    ):
        self.target_mean = target_mean
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self._gamma_buffer = EMABuffer(alpha=ema_alpha)

    def compensate(self, frame: np.ndarray) -> np.ndarray:
        """Apply dynamic exposure compensation to a BGR frame.

        Args:
            frame: BGR uint8 numpy array (H, W, 3).

        Returns:
            Exposure-compensated BGR uint8 numpy array (H, W, 3).
        """
        stats = compute_luminance_stats(frame)
        mean_lum = stats["mean"]

        # Compute ideal gamma for this frame.
        # Standard convention: gamma < 1 brightens, gamma > 1 darkens.
        # gamma = mean_lum / target:
        #   dark frame  → mean < target → gamma < 1 → brightens  ✓
        #   bright frame → mean > target → gamma > 1 → darkens   ✓
        if mean_lum < 1.0:
            # Essentially black frame -- apply strong brightening
            frame_gamma = self.min_gamma
        elif abs(mean_lum - self.target_mean) < 10:
            # Already near target -- no correction needed
            frame_gamma = 1.0
        else:
            ratio = mean_lum / self.target_mean
            frame_gamma = float(np.clip(ratio, self.min_gamma, self.max_gamma))

        # EMA-smooth the gamma value across frames
        gamma_arr = np.array([frame_gamma])
        smoothed = self._gamma_buffer.update(gamma_arr)
        smooth_gamma = float(smoothed[0])

        # Only apply if meaningfully different from 1.0
        if abs(smooth_gamma - 1.0) < 0.02:
            result = frame
        else:
            result = apply_gamma(frame, smooth_gamma)

        # Additional shadow lift for very dark frames only
        if stats["p5"] < 15 and mean_lum < 80:
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            L = lab[:, :, 0].astype(np.float32)
            shadow_mask = (L < 40).astype(np.float32)
            lift = shadow_mask * 5.0  # add up to 5 to very dark pixels
            L = np.clip(L + lift, 0, 255).astype(np.uint8)
            lab[:, :, 0] = L
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return result

    def reset(self):
        """Clear temporal state."""
        self._gamma_buffer.reset()
