"""
Histogram Utilities
====================
Pure numpy/OpenCV helpers for analyzing frame luminance.
Used by the Dynamic Exposure module and HDRNet calibration.
"""

import cv2
import numpy as np
from typing import Dict


def compute_luminance_stats(frame: np.ndarray) -> Dict[str, float]:
    """Compute luminance statistics from a BGR uint8 frame.

    Converts to LAB color space and analyzes the L channel.

    Args:
        frame: BGR uint8 numpy array (H, W, 3).

    Returns:
        Dictionary with keys:
            - ``mean``: Mean luminance (0–255).
            - ``median``: Median luminance.
            - ``p5``: 5th percentile (shadow level).
            - ``p25``: 25th percentile.
            - ``p75``: 75th percentile.
            - ``p95``: 95th percentile (highlight level).
            - ``std``: Standard deviation.
            - ``dynamic_range``: p95 - p5 (useful range).
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)  # L channel: 0–255

    return {
        "mean": float(np.mean(L)),
        "median": float(np.median(L)),
        "p5": float(np.percentile(L, 5)),
        "p25": float(np.percentile(L, 25)),
        "p75": float(np.percentile(L, 75)),
        "p95": float(np.percentile(L, 95)),
        "std": float(np.std(L)),
        "dynamic_range": float(np.percentile(L, 95) - np.percentile(L, 5)),
    }


def compute_histogram_scaling(
    stats: Dict[str, float],
    target_mean: float = 120.0,
    min_gain: float = 0.8,
    max_gain: float = 2.5,
) -> Dict[str, float]:
    """Derive gain and offset parameters from luminance stats.

    Used to pre-condition frames or adjust HDRNet tone strength.

    Args:
        stats: Output of ``compute_luminance_stats``.
        target_mean: Desired mean luminance after correction.
        min_gain: Minimum allowed gain (prevents darkening bright frames).
        max_gain: Maximum allowed gain (prevents blowing out dark frames).

    Returns:
        Dictionary with:
            - ``gain``: Multiplicative brightness factor.
            - ``gamma``: Suggested gamma correction value.
            - ``tone_strength``: HDRNet tone curve scaling (0.5–2.0).
    """
    mean_lum = max(stats["mean"], 1.0)  # avoid division by zero

    # Raw gain to reach target
    raw_gain = target_mean / mean_lum
    gain = float(np.clip(raw_gain, min_gain, max_gain))

    # Gamma: darker frames need lower gamma (more brightening)
    # Map mean luminance [0, 255] → gamma [0.3, 1.2]
    gamma = float(np.clip(0.3 + (mean_lum / 255.0) * 0.9, 0.3, 1.2))

    # Tone strength for HDRNet: inversely proportional to brightness
    # Very dark (mean<30) → strength 2.0; well-lit (mean>150) → strength 0.5
    tone_strength = float(np.clip(2.0 - (mean_lum / 100.0), 0.5, 2.0))

    return {
        "gain": gain,
        "gamma": gamma,
        "tone_strength": tone_strength,
    }


def apply_gamma(frame: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction to a BGR uint8 frame.

    Standard convention:
        gamma < 1  →  pixel^(<1)  →  brightens (lifts shadows)
        gamma > 1  →  pixel^(>1)  →  darkens
        gamma = 1  →  no change

    Args:
        frame: BGR uint8 numpy array (H, W, 3).
        gamma: Gamma exponent. Values below 1.0 brighten, above 1.0 darken.

    Returns:
        Gamma-corrected BGR uint8 numpy array.
    """
    gamma = max(gamma, 0.01)
    table = np.array(
        [(i / 255.0) ** gamma * 255.0 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(frame, table)
