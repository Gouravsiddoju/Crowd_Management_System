"""
EMA Buffer — Generic Exponential Moving Average
=================================================
Reusable buffer for temporal smoothing of frames, tensors,
or any numeric array. Used by the Temporal Stabilizer and
ContourNet temporal blending.

The EMA formula:
    ema_t = alpha * value_t + (1 - alpha) * ema_{t-1}
"""

import numpy as np


class EMABuffer:
    """Exponential Moving Average buffer for numpy arrays.

    Args:
        alpha: Smoothing factor in (0, 1]. Higher = more weight on current value.
            - 1.0 = no smoothing (passthrough)
            - 0.1 = very heavy smoothing (slow response)
            Recommended: 0.15–0.35 for CCTV temporal smoothing.
    """

    def __init__(self, alpha: float = 0.2):
        assert 0.0 < alpha <= 1.0, f"alpha must be in (0, 1], got {alpha}"
        self.alpha = alpha
        self._ema: np.ndarray = None

    def update(self, value: np.ndarray) -> np.ndarray:
        """Feed a new value and return the smoothed result.

        Args:
            value: Numpy array of any shape (must be consistent across calls).

        Returns:
            EMA-smoothed array of the same shape and dtype.
        """
        if self._ema is None:
            # First frame: initialize EMA directly
            self._ema = value.astype(np.float64)
        else:
            self._ema = (
                self.alpha * value.astype(np.float64)
                + (1.0 - self.alpha) * self._ema
            )
        return self._ema.astype(value.dtype)

    def get(self) -> np.ndarray:
        """Return the current EMA value (or None if no data yet)."""
        return self._ema

    def reset(self):
        """Clear the internal state."""
        self._ema = None

    @property
    def initialized(self) -> bool:
        """Whether the buffer has received at least one value."""
        return self._ema is not None
