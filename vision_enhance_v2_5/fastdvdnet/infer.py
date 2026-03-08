"""
FastDVDnet-Lite Inference Wrapper — Temporal Variant
=====================================================
Extends the base inference with automatic 3-frame sliding window
temporal denoising when ``temporal=True``.

When temporal mode is active:
  1. Maintains a 3-frame buffer (prev, current, next)
  2. Automatically pads the buffer at sequence start/end
  3. Concatenates 3 frames as 9-channel input for the model
"""

import os
from typing import Optional
from collections import deque

import numpy as np
import torch

from vision_enhance_v2_5.fastdvdnet.model_lite import FastDVDnetLite
from vision_enhance_v2_5.utils import (
    bgr_to_rgb_tensor, rgb_tensor_to_bgr,
    pad_to_multiple, unpad, autocast_context,
)


class FastDVDnetInfer:
    """FastDVDnet-Lite inference engine with temporal support.

    Args:
        weights_path: Path to pretrained ``weights.pth``. If ``"__skip__"``
            or the file does not exist, random weights are used.
        device: ``"cuda"`` or ``"cpu"``.
        sigma: Noise level (0–255 scale, internally normalized to [0,1]).
        temporal: If True, use 3-frame temporal denoising.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cuda",
        sigma: float = 15.0,
        temporal: bool = False,
    ):
        self.device = torch.device(device)
        self.sigma = sigma / 255.0
        self.temporal = temporal

        self.model = FastDVDnetLite(temporal=temporal)

        # Load weights (filter out shape-mismatched keys for temporal/non-temporal compat)
        if weights_path and weights_path != "__skip__" and os.path.isfile(weights_path):
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
            model_state = self.model.state_dict()
            # Only load keys that exist and have matching shape
            compatible = {
                k: v for k, v in state.items()
                if k in model_state and v.shape == model_state[k].shape
            }
            model_state.update(compatible)
            self.model.load_state_dict(model_state)
            skipped = len(state) - len(compatible)
            skip_msg = f" ({skipped} keys shape-adapted)" if skipped else ""
            print(f"  [FastDVDnet] Loaded weights from {weights_path}{skip_msg}")
        else:
            print(f"  [FastDVDnet] Using random initialization")

        self.model.to(self.device).eval()
        params = sum(p.numel() for p in self.model.parameters())
        mode_str = "temporal-3f" if temporal else "single-frame"
        print(f"  [FastDVDnet] Parameters: {params:,} | sigma={sigma:.1f} | {mode_str}")

        # Temporal frame buffer (stores RGB tensors, NCHW, on device)
        self._frame_buffer: deque = deque(maxlen=3)

    @torch.no_grad()
    def denoise(
        self,
        image: np.ndarray,
        sigma_override: Optional[float] = None,
    ) -> np.ndarray:
        """Denoise a BGR uint8 image.

        In temporal mode, this maintains a sliding window buffer.
        The first two frames use duplicated padding for the buffer.

        Args:
            image: BGR uint8 numpy array (H, W, 3).
            sigma_override: Override sigma for this call (0–255 scale).

        Returns:
            Denoised BGR uint8 numpy array (H, W, 3).
        """
        tensor = bgr_to_rgb_tensor(image, device=self.device)
        _, _, h, w = tensor.shape

        # Pad to multiple of 4
        tensor_padded, pad_hw = pad_to_multiple(tensor, 4)
        _, _, hp, wp = tensor_padded.shape

        # Build sigma map
        sigma = (sigma_override / 255.0) if sigma_override is not None else self.sigma
        sigma_map = torch.full(
            (1, 1, hp, wp), sigma,
            device=self.device, dtype=torch.float32,
        )

        if self.temporal:
            # Push current frame into the sliding buffer
            self._frame_buffer.append(tensor_padded)

            # Build 3-frame input
            if len(self._frame_buffer) == 1:
                # First frame: duplicate it 3 times
                frames_3 = torch.cat([
                    self._frame_buffer[0],
                    self._frame_buffer[0],
                    self._frame_buffer[0],
                ], dim=1)
            elif len(self._frame_buffer) == 2:
                # Second frame: prev, curr, curr
                frames_3 = torch.cat([
                    self._frame_buffer[0],
                    self._frame_buffer[1],
                    self._frame_buffer[1],
                ], dim=1)
            else:
                # Normal: prev, curr, next (using curr as last since it's real-time)
                # Actually for real-time, we use: t-2, t-1, t
                frames_3 = torch.cat([
                    self._frame_buffer[0],
                    self._frame_buffer[1],
                    self._frame_buffer[2],
                ], dim=1)

            with autocast_context(self.device):
                output = self.model(frames_3, sigma_map)
        else:
            with autocast_context(self.device):
                output = self.model(tensor_padded, sigma_map)

        output = unpad(output.float(), pad_hw)
        return rgb_tensor_to_bgr(output)

    def reset_temporal(self):
        """Reset temporal frame buffer (e.g. on scene cut)."""
        self._frame_buffer.clear()
