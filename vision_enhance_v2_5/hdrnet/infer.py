"""
HDRNet Inference Wrapper — Calibrated Mode
=============================================
Extends the base HDRNet inference with histogram-guided calibration
that auto-adjusts tone mapping strength based on scene brightness.

When ``calibrated=True``:
  1. Pre-analyzes frame histogram → computes tone strength scalar
  2. Scales the bilateral grid bias terms after forward pass
  3. Prevents overbright daytime / underbright deep-night frames
"""

import os
from typing import Optional

import cv2
import numpy as np
import torch

from vision_enhance_v2_5.hdrnet.model import HDRNet
from vision_enhance_v2_5.utils import (
    bgr_to_rgb_tensor, rgb_tensor_to_bgr,
    pad_to_multiple, unpad, autocast_context,
)
from vision_enhance_v2_5.exposure.histogram_utils import (
    compute_luminance_stats, compute_histogram_scaling,
)


class HDRNetInfer:
    """HDRNet inference engine with optional calibration.

    Args:
        weights_path: Path to pretrained ``weights.pth``. If ``"__skip__"``
            or the file does not exist, random weights are used.
        device: ``"cuda"`` or ``"cpu"``.
        grid_depth: Bilateral grid depth slices.
        lowres_size: Low-resolution branch target size.
        calibrated: If True, auto-adjust tone mapping based on scene brightness.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cuda",
        grid_depth: int = 8,
        lowres_size: int = 256,
        calibrated: bool = False,
    ):
        self.device = torch.device(device)
        self.calibrated = calibrated

        self.model = HDRNet(
            grid_depth=grid_depth,
            lowres_size=lowres_size,
        )

        # Load weights
        if weights_path and weights_path != "__skip__" and os.path.isfile(weights_path):
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state, strict=False)
            print(f"  [HDRNet] Loaded weights from {weights_path}")
        else:
            print(f"  [HDRNet] Using random initialization")

        self.model.to(self.device).eval()
        params = sum(p.numel() for p in self.model.parameters())
        cal_str = " (calibrated)" if calibrated else ""
        print(f"  [HDRNet] Parameters: {params:,}{cal_str}")

    @torch.no_grad()
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Apply HDRNet tone mapping to a BGR uint8 image.

        If calibrated mode is enabled, the tone mapping strength is
        automatically adjusted based on the input frame's luminance.

        Args:
            image: BGR uint8 numpy array (H, W, 3).

        Returns:
            Tone-mapped BGR uint8 numpy array (H, W, 3).
        """
        # Pre-analyze for calibration
        if self.calibrated:
            stats = compute_luminance_stats(image)
            scaling = compute_histogram_scaling(stats)
            tone_strength = scaling["tone_strength"]
        else:
            tone_strength = 1.0

        tensor = bgr_to_rgb_tensor(image, device=self.device)
        _, _, h, w = tensor.shape

        # Pad to multiple of 16 for strided convolutions
        tensor_padded, pad_hw = pad_to_multiple(tensor, 16)

        with autocast_context(self.device):
            output = self.model(tensor_padded)

        output = output.float()

        # Calibrated blending: scale the enhancement effect
        if tone_strength != 1.0:
            # Blend between original and enhanced based on tone_strength
            # tone_strength > 1 → more enhancement; < 1 → less
            original_padded = tensor_padded.float()
            # Compute residual (enhancement delta)
            residual = output - original_padded
            # Scale the residual by tone_strength
            output = original_padded + residual * tone_strength
            output = output.clamp(0.0, 1.0)

        # Remove padding
        output = unpad(output, pad_hw)

        return rgb_tensor_to_bgr(output)
