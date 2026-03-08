"""
ContourNet Inference Wrapper — Temporal Blending
==================================================
Extends the base ContourNet inference with EMA-smoothed contour
gate maps for stable edge reinforcement across time.

When ``temporal_blend=True``:
  1. Runs ContourNet to get the raw gate map
  2. EMA-blends the gate with previous frames' gates
  3. Uses the smoothed gate for final edge application
  → Stable, flicker-free edge boundaries
"""

import os
from typing import Optional

import numpy as np
import torch

from vision_enhance_v2_5.contournet.model import ContourNet
from vision_enhance_v2_5.utils import (
    bgr_to_rgb_tensor, rgb_tensor_to_bgr, autocast_context,
)


class ContourNetInfer:
    """ContourNet inference engine with temporal blending.

    Args:
        weights_path: Path to pretrained ``weights.pth``.
        device: ``"cuda"`` or ``"cpu"``.
        strength: Blend strength for edge reinforcement (0.0–1.0).
        temporal_blend: If True, EMA-smooth the contour gate across frames.
        temporal_alpha: EMA alpha for gate smoothing (0.1–0.5).
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cuda",
        strength: float = 1.0,
        temporal_blend: bool = False,
        temporal_alpha: float = 0.3,
    ):
        self.device = torch.device(device)
        self.strength = strength
        self.temporal_blend = temporal_blend
        self.temporal_alpha = temporal_alpha

        self.model = ContourNet()

        # Load weights
        if weights_path and weights_path != "__skip__" and os.path.isfile(weights_path):
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state, strict=False)
            print(f"  [ContourNet] Loaded weights from {weights_path}")
        else:
            print(f"  [ContourNet] Using random initialization")

        self.model.to(self.device).eval()
        params = sum(p.numel() for p in self.model.parameters())
        temporal_str = " (temporal EMA)" if temporal_blend else ""
        print(f"  [ContourNet] Parameters: {params:,} | strength={strength:.2f}{temporal_str}")

        # Temporal gate cache
        self._prev_gate: Optional[torch.Tensor] = None

    @torch.no_grad()
    def enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """Apply edge reinforcement with optional temporal blending.

        Args:
            image: BGR uint8 numpy array (H, W, 3).

        Returns:
            Edge-reinforced BGR uint8 numpy array (H, W, 3).
        """
        tensor = bgr_to_rgb_tensor(image, device=self.device)

        with autocast_context(self.device):
            # Run the model's sub-components manually to intercept the gate
            edges = self.model.sobel(tensor)
            inp = torch.cat([tensor, edges], dim=1)

            f1 = self.model.conv1(inp)
            f2 = self.model.conv2(f1)
            f3 = self.model.conv3(f2) + f1
            f4 = self.model.conv4(f3)

            residual = self.model.edge_residual(f4)
            gate = self.model.gate(f4)

        gate = gate.float()
        residual = residual.float()
        tensor = tensor.float()

        # Temporal EMA blending of the gate map
        if self.temporal_blend:
            if self._prev_gate is not None:
                # Ensure spatial dimensions match (handle resolution changes)
                if self._prev_gate.shape == gate.shape:
                    gate = (
                        self.temporal_alpha * gate
                        + (1.0 - self.temporal_alpha) * self._prev_gate
                    )
            self._prev_gate = gate.clone()

        # Apply: image + strength * gate * residual
        output = tensor + self.strength * gate * residual
        output = output.clamp(0.0, 1.0)

        return rgb_tensor_to_bgr(output)

    def reset_temporal(self):
        """Clear temporal gate cache (e.g. on scene cut)."""
        self._prev_gate = None
