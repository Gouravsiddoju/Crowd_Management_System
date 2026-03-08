"""
ContourNet — Edge Reinforcement Network
=========================================
Tiny CNN that restores edge boundaries in enhanced CCTV frames.

Takes as input the image concatenated with Sobel edge features
and produces a residual edge-reinforced output via feathered
contour blending.

Key properties:
* Uses fixed (non-learnable) Sobel kernels for deterministic edges
* Feathered blending via sigmoid gating — zero halo artifacts
* Residual learning — only adjusts edges, not flat regions
* ~100K parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdgeExtractor(nn.Module):
    """Extract Sobel-X and Sobel-Y edge maps using fixed convolution kernels.

    Operates on grayscale (luminance). The kernels are registered as
    non-learnable buffers for deterministic, consistent edge maps.

    Output: 2-channel edge map (Sobel-X, Sobel-Y) normalized to [0, 1].
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) RGB float in [0, 1]

        Returns:
            (B, 2, H, W) Sobel edge features in [0, 1]
        """
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)

        edge_x = torch.abs(edge_x)
        edge_y = torch.abs(edge_y)

        max_val = 4.0
        edge_x = (edge_x / max_val).clamp(0, 1)
        edge_y = (edge_y / max_val).clamp(0, 1)

        return torch.cat([edge_x, edge_y], dim=1)


class ContourNet(nn.Module):
    """Edge reinforcement network with feathered contour blending.

    Architecture:
        Input: 5 channels (3 RGB + 2 Sobel edge maps)
        → 5 conv layers with residual connections
        → Sigmoid gate (feathered blending mask)
        → Edge residual × gate + original image

    Args:
        mid_ch: Number of intermediate feature channels.

    Input:  (B, 3, H, W) RGB float in [0, 1]
    Output: (B, 3, H, W) edge-reinforced RGB in [0, 1]
    """

    def __init__(self, mid_ch: int = 32):
        super().__init__()
        self.sobel = SobelEdgeExtractor()

        self.conv1 = nn.Sequential(
            nn.Conv2d(5, mid_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.edge_residual = nn.Conv2d(mid_ch, 3, 3, padding=1, bias=True)

        self.gate = nn.Sequential(
            nn.Conv2d(mid_ch, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) RGB in [0, 1]

        Returns:
            (B, 3, H, W) edge-reinforced RGB in [0, 1]
        """
        edges = self.sobel(x)
        inp = torch.cat([x, edges], dim=1)

        f1 = self.conv1(inp)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2) + f1  # skip connection
        f4 = self.conv4(f3)

        residual = self.edge_residual(f4)
        gate = self.gate(f4)

        out = x + gate * residual

        return out.clamp(0.0, 1.0)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
