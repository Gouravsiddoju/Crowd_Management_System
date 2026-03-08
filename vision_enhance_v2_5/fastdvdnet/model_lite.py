"""
FastDVDnet-Lite — Lightweight Spatial Denoiser
===============================================
Minimal denoiser that removes sensor noise and compression block artifacts
while preserving fine texture detail.

Supports single-frame and optional 3-frame temporal modes.
Designed for CCTV / surveillance footage where texture preservation
is critical.

Key differences from full FastDVDnet:
* Single-frame default (no temporal buffer required)
* Fewer parameters (~300K vs ~2M)
* Conservative denoising to avoid oil-painting artifacts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoisingBlock(nn.Module):
    """Small U-Net-like encoder-decoder block with skip connections.

    Structure:
        Encoder: 2× (Conv3×3 → ReLU) with stride-2 downsample
        Decoder: 2× (ConvT3×3 → ReLU) with stride-2 upsample
        Skip connection from encoder to decoder
    """

    def __init__(self, in_ch: int, mid_ch: int = 32):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(mid_ch * 2, mid_ch, 3, stride=2,
                               padding=1, output_padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(mid_ch * 2, mid_ch, 3, stride=2,
                               padding=1, output_padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        # Final 1×1 to collapse to 3 channels
        self.out_conv = nn.Conv2d(mid_ch, 3, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)    # (B, mid, H, W)
        e2 = self.enc2(e1)   # (B, mid, H/2, W/2)
        e3 = self.enc3(e2)   # (B, mid, H/4, W/4)

        # Bottleneck
        bn = self.bottleneck(e3)  # (B, mid, H/4, W/4)

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([bn, e3], dim=1))
        if d3.shape[2:] != e2.shape[2:]:
            d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear',
                               align_corners=False)

        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        if d2.shape[2:] != e1.shape[2:]:
            d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear',
                               align_corners=False)

        out = self.out_conv(d2)
        return out


class FastDVDnetLite(nn.Module):
    """Lightweight spatial denoiser with optional temporal support.

    In single-frame mode, processes one frame with a noise-level map.
    In temporal mode, processes 3 frames (previous, current, next).

    Args:
        in_ch: Input channels per frame (3 for RGB).
        mid_ch: Internal feature channels.
        temporal: If True, accepts 3 concatenated frames.

    Input:
        x: (B, in_ch * n_frames + 1, H, W) where +1 is the sigma map
    Output:
        (B, 3, H, W) denoised frame
    """

    def __init__(self, in_ch: int = 3, mid_ch: int = 32,
                 temporal: bool = False):
        super().__init__()
        self.temporal = temporal
        n_frames = 3 if temporal else 1
        total_in = in_ch * n_frames + 1  # +1 for sigma map

        # Two-stage denoising
        self.stage1 = DenoisingBlock(in_ch=total_in, mid_ch=mid_ch)
        self.stage2 = DenoisingBlock(in_ch=3 + 1, mid_ch=mid_ch)  # denoised + sigma

    def forward(self, x: torch.Tensor, sigma_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) for single-frame or (B, 9, H, W) for temporal
            sigma_map: (B, 1, H, W) noise level map in [0, 1]

        Returns:
            (B, 3, H, W) denoised RGB in same range as input
        """
        # Stage 1: initial denoising
        inp1 = torch.cat([x, sigma_map], dim=1)
        residual1 = self.stage1(inp1)

        # Input image (center frame for temporal mode)
        if self.temporal and x.shape[1] == 9:
            center = x[:, 3:6, :, :]
        else:
            center = x[:, :3, :, :]

        denoised1 = center + residual1  # residual learning

        # Stage 2: refinement
        inp2 = torch.cat([denoised1, sigma_map], dim=1)
        residual2 = self.stage2(inp2)
        denoised2 = denoised1 + residual2

        return denoised2.clamp(0.0, 1.0)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
