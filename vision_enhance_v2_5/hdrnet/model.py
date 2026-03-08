"""
HDRNet — Bilateral Grid Tone Mapping
======================================
Google Research-inspired architecture for natural illumination enhancement.

Applies local affine color transforms via a learned bilateral grid.
* Zero hallucination — only transforms existing pixel values
* Natural shadow lifting and tone mapping
* Tiny model (~500K params), extremely fast

Reference:
    Gharbi et al., "Deep Bilateral Learning for Real-Time Image Enhancement",
    ACM SIGGRAPH 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1, use_bn: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LowLevelFeatures(nn.Module):
    """Extract spatial features from a low-resolution input.

    Progressively downsamples with strided convolutions.
    """

    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(in_ch, 8, stride=2),       # /2
            ConvBlock(8, 16, stride=2),           # /4
            ConvBlock(16, 32, stride=2),          # /8
            ConvBlock(32, 64, stride=2),          # /16
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class HighLevelFeatures(nn.Module):
    """Global context features via deeper convolutions + global pooling branch."""

    def __init__(self, in_ch: int = 64):
        super().__init__()
        self.local_path = nn.Sequential(
            ConvBlock(in_ch, 64, stride=2),       # /32
            ConvBlock(64, 64),
        )
        self.global_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_feat = self.local_path(x)
        global_feat = self.global_path(x)
        # Broadcast global features to spatial dims
        b, c = global_feat.shape
        global_feat = global_feat.view(b, c, 1, 1).expand_as(local_feat)
        return local_feat + global_feat


class BilateralGrid(nn.Module):
    """Predict a bilateral grid of affine coefficients.

    The grid has shape (B, 12, D, H, W) where:
    - 12 = 4 output channels × 3 input channels (affine matrix)
    - D = depth slices along intensity axis
    - H, W = spatial grid resolution

    For a 3→3 affine transform: out_c = sum(A[c,i] * in_i) for i in {R,G,B}
    plus a bias, giving 12 coefficients per grid cell.
    """

    def __init__(self, in_ch: int = 64, grid_depth: int = 8,
                 n_affine: int = 12):
        super().__init__()
        self.grid_depth = grid_depth
        self.n_affine = n_affine

        # Predict affine coefficients
        self.coeff_conv = nn.Sequential(
            ConvBlock(in_ch, 64),
            nn.Conv2d(64, n_affine * grid_depth, 1, bias=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns grid of shape (B, n_affine, D, H, W)."""
        b = features.shape[0]
        raw = self.coeff_conv(features)
        # Reshape: (B, n_affine*D, Hg, Wg) → (B, n_affine, D, Hg, Wg)
        _, _, hg, wg = raw.shape
        grid = raw.view(b, self.n_affine, self.grid_depth, hg, wg)
        return grid


class GuideMap(nn.Module):
    """Produce a scalar guide map from the full-resolution input.

    The guide determines where along the bilateral grid's depth axis
    each pixel looks up its affine coefficients.
    """

    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1, bias=True),
            nn.Sigmoid(),  # guide ∈ [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns guide map of shape (B, 1, H, W)."""
        return self.net(x)


def bilateral_slice(grid: torch.Tensor, guide: torch.Tensor,
                    full_res: torch.Tensor) -> torch.Tensor:
    """Slice the bilateral grid using the guide map and apply affine transform.

    Args:
        grid:     (B, 12, D, Hg, Wg) — bilateral grid of affine coefficients
        guide:    (B, 1, H, W) — guide map in [0, 1]
        full_res: (B, 3, H, W) — full resolution input

    Returns:
        (B, 3, H, W) — transformed output
    """
    b, n_affine, d, hg, wg = grid.shape
    _, _, h, w = full_res.shape

    # 1. Spatially upsample grid to full resolution
    grid_flat = grid.view(b, n_affine * d, hg, wg)
    grid_up = F.interpolate(grid_flat, size=(h, w), mode='bilinear',
                            align_corners=False)
    grid_up = grid_up.view(b, n_affine, d, h, w)

    # 2. Slice along depth using the guide map
    guide_scaled = guide * (d - 1)

    depth_lo = guide_scaled.long().clamp(0, d - 2)
    depth_hi = (depth_lo + 1).clamp(0, d - 1)
    alpha = (guide_scaled - depth_lo.float())

    depth_lo_exp = depth_lo.expand(b, n_affine, 1, h, w)
    depth_hi_exp = depth_hi.expand(b, n_affine, 1, h, w)

    coeff_lo = torch.gather(grid_up, 2, depth_lo_exp).squeeze(2)
    coeff_hi = torch.gather(grid_up, 2, depth_hi_exp).squeeze(2)

    alpha_exp = alpha.expand_as(coeff_lo)
    coeffs = coeff_lo * (1.0 - alpha_exp) + coeff_hi * alpha_exp

    # 3. Apply 3×3+3 affine transform per pixel
    matrix = coeffs[:, :9, :, :]
    bias = coeffs[:, 9:12, :, :]

    matrix = matrix.view(b, 3, 3, h, w)

    inp = full_res
    out = torch.zeros_like(inp)
    for c in range(3):
        for i in range(3):
            out[:, c:c+1, :, :] += matrix[:, c, i:i+1, :, :] * inp[:, i:i+1, :, :]
    out = out + bias

    return out.clamp(0.0, 1.0)


class HDRNet(nn.Module):
    """HDRNet — Bilateral Grid Tone Mapping Network.

    Full pipeline:
        1. Downsample input to low-res
        2. Extract low-level → high-level features
        3. Predict bilateral grid of affine coefficients
        4. Compute guide map at full resolution
        5. Slice grid with guide and apply affine transform

    Args:
        grid_depth: Number of depth slices in the bilateral grid.
        lowres_size: Target size for the low-resolution processing branch.

    Input:  (B, 3, H, W) RGB float32 in [0, 1]
    Output: (B, 3, H, W) RGB float32 in [0, 1]
    """

    def __init__(self, grid_depth: int = 8, lowres_size: int = 256):
        super().__init__()
        self.lowres_size = lowres_size

        self.low_level = LowLevelFeatures(in_ch=3)
        self.high_level = HighLevelFeatures(in_ch=64)
        self.grid = BilateralGrid(in_ch=64, grid_depth=grid_depth)
        self.guide = GuideMap(in_ch=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) RGB in [0, 1]
        Returns:
            (B, 3, H, W) tone-mapped RGB in [0, 1]
        """
        lowres = F.interpolate(x, size=(self.lowres_size, self.lowres_size),
                               mode='bilinear', align_corners=False)
        low_feat = self.low_level(lowres)
        high_feat = self.high_level(low_feat)
        grid = self.grid(high_feat)

        guide = self.guide(x)

        out = bilateral_slice(grid, guide, x)
        return out


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
