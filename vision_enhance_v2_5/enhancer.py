"""
Vision Enhancer v2.5 — Temporal + Exposure-Aware Pipeline
============================================================
Main entry point for the enhanced low-light CCTV enhancement engine.

Pipeline stages::

    Raw Frame_t
      → Stage 0 (opt): Temporal Stabilizer   — EMA flicker removal
      → Stage 1:        HDRNet (calibrated)   — Bilateral grid tone map
      → Stage 1.5 (opt): Dynamic Exposure     — Auto-gamma + shadow lift
      → Stage 2:        FastDVDnet-Lite       — Mild spatial/temporal denoise
      → Stage 3:        ContourNet            — Sigmoid-gated edge reinforce
      → Enhanced Frame_t

Every stage preserves original texture. No hallucination, no GANs.

Usage::

    from vision_enhance_v2_5 import VisionEnhancer

    enhancer = VisionEnhancer(
        device="cuda",
        temporal=True,
        temporal_alpha=0.2,
        exposure_comp=True,
        hdr_calibrated=True,
    )
    out = enhancer.enhance(frame_t)
"""

import os
import time
from typing import Optional

import cv2
import numpy as np
import torch

from vision_enhance_v2_5.hdrnet.infer import HDRNetInfer
from vision_enhance_v2_5.fastdvdnet.infer import FastDVDnetInfer
from vision_enhance_v2_5.contournet.infer import ContourNetInfer
from vision_enhance_v2_5.temporal.stabilizer import TemporalStabilizer
from vision_enhance_v2_5.exposure.dynamic_exposure import DynamicExposure
from vision_enhance_v2_5.utils import resize_if_needed


class VisionEnhancer:
    """Unified v2.5 enhancement pipeline with temporal + exposure features.

    Stages:
        0. Temporal Stabilizer — EMA frame fusion (optional)
        1. HDRNet — natural tone mapping (optional calibration)
        1.5. Dynamic Exposure — auto-gamma + shadow lift (optional)
        2. FastDVDnet-Lite — mild spatial/temporal denoising
        3. ContourNet — edge reinforcement (optional temporal blend)

    Args:
        device: ``"cuda"`` or ``"cpu"``.
        hdrnet_weights: Path to HDRNet checkpoint.
        dvdnet_weights: Path to FastDVDnet-Lite checkpoint.
        contour_weights: Path to ContourNet checkpoint.
        max_side: Downscale longest side for processing (0 = native).
        denoise_sigma: Noise level for denoiser (0–255). Rec: 10–25.
        contour_strength: Edge reinforcement blend (0.0–1.0).
        temporal: Enable temporal stabilization (Stage 0) and temporal
            denoising (Stage 2).
        temporal_alpha: EMA alpha for temporal stages (0.1–0.5).
        exposure_comp: Enable Dynamic Exposure compensation (Stage 1.5).
        hdr_calibrated: Enable histogram-guided HDRNet calibration.
        use_flow: Use optical flow for temporal stabilization.
        verbose: Print timing info per frame.
    """

    _DEFAULT_HDRNET = os.path.join("vision_enhance_v2_5", "hdrnet", "weights.pth")
    _DEFAULT_DVDNET = os.path.join("vision_enhance_v2_5", "fastdvdnet", "weights.pth")
    _DEFAULT_CONTOUR = os.path.join("vision_enhance_v2_5", "contournet", "weights.pth")

    def __init__(
        self,
        device: str = "cuda",
        hdrnet_weights: Optional[str] = None,
        dvdnet_weights: Optional[str] = None,
        contour_weights: Optional[str] = None,
        max_side: int = 0,
        denoise_sigma: float = 15.0,
        contour_strength: float = 1.0,
        temporal: bool = False,
        temporal_alpha: float = 0.2,
        exposure_comp: bool = False,
        hdr_calibrated: bool = False,
        use_flow: bool = False,
        verbose: bool = True,
    ):
        self.device = device
        self.max_side = max_side
        self.verbose = verbose
        self.temporal_enabled = temporal
        self.exposure_enabled = exposure_comp

        hdrnet_path = hdrnet_weights or self._DEFAULT_HDRNET
        dvdnet_path = dvdnet_weights or self._DEFAULT_DVDNET
        contour_path = contour_weights or self._DEFAULT_CONTOUR

        print("=" * 60)
        print("  Vision Enhancement Engine v2.5")
        print("  Temporal + Exposure-Aware Pipeline")
        print(f"  Device: {device}")
        print(f"  Temporal: {'ON' if temporal else 'OFF'}"
              f" (alpha={temporal_alpha}, flow={'ON' if use_flow else 'OFF'})")
        print(f"  Exposure comp: {'ON' if exposure_comp else 'OFF'}")
        print(f"  HDRNet calibrated: {'ON' if hdr_calibrated else 'OFF'}")
        print(f"  Denoise sigma: {denoise_sigma}")
        print(f"  Contour strength: {contour_strength}")
        print(f"  Max side: {max_side if max_side > 0 else 'native'}")
        print("=" * 60)

        # Stage 0: Temporal Stabilizer (optional)
        self.stabilizer = None
        if temporal:
            self.stabilizer = TemporalStabilizer(
                alpha=temporal_alpha,
                use_flow=use_flow,
            )
            print(f"  [Stage 0] Temporal Stabilizer: alpha={temporal_alpha}")

        # Stage 1: HDRNet (tone mapping)
        self.hdrnet = HDRNetInfer(
            weights_path=hdrnet_path,
            device=device,
            calibrated=hdr_calibrated,
        )

        # Stage 1.5: Dynamic Exposure (optional)
        self.exposure = None
        if exposure_comp:
            self.exposure = DynamicExposure(
                target_mean=120.0,
                ema_alpha=0.15,
            )
            print(f"  [Stage 1.5] Dynamic Exposure: target=120")

        # Stage 2: FastDVDnet-Lite (denoising)
        self.dvdnet = FastDVDnetInfer(
            weights_path=dvdnet_path,
            device=device,
            sigma=denoise_sigma,
            temporal=temporal,
        )

        # Stage 3: ContourNet (edge reinforcement)
        self.contournet = ContourNetInfer(
            weights_path=contour_path,
            device=device,
            strength=contour_strength,
            temporal_blend=temporal,
            temporal_alpha=temporal_alpha,
        )

        print("=" * 60)
        print("  [OK] Vision Enhancement Engine v2.5 ready")
        print("=" * 60)

    def enhance(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        """Enhance a single frame through the full pipeline.

        This is the primary API. Alias: ``enhance_frame()``.

        Args:
            frame: BGR uint8 numpy array (H, W, 3).
            **kwargs: Override parameters (see ``enhance_frame``).

        Returns:
            Enhanced BGR uint8 numpy array (H, W, 3).
        """
        return self.enhance_frame(frame, **kwargs)

    def enhance_frame(
        self,
        frame: np.ndarray,
        skip_stabilizer: bool = False,
        skip_hdrnet: bool = False,
        skip_exposure: bool = False,
        skip_denoise: bool = False,
        skip_contour: bool = False,
        denoise_sigma: Optional[float] = None,
    ) -> np.ndarray:
        """Enhance a single raw CCTV frame.

        Args:
            frame: BGR uint8 numpy array (H, W, 3).
            skip_stabilizer: Skip Stage 0 temporal stabilization.
            skip_hdrnet: Skip Stage 1 HDRNet tone mapping.
            skip_exposure: Skip Stage 1.5 exposure compensation.
            skip_denoise: Skip Stage 2 FastDVDnet-Lite denoising.
            skip_contour: Skip Stage 3 ContourNet edge reinforcement.
            denoise_sigma: Override denoising sigma for this frame.

        Returns:
            Enhanced BGR uint8 numpy array (H, W, 3).
        """
        assert frame is not None and len(frame.shape) == 3, \
            f"Expected BGR image (H, W, 3), got shape {getattr(frame, 'shape', None)}"
        assert frame.shape[2] == 3, \
            f"Expected 3-channel BGR, got {frame.shape[2]} channels"

        orig_h, orig_w = frame.shape[:2]
        t_start = time.perf_counter()

        # Optional downscale
        if self.max_side > 0:
            working, scale = resize_if_needed(frame, self.max_side)
        else:
            working = frame.copy()
            scale = 1.0

        t_stab = 0.0
        t_hdr = 0.0
        t_exp = 0.0
        t_denoise = 0.0
        t_contour = 0.0

        # STAGE 0: Temporal Stabilizer
        if self.stabilizer and not skip_stabilizer:
            t0 = time.perf_counter()
            working = self.stabilizer.stabilize(working)
            t_stab = time.perf_counter() - t0

        # STAGE 1: HDRNet — tone mapping
        if not skip_hdrnet:
            t1 = time.perf_counter()
            working = self.hdrnet.enhance(working)
            t_hdr = time.perf_counter() - t1

        # STAGE 1.5: Dynamic Exposure
        if self.exposure and not skip_exposure:
            t15 = time.perf_counter()
            working = self.exposure.compensate(working)
            t_exp = time.perf_counter() - t15

        # STAGE 2: FastDVDnet-Lite — denoising
        if not skip_denoise:
            t2 = time.perf_counter()
            working = self.dvdnet.denoise(working, sigma_override=denoise_sigma)
            t_denoise = time.perf_counter() - t2

        # STAGE 3: ContourNet — edge reinforcement
        if not skip_contour:
            t3 = time.perf_counter()
            working = self.contournet.enhance_edges(working)
            t_contour = time.perf_counter() - t3

        # Upscale back if we downscaled
        if scale < 1.0:
            working = cv2.resize(
                working, (orig_w, orig_h),
                interpolation=cv2.INTER_LANCZOS4,
            )

        t_total = time.perf_counter() - t_start

        if self.verbose:
            parts = []
            if t_stab > 0:
                parts.append(f"Stab: {t_stab*1000:.1f}ms")
            parts.append(f"HDR: {t_hdr*1000:.1f}ms")
            if t_exp > 0:
                parts.append(f"Exp: {t_exp*1000:.1f}ms")
            parts.append(f"Denoise: {t_denoise*1000:.1f}ms")
            parts.append(f"Contour: {t_contour*1000:.1f}ms")
            parts.append(f"Total: {t_total*1000:.1f}ms")
            print(f"[VisionEnhancer v2.5] {orig_w}x{orig_h} | {' | '.join(parts)}")

        return working

    def enhance_batch(
        self,
        frames: list,
        **kwargs,
    ) -> list:
        """Enhance a batch of frames sequentially.

        For temporal modes, frames should be in chronological order.

        Args:
            frames: List of BGR uint8 numpy arrays.
            **kwargs: Additional kwargs passed to enhance_frame.

        Returns:
            List of enhanced BGR uint8 numpy arrays.
        """
        return [self.enhance_frame(f, **kwargs) for f in frames]

    def reset_temporal(self):
        """Reset all temporal state (e.g. on scene cut).

        Clears buffers in the stabilizer, denoiser, contournet,
        and exposure module.
        """
        if self.stabilizer:
            self.stabilizer.reset()
        if hasattr(self.dvdnet, 'reset_temporal'):
            self.dvdnet.reset_temporal()
        if hasattr(self.contournet, 'reset_temporal'):
            self.contournet.reset_temporal()
        if self.exposure:
            self.exposure.reset()

    @staticmethod
    def get_gpu_memory_usage() -> dict:
        """Return current GPU memory usage (requires CUDA)."""
        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        }
