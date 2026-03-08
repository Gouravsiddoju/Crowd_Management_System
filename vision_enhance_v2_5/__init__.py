"""
Vision Enhancement Engine v2.5 — Temporal + Exposure-Aware Pipeline
====================================================================
Non-destructive low-light enhancement for railway CCTV footage,
with temporal stabilization and dynamic exposure compensation.

Pipeline::

    Raw Frame
      → Temporal Stabilizer (flicker removal)
      → HDRNet (calibrated tone map)
      → Dynamic Exposure (brightness stabilization)
      → FastDVDnet-Lite (temporal denoise)
      → ContourNet (temporal edge blend)
      → Enhanced Frame

Usage::

    from vision_enhance_v2_5 import VisionEnhancer

    enhancer = VisionEnhancer(device="cuda", temporal=True, exposure_comp=True)
    enhanced = enhancer.enhance(frame)
"""

from vision_enhance_v2_5.enhancer import VisionEnhancer

__all__ = ["VisionEnhancer"]
__version__ = "2.5.0"
