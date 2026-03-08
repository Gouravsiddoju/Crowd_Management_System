"""
Pipeline Benchmark Tool
========================
Measures FPS and per-stage latency of the v2.5 enhancement pipeline
at configurable resolutions.

Usage::

    python -m vision_enhance_v2_5.deploy.benchmark
    python -m vision_enhance_v2_5.deploy.benchmark --resolution 1280x720 --frames 50
    python -m vision_enhance_v2_5.deploy.benchmark --temporal --exposure
"""

import argparse
import time
import sys

import cv2
import numpy as np
import torch


def create_test_frame(height: int, width: int) -> np.ndarray:
    """Create a random dark CCTV-like test frame."""
    frame = np.random.randint(5, 50, (height, width, 3), dtype=np.uint8)
    # Add some structure
    for _ in range(5):
        cx, cy = np.random.randint(10, width-10), np.random.randint(10, height-10)
        r = np.random.randint(5, 30)
        color = tuple(int(c) for c in np.random.randint(30, 120, 3))
        cv2.circle(frame, (cx, cy), r, color, -1)
    return frame


def benchmark(
    device: str = "cpu",
    resolution: str = "640x480",
    num_frames: int = 30,
    temporal: bool = False,
    exposure: bool = False,
    calibrated: bool = False,
    skip_weights: bool = True,
    warmup: int = 3,
):
    """Run the benchmark."""
    from vision_enhance_v2_5.enhancer import VisionEnhancer

    w, h = map(int, resolution.split("x"))

    print("=" * 60)
    print("  Vision Enhancement v2.5 — Benchmark")
    print("=" * 60)
    print(f"  Resolution: {w}x{h}")
    print(f"  Frames: {num_frames}")
    print(f"  Device: {device}")
    print(f"  Temporal: {'ON' if temporal else 'OFF'}")
    print(f"  Exposure: {'ON' if exposure else 'OFF'}")
    print(f"  Calibrated HDR: {'ON' if calibrated else 'OFF'}")
    print("=" * 60)

    # Build enhancer
    skip_flag = "__skip__" if skip_weights else None
    enhancer = VisionEnhancer(
        device=device,
        hdrnet_weights=skip_flag,
        dvdnet_weights=skip_flag,
        contour_weights=skip_flag,
        temporal=temporal,
        exposure_comp=exposure,
        hdr_calibrated=calibrated,
        verbose=False,
    )

    # Generate test frames
    frames = [create_test_frame(h, w) for _ in range(num_frames)]

    # Warmup
    print(f"\n  Warming up ({warmup} frames)...")
    for i in range(min(warmup, num_frames)):
        enhancer.enhance(frames[i])
    enhancer.reset_temporal()

    # Benchmark
    print(f"  Running benchmark ({num_frames} frames)...")
    times = []
    for i in range(num_frames):
        t0 = time.perf_counter()
        _ = enhancer.enhance(frames[i])
        dt = time.perf_counter() - t0
        times.append(dt)

    times = np.array(times)
    fps = 1.0 / times.mean()
    total = times.sum()

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Mean time:   {times.mean() * 1000:.1f} ms/frame")
    print(f"  Median time: {np.median(times) * 1000:.1f} ms/frame")
    print(f"  Min time:    {times.min() * 1000:.1f} ms/frame")
    print(f"  Max time:    {times.max() * 1000:.1f} ms/frame")
    print(f"  Std dev:     {times.std() * 1000:.1f} ms")
    print(f"  FPS:         {fps:.1f}")
    print(f"  Total:       {total:.1f}s for {num_frames} frames")

    # GPU memory
    mem = enhancer.get_gpu_memory_usage()
    if mem:
        print(f"  GPU memory:  {mem['allocated_mb']:.1f}MB allocated, "
              f"{mem['reserved_mb']:.1f}MB reserved")

    # Param count
    total_params = (
        sum(p.numel() for p in enhancer.hdrnet.model.parameters())
        + sum(p.numel() for p in enhancer.dvdnet.model.parameters())
        + sum(p.numel() for p in enhancer.contournet.model.parameters())
    )
    print(f"  Total params: {total_params:,}")
    print("=" * 60)

    return {
        "fps": fps,
        "mean_ms": times.mean() * 1000,
        "median_ms": np.median(times) * 1000,
        "total_params": total_params,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark v2.5 pipeline")
    parser.add_argument("--device", default="cpu", help="cuda or cpu")
    parser.add_argument("--resolution", default="640x480", help="WxH")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames")
    parser.add_argument("--temporal", action="store_true", help="Enable temporal mode")
    parser.add_argument("--exposure", action="store_true", help="Enable exposure comp")
    parser.add_argument("--calibrated", action="store_true", help="Enable HDR calibration")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup frames")
    args = parser.parse_args()

    benchmark(
        device=args.device,
        resolution=args.resolution,
        num_frames=args.frames,
        temporal=args.temporal,
        exposure=args.exposure,
        calibrated=args.calibrated,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
