# Vision Enhancement Engine v2.5 — Temporal + Exposure-Aware Pipeline

Non-destructive low-light enhancement for **railway station CCTV** footage,
with **temporal stabilization** and **dynamic exposure compensation**.

---

## What's New in v2.5

| Feature | v2.0 | v2.5 |
|---------|------|------|
| Temporal flicker removal | ❌ | ✅ EMA + optical flow |
| Auto exposure compensation | ❌ | ✅ Per-frame gamma + shadow lift |
| HDRNet calibration | ❌ | ✅ Histogram-guided tone strength |
| Temporal denoising | ❌ | ✅ 3-frame sliding window |
| Temporal edge blending | ❌ | ✅ EMA-smoothed contour gate |
| ONNX/TensorRT export | Manual | ✅ CLI tools |
| Benchmarking | ❌ | ✅ Per-stage FPS tool |
| Parameter budget | ~900K | ~900K (same models) |

---

## Pipeline Architecture

```
RAW VIDEO FRAME_t
         │
         ▼
 ┌──────────────────────────────────────────────┐
 │  Stage 0: Temporal Stabilization (optional)  │
 │   • EMA frame fusion (alpha: 0.1–0.5)       │
 │   • Optional optical-flow warping            │
 │   • Removes flicker before enhancement       │
 └──────────────────────────────────────────────┘
         │
         ▼
 ┌──────────────────────────────────────────────┐
 │  Stage 1: HDRNet (Calibrated Mode)           │
 │   • Local affine tone transform              │
 │   • Histogram-guided tone scaling            │
 │   • Prevents overbright/underbright frames    │
 │   • ~500K parameters                         │
 └──────────────────────────────────────────────┘
         │
         ▼
 ┌──────────────────────────────────────────────┐
 │  Stage 1.5: Dynamic Exposure (optional)      │
 │   • Analyzes frame luminance (LAB L-channel) │
 │   • Auto-adjusts gamma + shadow lift         │
 │   • EMA-smooths brightness across frames     │
 └──────────────────────────────────────────────┘
         │
         ▼
 ┌──────────────────────────────────────────────┐
 │  Stage 2: FastDVDnet-Lite                    │
 │   • Mild spatial denoise                     │
 │   • Optional 3-frame temporal input          │
 │   • Residual learning (preserves texture)    │
 │   • ~300K parameters                         │
 └──────────────────────────────────────────────┘
         │
         ▼
 ┌──────────────────────────────────────────────┐
 │  Stage 3: ContourNet (Temporal Blend)        │
 │   • Sobel edges + CNN reinforcement          │
 │   • EMA-smoothed sigmoid gate                │
 │   • Stable boundaries across time            │
 │   • ~100K parameters                         │
 └──────────────────────────────────────────────┘
         │
         ▼
 ┌──────────────────────────────────────────────┐
 │ FINAL: Enhanced Frame_t (Stable + Enhanced)  │
 └──────────────────────────────────────────────┘
```

---

## Installation

```bash
pip install torch torchvision numpy opencv-python scipy
```

**Requirements:** Python 3.8+, PyTorch 1.12+, CUDA 11.x recommended.

---

## Quick Start

### Basic (static mode — same as v2)

```python
from vision_enhance_v2_5 import VisionEnhancer

enhancer = VisionEnhancer(device="cuda")
enhanced = enhancer.enhance(frame)
```

### Full v2.5 (temporal + exposure + calibrated)

```python
enhancer = VisionEnhancer(
    device="cuda",
    temporal=True,          # Enable Stage 0 + temporal denoising/edges
    temporal_alpha=0.2,     # EMA smoothing factor
    exposure_comp=True,     # Enable Stage 1.5
    hdr_calibrated=True,    # Enable histogram-guided HDRNet
    use_flow=False,         # Optical flow for PTZ cameras
)

# Process video frames sequentially
for frame in video_frames:
    enhanced = enhancer.enhance(frame)

# Reset on scene cut
enhancer.reset_temporal()
```

### Configurable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | `"cuda"` | `"cuda"` or `"cpu"` |
| `temporal` | `False` | Enable temporal features |
| `temporal_alpha` | `0.2` | EMA weight (0.1–0.5) |
| `exposure_comp` | `False` | Dynamic exposure compensation |
| `hdr_calibrated` | `False` | Histogram-guided tone mapping |
| `use_flow` | `False` | Optical flow alignment |
| `denoise_sigma` | `15.0` | Noise level (0–255 scale) |
| `contour_strength` | `1.0` | Edge blend strength (0–1) |
| `max_side` | `0` | Downscale limit (0 = native) |

---

## How Each New Feature Works

### Temporal Stabilization (Stage 0)

Applies exponential moving average (EMA) across consecutive raw frames.
This smooths sensor noise jitter and brightness flicker **before** any
neural network processing, giving the models cleaner, more consistent input.

Optional optical-flow warping (Farneback via OpenCV) aligns the previous
EMA frame to the current frame's geometry — useful for PTZ cameras.

### HDRNet Calibration

Before running HDRNet, the calibration module:
1. Computes luminance statistics (mean, percentiles) via LAB L-channel
2. Derives a `tone_strength` scalar from the histogram
3. Scales HDRNet's enhancement residual by this scalar

Result: very dark frames get stronger enhancement; already-bright frames
get gentler processing. Prevents overbright/clipped output.

### Dynamic Exposure (Stage 1.5)

After HDRNet, this module:
1. Analyzes output luminance
2. Computes ideal gamma correction to reach target brightness (default: 120/255)
3. EMA-smooths the gamma value across frames (prevents brightness flicker)
4. Applies gentle shadow lift for very dark regions

### Temporal Denoising (Stage 2)

FastDVDnet-Lite's model already supports 3-frame temporal input (9-channel).
The v2.5 infer wrapper maintains a sliding buffer and automatically:
- Pads the first frames (duplicate-fill)
- Concatenates [t-2, t-1, t] for each frame

### Temporal Edge Blending (Stage 3)

ContourNet's sigmoid gate map (which controls where edges are reinforced)
is EMA-blended across frames. This prevents:
- Flickering edge highlights
- Unstable boundary positions
- Jittery contour artifacts in video

---

## Testing

```bash
# Full test suite (random-init weights)
python test_enhance_v2_5.py --skip-weights --cpu

# Save output
python test_enhance_v2_5.py --skip-weights --save output_v25.png
```

---

## Generate Weights

```bash
python generate_weights_v2_5.py
```

Produces smart-initialized weights (~1.2 MB total) that work without training.

---

## Deployment (ONNX + TensorRT)

### Export to ONNX

```bash
python -m vision_enhance_v2_5.deploy.export_onnx --all
```

### Build TensorRT FP16 Engines

```bash
python -m vision_enhance_v2_5.deploy.build_trt --all --onnx-dir onnx_models/
```

### Benchmark

```bash
python -m vision_enhance_v2_5.deploy.benchmark --resolution 640x480 --frames 50
python -m vision_enhance_v2_5.deploy.benchmark --temporal --exposure --calibrated
```

---

## File Structure

```
vision_enhance_v2_5/
├── __init__.py
├── enhancer.py                  # Main orchestrator
├── utils.py                     # Tensor/image utilities
│
├── temporal/                    # Stage 0: Temporal Stabilization
│   ├── stabilizer.py            # EMA frame fusion + flow
│   ├── ema_buffer.py            # Generic EMA buffer
│   └── optical_flow.py          # Farneback flow + warping
│
├── exposure/                    # Stage 1.5: Dynamic Exposure
│   ├── dynamic_exposure.py      # Auto-gamma + shadow lift
│   └── histogram_utils.py       # LAB luminance analysis
│
├── hdrnet/                      # Stage 1: Tone Mapping
│   ├── model.py                 # HDRNet bilateral grid
│   ├── infer.py                 # Calibrated inference
│   └── weights.pth
│
├── fastdvdnet/                  # Stage 2: Denoising
│   ├── model_lite.py            # 2-stage residual denoiser
│   ├── infer.py                 # Temporal sliding window
│   └── weights.pth
│
├── contournet/                  # Stage 3: Edge Reinforcement
│   ├── model.py                 # Sobel + sigmoid-gated CNN
│   ├── infer.py                 # Temporal gate blending
│   └── weights.pth
│
├── deploy/                      # Deployment tools
│   ├── export_onnx.py           # ONNX export CLI
│   ├── build_trt.py             # TensorRT engine builder
│   └── benchmark.py             # FPS benchmarking
│
└── README_v2_5.md               # This file
```

---

## Performance

| Metric | Value |
|--------|-------|
| Total model params | ~900K |
| GPU VRAM | <500 MB |
| Speed (GPU, 640×480) | ~30–50 ms/frame |
| Temporal overhead | +5–15 ms (EMA + buffer) |
| Exposure overhead | +1–2 ms (histogram + gamma LUT) |
| FP16 autocast | Enabled on CUDA |

---

## License

MIT License. Model architectures inspired by:
- HDRNet: Gharbi et al., SIGGRAPH 2017
- FastDVDnet: Tassano et al., CVPR 2020
- ContourNet: Custom edge reinforcement network
