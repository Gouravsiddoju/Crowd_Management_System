"""
ONNX Export Tool
=================
Export individual v2.5 models (HDRNet, FastDVDnet-Lite, ContourNet)
to ONNX format with dynamic height/width axes.

Usage::

    python -m vision_enhance_v2_5.deploy.export_onnx --all
    python -m vision_enhance_v2_5.deploy.export_onnx --hdrnet
    python -m vision_enhance_v2_5.deploy.export_onnx --fastdvdnet
    python -m vision_enhance_v2_5.deploy.export_onnx --contournet
"""

import argparse
import os
import sys

import torch
import numpy as np


def export_hdrnet(output_dir: str, opset: int = 14):
    """Export HDRNet to ONNX."""
    from vision_enhance_v2_5.hdrnet.model import HDRNet

    print("[ONNX] Exporting HDRNet...")
    model = HDRNet()
    model.eval()

    # Load weights if available
    wpath = os.path.join("vision_enhance_v2_5", "hdrnet", "weights.pth")
    if os.path.isfile(wpath):
        model.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True), strict=False)

    dummy = torch.randn(1, 3, 480, 640).clamp(0, 1)
    out_path = os.path.join(output_dir, "hdrnet.onnx")

    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"},
        },
        opset_version=opset,
    )

    # Validate
    y = model(dummy)
    print(f"  [OK] HDRNet → {out_path} (shape: {dummy.shape} → {y.shape})")
    return out_path


def export_fastdvdnet(output_dir: str, opset: int = 14):
    """Export FastDVDnet-Lite (single-frame mode) to ONNX."""
    from vision_enhance_v2_5.fastdvdnet.model_lite import FastDVDnetLite

    print("[ONNX] Exporting FastDVDnet-Lite...")
    model = FastDVDnetLite(temporal=False)
    model.eval()

    wpath = os.path.join("vision_enhance_v2_5", "fastdvdnet", "weights.pth")
    if os.path.isfile(wpath):
        model.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True), strict=False)

    dummy_x = torch.randn(1, 3, 480, 640).clamp(0, 1)
    dummy_sigma = torch.full((1, 1, 480, 640), 0.06)
    out_path = os.path.join(output_dir, "fastdvdnet_lite.onnx")

    # Wrap model to accept single concatenated input for ONNX
    class DVDNetWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x, sigma_map):
            return self.model(x, sigma_map)

    wrapper = DVDNetWrapper(model)
    torch.onnx.export(
        wrapper, (dummy_x, dummy_sigma), out_path,
        input_names=["input", "sigma_map"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "sigma_map": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"},
        },
        opset_version=opset,
    )

    y = model(dummy_x, dummy_sigma)
    print(f"  [OK] FastDVDnet-Lite → {out_path} (shape: {dummy_x.shape} → {y.shape})")
    return out_path


def export_contournet(output_dir: str, opset: int = 14):
    """Export ContourNet to ONNX."""
    from vision_enhance_v2_5.contournet.model import ContourNet

    print("[ONNX] Exporting ContourNet...")
    model = ContourNet()
    model.eval()

    wpath = os.path.join("vision_enhance_v2_5", "contournet", "weights.pth")
    if os.path.isfile(wpath):
        model.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True), strict=False)

    dummy = torch.randn(1, 3, 480, 640).clamp(0, 1)
    out_path = os.path.join(output_dir, "contournet.onnx")

    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"},
        },
        opset_version=opset,
    )

    y = model(dummy)
    print(f"  [OK] ContourNet → {out_path} (shape: {dummy.shape} → {y.shape})")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Export v2.5 models to ONNX")
    parser.add_argument("--output", default="onnx_models", help="Output directory")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    parser.add_argument("--all", action="store_true", help="Export all models")
    parser.add_argument("--hdrnet", action="store_true", help="Export HDRNet")
    parser.add_argument("--fastdvdnet", action="store_true", help="Export FastDVDnet")
    parser.add_argument("--contournet", action="store_true", help="Export ContourNet")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.all or args.hdrnet:
        export_hdrnet(args.output, args.opset)
    if args.all or args.fastdvdnet:
        export_fastdvdnet(args.output, args.opset)
    if args.all or args.contournet:
        export_contournet(args.output, args.opset)

    if not (args.all or args.hdrnet or args.fastdvdnet or args.contournet):
        print("No model selected. Use --all or --hdrnet/--fastdvdnet/--contournet")
        sys.exit(1)

    print(f"\n[OK] ONNX exports saved to {args.output}/")


if __name__ == "__main__":
    main()
