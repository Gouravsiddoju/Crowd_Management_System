"""
TensorRT FP16 Engine Builder
==============================
Converts ONNX models to TensorRT engines with FP16 precision.
Requires TensorRT to be installed (``pip install tensorrt``).

Usage::

    python -m vision_enhance_v2_5.deploy.build_trt --onnx onnx_models/hdrnet.onnx
    python -m vision_enhance_v2_5.deploy.build_trt --onnx onnx_models/hdrnet.onnx --fp16
    python -m vision_enhance_v2_5.deploy.build_trt --all --onnx-dir onnx_models/
"""

import argparse
import os
import sys


def build_engine(
    onnx_path: str,
    engine_path: str = None,
    fp16: bool = True,
    min_shape: tuple = (1, 3, 240, 320),
    opt_shape: tuple = (1, 3, 480, 640),
    max_shape: tuple = (1, 3, 1080, 1920),
    workspace_mb: int = 512,
):
    """Build a TensorRT engine from an ONNX model.

    Args:
        onnx_path: Path to the ONNX model.
        engine_path: Output path for the TRT engine. If None, auto-generated.
        fp16: Enable FP16 precision.
        min_shape: Minimum input shape for dynamic axes.
        opt_shape: Optimal input shape.
        max_shape: Maximum input shape.
        workspace_mb: Workspace size in MB.

    Returns:
        Path to the built engine file.
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("[ERROR] TensorRT not installed. Install with: pip install tensorrt")
        print("        Or use trtexec CLI: trtexec --onnx=model.onnx --saveEngine=model.trt --fp16")
        return None

    if engine_path is None:
        engine_path = onnx_path.replace(".onnx", ".trt")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    print(f"[TRT] Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Parse error: {parser.get_error(i)}")
            return None

    # Build config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print(f"  FP16 enabled")
    elif fp16:
        print(f"  [WARN] FP16 not supported on this GPU, using FP32")

    # Dynamic shapes
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    print(f"  Building engine (this may take a few minutes)...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print(f"  [FAIL] Engine build failed")
        return None

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"  [OK] Engine saved: {engine_path} ({size_mb:.1f} MB)")
    return engine_path


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT FP16 engines from ONNX")
    parser.add_argument("--onnx", type=str, help="Single ONNX file to convert")
    parser.add_argument("--all", action="store_true", help="Convert all ONNX files in --onnx-dir")
    parser.add_argument("--onnx-dir", default="onnx_models", help="Directory with ONNX files")
    parser.add_argument("--output-dir", default=None, help="Output directory for engines")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 (default)")
    parser.add_argument("--fp32", action="store_true", help="Force FP32")
    parser.add_argument("--workspace", type=int, default=512, help="Workspace MB")
    args = parser.parse_args()

    use_fp16 = not args.fp32

    if args.onnx:
        out_dir = args.output_dir or os.path.dirname(args.onnx)
        engine_path = os.path.join(out_dir, os.path.basename(args.onnx).replace(".onnx", ".trt"))
        build_engine(args.onnx, engine_path, fp16=use_fp16, workspace_mb=args.workspace)
    elif args.all:
        onnx_dir = args.onnx_dir
        out_dir = args.output_dir or onnx_dir
        if not os.path.isdir(onnx_dir):
            print(f"[ERROR] ONNX directory not found: {onnx_dir}")
            sys.exit(1)
        for fname in sorted(os.listdir(onnx_dir)):
            if fname.endswith(".onnx"):
                onnx_path = os.path.join(onnx_dir, fname)
                engine_path = os.path.join(out_dir, fname.replace(".onnx", ".trt"))
                build_engine(onnx_path, engine_path, fp16=use_fp16, workspace_mb=args.workspace)
    else:
        print("Specify --onnx <file> or --all. Use --help for options.")
        sys.exit(1)


if __name__ == "__main__":
    main()
