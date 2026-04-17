"""
Convert YOLO models to ONNX (and optionally TensorRT engine).

Usage:
    python convert_onnx.py                     # convert all 3 models
    python convert_onnx.py --model path/to.pt  # single model
    python convert_onnx.py --trt               # also build TensorRT engine (GPU only)
    python convert_onnx.py --imgsz 640         # export at 640px (faster, less accurate)
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))


DEFAULT_MODELS = [
    "C:\\Users\\shche\\Desktop\\Application_for_models\\models\\BestBoots_v2.pt",
    "C:\\Users\\shche\\Desktop\\Application_for_models\\models\\person.pt",
    "C:\\Users\\shche\\Desktop\\Application_for_models\\models\\BestPose.pt",
]


def parse_args():
    p = argparse.ArgumentParser(description="Export YOLO .pt → ONNX (+ optional TensorRT)")
    p.add_argument("--model",  nargs="+", default=None,
                   help="Path(s) to .pt files. Defaults to all 3 project models.")
    p.add_argument("--imgsz",  type=int, default=640,
                   help="Export image size (square). 640 recommended for speed.")
    p.add_argument("--batch",  type=int, default=1)
    p.add_argument("--half",   action="store_true",
                   help="FP16 export (GPU only, faster on NVIDIA)")
    p.add_argument("--trt",    action="store_true",
                   help="Also export TensorRT .engine after ONNX (requires CUDA)")
    p.add_argument("--opset",  type=int, default=17,
                   help="ONNX opset version (17 recommended for YOLOv8)")
    p.add_argument("--dynamic", action="store_true",
                   help="Dynamic batch/image size axes")
    p.add_argument("--simplify", action="store_true", default=True,
                   help="Run onnx-simplifier after export (cleaner graph)")
    p.add_argument("--no-simplify", dest="simplify", action="store_false")
    return p.parse_args()


def export_model(pt_path: str, args) -> str | None:
    """Export a single .pt model to ONNX. Returns path to .onnx or None on failure."""
    from ultralytics import YOLO

    if not os.path.exists(pt_path):
        print(f"  [SKIP] Not found: {pt_path}")
        return None

    size_mb = os.path.getsize(pt_path) / 1024 / 1024
    print(f"\n  Model  : {pt_path}  ({size_mb:.1f} MB)")
    print(f"  imgsz  : {args.imgsz}   half={args.half}   opset={args.opset}")

    model = YOLO(pt_path)
    t0    = time.time()

    try:
        exported = model.export(
            format   = "onnx",
            imgsz    = args.imgsz,
            batch    = args.batch,
            half     = args.half and torch.cuda.is_available(),
            opset    = args.opset,
            dynamic  = args.dynamic,
            simplify = args.simplify,
        )
    except Exception as exc:
        print(f"  [ERROR] Export failed: {exc}")
        return None

    elapsed = time.time() - t0
    onnx_path = str(exported)

    if not os.path.exists(onnx_path):
        # Ultralytics sometimes returns the path differently
        onnx_path = pt_path.replace(".pt", ".onnx")

    if os.path.exists(onnx_path):
        onnx_mb = os.path.getsize(onnx_path) / 1024 / 1024
        print(f"  [OK]   ONNX saved: {onnx_path}  ({onnx_mb:.1f} MB)  [{elapsed:.1f}s]")
    else:
        print(f"  [WARN] Could not locate output ONNX file")
        return None

    return onnx_path


def export_trt(onnx_path: str, args) -> str | None:
    """Export ONNX → TensorRT .engine."""
    if not torch.cuda.is_available():
        print("  [SKIP] TensorRT requires CUDA — no GPU detected")
        return None

    from ultralytics import YOLO
    print(f"\n  Building TensorRT engine from {onnx_path} ...")
    model = YOLO(onnx_path)
    t0    = time.time()
    try:
        exported = model.export(
            format  = "engine",
            imgsz   = args.imgsz,
            batch   = args.batch,
            half    = args.half,
            dynamic = args.dynamic,
        )
    except Exception as exc:
        print(f"  [ERROR] TRT export failed: {exc}")
        return None

    elapsed    = time.time() - t0
    trt_path   = str(exported)
    if os.path.exists(trt_path):
        trt_mb = os.path.getsize(trt_path) / 1024 / 1024
        print(f"  [OK]   TRT engine: {trt_path}  ({trt_mb:.1f} MB)  [{elapsed:.1f}s]")
    return trt_path


def main():
    args        = parse_args()
    model_paths = args.model if args.model else DEFAULT_MODELS

    has_cuda = torch.cuda.is_available()
    print("=" * 60)
    print("  YOLO → ONNX Converter")
    print("=" * 60)
    print(f"  Device     : {'CUDA (' + torch.cuda.get_device_name(0) + ')' if has_cuda else 'CPU'}")
    print(f"  Models     : {len(model_paths)}")
    print(f"  imgsz      : {args.imgsz}")
    print(f"  FP16 half  : {args.half and has_cuda}")
    print(f"  TensorRT   : {args.trt and has_cuda}")
    print("=" * 60)

    results = []
    for pt_path in model_paths:
        onnx_path = export_model(pt_path, args)
        if onnx_path and args.trt:
            trt_path = export_trt(onnx_path, args)
            results.append((pt_path, onnx_path, trt_path))
        else:
            results.append((pt_path, onnx_path, None))

    print("\n" + "=" * 60)
    print("  Summary:")
    for pt, onnx, trt in results:
        name = os.path.basename(pt)
        onnx_status = "✓ " + os.path.basename(onnx) if onnx else "✗ failed"
        trt_status  = ("✓ " + os.path.basename(trt)) if trt else ("✗ skipped" if args.trt else "—")
        print(f"    {name:<25} ONNX: {onnx_status:<30} TRT: {trt_status}")

    print("=" * 60)
    print("\n  Next step — run benchmark:")
    print("    python benchmark_onnx.py --input video.mp4")


if __name__ == "__main__":
    main()
