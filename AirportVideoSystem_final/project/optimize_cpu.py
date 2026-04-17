"""
CPU Optimization: ONNX + OpenVINO + INT8 quantization

На CPU PyTorch быстрее голого ONNX Runtime, потому что использует MKL-DNN.
Настоящее ускорение на CPU дают:
  1. OpenVINO Execution Provider  (~2-4x vs PT на Intel CPU)
  2. INT8 квантизация             (~1.5-2x vs FP32, небольшая потеря качества)

Usage:
    # Установить зависимости:
    pip install onnxruntime openvino-dev onnxruntime-openvino

    # Экспорт в OpenVINO формат:
    python optimize_cpu.py --mode openvino

    # INT8 квантизация ONNX:
    python optimize_cpu.py --mode int8 --input video.mp4

    # Бенчмарк всех форматов:
    python optimize_cpu.py --mode benchmark --input video.mp4
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

MODELS_PT = {
    "wheels": "C:\\Users\\shche\\Desktop\\Application_for_models\\models\\BestBoots_v2.pt",
    "person": "C:\\Users\\shche\\Desktop\\Application_for_models\\models\\person.pt",
    "pose":   "C:\\Users\\shche\\Desktop\\Application_for_models\\models\\BestPose.pt",
}

def onnx_path(pt): return pt.replace(".pt", ".onnx")
def ovino_path(pt): return pt.replace(".pt", "_openvino_model")
def int8_path(pt):  return pt.replace(".pt", "_int8.onnx")


# ─────────────────────────────────────────────────────────────────────────────
# 1. OpenVINO export
# ─────────────────────────────────────────────────────────────────────────────

def export_openvino(args):
    """Export .pt → OpenVINO IR via Ultralytics."""
    from ultralytics import YOLO
    print("\n[OpenVINO Export]")
    print("  Exports .pt directly to OpenVINO IR (XML + BIN)")
    print("  Best for Intel CPU / iGPU\n")

    for name, pt in MODELS_PT.items():
        if not os.path.exists(pt):
            print(f"  [SKIP] {name}: {pt} not found")
            continue
        print(f"  Exporting {name} ({os.path.basename(pt)})...")
        model = YOLO(pt)
        t0 = time.time()
        out = model.export(format="openvino", imgsz=args.imgsz, half=False)
        elapsed = time.time() - t0
        print(f"  → {out}  [{elapsed:.1f}s]")


# ─────────────────────────────────────────────────────────────────────────────
# 2. INT8 quantization
# ─────────────────────────────────────────────────────────────────────────────

def export_int8(args):
    """Quantize FP32 ONNX → INT8 ONNX using onnxruntime quantization."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("[ERROR] pip install onnxruntime  (>=1.16)")
        return

    print("\n[INT8 Dynamic Quantization]")
    print("  Quantizes weights to INT8 (activations stay FP32 at runtime)")
    print("  ~1.5-2x speedup on CPU, model size ~4x smaller\n")

    for name, pt in MODELS_PT.items():
        op = onnx_path(pt)
        if not os.path.exists(op):
            print(f"  [SKIP] {name}: ONNX not found ({op})")
            print(f"         Run: python convert_onnx.py --imgsz {args.imgsz}")
            continue

        out = int8_path(pt)
        print(f"  Quantizing {os.path.basename(op)} → {os.path.basename(out)} ...")
        t0 = time.time()
        try:
            quantize_dynamic(
                model_input=op,
                model_output=out,
                weight_type=QuantType.QInt8,
                per_channel=False,
            )
            elapsed = time.time() - t0
            size_fp32 = os.path.getsize(op)  / 1024 / 1024
            size_int8 = os.path.getsize(out) / 1024 / 1024
            print(f"  → {out}")
            print(f"     FP32: {size_fp32:.1f} MB  →  INT8: {size_int8:.1f} MB  "
                  f"(x{size_fp32/size_int8:.1f} smaller)  [{elapsed:.1f}s]")
        except Exception as e:
            print(f"  [ERROR] {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Benchmark all formats
# ─────────────────────────────────────────────────────────────────────────────

def sample_frames(video_path: str, n: int) -> list:
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, total // n)
    frames, idx = [], 0
    while cap.isOpened() and len(frames) < n:
        ret, frame = cap.read()
        if not ret: break
        if idx % step == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames[:n]


def bench_format(model, frames, imgsz, warmup=5, label=""):
    """Returns mean ms over frames (after warmup)."""
    all_frames = frames[:warmup] + frames
    times = []
    for i, f in enumerate(all_frames):
        t0 = time.perf_counter()
        model.predict(f, imgsz=imgsz, conf=0.25, verbose=False)
        ms = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            times.append(ms)
        if i % 20 == 0:
            print(f"\r    {label}  {i-warmup+1}/{len(frames)} frames  {ms:.0f}ms",
                  end="", flush=True)
    print(f"\r    {label}  {len(frames)}/{len(frames)} done" + " " * 20)
    return float(np.mean(times)), float(np.min(times)), float(np.max(times))


def benchmark_all(args):
    from ultralytics import YOLO
    import onnxruntime as ort

    print(f"\n[Benchmark — {args.frames} frames, imgsz={args.imgsz}]")
    print(f"  Available ONNX providers: {ort.get_available_providers()}\n")

    frames = sample_frames(args.input, args.frames + 5)
    if not frames:
        print("[ERROR] Cannot read video"); return

    # ── Detect available providers ────────────────────────────────────────
    providers   = ort.get_available_providers()
    has_ovino   = "OpenVINOExecutionProvider"   in providers
    has_cuda    = "CUDAExecutionProvider"        in providers
    has_trt     = "TensorrtExecutionProvider"    in providers

    print(f"  OpenVINO provider : {'✓' if has_ovino else '✗ (pip install onnxruntime-openvino)'}")
    print(f"  CUDA provider     : {'✓' if has_cuda  else '✗'}")
    print(f"  TRT provider      : {'✓' if has_trt   else '✗'}\n")

    col_w = [28, 10, 9, 9, 9, 9]
    hdr   = ["Model", "Format", "mean ms", "min", "max", "FPS"]
    sep   = "  ".join("─" * w for w in col_w)
    row   = "  ".join(f"{{:<{w}}}" for w in col_w)

    print(sep)
    print(row.format(*hdr))
    print(sep)

    results = []  # (name, fmt, mean_ms)

    for name, pt in MODELS_PT.items():
        base = os.path.basename(pt)

        # PT
        if os.path.exists(pt):
            mean, mn, mx = bench_format(YOLO(pt), frames, args.imgsz, label=f"[pt/{name}]")
            fps = 1000 / mean
            print(row.format(base, "pt", f"{mean:.1f}", f"{mn:.1f}", f"{mx:.1f}", f"{fps:.1f}"))
            results.append((name, "pt", mean))

        # ONNX CPU
        op = onnx_path(pt)
        if os.path.exists(op):
            mean, mn, mx = bench_format(YOLO(op), frames, args.imgsz, label=f"[onnx/{name}]")
            fps = 1000 / mean
            print(row.format(base.replace(".pt",".onnx"), "onnx-cpu",
                             f"{mean:.1f}", f"{mn:.1f}", f"{mx:.1f}", f"{fps:.1f}"))
            results.append((name, "onnx-cpu", mean))

        # ONNX + OpenVINO Execution Provider
        # Falls back to CPU silently if openvino.dll is missing —
        # we detect that case and mark results accordingly.
        if os.path.exists(op) and has_ovino:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Suppress the DLL-missing stderr noise
                    import io as _io, contextlib as _ctx
                    _buf = _io.StringIO()
                    with _ctx.redirect_stderr(_buf):
                        sess = ort.InferenceSession(
                            op,
                            providers=["OpenVINOExecutionProvider",
                                       "CPUExecutionProvider"])
                active_providers = sess.get_providers()
                used_ov = "OpenVINOExecutionProvider" in active_providers
                provider_tag = "onnx-openvino" if used_ov else "onnx-openvino(cpu)"

                inp_name = sess.get_inputs()[0].name
                imgsz_ov = args.imgsz

                mean_ms_list = []
                for f in frames:
                    img = cv2.resize(f, (imgsz_ov, imgsz_ov))
                    img = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
                    t0  = time.perf_counter()
                    sess.run(None, {inp_name: img})
                    mean_ms_list.append((time.perf_counter() - t0) * 1000)

                mean = float(np.mean(mean_ms_list))
                fps  = 1000 / mean
                note = "" if used_ov else "  ← OV DLL missing, ran on CPU"
                print(row.format(base.replace(".pt", ".onnx"), provider_tag,
                                 f"{mean:.1f}", f"{min(mean_ms_list):.1f}",
                                 f"{max(mean_ms_list):.1f}", f"{fps:.1f}") + note)
                if not used_ov:
                    print("    ⚠ openvino.dll not found → install: pip install openvino")
                results.append((name, provider_tag, mean))
            except Exception as e:
                print(f"  [OpenVINO provider error] {e}")

        # INT8 ONNX
        i8 = int8_path(pt)
        if os.path.exists(i8):
            mean, mn, mx = bench_format(YOLO(i8), frames, args.imgsz,
                                        label=f"[int8/{name}]")
            fps = 1000 / mean
            print(row.format(base.replace(".pt", "_int8.onnx"), "int8",
                             f"{mean:.1f}", f"{mn:.1f}", f"{mx:.1f}", f"{fps:.1f}"))
            results.append((name, "int8", mean))

        # OpenVINO IR  — Ultralytics expects the DIRECTORY, not the .xml file
        ov_dir = ovino_path(pt)
        if os.path.exists(ov_dir):
            try:
                mean, mn, mx = bench_format(
                    YOLO(ov_dir),          # pass directory, not .xml
                    frames, args.imgsz,
                    label=f"[ovino/{name}]")
                fps = 1000 / mean
                print(row.format(os.path.basename(ov_dir), "openvino-ir",
                                 f"{mean:.1f}", f"{mn:.1f}", f"{mx:.1f}", f"{fps:.1f}"))
                results.append((name, "openvino-ir", mean))
            except Exception as e:
                print(f"  [OpenVINO IR error] {e}")

        print()

    print(sep)

    # ── Combined throughput ───────────────────────────────────────────────
    print("\n  Combined (all 3 models, same frame):")
    fmt_totals = {}
    for _, fmt, ms in results:
        fmt_totals[fmt] = fmt_totals.get(fmt, 0) + ms
    baseline = fmt_totals.get("pt", 1)
    for fmt, total_ms in sorted(fmt_totals.items(), key=lambda x: x[1]):
        fps     = 1000 / total_ms
        speedup = baseline / total_ms
        print(f"    {fmt:<18} {total_ms:.0f} ms/frame  →  {fps:.1f} fps  "
              f"({speedup:.2f}x vs PT)")

    print("\n  Recommendation:")
    best = min(fmt_totals, key=fmt_totals.get)
    print(f"    Fastest format on this machine: [{best}]  "
          f"({1000/fmt_totals[best]:.1f} fps combined)")
    if best != "pt":
        suffix_map = {
            "onnx-cpu":      ".onnx",
            "onnx-openvino": ".onnx",
            "int8":          "_int8.onnx",
            "openvino-ir":   "_openvino_model/<name>.xml",
        }
        ext = suffix_map.get(best, ".onnx")
        print(f"    Use {ext} paths in process_video.py / web_app.py")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",   choices=["openvino","int8","benchmark","all"],
                   default="benchmark")
    p.add_argument("--input",  default=None, help="Video for benchmark frames")
    p.add_argument("--imgsz",  type=int, default=640)
    p.add_argument("--frames", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print("  CPU Optimization Tool")
    print("=" * 65)

    if args.mode in ("openvino", "all"):
        export_openvino(args)
    if args.mode in ("int8", "all"):
        export_int8(args)
    if args.mode in ("benchmark", "all"):
        if not args.input:
            print("[ERROR] --input required for benchmark")
            return
        benchmark_all(args)


if __name__ == "__main__":
    main()
