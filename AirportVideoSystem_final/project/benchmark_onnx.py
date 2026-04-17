"""
Benchmark: PyTorch (.pt) vs ONNX (.onnx) vs TensorRT (.engine)

Measures both SPEED and QUALITY for each model format.

Speed metrics (per model, per format):
  - mean / min / max / p95 inference time, FPS, speedup vs PT

Quality metrics (ONNX/TRT vs PT baseline on identical frames):
  - Detection count match rate
  - Mean IoU of matched bounding boxes
  - Mean confidence delta
  - Class label agreement rate
  - Overall quality score (0-100)

Usage:
    python benchmark_onnx.py --input video.mp4
    python benchmark_onnx.py --input video.mp4 --frames 150 --imgsz 640
    python benchmark_onnx.py --input video.mp4 --skip-trt
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# Model paths
# ─────────────────────────────────────────────────────────────────────────────

MODELS_PT = {
    "wheels": "C:\\Users\\shche\\Desktop\\Application_for_models\\models\\BestBoots_v2.pt",
    "person": "C:\\Users\\shche\\Desktop\\Application_for_models\\models\\person.pt",
    "pose":   "C:\\Users\\shche\\Desktop\\Application_for_models\\models\\BestPose.pt",
}

def onnx_path(pt: str) -> str:
    return pt.replace(".pt", ".onnx")

def trt_path(pt: str) -> str:
    return pt.replace(".pt", ".engine")


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Speed + Quality benchmark: PT vs ONNX vs TRT")
    p.add_argument("--input",    required=True, help="Video file for sample frames")
    p.add_argument("--frames",   type=int, default=150)
    p.add_argument("--warmup",   type=int, default=10)
    p.add_argument("--imgsz",    type=int, default=640)
    p.add_argument("--conf",     type=float, default=0.25)
    p.add_argument("--iou-match", type=float, default=0.5,
                   help="IoU threshold to consider two boxes as matched")
    p.add_argument("--skip-trt", action="store_true")
    p.add_argument("--save-csv", action="store_true",
                   help="Save detailed per-frame results to benchmark_results.csv")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    x1: float; y1: float; x2: float; y2: float
    conf: float; cls: int


@dataclass
class SpeedResult:
    model_name: str
    fmt: str
    times_ms: list = field(default_factory=list)

    @property
    def mean_ms(self): return float(np.mean(self.times_ms)) if self.times_ms else 0.0
    @property
    def min_ms(self):  return float(np.min(self.times_ms))  if self.times_ms else 0.0
    @property
    def max_ms(self):  return float(np.max(self.times_ms))  if self.times_ms else 0.0
    @property
    def p95_ms(self):
        return float(np.percentile(self.times_ms, 95)) if self.times_ms else 0.0
    @property
    def fps(self):     return 1000.0 / self.mean_ms if self.mean_ms > 0 else 0.0


@dataclass
class QualityResult:
    model_name: str
    fmt: str                           # "onnx" or "trt"
    n_frames: int = 0
    # per-frame quality lists
    count_match_rates: list = field(default_factory=list)  # fraction of frames where #dets matches
    matched_ious: list      = field(default_factory=list)  # IoU of matched box pairs
    conf_deltas: list       = field(default_factory=list)  # |conf_onnx - conf_pt|
    cls_agreements: list    = field(default_factory=list)  # 1/0 per matched pair
    unmatched_pt: list      = field(default_factory=list)  # missed detections (false negatives)
    unmatched_fmt: list     = field(default_factory=list)  # extra detections (false positives)

    def mean(self, lst): return float(np.mean(lst)) if lst else float("nan")

    @property
    def mean_iou(self):        return self.mean(self.matched_ious)
    @property
    def mean_conf_delta(self): return self.mean(self.conf_deltas)
    @property
    def cls_agreement(self):   return self.mean(self.cls_agreements)
    @property
    def count_match_rate(self):return self.mean(self.count_match_rates)
    @property
    def false_neg_rate(self):
        total_pt = sum(self.unmatched_pt) + len(self.matched_ious)
        return sum(self.unmatched_pt) / max(total_pt, 1)
    @property
    def false_pos_rate(self):
        total_fmt = sum(self.unmatched_fmt) + len(self.matched_ious)
        return sum(self.unmatched_fmt) / max(total_fmt, 1)

    @property
    def quality_score(self) -> float:
        """Composite 0-100 quality score vs PT baseline."""
        if not self.matched_ious:
            return 0.0
        iou_score   = self.mean_iou * 40          # max 40 pts
        cls_score   = self.cls_agreement * 25     # max 25 pts
        fn_score    = (1 - self.false_neg_rate) * 20   # max 20 pts
        fp_score    = (1 - min(self.false_pos_rate, 1)) * 10  # max 10 pts
        conf_score  = max(0, 1 - self.mean_conf_delta / 0.1) * 5  # max 5 pts
        return round(iou_score + cls_score + fn_score + fp_score + conf_score, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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


def parse_detections(result) -> list[Detection]:
    """Extract Detection objects from a YOLO result."""
    dets = []
    if result.boxes is None:
        return dets
    boxes  = result.boxes.xyxy.cpu().numpy()   if result.boxes.xyxy  is not None else []
    confs  = result.boxes.conf.cpu().numpy()   if result.boxes.conf  is not None else []
    clsids = result.boxes.cls.cpu().numpy()    if result.boxes.cls   is not None else []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        dets.append(Detection(x1, y1, x2, y2, float(confs[i]), int(clsids[i])))
    return dets


def box_iou(a: Detection, b: Detection) -> float:
    ix1 = max(a.x1, b.x1); iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2); iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter / (area_a + area_b - inter)


def match_detections(pt_dets: list[Detection],
                     fmt_dets: list[Detection],
                     iou_thresh: float) -> tuple:
    """
    Greedy IoU matching between PT and format detections.
    Returns (matched_pairs, unmatched_pt_count, unmatched_fmt_count).
    """
    matched = []
    used_fmt = set()

    for pt_d in pt_dets:
        best_iou, best_j = 0.0, -1
        for j, fmt_d in enumerate(fmt_dets):
            if j in used_fmt:
                continue
            iou = box_iou(pt_d, fmt_d)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thresh:
            matched.append((pt_d, fmt_dets[best_j], best_iou))
            used_fmt.add(best_j)

    unmatched_pt  = len(pt_dets)  - len(matched)
    unmatched_fmt = len(fmt_dets) - len(matched)
    return matched, unmatched_pt, unmatched_fmt


def load_model(path: str):
    from ultralytics import YOLO
    if not os.path.exists(path):
        return None
    try:
        return YOLO(path)
    except Exception as exc:
        print(f"    [WARN] Cannot load {path}: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Core benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_speed_benchmark(model, frames: list, imgsz: int, conf: float,
                        warmup: int, model_name: str, fmt: str) -> SpeedResult:
    result = SpeedResult(model_name=model_name, fmt=fmt)
    all_frames = frames[:warmup] + frames

    for i, frame in enumerate(all_frames):
        t0 = time.perf_counter()
        model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)
        ms = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            result.times_ms.append(ms)
        if (i - warmup) % 20 == 0 and i >= warmup:
            done = i - warmup + 1
            print(f"\r      {done}/{len(frames)}  {ms:.0f}ms/frame", end="", flush=True)

    print(f"\r      {len(frames)}/{len(frames)}  done{' '*20}")
    return result


def run_quality_comparison(pt_model, fmt_model,
                           frames: list, imgsz: int, conf: float,
                           iou_thresh: float,
                           model_name: str, fmt: str) -> QualityResult:
    """Run PT and fmt_model on identical frames, compare detections."""
    result = QualityResult(model_name=model_name, fmt=fmt, n_frames=len(frames))

    for i, frame in enumerate(frames):
        pt_preds  = pt_model.predict(frame,  imgsz=imgsz, conf=conf, verbose=False)
        fmt_preds = fmt_model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)

        pt_dets  = parse_detections(pt_preds[0])
        fmt_dets = parse_detections(fmt_preds[0])

        # count match
        result.count_match_rates.append(
            1.0 if len(pt_dets) == len(fmt_dets) else
            1.0 - abs(len(pt_dets) - len(fmt_dets)) / max(len(pt_dets), 1, len(fmt_dets))
        )

        matched, unm_pt, unm_fmt = match_detections(pt_dets, fmt_dets, iou_thresh)
        result.unmatched_pt.append(unm_pt)
        result.unmatched_fmt.append(unm_fmt)

        for pt_d, fmt_d, iou in matched:
            result.matched_ious.append(iou)
            result.conf_deltas.append(abs(pt_d.conf - fmt_d.conf))
            result.cls_agreements.append(1 if pt_d.cls == fmt_d.cls else 0)

        if i % 20 == 0:
            print(f"\r      quality {i+1}/{len(frames)}", end="", flush=True)

    print(f"\r      quality {len(frames)}/{len(frames)} done{' '*10}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_speed_table(results: list[SpeedResult]):
    W = [28, 6, 9, 8, 8, 8, 8, 9]
    hdr = ["Model", "fmt", "mean ms", "min", "max", "p95", "FPS", "speedup"]
    sep = "  ".join("─" * w for w in W)
    row = "  ".join(f"{{:<{w}}}" for w in W)
    print("\n" + sep)
    print(row.format(*hdr))
    print(sep)

    baseline: dict[str, float] = {}
    for r in results:
        key = r.model_name
        if r.fmt == "pt":
            baseline[key] = r.fps
        base   = baseline.get(key, r.fps)
        speedup = f"{r.fps / base:.2f}x" if base > 0 else "—"
        print(row.format(
            r.model_name[:W[0]], r.fmt,
            f"{r.mean_ms:.1f}", f"{r.min_ms:.1f}",
            f"{r.max_ms:.1f}", f"{r.p95_ms:.1f}",
            f"{r.fps:.1f}", speedup,
        ))
    print(sep)


def quality_bar(score: float, width: int = 20) -> str:
    filled = int(score / 100 * width)
    bar    = "█" * filled + "░" * (width - filled)
    color  = "✓" if score >= 90 else ("~" if score >= 75 else "✗")
    return f"[{bar}] {score:.1f}/100 {color}"


def print_quality_table(results: list[QualityResult]):
    print("\n  Quality vs PT baseline  (higher = closer to original PyTorch output)")
    print("  " + "─" * 80)
    hdr = f"  {'Model+Format':<32} {'Score':>8}  {'IoU':>6}  {'Cls%':>6}  {'FN%':>6}  {'FP%':>6}  {'ΔConf':>7}"
    print(hdr)
    print("  " + "─" * 80)

    for r in results:
        label = f"{r.model_name} [{r.fmt}]"
        fn_pct = r.false_neg_rate * 100
        fp_pct = r.false_pos_rate * 100
        iou    = r.mean_iou
        cls_ag = r.cls_agreement * 100
        dconf  = r.mean_conf_delta

        score_bar = quality_bar(r.quality_score)
        print(f"  {label:<32} {score_bar}")
        print(f"  {'':32}  IoU={iou:.3f}  Cls={cls_ag:.1f}%  "
              f"FN={fn_pct:.1f}%  FP={fp_pct:.1f}%  ΔConf={dconf:.4f}")
        print()


def verdict(quality_results: list[QualityResult]) -> None:
    print("  Verdict:")
    for r in quality_results:
        score = r.quality_score
        if score >= 90:
            v = f"SAFE to use — quality essentially identical to PT"
        elif score >= 75:
            v = f"ACCEPTABLE — minor differences, check FN/FP rates"
        elif score >= 55:
            v = f"DEGRADED — noticeable quality loss, use with caution"
        else:
            v = f"NOT RECOMMENDED — significant quality difference vs PT"
        print(f"    [{r.fmt}] {r.model_name}: {v}  (score={score})")


def save_csv(speed_results: list[SpeedResult],
             quality_results: list[QualityResult], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["section", "model", "format",
                    "mean_ms", "fps", "speedup",
                    "quality_score", "mean_iou", "cls_agreement_pct",
                    "false_neg_pct", "false_pos_pct", "conf_delta"])

        baseline: dict[str, float] = {}
        speed_map = {(r.model_name, r.fmt): r for r in speed_results}
        qual_map  = {(r.model_name, r.fmt): r for r in quality_results}

        for r in speed_results:
            if r.fmt == "pt":
                baseline[r.model_name] = r.fps

        for (name, fmt), sr in speed_map.items():
            base   = baseline.get(name, sr.fps)
            speedup = sr.fps / base if base > 0 else 1.0
            qr = qual_map.get((name, fmt))
            w.writerow([
                "benchmark", name, fmt,
                f"{sr.mean_ms:.2f}", f"{sr.fps:.2f}", f"{speedup:.3f}",
                f"{qr.quality_score:.1f}" if qr else "—",
                f"{qr.mean_iou:.4f}"       if qr else "—",
                f"{qr.cls_agreement*100:.1f}" if qr else "—",
                f"{qr.false_neg_rate*100:.1f}" if qr else "—",
                f"{qr.false_pos_rate*100:.1f}" if qr else "—",
                f"{qr.mean_conf_delta:.5f}" if qr else "—",
            ])
    print(f"\n  Detailed results saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    has_cuda = torch.cuda.is_available()

    print("=" * 65)
    print("  Speed + Quality Benchmark: PT vs ONNX vs TRT")
    print("=" * 65)
    print(f"  Device     : {'CUDA (' + torch.cuda.get_device_name(0) + ')' if has_cuda else 'CPU'}")
    print(f"  Video      : {args.input}")
    print(f"  Frames     : {args.frames} bench + {args.warmup} warmup")
    print(f"  imgsz      : {args.imgsz}   conf={args.conf}")
    print(f"  IoU match  : {args.iou_match}")
    print("=" * 65)

    frames = sample_frames(args.input, args.frames + args.warmup)
    bench_frames  = frames[args.warmup:]
    print(f"\n  Sampled {len(frames)} frames "
          f"({bench_frames[0].shape[1]}x{bench_frames[0].shape[0]})\n")

    speed_results:   list[SpeedResult]   = []
    quality_results: list[QualityResult] = []

    for name, pt in MODELS_PT.items():
        base_name = os.path.basename(pt)
        print(f"{'═'*65}")
        print(f"  Model: {name}  ({base_name})")
        print(f"{'═'*65}")

        pt_model = load_model(pt)
        if pt_model is None:
            print(f"  [SKIP] .pt not found: {pt}\n")
            continue

        # ── Speed: PT ────────────────────────────────────────────────────────
        print(f"\n  [1] Speed  — PyTorch (.pt)")
        sr_pt = run_speed_benchmark(
            pt_model, bench_frames, args.imgsz, args.conf,
            args.warmup, base_name, "pt")
        speed_results.append(sr_pt)
        print(f"      → {sr_pt.mean_ms:.1f} ms/frame  {sr_pt.fps:.1f} FPS")

        for fmt_label, fmt_path_fn in [("onnx", onnx_path),
                                        ("trt",  trt_path)]:
            if fmt_label == "trt" and args.skip_trt:
                continue

            path = fmt_path_fn(pt)
            if not os.path.exists(path):
                print(f"\n  [{fmt_label.upper()}] Not found: {path}")
                print(f"       Run: python convert_onnx.py --imgsz {args.imgsz}"
                      + (" --trt" if fmt_label == "trt" else ""))
                continue

            fmt_model = load_model(path)
            if fmt_model is None:
                continue

            # ── Speed ────────────────────────────────────────────────────────
            print(f"\n  [2] Speed  — {fmt_label.upper()} ({os.path.basename(path)})")
            sr = run_speed_benchmark(
                fmt_model, bench_frames, args.imgsz, args.conf,
                args.warmup, base_name, fmt_label)
            speed_results.append(sr)
            speedup = sr.fps / sr_pt.fps if sr_pt.fps > 0 else 1.0
            print(f"      → {sr.mean_ms:.1f} ms/frame  {sr.fps:.1f} FPS  "
                  f"({speedup:.2f}x vs PT)")

            # ── Quality ──────────────────────────────────────────────────────
            print(f"\n  [3] Quality — {fmt_label.upper()} vs PT (same {len(bench_frames)} frames)")
            qr = run_quality_comparison(
                pt_model, fmt_model, bench_frames,
                args.imgsz, args.conf, args.iou_match,
                base_name, fmt_label)
            quality_results.append(qr)
            print(f"      → score={qr.quality_score}/100  "
                  f"IoU={qr.mean_iou:.3f}  "
                  f"cls={qr.cls_agreement*100:.1f}%  "
                  f"FN={qr.false_neg_rate*100:.1f}%  "
                  f"FP={qr.false_pos_rate*100:.1f}%")

        print()

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SPEED RESULTS")
    print("=" * 65)
    print_speed_table(speed_results)

    # Combined FPS
    for fmt in ("pt", "onnx", "trt"):
        total_ms = sum(r.mean_ms for r in speed_results if r.fmt == fmt)
        if total_ms > 0:
            print(f"  Combined [{fmt:4s}] all models: {total_ms:.0f} ms/frame"
                  f"  →  {1000/total_ms:.1f} fps")

    if quality_results:
        print("\n" + "=" * 65)
        print("  QUALITY RESULTS")
        print("=" * 65)
        print_quality_table(quality_results)
        verdict(quality_results)

    # ── ONNX recommendation ───────────────────────────────────────────────────
    best_fmt = None
    best_score = -1
    for r in quality_results:
        if r.quality_score > best_score:
            best_score = r.quality_score
            best_fmt   = r.fmt

    if best_fmt and best_score >= 75:
        print(f"\n  → Use {best_fmt.upper()} models in process_video.py:")
        fn = onnx_path if best_fmt == "onnx" else trt_path
        for name, pt in MODELS_PT.items():
            print(f"    --model-{name:<7} \"{fn(pt)}\"")
    elif quality_results:
        print("\n  → Quality below threshold; stick with PyTorch .pt models.")

    if args.save_csv:
        save_csv(speed_results, quality_results, "benchmark_results.csv")

    print()


if __name__ == "__main__":
    main()
