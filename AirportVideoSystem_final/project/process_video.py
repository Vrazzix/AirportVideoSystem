"""
Standalone video processing script — no Streamlit required.

Usage:
    python process_video.py --input video.mp4 [options]

Outputs (in --output-dir):
    processed_video.mp4   — annotated video
    slam_map.html         — interactive 3-D SLAM map (Plotly)
    slam_trajectory.csv   — camera trajectory (x, y, z per frame)
"""

import argparse
import csv
import io
import os
import sys
import time

import cv2
import torch

# ── Make sure project root is on path ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from modules import SimpleTracker, SemanticSLAM, build_slam_plotly, process_frame
from ultralytics import YOLO

_USE_HALF = torch.cuda.is_available()


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Aircraft & Human Detection — offline video processor"
    )

    # I/O
    p.add_argument("--input",      required=True, help="Path to input video file")
    p.add_argument("--output-dir", default="output", help="Directory for output files")

    # Models
    p.add_argument("--model-wheels", default="C:\\Users\\shche\\Desktop\\Application_for_models\\models\\BestBoots_v2.pt")
    p.add_argument("--model-person", default="C:\\Users\\shche\\Desktop\\Application_for_models\\models\\person.pt")
    p.add_argument("--model-pose",   default="C:\\Users\\shche\\Desktop\\Application_for_models\\models\\BestPose.pt")

    # Class IDs
    p.add_argument("--wheel-class", type=int, default=1)
    p.add_argument("--chock-class", type=int, default=0)

    # Thresholds
    p.add_argument("--conf-wheels",   type=float, default=0.15)
    p.add_argument("--conf-person",   type=float, default=0.50)
    p.add_argument("--conf-pose-kpt", type=float, default=0.50)
    p.add_argument("--imgsz",         type=int,   default=1280)
    p.add_argument("--every-n",       type=int,   default=1,
                   help="Process every N-th frame (skip others)")

    # Active modules
    p.add_argument("--no-wheels", action="store_true", help="Disable wheel/chock detection")
    p.add_argument("--no-person", action="store_true", help="Disable person detection")
    p.add_argument("--no-pose",   action="store_true", help="Disable pose estimation")
    p.add_argument("--no-slam",   action="store_true", help="Disable Semantic SLAM")
    p.add_argument("--no-track",  action="store_true", help="Disable object tracking")

    # FP filter
    p.add_argument("--no-filter",      action="store_true", help="Disable FP filter")
    p.add_argument("--wheel-min-area", type=int,   default=3000)
    p.add_argument("--wheel-max-asp",  type=float, default=2.5)
    p.add_argument("--wheel-min-asp",  type=float, default=0.4)
    p.add_argument("--chock-min-area", type=int,   default=1000)
    p.add_argument("--chock-max-area", type=int,   default=40000)
    p.add_argument("--chock-max-asp",  type=float, default=4.0)

    # Tracking
    p.add_argument("--iou-thresh", type=float, default=0.3)

    # SLAM
    p.add_argument("--slam-min-obs",      type=int,   default=3)
    p.add_argument("--slam-min-feat",     type=int,   default=2)
    p.add_argument("--slam-masking",      action="store_true", default=True,
                   help="Enable YOLOv8-seg dynamic masking before SIFT")
    p.add_argument("--no-slam-masking",   action="store_true", help="Disable dynamic masking")
    p.add_argument("--slam-yolo-seg",     default="yolov8n-seg.pt")
    p.add_argument("--slam-yolo-conf",    type=float, default=0.25)

    # Visualisation
    p.add_argument("--line-thickness", type=int,   default=2)
    p.add_argument("--font-scale",     type=float, default=0.6)
    p.add_argument("--no-status-bar",  action="store_true")
    p.add_argument("--no-track-ids",   action="store_true")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model_safe(path: str):
    if not os.path.exists(path):
        print(f"  [WARN] Model not found: {path}")
        return None
    m = YOLO(path)
    print(f"  [OK]   Loaded {path}")
    return m


def progress_bar(current: int, total: int, start_time: float, width: int = 40) -> str:
    pct     = current / max(total, 1)
    filled  = int(width * pct)
    bar     = "█" * filled + "░" * (width - filled)
    elapsed = time.time() - start_time
    eta     = (elapsed / max(current, 1)) * (total - current)
    return f"\r[{bar}] {current}/{total}  {pct*100:.1f}%  ETA {eta:.0f}s"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Video info ────────────────────────────────────────────────────────────
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.input)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print("=" * 60)
    print("  Aircraft & Human Detection — Offline Processor")
    print("=" * 60)
    print(f"  Input   : {args.input}")
    print(f"  Output  : {args.output_dir}/")
    print(f"  Video   : {vid_w}x{vid_h}  {fps:.1f} fps  {total_frames} frames")
    print(f"  Device  : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n[1/4] Loading models...")
    loaded = {}
    use_wheels = not args.no_wheels
    use_person = not args.no_person
    use_pose   = not args.no_pose

    if use_wheels:
        m = load_model_safe(args.model_wheels)
        if m: loaded["combo"] = m
    if use_person:
        m = load_model_safe(args.model_person)
        if m: loaded["person"] = m
    if use_pose:
        m = load_model_safe(args.model_pose)
        if m: loaded["pose"] = m

    if not loaded:
        print("[ERROR] No models loaded. Exiting.")
        sys.exit(1)
    print(f"  Active : {', '.join(loaded.keys())}")

    # ── Filter config ─────────────────────────────────────────────────────────
    filter_cfg = {
        "filter_enabled":  not args.no_filter,
        "wheel_class_id":  args.wheel_class,
        "chock_class_id":  args.chock_class,
        "wheel_min_area":  args.wheel_min_area,
        "wheel_max_aspect": args.wheel_max_asp,
        "wheel_min_aspect": args.wheel_min_asp,
        "use_zone_filter": False,
        "zone_pct":        70,
        "chock_min_area":  args.chock_min_area,
        "chock_max_area":  args.chock_max_area,
        "chock_max_aspect": args.chock_max_asp,
    }

    # ── Tracker & SLAM ────────────────────────────────────────────────────────
    print("\n[2/4] Initialising tracker & SLAM...")
    tracking_enabled = not args.no_track
    slam_enabled     = not args.no_slam

    tracker = SimpleTracker(iou_threshold=args.iou_thresh) if tracking_enabled else None

    enable_masking = args.slam_masking and not args.no_slam_masking
    slam = (
        SemanticSLAM(
            min_observations=args.slam_min_obs,
            min_features=args.slam_min_feat,
            enable_visual_masking=enable_masking,
            yolo_model=args.slam_yolo_seg,
            yolo_conf=args.slam_yolo_conf,
        )
        if slam_enabled else None
    )
    print(f"  Tracking : {'ON' if tracking_enabled else 'OFF'}")
    print(f"  SLAM     : {'ON (masking=' + str(enable_masking) + ')' if slam_enabled else 'OFF'}")

    # ── Video writer ──────────────────────────────────────────────────────────
    print("\n[3/4] Processing video...")
    raw_out  = os.path.join(args.output_dir, "processed_raw.mp4")
    out_fps  = fps / args.every_n
    writer   = cv2.VideoWriter(
        raw_out, cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps, (vid_w, vid_h)
    )

    cap           = cv2.VideoCapture(args.input)
    frame_idx     = 0
    processed_cnt = 0
    svc_status    = "ОЖИДАНИЕ: Установите колодки"
    svc_color     = (0, 0, 255)
    start_time    = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.every_n == 0:
            annotated, svc_status, svc_color = process_frame(
                frame,
                loaded.get("combo"), loaded.get("person"), loaded.get("pose"),
                use_wheels, use_person, use_pose,
                args.conf_wheels, args.conf_person, args.conf_pose_kpt,
                args.imgsz, args.line_thickness, args.font_scale,
                svc_status, svc_color, not args.no_status_bar,
                tracker, not args.no_track_ids, True,
                filter_cfg=filter_cfg,
                tracking_enabled=tracking_enabled,
                slam=slam,
                show_slam_stats=True,
            )
            writer.write(annotated)
            processed_cnt += 1

        if frame_idx % 10 == 0:
            print(progress_bar(frame_idx + 1, total_frames, start_time), end="", flush=True)

        frame_idx += 1

    cap.release()
    writer.release()
    print(progress_bar(total_frames, total_frames, start_time))

    total_time = time.time() - start_time
    print(f"\n  Done: {processed_cnt} frames in {total_time:.1f}s "
          f"({processed_cnt/total_time:.1f} fps)")

    # ── Re-encode to H.264 ────────────────────────────────────────────────────
    final_video = os.path.join(args.output_dir, "processed_video.mp4")
    print(f"\n  Re-encoding to H.264 -> {final_video}")
    ret = os.system(
        f'ffmpeg -y -i "{raw_out}" -vcodec libx264 -acodec aac '
        f'"{final_video}" -loglevel error'
    )
    if ret == 0 and os.path.exists(final_video) and os.path.getsize(final_video) > 0:
        os.unlink(raw_out)
    else:
        print("  [WARN] ffmpeg failed — keeping raw mp4v output")
        os.rename(raw_out, final_video)

    # ── SLAM outputs ──────────────────────────────────────────────────────────
    print("\n[4/4] Saving SLAM outputs...")

    if slam is not None:
        total_obj, verified_obj = slam.get_stats()
        print(f"  SLAM objects : {total_obj} total, {verified_obj} verified")
        print(f"  Point cloud  : {len(slam.point_cloud)} points")
        print(f"  Trajectory   : {len(slam.cam_traj)} poses")

        mask_stats = slam.get_mask_stats()
        if mask_stats.get("enabled"):
            print(f"  Masking      : {mask_stats['filter_rate_pct']}% kpts filtered")

        # ── HTML interactive 3-D map ──────────────────────────────────────────
        if len(slam.cam_traj) > 1 or len(slam.point_cloud) > 0:
            fig      = build_slam_plotly(slam, args.wheel_class, args.chock_class)
            html_out = os.path.join(args.output_dir, "slam_map.html")
            html_buf = io.StringIO()
            fig.write_html(html_buf, include_plotlyjs="cdn")
            with open(html_out, "w", encoding="utf-8") as f:
                f.write(html_buf.getvalue())
            print(f"  3-D map      : {html_out}")
        else:
            print("  [WARN] Not enough SLAM data to build map")

        # ── Trajectory CSV ────────────────────────────────────────────────────
        import numpy as np
        traj_out = os.path.join(args.output_dir, "slam_trajectory.csv")
        with open(traj_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "x", "y", "z"])
            for i, pt in enumerate(slam.cam_traj):
                if isinstance(pt, np.ndarray) and pt.shape[0] >= 3:
                    w.writerow([i, f"{pt[0]:.6f}", f"{pt[1]:.6f}", f"{pt[2]:.6f}"])
        print(f"  Trajectory   : {traj_out}")

        slam.stop()
    else:
        print("  SLAM disabled — skipping map outputs")

    print("\n" + "=" * 60)
    print("  Output files:")
    for fname in os.listdir(args.output_dir):
        fpath = os.path.join(args.output_dir, fname)
        size  = os.path.getsize(fpath) / (1024 * 1024)
        print(f"    {fname:<30} {size:.1f} MB")
    print("=" * 60)
    print("  Done!")


if __name__ == "__main__":
    main()
