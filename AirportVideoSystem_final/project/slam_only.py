"""
SLAM-only offline processor — no YOLO detection models.

Runs pure visual SLAM (SIFT features + Essential Matrix) on every frame.
Outputs:
  slam_map.html          — interactive 3-D point cloud + camera trajectory
  slam_trajectory.csv    — per-frame camera position (x, y, z)
  slam_video.mp4         — video with trajectory overlay (optional)

Usage:
    python slam_only.py --input video.mp4
    python slam_only.py --input video.mp4 --every-n 2 --no-video
    python slam_only.py --input video.mp4 --output-dir results/
"""

import argparse
import csv
import io
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from modules.slam import SemanticSLAM
from modules.visualization import build_slam_plotly


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SLAM-only offline processor (no YOLO)")
    p.add_argument("--input",      required=True)
    p.add_argument("--output-dir", default="slam_output")
    p.add_argument("--every-n",    type=int,   default=1,
                   help="Process every N-th frame (2-3 recommended for speed)")
    p.add_argument("--resize",     type=float, default=1.0,
                   help="Resize factor before SIFT (0.5 = half size, much faster)")
    p.add_argument("--slam-min-obs",  type=int,   default=3)
    p.add_argument("--slam-min-feat", type=int,   default=2)
    p.add_argument("--no-video",   action="store_true",
                   help="Skip video output (faster — map only)")
    p.add_argument("--no-masking", action="store_true",
                   help="Disable YOLOv8-seg dynamic masking (no YOLO at all)")
    p.add_argument("--slam-yolo-seg",  default="yolov8n-seg.pt")
    p.add_argument("--slam-yolo-conf", type=float, default=0.25)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory overlay
# ─────────────────────────────────────────────────────────────────────────────

def draw_trajectory_overlay(frame: np.ndarray, traj: list,
                             map_size: int = 200, margin: int = 10) -> np.ndarray:
    """Draw 2-D top-down trajectory in top-right corner."""
    if len(traj) < 2:
        return frame

    pts = np.array(
        [(float(p[0]), float(p[2])) for p in traj
         if isinstance(p, np.ndarray) and p.shape[0] >= 3],
        dtype=np.float32,
    )
    if len(pts) < 2:
        return frame

    mn, mx = pts.min(0), pts.max(0)
    rng = mx - mn
    rng[rng < 1e-6] = 1.0

    def to_px(p):
        nx = int((p[0] - mn[0]) / rng[0] * (map_size - 20) + 10)
        ny = int((1.0 - (p[1] - mn[1]) / rng[1]) * (map_size - 20) + 10)
        return nx, ny

    h, w = frame.shape[:2]
    x0 = w - map_size - margin
    y0 = margin

    # semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + map_size, y0 + map_size),
                  (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # path
    for i in range(1, len(pts)):
        p1 = (x0 + to_px(pts[i - 1])[0], y0 + to_px(pts[i - 1])[1])
        p2 = (x0 + to_px(pts[i])[0],     y0 + to_px(pts[i])[1])
        alpha = 0.3 + 0.7 * (i / len(pts))
        color = (int(255 * alpha), int(255 * alpha), int(255 * alpha))
        cv2.line(frame, p1, p2, color, 1)

    # current position
    cur = (x0 + to_px(pts[-1])[0], y0 + to_px(pts[-1])[1])
    cv2.circle(frame, cur, 4, (0, 255, 136), -1)

    cv2.putText(frame, f"SLAM traj ({len(pts)} pts)",
                (x0 + 4, y0 + map_size - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 200, 255), 1)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Progress
# ─────────────────────────────────────────────────────────────────────────────

def progress_bar(cur, total, t0, width=40):
    pct    = cur / max(total, 1)
    filled = int(width * pct)
    bar    = "█" * filled + "░" * (width - filled)
    elapsed = time.time() - t0
    eta     = (elapsed / max(cur, 1)) * (total - cur)
    fps_now = cur / max(elapsed, 1e-6)
    return (f"\r[{bar}] {cur}/{total}  {pct*100:.1f}%  "
            f"{fps_now:.1f} fps  ETA {eta:.0f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.input):
        print(f"[ERROR] File not found: {args.input}")
        sys.exit(1)

    cap          = cv2.VideoCapture(args.input)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    out_w = int(vid_w * args.resize)
    out_h = int(vid_h * args.resize)

    use_masking = not args.no_masking

    print("=" * 60)
    print("  SLAM-Only Offline Processor")
    print("=" * 60)
    print(f"  Input      : {args.input}")
    print(f"  Output dir : {args.output_dir}/")
    print(f"  Resolution : {vid_w}x{vid_h}"
          + (f"  →  {out_w}x{out_h}" if args.resize != 1.0 else ""))
    print(f"  Every-N    : {args.every_n}")
    print(f"  Masking    : {'ON (yolov8n-seg.pt)' if use_masking else 'OFF (pure SIFT)'}")
    print(f"  Video out  : {'NO' if args.no_video else 'YES'}")
    print("=" * 60)

    # ── SLAM init ─────────────────────────────────────────────────────────────
    slam = SemanticSLAM(
        min_observations=args.slam_min_obs,
        min_features=args.slam_min_feat,
        enable_visual_masking=use_masking,
        yolo_model=args.slam_yolo_seg,
        yolo_conf=args.slam_yolo_conf,
    )

    # ── Video writer ──────────────────────────────────────────────────────────
    writer = None
    raw_out = os.path.join(args.output_dir, "slam_raw.mp4")
    if not args.no_video:
        out_fps = max(fps / args.every_n, 1.0)
        writer  = cv2.VideoWriter(
            raw_out, cv2.VideoWriter_fourcc(*"mp4v"),
            out_fps, (vid_w, vid_h),
        )

    # ── Processing loop ───────────────────────────────────────────────────────
    print("\nProcessing...")
    cap           = cv2.VideoCapture(args.input)
    frame_idx     = 0
    processed_cnt = 0
    t0            = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.every_n == 0:
            # Resize for SIFT if requested
            proc_frame = (cv2.resize(frame, (out_w, out_h))
                          if args.resize != 1.0 else frame)

            # SLAM update (no detections — empty list)
            slam.update(proc_frame, [])
            processed_cnt += 1

            if writer is not None:
                # Draw trajectory overlay on original-size frame
                annotated = draw_trajectory_overlay(frame.copy(), slam.cam_traj)
                # SLAM stats text
                n_pts = len(slam.point_cloud)
                n_pos = len(slam.cam_traj)
                cv2.putText(
                    annotated,
                    f"SLAM | pts:{n_pts} | poses:{n_pos}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 136), 2,
                )
                writer.write(annotated)

        if frame_idx % 30 == 0:
            print(progress_bar(frame_idx + 1, total_frames, t0),
                  end="", flush=True)
        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    print(progress_bar(total_frames, total_frames, t0))

    elapsed = time.time() - t0
    print(f"\n  Processed {processed_cnt} frames in {elapsed:.1f}s  "
          f"({processed_cnt / elapsed:.2f} fps)\n")

    # ── Re-encode ─────────────────────────────────────────────────────────────
    if writer is not None:
        final_video = os.path.join(args.output_dir, "slam_video.mp4")
        print(f"  Re-encoding -> {final_video}")
        ret_code = os.system(
            f'ffmpeg -y -i "{raw_out}" -vcodec libx264 -preset fast '
            f'"{final_video}" -loglevel error'
        )
        if ret_code == 0 and os.path.getsize(final_video) > 0:
            os.unlink(raw_out)
        else:
            os.rename(raw_out, final_video)

    # ── SLAM outputs ──────────────────────────────────────────────────────────
    print("  Saving SLAM map...")
    total_obj, verified_obj = slam.get_stats()
    print(f"  Objects    : {total_obj} total, {verified_obj} verified")
    print(f"  Point cloud: {len(slam.point_cloud)} pts")
    print(f"  Trajectory : {len(slam.cam_traj)} poses")

    # HTML map
    if len(slam.cam_traj) > 1 or len(slam.point_cloud) > 0:
        fig      = build_slam_plotly(slam, wheel_cls=-1, chock_cls=-2)  # no semantic labels
        html_out = os.path.join(args.output_dir, "slam_map.html")
        buf      = io.StringIO()
        fig.write_html(buf, include_plotlyjs="cdn")
        with open(html_out, "w", encoding="utf-8") as f:
            f.write(buf.getvalue())
        print(f"  3-D map    : {html_out}")
    else:
        print("  [WARN] Not enough data for 3-D map")

    # Trajectory CSV
    traj_out = os.path.join(args.output_dir, "slam_trajectory.csv")
    with open(traj_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_processed", "x", "y", "z"])
        for i, pt in enumerate(slam.cam_traj):
            if isinstance(pt, np.ndarray) and pt.shape[0] >= 3:
                w.writerow([i, f"{pt[0]:.6f}", f"{pt[1]:.6f}", f"{pt[2]:.6f}"])
    print(f"  Trajectory : {traj_out}")

    slam.stop()

    print("\n" + "=" * 60)
    print("  Output files:")
    for fname in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, fname)
        size  = os.path.getsize(fpath) / (1024 * 1024)
        print(f"    {fname:<30} {size:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
