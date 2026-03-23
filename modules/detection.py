"""
Detection pipeline: false-positive filtering and per-frame processing.

filter_detection() accepts an explicit FilterConfig dict so it has no
dependency on Streamlit sidebar globals — making it fully testable.
"""

import cv2
import numpy as np
import torch

from .utils import put_cyrillic_text, is_point_near_box

_USE_HALF = torch.cuda.is_available()

# ──────────────────────────────────────────────────────────────────────────────
# Skeleton constants (MPII 16 keypoints)
# ──────────────────────────────────────────────────────────────────────────────

SKELETON_MPII = [
    (0, 1), (1, 2), (2, 6),
    (5, 4), (4, 3), (3, 6),
    (6, 7), (7, 8), (8, 9),
    (7, 12), (12, 11), (11, 10),
    (7, 13), (13, 14), (14, 15),
]
MPII_N = 16
KPT_COLORS = [
    (0, 0, 255),   (0, 85, 255),  (0, 170, 255), (0, 255, 170),
    (0, 255, 85),  (0, 255, 0),   (255, 255, 0), (255, 170, 0),
    (255, 85, 0),  (255, 0, 0),   (255, 0, 170), (255, 0, 255),
    (170, 0, 255), (85, 0, 255),  (0, 0, 255),   (0, 85, 255),
]
SKEL_COLOR = (0, 255, 0)


# ──────────────────────────────────────────────────────────────────────────────
# False-positive filter
# ──────────────────────────────────────────────────────────────────────────────

def filter_detection(
    x1: int, y1: int, x2: int, y2: int,
    cls_id: int, frame_h: int, frame_w: int,
    cfg: dict,
) -> bool:
    """
    Returns True if the detection should be KEPT, False if it should be dropped.

    Args:
        x1, y1, x2, y2: bounding-box coordinates.
        cls_id:          detected class id.
        frame_h/w:       frame dimensions (pixels).
        cfg: dict with keys:
            filter_enabled, wheel_class_id, chock_class_id,
            wheel_min_area, wheel_max_aspect, wheel_min_aspect,
            use_zone_filter, zone_pct,
            chock_min_area, chock_max_area, chock_max_aspect
    """
    if not cfg.get("filter_enabled", True):
        return True

    bw = x2 - x1
    bh = y2 - y1
    area = bw * bh
    aspect = max(bw, bh) / (min(bw, bh) + 1e-6)

    if cls_id == cfg["wheel_class_id"]:
        if area < cfg["wheel_min_area"]:
            return False
        if aspect > cfg["wheel_max_aspect"] or aspect < 1.0 / cfg["wheel_min_aspect"]:
            return False
        if cfg.get("use_zone_filter", False):
            cy = (y1 + y2) / 2
            if cy < frame_h * (1 - cfg["zone_pct"] / 100.0):
                return False

    elif cls_id == cfg["chock_class_id"]:
        if area < cfg["chock_min_area"]:
            return False
        if area > cfg["chock_max_area"]:
            return False
        if aspect > cfg["chock_max_aspect"]:
            return False

    return True


# ──────────────────────────────────────────────────────────────────────────────
# Per-frame processing
# ──────────────────────────────────────────────────────────────────────────────

def process_frame(
    frame: np.ndarray,
    combo_model,
    det_model,
    pose_model,
    use_w: bool,
    use_p: bool,
    use_po: bool,
    conf_w: float,
    conf_p: float,
    conf_kpt: float,
    imgsz_c: int,
    lw: int,
    fs: float,
    svc_status: str,
    svc_color: tuple,
    show_bar: bool,
    tracker,
    show_ids: bool,
    show_filt_stats: bool,
    filter_cfg: dict,
    tracking_enabled: bool,
    slam=None,
    show_slam_stats: bool = False,
) -> tuple:
    """
    Run detection, tracking, SLAM verification, pose estimation and draw
    annotations on a single frame.

    Args:
        frame:            BGR image.
        combo_model:      YOLO model for wheels + chocks (or None).
        det_model:        YOLO model for person detection (or None).
        pose_model:       YOLO pose model (or None).
        use_w/p/po:       flags to enable each pipeline stage.
        conf_w/p/kpt:     confidence thresholds.
        imgsz_c:          inference image size for combo model.
        lw:               line thickness for drawing.
        fs:               font scale for cv2.putText.
        svc_status/color: current chock-placement status (mutated if chocks placed).
        show_bar:         draw the Cyrillic status bar at the top.
        tracker:          SimpleTracker instance (or None).
        show_ids:         overlay track IDs on detections.
        show_filt_stats:  overlay filtered-detection counter.
        filter_cfg:       dict passed to filter_detection().
        tracking_enabled: whether to use the tracker.
        slam:             SemanticSLAM instance (or None).
        show_slam_stats:  overlay SLAM verified/total counter.

    Returns:
        (annotated_frame, svc_status, svc_color)
    """
    h, w = frame.shape[:2]
    annotated = frame
    global_chocks = []
    filtered_count = 0

    w_cls = filter_cfg["wheel_class_id"]
    c_cls = filter_cfg["chock_class_id"]

    # ── Stage 1: Wheels + Chocks ──────────────────────────────────────
    combo_detections = []
    if use_w and combo_model is not None:
        combo_res = combo_model(frame, conf=conf_w, imgsz=imgsz_c,
                                verbose=False, half=_USE_HALF)
        for box in combo_res[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            conf_val = float(box.conf[0].cpu().numpy())
            if not filter_detection(x1, y1, x2, y2, cls_id, h, w, filter_cfg):
                filtered_count += 1
                continue
            combo_detections.append((x1, y1, x2, y2, cls_id, conf_val))

    # ── Tracking ──────────────────────────────────────────────────────
    if tracking_enabled and tracker is not None:
        tracked = tracker.update(combo_detections)
    else:
        tracked = [(*d, -1) for d in combo_detections]

    # ── SLAM verification ─────────────────────────────────────────────
    verified_ids: set = set()
    if slam is not None:
        verified_ids = slam.update(frame, list(tracked))

    # ── Draw wheel / chock detections ─────────────────────────────────
    for item in tracked:
        x1, y1, x2, y2, cls_id, conf_val, track_id = item
        is_verified = (track_id in verified_ids) if slam is not None else True

        if cls_id == w_cls:
            color = (255, 0, 0)
            label = f"Wheel {conf_val:.2f}"
            if show_ids and track_id > 0:
                label += f" [#{track_id}]"
            if not is_verified:
                label += " ?"
                cv2.rectangle(annotated, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (160, 160, 160), 1)
            else:
                cv2.rectangle(annotated, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 255, 100), 1)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, lw)
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, color, 2)

        elif cls_id == c_cls:
            global_chocks.append([x1, y1, x2, y2])
            color = (0, 165, 255)
            label = f"Chock {conf_val:.2f}"
            if show_ids and track_id > 0:
                label += f" [#{track_id}]"
            if not is_verified:
                label += " ?"
                cv2.rectangle(annotated, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (160, 160, 160), 1)
            else:
                cv2.rectangle(annotated, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 255, 100), 1)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, lw + 1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, color, 2)

    # ── Overlays ──────────────────────────────────────────────────────
    if show_slam_stats and slam is not None:
        total_obj, verified_obj = slam.get_stats()
        cv2.putText(annotated, f"SLAM: {verified_obj}/{total_obj} verified",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

    if show_filt_stats and filtered_count > 0:
        cv2.putText(annotated, f"Filtered: {filtered_count}",
                    (w - 250, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

    # ── Stage 2: Person detection ─────────────────────────────────────
    person_boxes = []
    if use_p and det_model is not None:
        det_res = det_model(frame, classes=[0], conf=conf_p,
                            verbose=False, half=_USE_HALF)
        person_boxes = det_res[0].boxes.xyxy.cpu().numpy()

    # ── Stage 3: Pose estimation (crop-based, batched) ────────────────
    if use_po and pose_model is not None and len(person_boxes) > 0:
        crops = []
        crop_coords = []
        for pbox in person_boxes:
            bx1, by1, bx2, by2 = map(int, pbox)
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w, bx2), min(h, by2)
            crop = frame[by1:by2, bx1:bx2]
            if crop.size == 0:
                continue
            crops.append(crop)
            crop_coords.append((bx1, by1, bx2, by2))

        if crops:
            pose_results = pose_model(crops, verbose=False, half=_USE_HALF)
            for pose_res, (bx1, by1, bx2, by2) in zip(pose_results, crop_coords):
                keypoints = pose_res.keypoints
                if keypoints is None or len(keypoints.data) == 0:
                    continue

                kpts = keypoints.data[0].cpu().numpy()
                num_kpts = kpts.shape[0]
                conf_kpt_val = conf_kpt

                # Skeleton lines
                for p1_idx, p2_idx in SKELETON_MPII:
                    if p1_idx < num_kpts and p2_idx < num_kpts:
                        kx1, ky1, c1 = kpts[p1_idx]
                        kx2, ky2, c2 = kpts[p2_idx]
                        if c1 > conf_kpt_val and c2 > conf_kpt_val:
                            pt1 = (int(kx1) + bx1, int(ky1) + by1)
                            pt2 = (int(kx2) + bx1, int(ky2) + by1)
                            cv2.line(annotated, pt1, pt2, SKEL_COLOR, lw + 1, cv2.LINE_AA)

                # Keypoints
                for idx in range(min(num_kpts, MPII_N)):
                    kx, ky, kc = kpts[idx]
                    if kc > conf_kpt_val:
                        cx_k, cy_k = int(kx) + bx1, int(ky) + by1
                        cv2.circle(annotated, (cx_k, cy_k), 5, KPT_COLORS[idx], -1, cv2.LINE_AA)
                        cv2.circle(annotated, (cx_k, cy_k), 6, (255, 255, 255), 1, cv2.LINE_AA)

                # Chock-placement logic
                knees_y = []
                if num_kpts > 4:
                    if kpts[1][2] > conf_kpt_val:
                        knees_y.append(kpts[1][1] + by1)
                    if kpts[4][2] > conf_kpt_val:
                        knees_y.append(kpts[4][1] + by1)

                wrists = []
                if num_kpts > 15:
                    if kpts[10][2] > 0.4:
                        wrists.append((kpts[10][0] + bx1, kpts[10][1] + by1))
                    if kpts[15][2] > 0.4:
                        wrists.append((kpts[15][0] + bx1, kpts[15][1] + by1))

                is_bending = any(
                    wy > min(knees_y)
                    for wx, wy in wrists
                ) if knees_y and wrists else False

                is_touching_chock = False
                if is_bending:
                    for wx, wy in wrists:
                        for c_box in global_chocks:
                            if is_point_near_box(wx, wy, c_box, margin=50):
                                is_touching_chock = True
                                break

                if is_bending and is_touching_chock:
                    svc_status = "ГОТОВО: Колодки установлены!"
                    svc_color = (0, 255, 0)

    # Person bounding boxes
    if use_p:
        for pbox in person_boxes:
            bx1, by1, bx2, by2 = map(int, pbox)
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (50, 205, 50), lw)
            cv2.putText(annotated, "Person", (bx1, by1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (50, 205, 50), 2)

    # ── Status bar (Cyrillic) ─────────────────────────────────────────
    if show_bar:
        cv2.rectangle(annotated, (0, 0), (w, 70), (0, 0, 0), -1)
        annotated = put_cyrillic_text(annotated, svc_status, (20, 12),
                                      svc_color, font_size=32)

    return annotated, svc_status, svc_color
