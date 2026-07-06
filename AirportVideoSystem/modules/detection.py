"""
Detection pipeline: false-positive filtering and per-frame processing.

filter_detection() accepts an explicit FilterConfig dict so it has no
dependency on Streamlit sidebar globals — making it fully testable.
"""

import cv2
import numpy as np
import torch

from .utils import put_cyrillic_text, is_box_near_box

_USE_HALF = torch.cuda.is_available()


# ──────────────────────────────────────────────────────────────────────────────
# Event state machine — temporal debouncing (hysteresis)
# ──────────────────────────────────────────────────────────────────────────────

class ChockServiceState:
    """
    Confirms the "chock placed at wheel" service status with hysteresis.

    A raw per-frame condition is noisy (a single mis-detection can flip it), so
    we only switch the *confirmed* state after the condition has held steady for
    a number of consecutive processed frames:

        * OFF → ON  requires `on_frames` consecutive frames with condition True
        * ON  → OFF requires `off_frames` consecutive frames with condition False

    This removes flicker and — unlike the old logic — allows the status to fall
    back to "waiting" when the chock is removed, so a removal event can fire.

    Crucially, removal is only inferred when the scene is *observable* (the wheel
    is in view but no chock is next to it). If the wheel itself leaves the frame
    or is occluded, the situation is undecidable — disappearance is NOT removal —
    so the confirmed state is frozen until the wheel becomes visible again.
    """

    def __init__(self, on_frames: int = 5, off_frames: int = 15):
        self.on_frames  = max(1, int(on_frames))
        self.off_frames = max(1, int(off_frames))
        self.active     = False
        self._on_count  = 0
        self._off_count = 0

    def update(self, condition_met: bool, observable: bool = True) -> bool:
        """
        Feed the raw per-frame condition; returns the confirmed active state.

        Args:
            condition_met: True if a chock is currently next to a wheel.
            observable:    True if the scene allows a judgement (i.e. at least
                           one wheel is in view). When False, the state is held
                           and transient counters are reset, so a wheel leaving
                           the frame cannot be mistaken for a chock removal.
        """
        if not observable:
            self._on_count  = 0
            self._off_count = 0
            return self.active

        if condition_met:
            self._on_count += 1
            self._off_count = 0
            if not self.active and self._on_count >= self.on_frames:
                self.active = True
        else:
            self._off_count += 1
            self._on_count = 0
            if self.active and self._off_count >= self.off_frames:
                self.active = False
        return self.active


# ──────────────────────────────────────────────────────────────────────────────
# ONNX / format helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_onnx_like(model) -> bool:
    """True if the model backend is ONNX / OpenVINO / TFLite (non-PyTorch)."""
    try:
        pt = str(getattr(model, 'model', '') or
                 getattr(model, 'ckpt_path', '') or '')
        if any(pt.lower().endswith(e) for e in ('.onnx', '.xml', '.tflite', '.pb')):
            return True
        # Ultralytics stores format in overrides after export
        fmt = (getattr(model, 'overrides', {}) or {}).get('format', '')
        return fmt in ('onnx', 'openvino', 'tflite', 'tensorflow')
    except Exception:
        return False


def _native_imgsz(model) -> int | None:
    """
    Read fixed input H from an ONNX or OpenVINO IR model.
    Returns int if static shape, None if dynamic or unknown.
    """
    import os
    pt = str(getattr(model, 'model', '') or '')

    # ── ONNX: read shape via onnxruntime ─────────────────────────────────────
    if pt.lower().endswith('.onnx'):
        try:
            import onnxruntime as ort
            sess  = ort.InferenceSession(pt, providers=['CPUExecutionProvider'])
            shape = sess.get_inputs()[0].shape   # e.g. [1, 3, 640, 640]
            h = shape[2] if len(shape) >= 4 else None
            if isinstance(h, int) and h > 0:
                return h
        except Exception:
            pass
        return None

    # ── OpenVINO IR: try metadata.yaml first, then read shape from model ────
    # model path may be the directory or the .xml file inside it
    xml_path = None
    if pt.lower().endswith('.xml') and os.path.isfile(pt):
        xml_path = pt
    elif os.path.isdir(pt):
        xmls = [f for f in os.listdir(pt) if f.lower().endswith('.xml')]
        if xmls:
            xml_path = os.path.join(pt, xmls[0])

    if xml_path:
        ov_dir = os.path.dirname(xml_path)
        # 1) try metadata.yaml
        meta_path = os.path.join(ov_dir, 'metadata.yaml')
        if os.path.isfile(meta_path):
            try:
                import yaml
                with open(meta_path, 'r') as f:
                    meta = yaml.safe_load(f)
                imgsz = meta.get('imgsz')
                if imgsz:
                    if isinstance(imgsz, (list, tuple)):
                        return int(imgsz[0])
                    return int(imgsz)
            except Exception:
                pass
        # 2) read input shape directly from the OpenVINO model
        try:
            import openvino as ov
            core = ov.Core()
            ov_model = core.read_model(xml_path)
            shape = ov_model.input().shape  # e.g. [1, 3, 640, 640]
            if len(shape) >= 4:
                h = int(shape[2])
                if h > 0:
                    return h
        except Exception:
            pass

    return None


def _safe_imgsz(model, requested: int) -> int:
    """
    Return the correct imgsz to use for this model:
    - For static ONNX: always the model's native size (ignores user's request).
    - For dynamic ONNX / PyTorch: user's requested size.
    Caches result on the model object after first call.
    """
    if not hasattr(model, '_cached_imgsz'):
        model._cached_imgsz = _native_imgsz(model)
    fixed = model._cached_imgsz
    return fixed if fixed else requested


def _safe_half(model) -> bool:
    """FP16 only for PyTorch on CUDA; ONNX/OpenVINO always FP32."""
    return _USE_HALF and not _is_onnx_like(model)


# ──────────────────────────────────────────────────────────────────────────────
# Tiled inference — для обнаружения мелких/дальних объектов
# ──────────────────────────────────────────────────────────────────────────────

def _nms_boxes(boxes, scores, iou_thr=0.5):
    if not boxes:
        return []
    b = np.array(boxes, dtype=np.float32)
    s = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = b[:,0], b[:,1], b[:,2], b[:,3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = s.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thr]
    return keep


def tiled_predict(model, img, conf=0.25, tile_size=640, overlap=0.25, imgsz=640):
    """
    Запускает модель на перекрывающихся тайлах и возвращает результаты
    в координатах исходного изображения.

    Используй для обнаружения мелких/дальних объектов (колёса, колодки вдали).
    """
    H, W  = img.shape[:2]
    step  = int(tile_size * (1 - overlap))

    all_boxes, all_scores, all_classes = [], [], []

    for y in range(0, H, step):
        for x in range(0, W, step):
            x2 = min(x + tile_size, W)
            y2 = min(y + tile_size, H)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            tile = img[y1:y2, x1:x2]

            results = model(tile, imgsz=imgsz, conf=conf, verbose=False, half=False)
            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box, score, cls in zip(r.boxes.xyxy.cpu().numpy(),
                                        r.boxes.conf.cpu().numpy(),
                                        r.boxes.cls.cpu().numpy().astype(int)):
                bx1, by1, bx2, by2 = box
                all_boxes.append([bx1+x1, by1+y1, bx2+x1, by2+y1])
                all_scores.append(float(score))
                all_classes.append(int(cls))

    if not all_boxes:
        return None  # совместимо с обычным возвратом

    keep = _nms_boxes(all_boxes, all_scores, iou_thr=0.5)
    return {
        'boxes':   [all_boxes[i]  for i in keep],
        'scores':  [all_scores[i] for i in keep],
        'classes': [all_classes[i] for i in keep],
    }

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
    chock_state=None,
    chock_wheel_margin: int = 80,
    device=None,
    use_half: bool = False,
    event_engine=None,
    door_model=None,
    conf_door: float = 0.35,
    door_classes=None,
    frame_idx: int = 0,
    time_sec: float = 0.0,
) -> tuple:
    """
    Run detection, tracking, pose estimation and draw annotations on a single frame.

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
        tracker:          не используется (оставлен для совместимости).
        show_ids:         overlay track IDs on detections.
        show_filt_stats:  overlay filtered-detection counter.
        filter_cfg:       dict passed to filter_detection().
        tracking_enabled: whether to use the tracker.
        chock_state:      optional ChockServiceState for temporal debouncing of
                          the "chock placed at wheel" status. If None, the status
                          is derived from the raw per-frame condition (no debounce).
        chock_wheel_margin: max gap in pixels between a chock and a wheel box for
                          the chock to count as "placed at the wheel".
        device:           inference device ('cpu', 'cuda', '0', or None=auto).
        use_half:         enable FP16 (opt-in; CUDA only).
        event_engine:     optional EventEngine. When given, it evaluates all
                          declared rules and owns event logging + status lines;
                          chock_state/chock_wheel_margin are then unused.
        door_model:       optional YOLO model detecting door/hatch classes,
                          feeding the 'door' detection source for presence rules.
        conf_door:        confidence threshold for the door model.
        door_classes:     optional list of class ids to keep from the door model
                          (e.g. [door, hatch]); other classes (people, …) are
                          neither detected nor drawn. None = keep all.
        frame_idx/time_sec: frame index and elapsed seconds, attached to events.

    Returns:
        (annotated_frame, svc_status, svc_color)
    """
    h, w = frame.shape[:2]
    annotated = frame
    global_chocks = []
    global_wheels = []
    filtered_count = 0

    # Detections grouped per source/class for the declarative event engine:
    #   dets = {'combo': {cls: [[x1,y1,x2,y2], ...]}, 'door': {...}, 'person': {...}}
    dets = {'combo': {}, 'door': {}, 'person': {}}

    w_cls = filter_cfg["wheel_class_id"]
    c_cls = filter_cfg["chock_class_id"]

    # FP16 is opt-in (a common cause of "CUDA error: unknown error" on consumer
    # GPUs) and only valid on a CUDA device — always FP32 on CPU.
    def _half_for(m):
        return use_half and _safe_half(m) and device != 'cpu'

    # ── Stage 1: Wheels + Chocks ──────────────────────────────────────
    combo_detections = []
    if use_w and combo_model is not None:
        _imgsz = _safe_imgsz(combo_model, imgsz_c)
        _half  = _half_for(combo_model)
        if tracking_enabled:
            combo_res = combo_model.track(
                frame, conf=conf_w, imgsz=_imgsz,
                verbose=False, half=_half, device=device,
                tracker='bytetrack.yaml', persist=True,
            )
        else:
            combo_res = combo_model(
                frame, conf=conf_w, imgsz=_imgsz,
                verbose=False, half=_half, device=device,
            )
        boxes_res = combo_res[0].boxes
        if boxes_res is not None and len(boxes_res):
            ids = boxes_res.id  # Tensor (N,) or None
            for i, box in enumerate(boxes_res):
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cls_id   = int(box.cls[0].cpu().numpy())
                conf_val = float(box.conf[0].cpu().numpy())
                if not filter_detection(x1, y1, x2, y2, cls_id, h, w, filter_cfg):
                    filtered_count += 1
                    continue
                tid = int(ids[i].cpu().numpy()) if ids is not None else -1
                combo_detections.append((x1, y1, x2, y2, cls_id, conf_val, tid))

    tracked = combo_detections

    # ── Draw wheel / chock detections ─────────────────────────────────
    for item in tracked:
        x1, y1, x2, y2, cls_id, conf_val, track_id = item

        dets['combo'].setdefault(cls_id, []).append([x1, y1, x2, y2])

        if cls_id == w_cls:
            global_wheels.append([x1, y1, x2, y2])
            color = (255, 0, 0)
            label = f"Wheel {conf_val:.2f}"
            if show_ids and track_id > 0:
                label += f" [#{track_id}]"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, lw)
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, color, 2)

        elif cls_id == c_cls:
            global_chocks.append([x1, y1, x2, y2])
            color = (0, 165, 255)
            label = f"Chock {conf_val:.2f}"
            if show_ids and track_id > 0:
                label += f" [#{track_id}]"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, lw + 1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, color, 2)

    # ── Overlays ──────────────────────────────────────────────────────
    if show_filt_stats and filtered_count > 0:
        cv2.putText(annotated, f"Filtered: {filtered_count}",
                    (w - 250, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

    # ── Stage 1b: Door / hatch detection (separate model) ─────────────
    # Restrict to the relevant classes (door/hatch) so the door model's other
    # outputs (e.g. people) are not detected or drawn over the person model.
    if door_model is not None:
        door_res = door_model(frame, conf=conf_door,
                              imgsz=_safe_imgsz(door_model, 640),
                              classes=door_classes,
                              verbose=False, half=_half_for(door_model), device=device)
        d_boxes = door_res[0].boxes
        d_names = getattr(door_model, 'names', {}) or {}
        if d_boxes is not None and len(d_boxes):
            for box in d_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cls_id   = int(box.cls[0].cpu().numpy())
                conf_val = float(box.conf[0].cpu().numpy())
                dets['door'].setdefault(cls_id, []).append([x1, y1, x2, y2])
                color = (255, 0, 255)
                label = f"{d_names.get(cls_id, cls_id)} {conf_val:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, lw)
                cv2.putText(annotated, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, color, 2)

    # ── Stage 2: Person detection ─────────────────────────────────────
    person_boxes = []
    if use_p and det_model is not None:
        det_res = det_model(frame, classes=[0], conf=conf_p,
                            imgsz=_safe_imgsz(det_model, 640),
                            verbose=False, half=_half_for(det_model), device=device)
        person_boxes = det_res[0].boxes.xyxy.cpu().numpy()
        dets['person'][0] = [[int(a), int(b), int(c), int(d)] for a, b, c, d in person_boxes]

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
            # Process crops one-at-a-time: ONNX static-shape models don't
            # support variable batch sizes or variable spatial dimensions.
            pose_imgsz = _safe_imgsz(pose_model, 640)
            pose_half  = _half_for(pose_model)
            pose_results = [
                pose_model(crop, imgsz=pose_imgsz,
                           verbose=False, half=pose_half, device=device)[0]
                for crop in crops
            ]
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

    # ── Events ─────────────────────────────────────────────────────────
    # Declarative path: the engine evaluates all configured rules (chock↔wheel,
    # door→boarding, …) and records transitions internally. Status lines for the
    # bar come from the engine. The legacy chock_state path is kept for callers
    # that don't pass an engine.
    status_lines = None
    if event_engine is not None:
        event_engine.update(dets, frame_idx=frame_idx, time_sec=time_sec)
        status_lines = event_engine.status_lines()
        if status_lines:
            svc_status, svc_color = status_lines[0]
    else:
        # Direct spatial check (chock box near a wheel box), then debounced
        # through the state machine so a single noisy frame can't flip status.
        chock_at_wheel = any(
            is_box_near_box(c_box, w_box, margin=chock_wheel_margin)
            for c_box in global_chocks
            for w_box in global_wheels
        )
        # Removal can only be judged while the wheel (the anchor) is visible.
        # If the wheel left the frame / is occluded, we must not infer removal.
        wheels_visible = len(global_wheels) > 0

        if chock_state is not None:
            confirmed = chock_state.update(chock_at_wheel, observable=wheels_visible)
        else:
            confirmed = chock_at_wheel

        if confirmed:
            svc_status = "ГОТОВО: Колодки установлены!"
            svc_color  = (0, 255, 0)
        else:
            svc_status = "ОЖИДАНИЕ: Установите колодки"
            svc_color  = (0, 0, 255)

    # Person bounding boxes
    if use_p:
        for pbox in person_boxes:
            bx1, by1, bx2, by2 = map(int, pbox)
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (50, 205, 50), lw)
            cv2.putText(annotated, "Person", (bx1, by1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (50, 205, 50), 2)

    # ── Status bar (Cyrillic) ─────────────────────────────────────────
    if show_bar:
        # Multiple event statuses (engine) stack vertically; otherwise one line.
        lines = status_lines if status_lines else [(svc_status, svc_color)]
        bar_h = 16 + 38 * len(lines)
        cv2.rectangle(annotated, (0, 0), (w, bar_h), (0, 0, 0), -1)
        for i, (text, color) in enumerate(lines):
            annotated = put_cyrillic_text(annotated, text, (20, 8 + i * 38),
                                          color, font_size=30)

    return annotated, svc_status, svc_color
