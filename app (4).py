"""
✈️ Aircraft & Human Detection System v3
- False positive filtering (aspect ratio, min/max size, context zones)
- Object tracking (ByteTrack via ultralytics or simple IoU tracker)
- Cyrillic status bar via PIL
- Crop-based pose (MPII 16 keypoints)
"""

import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import time
import torch
from collections import defaultdict

_USE_HALF = torch.cuda.is_available()

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Aircraft & Human Detection",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');
    .main-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { font-family: 'Inter', sans-serif; font-size: 1rem; color: #6b7280; margin-bottom: 2rem; }
    .model-card { background: linear-gradient(145deg, #f8fafc, #e2e8f0); border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem; border-left: 4px solid; }
    .model-card-wheels { border-left-color: #f59e0b; }
    .model-card-person { border-left-color: #3b82f6; }
    .model-card-pose   { border-left-color: #10b981; }
    .status-ready  { color: #10b981; font-weight: 600; }
    .status-missing { color: #ef4444; font-weight: 600; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a, #1e293b); }
    div[data-testid="stSidebar"] .stMarkdown p, div[data-testid="stSidebar"] .stMarkdown li, div[data-testid="stSidebar"] label { color: #cbd5e1 !important; }
    div[data-testid="stSidebar"] .stMarkdown h1, div[data-testid="stSidebar"] .stMarkdown h2, div[data-testid="stSidebar"] .stMarkdown h3 { color: #f1f5f9 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">✈️ Aircraft & Human Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Детекция колёс/колодок, людей и позы · Фильтрация FP · Трекинг объектов</p>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Модели")
    model_path_wheels = st.text_input("🛞 Комбо-модель (колёса+колодки)", value="models/bestBoots_v2.pt")
    model_path_person = st.text_input("🧑 Модель людей", value="yolov8n.pt")
    model_path_pose = st.text_input("🦴 Модель позы (MPII 16)", value="models/BestPose.pt")

    st.markdown("---")
    st.markdown("## 🏷️ ID классов")
    wheel_class_id = st.number_input("ID «Колесо»", 0, 20, 1)
    chock_class_id = st.number_input("ID «Колодка»", 0, 20, 0)

    st.markdown("---")
    st.markdown("## 🎛️ Пороги")
    conf_wheels = st.slider("Conf — колёса/колодки", 0.05, 1.0, 0.15, 0.05)
    conf_person = st.slider("Conf — люди", 0.1, 1.0, 0.5, 0.05)
    conf_pose_kpt = st.slider("Conf — ключевые точки", 0.1, 1.0, 0.5, 0.05)
    combo_imgsz = st.selectbox("Размер инференса комбо", [640, 960, 1280], index=2)
    process_every_n = st.slider("Каждый N-й кадр", 1, 10, 1)

    st.markdown("---")
    st.markdown("## 🔍 Фильтрация ложных срабатываний")

    filter_enabled = st.checkbox("Включить фильтрацию FP", value=True)

    st.markdown("**Колесо (Wheel):**")
    wheel_min_area = st.slider("Мин. площадь Wheel (px²)", 500, 20000, 3000, 500,
                               help="Отсекает мелкие ложные детекции (буквы, мусор)")
    wheel_max_aspect = st.slider("Макс. aspect ratio Wheel", 1.0, 5.0, 2.5, 0.1,
                                 help="Колесо ~круглое (≈1.0). Отсечёт вытянутые объекты")
    wheel_min_aspect = st.slider("Мин. aspect ratio Wheel", 0.1, 1.0, 0.4, 0.05,
                                 help="Отсечёт слишком узкие вертикальные детекции")

    st.markdown("**Колодка (Chock):**")
    chock_min_area = st.slider("Мин. площадь Chock (px²)", 200, 10000, 1000, 200)
    chock_max_area = st.slider("Макс. площадь Chock (px²)", 5000, 200000, 40000, 5000,
                               help="Отсекает крупные объекты (машины и т.п.), ошибочно детектированные как колодки")
    chock_max_aspect = st.slider("Макс. aspect ratio Chock", 1.0, 8.0, 4.0, 0.5)

    st.markdown("**Зона нижней части кадра:**")
    use_zone_filter = st.checkbox("Колёса только в нижних N% кадра", value=False,
                                  help="Колёса шасси обычно в нижней части кадра")
    zone_pct = st.slider("Нижний % кадра для колёс", 30, 100, 70, 5)

    st.markdown("---")
    st.markdown("## 🔗 Трекинг объектов")
    tracking_enabled = st.checkbox("Включить трекинг (IoU-based)", value=True)
    iou_track_thresh = st.slider("IoU порог для трекинга", 0.1, 0.8, 0.3, 0.05)

    st.markdown("---")
    st.markdown("## 🗺️ Semantic SLAM")
    slam_enabled = st.checkbox("Включить Semantic SLAM", value=True,
                               help="Верифицирует детекции по накопленным наблюдениям и ORB-признакам")
    slam_min_obs = st.slider("Мин. наблюдений для верификации", 2, 15, 3,
                             help="Объект подтверждается только после N кадров стабильного трекинга")
    slam_min_feat = st.slider("Мин. ORB-признаков в bbox", 0, 10, 2,
                              help="Фильтрует детекции на однородном фоне без текстуры")
    show_slam_overlay = st.checkbox("Показывать статистику SLAM", value=True)

    st.markdown("---")
    st.markdown("## 🎨 Активные модели")
    use_wheels = st.checkbox("🛞 Колёса и колодки", value=True)
    use_person = st.checkbox("🧑 Детекция людей", value=True)
    use_pose = st.checkbox("🦴 Поза человека", value=True)

    st.markdown("---")
    st.markdown("## 📊 Визуализация")
    line_thickness = st.slider("Толщина линий", 1, 5, 2)
    font_scale = st.slider("Размер шрифта", 0.3, 1.5, 0.6, step=0.1)
    show_status_bar = st.checkbox("Статус-бар (логика колодок)", value=True)
    show_track_ids = st.checkbox("Показывать ID трекинга", value=True)
    show_filtered_stats = st.checkbox("Показывать счётчик отфильтрованных", value=True)


# ══════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════
@st.cache_resource
def load_model(model_path: str):
    if not os.path.exists(model_path):
        return None
    return YOLO(model_path)


def check_model_status(path):
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        return "ready", f"✅ ({size_mb:.1f} MB)"
    return "missing", f"❌ `{path}`"


# Model cards
st.markdown("### 📦 Статус моделей")
mc1, mc2, mc3 = st.columns(3)
for name, path, css, enabled, col in [
    ("🛞 Колёса+колодки", model_path_wheels, "model-card-wheels", use_wheels, mc1),
    ("🧑 Люди", model_path_person, "model-card-person", use_person, mc2),
    ("🦴 Поза MPII-16", model_path_pose, "model-card-pose", use_pose, mc3),
]:
    s, msg = check_model_status(path)
    with col:
        sc = "status-ready" if s == "ready" else "status-missing"
        st.markdown(f'<div class="model-card {css}"><strong>{name}</strong><br><span class="{sc}">{msg}</span><br><small>{"Вкл" if enabled else "Выкл"}</small></div>', unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════
# SKELETON (MPII 16)
# ══════════════════════════════════════════════
SKELETON_MPII = [
    (0, 1), (1, 2), (2, 6),
    (5, 4), (4, 3), (3, 6),
    (6, 7), (7, 8), (8, 9),
    (7, 12), (12, 11), (11, 10),
    (7, 13), (13, 14), (14, 15),
]
MPII_N = 16
KPT_COLORS = [
    (0,0,255),(0,85,255),(0,170,255),(0,255,170),
    (0,255,85),(0,255,0),(255,255,0),(255,170,0),
    (255,85,0),(255,0,0),(255,0,170),(255,0,255),
    (170,0,255),(85,0,255),(0,0,255),(0,85,255),
]
SKEL_COLOR = (0, 255, 0)


# ══════════════════════════════════════════════
# FALSE POSITIVE FILTER
# ══════════════════════════════════════════════
def filter_detection(x1, y1, x2, y2, cls_id, frame_h, frame_w):
    """Returns True if detection should be KEPT, False if filtered out."""
    if not filter_enabled:
        return True

    bw = x2 - x1
    bh = y2 - y1
    area = bw * bh
    aspect = max(bw, bh) / (min(bw, bh) + 1e-6)

    if cls_id == wheel_class_id:
        if area < wheel_min_area:
            return False
        if aspect > wheel_max_aspect or aspect < 1.0 / wheel_min_aspect:
            return False
        # Zone filter: wheel center must be in bottom N% of frame
        if use_zone_filter:
            cy = (y1 + y2) / 2
            if cy < frame_h * (1 - zone_pct / 100.0):
                return False

    elif cls_id == chock_class_id:
        if area < chock_min_area:
            return False
        if area > chock_max_area:
            return False
        if aspect > chock_max_aspect:
            return False

    return True


# ══════════════════════════════════════════════
# SIMPLE IoU TRACKER
# ══════════════════════════════════════════════
class SimpleTracker:
    """IoU-based tracker: assigns persistent IDs to detections across frames."""

    def __init__(self, iou_threshold=0.3, max_lost=15):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks = {}  # track_id -> {"box": [x1,y1,x2,y2], "cls": int, "lost": int}

    @staticmethod
    def _iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0

    def update(self, detections):
        """
        detections: list of (x1, y1, x2, y2, cls_id, conf)
        Returns: list of (x1, y1, x2, y2, cls_id, conf, track_id)
        """
        if not detections:
            # Increment lost counter
            for tid in list(self.tracks):
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]
            return []

        # Build cost matrix
        track_ids = list(self.tracks.keys())
        matched_det = set()
        matched_trk = set()
        results = []

        # Greedy matching by highest IoU
        pairs = []
        for di, det in enumerate(detections):
            for ti, tid in enumerate(track_ids):
                trk = self.tracks[tid]
                if det[4] == trk["cls"]:  # same class only
                    iou = self._iou(det[:4], trk["box"])
                    if iou >= self.iou_threshold:
                        pairs.append((iou, di, ti, tid))

        pairs.sort(key=lambda x: -x[0])
        for iou_val, di, ti, tid in pairs:
            if di in matched_det or tid in matched_trk:
                continue
            matched_det.add(di)
            matched_trk.add(tid)
            det = detections[di]
            self.tracks[tid]["box"] = list(det[:4])
            self.tracks[tid]["lost"] = 0
            results.append((*det, tid))

        # Unmatched detections -> new tracks
        for di, det in enumerate(detections):
            if di not in matched_det:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"box": list(det[:4]), "cls": det[4], "lost": 0}
                results.append((*det, tid))

        # Unmatched tracks -> increment lost
        for tid in track_ids:
            if tid not in matched_trk:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]

        return results


# ══════════════════════════════════════════════
# SEMANTIC SLAM — object position verifier
# ══════════════════════════════════════════════
class SemanticSLAM:
    """
    Lightweight Semantic SLAM for verifying detected object positions.

    For each tracked object, counts how many frames it has been
    consistently observed and how many ORB texture features fall
    inside its bounding box.  A detection is marked *verified* only
    when both thresholds are met, which suppresses false positives
    that appear for only one or two frames (reflections, blur, noise).

    Camera-pose estimation (Essential-matrix decomposition) is also
    performed so that, in future extensions, true 3-D triangulation
    can be added without refactoring the interface.
    """

    def __init__(self, min_observations: int = 3, min_features: int = 2):
        self.orb = cv2.ORB_create(nfeatures=1500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_gray = None
        self.prev_kpts = None
        self.prev_desc = None
        self.camera_matrix = None
        self.pose = np.eye(4)           # cumulative camera pose (world-to-cam)
        self.semantic_map = {}          # track_id -> dict
        self.min_observations = min_observations
        self.min_features = min_features
        self.verified_ids: set = set()
        self.frame_count = 0
        # ── 3-D map data ──────────────────────────────────────────────
        self.cam_traj: list = [np.zeros(3)]   # camera positions in world frame
        self.prev_centers: dict = {}           # track_id -> (cx, cy) prev frame
        self._frame_size: tuple = (1920, 1080)
        # ── 3-D point cloud (reconstructed environment) ──────────────
        self.point_cloud: list = []            # [(x, y, z, cls_label)] world pts
        self._max_cloud_pts: int = 10000       # ring-buffer cap

    def _init_camera(self, w: int, h: int):
        """Estimate pinhole camera matrix from frame dimensions."""
        f = float(max(w, h))
        self.camera_matrix = np.array(
            [[f, 0, w / 2.0],
             [0, f, h / 2.0],
             [0, 0, 1.0]], dtype=np.float64)

    def update(self, frame: np.ndarray, tracked_detections: list) -> set:
        """
        Process one frame.

        tracked_detections: list of (x1, y1, x2, y2, cls_id, conf, track_id)
        Returns: set of track_ids whose position is verified.
        """
        h, w = frame.shape[:2]
        if self.camera_matrix is None:
            self._init_camera(w, h)
        self._frame_size = (w, h)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kpts, desc = self.orb.detectAndCompute(gray, None)

        # ── Camera-motion estimation ──────────────────────────────────
        if (self.prev_gray is not None
                and desc is not None
                and self.prev_desc is not None
                and kpts is not None and len(kpts) >= 8):
            matches = self.bf.match(self.prev_desc, desc)
            matches = sorted(matches, key=lambda m: m.distance)[:200]
            if len(matches) >= 8:
                pts1 = np.float32([self.prev_kpts[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kpts[m.trainIdx].pt for m in matches])
                E, mask = cv2.findEssentialMat(
                    pts1, pts2, self.camera_matrix,
                    method=cv2.RANSAC, prob=0.999, threshold=1.0,
                )
                if E is not None:
                    _, R, t, _ = cv2.recoverPose(
                        E, pts1, pts2, self.camera_matrix, mask=mask)
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = t.flatten()
                    self.pose = T @ self.pose

                    # ── Triangulate ALL matched ORB features ──────────
                    K = self.camera_matrix
                    P1 = K @ np.eye(3, 4)
                    P2 = K @ np.hstack([R, t])
                    pose_inv = np.linalg.inv(self.pose)
                    R_w = pose_inv[:3, :3]
                    t_w = pose_inv[:3, 3]

                    # Batch-triangulate all matched feature pairs
                    pts4d_all = cv2.triangulatePoints(
                        P1, P2, pts1.T, pts2.T)        # (4, N)

                    for i in range(pts4d_all.shape[1]):
                        w4 = pts4d_all[3, i]
                        if abs(w4) < 1e-7:
                            continue
                        pos_cam = pts4d_all[:3, i] / w4
                        if pos_cam[2] <= 0 or pos_cam[2] > 200:
                            continue
                        pos_world = R_w @ pos_cam + t_w

                        # Label: which detection bbox contains this 2D point?
                        px, py = pts2[i]
                        lbl = -1            # background
                        for det in tracked_detections:
                            if det[0] <= px <= det[2] and det[1] <= py <= det[3]:
                                lbl = int(det[4])
                                break
                        self.point_cloud.append(
                            (float(pos_world[0]), float(pos_world[1]),
                             float(pos_world[2]), lbl))

                    # Cap point-cloud size (keep most recent)
                    if len(self.point_cloud) > self._max_cloud_pts:
                        self.point_cloud = self.point_cloud[-self._max_cloud_pts:]

                    # ── Also triangulate object centers for pos3d ─────
                    for det in tracked_detections:
                        tid = det[6]
                        if tid < 0 or tid not in self.prev_centers:
                            continue
                        pcx, pcy = self.prev_centers[tid]
                        cx2 = (det[0] + det[2]) / 2.0
                        cy2 = (det[1] + det[3]) / 2.0
                        pts4d = cv2.triangulatePoints(
                            P1, P2,
                            np.float32([[pcx, pcy]]).T,
                            np.float32([[cx2, cy2]]).T,
                        )
                        w4v = pts4d[3, 0]
                        if abs(w4v) < 1e-7:
                            continue
                        pc = (pts4d[:3, 0] / w4v).astype(float)
                        if pc[2] <= 0:
                            continue
                        pw = R_w @ pc + t_w
                        e = self.semantic_map.get(tid)
                        if e is not None:
                            if e["pos3d_sum"] is None:
                                e["pos3d_sum"] = pw.copy()
                            else:
                                e["pos3d_sum"] += pw
                            e["pos3d_n"] += 1

                    # ── Record camera position in world frame ─────────
                    self.cam_traj.append(t_w.copy())

        # ── Update semantic map ───────────────────────────────────────
        for det in tracked_detections:
            x1, y1, x2, y2, cls_id, conf, track_id = det
            if track_id < 0:
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Count ORB keypoints that land inside this bounding box
            n_kpts = 0
            if kpts:
                n_kpts = sum(
                    1 for kp in kpts
                    if x1 <= kp.pt[0] <= x2 and y1 <= kp.pt[1] <= y2
                )

            if track_id not in self.semantic_map:
                self.semantic_map[track_id] = {
                    "cls": cls_id,
                    "obs": 1,
                    "kpts_sum": n_kpts,
                    "consistent": True,
                    "cx_sum": cx,
                    "cy_sum": cy,
                    "conf_sum": conf,
                    "pos3d_sum": None,
                    "pos3d_n": 0,
                }
            else:
                e = self.semantic_map[track_id]
                e["obs"] += 1
                e["kpts_sum"] += n_kpts
                e["cx_sum"] += cx
                e["cy_sum"] += cy
                e["conf_sum"] += conf
                if e["cls"] != cls_id:
                    e["consistent"] = False     # class flip → false positive

        # ── Save centers for next-frame triangulation ─────────────────
        self.prev_centers = {
            det[6]: ((det[0] + det[2]) / 2.0, (det[1] + det[3]) / 2.0)
            for det in tracked_detections if det[6] >= 0
        }

        # ── Decide which tracks are verified ─────────────────────────
        self.verified_ids = set()
        for tid, e in self.semantic_map.items():
            avg_kpts = e["kpts_sum"] / max(e["obs"], 1)
            if (e["obs"] >= self.min_observations
                    and e["consistent"]
                    and avg_kpts >= self.min_features):
                self.verified_ids.add(tid)

        self.prev_gray = gray
        self.prev_kpts = kpts
        self.prev_desc = desc
        self.frame_count += 1
        return self.verified_ids

    def get_stats(self) -> tuple:
        """Returns (total_tracked, verified_count)."""
        return len(self.semantic_map), len(self.verified_ids)

    def get_map_data(self) -> dict:
        """
        Returns all data needed to render the 3-D semantic map.

        Each object entry contains:
          - track_id, cls, verified, obs
          - cx_mean, cy_mean  : average image-space center (pixels)
          - conf_avg          : average detection confidence
          - pos3d             : triangulated world position (x,y,z) or None
        """
        fw, fh = self._frame_size
        objects = []
        for tid, e in self.semantic_map.items():
            obs = max(e["obs"], 1)
            pos3d = None
            if e["pos3d_n"] > 0 and e["pos3d_sum"] is not None:
                pos3d = e["pos3d_sum"] / e["pos3d_n"]
            objects.append({
                "track_id": tid,
                "cls": e["cls"],
                "verified": tid in self.verified_ids,
                "obs": e["obs"],
                "cx_mean": e["cx_sum"] / obs,
                "cy_mean": e["cy_sum"] / obs,
                "conf_avg": e["conf_sum"] / obs,
                "pos3d": pos3d,
            })
        return {
            "objects": objects,
            "cam_traj": self.cam_traj,
            "point_cloud": self.point_cloud,   # [(x,y,z,cls_label), ...]
            "frame_w": fw,
            "frame_h": fh,
        }


# ══════════════════════════════════════════════
# SEMANTIC 3-D MAP RENDERER
# ══════════════════════════════════════════════

_MAP_BG         = (20, 30, 48)
_MAP_GRID       = (35, 48, 68)
_CLOUD_BG_COLOR = (55, 65, 80)        # background (environment) points
_CLOUD_OBJ_COLORS = {}                # filled dynamically per cls
_MAP_TRAJ       = (200, 200, 200)
_CANVAS_W, _CANVAS_H = 820, 780
_MARGIN = 70
_LEGEND_H = 70
_PLOT_W = _CANVAS_W - 2 * _MARGIN
_PLOT_H = _CANVAS_H - 2 * _MARGIN - _LEGEND_H


def _cls_color_bgr(cls_id: int, wheel_cls: int, chock_cls: int):
    """Returns (name, BGR) for a class id."""
    if cls_id == wheel_cls:
        return "Wheel",  (220,  60,  60)
    if cls_id == chock_cls:
        return "Chock",  ( 50, 165, 255)
    return "Person", ( 50, 205,  50)


def _w2c(x, z, x_min, x_max, z_min, z_max):
    """World XZ → canvas pixel (col, row)."""
    xr = max(x_max - x_min, 1e-3)
    zr = max(z_max - z_min, 1e-3)
    col = _MARGIN + int((x - x_min) / xr * _PLOT_W)
    row = _MARGIN + int((1.0 - (z - z_min) / zr) * _PLOT_H)
    return col, row


def _pseudo_xz(cx_mean, cy_mean, fw, fh):
    """Rough ground-plane XZ from image coordinates."""
    z = max((1.0 - cy_mean / fh) * 10.0, 0.05)
    x = (cx_mean / fw - 0.5) * z * (fw / fh) * 2.0
    return float(x), float(z)


def render_map_3d(slam: "SemanticSLAM", wheel_cls: int, chock_cls: int) -> np.ndarray:
    """
    Render a bird's-eye view of the **reconstructed 3-D environment**.

    The canvas shows:
      1. Sparse point cloud — triangulated ORB features that form the
         visible "space".  Background points are dark-grey; points that
         fell inside a detection bbox are tinted with the class colour.
      2. Camera trajectory — white polyline with a bright dot at the
         current position.
      3. Verified objects — large coloured circles with labels.  These
         are the detections confirmed to occupy real 3-D structure.
      4. Unverified objects — small hollow grey circles.

    Returns a BGR uint8 image of shape (_CANVAS_H, _CANVAS_W, 3).
    """
    data = slam.get_map_data()
    objects    = data["objects"]
    cam_traj   = data["cam_traj"]
    cloud      = data["point_cloud"]     # [(x,y,z,cls_label), ...]
    fw, fh     = data["frame_w"], data["frame_h"]

    canvas = np.full((_CANVAS_H, _CANVAS_W, 3), _MAP_BG, dtype=np.uint8)

    # ── Collect every XZ we will draw so we can compute bounds ────────
    all_xz = []

    # Point cloud (use X and Z)
    for (px, py, pz, _lbl) in cloud:
        all_xz.append((px, pz))

    # Camera trajectory
    traj_xz = []
    for p in cam_traj:
        if isinstance(p, np.ndarray) and p.shape[0] >= 3:
            traj_xz.append((float(p[0]), float(p[2])))
    all_xz.extend(traj_xz)

    # Object centres (triangulated or pseudo)
    obj_xz = []
    for obj in objects:
        if obj["pos3d"] is not None:
            p = obj["pos3d"]
            ox, oz = float(p[0]), float(p[2])
        else:
            ox, oz = _pseudo_xz(obj["cx_mean"], obj["cy_mean"], fw, fh)
        obj_xz.append((ox, oz))
        all_xz.append((ox, oz))

    if not all_xz:
        cv2.putText(canvas, "Waiting for 3-D data...",
                    (180, _CANVAS_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (130, 130, 130), 2)
        return canvas

    xs = [p[0] for p in all_xz]
    zs = [p[1] for p in all_xz]

    # Robust bounds: clip outliers at 2nd / 98th percentile
    if len(xs) > 20:
        xs_s = sorted(xs)
        zs_s = sorted(zs)
        lo = int(len(xs_s) * 0.02)
        hi = int(len(xs_s) * 0.98)
        xs_s = xs_s[lo:hi + 1]
        zs_s = zs_s[lo:hi + 1]
    else:
        xs_s, zs_s = xs, zs

    pad_x = max((max(xs_s) - min(xs_s)) * 0.12, 0.3)
    pad_z = max((max(zs_s) - min(zs_s)) * 0.12, 0.3)
    x_min, x_max = min(xs_s) - pad_x, max(xs_s) + pad_x
    z_min, z_max = min(zs_s) - pad_z, max(zs_s) + pad_z

    # ── Grid ──────────────────────────────────────────────────────────
    for i in range(6):
        frac = i / 5.0
        cx_g, _ = _w2c(x_min + frac * (x_max - x_min), z_min,
                        x_min, x_max, z_min, z_max)
        _, rz_g = _w2c(x_min, z_min + frac * (z_max - z_min),
                        x_min, x_max, z_min, z_max)
        cv2.line(canvas, (cx_g, _MARGIN), (cx_g, _MARGIN + _PLOT_H),
                 _MAP_GRID, 1)
        cv2.line(canvas, (_MARGIN, rz_g), (_MARGIN + _PLOT_W, rz_g),
                 _MAP_GRID, 1)

    # ── 1. Point cloud (environment) ─────────────────────────────────
    for (px, py, pz, lbl) in cloud:
        col, row = _w2c(px, pz, x_min, x_max, z_min, z_max)
        if not (_MARGIN <= col < _MARGIN + _PLOT_W
                and _MARGIN <= row < _MARGIN + _PLOT_H):
            continue
        if lbl < 0:
            # Background point — tiny dark dot
            canvas[row, col] = _CLOUD_BG_COLOR
            # Cross-hair 1px to make it more visible
            if col + 1 < _CANVAS_W:
                canvas[row, col + 1] = _CLOUD_BG_COLOR
            if row + 1 < _CANVAS_H:
                canvas[row + 1, col] = _CLOUD_BG_COLOR
        else:
            # Object point — coloured 2×2 block
            _, c = _cls_color_bgr(lbl, wheel_cls, chock_cls)
            # Brighter tint for cloud points inside bbox
            bright = tuple(min(255, int(ch * 0.6 + 80)) for ch in c)
            for dr in range(3):
                for dc in range(3):
                    rr, cc = row + dr - 1, col + dc - 1
                    if (_MARGIN <= cc < _MARGIN + _PLOT_W
                            and _MARGIN <= rr < _MARGIN + _PLOT_H):
                        canvas[rr, cc] = bright

    # ── 2. Camera trajectory ──────────────────────────────────────────
    tpts = [_w2c(tx, tz, x_min, x_max, z_min, z_max) for tx, tz in traj_xz]
    if len(tpts) > 1:
        for i in range(len(tpts) - 1):
            cv2.line(canvas, tpts[i], tpts[i + 1], _MAP_TRAJ, 2, cv2.LINE_AA)
    if tpts:
        # Camera current pos — bright white circle with halo
        cv2.circle(canvas, tpts[-1], 7, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, tpts[-1], 10, (255, 255, 255), 1, cv2.LINE_AA)

    # ── 3. Verified object markers ────────────────────────────────────
    legend_entries: dict = {}
    for obj, (ox, oz) in zip(objects, obj_xz):
        col, row = _w2c(ox, oz, x_min, x_max, z_min, z_max)
        name, color = _cls_color_bgr(obj["cls"], wheel_cls, chock_cls)
        verified = obj["verified"]
        tid = obj["track_id"]
        conf = obj["conf_avg"]

        if verified:
            r = max(12, min(24, 12 + obj["obs"] // 3))
            cv2.circle(canvas, (col, row), r, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (col, row), r + 2, (255, 255, 255), 2, cv2.LINE_AA)
            lbl_txt = f"{name}#{tid} {conf:.0%}"
            cv2.putText(canvas, lbl_txt, (col + r + 5, row + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            legend_entries[name] = color
        else:
            cv2.circle(canvas, (col, row), 5, (100, 100, 100), 1, cv2.LINE_AA)

    # ── Plot border ───────────────────────────────────────────────────
    cv2.rectangle(canvas, (_MARGIN, _MARGIN),
                  (_MARGIN + _PLOT_W, _MARGIN + _PLOT_H), (80, 100, 130), 1)

    # ── Title ─────────────────────────────────────────────────────────
    title = "Semantic SLAM — 3D Environment"
    n_pts = len(cloud)
    n_ver = sum(1 for o in objects if o["verified"])
    sub = f"  |  {n_pts} pts   {len(traj_xz)} cam poses   {n_ver} verified obj"
    cv2.putText(canvas, title, (_MARGIN, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(canvas, sub, (_MARGIN, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 160, 175), 1, cv2.LINE_AA)

    # ── Axis labels ───────────────────────────────────────────────────
    cv2.putText(canvas, "X", (_MARGIN + _PLOT_W + 5, _MARGIN + _PLOT_H + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 160, 175), 1)
    cv2.putText(canvas, "Z (depth)", (_MARGIN - 5, _MARGIN - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 160, 175), 1)

    # ── Legend strip ──────────────────────────────────────────────────
    ly = _CANVAS_H - _LEGEND_H + 15
    cv2.line(canvas, (_MARGIN, ly - 12), (_CANVAS_W - _MARGIN, ly - 12),
             _MAP_GRID, 1)
    lx = _MARGIN
    # Cloud dot
    cv2.circle(canvas, (lx + 6, ly + 8), 3, _CLOUD_BG_COLOR, -1)
    cv2.putText(canvas, "Environment", (lx + 14, ly + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 140, 150), 1)
    lx += 110
    # Camera
    cv2.circle(canvas, (lx + 6, ly + 8), 5, (255, 255, 255), -1)
    cv2.putText(canvas, "Camera", (lx + 16, ly + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    lx += 90
    # Per-class entries
    for lname, lcol in legend_entries.items():
        cv2.circle(canvas, (lx + 6, ly + 8), 7, lcol, -1)
        cv2.putText(canvas, lname, (lx + 18, ly + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, lcol, 1)
        lx += 90
    # Unverified
    cv2.circle(canvas, (lx + 6, ly + 8), 5, (100, 100, 100), 1)
    cv2.putText(canvas, "Unverified", (lx + 16, ly + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    return canvas


# ══════════════════════════════════════════════
# CYRILLIC TEXT ON FRAME (PIL-based)
# ══════════════════════════════════════════════
_font_cache = {}

def _get_font(font_size):
    if font_size in _font_cache:
        return _font_cache[font_size]
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()
    _font_cache[font_size] = font
    return font


def put_cyrillic_text(frame, text, pos, color_bgr, font_size=36, bg_color=(0, 0, 0)):
    """Draw Cyrillic text on OpenCV frame using PIL — only converts a small strip."""
    font = _get_font(font_size)

    # Measure text size on a dummy image to know the strip height
    dummy = Image.new("RGB", (1, 1))
    bbox = ImageDraw.Draw(dummy).textbbox(pos, text, font=font)
    pad = 8
    strip_y1 = max(0, bbox[1] - pad)
    strip_y2 = min(frame.shape[0], bbox[3] + pad + 1)
    strip_x1 = max(0, bbox[0] - pad)
    strip_x2 = min(frame.shape[1], bbox[2] + pad + 1)

    # Convert only the small strip to PIL
    strip = frame[strip_y1:strip_y2, strip_x1:strip_x2]
    pil_strip = Image.fromarray(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_strip)

    # Adjusted coordinates relative to strip
    adj_pos = (pos[0] - strip_x1, pos[1] - strip_y1)
    adj_bbox = (bbox[0] - strip_x1, bbox[1] - strip_y1, bbox[2] - strip_x1, bbox[3] - strip_y1)

    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.rectangle(
        [adj_bbox[0] - pad, adj_bbox[1] - pad, adj_bbox[2] + pad, adj_bbox[3] + pad],
        fill=(bg_color[2], bg_color[1], bg_color[0]),
    )
    draw.text(adj_pos, text, font=font, fill=color_rgb)

    frame[strip_y1:strip_y2, strip_x1:strip_x2] = cv2.cvtColor(np.array(pil_strip), cv2.COLOR_RGB2BGR)
    return frame


def is_point_near_box(px, py, box, margin=50):
    x1, y1, x2, y2 = box
    return (x1 - margin <= px <= x2 + margin) and (y1 - margin <= py <= y2 + margin)


# ══════════════════════════════════════════════
# PROCESS FRAME
# ══════════════════════════════════════════════
def process_frame(
    frame, combo_model, det_model, pose_model,
    use_w, use_p, use_po,
    conf_w, conf_p, conf_kpt,
    w_cls, c_cls, imgsz_c,
    lw, fs,
    svc_status, svc_color, show_bar,
    tracker, show_ids, show_filt_stats,
    slam=None, show_slam_stats=False,
):
    h, w = frame.shape[:2]
    annotated = frame
    global_chocks = []
    filtered_count = 0

    # ── ЭТАП 1: Колёса + колодки ──
    combo_detections = []
    if use_w and combo_model is not None:
        combo_res = combo_model(frame, conf=conf_w, imgsz=imgsz_c, verbose=False, half=_USE_HALF)
        for box in combo_res[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            conf_val = float(box.conf[0].cpu().numpy())

            # ★ ФИЛЬТРАЦИЯ ★
            if not filter_detection(x1, y1, x2, y2, cls_id, h, w):
                filtered_count += 1
                continue

            combo_detections.append((x1, y1, x2, y2, cls_id, conf_val))

    # ★ ТРЕКИНГ ★
    if tracking_enabled and tracker is not None:
        tracked = tracker.update(combo_detections)
    else:
        tracked = [(* d, -1) for d in combo_detections]

    # ── SLAM verification ─────────────────────────────────────────────
    verified_ids: set = set()
    if slam is not None:
        verified_ids = slam.update(frame, list(tracked))

    # Draw combo detections
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
                # Thin outer grey border indicates "pending verification"
                cv2.rectangle(annotated, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (160, 160, 160), 1)
            else:
                # Extra thin bright-green outer border confirms position
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

    # SLAM statistics overlay
    if show_slam_stats and slam is not None:
        total_obj, verified_obj = slam.get_stats()
        slam_text = f"SLAM: {verified_obj}/{total_obj} verified"
        cv2.putText(annotated, slam_text, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

    # Filtered stats overlay
    if show_filt_stats and filtered_count > 0:
        cv2.putText(annotated, f"Filtered: {filtered_count}", (w - 250, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

    # ── ЭТАП 2: Люди ──
    person_boxes = []
    if use_p and det_model is not None:
        det_res = det_model(frame, classes=[0], conf=conf_p, verbose=False, half=_USE_HALF)
        person_boxes = det_res[0].boxes.xyxy.cpu().numpy()

    # ── ЭТАП 3: Поза (crop-based, batched) ──
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

                # Skeleton
                for p1_idx, p2_idx in SKELETON_MPII:
                    if p1_idx < num_kpts and p2_idx < num_kpts:
                        kx1, ky1, c1 = kpts[p1_idx]
                        kx2, ky2, c2 = kpts[p2_idx]
                        if c1 > conf_kpt and c2 > conf_kpt:
                            pt1 = (int(kx1) + bx1, int(ky1) + by1)
                            pt2 = (int(kx2) + bx1, int(ky2) + by1)
                            cv2.line(annotated, pt1, pt2, SKEL_COLOR, lw + 1, cv2.LINE_AA)

                # Keypoints
                for idx in range(min(num_kpts, MPII_N)):
                    kx, ky, kc = kpts[idx]
                    if kc > conf_kpt:
                        cx, cy = int(kx) + bx1, int(ky) + by1
                        cv2.circle(annotated, (cx, cy), 5, KPT_COLORS[idx], -1, cv2.LINE_AA)
                        cv2.circle(annotated, (cx, cy), 6, (255, 255, 255), 1, cv2.LINE_AA)

                # Chock placement logic
                knees_y = []
                if num_kpts > 4:
                    if kpts[1][2] > conf_kpt: knees_y.append(kpts[1][1] + by1)
                    if kpts[4][2] > conf_kpt: knees_y.append(kpts[4][1] + by1)

                wrists = []
                if num_kpts > 15:
                    if kpts[10][2] > 0.4: wrists.append((kpts[10][0] + bx1, kpts[10][1] + by1))
                    if kpts[15][2] > 0.4: wrists.append((kpts[15][0] + bx1, kpts[15][1] + by1))

                is_bending = False
                if knees_y and wrists:
                    min_knee_y = min(knees_y)
                    for wx, wy in wrists:
                        if wy > min_knee_y:
                            is_bending = True
                            break

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

    # Person boxes
    if use_p:
        for pbox in person_boxes:
            bx1, by1, bx2, by2 = map(int, pbox)
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (50, 205, 50), lw)
            cv2.putText(annotated, "Person", (bx1, by1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (50, 205, 50), 2)

    # ── Status bar (Cyrillic via PIL) ──
    if show_bar:
        # Black bar at top
        cv2.rectangle(annotated, (0, 0), (w, 70), (0, 0, 0), -1)
        annotated = put_cyrillic_text(annotated, svc_status, (20, 12), svc_color, font_size=32)

    return annotated, svc_status, svc_color


# ══════════════════════════════════════════════
# VIDEO UPLOAD & PROCESSING
# ══════════════════════════════════════════════
st.markdown("### 🎬 Загрузка видео")

uploaded_file = st.file_uploader(
    "Перетащите видеофайл или нажмите для выбора",
    type=["mp4", "avi", "mov", "mkv", "wmv"],
)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.flush()
    input_path = tfile.name

    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📐 Разрешение", f"{vid_w}×{vid_h}")
    c2.metric("🎞️ Кадров", f"{total_frames}")
    c3.metric("⏱️ Длительность", f"{duration:.1f} сек")
    c4.metric("🔄 FPS", f"{fps:.1f}")

    st.markdown("---")

    with st.expander("👁️ Предпросмотр", expanded=False):
        st.video(input_path)

    if st.button("🚀 Запустить обработку", type="primary", use_container_width=True):

        loaded = {}
        with st.spinner("Загрузка моделей..."):
            if use_wheels:
                m = load_model(model_path_wheels)
                if m: loaded["combo"] = m
                else: st.warning(f"⚠️ {model_path_wheels}")
            if use_person:
                m = load_model(model_path_person)
                if m: loaded["person"] = m
                else: st.warning(f"⚠️ {model_path_person}")
            if use_pose:
                m = load_model(model_path_pose)
                if m: loaded["pose"] = m
                else: st.warning(f"⚠️ {model_path_pose}")

        if not loaded:
            st.error("❌ Ни одна модель не загружена.")
        else:
            st.success(f"✅ Загружено: {', '.join(loaded.keys())}")

            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            cap = cv2.VideoCapture(input_path)
            out_fps = fps / process_every_n
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (vid_w, vid_h))

            progress_bar = st.progress(0, text="Обработка...")
            col_vid, col_map = st.columns([3, 2])
            frame_display = col_vid.empty()
            map_display = col_map.empty() if slam_enabled else None

            tracker = SimpleTracker(iou_threshold=iou_track_thresh) if tracking_enabled else None
            slam = SemanticSLAM(min_observations=slam_min_obs, min_features=slam_min_feat) if slam_enabled else None

            frame_idx = 0
            processed_count = 0
            start_time = time.time()
            svc_status = "ОЖИДАНИЕ: Установите колодки"
            svc_color = (0, 0, 255)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % process_every_n == 0:
                    annotated, svc_status, svc_color = process_frame(
                        frame, loaded.get("combo"), loaded.get("person"), loaded.get("pose"),
                        use_wheels, use_person, use_pose,
                        conf_wheels, conf_person, conf_pose_kpt,
                        wheel_class_id, chock_class_id, combo_imgsz,
                        line_thickness, font_scale,
                        svc_status, svc_color, show_status_bar,
                        tracker, show_track_ids, show_filtered_stats,
                        slam=slam, show_slam_stats=show_slam_overlay,
                    )
                    writer.write(annotated)
                    processed_count += 1

                    if processed_count % 15 == 0:
                        frame_display.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                        if slam is not None and map_display is not None:
                            _map_rgb = cv2.cvtColor(
                                render_map_3d(slam, wheel_class_id, chock_class_id),
                                cv2.COLOR_BGR2RGB,
                            )
                            map_display.image(_map_rgb, caption="SLAM · bird's-eye (live)", use_container_width=True)

                if frame_idx % 10 == 0 or frame_idx == total_frames - 1:
                    progress = (frame_idx + 1) / total_frames
                    elapsed = time.time() - start_time
                    remaining = max(0, (elapsed / (frame_idx + 1) * total_frames) - elapsed) if frame_idx > 0 else 0
                    progress_bar.progress(min(progress, 1.0),
                        text=f"Кадр {frame_idx+1}/{total_frames} | Обработано: {processed_count} | ~{remaining:.0f} сек")
                frame_idx += 1

            cap.release()
            writer.release()

            total_time = time.time() - start_time
            progress_bar.progress(1.0, text="✅ Готово!")

            st.markdown("### 📊 Результаты")
            r1, r2, r3 = st.columns(3)
            r1.metric("⏱️ Время", f"{total_time:.1f} сек")
            r2.metric("🎞️ Кадров", f"{processed_count}")
            r3.metric("⚡ FPS", f"{processed_count/total_time:.1f}" if total_time > 0 else "—")

            # ── Semantic 3-D map — финальное обновление + скачать ────
            if slam is not None:
                total_obj, verified_obj = slam.get_stats()
                st.markdown("### 🗺️ Semantic SLAM — итог")
                sm1, sm2 = st.columns(2)
                sm1.metric("Всего объектов (треков)", total_obj)
                sm2.metric("Верифицированных", verified_obj)

                # Render final map and push to the live placeholder
                map_img_bgr = render_map_3d(slam, wheel_class_id, chock_class_id)
                map_img_rgb = cv2.cvtColor(map_img_bgr, cv2.COLOR_BGR2RGB)
                if map_display is not None:
                    map_display.image(map_img_rgb, caption="SLAM · bird's-eye (финал)", use_container_width=True)

                import io as _io
                buf = _io.BytesIO()
                Image.fromarray(map_img_rgb).save(buf, format="PNG")
                st.download_button("⬇️ Скачать карту (PNG)", buf.getvalue(),
                                   "slam_map.png", "image/png")

            st.markdown("### 🎬 Результат")
            h264_path = output_path.replace(".mp4", "_h264.mp4")
            os.system(f'ffmpeg -y -i "{output_path}" -vcodec libx264 -acodec aac "{h264_path}" -loglevel quiet')
            final = h264_path if (os.path.exists(h264_path) and os.path.getsize(h264_path) > 0) else output_path

            st.video(final)
            with open(final, "rb") as f:
                st.download_button("⬇️ Скачать видео", f, "processed_video.mp4", "video/mp4", use_container_width=True)

            for p in [input_path, output_path, h264_path]:
                try:
                    if os.path.exists(p): os.unlink(p)
                except: pass
else:
    st.markdown("""
    <div style="border:2px dashed #4a5568;border-radius:16px;padding:3rem;text-align:center;background:linear-gradient(145deg,#f7fafc,#edf2f7);margin:2rem 0;">
        <p style="font-size:3rem;margin:0;">🎥</p>
        <p style="font-size:1.2rem;color:#4a5568;font-weight:600;">Загрузите видеофайл</p>
        <p style="font-size:0.9rem;color:#718096;">MP4, AVI, MOV, MKV, WMV</p>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div style="text-align:center;color:#9ca3af;font-size:0.85rem;">✈️ Aircraft Detection v3 | YOLOv8 + MPII Pose + Tracking + FP Filter</div>', unsafe_allow_html=True)
