"""
Semantic SLAM module — with Visual SLAM dynamic masking.

SemanticSLAM теперь использует DynamicMasker из sources/ (slam.zip):
  • YOLOv8-seg маскирует person + car в каждом кадре в фоновом потоке.
  • SIFT feature detection запускается только на незамаскированных
    (статичных) пикселях — люди, тележки, машины не загрязняют облако точек.

Совместимость с app.py сохранена полностью:
  SemanticSLAM(min_observations, min_features) — тот же конструктор.
  .update(frame, tracked_detections) -> set    — тот же интерфейс.
  .get_stats() / .get_map_data()               — без изменений.
"""

import sys
import os

import cv2
import numpy as np

# ------------------------------------------------------------------
# Подключаем DynamicMasker из sources/ (slam.zip)
# ------------------------------------------------------------------
_SOURCES = os.path.join(os.path.dirname(__file__), '..', 'sources')
if _SOURCES not in sys.path:
    sys.path.insert(0, _SOURCES)

try:
    from sources.masker import DynamicMasker
    _MASKER_AVAILABLE = True
except ImportError:
    _MASKER_AVAILABLE = False


class SemanticSLAM:
    """
    Semantic SLAM с SIFT-фичами, keyframe-триангуляцией и фильтрацией
    по ошибке репроекции.

    Расширено: DynamicMasker из slam.zip маскирует динамические объекты
    (person, car) перед SIFT-детектированием, что резко улучшает качество
    точечного облака — колёса и колодки не теряются в «шуме» от людей.
    """

    _KEYFRAME_MIN_MATCHES  = 40
    _KEYFRAME_MIN_MOVEMENT = 0.005
    _MAX_ROTATION_DEG      = 7.0   # reject poses with rotation > this per frame
    _TRAJ_SMOOTH_ALPHA     = 0.5   # EMA smoothing for trajectory (0=frozen, 1=raw)
    _MIN_DISP_PX           = 2.5   # median px displacement below this → car stationary
    _ROT_TRANS_DECAY_START = 2.0   # degrees: above this rotation starts damping translation
    _ROT_TRANS_DECAY_END   = 6.0   # degrees: at this rotation, translation → 0

    def __init__(
        self,
        min_observations: int = 3,
        min_features: int = 2,
        # --- DynamicMasker параметры ---
        enable_visual_masking: bool = True,
        yolo_model:  str   = 'yolov8n-seg.pt',
        yolo_conf:   float = 0.25,
        yolo_imgsz:  tuple = (480, 480),
    ):
        self.sift    = cv2.SIFT_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        self.prev_gray  = None
        self.prev_kpts  = None
        self.prev_desc  = None
        self.camera_matrix = None
        self.pose = np.eye(4)

        self.semantic_map: dict = {}
        self.min_observations   = min_observations
        self.min_features       = min_features
        self.verified_ids: set  = set()
        self.frame_count        = 0

        self.cam_traj: list     = [np.zeros(3)]
        self.prev_centers: dict = {}
        self._frame_size: tuple = (1920, 1080)
        self.point_cloud: list  = []
        self._max_cloud_pts     = 15000

        self._kf_gray  = None
        self._kf_kpts  = None
        self._kf_desc  = None
        self._kf_pose  = np.eye(4)
        self._frames_since_kf = 0

        # --- DynamicMasker ---
        self._masker = None
        self._masking_enabled = False
        self._mask_total_kpts    = 0
        self._mask_filtered_kpts = 0

        if enable_visual_masking and _MASKER_AVAILABLE:
            try:
                self._masker = DynamicMasker(
                    model_path=yolo_model,
                    input_size=yolo_imgsz,
                    conf_threshold=yolo_conf,
                )
                self._masker.start()
                self._masking_enabled = True
                print("[SemanticSLAM] DynamicMasker запущен — "
                      "person/car маскируются перед SIFT")
            except Exception as exc:
                print(f"[SemanticSLAM] DynamicMasker недоступен ({exc}), "
                      "работаем без маскировки.")
        elif enable_visual_masking and not _MASKER_AVAILABLE:
            print("[SemanticSLAM] sources/ не найден — работаем без маскировки.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self):
        """Останавливает фоновый поток маскировщика."""
        if self._masker is not None:
            self._masker.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_camera(self, w: int, h: int) -> None:
        f = float(max(w, h))
        self.camera_matrix = np.array(
            [[f, 0, w / 2.0],
             [0, f, h / 2.0],
             [0, 0, 1.0]], dtype=np.float64)

    def _match_features(self, desc1, desc2, ratio: float = 0.75) -> list:
        if desc1 is None or desc2 is None:
            return []
        raw = self.matcher.knnMatch(desc1, desc2, k=2)
        return [m for pair in raw if len(pair) == 2
                for m, n in [pair] if m.distance < ratio * n.distance]

    def _get_mask(self, frame: np.ndarray):
        """
        Возвращает статическую маску:
          255 = статичный фон (разрешено для SIFT)
            0 = динамический объект (маскировать)
        """
        if not self._masking_enabled or self._masker is None:
            return None
        self._masker.submit_frame(frame)
        return self._masker.get_mask_for_frame(frame.shape)

    def _apply_mask_to_gray(self, gray: np.ndarray, mask) -> np.ndarray:
        """Заменяет пиксели динамических объектов на 128."""
        if mask is None:
            return gray
        out = gray.copy()
        out[mask == 0] = 128
        return out

    def _count_filtered_kpts(self, kpts, mask) -> int:
        if mask is None or not kpts:
            return 0
        h, w = mask.shape[:2]
        return sum(
            1 for kp in kpts
            if 0 <= int(kp.pt[1]) < h
            and 0 <= int(kp.pt[0]) < w
            and mask[int(kp.pt[1]), int(kp.pt[0])] == 0
        )

    def _triangulate_and_filter(
        self, pts1, pts2, R, t, tracked_detections
    ) -> tuple:
        K  = self.camera_matrix
        P1 = K @ np.eye(3, 4)
        P2 = K @ np.hstack([R, t])

        pts4d    = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pose_inv = np.linalg.inv(self.pose)
        R_w      = pose_inv[:3, :3]
        t_w      = pose_inv[:3, 3]

        new_points = []
        for i in range(pts4d.shape[1]):
            w4 = pts4d[3, i]
            if abs(w4) < 1e-7:
                continue
            pos_cam = pts4d[:3, i] / w4
            if pos_cam[2] <= 0.1 or pos_cam[2] > 150:
                continue
            proj = K @ pos_cam
            if abs(proj[2]) < 1e-7:
                continue
            proj_px = proj[:2] / proj[2]
            if np.linalg.norm(proj_px - pts2[i]) > 3.0:
                continue
            pos_world = R_w @ pos_cam + t_w
            px, py = pts2[i]
            lbl = -1
            for det in tracked_detections:
                if det[0] <= px <= det[2] and det[1] <= py <= det[3]:
                    lbl = int(det[4])
                    break
            new_points.append((
                float(pos_world[0]), float(pos_world[1]),
                float(pos_world[2]), lbl,
            ))
        return new_points, R_w, t_w

    def _cap_cloud(self) -> None:
        if len(self.point_cloud) > self._max_cloud_pts:
            self.point_cloud = self.point_cloud[-self._max_cloud_pts:]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray, tracked_detections: list) -> set:
        """
        Обрабатывает один кадр.

        Args:
            frame:              BGR изображение.
            tracked_detections: list of (x1,y1,x2,y2,cls_id,conf,track_id).
        Returns:
            Множество верифицированных track_id.
        """
        h, w = frame.shape[:2]
        if self.camera_matrix is None:
            self._init_camera(w, h)
        self._frame_size = (w, h)

        # ── Маска динамических объектов ───────────────────────────────
        dyn_mask    = self._get_mask(frame)

        # ── Greyscale + маскировка → SIFT ─────────────────────────────
        gray        = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_masked = self._apply_mask_to_gray(gray, dyn_mask)
        kpts, desc  = self.sift.detectAndCompute(gray_masked, None)

        # Статистика
        if kpts:
            self._mask_total_kpts    += len(kpts)
            self._mask_filtered_kpts += self._count_filtered_kpts(kpts, dyn_mask)

        # ── Sequential motion estimation ──────────────────────────────
        if (self.prev_desc is not None and desc is not None
                and kpts is not None and len(kpts) >= 8):
            good = self._match_features(self.prev_desc, desc)
            if len(good) >= 8:
                pts1 = np.float32([self.prev_kpts[m.queryIdx].pt for m in good])
                pts2 = np.float32([kpts[m.trainIdx].pt          for m in good])
                E, mask = cv2.findEssentialMat(
                    pts1, pts2, self.camera_matrix,
                    method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E is not None:
                    _, R, t, pose_mask = cv2.recoverPose(
                        E, pts1, pts2, self.camera_matrix, mask=mask)

                    inlier_idx = pose_mask.ravel() > 0

                    # ── Problem 1 fix: stationary detection ───────────
                    # If median feature displacement is tiny, the car is
                    # not moving — skip pose update entirely to prevent
                    # backward drift caused by noise / vibration.
                    if np.any(inlier_idx):
                        disp_vals = np.linalg.norm(
                            pts2[inlier_idx] - pts1[inlier_idx], axis=1)
                        median_disp = float(np.median(disp_vals))
                    else:
                        median_disp = 0.0

                    if median_disp < self._MIN_DISP_PX:
                        pass  # car stationary — freeze trajectory
                    else:
                        # ── Sanity check: reject degenerate rotation ──
                        rot_angle = np.degrees(
                            np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
                        )
                        if rot_angle > self._MAX_ROTATION_DEG:
                            pass  # degenerate Essential Matrix — skip
                        else:
                            # ── Scale translation by pixel displacement ─
                            scale = median_disp / float(max(w, h))
                            scale = np.clip(scale, 0.0, 0.15)

                            # ── Problem 2 fix: rotation-translation decay
                            # When rotation dominates (car turning), the
                            # unit translation vector from recoverPose
                            # points sideways rather than forward, causing
                            # sharp bends. Damp translation proportionally
                            # to how rotational the motion is.
                            if rot_angle > self._ROT_TRANS_DECAY_START:
                                decay_range = (self._ROT_TRANS_DECAY_END
                                               - self._ROT_TRANS_DECAY_START)
                                rot_factor = 1.0 - np.clip(
                                    (rot_angle - self._ROT_TRANS_DECAY_START)
                                    / decay_range, 0.0, 1.0)
                                scale *= rot_factor

                            T = np.eye(4)
                            T[:3, :3] = R
                            T[:3, 3]  = t.flatten() * scale
                            self.pose = T @ self.pose

                            pts1_in, pts2_in = pts1[inlier_idx], pts2[inlier_idx]
                            if len(pts1_in) >= 8:
                                new_pts, R_w, t_w = self._triangulate_and_filter(
                                    pts1_in, pts2_in, R, t, tracked_detections)
                                self.point_cloud.extend(new_pts)
                                self._cap_cloud()

                            pose_inv = np.linalg.inv(self.pose)
                            R_w      = pose_inv[:3, :3]
                            t_w      = pose_inv[:3, 3]
                            K        = self.camera_matrix
                            P1 = K @ np.eye(3, 4)
                            P2 = K @ np.hstack([R, t])
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
                                    np.float32([[cx2, cy2]]).T)
                                w4v = pts4d[3, 0]
                                if abs(w4v) < 1e-7:
                                    continue
                                pc = (pts4d[:3, 0] / w4v).astype(float)
                                if pc[2] <= 0:
                                    continue
                                pw = R_w @ pc + t_w
                                e  = self.semantic_map.get(tid)
                                if e is not None:
                                    if e["pos3d_sum"] is None:
                                        e["pos3d_sum"] = pw.copy()
                                    else:
                                        e["pos3d_sum"] += pw
                                    e["pos3d_n"] += 1

                            # ── EMA-smoothed trajectory append ────────
                            a = self._TRAJ_SMOOTH_ALPHA
                            prev = self.cam_traj[-1]
                            smoothed = a * t_w + (1.0 - a) * prev
                            self.cam_traj.append(smoothed)

        # ── Keyframe triangulation ────────────────────────────────────
        self._frames_since_kf += 1
        if (self._kf_desc is not None and desc is not None
                and self._frames_since_kf >= 5):
            kf_good = self._match_features(self._kf_desc, desc)
            if len(kf_good) >= self._KEYFRAME_MIN_MATCHES:
                kf_pts1 = np.float32([self._kf_kpts[m.queryIdx].pt for m in kf_good])
                kf_pts2 = np.float32([kpts[m.trainIdx].pt          for m in kf_good])
                disp = np.median(np.linalg.norm(kf_pts2 - kf_pts1, axis=1))
                if disp > self._KEYFRAME_MIN_MOVEMENT * max(w, h):
                    E_kf, mask_kf = cv2.findEssentialMat(
                        kf_pts1, kf_pts2, self.camera_matrix,
                        method=cv2.RANSAC, prob=0.999, threshold=1.0)
                    if E_kf is not None:
                        _, R_kf, t_kf, pm_kf = cv2.recoverPose(
                            E_kf, kf_pts1, kf_pts2, self.camera_matrix, mask=mask_kf)
                        inl = pm_kf.ravel() > 0
                        if np.sum(inl) >= 10:
                            new_kf_pts, _, _ = self._triangulate_and_filter(
                                kf_pts1[inl], kf_pts2[inl],
                                R_kf, t_kf, tracked_detections)
                            self.point_cloud.extend(new_kf_pts)
                            self._cap_cloud()
                    self._kf_gray  = gray_masked.copy()
                    self._kf_kpts  = kpts
                    self._kf_desc  = desc.copy() if desc is not None else None
                    self._kf_pose  = self.pose.copy()
                    self._frames_since_kf = 0

        if self._kf_desc is None and desc is not None:
            self._kf_gray  = gray_masked.copy()
            self._kf_kpts  = kpts
            self._kf_desc  = desc.copy()
            self._kf_pose  = self.pose.copy()
            self._frames_since_kf = 0

        # ── Семантическая карта ───────────────────────────────────────
        for det in tracked_detections:
            x1, y1, x2, y2, cls_id, conf, track_id = det
            if track_id < 0:
                continue
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            n_kpts = 0
            if kpts:
                n_kpts = sum(1 for kp in kpts
                             if x1 <= kp.pt[0] <= x2 and y1 <= kp.pt[1] <= y2)
            if track_id not in self.semantic_map:
                self.semantic_map[track_id] = {
                    "cls": cls_id, "obs": 1, "kpts_sum": n_kpts,
                    "consistent": True, "cx_sum": cx, "cy_sum": cy,
                    "conf_sum": conf, "pos3d_sum": None, "pos3d_n": 0,
                }
            else:
                e = self.semantic_map[track_id]
                e["obs"]      += 1
                e["kpts_sum"] += n_kpts
                e["cx_sum"]   += cx
                e["cy_sum"]   += cy
                e["conf_sum"] += conf
                if e["cls"] != cls_id:
                    e["consistent"] = False

        self.prev_centers = {
            det[6]: ((det[0] + det[2]) / 2.0, (det[1] + det[3]) / 2.0)
            for det in tracked_detections if det[6] >= 0
        }

        # ── Верификация треков ────────────────────────────────────────
        self.verified_ids = set()
        for tid, e in self.semantic_map.items():
            avg_kpts = e["kpts_sum"] / max(e["obs"], 1)
            if (e["obs"] >= self.min_observations
                    and e["consistent"]
                    and avg_kpts >= self.min_features):
                self.verified_ids.add(tid)

        self.prev_gray  = gray_masked
        self.prev_kpts  = kpts
        self.prev_desc  = desc
        self.frame_count += 1
        return self.verified_ids

    # ------------------------------------------------------------------
    # Stats & map data
    # ------------------------------------------------------------------

    def get_stats(self) -> tuple:
        """Возвращает (всего треков, верифицированных). Совместимо с app.py."""
        return len(self.semantic_map), len(self.verified_ids)

    def get_mask_stats(self) -> dict:
        """Статистика маскировки динамических объектов."""
        if not self._masking_enabled:
            return {"enabled": False}
        total   = self._mask_total_kpts
        filt    = self._mask_filtered_kpts
        rate    = filt / total if total > 0 else 0.0
        return {
            "enabled":         True,
            "total_kpts":      total,
            "filtered_kpts":   filt,
            "filter_rate_pct": round(rate * 100, 1),
        }

    def get_map_data(self) -> dict:
        """Все данные для build_slam_plotly(). Совместимо с visualization.py."""
        fw, fh  = self._frame_size
        objects = []
        for tid, e in self.semantic_map.items():
            obs   = max(e["obs"], 1)
            pos3d = None
            if e["pos3d_n"] > 0 and e["pos3d_sum"] is not None:
                pos3d = e["pos3d_sum"] / e["pos3d_n"]
            objects.append({
                "track_id": tid,
                "cls":      e["cls"],
                "verified": tid in self.verified_ids,
                "obs":      e["obs"],
                "cx_mean":  e["cx_sum"] / obs,
                "cy_mean":  e["cy_sum"] / obs,
                "conf_avg": e["conf_sum"] / obs,
                "pos3d":    pos3d,
            })
        return {
            "objects":     objects,
            "cam_traj":    self.cam_traj,
            "point_cloud": self.point_cloud,
            "frame_w":     fw,
            "frame_h":     fh,
        }
