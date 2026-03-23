"""
Semantic SLAM module.

Builds a sparse 3-D point cloud from SIFT features using keyframe-based
triangulation with reprojection-error filtering.  Also maintains a semantic
map that verifies detected objects by requiring consistent multi-frame
observations and sufficient local texture.
"""

import cv2
import numpy as np


class SemanticSLAM:
    """
    Semantic SLAM with SIFT features, keyframe-based triangulation,
    and reprojection-error filtering for high-quality 3D point clouds.
    """

    _KEYFRAME_MIN_MATCHES = 40
    _KEYFRAME_MIN_MOVEMENT = 0.005   # min median pixel displacement (fraction of frame size)

    def __init__(self, min_observations: int = 3, min_features: int = 2):
        self.sift = cv2.SIFT_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.prev_gray = None
        self.prev_kpts = None
        self.prev_desc = None
        self.camera_matrix = None
        self.pose = np.eye(4)
        self.semantic_map: dict = {}
        self.min_observations = min_observations
        self.min_features = min_features
        self.verified_ids: set = set()
        self.frame_count = 0
        # 3-D data
        self.cam_traj: list = [np.zeros(3)]
        self.prev_centers: dict = {}
        self._frame_size: tuple = (1920, 1080)
        # 3-D point cloud: list of (x, y, z, cls_label)
        self.point_cloud: list = []
        self._max_cloud_pts: int = 15000
        # Keyframe state
        self._kf_gray = None
        self._kf_kpts = None
        self._kf_desc = None
        self._kf_pose = np.eye(4)
        self._frames_since_kf = 0

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _init_camera(self, w: int, h: int) -> None:
        f = float(max(w, h))
        self.camera_matrix = np.array(
            [[f, 0, w / 2.0],
             [0, f, h / 2.0],
             [0, 0, 1.0]], dtype=np.float64)

    def _match_features(self, desc1, desc2, ratio: float = 0.75) -> list:
        """Lowe's ratio test for robust SIFT matching."""
        if desc1 is None or desc2 is None:
            return []
        raw = self.matcher.knnMatch(desc1, desc2, k=2)
        return [m for pair in raw if len(pair) == 2
                for m, n in [pair] if m.distance < ratio * n.distance]

    def _triangulate_and_filter(
        self, pts1: np.ndarray, pts2: np.ndarray,
        R: np.ndarray, t: np.ndarray,
        tracked_detections: list,
    ) -> tuple:
        """
        Triangulate a set of point correspondences and filter by:
          - positive depth
          - reprojection error < 3 px
        Returns (new_points, R_world, t_world).
        """
        K = self.camera_matrix
        P1 = K @ np.eye(3, 4)
        P2 = K @ np.hstack([R, t])

        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pose_inv = np.linalg.inv(self.pose)
        R_w = pose_inv[:3, :3]
        t_w = pose_inv[:3, 3]

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

    # ──────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray, tracked_detections: list) -> set:
        """
        Process one video frame.

        Args:
            frame: BGR image (numpy array).
            tracked_detections: list of (x1, y1, x2, y2, cls_id, conf, track_id).
        Returns:
            Set of track_ids whose position is verified.
        """
        h, w = frame.shape[:2]
        if self.camera_matrix is None:
            self._init_camera(w, h)
        self._frame_size = (w, h)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kpts, desc = self.sift.detectAndCompute(gray, None)

        # ── Sequential-frame camera-motion estimation ─────────────────
        if (self.prev_desc is not None and desc is not None
                and kpts is not None and len(kpts) >= 8):
            good = self._match_features(self.prev_desc, desc)
            if len(good) >= 8:
                pts1 = np.float32([self.prev_kpts[m.queryIdx].pt for m in good])
                pts2 = np.float32([kpts[m.trainIdx].pt for m in good])
                E, mask = cv2.findEssentialMat(
                    pts1, pts2, self.camera_matrix,
                    method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E is not None:
                    _, R, t, pose_mask = cv2.recoverPose(
                        E, pts1, pts2, self.camera_matrix, mask=mask)
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = t.flatten()
                    self.pose = T @ self.pose

                    inlier_idx = pose_mask.ravel() > 0
                    pts1_in, pts2_in = pts1[inlier_idx], pts2[inlier_idx]
                    if len(pts1_in) >= 8:
                        new_pts, R_w, t_w = self._triangulate_and_filter(
                            pts1_in, pts2_in, R, t, tracked_detections)
                        self.point_cloud.extend(new_pts)
                        self._cap_cloud()

                    # Triangulate object centers for pos3d
                    pose_inv = np.linalg.inv(self.pose)
                    R_w = pose_inv[:3, :3]
                    t_w = pose_inv[:3, 3]
                    K = self.camera_matrix
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
                        e = self.semantic_map.get(tid)
                        if e is not None:
                            if e["pos3d_sum"] is None:
                                e["pos3d_sum"] = pw.copy()
                            else:
                                e["pos3d_sum"] += pw
                            e["pos3d_n"] += 1

                    self.cam_traj.append(t_w.copy())

        # ── Keyframe-based triangulation (wider baseline) ─────────────
        self._frames_since_kf += 1
        if (self._kf_desc is not None and desc is not None
                and self._frames_since_kf >= 5):
            kf_good = self._match_features(self._kf_desc, desc)
            if len(kf_good) >= self._KEYFRAME_MIN_MATCHES:
                kf_pts1 = np.float32([self._kf_kpts[m.queryIdx].pt for m in kf_good])
                kf_pts2 = np.float32([kpts[m.trainIdx].pt for m in kf_good])
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
                                kf_pts1[inl], kf_pts2[inl], R_kf, t_kf, tracked_detections)
                            self.point_cloud.extend(new_kf_pts)
                            self._cap_cloud()
                    # Promote current frame to keyframe
                    self._kf_gray = gray.copy()
                    self._kf_kpts = kpts
                    self._kf_desc = desc.copy() if desc is not None else None
                    self._kf_pose = self.pose.copy()
                    self._frames_since_kf = 0

        # Init first keyframe
        if self._kf_desc is None and desc is not None:
            self._kf_gray = gray.copy()
            self._kf_kpts = kpts
            self._kf_desc = desc.copy()
            self._kf_pose = self.pose.copy()
            self._frames_since_kf = 0

        # ── Update semantic map ───────────────────────────────────────
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
                e["obs"] += 1
                e["kpts_sum"] += n_kpts
                e["cx_sum"] += cx
                e["cy_sum"] += cy
                e["conf_sum"] += conf
                if e["cls"] != cls_id:
                    e["consistent"] = False

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
        Keys: objects, cam_traj, point_cloud, frame_w, frame_h.
        """
        fw, fh = self._frame_size
        objects = []
        for tid, e in self.semantic_map.items():
            obs = max(e["obs"], 1)
            pos3d = None
            if e["pos3d_n"] > 0 and e["pos3d_sum"] is not None:
                pos3d = e["pos3d_sum"] / e["pos3d_n"]
            objects.append({
                "track_id": tid, "cls": e["cls"],
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
            "point_cloud": self.point_cloud,
            "frame_w": fw,
            "frame_h": fh,
        }
