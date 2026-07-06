import numpy as np
import cv2 as cv
from typing import Tuple, List, Optional, Dict, Any
import math
import copy
from collections import deque

from sources.masker import DynamicMasker
from sources.logger import SLAMLogger

class Frame():
    def __init__(self) -> None:
        self.pixels = None
        self.kps = None
        self.des = None
        self.E = None
        self.pose = dict()
        self.pose['R'] = np.eye(3)
        self.pose['t'] = np.zeros((3, 1))
    
    def copy(self, that_frame, pixels):
        self.pixels = pixels.copy()
        self.kps = that_frame.kps
        self.des = that_frame.des
    
    def __str__(self) -> str:
        return f"Data: {len(self.pixels)}, kps: {self.kps}, des: {self.des}, E: {self.E}, pose: {self.pose}"

class Vision():
    def __init__(self, video_dim: Tuple[int, int], _focal: float) -> None:
        self.orb = cv.ORB_create()
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.feats = None
        self.focal = _focal
        self.dist_coeffs = np.zeros(5)
        self.dist_coeffs[0] = -0.3
        self.cx = video_dim[0] // 2
        self.cy = video_dim[1] // 2
        self.K = np.array([[self.focal, 0, self.cx],
                           [0, self.focal, self.cy],
                           [0, 0, 1]])
        print("Camera intrinsics matrix:")
        print(self.K)
        self.current_frame = Frame()
        self.last_frame = Frame()
        self.matches = None
        self.camera_poses = []
        self.T_total = np.zeros((3, 1))
        self.R_total = np.eye(3)
        self.masker = DynamicMasker(
            model_path='yolov8n-seg.pt',
            input_size=(480, 480),
            conf_threshold=0.25
        )
        self.masker.start()
        print("YOLO Masker initialized")
        self.logger = SLAMLogger(log_dir="logs")
        self.logger.open()
        print("Logger initialized")

        self.stationary_frame_count = 0
        self.is_stationary = False

        self.PIXEL_MOVE_THRESHOLD = 2.5
        self.PIXEL_STOP_THRESHOLD = 1.3
        self.STATIONARY_FRAMES_REQUIRED = 5

        self.motion_history = deque(maxlen=5)

        self.last_T_total = np.zeros((3, 1))

        self.points_total = 0
        self.points_filtered_by_mask = 0
        self.current_motion_magnitude = 0.0
        self.current_inlier_ratio = 0.0
        self.current_mask_coverage = 0.0

        self.total_matches_all_frames = 0
        self.filtered_by_mask_all_frames = 0

        self.frame_id = 0
    
    def get_camera_poses(self):
        return self.camera_poses
    
    def camera_pose_to_opengl(self, T_total, R_total):
        pose = dict()
        pose['R'] = R_total
        pose['t'] = T_total
        corrected_pose = copy.deepcopy(pose)
        corrected_pose['t'][1] = 0
        corrected_pose['t'][2] *= -1
        return corrected_pose

    def get_camera_pose(self, matches: List[Tuple[Tuple[float, float], Tuple[float, float]]], 
                       frame_id: int = 0, static_mask: Optional[np.ndarray] = None):
        assert matches is not None, "No matches given"
        
        c1 = [pt1 for pt1, pt2 in matches]
        c2 = [pt2 for pt1, pt2 in matches]
        c1 = np.array(c1)
        c2 = np.array(c2)
        pp = (self.cx, self.cy)

        pixel_displacements = np.sqrt(np.sum((c1 - c2) ** 2, axis=1))
        avg_pixel_motion = np.mean(pixel_displacements) if len(pixel_displacements) > 0 else 0
        self.motion_history.append(avg_pixel_motion)
        smoothed_motion = np.mean(self.motion_history)

        E, mask = cv.findEssentialMat(c1, c2, self.focal, pp, cv.RANSAC, 0.999, 1.0)
        inlier_ratio = np.sum(mask) / len(mask) if len(mask) > 0 else 0
        self.current_inlier_ratio = inlier_ratio
        
        _, R, t, _ = cv.recoverPose(E, c1, c2, self.K, pp)
        motion_magnitude = np.linalg.norm(t)
        self.current_motion_magnitude = avg_pixel_motion

        if not self.is_stationary:
            if smoothed_motion < self.PIXEL_STOP_THRESHOLD:
                self.stationary_frame_count += 1
                if self.stationary_frame_count >= self.STATIONARY_FRAMES_REQUIRED:
                    self.is_stationary = True
                    print(f"[ZUPT] → STOPPED (pixel_motion: {smoothed_motion:.2f})")
            else:
                self.stationary_frame_count = 0
        else:
            if smoothed_motion > self.PIXEL_MOVE_THRESHOLD:
                self.is_stationary = False
                self.stationary_frame_count = 0
                print(f"[ZUPT] → MOVING (pixel_motion: {smoothed_motion:.2f})")

        pose = {'R': R, 't': np.zeros((3, 1)) if self.is_stationary else t}
        
        self.current_frame.E = E
        self.current_frame.pose = pose
        self.camera_poses.append(self.camera_pose_to_opengl(self.T_total, self.R_total))
        
        if not self.is_stationary:
            self.T_total += self.R_total @ pose['t']
            self.R_total = self.R_total @ pose['R']
            self.last_T_total = self.T_total.copy()

        mask_coverage = 0.0
        if static_mask is not None:
            total_px = static_mask.size
            masked_px = np.sum(static_mask == 0)
            mask_coverage = (masked_px / total_px) * 100 if total_px > 0 else 0
        self.current_mask_coverage = mask_coverage
        
        log_data = {
            'frame_id': frame_id,
            'total_matches': len(matches),
            'filtered_by_mask': getattr(self, 'points_filtered_by_mask', 0),
            'used_for_slam': len(matches) - getattr(self, 'points_filtered_by_mask', 0),
            'motion_magnitude': f"{smoothed_motion:.6f}",  # Теперь пиксели
            'status': 'STOPPED' if self.is_stationary else 'MOVING',
            'stationary_frames': self.stationary_frame_count,
            'pos_x': f"{self.T_total[0, 0]:.4f}",
            'pos_y': f"{self.T_total[1, 0]:.4f}",
            'pos_z': f"{self.T_total[2, 0]:.4f}",
            'mask_coverage_percent': f"{mask_coverage:.2f}",
            'inlier_ratio': f"{inlier_ratio:.4f}"
        }
        self.logger.log_frame(log_data)
    
    def get_pose_cumulation(self):
        return self.R_total, self.T_total

    def distance_between_points(self, pt1: float, pt2: float):
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

    def find_matching_points(self, current_frame: Frame):
        assert current_frame.pixels is not None, "No frame passed"

        static_mask = self.masker.get_mask_for_frame(current_frame.pixels.shape)

        frame_for_detection = current_frame.pixels.copy()

        if static_mask is not None:
            frame_for_detection[static_mask == 0] = 128
        
        match = np.mean(frame_for_detection, axis=2).astype(np.uint8)
        feats = cv.goodFeaturesToTrack(match, maxCorners=3000, qualityLevel=0.01, minDistance=7)
        
        if feats is None:
            return None
            
        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats]
        kps, des = self.orb.compute(frame_for_detection, kps)
        
        self.feats = feats
        self.current_frame.kps = kps
        self.current_frame.des = des
        
        if self.last_frame.kps is None or self.last_frame.des is None:
            return None

        frame_total = 0
        frame_filtered = 0
        
        self.matches = []
        
        for m in self.matcher.match(des, self.last_frame.des):
            kp1 = self.current_frame.kps[m.queryIdx].pt
            kp2 = self.last_frame.kps[m.trainIdx].pt
            
            frame_total += 1
            
            if self.distance_between_points(kp1, kp2) > 25:
                continue
            if kp1 != kp2:
                if static_mask is not None:
                    x, y = int(kp1[0]), int(kp1[1])
                    h, w = static_mask.shape
                    if 0 <= y < h and 0 <= x < w:
                        if static_mask[y, x] == 0:
                            frame_filtered += 1
                            continue
                self.matches.append((kp1, kp2))

        self.total_matches_all_frames += frame_total
        self.filtered_by_mask_all_frames += frame_filtered

        self.points_total = frame_total
        self.points_filtered_by_mask = frame_filtered
                
        if len(self.matches) == 0:
            return None
        return self.matches

    def view_interest_points(self, frame: Frame, matches: List[Tuple[Tuple[float, float], Tuple[float, float]]]):
        assert matches != None, "No matches passed"
        assert frame.pixels is not None, "No frame passed"

        for i, (pt1, pt2) in enumerate(matches):
            assert pt1 != pt2, "Points are the same"
            cv.circle(frame.pixels, (int(pt1[0]), int(pt1[1])), color=(0, 255, 255), radius=4)
            cv.circle(frame.pixels, (int(pt2[0]), int(pt2[1])), color=(255, 0, 255), radius=3)
            cv.line(frame.pixels, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color=(38, 207, 63), thickness=1)
        return frame.pixels

    def apply_mask_visualization(self, frame: np.ndarray, mode: str = "black_fill") -> np.ndarray:
        static_mask = self.masker.get_mask_for_frame(frame.shape)
        if static_mask is None:
            return frame
            
        result = frame.copy()
        
        if mode == "black_fill":
            result[static_mask == 0] = [0, 0, 0]
            cv.putText(result, "MASKED: Person/Car (BLACK)", (10, 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif mode == "red_overlay":
            overlay = frame.copy()
            overlay[static_mask == 0] = [0, 0, 255]
            result = cv.addWeighted(overlay, 0.4, frame, 0.6, 0)
            cv.putText(result, "MASKED: Person/Car (RED)", (10, 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result

    def debug_show_mask(self, frame: np.ndarray, window_name: str = "Mask Debug"):
        static_mask = self.masker.get_mask_for_frame(frame.shape)
        
        if static_mask is None:
            debug_img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            cv.putText(debug_img, "NO MASK YET", (50, 100), 
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv.imshow(window_name, debug_img)
            return
            
        mask_bgr = cv.cvtColor(static_mask, cv.COLOR_GRAY2BGR)
        
        total_pixels = static_mask.size
        masked_pixels = np.sum(static_mask == 0)
        mask_percent = (masked_pixels / total_pixels) * 100
        
        info_text = [
            f"Masked: {masked_pixels}/{total_pixels} px",
            f"Coverage: {mask_percent:.1f}%",
            f"Classes: person(0), car(2)"
        ]
        
        for i, text in enumerate(info_text):
            cv.putText(mask_bgr, text, (10, 30 + i * 25), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv.imshow(window_name, mask_bgr)
        
    def print_mask_statistics(self):
        print("\n" + "="*60)
        print("📊 YOLO MASK STATISTICS")
        print("="*60)
        print(f"Total matches found:     {self.total_matches_all_frames}")
        print(f"Filtered by mask:        {self.filtered_by_mask_all_frames}")
        
        if self.total_matches_all_frames > 0:
            percent = (self.filtered_by_mask_all_frames / self.total_matches_all_frames) * 100
            print(f"Filter rate:             {percent:.2f}%")
        else:
            print(f"Filter rate:             0%")
            
        print(f"Used for SLAM:           {self.total_matches_all_frames - self.filtered_by_mask_all_frames}")
        print("="*60)
        
        if self.filtered_by_mask_all_frames > 0:
            print("Mask is WORKING - dynamic objects were ignored!")
        else:
            print("Mask might not be working - no points were filtered")
        print("="*60 + "\n")

    def close_logger(self):
        summary = {
            'total_frames': self.total_matches_all_frames,
            'total_filtered': self.filtered_by_mask_all_frames,
            'filter_rate': f"{(self.filtered_by_mask_all_frames / max(self.total_matches_all_frames, 1)) * 100:.2f}%",
            'final_position': f"({self.T_total[0,0]:.4f}, {self.T_total[1,0]:.4f}, {self.T_total[2,0]:.4f})"
        }
        self.logger.log_summary(summary)
        self.logger.close()
        

class Slam():
    def __init__(self, width: int, height: int) -> None:
        self.vision = Vision(video_dim=(width, height), _focal=811.27)
        self._projection_matrix = None
        self._past_projection_matrix = None
        self.E_buffer = None
        self.pose_buffer = None
        self.points_centroid = None
        self.points3Dcumulative = []
    
    @property
    def projection_matrix(self):
        return self._projection_matrix
    
    @property
    def past_projection_matrix(self):
        return self._past_projection_matrix
    
    def get_camera_poses(self):
        return self.vision.get_camera_poses()

    def update_frame_pixels(self, current_frame_pixels: np.ndarray, last_frame_pixels: np.ndarray):
        assert current_frame_pixels is not None, "No frame passed"
        assert last_frame_pixels is not None, "No last frame passed"
        assert (current_frame_pixels==last_frame_pixels).all() == False, "Frames are the same"
        if last_frame_pixels is not None:
            self.vision.last_frame.copy(that_frame=self.vision.current_frame, pixels=last_frame_pixels)
        self.vision.current_frame.pixels = current_frame_pixels.copy()
    
    def get_vision_matches(self, render_frame):
        assert render_frame is not None, "No frame for rendering"
        assert self.vision.current_frame.pixels is not None, "No frame passed"
        assert self.vision.last_frame.pixels is not None, "No last frame"
        assert (self.vision.current_frame.pixels==self.vision.last_frame.pixels).all() == False, "Frames are the same"
        matches = self.vision.find_matching_points(self.vision.current_frame)
        if matches is not None:
            render_frame = self.vision.view_interest_points(self.vision.current_frame, matches)
            self.vision.get_camera_pose(matches)
            return matches, render_frame
        print("No matches found")
        return None, render_frame

    def hand_rule_change(self, points3D):
        assert points3D is not None, "points3D None"
        assert points3D.shape[0] > 4, "Points4D not 4xN"
        points3D[:, 1] *= -1
        points3D[:, 2] *= -1
        return points3D
    
    def transform_points_3D_openGL(self, points3D):
        assert points3D is not None, "points3D None"
        assert points3D.shape[0] == 3, "Points3D not 3D"
        R_total, t_total = self.vision.get_pose_cumulation()
        pose_corrected = self.vision.camera_pose_to_opengl(t_total, R_total)
        return np.dot(points3D.T, -pose_corrected['R']) + pose_corrected['t'].T

    def project_points(self, points3D):
        inv = np.linalg.inv(self.vision.K)
        points3D_projected = np.dot(inv, points3D)
        return points3D_projected
    
    def triangulate(self, matches: List[Tuple[Tuple[float, float], Tuple[float, float]]]):
        assert matches != None, "matches is None"
        assert len(matches) > 0, "No matches passed"
        assert self.vision.current_frame.E is not None, "current essential matrix is None"
        assert self.vision.current_frame.pose is not None, "current pose is None"
        if self.vision.last_frame.E is None:
            self.vision.last_frame.E = self.vision.current_frame.E
            self.vision.last_frame.pose = self.vision.current_frame.pose
        self._projection_matrix = np.hstack((self.vision.current_frame.pose['R'],
                                       self.vision.current_frame.pose['t']))
        self._past_projection_matrix = np.hstack((self.vision.last_frame.pose['R'],
                                            self.vision.last_frame.pose['t']))
        projPoints1 = []
        projPoints2 = []
        for kp1, kp2 in matches:
            projPoints1.append([kp1[0], kp1[1]])
            projPoints2.append([kp2[0], kp2[1]])
        projPoints1 = np.array(projPoints1).T
        projPoints2 = np.array(projPoints2).T
        K = self.vision.K
        projMat1 = K @ cv.hconcat([self.vision.current_frame.pose['R'], self.vision.current_frame.pose['t']])
        projMat2 = K @ cv.hconcat([self.vision.last_frame.pose['R'], self.vision.last_frame.pose['t']])
        
        # points1u = cv.undistortPoints(projPoints1, K, 1, None, K)
        # points2u = cv.undistortPoints(projPoints2, K, 1, None, K)

        points1u = cv.undistortPoints(projPoints1, self.vision.K, self.vision.dist_coeffs, None, self.vision.K)
        points2u = cv.undistortPoints(projPoints2, self.vision.K, self.vision.dist_coeffs, None, self.vision.K)
        
        points4D = cv.triangulatePoints(projMat1, projMat2, points1u, points2u)

        points3D = (points4D[:3] / points4D[3])
        goods = np.abs(points4D[3, :]) > 0.005
        goods &= points3D[1, :] > 0
        goods &= points3D[2, :] > 0
        goods &= points3D[2, :] < 500
        goods &= np.abs(points3D[0, :]) < 500
        if len(goods[goods == True]) < 25:
            print("error: not enough points")
            return None
        points3D = points3D[:, goods]
        points3D = self.transform_points_3D_openGL(points3D)
        self.points_centroid = sum([v for v in points3D]) / len(points3D)
        print("SLAM: points centroid: ", self.points_centroid)
        print(f"SLAM: estimate position: {self.vision.T_total[0]}, {self.vision.T_total[1]}, {self.vision.T_total[2]}")
        self.vision.last_frame.E = self.vision.current_frame.E
        self.vision.last_frame.pose = self.vision.current_frame.pose
        point_info = (points3D, self.points_centroid)
        self.points3Dcumulative.append(point_info)
        return self.points3Dcumulative