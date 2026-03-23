"""IoU-based object tracker with persistent track IDs."""


class SimpleTracker:
    """Greedy IoU tracker: assigns persistent IDs to detections across frames."""

    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 15):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks: dict = {}  # track_id -> {"box", "cls", "lost"}

    @staticmethod
    def _iou(box1, box2) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections: list) -> list:
        """
        Args:
            detections: list of (x1, y1, x2, y2, cls_id, conf)
        Returns:
            list of (x1, y1, x2, y2, cls_id, conf, track_id)
        """
        if not detections:
            for tid in list(self.tracks):
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]
            return []

        track_ids = list(self.tracks.keys())
        matched_det: set = set()
        matched_trk: set = set()
        results = []

        # Greedy matching by highest IoU (same class only)
        pairs = []
        for di, det in enumerate(detections):
            for tid in track_ids:
                trk = self.tracks[tid]
                if det[4] == trk["cls"]:
                    iou = self._iou(det[:4], trk["box"])
                    if iou >= self.iou_threshold:
                        pairs.append((iou, di, tid))

        pairs.sort(key=lambda x: -x[0])
        for _, di, tid in pairs:
            if di in matched_det or tid in matched_trk:
                continue
            matched_det.add(di)
            matched_trk.add(tid)
            det = detections[di]
            self.tracks[tid]["box"] = list(det[:4])
            self.tracks[tid]["lost"] = 0
            results.append((*det, tid))

        # Unmatched detections → new tracks
        for di, det in enumerate(detections):
            if di not in matched_det:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"box": list(det[:4]), "cls": det[4], "lost": 0}
                results.append((*det, tid))

        # Unmatched tracks → increment lost counter
        for tid in track_ids:
            if tid not in matched_trk:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]

        return results
