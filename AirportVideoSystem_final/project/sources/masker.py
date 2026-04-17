#!/usr/bin/python3
import cv2
import numpy as np
from ultralytics import YOLO
import threading
from collections import deque
from typing import Optional, Tuple

class DynamicMasker:
    """
    Асинхронный маскировщик на базе YOLOv8-Seg.
    Маскирует person и car для исключения из SLAM.
    """
    
    # COCO class IDs: person=0, car=2
    CLASSES_TO_MASK = {0, 2}
    
    def __init__(self, model_path: str = 'yolov8n-seg.pt', 
                 input_size: Tuple[int, int] = (480, 480),
                 conf_threshold: float = 0.25):
        self.model = YOLO(model_path)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        
        # Кэш масок
        self._mask_cache = deque(maxlen=3)
        self._lock = threading.Lock()
        
        # Потоки
        self._running = False
        self._thread = None
        self._input_queue = deque(maxlen=1)
        
    def start(self):
        """Запускает фоновый поток обработки"""
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        print("✅ DynamicMasker started")
        
    def stop(self):
        """Останавливает фоновый поток"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("✅ DynamicMasker stopped")
        
    def submit_frame(self, frame: np.ndarray):
        """Отправляет кадр в очередь обработки"""
        if self._running and frame is not None:
            self._input_queue.append(frame.copy())
            
    def _process_loop(self):
        """Основной цикл обработки в отдельном потоке"""
        while self._running:
            if not self._input_queue:
                threading.Event().wait(0.01)
                continue
                
            frame = self._input_queue.popleft()
            mask = self._generate_mask(frame)
            
            with self._lock:
                self._mask_cache.append(mask)
                
    def _generate_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Генерирует бинарную маску: 255=статика(фон), 0=динамика(person/car)
        """
        h, w = frame.shape[:2]
        static_mask = np.ones((h, w), dtype=np.uint8) * 255
        
        try:
            results = self.model(frame, verbose=False, conf=self.conf_threshold, 
                                imgsz=self.input_size)
            
            if results and results[0].masks is not None and results[0].boxes is not None:
                boxes = results[0].boxes
                masks = results[0].masks
                
                for i, class_id in enumerate(boxes.cls.cpu().numpy().astype(int)):
                    if class_id in self.CLASSES_TO_MASK:
                        if i < len(masks.xy):
                            mask_poly = masks.xy[i].astype(np.int32)
                            cv2.fillPoly(static_mask, [mask_poly], 0)
        except Exception as e:
            print(f"[Masker Warning] {e}")
            
        return static_mask
    
    def get_mask_for_frame(self, frame_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
        """
        Возвращает последнюю доступную маску
        frame_shape: (height, width, channels)
        """
        with self._lock:
            if not self._mask_cache:
                return None
            mask = self._mask_cache[-1].copy()
            
        # Ресайз если размеры не совпадают
        if mask.shape != frame_shape[:2]:
            mask = cv2.resize(mask, (frame_shape[1], frame_shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
        return mask