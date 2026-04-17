#!/usr/bin python3
import csv
import os
from datetime import datetime
from typing import Optional, Dict, Any

class SLAMLogger:
    """
    Логгер для записи статистики SLAM в CSV файл.
    """
    
    def __init__(self, log_dir: str = "logs", filename: Optional[str] = None):
        """
        Args:
            log_dir: Папка для сохранения логов
            filename: Имя файла (если None, создается автоматически)
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"slam_log_{timestamp}.csv"
        
        self.filepath = os.path.join(log_dir, filename)
        self.file = None
        self.writer = None
        self.frame_count = 0
        
        # Заголовки CSV — ДОБАВЛЕНЫ yaw_angle и rotation_score
        self.headers = [
            'timestamp',
            'frame_id',
            'total_matches',
            'filtered_by_mask',
            'used_for_slam',
            'motion_magnitude',
            'rotation_score',      # <-- НОВОЕ
            'status',              # MOVING / STOPPED / ROTATING
            'stationary_frames',
            'pos_x',
            'pos_y',
            'pos_z',
            'yaw_angle',           # <-- НОВОЕ
            'mask_coverage_percent',
            'inlier_ratio'
        ]
        
    def open(self):
        """Открывает файл для записи"""
        self.file = open(self.filepath, 'w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=self.headers)
        self.writer.writeheader()
        print(f"📝 Logger started: {self.filepath}")
        
    def close(self):
        """Закрывает файл и выводит статистику"""
        if self.file:
            self.file.close()
            print(f"✅ Logger closed: {self.filepath}")
            print(f"📊 Total frames logged: {self.frame_count}")
            
    def log_frame(self, data: Dict[str, Any]):
        """
        Записывает данные кадра в лог.
        
        Args:
            data: Словарь с данными (должен содержать ключи из self.headers)
        """
        if self.writer is None:
            return
            
        # Добавляем timestamp и frame_id если нет
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        if 'frame_id' not in data:
            self.frame_count += 1
            data['frame_id'] = self.frame_count
        else:
            self.frame_count = max(self.frame_count, data['frame_id'])
        
        # Записываем строку
        self.writer.writerow(data)
        self.file.flush()  # Сразу записываем на диск
        
    def log_summary(self, summary_data: Dict[str, Any]):
        """
        Записывает итоговую статистику в конец файла.
        """
        if self.file:
            self.file.write("\n")
            self.file.write("# SUMMARY\n")
            for key, value in summary_data.items():
                self.file.write(f"# {key}: {value}\n")
            self.file.flush()