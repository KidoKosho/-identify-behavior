import os
import time
import cv2
import numpy as np
from datetime import datetime

class EventProcessor:
    """
    Xử lý các sự kiện đã được CONFIRMED từ TemporalBuffer.
    Thực hiện lưu snapshot và ghi log sự kiện.
    """
    
    def __init__(self, output_dir="outputs/snapshots", camera_id="cam_01"):
        self.output_dir = output_dir
        self.camera_id = camera_id
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            
    def process_event(self, event_type: str, confidence: float, frame: np.ndarray, bbox: list = None):
        """
        Ghi nhận sự kiện CONFIRMED và lưu snapshot.
        """
        # Format: fight_81_cam0_TIMESTAMP.jpg
        conf_int = int(confidence * 100)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19] # YYYYMMDD_HHMMSS_mmm
        
        filename = f"{event_type}_{conf_int}_{self.camera_id}_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        # Vẽ bounding box lên frame nếu có (copy để không ảnh hưởng luồng chính)
        save_frame = frame.copy()
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(save_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(save_frame, f"{event_type.upper()} {conf_int}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
        cv2.imwrite(filepath, save_frame)
        print(f"🚨 [EVENT CONFIRMED] {event_type.upper()} ({conf_int}%) - Saved snapshot to {filename}")
