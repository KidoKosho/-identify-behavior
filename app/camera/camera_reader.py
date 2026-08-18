import cv2
import threading
import time
import numpy as np
from typing import Optional

class CameraReader:
    """
    Chịu trách nhiệm đọc luồng video/camera trong một thread riêng biệt.
    Tuyệt đối không chạy AI inference trong class này.
    Chỉ lưu lại frame mới nhất (latest_frame).
    """
    
    def __init__(self, source: str, camera_id: str = "cam_01"):
        self.source = source
        self.camera_id = camera_id
        self.cap = None
        
        self.latest_frame = None
        self.is_running = False
        self.lock = threading.Lock()
        self.thread = None
        
        # Thống kê ingestion
        self.frame_count = 0
        self.start_time = 0.0

    def start(self):
        if self.is_running:
            return
            
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self.source}")
            
        self.is_running = True
        self.start_time = time.time()
        
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        print(f"[{self.camera_id}] Camera thread started.")

    def _update_loop(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 30.0
        frame_delay = 1.0 / fps
        
        while self.is_running:
            t0 = time.time()
            ret, frame = self.cap.read()
            if not ret:
                print(f"[{self.camera_id}] End of stream or read error.")
                self.is_running = False
                break
                
            with self.lock:
                self.latest_frame = frame
                self.frame_count += 1
                
            # Sync to original video FPS
            elapsed = time.time() - t0
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Lấy frame mới nhất. Có thể trả về None nếu stream chưa bắt đầu
        hoặc không có frame mới.
        """
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def stop(self):
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
        print(f"[{self.camera_id}] Camera thread stopped. Processed {self.frame_count} frames.")
