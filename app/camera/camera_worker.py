# camera/camera_worker.py
import cv2
import threading
import time
from app.camera.latest_frame_store import LatestFrameStore

class CameraWorker:
    def __init__(self, cam_id, src=0):
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(src)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.running = True
        self.frame_count = 0

    def start(self):
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_count += 1
                LatestFrameStore.set(self.cam_id, frame, self.frame_count)
            else:
                time.sleep(0.01)