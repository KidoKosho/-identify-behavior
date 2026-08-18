import threading
import time

class LatestFrameStore:
    _frames = {}  # cam_id -> (frame, frame_count, timestamp)
    _lock = threading.Lock()

    @classmethod
    def set(cls, cam_id, frame, frame_count):
        with cls._lock:
            cls._frames[cam_id] = (frame, frame_count, time.time())

    @classmethod
    def get(cls, cam_id):
        with cls._lock:
            return cls._frames.get(cam_id)