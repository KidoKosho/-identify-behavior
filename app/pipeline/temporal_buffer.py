import time
from collections import deque
import numpy as np
from app.utils.helpers import compute_iou

class EventTrack:
    """Đại diện cho một đám cháy hoặc đám khói độc lập đang được theo dõi trên màn hình."""
    def __init__(self, track_id, init_bbox, track_type="fire"):
        self.track_id = track_id
        self.type = track_type # "fire" hoặc "smoke"
        self.history = deque(maxlen=15) # Lưu các thông tin (bbox, time, conf, area)
        self.persistence_count = 0
        self.last_update_time = time.time()
        
    def add_record(self, bbox, conf):
        w = max(1, bbox[2] - bbox[0])
        h = max(1, bbox[3] - bbox[1])
        area = w * h
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.history.append({
            "bbox": bbox,
            "conf": conf,
            "area": area,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "time": time.time()
        })
        self.persistence_count = min(15, self.persistence_count + 1)
        self.last_update_time = time.time()

class TemporalBuffer:
    """
    Quản lý bộ nhớ đệm (buffer) cho Violence và
    Hệ thống Theo vết Đa mục tiêu (Multi-Object Tracking) cho Fire/Smoke.
    """
    def __init__(self, violence_seq_len=16, window_size=5, confirm_threshold=3, cooldown_sec=10.0):
        self.violence_seq_len = violence_seq_len
        self.window_size = window_size
        self.confirm_threshold = confirm_threshold
        self.cooldown_sec = cooldown_sec

        # Buffer cho sequence model (Violence)
        self.frame_buffer = deque(maxlen=self.violence_seq_len)
        
        # Lịch sử phát hiện cho FIGHT
        self.event_history = {
            "fight": deque(maxlen=self.window_size)
        }
        
        self.last_event_time = {
            "fight": 0.0,
            "fire_smoke": 0.0
        }
        
        # Multi-Object Tracking cho Fire/Smoke
        self.tracks = []
        self.next_track_id = 1

    def add_frame(self, frame: np.ndarray):
        """Thêm frame vào buffer cho model Violence."""
        self.frame_buffer.append(frame)

    def get_violence_sequence(self):
        if len(self.frame_buffer) == self.violence_seq_len:
            return list(self.frame_buffer)
        return None

    def update_history(self, event_type: str, is_positive: bool) -> bool:
        """Sử dụng riêng cho Fight. Lửa/Khói đã có hàm track_and_score riêng."""
        if event_type not in self.event_history:
            return False
            
        self.event_history[event_type].append(1 if is_positive else 0)
        positive_count = sum(self.event_history[event_type])
        
        if positive_count >= self.confirm_threshold:
            now = time.time()
            if now - self.last_event_time[event_type] > self.cooldown_sec:
                self.last_event_time[event_type] = now
                return True
        return False
        
    def _cleanup_tracks(self, now):
        """Xoá các track đã mất dấu hơn 1 giây."""
        self.tracks = [t for t in self.tracks if now - t.last_update_time < 1.0]

    def track_and_score_fire_smoke(self, pred_class: str, conf: float, bbox: list) -> float:
        """
        Theo vết và tính toán điểm (FireScore/SmokeScore) phức hợp cho từng đối tượng độc lập.
        """
        now = time.time()
        self._cleanup_tracks(now)
        
        best_iou = 0.0
        best_track = None
        
        for t in self.tracks:
            if t.type == pred_class:
                last_bbox = t.history[-1]["bbox"]
                iou = compute_iou(bbox, last_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track = t
                    
        if best_iou > 0.05 and best_track is not None:
            best_track.add_record(bbox, conf)
            track = best_track
        else:
            track = EventTrack(self.next_track_id, bbox, track_type=pred_class)
            track.add_record(bbox, conf)
            self.tracks.append(track)
            self.next_track_id += 1
            
        return self._calculate_score(track)
        
    def calculate_fire_score(self, conf: float, bbox: list) -> float:
        """Helper tính điểm cháy cho 1 bbox và conf."""
        return self.track_and_score_fire_smoke("fire", conf, bbox)
        
    def _calculate_score(self, track: EventTrack) -> float:
        """Tính điểm dựa trên luật Heuristic ngặt nghèo."""
        current = track.history[-1]
        model_conf = current["conf"]
        
        # 1. Model Score (35%)
        s_model = min(1.0, model_conf) * 0.35
        
        # 2. Persistence Score (20% - Phải tồn tại qua nhiều frame)
        s_persist = min(1.0, track.persistence_count / 5.0) * 0.20
        
        # Mapping model classes
        is_fire = track.type in ["fire", "fire_only", "fire_smoke"]
        is_smoke = track.type in ["smoke", "smoke_only", "fire_smoke"]
        
        if is_fire:
            # FireScore = 0.35(model) + 0.20(spatial) + 0.20(persist) + 0.15(motion) + 0.10(smoke)
            
            # Spatial Consistency: Tâm lửa không dịch chuyển quá rộng so với kích thước
            s_spatial = 0.20
            if len(track.history) >= 2:
                prev = track.history[-2]
                dist = np.sqrt((current["cx"] - prev["cx"])**2 + (current["cy"] - prev["cy"])**2)
                if dist > current["w"] * 1.5: 
                    s_spatial = 0.05 # Bị trừ điểm nếu nhảy cóc
                    
            # Motion/Shape: Tỷ lệ H/W thường từ 0.5 đến 3.0
            ar = current["h"] / max(1, float(current["w"]))
            s_motion = 0.15 if 0.5 < ar < 3.0 else 0.05
            
            # Smoke Corroboration: Kiểm tra xem có track Khói nào ở gần đó không
            s_smoke = 0.0
            for t in self.tracks:
                if t.type in ["smoke", "smoke_only", "fire_smoke"]:
                    if compute_iou(current["bbox"], t.history[-1]["bbox"]) > 0.01:
                        s_smoke = 0.10
                        break
                        
            return s_model + s_spatial + s_persist + s_motion + s_smoke
            
        elif is_smoke:
            # SmokeScore = 0.35(model) + 0.25(expansion) + 0.20(persist) + 0.20(shape_consistency)
            
            # Expansion: Khói lan tỏa, diện tích to dần
            s_expand = 0.05
            if len(track.history) >= 3:
                first = track.history[0]
                if current["area"] > first["area"] * 1.15: # Nở ra > 15%
                    s_expand = 0.25
                elif current["area"] > first["area"]:
                    s_expand = 0.15
                    
            # Shape Consistency: Khói không nhảy cóc biến mất
            s_shape = 0.20
            if len(track.history) >= 2:
                prev = track.history[-2]
                iou = compute_iou(current["bbox"], prev["bbox"])
                if iou < 0.1: # Độ gối lên nhau quá thấp -> Báo động giả (gray blur)
                    s_shape = 0.05
                    
            return s_model + s_expand + s_persist + s_shape

    def check_fire_smoke_cooldown(self):
        """Bảo vệ không Spam log khi cháy liên tục."""
        now = time.time()
        if now - self.last_event_time["fire_smoke"] > self.cooldown_sec:
            self.last_event_time["fire_smoke"] = now
            return True
        return False
