import time
from collections import deque
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple
from app.utils.helpers import get_centroid, compute_iou

class ObjectTrack:
    def __init__(self, track_id, bbox, label="person"):
        self.track_id = track_id
        self.label = label
        self.history = deque(maxlen=15)
        self.missed_frames = 0
        self.update(bbox)
        
    def update(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        self.history.append({
            "bbox": bbox, "cx": cx, "cy": cy, "w": w, "h": h, "time": time.time()
        })
        self.missed_frames = 0
        
    def get_velocity(self):
        if len(self.history) < 2:
            return 0.0, 0.0
        first = self.history[0]
        last = self.history[-1]
        dt = max(0.01, last["time"] - first["time"])
        vx = (last["cx"] - first["cx"]) / dt
        vy = (last["cy"] - first["cy"]) / dt
        return vx, vy
        
    def get_motion_intensity(self):
        if len(self.history) < 2:
            return 0.0
        vx, vy = self.get_velocity()
        return np.sqrt(vx**2 + vy**2)
        
    def get_deceleration(self):
        # Calculate acceleration (change in speed)
        if len(self.history) < 3:
            return 0.0
        # Split history into two halves
        mid = len(self.history) // 2
        
        first = self.history[0]
        middle = self.history[mid]
        last = self.history[-1]
        
        dt1 = max(0.01, middle["time"] - first["time"])
        v1 = np.sqrt(((middle["cx"] - first["cx"]) / dt1)**2 + ((middle["cy"] - first["cy"]) / dt1)**2)
        
        dt2 = max(0.01, last["time"] - middle["time"])
        v2 = np.sqrt(((last["cx"] - middle["cx"]) / dt2)**2 + ((last["cy"] - middle["cy"]) / dt2)**2)
        
        # If speed drops, deceleration is positive
        return max(0.0, v1 - v2)

class CandidateFilter:
    """
    Tích hợp 3 Engines Rule-Based:
    1. Fight Engine (Kinetic)
    2. Accident Engine (Kinetic Vehicle Collision)
    3. Fire/Smoke Engine (MOG2 + HSV)
    """

    def __init__(self, fight_threshold=0.35, accident_threshold=0.55):
        self.fight_threshold = fight_threshold
        self.accident_threshold = accident_threshold
        
        # Tracking cho Người và Xe
        self.tracks = {}
        self.next_track_id = 0
        self.pair_history = {} # Dùng chung cho Fight và Accident
        
        # Tracking cho Fire/Smoke
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25, detectShadows=False)
        self.fire_tracks = {}
        self.next_fire_track_id = 0

    def _update_tracks(self, objects: List[Dict[str, Any]]):
        unmatched_tracks = set(self.tracks.keys())
        for p in objects:
            box = p["bbox"]
            best_id = -1
            best_iou = 0.2
            for tid in unmatched_tracks:
                trk = self.tracks[tid]
                if trk.label != p["label"]: continue
                iou = compute_iou(box, trk.history[-1]["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid
            if best_id != -1:
                self.tracks[best_id].update(box)
                p["track_id"] = best_id
                unmatched_tracks.remove(best_id)
            else:
                self.tracks[self.next_track_id] = ObjectTrack(self.next_track_id, box, p["label"])
                p["track_id"] = self.next_track_id
                self.next_track_id += 1
                
        for tid in list(unmatched_tracks):
            self.tracks[tid].missed_frames += 1
            if self.tracks[tid].missed_frames > 5:
                del self.tracks[tid]
                keys_to_del = [k for k in self.pair_history if k[0] == tid or k[1] == tid]
                for k in keys_to_del: del self.pair_history[k]

    def process_objects(self, detected_objects: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Nhận vào toàn bộ YOLO objects (person, car, etc).
        Cập nhật tracking và xuất ra danh sách ứng viên Fight và Accident.
        """
        persons = [obj for obj in detected_objects if obj["label"] == "person"]
        vehicles = [obj for obj in detected_objects if obj["label"] in ["car", "truck", "bus", "motorcycle"]]
        
        self._update_tracks(persons + vehicles)
        
        fight_candidates = self._filter_fights(persons)
        accident_candidates = self._filter_accidents(vehicles)
        fall_candidates = self._filter_falls(persons)
        
        return fight_candidates, accident_candidates, fall_candidates

    def filter_fight_candidates(self, detected_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Lọc ứng viên ẩu đả từ danh sách detected objects."""
        persons = [obj for obj in detected_objects if obj.get("label") == "person"]
        self._update_tracks(persons)
        return self._filter_fights(persons)

    def _filter_falls(self, persons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates = []
        for p in persons:
            tid = p.get("track_id")
            if tid is None or tid not in self.tracks: continue
            trk = self.tracks[tid]
            
            box = p["bbox"]
            w = max(1, box[2] - box[0])
            h = max(1, box[3] - box[1])
            ar = h / float(w)
            
            # FallScore = 0.25*posture + 0.25*vertical_motion + 0.20*aspect_change + 0.15*velocity + 0.15*temporal
            s_posture = 1.0 if ar < 1.0 else max(0.0, 1.0 - (ar - 1.0)) # ar < 1.0 là dáng nằm
            
            s_vert = 0.0
            s_aspect = 0.0
            if len(trk.history) >= 5:
                past = trk.history[-5]
                dy = trk.history[-1]["cy"] - past["cy"]
                if dy > 0: # Trọng tâm hạ xuống
                    s_vert = min(1.0, dy / max(1, past["h"] * 0.5))
                
                past_ar = past["h"] / max(1, float(past["w"]))
                if past_ar > 1.2 and ar < 1.0: # Từ đứng chuyển sang nằm
                    s_aspect = 1.0
                    
            intensity = trk.get_motion_intensity()
            s_vel = min(1.0, intensity / 150.0)
            
            s_temp = min(1.0, len(trk.history) / 10.0)
            
            score = 0.25 * s_posture + 0.25 * s_vert + 0.20 * s_aspect + 0.15 * s_vel + 0.15 * s_temp
            
            if score >= 0.80:
                candidates.append({
                    "type": "fall_candidate", "score": score, "bbox": box, "reason": "fall_kinetic"
                })
        return candidates

    def _filter_fights(self, persons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates = []
        if len(persons) < 2: return candidates
            
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                p1, p2 = persons[i], persons[j]
                tid1, tid2 = p1.get("track_id"), p2.get("track_id")
                if tid1 is None or tid2 is None: continue
                if tid1 > tid2: tid1, tid2 = tid2, tid1
                pair_key = (tid1, tid2)
                
                trk1 = self.tracks[tid1]
                trk2 = self.tracks[tid2]
                
                box1, box2 = p1["bbox"], p2["bbox"]
                w1, h1 = max(1, box1[2] - box1[0]), max(1, box1[3] - box1[1])
                w2, h2 = max(1, box2[2] - box2[0]), max(1, box2[3] - box2[1])
                
                dist = np.sqrt((trk1.history[-1]["cx"] - trk2.history[-1]["cx"])**2 + (trk1.history[-1]["cy"] - trk2.history[-1]["cy"])**2)
                avg_w = (w1 + w2) / 2.0
                s_pair = max(0.0, 1.0 - (dist / (2.2 * avg_w)))
                
                iou = compute_iou(box1, box2)
                s_contact = min(1.0, iou / 0.30) if iou > 0.0 else 0.0
                
                v1x, v1y = trk1.get_velocity()
                v2x, v2y = trk2.get_velocity()
                rel_motion = np.sqrt((v1x - v2x)**2 + (v1y - v2y)**2)
                s_rel = min(1.0, rel_motion / 150.0)
                
                s_int = min(1.0, (np.sqrt(v1x**2+v1y**2) + np.sqrt(v2x**2+v2y**2)) / 200.0)
                
                # Rule: Đứng cạnh / đi sát mà không có motion mạnh/contact thì bỏ
                if s_rel < 0.2 and s_int < 0.2:
                    s_pair = 0.0
                    s_contact = 0.0
                    
                if s_contact > 0.1 and s_int > 0.2:
                    self.pair_history[pair_key] = self.pair_history.get(pair_key, 0) + 1
                else:
                    self.pair_history[pair_key] = max(0, self.pair_history.get(pair_key, 0) - 1)
                
                s_temp = min(1.0, self.pair_history.get(pair_key, 0) / 10.0)
                
                score = 0.20 * s_pair + 0.20 * s_contact + 0.25 * s_rel + 0.20 * s_int + 0.15 * s_temp
                
                if score >= self.fight_threshold:
                    candidates.append({
                        "type": "fight_candidate", "score": score,
                        "bbox": [min(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[2], box2[2]), max(box1[3], box2[3])],
                        "reason": "pair_kinetic"
                    })
        return candidates

    def _filter_accidents(self, vehicles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates = []
        if len(vehicles) < 2: return candidates
        
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                v1, v2 = vehicles[i], vehicles[j]
                tid1, tid2 = v1.get("track_id"), v2.get("track_id")
                if tid1 is None or tid2 is None: continue
                if tid1 > tid2: tid1, tid2 = tid2, tid1
                
                trk1, trk2 = self.tracks[tid1], self.tracks[tid2]
                box1, box2 = v1["bbox"], v2["bbox"]
                
                # CollisionScore = 0.15*track_stability + 0.20*relative_velocity + 0.20*distance_closing + 0.20*collision_geometry + 0.15*velocity_change + 0.10*temporal_score
                
                s_track = min(1.0, (len(trk1.history) + len(trk2.history)) / 30.0)
                
                v1x, v1y = trk1.get_velocity()
                v2x, v2y = trk2.get_velocity()
                rel_vel = np.sqrt((v1x - v2x)**2 + (v1y - v2y)**2)
                s_rel = min(1.0, rel_vel / 200.0)
                
                # Distance closing
                dist_now = np.sqrt((trk1.history[-1]["cx"] - trk2.history[-1]["cx"])**2 + (trk1.history[-1]["cy"] - trk2.history[-1]["cy"])**2)
                dist_past = dist_now
                if len(trk1.history) > 2 and len(trk2.history) > 2:
                    dist_past = np.sqrt((trk1.history[0]["cx"] - trk2.history[0]["cx"])**2 + (trk1.history[0]["cy"] - trk2.history[0]["cy"])**2)
                s_dist_closing = 1.0 if dist_now < dist_past - 10 else 0.0
                
                iou = compute_iou(box1, box2)
                s_geom = min(1.0, iou / 0.5) if iou > 0.05 else 0.0
                
                dec1 = trk1.get_deceleration()
                dec2 = trk2.get_deceleration()
                s_dec = min(1.0, max(dec1, dec2) / 200.0)
                
                pair_key = (tid1, tid2)
                if s_geom > 0.2 or s_dec > 0.5:
                    self.pair_history[pair_key] = self.pair_history.get(pair_key, 0) + 1
                else:
                    self.pair_history[pair_key] = max(0, self.pair_history.get(pair_key, 0) - 1)
                
                s_temp = min(1.0, self.pair_history.get(pair_key, 0) / 10.0)
                
                score = 0.15 * s_track + 0.20 * s_rel + 0.20 * s_dist_closing + 0.20 * s_geom + 0.15 * s_dec + 0.10 * s_temp
                
                if score >= self.accident_threshold:
                    candidates.append({
                        "type": "vehicle_collision_candidate",
                        "score": score,
                        "bbox": [min(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[2], box2[2]), max(box1[3], box2[3])],
                        "reason": "vehicle_collision"
                    })
        return candidates

    def filter_fire_candidates(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Sử dụng MOG2 + Phổ màu HSV mở rộng + Dilation để phát hiện Lửa/Khói (Giai đoạn v1).
        Mở rộng vùng bao (bbox) và tăng độ nhạy để giai đoạn v2 (AI Classifier) dễ nhận diện hơn.
        """
        if frame is None or frame.size == 0:
            return []

        h_orig, w_orig = frame.shape[:2]
        # Resize để xử lý nhanh và giảm nhiễu cục bộ
        small = cv2.resize(frame, (480, 270))
        fg_mask = self.bg_subtractor.apply(small)
        
        # Lọc nhiễu chuyển động
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 1. Fire HSV Mở rộng: Đỏ/Cam/Vàng + Lõi sáng chói
        fire_color = ((h <= 42) | (h >= 155)) & (s >= 35) & (v >= 75)
        fire_core = (v >= 190) & (s <= 120)
        raw_fire_mask = (fire_color | fire_core).astype(np.uint8) * 255
        
        # Kết hợp chuyển động MOG2 + màu sắc lửa
        moving_fire = cv2.bitwise_and(raw_fire_mask, fg_mask)
        # Giữ lại cả ngọn lửa tĩnh có diện tích rõ ràng
        static_fire = cv2.morphologyEx(raw_fire_mask, cv2.MORPH_OPEN, kernel)
        combined_fire = cv2.bitwise_or(moving_fire, static_fire)
        
        # Dilation (Giãn nở) để gộp các đốm lửa rời rạc thành một khối to, liền mạch
        merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        active_fire = cv2.dilate(combined_fire, merge_kernel, iterations=2)
        
        # 2. Smoke Mask: Vùng khói mờ xám/trắng kèm chuyển động
        smoke_color = (s <= 65) & (v >= 55) & (v <= 225)
        smoke_mask = smoke_color.astype(np.uint8) * 255
        moving_smoke = cv2.bitwise_and(smoke_mask, fg_mask)
        moving_smoke = cv2.dilate(moving_smoke, merge_kernel, iterations=1)
        
        candidates = []
        scale_x = w_orig / 480.0
        scale_y = h_orig / 270.0
        
        # --- Phân tích đốm Lửa (Hạ ngưỡng diện tích từ 100 -> 35 để bắt nhạy hơn) ---
        contours, _ = cv2.findContours(active_fire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 35:
                x, y, w_b, h_b = cv2.boundingRect(cnt)
                
                # Mở rộng bounding box thêm 30% để bao quát ngữ cảnh xung quanh ngọn lửa cho v2
                pad_x = max(10, int(w_b * 0.35))
                pad_y = max(10, int(h_b * 0.35))
                
                bx1 = max(0, int((x - pad_x) * scale_x))
                by1 = max(0, int((y - pad_y) * scale_y))
                bx2 = min(w_orig, int((x + w_b + pad_x) * scale_x))
                by2 = min(h_orig, int((y + h_b + pad_y) * scale_y))
                
                # Đảm bảo kích thước tối thiểu để mô hình AI v2 có đủ thông tin
                if (bx2 - bx1) >= 40 and (by2 - by1) >= 40:
                    candidates.append({
                        "type": "fire",
                        "score": min(1.0, float(area) / 600.0 + 0.3),
                        "bbox": [bx1, by1, bx2, by2]
                    })
                
        # --- Phân tích đốm Khói ---
        contours_smoke, _ = cv2.findContours(moving_smoke, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_smoke:
            area = cv2.contourArea(cnt)
            if area > 80:
                x, y, w_b, h_b = cv2.boundingRect(cnt)
                if h_b / float(max(1, w_b)) > 0.4:
                    pad_x = max(15, int(w_b * 0.30))
                    pad_y = max(15, int(h_b * 0.30))
                    
                    bx1 = max(0, int((x - pad_x) * scale_x))
                    by1 = max(0, int((y - pad_y) * scale_y))
                    bx2 = min(w_orig, int((x + w_b + pad_x) * scale_x))
                    by2 = min(h_orig, int((y + h_b + pad_y) * scale_y))
                    
                    candidates.append({
                        "type": "smoke",
                        "score": min(1.0, float(area) / 1000.0 + 0.2),
                        "bbox": [bx1, by1, bx2, by2]
                    })
                
        return candidates
