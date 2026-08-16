import numpy as np
import config
from utils import compute_iou, get_centroid


class Detector:
    def __init__(self, models):
        self.models = models
        self.prev_positions = {}
        self.next_id = 0
        self.fight_history = {}
        self.tracking_records = []

    def _get_model_names(self, model):
        try:
            names = getattr(model, "names", None)
            if callable(names):
                names = names()
            if isinstance(names, dict):
                return {int(k): str(v).lower() for k, v in names.items()}
            if isinstance(names, list):
                return {idx: str(name).lower() for idx, name in enumerate(names)}
        except Exception:
            pass
        return {}

    def _matches_label(self, box, model, accepted_tokens, fallback_to_zero=False):
        cls_id = int(box.cls[0]) if hasattr(box, "cls") and len(box.cls) else None
        if cls_id is None:
            return False

        model_names = self._get_model_names(model)
        if model_names:
            label = model_names.get(cls_id, "")
            if any(token in label for token in accepted_tokens):
                return True
            return False

        if fallback_to_zero and cls_id == 0:
            return True
        return False

    def _deduplicate_boxes(self, boxes, iou_threshold=0.5):
        
        if not boxes:
            return []
        
        sorted_boxes = sorted(boxes, key=lambda x: x[1], reverse=True)
        deduplicated = []
        used_indices = set()
        
        for i, (box_i, conf_i) in enumerate(sorted_boxes):
            if i in used_indices:
                continue
            deduplicated.append((box_i, conf_i))

            
            for j in range(i + 1, len(sorted_boxes)):
                if j in used_indices:
                    continue
                box_j, conf_j = sorted_boxes[j]
                iou = compute_iou(box_i, box_j)
                if iou > iou_threshold:
                    used_indices.add(j)
        
        return deduplicated

    # detection khoi
    def detect_smoke(self, frame):
        if not config.ENABLE_FIRE_DETECTION or self.models.fire_model is None:
            return []

        results = self.models.fire_model(frame, imgsz=config.IMGSZ_SMOKE, conf=config.SMOKE_CONF)
        smoke_boxes = []
        h, w = frame.shape[:2]
        min_area = max(1, int(w * h * getattr(config, 'FIRE_MIN_AREA_RATIO', 0.006)))

        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < config.SMOKE_CONF:
                continue
            if not self._matches_label(box, self.models.fire_model,
                                      ["smoke", "fire", "flame", "burn", "smoke_fire"], fallback_to_zero=True):
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = max(0.0, (x2 - x1) * (y2 - y1))
            if area < min_area:
                continue
            smoke_boxes.append(([x1, y1, x2, y2], conf))
        return smoke_boxes

    # dection va cham
    def detect_accident(self, frame):
        if not config.ENABLE_ACCIDENT_DETECTION or self.models.accident_model is None:
            return []

        accident_conf = getattr(config, 'ACCIDENT_CONF', 0.72)
        results = self.models.accident_model(frame, conf=accident_conf)
        accident_boxes = []
        h, w = frame.shape[:2]
        min_area = max(1, int(w * h * config.ACCIDENT_MIN_AREA_RATIO))
        min_width = getattr(config, 'ACCIDENT_MIN_WIDTH', 80)
        min_height = getattr(config, 'ACCIDENT_MIN_HEIGHT', 50)

        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < accident_conf:
                continue
            if not self._matches_label(box, self.models.accident_model,
                                      ["accident", "crash", "collision", "damage"], fallback_to_zero=True):
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            area = width * height
            if area < min_area:
                continue
            if width < min_width or height < min_height:
                continue
            accident_boxes.append(([x1, y1, x2, y2], conf))
        return accident_boxes


    def detect_objects(self, frame):
        if self.models.coco_model is None:
            return [], [], []

        results = self.models.coco_model(frame, imgsz=config.IMGSZ_COCO, conf=config.CONF_THRESHOLD)
        persons, vehicles, trains = [], [], []
        h, w = frame.shape[:2]
        ground_y_min = int(h * config.GROUND_Y_MIN)
        ground_y_max = int(h * config.GROUND_Y_MAX)

        for box in results[0].boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            width = max(x2 - x1, 1.0)
            height = max(y2 - y1, 1.0)
            aspect_ratio = width / height
            # Chỉ lấy object khi confidence >= CONF_THRESHOLD (không relax xuống)
            if conf < config.CONF_THRESHOLD:
                continue
            if config.BOUND_GROUND and not (ground_y_min <= int(y2) <= ground_y_max):
                continue
            names = self._get_model_names(self.models.coco_model)
            if names:
                label = names.get(cls, "").lower()
                is_train_like = "train" in label or ("truck" in label and aspect_ratio >= config.TRAIN_MIN_ASPECT_RATIO)
                if config.ENABLE_PERSON_DETECTION and ("person" in label or "man" in label or "woman" in label):
                    persons.append(([x1, y1, x2, y2], conf))
                elif config.ENABLE_COLLISION_FIGHT_DETECTION and any(token in label for token in ["car", "bus", "truck", "motorcycle", "bicycle", "vehicle", "train"]):
                    if is_train_like or (aspect_ratio >= config.TRAIN_MIN_ASPECT_RATIO and width > 80):
                        trains.append(([x1, y1, x2, y2], conf))
                    else:
                        vehicles.append(([x1, y1, x2, y2], conf))
            else:
                is_train_like = cls == 6 or (cls == 7 and aspect_ratio >= config.TRAIN_MIN_ASPECT_RATIO)
                if cls == 0 and config.ENABLE_PERSON_DETECTION:
                    persons.append(([x1, y1, x2, y2], conf))
                elif cls in [2, 3, 5, 6, 7, 8, 9] and config.ENABLE_COLLISION_FIGHT_DETECTION:
                    if is_train_like or (aspect_ratio >= config.TRAIN_MIN_ASPECT_RATIO and width > 80):
                        trains.append(([x1, y1, x2, y2], conf))
                    else:
                        vehicles.append(([x1, y1, x2, y2], conf))
        
        # Loại bỏ các detection trùng lặp (ví dụ: 1 xe được detect thành 2)
        persons = self._deduplicate_boxes(persons, iou_threshold=0.3)
        vehicles = self._deduplicate_boxes(vehicles, iou_threshold=0.3)
        trains = self._deduplicate_boxes(trains, iou_threshold=0.3)
        
        return persons, vehicles, trains

    def detect_collisions(self, persons, vehicles, trains):
        if not config.ENABLE_COLLISION_FIGHT_DETECTION:
            return []
        """Va chạm thông thường (dùng IOU_COLLISION)"""
        p_boxes = [p[0] for p in persons]
        v_boxes = [v[0] for v in vehicles]
        t_boxes = [t[0] for t in trains]
        collisions = []

        for i in range(len(p_boxes)):
            for j in range(i+1, len(p_boxes)):
                iou = compute_iou(p_boxes[i], p_boxes[j])
                if iou > config.IOU_COLLISION:
                    collisions.append(("person-person", p_boxes[i], p_boxes[j], iou))

        for i in range(len(v_boxes)):
            for j in range(i+1, len(v_boxes)):
                iou = compute_iou(v_boxes[i], v_boxes[j])
                if iou > config.IOU_COLLISION:
                    collisions.append(("vehicle-vehicle", v_boxes[i], v_boxes[j], iou))
        # Vehicle-train (tàu hoả và ô tô / xe khác)
        for v in v_boxes:
            for t in t_boxes:
                iou = compute_iou(v, t)
                if iou > config.IOU_COLLISION:
                    collisions.append(("vehicle-train", v, t, iou))
        # Train-train
        for i in range(len(t_boxes)):
            for j in range(i+1, len(t_boxes)):
                iou = compute_iou(t_boxes[i], t_boxes[j])
                if iou > config.IOU_COLLISION:
                    collisions.append(("train-train", t_boxes[i], t_boxes[j], iou))
        # Person-vehicle
        for p in p_boxes:
            for v in v_boxes:
                iou = compute_iou(p, v)
                if iou > config.IOU_COLLISION:
                    collisions.append(("person-vehicle", p, v, iou))
        # Person-train
        for p in p_boxes:
            for t in t_boxes:
                iou = compute_iou(p, t)
                if iou > config.IOU_COLLISION:
                    collisions.append(("person-train", p, t, iou))
        # Vehicle-train
        for v in v_boxes:
            for t in t_boxes:
                iou = compute_iou(v, t)
                if iou > config.IOU_COLLISION:
                    collisions.append(("vehicle-train", v, t, iou))
        return collisions

    def detect_fight(self, persons, velocity_map):
        if not config.ENABLE_COLLISION_FIGHT_DETECTION:
            return []
        """
        Phát hiện đánh nhau dựa trên:
        - ít nhất 2 người
        - IoU > IOU_FIGHT
        - cả hai người đều có tốc độ đáng kể hoặc tối thiểu 1 người có tốc độ cao
        - không phải camera pan / người đi bộ đơn lẻ
        """
        if len(persons) < config.FIGHT_MIN_PERSONS:
            return []

        p_boxes = [p[0] for p in persons]
        fights = []
        for i in range(len(p_boxes)):
            for j in range(i+1, len(p_boxes)):
                iou = compute_iou(p_boxes[i], p_boxes[j])
                if iou <= config.IOU_FIGHT:
                    continue

                c1 = get_centroid(p_boxes[i])
                c2 = get_centroid(p_boxes[j])
                v1 = 0
                v2 = 0
                for tid, vel in velocity_map.items():
                    px, py = self.prev_positions.get(tid, (0, 0))
                    if abs(px - c1[0]) < 10 and abs(py - c1[1]) < 10:
                        v1 = vel
                    if abs(px - c2[0]) < 10 and abs(py - c2[1]) < 10:
                        v2 = vel

                max_vel = max(v1, v2)
                min_vel = min(v1, v2)
                if max_vel > config.FIGHT_VELOCITY_THRESHOLD and min_vel > 3:
                    fights.append((p_boxes[i], p_boxes[j], iou, max_vel))
        return fights

    def track_and_compute_velocity(self, boxes, frame_id=None):
        velocity_map = {}
        current_positions = {}
        frame_records = []
        assignment = {}

        candidates = []
        for box_idx, box in enumerate(boxes):
            cx, cy = get_centroid(box)
            for tid, (px, py) in self.prev_positions.items():
                dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                if dist <= config.TRACKING_MAX_DIST:
                    candidates.append((dist, tid, box_idx))

        used_prev_ids = set()
        used_box_indices = set()
        for dist, tid, box_idx in sorted(candidates, key=lambda item: (item[0], item[1], item[2])):
            if tid in used_prev_ids or box_idx in used_box_indices:
                continue
            assignment[box_idx] = tid
            used_prev_ids.add(tid)
            used_box_indices.add(box_idx)

        for box_idx, box in enumerate(boxes):
            cx, cy = get_centroid(box)
            track_id = assignment.get(box_idx)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1

            if track_id in self.prev_positions:
                px, py = self.prev_positions[track_id]
                dx = cx - px
                dy = cy - py
                vel = np.sqrt(dx ** 2 + dy ** 2)
            else:
                vel = 0.0

            velocity_map[track_id] = vel
            current_positions[track_id] = (cx, cy)
            frame_records.append({
                "frame_id": frame_id,
                "track_id": track_id,
                "x_center": float(cx),
                "y_center": float(cy),
                "bbox_width": float(box[2] - box[0]),
                "bbox_height": float(box[3] - box[1]),
            })

        self.prev_positions = current_positions
        self.tracking_records.extend(frame_records)
        if frame_id is not None:
            return velocity_map, frame_records
        return velocity_map