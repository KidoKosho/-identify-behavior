import cv2
import os
from urllib.parse import urlparse
import config
from detector import Detector
from utils import draw_velocity, draw_collision_line, draw_smoke_circle, compute_iou, get_centroid, export_tracking_records


class Pipeline:
    def __init__(self, models, video_name="video", output_dir=None):
        self.detector = Detector(models)
        source = video_name
        parsed = urlparse(video_name)
        if parsed.scheme:
            source = parsed.path
        self.video_name = os.path.splitext(os.path.basename(source))[0] or "video"
        self.output_dir = output_dir or config.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.frame_count = 0
        self.fight_frame_buffer = {}
        self.saved_events = set()
        
        # Buffer để lưu smoke detections trong 5 frame liên tiếp
        self.smoke_buffer = []  # [(frame, smoke_boxes, smoke_conf), ...]
        self.smoke_buffer_max_frames = 5
        
        # Buffer để lưu accident detections
        self.accident_buffer = []  # [(frame, accident_boxes, accident_conf), ...]
        self.accident_buffer_max_frames = 5

    def _is_real_collision(self, v1, v2, iou):
       
        if v1 < 5 and v2 < 5:
            return False

        relative_velocity = abs(v1 - v2)
        max_velocity = max(v1, v2)

        # Hai xe chạy song song cùng tốc độ -> không phải va chạm
        if iou > 0.8 and relative_velocity < 5:
            return False

        # Nếu cả 2 đều đang chuyển động mạnh và có overlap -> va chạm thật
        if v1 > 10 and v2 > 10 and iou > 0.15:
            return True

        # Nếu 1 xe đứng, 1 xe lao vào mạnh -> chỉ cảnh báo khi overlap đủ sâu
        if (v1 < 5 or v2 < 5) and max_velocity > 35:
            return iou > 0.25

        # Nếu cả 2 đều đang di chuyển với tốc độ cao -> cần overlap đủ để xác nhận
        if max_velocity > config.VELOCITY_THRESHOLD:
            return iou > 0.25

        # Với vận tốc trung bình, chỉ chốt khi overlap rất rõ
        if max_velocity > 24 and iou > 0.4:
            return True

        return False

    def _process_smoke_buffer(self):
        """
        Xử lý smoke buffer: nếu có smoke trong nhiều frame liên tiếp,
        chỉ lưu 1 frame tốt nhất (confidence cao nhất).
        """
        if not self.smoke_buffer:
            return
        
        # Tìm frame có confidence cao nhất
        best_idx = max(range(len(self.smoke_buffer)), 
                       key=lambda i: self.smoke_buffer[i][2])
        best_frame, best_boxes, best_conf = self.smoke_buffer[best_idx]
        
        # Lưu frame tốt nhất
        self._save_event_image(best_frame, "khoi", best_conf)
        
        # Clear buffer
        self.smoke_buffer = []

    def _process_accident_buffer(self):
        """
        Xử lý accident buffer: lưu TẤT CẢ frame có confidence >= 0.55.
        """
        if not self.accident_buffer:
            return
        
        # Lưu tất cả frame có confidence >= 0.55 để capture toàn bộ sự kiện va chạm
        for frame, accident_boxes, conf in self.accident_buffer:
            if conf >= config.ACCIDENT_CONF:
                self._save_event_image(frame, "accident", conf)
        
        # Clear buffer
        self.accident_buffer = []

    def _draw_ground_bound(self, frame):
        """
        Vẽ vùng ground bound trên frame để visualize.
        """
        h, w = frame.shape[:2]
        ground_y_min = int(h * config.GROUND_Y_MIN)
        ground_y_max = int(h * config.GROUND_Y_MAX)
        
        # Vẽ hình chữ nhật vùng ground (màu xanh lá)
        cv2.rectangle(frame, (0, ground_y_min), (w, ground_y_max), (0, 255, 0), 2)
        cv2.putText(frame, f"GROUND: {config.GROUND_Y_MIN:.0%}-{config.GROUND_Y_MAX:.0%}", 
                    (10, ground_y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame

    def _is_falling_collision(self, box1, box2, prev_pos1, prev_pos2):
        """
        Phát hiện va chạm từ vật rơi từ trên xuống.
        Nếu vật di chuyển chủ yếu theo hướng dọc (Y), đó có thể là vật rơi.
        """
        c1 = get_centroid(box1)
        c2 = get_centroid(box2)
        
        if not prev_pos1 or not prev_pos2:
            return False
        
        # Tính displacement (sự dịch chuyển)
        dx1 = abs(c1[0] - prev_pos1[0])
        dy1 = abs(c1[1] - prev_pos1[1])
        
        dx2 = abs(c2[0] - prev_pos2[0])
        dy2 = abs(c2[1] - prev_pos2[1])
        
        # Nếu di chuyển chủ yếu theo Y (dy >> dx), có thể là vật rơi
        # Ngưỡng: dy > 3*dx (chủ yếu dọc)
        falling1 = dy1 > 3 * dx1 and dy1 > 5
        falling2 = dy2 > 3 * dx2 and dy2 > 5
        
        return falling1 or falling2

    def _save_event_image(self, frame, prefix, confidence, suffix="", frame_id=None):
        accuracy = max(0, min(99, int(confidence * 100)))
        postfix = f"_{frame_id}" if frame_id is not None else ""
        if suffix:
            filename = f"{prefix}_{accuracy}{postfix}_{suffix}_{self.video_name}.jpg"
        else:
            filename = f"{prefix}_{accuracy}{postfix}_{self.video_name}.jpg"
        filepath = os.path.join(self.output_dir, filename)

        # Chỉ ngăn duplicate trong cùng 1 frame. Nếu cùng loại sự kiện xảy ra ở frame mới,
        # vẫn lưu lại để người dùng có thể kiểm tra toàn bộ cảnh báo.
        key = (prefix, self.video_name, frame_id if frame_id is not None else accuracy, suffix)
        if key in self.saved_events:
            return
        self.saved_events.add(key)

        ok = cv2.imwrite(filepath, frame)
        print(f"💾 save_event: {filepath} -> {'OK' if ok else 'FAILED'}")
        if not ok:
            print(f"⚠️ Unable to save image to {filepath}")

    def export_tracking_data(self, output_path=None):
        if not config.TRACKING_EXPORT_ENABLED:
            return None
        if output_path is None:
            output_path = config.TRACKING_EXPORT_PATH
        export_tracking_records(
            self.detector.tracking_records,
            output_path,
            scale=getattr(config, "PIXEL_TO_METER", 1.0),
            homography=getattr(config, "HOMOGRAPHY_MATRIX", None),
            columns=["frame_id", "track_id", "x_center", "y_center", "bbox_width", "bbox_height"],
        )
        if config.TRACKING_EXPORT_JSON:
            import json
            with open(config.TRACKING_EXPORT_JSON, "w", encoding="utf-8") as f:
                json.dump(self.detector.tracking_records, f, ensure_ascii=False, indent=2)
        return output_path

    def process_frame(self, frame):
        self.frame_count += 1

        if config.PROCESS_EVERY_N_FRAMES > 1 and self.frame_count % config.PROCESS_EVERY_N_FRAMES != 0:
            return frame

        # 1. Chỉ chạy model người nếu cấu hình cho phép
        persons, vehicles, trains = self.detector.detect_objects(frame)

        # 2. Phát hiện khói/lửa - lưu vào buffer thay vì lưu ngay
        smoke_boxes = self.detector.detect_smoke(frame)
        smoking_persons = []
        
        if smoke_boxes:
            print(f"🔥 Smoke/Fire detections on this frame: {len(smoke_boxes)}")
            # Tìm confidence cao nhất
            max_conf = max([sconf for _, sconf in smoke_boxes])
            # Thêm vào buffer
            self.smoke_buffer.append((frame.copy(), smoke_boxes, max_conf))
            
            # Giới hạn kích thước buffer
            if len(self.smoke_buffer) >= self.smoke_buffer_max_frames:
                self._process_smoke_buffer()
        else:
            # Nếu không có smoke mà buffer có dữ liệu, xử lý buffer
            if self.smoke_buffer:
                self._process_smoke_buffer()

        # Xử lý smoking_persons cho visualization
        for sbox, sconf in smoke_boxes:
            best_iou = 0
            best_person = None
            if config.ENABLE_PERSON_DETECTION:
                for pbox, pconf in persons:
                    iou = compute_iou(sbox, pbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_person = pbox
            if best_person is not None and best_iou > 0.15:
                smoking_persons.append((best_person, sconf))

        # 3. Phát hiện tai nạn (accident): chạy theo model, không cần check lại bằng va chạm/IoU
        accident_boxes = self.detector.detect_accident(frame)
        
        if accident_boxes:
            print(f"⚠️ Accident detected: {len(accident_boxes)} box(es)")
            max_conf = max([aconf for _, aconf in accident_boxes])
            # Lưu ngay khi model phát hiện accident, không chờ buffer hay check thêm nữa
            self._save_event_image(frame.copy(), "accident", max_conf, frame_id=self.frame_count)
            self.accident_buffer = []
        else:
            # Nếu không có accident mà buffer có dữ liệu, xử lý buffer
            if self.accident_buffer:
                self._process_accident_buffer()

        if not config.ENABLE_PERSON_DETECTION or not config.ENABLE_COLLISION_FIGHT_DETECTION:
            return frame

        # 4. Tính vận tốc
        all_boxes = [p[0] for p in persons] + [v[0] for v in vehicles] + [t[0] for t in trains]
        tracking_result = self.detector.track_and_compute_velocity(all_boxes, frame_id=self.frame_count)
        if isinstance(tracking_result, tuple):
            velocity_map, _ = tracking_result
        else:
            velocity_map = tracking_result

        # 5. Phát hiện va chạm thông thường
        collisions = self.detector.detect_collisions(persons, vehicles, trains)
        for typ, box1, box2, iou in collisions:
            c1 = get_centroid(box1)
            c2 = get_centroid(box2)
            v1 = 0
            v2 = 0
            prev_pos1 = None
            prev_pos2 = None
            
            # Tìm vận tốc và previous position của 2 vật thể
            for tid, vel in velocity_map.items():
                px, py = self.detector.prev_positions.get(tid, (0, 0))
                if abs(px - c1[0]) < 10 and abs(py - c1[1]) < 10:
                    v1 = vel
                    prev_pos1 = (px, py)
                if abs(px - c2[0]) < 10 and abs(py - c2[1]) < 10:
                    v2 = vel
                    prev_pos2 = (px, py)
            
            # Kiểm tra xem đó có phải va chạm thực sự (không phải camera pan)
            if not self._is_real_collision(v1, v2, iou):
                # Nhưng nếu là falling object (vật rơi từ trên xuống), vẫn lưu
                if not self._is_falling_collision(box1, box2, prev_pos1, prev_pos2):
                    continue
            
            max_vel = max(v1, v2)

            # Chặn false positive cho train-car / person-train
            if typ in ("person-train", "vehicle-train"):
                width1 = max(0.0, box1[2] - box1[0])
                width2 = max(0.0, box2[2] - box2[0])
                if iou < 0.5 or max_vel < 20:
                    continue
                if abs(v1 - v2) < 8 and iou > 0.8:
                    continue
                if (width1 > 150 and width2 < 100) or (width2 > 150 and width1 < 100):
                    continue

            if typ == "vehicle-vehicle":
                prefix = "xetainan"
            elif typ == "train-train":
                prefix = "taunao"
            elif typ == "person-vehicle":
                prefix = "va_cham_nguoi_xe"
            elif typ == "person-train" or typ == "vehicle-train":
                prefix = "taucar"
            else:
                prefix = "collision"
            suffix = "canh_bao" if max_vel > config.VELOCITY_THRESHOLD else ""
            self._save_event_image(frame, prefix, iou, suffix, frame_id=self.frame_count)

        fights = self.detector.detect_fight(persons, velocity_map)
        for box1, box2, iou, max_vel in fights:
            self._save_event_image(frame, "danhnhau", iou, f"v{int(max_vel)}", frame_id=self.frame_count)

        for pbox, pconf in persons:
            cv2.rectangle(frame, (int(pbox[0]), int(pbox[1])), (int(pbox[2]), int(pbox[3])), (255, 0, 0), 2)
            cx, cy = get_centroid(pbox)
            vel = 0
            for tid, v in velocity_map.items():
                px, py = self.detector.prev_positions.get(tid, (0, 0))
                if abs(px - cx) < 10 and abs(py - cy) < 10:
                    vel = v
                    break
            draw_velocity(frame, cx, cy, vel)

        for vbox, vconf in vehicles:
            cv2.rectangle(frame, (int(vbox[0]), int(vbox[1])), (int(vbox[2]), int(vbox[3])), (0, 255, 0), 2)
            cx, cy = get_centroid(vbox)
            vel = 0
            for tid, v in velocity_map.items():
                px, py = self.detector.prev_positions.get(tid, (0, 0))
                if abs(px - cx) < 10 and abs(py - cy) < 10:
                    vel = v
                    break
            draw_velocity(frame, cx, cy, vel)

        for tbox, tconf in trains:
            cv2.rectangle(frame, (int(tbox[0]), int(tbox[1])), (int(tbox[2]), int(tbox[3])), (0, 255, 255), 2)
            cx, cy = get_centroid(tbox)
            vel = 0
            for tid, v in velocity_map.items():
                px, py = self.detector.prev_positions.get(tid, (0, 0))
                if abs(px - cx) < 10 and abs(py - cy) < 10:
                    vel = v
                    break
            draw_velocity(frame, cx, cy, vel)

        for pbox, sconf in smoking_persons:
            draw_smoke_circle(frame, pbox)
            cv2.putText(frame, f"SMOKE {sconf:.2f}", (int(pbox[0]), int(pbox[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        for typ, box1, box2, iou in collisions:
            color = (0, 0, 255)
            c1 = get_centroid(box1)
            c2 = get_centroid(box2)
            v1 = 0
            v2 = 0
            for tid, v in velocity_map.items():
                px, py = self.detector.prev_positions.get(tid, (0, 0))
                if abs(px - c1[0]) < 10 and abs(py - c1[1]) < 10:
                    v1 = v
                if abs(px - c2[0]) < 10 and abs(py - c2[1]) < 10:
                    v2 = v
            if max(v1, v2) > config.VELOCITY_THRESHOLD:
                color = (0, 0, 255)
                label = f"{typ} ⚠️{iou:.2f}"
            else:
                label = f"{typ} {iou:.2f}"
            draw_collision_line(frame, box1, box2, label, color)

        for box1, box2, iou, max_vel in fights:
            color = (0, 0, 255)
            label = f"FIGHT {iou:.2f} v{int(max_vel)}"
            draw_collision_line(frame, box1, box2, label, color)

        # Vẽ vùng ground bound để visualize
        frame = self._draw_ground_bound(frame)

        return frame