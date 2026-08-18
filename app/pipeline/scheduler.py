import time
import cv2
import numpy as np
import threading

from app.pipeline.candidate_filter import CandidateFilter
from app.pipeline.temporal_buffer import TemporalBuffer
from app.pipeline.event_processor import EventProcessor
from app.models.model_loader import ModelLoader
from app.utils.performance import PerformanceMonitor
from app.pipeline.embedding_worker import EmbeddingWorker

class InferenceScheduler:
    """
    Điều phối luồng Inference đảm bảo tốc độ tối đa là 5 FPS
    bằng cách dùng time.monotonic(). Tích hợp toàn bộ pipeline 5-Level Cascade.
    """
    
    def __init__(self, camera_reader, models: ModelLoader, target_fps=5.0, display=False):
        self.camera = camera_reader
        self.models = models
        self.target_fps = target_fps
        self.interval = 1.0 / target_fps
        self.display = display
        
        self.is_running = False
        self.last_inference_time = 0.0
        self.last_heavy_time = 0.0
        
        # Cache for smooth rendering
        self.last_detected_objects = []
        self.last_fight_candidates = []
        self.last_fire_candidates = []
        self.last_accident_candidates = []
        self.last_fall_candidates = []
        # Lists that hold only confirmed events for rendering
        self.confirmed_fight = []
        self.confirmed_fire = []
        self.confirmed_accident = []
        self.confirmed_fall = []
        self.last_fire_crop = None
        self.last_fire_crop_info = ""
        
        self.inference_count = 0
        self.start_time = 0.0
        
        # Các module pipeline
        self.candidate_filter = CandidateFilter()
        self.temporal_buffer = TemporalBuffer(violence_seq_len=16)
        self.event_processor = EventProcessor(camera_id=self.camera.camera_id)
        self.perf_monitor = PerformanceMonitor()
        self.embedding_worker = EmbeddingWorker()
        
    def start(self):
        if self.is_running:
            return
            
        self.is_running = True
        self.start_time = time.monotonic()
        self.embedding_worker.start()
        print(f"[{self.camera.camera_id}] Scheduler started at {self.target_fps} FPS.")
        
        self._run_loop()

    def _run_loop(self):
        yolo = self.models.get_yolo()
        violence_model = self.models.get_violence()
        fire_smoke_model = self.models.get_fire_smoke()
        
        ai_thread = None
        
        def ai_task(frame_to_process, run_heavy_models):
            new_confirmed_fight = []
            new_confirmed_fire = []
            new_confirmed_accident = []
            new_confirmed_fall = []

            # LEVEL 1: YOLO Object Detection
            t0 = time.time()
            detected_objects = yolo.predict(frame_to_process)
            self.perf_monitor.add_latency("yolo", time.time() - t0)
            
            # LEVEL 2: Candidate Filtering
            fight_candidates, accident_candidates, fall_candidates = self.candidate_filter.process_objects(detected_objects)
            
            t_f = time.time()
            fire_candidates = self.candidate_filter.filter_fire_candidates(frame_to_process)
            self.perf_monitor.add_latency("fire", time.time() - t_f)
            
            # Cập nhật cache cho màn hình
            self.last_detected_objects = detected_objects
            self.last_fight_candidates = fight_candidates
            self.last_accident_candidates = accident_candidates
            self.last_fire_candidates = fire_candidates
            self.last_fall_candidates = fall_candidates
            
            self.perf_monitor.increment_counter("candidate_count", len(fight_candidates) + len(fire_candidates) + len(accident_candidates) + len(fall_candidates))
            
            # LEVEL 3 & 4: Specialized Models & Temporal Confirmation
            
            # --- FALL PIPELINE ---
            is_fall_positive = False
            best_fall_bbox = None
            current_fall_conf = 0.0
            for cand in fall_candidates:
                if cand["score"] >= 0.80:
                    is_fall_positive = True
                    if cand["score"] > current_fall_conf:
                        current_fall_conf = cand["score"]
                        best_fall_bbox = cand["bbox"]
                        
            if self.temporal_buffer.update_history("fall", is_fall_positive):
                self.perf_monitor.increment_counter("confirmed_count")
                self.event_processor.process_event("fall_confirmed", current_fall_conf, frame_to_process, best_fall_bbox)
                self._dispatch_embedding(frame_to_process, best_fall_bbox, "fall_confirmed")
                # Store confirmed fall for rendering
                new_confirmed_fall.append({"bbox": best_fall_bbox, "score": current_fall_conf, "label": "FALL"})
            
            # --- VIOLENCE PIPELINE ---
            is_fight_positive = False
            best_fight_bbox = None
            current_fight_conf = 0.0
            if len(fight_candidates) > 0 and violence_model is not None:
                seq = self.temporal_buffer.get_violence_sequence()
                if seq is not None:
                    t1 = time.time()
                    pred_class, conf = violence_model.predict_clip(seq)
                    self.perf_monitor.add_latency("violence", time.time() - t1)
                    
                    if pred_class == 1:
                        current_fight_conf = conf
                        for cand in fight_candidates:
                            cand["score"] = conf
                            
                        if conf >= 0.40:
                            is_fight_positive = True
                            best_fight_bbox = fight_candidates[0]["bbox"]
            
            if self.temporal_buffer.update_history("fight", is_fight_positive):
                self.perf_monitor.increment_counter("confirmed_count")
                self.event_processor.process_event("fight_confirmed", current_fight_conf, frame_to_process, best_fight_bbox)
                self._dispatch_embedding(frame_to_process, best_fight_bbox, "fight_confirmed")
                # Store confirmed fight for rendering
                new_confirmed_fight.append({"bbox": best_fight_bbox, "score": current_fight_conf, "label": "FIGHT"})
                
            # --- HEAVY MODELS THROTTLING ---
            if not run_heavy_models:
                self.confirmed_fall = new_confirmed_fall
                self.confirmed_fight = new_confirmed_fight
                self.inference_count += 1
                return

            # --- FIRE & SMOKE PIPELINE (RULE-BASED MOG2 & AI MODEL) ---
            for cand in fire_candidates:
                pred_class = cand["type"]
                conf = cand["score"]
                
                if fire_smoke_model is not None:
                    x1, y1, x2, y2 = cand["bbox"]
                    h, w = frame_to_process.shape[:2]
                    box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
                    pad_x = max(25, int(box_w * 0.25))
                    pad_y = max(20, int(box_h * 0.25))
                    crop_y1, crop_y2 = max(0, y1 - pad_y), min(h, y2 + pad_y)
                    crop_x1, crop_x2 = max(0, x1 - pad_x), min(w, x2 + pad_x)
                    crop = frame_to_process[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    if crop.size > 0:
                        t2 = time.time()
                        ai_class, ai_conf = fire_smoke_model.predict(crop)
                        self.perf_monitor.add_latency("fire_ai", time.time() - t2)
                        
                        self.last_fire_crop = crop.copy()
                        self.last_fire_crop_info = f"{ai_class} ({ai_conf:.2%})"
                        
                        cand["type"] = ai_class
                        cand["score"] = ai_conf
                        
                        if ai_class != 'no_fire_no_smoke':
                            fire_score = self.temporal_buffer.track_and_score_fire_smoke(ai_class, ai_conf, cand["bbox"])
                            if fire_score >= 0.70:
                                event_type = f"{ai_class}_confirmed"
                                if ai_class == "smoke_only":
                                    disp_label = "SMOKE"
                                elif ai_class == "fire_smoke":
                                    disp_label = "FIRE + SMOKE"
                                else:
                                    disp_label = "FIRE"

                                if self.temporal_buffer.check_fire_smoke_cooldown():
                                    self.perf_monitor.increment_counter("confirmed_count")
                                    self.event_processor.process_event(event_type, fire_score, frame_to_process, cand["bbox"])
                                    self._dispatch_embedding(frame_to_process, cand["bbox"], event_type)
                                    
                                new_confirmed_fire.append({
                                    "bbox": cand["bbox"],
                                    "score": fire_score,
                                    "label": disp_label,
                                    "class_type": ai_class
                                })
                        continue

                fire_score = self.temporal_buffer.track_and_score_fire_smoke(pred_class, conf, cand["bbox"])
                cand["score"] = fire_score
                
                if fire_score >= 0.75:
                    disp_label = "SMOKE" if "smoke" in pred_class.lower() else "FIRE"
                    event_type = f"{pred_class}_confirmed"
                    if self.temporal_buffer.check_fire_smoke_cooldown():
                        self.perf_monitor.increment_counter("confirmed_count")
                        self.event_processor.process_event(event_type, fire_score, frame_to_process, cand["bbox"])
                        self._dispatch_embedding(frame_to_process, cand["bbox"], event_type)
                        
                    new_confirmed_fire.append({
                        "bbox": cand["bbox"],
                        "score": fire_score,
                        "label": disp_label,
                        "class_type": pred_class
                    })

            # --- ACCIDENT PIPELINE (RULE-BASED VEHICLE TRACKING) ---
            is_accident_positive = False
            best_accident_bbox = None
            current_accident_conf = 0.0
            
            for cand in accident_candidates:
                conf = cand["score"]
                if conf > current_accident_conf:
                    current_accident_conf = conf
                    best_accident_bbox = cand["bbox"]
                    is_accident_positive = True
            
            if is_accident_positive:
                self.perf_monitor.increment_counter("confirmed_count")
                self.event_processor.process_event("accident_confirmed", current_accident_conf, frame_to_process, best_accident_bbox)
                self._dispatch_embedding(frame_to_process, best_accident_bbox, "accident_confirmed")
                # Store confirmed accident for rendering
                new_confirmed_accident.append({"bbox": best_accident_bbox, "score": current_accident_conf, "label": "ACCIDENT"})

            self.confirmed_fall = new_confirmed_fall
            self.confirmed_fight = new_confirmed_fight
            self.confirmed_fire = new_confirmed_fire
            self.confirmed_accident = new_confirmed_accident
            self.inference_count += 1
            
        while self.is_running:
            now = time.monotonic()
            
            frame = self.camera.get_latest_frame()
            if frame is None:
                if not self.camera.is_running:
                    print("Camera stream ended. Stopping scheduler.")
                    break
                time.sleep(0.01)
                continue
                
            # --- RENDER BLOCK ---
            if self.display:
                display_frame = frame.copy()
                
                # Render detected objects (YOLO)
                for obj in self.last_detected_objects:
                    x1, y1, x2, y2 = obj["bbox"]
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{obj['label']} {obj['confidence']:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Render Confirmed Events
                # Fight
                for cand in self.confirmed_fight:
                    if cand.get("bbox"):
                        x1, y1, x2, y2 = cand["bbox"]
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(display_frame, f"FIGHT {cand['score']:.2f}", (x1, max(0, y1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Fire / Smoke (Differentiated rendering)
                for cand in self.confirmed_fire:
                    if cand.get("bbox"):
                        x1, y1, x2, y2 = cand["bbox"]
                        label = cand.get("label", "FIRE")
                        class_type = cand.get("class_type", "").lower()
                        score_val = cand.get("score", 0.0)
                        
                        # Distinct colors and banners:
                        # SMOKE -> Smokey Slate Gray (180, 180, 180)
                        # FIRE + SMOKE -> Deep Orange / Coral (0, 140, 255)
                        # FIRE -> Bright Scarlet / Red-Orange (0, 60, 255)
                        if "smoke_only" in class_type or label == "SMOKE":
                            box_color = (190, 190, 190)
                            txt_color = (0, 0, 0)
                        elif "fire_smoke" in class_type or "+" in label:
                            box_color = (0, 140, 255)
                            txt_color = (255, 255, 255)
                        else:
                            box_color = (0, 60, 255)
                            txt_color = (255, 255, 255)
                            
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 3)
                        text_str = f"{label} {score_val:.2f}"
                        (tw, th), _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        # Text background header banner
                        cv2.rectangle(display_frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), box_color, -1)
                        cv2.putText(display_frame, text_str, (x1 + 4, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_color, 2)

                # Accident
                for cand in self.confirmed_accident:
                    if cand.get("bbox"):
                        x1, y1, x2, y2 = cand["bbox"]
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                        cv2.putText(display_frame, f"ACCIDENT {cand['score']:.2f}", (x1, max(0, y1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Fall
                for cand in self.confirmed_fall:
                    if cand.get("bbox"):
                        x1, y1, x2, y2 = cand["bbox"]
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                        cv2.putText(display_frame, f"FALL {cand['score']:.2f}", (x1, max(0, y1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Render Fire Crop overlay & Dedicated Window
                if self.last_fire_crop is not None:
                    # Show dedicated window
                    crop_display = cv2.resize(self.last_fire_crop, (400, 200))
                    cv2.putText(crop_display, self.last_fire_crop_info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.imshow("Fire_Smoke_Crop", crop_display)
                    
                    # Also overlay Picture-in-Picture on main frame
                    fh, fw = display_frame.shape[:2]
                    pip_w, pip_h = min(200, fw // 3), min(100, fh // 3)
                    if pip_w > 10 and pip_h > 10:
                        pip_crop = cv2.resize(self.last_fire_crop, (pip_w, pip_h))
                        display_frame[10:10+pip_h, fw-pip_w-10:fw-10] = pip_crop
                        cv2.rectangle(display_frame, (fw-pip_w-10, 10), (fw-10, 10+pip_h), (0, 165, 255), 2)
                        cv2.putText(display_frame, "CROP AI", (fw-pip_w-5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

                cv2.imshow(self.camera.camera_id, display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    break

            if now - self.last_inference_time >= self.interval:
                if ai_thread is None or not ai_thread.is_alive():
                    self.last_inference_time = now
                    self.temporal_buffer.add_frame(frame)
                    
                    # Heavy models run at ~2 FPS max
                    run_heavy = (now - self.last_heavy_time) >= 0.5
                    if run_heavy:
                        self.last_heavy_time = now
                        
                    ai_thread = threading.Thread(target=ai_task, args=(frame.copy(), run_heavy), daemon=True)
                    ai_thread.start()
                    
            time.sleep(0.01)
            
            elapsed = time.monotonic() - self.start_time
            if elapsed > 0:
                ai_fps = self.inference_count / elapsed
                input_fps = self.camera.frame_count / elapsed if self.camera else 0.0
                self.perf_monitor.log_performance(input_fps, ai_fps)

    def _dispatch_embedding(self, frame, bbox, event_type):
        if bbox is None: return
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        crop_y1, crop_y2 = max(0, y1-10), min(h, y2+10)
        crop_x1, crop_x2 = max(0, x1-10), min(w, x2+10)
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size > 0:
            self.embedding_worker.enqueue(crop, event_type)

    def stop(self):
        self.is_running = False
        self.embedding_worker.stop()
        if self.display:
            cv2.destroyAllWindows()
        print(f"[{self.camera.camera_id}] Scheduler stopped.")
