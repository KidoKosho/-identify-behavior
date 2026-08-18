import os
from typing import List, Dict, Any
from ultralytics import YOLO
import numpy as np

from app.models.base_detector import ObjectDetector

class YOLODetector(ObjectDetector):
    """
    Wrapper cho YOLO11n.
    Phát hiện person và vehicles để làm tiền đề cho Candidate Filtering.
    """
    
    # Mapping chuẩn của COCO dataset
    PERSON_CLASS = 0
    VEHICLE_CLASSES = {2, 3, 5, 7} # car, motorcycle, bus, truck

    def __init__(self, model_path: str = "models/yolo11x.pt", conf_threshold: float = 0.15):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
    def predict(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Input: numpy array (H, W, 3) BGR format (OpenCV)
        Output: Danh sách các detected object (person, vehicle).
        """
        # Run YOLO inference
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        
        detected_objects = []
        if not results or len(results) == 0:
            return detected_objects
            
        boxes = results[0].boxes
        
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            if conf < self.conf_threshold:
                continue
                
            # Chỉ lấy person và vehicles
            if cls_id == self.PERSON_CLASS:
                label = "person"
            elif cls_id in self.VEHICLE_CLASSES:
                if conf < 0.50:
                    continue # Video2Action: Loại bỏ xe cộ có confidence < 0.50
                label = "vehicle"
            else:
                continue
                
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            detected_objects.append({
                "label": label,
                "class_id": cls_id,
                "confidence": conf,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })
            
        return detected_objects
