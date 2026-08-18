import pytest
import numpy as np
from app.models.yolo_detector import YOLODetector

def test_yolo_initialization():
    # Khởi tạo YOLO với file trọng số mặc định (phải có file yolo11n.pt trong weights/)
    detector = YOLODetector(model_path="models/yolov8n.pt", conf_threshold=0.5)
    assert detector is not None
    assert detector.conf_threshold == 0.5

def test_yolo_predict_empty_frame():
    detector = YOLODetector(model_path="models/yolov8n.pt", conf_threshold=0.5)
    # Tạo frame giả ngẫu nhiên 640x480 RGB
    fake_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    # Dự đoán (khả năng rất cao frame nhiễu ngẫu nhiên không chứa person/vehicle)
    results = detector.predict(fake_frame)
    
    assert isinstance(results, list)
    # Nếu có detect được gì thì label phải thuộc nhóm đã định
    for obj in results:
        assert obj["label"] in ["person", "vehicle"]
        assert "bbox" in obj
        assert len(obj["bbox"]) == 4
        assert obj["confidence"] >= 0.5
