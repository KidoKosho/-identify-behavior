import pytest
import numpy as np
import torch
from app.models.violence_detector import ViolenceDetector

def test_violence_initialization():
    # Giả lập chưa có file weights chuẩn để tránh tốn thời gian CI, ta chỉ test cấu trúc
    try:
        detector = ViolenceDetector(model_path="models/best_violence_model.pth")
    except Exception as e:
        pytest.skip(f"Skipping test due to missing/invalid weights: {e}")
        
    assert detector is not None

def test_violence_input_shape():
    detector = ViolenceDetector(model_path="models/best_violence_model.pth")
    # Tạo chuỗi 16 frames giả ngẫu nhiên 224x224 RGB
    fake_sequence = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(16)]
    
    # Dự đoán bằng predict_clip
    result = detector.predict_clip(fake_sequence)
    
    assert result is not None
    pred_class, conf = result
    assert isinstance(pred_class, int)
    assert 0.0 <= conf <= 1.0
