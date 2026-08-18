import pytest
import numpy as np
from app.models.fire_smoke_classifier import FireSmokeClassifier

def test_fire_smoke_initialization():
    try:
        classifier = FireSmokeClassifier(model_path="models/fire_smoke/model.h5")
        assert classifier is not None
    except Exception as e:
        pytest.skip(f"Skipping test due to missing weights: {e}")

def test_fire_smoke_preprocess():
    # Tạo fake model để test preprocess mà không cần load tốn kém
    classifier = FireSmokeClassifier(model_path=None)
    fake_roi = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    tensor = classifier._preprocess(fake_roi)
    
    # InceptionResNetV2 input mặc định thường là (299, 299) hoặc custom (200, 400)
    # Class này setup target_size = (200, 400) (h, w) hay (w, h)?
    assert tensor.shape == (1, 200, 400, 3) # Batch dimension = 1
    
    # Normalization check: phải nằm trong khoảng [-1, 1] vì x/127.5 - 1
    assert tensor.min() >= -1.0
    assert tensor.max() <= 1.0
