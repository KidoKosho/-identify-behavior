import os
import cv2
import numpy as np
try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None

class FireSmokeClassifier:
    """
    Classifier mô hình InceptionResNetV2-forest fire.
    """
    def __init__(self, model_path="models/fire_smoke/model.h5"):
        self.model = None
        if model_path and os.path.exists(model_path) and load_model is not None:
            try:
                self.model = load_model(model_path)
                print(f"FireSmokeClassifier loaded successfully from {model_path}")
            except Exception as e:
                print(f"ERROR loading FireSmokeClassifier model from {model_path}: {e}")
        else:
            print("WARNING: TensorFlow not found or model file missing! FireSmokeClassifier is returning MOCK non_fire.")

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        # Resize về kích thước chuẩn Inception (giả định 200x400 theo dataset cũ)
        resized = cv2.resize(roi, (400, 200)) # w, h
        # Normalize -1 to 1
        normalized = (resized / 127.5) - 1.0
        return np.expand_dims(normalized, axis=0)

    def predict(self, roi: np.ndarray):
        """
        Dự đoán ảnh ROI.
        Trả về (pred_class_name, confidence)
        """
        if self.model is None:
            return ("non_fire", 0.0)
            
        tensor = self._preprocess(roi)
        preds = self.model.predict(tensor, verbose=0)[0]
        # Giả định [0] = fire, [1] = non-fire/smoke tuỳ checkpoint
        conf = float(np.max(preds))
        idx = int(np.argmax(preds))
        
        class_name = "fire" if idx == 0 else "smoke"
        return class_name, conf
