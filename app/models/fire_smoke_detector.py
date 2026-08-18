import cv2
import numpy as np
import tensorflow as tf
from app.models.base_detector import ImageClassifier

class FireSmokeDetector(ImageClassifier):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.interpreter = None
        self.input_shape = (400, 200)  # (width, height) – will be overwritten for TFLite
        self.classes = ['fire_only', 'fire_smoke', 'no_fire_no_smoke', 'smoke_only']
        self.is_tflite = model_path.lower().endswith('.tflite')
        self.load_model()

    def load_model(self):
        print(f"Loading Fire/Smoke Model from {self.model_path}...")
        if self.is_tflite:
            try:
                self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
                self.interpreter.allocate_tensors()
                input_details = self.interpreter.get_input_details()
                # Assume single input tensor
                self.input_shape = (input_details[0]["shape"][2], input_details[0]["shape"][1])  # width, height
            except Exception as e:
                raise RuntimeError(f"Failed to load TFLite Fire/Smoke model: {e}")
            print("TFLite Fire/Smoke Model loaded successfully!")
        else:
            try:
                import keras
                keras.config.enable_unsafe_deserialization()
                self.model = tf.keras.models.load_model(self.model_path, compile=False, safe_mode=False)
            except Exception as e:
                raise RuntimeError(f"Failed to load Fire/Smoke model: {e}")
            print("Fire/Smoke Model loaded successfully!")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, self.input_shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 127.5 - 1.0
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, frame: np.ndarray):
        if self.is_tflite:
            if self.interpreter is None:
                raise RuntimeError("TFLite interpreter not initialized.")
            input_data = self.preprocess(frame)
            input_index = self.interpreter.get_input_details()[0]["index"]
            self.interpreter.set_tensor(input_index, input_data)
            self.interpreter.invoke()
            output_details = self.interpreter.get_output_details()[0]
            preds = self.interpreter.get_tensor(output_details["index"])[0]
        else:
            if self.model is None:
                raise ValueError("Model chưa được load.")
            input_data = self.preprocess(frame)
            preds = self.model.predict(input_data, verbose=0)[0]

        class_idx = int(np.argmax(preds))
        confidence = float(preds[class_idx])
        predicted_class = self.classes[class_idx]
        
        # Log chi tiết phân phối xác suất của từng nhãn
        scores_detail = ", ".join([f"{cls}: {float(preds[i]):.3f}" for i, cls in enumerate(self.classes)])
        print(f"🔥 [FireSmoke AI] Top: {predicted_class} ({confidence:.2%}) | All classes -> [{scores_detail}]")
        
        return predicted_class, confidence
