import os
import torch
import warnings

from app.models.yolo_detector import YOLODetector
from app.models.violence_detector import ViolenceDetector

class ModelLoader:
    """
    Quản lý việc tải các mô hình AI (YOLO và Violence).
    Sử dụng Singleton Pattern hoặc truy cập tĩnh nếu cần.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self, device: str = None):
        if self._initialized:
            return
            
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.yolo = None
        self.violence = None
        
        self._initialized = True
        
    def load_yolo(self, weight_path: str = "models/yolo11x.pt"):
        if self.yolo is None:
            if os.path.exists(weight_path):
                self.yolo = YOLODetector(model_path=weight_path, conf_threshold=0.15)
            else:
                print(f"Warning: YOLO weight not found at {weight_path}")
        return self.yolo
        
    def load_violence(self, weight_path: str = "models/best_violence_model.pth"):
        if self.violence is None:
            if os.path.exists(weight_path):
                self.violence = ViolenceDetector(model_path=weight_path, device=self.device)
            else:
                print(f"Warning: Violence weight not found at {weight_path}")
        return self.violence
        
    def load_fire_smoke(self, weight_path: str = "model.tflite"):
        from app.models.fire_smoke_detector import FireSmokeDetector
        if not hasattr(self, 'fire_smoke') or self.fire_smoke is None:
            if os.path.exists(weight_path):
                self.fire_smoke = FireSmokeDetector(model_path=weight_path)
            else:
                print(f"Warning: Fire/Smoke weight not found at {weight_path}")
                self.fire_smoke = None
        return getattr(self, 'fire_smoke', None)
        
    def get_yolo(self):
        if self.yolo is None:
            raise RuntimeError("YOLO model not loaded yet. Call load_yolo() first.")
        return self.yolo
        
    def get_violence(self):
        return self.violence

    def get_fire_smoke(self):
        return getattr(self, 'fire_smoke', None)