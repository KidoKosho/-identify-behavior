# app/config/model_config.py
import os
from app.config.performance_config import PerformanceConfig as PConf

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

class ModelConfig:
    # Model paths
    COCO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolov8n.pt')
    YOLO_S_PATH = os.path.join(BASE_DIR, 'models', 'yolov8s.pt')
    VIOLENCE_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_violence_model.pth')
    FIRE_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'fire_smoke', 'model.h5')
    FIRE_TFLITE_PATH = os.path.join(BASE_DIR, 'models', 'fire_smoke', 'model.tflite')
    FIRE_KERAS_PATH = os.path.join(BASE_DIR, 'models', 'fire_smoke', 'fire_smoke_detector.keras')
    
    # Thresholds
    CONF_THRESHOLD = 0.25
    SMOKE_CONF = 0.30
    ACCIDENT_CONF = 0.55
    FIGHT_IOU_THRESHOLD = 0.58
    VEHICLE_CONF_THRESHOLD = 0.65
    
    # Fire model input size
    FIRE_IMGSZ = 224
    
    # Tracking
    MAX_TRACK_HISTORY = PConf.MAX_TRACK_HISTORY