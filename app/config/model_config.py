# config/model_config.py
import os
from config.performance_config import PerformanceConfig as PConf

class ModelConfig:
    # Model paths
    COCO_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'yolov8n.pt')
    FIRE_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'fire.pt')
    ACCIDENT_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'accident.pt')
    
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