from ultralytics import YOLO
import os
import config

class ModelLoader:
    def __init__(self):
        self.coco_model = None
        self.fire_model = None
        self.accident_model = None
        self.load_models()

    def load_models(self):
        print("Loading models...")
        if config.ENABLE_PERSON_DETECTION:
            self.coco_model = YOLO(config.COCO_MODEL)
            print("✅ COCO model loaded")
        else:
            print("ℹ️ Person detection disabled; skipping COCO model load")

        if os.path.exists(config.FIRE_MODEL_PATH):
            self.fire_model = YOLO(config.FIRE_MODEL_PATH)
            print("✅ Fire & Smoke model loaded")
        else:
            print("⚠️ Fire model not found!")

        if os.path.exists(config.ACCIDENT_MODEL_PATH):
            self.accident_model = YOLO(config.ACCIDENT_MODEL_PATH)
            print("✅ Accident model loaded")
        else:
            print("⚠️ Accident model not found")