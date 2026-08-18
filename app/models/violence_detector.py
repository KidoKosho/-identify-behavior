"""
Violence detector dùng trong pipeline real-time.
"""
import os
import cv2
import torch
import numpy as np
from app.models.video_backbone import ViolenceVideoModel


class ViolenceDetector:
    def __init__(self, model_path=None, num_frames=16, device='cpu', threshold=0.5):
        self.num_frames = num_frames
        self.device = device
        self.threshold = threshold
        self.model = ViolenceVideoModel(num_frames=num_frames).to(device)
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path):
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Loaded violence model from {model_path}")

    def predict_clip(self, frames):
        """
        frames: list of numpy arrays (H, W, 3) RGB, length >= num_frames
        Returns: (pred_class, confidence)
        """
        if len(frames) < self.num_frames:
            return None

        # Lấy num_frames frame gần nhất
        frames = frames[-self.num_frames:]
        processed = []
        for frame in frames:
            frame = cv2.resize(frame, (224, 224))
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            processed.append(frame)

        x = torch.stack(processed, dim=0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        return pred_class, confidence

    def is_violence(self, frame_buffer):
        """Trả về (is_violence, confidence)"""
        result = self.predict_clip(frame_buffer)
        if result:
            pred_class, conf = result
            if pred_class == 1 and conf >= self.threshold:
                return True, conf
        return False, 0.0
