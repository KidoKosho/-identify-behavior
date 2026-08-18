import abc
import numpy as np
from typing import List, Tuple, Any, Dict

class BaseDetector(abc.ABC):
    """
    Interface gốc cho tất cả các model trong pipeline.
    """
    pass

class ImageClassifier(BaseDetector):
    """
    Interface cho các model phân loại trên một khung hình đơn (vd: Fire/Smoke).
    """
    @abc.abstractmethod
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Input: image (H, W, C) numpy array
        Output: (class_name, confidence)
        """
        pass

class VideoDetector(BaseDetector):
    """
    Interface cho các model yêu cầu sequence frames (vd: Violence, Accident).
    """
    @abc.abstractmethod
    def predict_sequence(self, frames: List[np.ndarray]) -> Tuple[str, float]:
        """
        Input: list of images (H, W, C) numpy arrays
        Output: (class_name, confidence)
        """
        pass

class ObjectDetector(BaseDetector):
    """
    Interface cho các model detect object (vd: YOLO).
    """
    @abc.abstractmethod
    def predict(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Input: frame (H, W, C) numpy array
        Output: list of dictionaries containing bbox, class, confidence
        """
        pass
