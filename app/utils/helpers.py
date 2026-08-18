import numpy as np
import cv2

def get_centroid(box):
    """
    Tính tọa độ trung tâm của bounding box [x1, y1, x2, y2].
    """
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

def compute_iou(box1, box2):
    """
    Tính Intersection over Union (IoU) giữa 2 bounding box [x1, y1, x2, y2].
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    box2_area = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0
    return intersection_area / union_area
