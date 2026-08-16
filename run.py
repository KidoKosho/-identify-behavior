import cv2
import numpy as np

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def get_centroid(box):
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

def draw_velocity(frame, cx, cy, vel, color=(0, 255, 0)):
    cv2.putText(frame, f"v:{vel:.1f}", (int(cx), int(cy) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_collision_line(frame, box1, box2, label, color=(0, 0, 255)):
    try:
        x1, y1, x2, y2 = map(int, box1)
        x3, y3, x4, y4 = map(int, box2)
        c1 = ((x1 + x2) // 2, (y1 + y2) // 2)
        c2 = ((x3 + x4) // 2, (y3 + y4) // 2)
        cv2.line(frame, c1, c2, color, 2)
        cv2.putText(frame, label, (min(c1[0], c2[0]), min(c1[1], c2[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    except:
        pass

def draw_smoke_circle(frame, box, color=(0, 0, 255)):
    try:
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        radius = max((x2 - x1), (y2 - y1)) // 2
        cv2.circle(frame, (cx, cy), radius, color, 2)
        cv2.putText(frame, "SMOKING", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    except:
        pass