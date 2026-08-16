import csv

import cv2
import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


def pixel_to_meter(x, y, scale=1.0, homography=None):
    """Convert pixel coordinates to real-world coordinates in meters.

    If a homography matrix is available, it is used first. Otherwise a simple
    scalar scale factor is applied.
    """
    if homography is not None:
        try:
            h = np.asarray(homography, dtype=float).reshape(3, 3)
            point = np.array([x, y, 1.0], dtype=float)
            warped = h @ point
            if abs(warped[2]) > 1e-8:
                warped = warped / warped[2]
            return float(warped[0]), float(warped[1])
        except Exception:
            pass
    return float(x * scale), float(y * scale)


def export_tracking_records(records, output_path, scale=1.0, homography=None, columns=None):
    """Save tracking records to CSV and return the records as a DataFrame when pandas is available."""
    resolved_columns = columns or ["frame_id", "track_id", "x_center", "y_center"]

    if not records:
        if pd is not None:
            empty = pd.DataFrame(columns=resolved_columns)
            if output_path:
                empty.to_csv(output_path, index=False)
            return empty
        if output_path:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(resolved_columns)
        return []

    if pd is not None:
        df = pd.DataFrame(records)
        if columns:
            df = df.reindex(columns=columns)

        if "x_center" in df.columns and "y_center" in df.columns:
            meters = [pixel_to_meter(float(x), float(y), scale=scale, homography=homography)
                      for x, y in zip(df["x_center"], df["y_center"])]
            df["x_center_m"] = [m[0] for m in meters]
            df["y_center_m"] = [m[1] for m in meters]

        if output_path:
            df.to_csv(output_path, index=False)
        return df

    if output_path:
        fieldnames = list(records[0].keys()) if records else resolved_columns
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in records:
                out = dict(row)
                if "x_center" in out and "y_center" in out:
                    x_m, y_m = pixel_to_meter(float(out["x_center"]), float(out["y_center"]), scale=scale, homography=homography)
                    out["x_center_m"] = x_m
                    out["y_center_m"] = y_m
                writer.writerow(out)
    return records


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

def draw_velocity(frame, cx, cy, vel, color=(0,255,0)):
    cv2.putText(frame, f"v:{vel:.1f}", (int(cx), int(cy)-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_collision_line(frame, box1, box2, label, color=(0,0,255)):
    try:
        x1, y1, x2, y2 = map(int, box1)
        x3, y3, x4, y4 = map(int, box2)
        c1 = ((x1 + x2)//2, (y1 + y2)//2)
        c2 = ((x3 + x4)//2, (y3 + y4)//2)
        cv2.line(frame, c1, c2, color, 2)
        cv2.putText(frame, label, (min(c1[0], c2[0]), min(c1[1], c2[1])-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    except:
        pass

def draw_smoke_circle(frame, box, color=(0,0,255)):
    try:
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2)//2
        cy = (y1 + y2)//2
        radius = max((x2-x1), (y2-y1))//2
        cv2.circle(frame, (cx, cy), radius, color, 2)
        cv2.putText(frame, "SMOKING", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    except:
        pass
