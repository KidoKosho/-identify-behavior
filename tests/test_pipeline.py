import pytest
import numpy as np
from app.pipeline.candidate_filter import CandidateFilter

def test_fight_candidate_filter():
    filter_mod = CandidateFilter(fight_threshold=0.3)
    
    # Frame 1: 2 người đứng gần nhau
    frame1 = [
        {"label": "person", "class_id": 0, "confidence": 0.9, "bbox": [100, 100, 150, 300]},
        {"label": "person", "class_id": 0, "confidence": 0.8, "bbox": [120, 120, 160, 310]}
    ]
    filter_mod.filter_fight_candidates(frame1)
    
    # Frame 2: Có chuyển động mạnh và giao thoa
    frame2 = [
        {"label": "person", "class_id": 0, "confidence": 0.9, "bbox": [115, 100, 165, 300]},
        {"label": "person", "class_id": 0, "confidence": 0.8, "bbox": [110, 120, 150, 310]}
    ]
    candidates = filter_mod.filter_fight_candidates(frame2)
    
    assert len(candidates) >= 1
    assert candidates[0]["type"] == "fight_candidate"
    
def test_fight_no_candidate():
    filter_mod = CandidateFilter(fight_threshold=0.5)
    
    # Mock detected objects: 2 người cách xa nhau vạn dặm
    detected_objects = [
        {"label": "person", "class_id": 0, "confidence": 0.9, "bbox": [0, 0, 50, 100]},
        {"label": "person", "class_id": 0, "confidence": 0.8, "bbox": [500, 500, 550, 600]}
    ]
    
    candidates = filter_mod.filter_fight_candidates(detected_objects)
    
    # Không thể thành Fight candidate vì khoảng cách xa
    assert len(candidates) == 0

def test_fire_candidate_filter():
    filter_mod = CandidateFilter()
    
    # Khung ảnh ngẫu nhiên tối thui
    dark_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    candidates1 = filter_mod.filter_fire_candidates(dark_frame)
    assert len(candidates1) == 0
    
    # Khung ảnh có một vùng đỏ rực lớn
    red_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    red_frame[200:300, 200:300] = [0, 0, 255] # BGR
    
    candidates2 = filter_mod.filter_fire_candidates(red_frame)
    assert len(candidates2) > 0
    assert candidates2[0]["type"] in ["fire", "fire_smoke", "smoke"]
