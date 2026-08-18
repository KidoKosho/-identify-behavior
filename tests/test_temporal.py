import pytest
from app.pipeline.temporal_buffer import TemporalBuffer

def test_temporal_confirmation():
    buffer = TemporalBuffer(window_size=5, confirm_threshold=3, cooldown_sec=1.0)
    
    # Khung 1: Fight -> Chưa đủ 3
    assert not buffer.update_history("fight", True)
    
    # Khung 2: Không Fight -> Chưa đủ 3
    assert not buffer.update_history("fight", False)
    
    # Khung 3: Fight -> Chưa đủ 3
    assert not buffer.update_history("fight", True)
    
    # Khung 4: Fight -> 3/5 positive -> CONFIRMED
    assert buffer.update_history("fight", True) == True
    
    # Khung 5: Fight -> Blocked by Cooldown!
    assert not buffer.update_history("fight", True)

def test_fire_score_calculation():
    buffer = TemporalBuffer()
    
    # Model conf = 0.9, bbox = [0, 0, 100, 100]
    score1 = buffer.calculate_fire_score(0.9, [0, 0, 100, 100])
    
    # Lần 2: Cùng vị trí -> được thưởng Persistence và Spatial
    score2 = buffer.calculate_fire_score(0.9, [10, 10, 110, 110])
    
    assert score2 > score1 # Score phải tăng vì tính bền vững (Persistence)
