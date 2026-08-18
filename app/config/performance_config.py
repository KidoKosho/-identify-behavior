# config/performance_config.py
import os

class PerformanceConfig:
    # CPU threads
    OMP_NUM_THREADS = 2
    MKL_NUM_THREADS = 2
    TORCH_NUM_THREADS = 2

    # Inference frequency
    INFERENCE_FPS = 5  # AI inference mỗi giây
    PROCESS_EVERY_N_FRAMES = 6  # Với camera 30 FPS, 30/5 = 6

    # Giới hạn RAM (lịch sử tracking)
    MAX_TRACK_HISTORY = 15  # frame

    # Cooldown snapshot
    SNAPSHOT_COOLDOWN = 10  # giây