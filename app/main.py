# Main entry point for Video2Action pipeline
import os
import sys
import time
import argparse

# 1. Non-Negotiable CPU Threading Limits
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["TORCH_NUM_THREADS"] = "2"
try:
    import torch
    torch.set_num_threads(2)
except ImportError:
    pass

try:
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
except ImportError:
    pass

from app.models.model_loader import ModelLoader
from app.pipeline.camera import CameraReader
from app.pipeline.scheduler import InferenceScheduler

def build_parser():
    parser = argparse.ArgumentParser(description="Run YOLO Multi-Model Inference Pipeline")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to input video or RTSP/HLS stream URL")
    parser.add_argument("--display", action="store_true", help="Hiển thị video với Bounding Boxes")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    video_path = args.video
    if not video_path.startswith("http") and not video_path.startswith("rtsp") and not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)

    print(f"📹 Starting Pipeline for: {video_path}")

    # Khởi tạo mô hình (Chỉ 1 lần duy nhất)
    models = ModelLoader()
    # Để tránh download trong lúc demo, ta gọi load_yolo (nếu file tồn tại)
    models.load_yolo("models/yolov8s.pt")
    models.load_violence("models/best_violence_model.pth")
    models.load_fire_smoke("model.tflite")

    
    # Khởi tạo Camera (chạy thread riêng)
    camera = CameraReader(source=video_path, camera_id="cam_01")
    
    # Khởi tạo Scheduler (5 FPS max)
    scheduler = InferenceScheduler(camera_reader=camera, models=models, target_fps=5.0, display=args.display)

    try:
        # Bắt đầu luồng đọc camera
        camera.start()
        
        # Đợi một chút để camera có frame đầu tiên
        time.sleep(1.0)
        
        if not camera.is_running:
            print("❌ Failed to start camera stream.")
            sys.exit(1)
            
        # Chạy scheduler trên main thread (hoặc có thể chạy thread riêng)
        # Trong ví dụ này, run_loop() là blocking loop
        scheduler.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Graceful shutdown initiated by user...")
    finally:
        # Resource Cleanup (Mục 35)
        scheduler.stop()
        camera.stop()
        print("✅ Pipeline terminated safely.")

if __name__ == "__main__":
    main()