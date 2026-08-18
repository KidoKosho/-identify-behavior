import time
import psutil
import os
import torch

class PerformanceMonitor:
    """
    Theo dõi hiệu suất phần cứng (CPU, RAM) và độ trễ (Latency)
    của các thành phần AI trong pipeline.
    """
    def __init__(self, log_interval=5.0):
        self.log_interval = log_interval
        self.last_log_time = time.time()
        self.process = psutil.Process(os.getpid())
        
        # Setup threading limits cho CPU based inference
        os.environ["OMP_NUM_THREADS"] = "2"
        os.environ["MKL_NUM_THREADS"] = "2"
        os.environ["TORCH_NUM_THREADS"] = "2"
        if torch.cuda.is_available():
            torch.set_num_threads(2)

        self.reset_metrics()

    def reset_metrics(self):
        self.metrics = {
            "yolo_latency_ms": [],
            "violence_latency_ms": [],
            "fire_latency_ms": [],
            "candidate_count": 0,
            "confirmed_count": 0
        }

    def add_latency(self, component: str, latency_sec: float):
        key = f"{component}_latency_ms"
        if key in self.metrics:
            self.metrics[key].append(latency_sec * 1000.0)

    def increment_counter(self, counter: str, count: int = 1):
        if counter in self.metrics:
            self.metrics[counter] += count

    def log_performance(self, input_fps: float, ai_fps: float):
        now = time.time()
        if now - self.last_log_time >= self.log_interval:
            # Thu thập tài nguyên
            cpu_percent = self.process.cpu_percent(interval=None) / psutil.cpu_count()
            ram_mb = self.process.memory_info().rss / (1024 * 1024)
            
            # Tính toán latency trung bình
            yolo_avg = sum(self.metrics["yolo_latency_ms"]) / max(1, len(self.metrics["yolo_latency_ms"]))
            viol_avg = sum(self.metrics["violence_latency_ms"]) / max(1, len(self.metrics["violence_latency_ms"]))
            fire_avg = sum(self.metrics["fire_latency_ms"]) / max(1, len(self.metrics["fire_latency_ms"]))
            
            print(f"[{time.strftime('%H:%M:%S')}] ⚙️ PERF LOG:")
            print(f"  ├─ Hardware : CPU: {cpu_percent:.1f}% | RAM: {ram_mb:.1f} MB")
            print(f"  ├─ Throughpt: Input: {input_fps:.1f} FPS | AI: {ai_fps:.1f} FPS")
            print(f"  ├─ Latency  : YOLO: {yolo_avg:.1f}ms | Fight: {viol_avg:.1f}ms | Fire: {fire_avg:.1f}ms")
            print(f"  └─ Events   : Candidates: {self.metrics['candidate_count']} | Confirmed: {self.metrics['confirmed_count']}")
            
            self.last_log_time = now
            self.reset_metrics()
