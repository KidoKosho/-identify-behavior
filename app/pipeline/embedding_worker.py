import queue
import threading
import time
import cv2
import numpy as np

class EmbeddingWorker:
    """
    Worker xử lý Embedding (MobileNet + FAISS) bất đồng bộ.
    Mục đích: Không block luồng Inference chính.
    """
    
    def __init__(self, max_queue_size=50):
        self.q = queue.Queue(maxsize=max_queue_size)
        self.is_running = False
        self.thread = None
        # Mock models
        self.mobilenet = None 
        self.faiss_index = None

    def start(self):
        if self.is_running:
            return
            
        self.is_running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        print("[EmbeddingWorker] Started Async Queue.")

    def enqueue(self, crop: np.ndarray, event_type: str):
        """
        Đưa crop vào queue. Nếu queue đầy, sẽ drop frame cũ (non-blocking).
        """
        if not self.is_running:
            return
            
        try:
            self.q.put_nowait((crop, event_type, time.time()))
        except queue.Full:
            print("[EmbeddingWorker] Queue is full, dropping frame.")

    def _worker_loop(self):
        while self.is_running:
            try:
                # Đợi tối đa 1s để lấy việc, nếu không thì lặp lại để check is_running
                item = self.q.get(timeout=1.0)
                crop, event_type, timestamp = item
                
                # Mock xử lý Embedding:
                # 1. Resize crop
                # resized = cv2.resize(crop, (224, 224))
                # 2. Extract feature
                # feature = self.mobilenet.predict(resized)
                # 3. FAISS Search
                # Dists, Ids = self.faiss_index.search(feature, k=5)
                
                # Simulating processing time (0.05s)
                time.sleep(0.05)
                
                # print(f"[EmbeddingWorker] Processed {event_type} crop asynchronously.")
                
                self.q.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[EmbeddingWorker] Error: {e}")

    def stop(self):
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        print("[EmbeddingWorker] Stopped.")
