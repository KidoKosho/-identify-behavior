# 🚨 Identify-Behavior: Multi-Model Real-Time Behavior & Event Detection System

Hệ thống AI giám sát và phân tích video/camera thông minh thời gian thực (Real-time AI Video Analytics). Hệ thống áp dụng kiến trúc **5-Level Cascade** để tự động phát hiện các hành vi và sự cố bất thường: **Ẩu đả/Bạo lực (Violence/Fight)**, **Hỏa hoạn & Khói (Fire & Smoke)**, **Tai nạn giao thông (Vehicle Accident)** và **Ngã (Fall Detection)** với hiệu năng tối ưu, tiêu thụ ít tài nguyên phần cứng.

---

## 🏗️ Kiến trúc Hệ thống (5-Level Cascade Architecture)

```
[ Camera Stream / Video (30 FPS) ]
            │
            ▼
┌───────────────────────────────────────┐
│   CameraWorker (Background Thread)    │  Đọc luồng liên tục, độ trễ tối thiểu
└───────────────────┬───────────────────┘
                    │ Ghi đè khung hình mới nhất (Zero Frame Queue)
                    ▼
┌───────────────────────────────────────┐
│   LatestFrameStore (Thread-Safe)      │  Lưu 1 frame duy nhất cho mỗi camera
└───────────────────┬───────────────────┘
                    │ Polling ở tốc độ tối ưu (Inference Target: 5 FPS)
                    ▼
┌───────────────────────────────────────┐
│   InferenceScheduler                  │  Điều phối tải (CPU Limit: 2 Threads)
└───────────────────┬───────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  LEVEL 1: Shared YOLO Detection                                        │
│  - Phát hiện nhanh Người (person) và Phương tiện (car, truck, bus, moto)│
└───────────────────┬────────────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  LEVEL 2: Rule-Based Candidate Filter (Bộ lọc Động học & Không gian)  │
│  ├── 1. Fight Engine: Khoảng cách cặp người, vận tốc tương đối & IOU  │
│  ├── 2. Accident Engine: Vector áp sát, giảm tốc đột ngột, va chạm     │
│  ├── 3. Fall Engine: Tỷ lệ co h/w (đứng -> nằm) & tốc độ rơi trọng tâm │
│  └── 4. Fire/Smoke Engine: MOG2 tách nền + Phổ màu HSV (Lửa & Khói)   │
└───────────────────┬────────────────────────────────────────────────────┘
                    │
                    ▼ (Chỉ kích hoạt khi phát hiện ứng viên khả nghi)
┌────────────────────────────────────────────────────────────────────────┐
│  LEVEL 3: Deep AI Specialist Models                                    │
│  ├── Violence Model (PyTorch): Phân loại chuỗi 16-frame liên tiếp      │
│  └── Fire/Smoke Model (Keras/TensorFlow): Phân loại vùng ảnh khả nghi  │
└───────────────────┬────────────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  LEVEL 4: Temporal Confirmation & Anti-False Positive (State Machine)  │
│  - Khử nhiễu qua cửa sổ trượt (Cần xác nhận 3/5 frame liên tiếp)      │
│  - Chấm điểm độ bền vững không gian (Spatial Persistence)              │
└───────────────────┬────────────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  LEVEL 5: Event Processing & Snapshot Manager                          │
│  - Xuất sự kiện cảnh báo, vẽ Bounding Box, lưu Snapshot có cooldown   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Các tính năng nổi bật

1. **Hiệu năng cao & Tiết kiệm tài nguyên:**
   - Inference decoupling: Camera stream 30 FPS nhưng AI inference chỉ chạy ở $\le 5\text{ FPS}$ giúp tiết kiệm tới **80% CPU/GPU**.
   - Bộ lọc Rule-based loại bỏ >90% khung hình bình thường trước khi gọi đến các mô hình AI chuyên sâu (Level 3).
2. **Chống báo động giả cực mạnh (Anti-False-Positive):**
   - Không kích hoạt cảnh báo tức thời từ 1 frame đơn lẻ.
   - Cơ chế cửa sổ thời gian (Temporal Buffer) yêu cầu sự kiện duy trì liên tục qua nhiều khung hình.
3. **Quản lý trọng số Model qua Hugging Face Hub:**
   - Tự động đồng bộ và tải toàn bộ trọng số (Weights) chỉ với một lệnh.

---

## 📂 Cấu trúc Thư mục

```text
├── app/
│   ├── camera/
│   │   ├── camera_worker.py        # Đọc stream RTSP / Video file
│   │   └── latest_frame_store.py   # Bộ nhớ đệm thread-safe 1-slot
│   ├── config/
│   │   ├── model_config.py         # Cấu hình ngưỡng & đường dẫn model
│   │   └── performance_config.py   # Cấu hình giới hạn luồng & FPS
│   ├── models/
│   │   ├── base_detector.py        # Base class chuẩn hóa các detector
│   │   ├── model_loader.py         # Singleton Model Loader
│   │   ├── yolo_detector.py        # YOLO Object Detector
│   │   ├── violence_detector.py    # PyTorch Violence Sequence Model
│   │   └── fire_smoke_classifier.py# Keras Fire/Smoke Classifier
│   ├── pipeline/
│   │   ├── candidate_filter.py     # Bộ lọc ứng viên Rule-Based (Level 2)
│   │   ├── temporal_buffer.py      # Bộ nhớ đệm thời gian chống nhiễu (Level 4)
│   │   ├── scheduler.py            # Bộ điều phối Inference 5 FPS
│   │   └── event_processor.py      # Xử lý sự kiện & Snapshot (Level 5)
│   ├── utils/                      # Helper tính toán IOU, vector, metrics
│   └── main.py                     # Entrypoint chạy hệ thống
├── docs/                           # Tài liệu kiến trúc chi tiết
├── models/                         # Thư mục chứa trọng số weights (.pt, .pth, .h5)
├── scripts/
│   ├── upload_to_hf.py             # Script đẩy models lên Hugging Face
│   ├── download_models.py           # Script tải models từ Hugging Face
│   └── run_tests.py                # Script chạy toàn bộ test
├── tests/                          # Bộ Unit Tests toàn diện
├── requirements.txt
└── README.md
```

---

## 🚀 Cài đặt & Hướng dẫn Sử dụng

### 1. Cài đặt Môi trường
Khuyến nghị sử dụng Python 3.10 hoặc 3.11:

```bash
# Clone repository
git clone https://github.com/KidoKosho/-identify-behavior.git
cd -identify-behavior

# Tạo và kích hoạt môi trường ảo
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

---

### 2. Tải Trọng số Model từ Hugging Face Hub
Dự án lưu trữ các file trọng số AI lớn trên **Hugging Face Hub**. Chạy lệnh sau để tải toàn bộ model về thư mục `models/`:

```bash
python scripts/download_models.py --repo-id kidokosho/identify-behavior-models
```

*(Hoặc nếu bạn muốn upload model mới lên Hugging Face Hub)*:
```bash
python scripts/upload_to_hf.py --repo-id kidokosho/identify-behavior-models --token <hf_token_cua_ban>
```

---

### 3. Chạy Phân tích Video / Camera Stream

* **Chạy với file video:**
  ```bash
  python app/main.py --source "path/to/video.mp4"
  ```

* **Chạy với luồng Camera RTSP / Webcam:**
  ```bash
  # Webcam máy tính
  python app/main.py --source 0

  # Camera RTSP
  python app/main.py --source "rtsp://admin:password@192.168.1.100:554/stream"
  ```

* **Lưu video kết quả phân tích:**
  ```bash
  python app/main.py --source "video.mp4" --save-output --output-dir "outputs/"
  ```

---

## 🧪 Kiểm thử Tự động (Testing)

Dự án được bảo vệ chặt chẽ bởi bộ Unit Test:

```bash
# Chạy toàn bộ tests với pytest
pytest

# Hoặc chạy thông qua script:
python scripts/run_tests.py
```

---

## 📄 License
Dự án được phát triển cho mục đích nghiên cứu và ứng dụng giám sát an ninh thông minh.
