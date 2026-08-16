# Hệ thống Phát hiện Vật thể & Sự kiện an ninh

Dự án này sử dụng YOLOv8 để phát hiện và theo track các vật thể trong video, bao gồm:
- Phát hiện khói/đám cháy (fire.pt)
- Phát hiện tai nạn (accident.pt)
- Phát hiện va chạm (collision detection)
- Phát hiện đánh nhau (fight detection)
- Theo dõi vận tốc và lưu dữ liệu track

## Cấu trúc dự án

```
e:\Code\python\test\
├── main.py              # Điểm vào chính, xử lý video
├── config.py            # Cấu hình toàn cục
├── run.py               # Các hàm tiện ích (IOU, centroid, vẽ)
├── models.py            # Tải và quản lý các model YOLO
├── detector.py          # Các hàm phát hiện (smoke, accident, collision, fight)
├── pipeline.py          # Xử lý luồng video, buffer, export dữ liệu
├── utils.py             # Hàm toán học, xuất tracking records
├── run_demo.py          # Chạy demo nhanh
├── requirements.txt     # Phụ thuộc Python
├── models/
│   ├── accident.pt      # Model tai nạn
│   └── fire.pt          # Model khói/đám cháy
├── imgout/              # Thư mục đầu ra (ảnh sự kiện, video)
└── README.md            # Tài liệu này
```

## Cài đặt

### 1. Yêu cầu hệ thống
- Python 3.8+
- GPU khuyến nghị (để chạy YOLO nhanh hơn)

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

Dependencies chính (từ `requirements.txt`):
- `ultralytics` - YOLOv8
- `opencv-python` - Xử lý ảnh
- `numpy` - Toán số
- `pandas` - Xử lý dữ liệu
- `torch` - PyTorch
- `yt-dlp` - Xử lý stream HLS
- `huggingface-hub` - Tải model

### 3. Model files
Dự án sử dụng 2 model đã training:
- `models/fire.pt` - Phát hiện khói/đám cháy
- `models/accident.pt` - Phát hiện tai nạn

File `yolov8n.pt` là model COCO mặc định của ultralytics (sẽ tự động tải khi chạy lần đầu).

## Cách chạy

### Chạy cơ bản
```bash
python main.py --invideo <đường_dẫn_video> --outimg ./imgout
```

### Các tham số quan trọng

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--invideo` / `--video` | Bắt buộc | Đường dẫn video hoặc URL HLS stream |
| `--outimg` / `--output` | `./imgout` | Thư mục lưu ảnh sự kiện |
| `--display` | `True` | Hiển thị video xử lý trong cửa sổ OpenCV |
| `--window-scale` | `1.0` | Phóng to/thu nhỏ cửa sổ hiển thị (0.5 = 50%) |

### Ví dụ thực tế

**Xử lý video file cục bộ:**
```bash
python main.py --invideo data/video.mp4 --outimg ./output_images
```

**Xử lý stream HLS/URL:**
```bash
python main.py --invideo "https://example.com/stream.m3u8" --outimg ./stream_output
```

**Chạy không hiển thị giao diện (chỉ lưu dữ liệu):**
```bash
python main.py --invideo video.mp4 --outimg ./imgout --display False
```

## Cấu hình

Mở file `config.py` để tùy chỉnh:

### Cấu hình phát hiện
```python
# Kích/tắt các loại phát hiện
ENABLE_PERSON_DETECTION = True       # Phát hiện người
ENABLE_COLLISION_FIGHT_DETECTION = True  # Phát hiện va chạm/đánh nhau
ENABLE_FIRE_DETECTION = True         # Phát hiện khói/đám cháy
ENABLE_ACCIDENT_DETECTION = True     # Phát hiện tai nạn

# Ngưỡng confidence
CONF_THRESHOLD = 0.22                # Ngưỡng cho COCO person/vehicle
SMOKE_CONF = 0.30                    # Ngưỡng cho smoke/fire detection
ACCIDENT_CONF = 0.55                 # Ngưỡng cho accident detection

# Params va chạm
IOU_COLLISION = 0.45                 # Ngưỡng IoU để coi là va chạm
IOU_FIGHT = 0.58                     # Ngưỡng IoU cho phát hiện đánh nhau

# Params tai nạn
ACCIDENT_MIN_AREA_RATIO = 0.016
ACCIDENT_MIN_WIDTH = 65
ACCIDENT_MIN_HEIGHT = 40

# Params đánh nhau
FIGHT_MIN_PERSONS = 2                # Số người tối thiểu để detect fight
FIGHT_VELOCITY_THRESHOLD = 10        # Ngưỡng tốc độ cho fight

# Cấu hình khác
PIXEL_TO_METER = 0.05                # Chuyển đổi pixel -> mét
TRACKING_EXPORT_ENABLED = True       # Xuất dữ liệu track
TRACKING_EXPORT_PATH = "./tracking_data.csv"
TRACKING_EXPORT_JSON = "./tracking_data.json"
OUTPUT_DIR = "./imgout"
```

## Kết quả

### Ảnh sự kiện (lưu vào `imgout/`)
Dự án sẽ tự động lưu các ảnh khi phát hiện:
- **khoi_XX_YYYY_videoName.jpg** - Sự kiện khói/đám cháy
- **accident_XX_YYYY_videoName.jpg** - Tai nạn
- **collision_XX_YYYY_videoName.jpg** - Va chạm xe cộ/người xe
- **taucar_XX_YYYY_videoName.jpg** - Va chạm tàu/xe cộ
- **tau_nao_XX_YYYY_videoName.jpg** - Va chạm tàu tàu
- **danhnhau_XX_YYYY_videoName.jpg** - Đánh nhau

### Dữ liệu track (lưu vào `tracking_data.csv` / `tracking_data.json`)
Bao gồm các cột: `frame_id`, `track_id`, `x_center`, `y_center`, `bbox_width`, `bbox_height`
- Dữ liệu có thể chuyển đổi sang mét bằng `PIXEL_TO_METER`
- Có thể xuất JSON để phân tích thêm

## Cách hoạt động

1. **Phát hiện frame**: Mỗi frame được xử lý qua các model YOLO
2. **Buffer smoke**: Phát hiện khói được lưu trong 5 frame liên tiếp, sau đó lưu frame có confidence cao nhất
3. **Buffer accident**: Tất cả frame có accident được lưu (để capture toàn bộ sự kiện)
4. **Track & Velocity**: Theo dõi các vật thể, tính vận tốc giữa các frame
5. **Phát hiện va chạm**: Sử dụng IoU và vận tốc để xác định va chạm thực sự
6. **Phát hiện fight**: Cần ít nhất 2 người, IoU cao và tốc độ đáng kể

## Troubleshooting

### Model không tải được
- Kiểm tra file `models/fire.pt` và `models/accident.pt` có tồn tại không
- Model COCO (`yolov8n.pt`) sẽ tự động tải từ HuggingFace nếu chưa có local

### Không phát hiện được vật thể
- Điều chỉnh `CONF_THRESHOLD` trong `config.py` (giảm nếu không phát hiện)
- Kiểm tra video có rõ không, ánh sáng đủ tốt không

### Export dữ liệu rỗng
- Đảm bảo `TRACKING_EXPORT_ENABLED = True` trong `config.py`
- Video có chứa vật thể được detect không?

### Cửa sổ hiển thị không xuất hiện
- Chạy trên hệ thống có GUI (không phải headless server)
- Sử dụng `--display False` nếu chạy trên server distante

## Phát triển thêm

### Thêm model mới
1. Đặt file `.pt` vào thư mục `models/`
2. Thêm đường dẫn vào `config.py`
3. Thêm hàm detect tương ứng trong `detector.py`
4. Xử lý kết quả trong `pipeline.py`

### Tùy chỉnh xuất dữ liệu
- Hàm `export_tracking_data()` trong `pipeline.py` có thể tùy chỉnh cột và định dạng
- `utils.py` chứa `pixel_to_meter()` và `export_tracking_records()` để xử lý dữ liệu

## Liên hệ

Nếu có câu hỏi hoặc cần hỗ trợ, vui lòng kiểm tra file `.env` hoặc liên hệ với người phát triển dự án.