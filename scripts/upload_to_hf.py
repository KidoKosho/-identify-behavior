"""
Script đẩy toàn bộ models lên Hugging Face Hub.

Sử dụng:
    1. Cài đặt thư viện: pip install huggingface_hub
    2. Đăng nhập: huggingface-cli login  (hoặc truyền token qua --token)
    3. Chạy script:
       python scripts/upload_to_hf.py --repo-id <username>/<model-repo-name>
"""

import os
import argparse
import sys
from pathlib import Path

def upload_models(repo_id: str, token: str = None, private: bool = False):
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("[!] Thư viện 'huggingface_hub' chưa được cài đặt. Đang cài đặt...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)

    # 1. Tạo repository trên Hugging Face nếu chưa tồn tại
    print(f"[*] Đang kiểm tra/tạo repository: {repo_id} ...")
    try:
        create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True, token=token)
        print(f"[✓] Repository '{repo_id}' đã sẵn sàng!")
    except Exception as e:
        print(f"[!] Lỗi khi tạo repo (hoặc repo đã tồn tại): {e}")

    # 2. Danh sách file/folder cần upload
    root_dir = Path(__file__).resolve().parent.parent
    files_to_upload = [
        # Thư mục models
        ("models", "models"),
        # Các file model trọng số lớn ở root (nếu có)
        ("InceptionResNetV2_4class_fire_smoke_no_fire_balanced.h5", "InceptionResNetV2_4class_fire_smoke_no_fire_balanced.h5"),
        ("fire_smoke_detector.keras", "fire_smoke_detector.keras"),
        ("model.tflite", "model.tflite"),
    ]

    for local_path, repo_path in files_to_upload:
        full_local_path = root_dir / local_path
        if not full_local_path.exists():
            continue

        if full_local_path.is_dir():
            print(f"[*] Đang tải lên thư mục: {local_path} -> {repo_path} ...")
            api.upload_folder(
                folder_path=str(full_local_path),
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="model",
                token=token
            )
            print(f"[✓] Đã tải lên thư mục: {local_path}")
        else:
            print(f"[*] Đang tải lên file: {local_path} ({full_local_path.stat().st_size / 1e6:.2f} MB) ...")
            api.upload_file(
                path_or_fileobj=str(full_local_path),
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="model",
                token=token
            )
            print(f"[✓] Đã tải lên: {local_path}")

    print(f"\n[🎉] Hoàn tất! Tất cả models đã được tải lên: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload models to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True, help="HF Repo ID (ví dụ: KidoKosho/identify-behavior-models)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face Access Token (hoặc dùng huggingface-cli login trước)")
    parser.add_argument("--private", action="store_true", help="Tạo repo ở chế độ Private")

    args = parser.parse_args()
    upload_models(repo_id=args.repo_id, token=args.token, private=args.private)
