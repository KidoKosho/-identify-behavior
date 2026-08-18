"""
Script tự động tải models từ Hugging Face Hub về thư mục models/ cục bộ.

Sử dụng:
    python scripts/download_models.py --repo-id <username>/<model-repo-name>
"""

import os
import argparse
import sys
from pathlib import Path

DEFAULT_REPO_ID = "KidoKosho/identify-behavior-models"

def download_models(repo_id: str = DEFAULT_REPO_ID, token: str = None):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[!] Thư viện 'huggingface_hub' chưa được cài đặt. Đang cài đặt...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import snapshot_download

    root_dir = Path(__file__).resolve().parent.parent
    models_dir = root_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Đang tải weights & models từ Hugging Face: {repo_id} ...")
    try:
        download_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(root_dir),
            local_dir_use_symlinks=False,
            token=token
        )
        print(f"[✓] Tải thành công! Toàn bộ model đã sẵn sàng tại: {models_dir}")
    except Exception as e:
        print(f"[!] Lỗi khi tải models từ Hugging Face Hub: {e}")
        print("[*] Gợi ý: Hãy đảm bảo bạn đã upload model lên repo và truyền đúng --repo-id.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models from Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help=f"HF Repo ID (mặc định: {DEFAULT_REPO_ID})")
    parser.add_argument("--token", type=str, default=None, help="HF Token (nếu là Private repo)")

    args = parser.parse_args()
    download_models(repo_id=args.repo_id, token=args.token)
