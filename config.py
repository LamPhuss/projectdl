"""
Cấu hình tập trung cho Đề tài 4 — Phát hiện đối tượng (YOLO).

Mọi notebook đều import từ đây để đảm bảo:
  - cùng một seed  -> kết quả tái lập được
  - cùng một cấu trúc thư mục -> README chạy lại được
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# 1. Thư mục
# --------------------------------------------------------------------------
# PROJECT_ROOT = thư mục chứa data/, runs/, artifacts/, logs/.
# Hỗ trợ cả hai bố cục: file này nằm trong src/ (như README mô tả) hoặc nằm
# phẳng ở gốc dự án. Không hard-code parents[1] vì bố cục phẳng sẽ trỏ sai ra
# ngoài dự án và mọi đường dẫn dữ liệu đều lệch.
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent if _HERE.name == "src" else _HERE

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # dữ liệu gốc (VOC XML + ảnh), KHÔNG chỉnh sửa
YOLO_DIR = DATA_DIR / "yolo"        # dữ liệu đã chuẩn hóa về định dạng YOLO
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"   # bảng số liệu, hình cho báo cáo
RUNS_DIR = PROJECT_ROOT / "runs"    # output của ultralytics
LOG_DIR = PROJECT_ROOT / "logs"     # nhật ký thí nghiệm (CSV)
CONFIG_DIR = PROJECT_ROOT / "configs"

for _d in (DATA_DIR, RAW_DIR, YOLO_DIR, ARTIFACT_DIR, RUNS_DIR, LOG_DIR, CONFIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATA_YAML = CONFIG_DIR / "helmet.yaml"
EXPERIMENT_LOG = LOG_DIR / "experiment_log.csv"

# --------------------------------------------------------------------------
# 2. Bài toán
# --------------------------------------------------------------------------
# Dataset: Safety Helmet Detection (andrewmvd/hard-hat-detection) — 5.000 ảnh,
# nhãn VOC XML, 3 lớp. Đây là bài toán "phát hiện đội mũ bảo hiểm" trong danh
# mục đề tài, giới hạn <= 5.000 ảnh theo yêu cầu.
CLASS_NAMES = ["helmet", "head", "person"]
CLASS_NAMES_VI = {"helmet": "có đội mũ", "head": "đầu trần", "person": "người (toàn thân)"}
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

KAGGLE_DATASET = "andrewmvd/hard-hat-detection"

# --------------------------------------------------------------------------
# 3. Chia tập & seed
# --------------------------------------------------------------------------
SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLITS = ("train", "val", "test")

# Ngưỡng diện tích theo quy ước COCO (đơn vị: pixel^2, tính trên ảnh gốc)
AREA_SMALL = 32 ** 2      # < 1024  -> small
AREA_MEDIUM = 96 ** 2     # 1024..9216 -> medium ; > 9216 -> large


def size_bucket(area_px: float) -> str:
    """Phân loại kích thước hộp theo quy ước COCO."""
    if area_px < AREA_SMALL:
        return "small"
    if area_px < AREA_MEDIUM:
        return "medium"
    return "large"


# --------------------------------------------------------------------------
# 4. Tái lập
# --------------------------------------------------------------------------
def set_seed(seed: int = SEED, deterministic: bool = True) -> int:
    """Cố định seed cho random / numpy / torch. Trả về seed đã dùng."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # cudnn.deterministic làm chậm ~10-15% nhưng cần cho tái lập
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed


def describe_environment() -> dict:
    """Thu thập thông tin môi trường để ghi vào báo cáo (phần tái lập)."""
    import platform
    import sys

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": SEED,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1
            )
    except ImportError:
        info["torch"] = None
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        info["ultralytics"] = None
    return info
