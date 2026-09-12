"""
Nhật ký thí nghiệm — bằng chứng cho phần chấm "Thí nghiệm và phân tích".

Mọi lượt chạy (train / val / quét ngưỡng / đo tốc độ) đều gọi `log_run(...)`,
kết quả được nối thêm (append) vào logs/experiment_log.csv.
"""
from __future__ import annotations

import platform
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import EXPERIMENT_LOG

# Thứ tự cột cố định để file CSV luôn đọc được, kể cả khi thiếu trường
COLUMNS = [
    "timestamp",
    "run_id",
    "stage",          # train | val | sweep_conf | sweep_iou | speed | ...
    "model",
    "dataset",
    "split",
    "imgsz",
    "epochs",
    "batch",
    "seed",
    "conf",
    "iou_nms",
    "mAP50",
    "mAP50_95",
    "precision",
    "recall",
    "f1",
    "fps",
    "ms_per_img",
    "train_time_min",
    "gpu",
    "notes",
]


def log_run(path: Path = EXPERIMENT_LOG, **fields) -> pd.DataFrame:
    """Ghi một dòng vào nhật ký thí nghiệm và trả về 5 dòng cuối để xem nhanh."""
    row = {c: fields.pop(c, None) for c in COLUMNS}
    row["timestamp"] = row["timestamp"] or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if row["gpu"] is None:
        try:
            import torch

            row["gpu"] = (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()
            )
        except ImportError:
            row["gpu"] = platform.processor()

    # trường phụ (nếu có) được gộp vào cột notes để không mất thông tin
    if fields:
        extra = "; ".join(f"{k}={v}" for k, v in fields.items())
        row["notes"] = f"{row['notes']}; {extra}" if row["notes"] else extra

    df_new = pd.DataFrame([row], columns=COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    df_new.to_csv(path, mode="a", header=header, index=False, encoding="utf-8")
    return read_log(path).tail(5)


def read_log(path: Path = EXPERIMENT_LOG) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def metrics_from_ultralytics(metrics) -> dict:
    """Rút các chỉ số chính từ đối tượng trả về của `model.val()`."""
    box = metrics.box
    out = {
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }
    p, r = out["precision"], out["recall"]
    out["f1"] = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    speed = getattr(metrics, "speed", None)
    if isinstance(speed, dict):
        total_ms = sum(v for v in speed.values() if isinstance(v, (int, float)))
        out["ms_per_img"] = round(total_ms, 3)
        out["fps"] = round(1000.0 / total_ms, 2) if total_ms > 0 else None
    return out


def per_class_table(metrics, class_names: list[str]) -> pd.DataFrame:
    """Bảng AP / P / R theo từng lớp.

    Lưu ý: ultralytics chỉ trả mảng chỉ số cho các lớp THỰC SỰ xuất hiện trong
    tập đánh giá, kèm `box.ap_class_index` để ánh xạ. Ta dùng chỉ mục này thay vì
    giả định mảng đã đúng thứ tự — nếu một lớp vắng mặt, giá trị sẽ là NaN.
    """
    box = metrics.box
    idx = [int(i) for i in box.ap_class_index]
    rows = []
    for ci, name in enumerate(class_names):
        if ci in idx:
            j = idx.index(ci)
            rows.append(
                {
                    "class": name,
                    "AP50": round(float(box.ap50[j]), 4),
                    "AP50_95": round(float(box.ap[j]), 4),
                    "precision": round(float(box.p[j]), 4),
                    "recall": round(float(box.r[j]), 4),
                }
            )
        else:
            rows.append(
                {"class": name, "AP50": pd.NA, "AP50_95": pd.NA,
                 "precision": pd.NA, "recall": pd.NA}
            )
    df = pd.DataFrame(rows)
    df["F1"] = (
        2 * df["precision"] * df["recall"] / (df["precision"] + df["recall"]).replace(0, pd.NA)
    ).round(4)
    return df
