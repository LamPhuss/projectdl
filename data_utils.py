"""
Tiện ích xử lý dữ liệu: đọc nhãn VOC, chuyển sang định dạng YOLO,
chia tập có phân tầng và thống kê mô tả.
"""
from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    AREA_MEDIUM,
    AREA_SMALL,
    CLASS_NAMES,
    CLASS_TO_ID,
    SEED,
    SPLIT_RATIOS,
    SPLITS,
    size_bucket,
)

# --------------------------------------------------------------------------
# 1. Đọc nhãn VOC
# --------------------------------------------------------------------------


def parse_voc_xml(xml_path: Path) -> dict:
    """Đọc một file annotation VOC.

    Trả về dict: {filename, width, height, objects: [{name, xmin, ymin, xmax, ymax}]}
    Các hộp bị lật ngược hoặc tràn biên được sửa lại tại đây (clamping).
    """
    root = ET.parse(xml_path).getroot()

    size = root.find("size")
    width = int(float(size.find("width").text))
    height = int(float(size.find("height").text))

    filename_node = root.find("filename")
    filename = filename_node.text if filename_node is not None else xml_path.stem

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)

        # sửa hộp lật ngược
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        if ymax < ymin:
            ymin, ymax = ymax, ymin
        # cắt về trong ảnh
        xmin = max(0.0, min(xmin, width - 1))
        ymin = max(0.0, min(ymin, height - 1))
        xmax = max(1.0, min(xmax, float(width)))
        ymax = max(1.0, min(ymax, float(height)))

        objects.append(
            {"name": name, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
        )

    return {
        "filename": filename,
        "stem": xml_path.stem,
        "width": width,
        "height": height,
        "objects": objects,
    }


def build_annotation_dataframe(records: list[dict]) -> pd.DataFrame:
    """Dàn phẳng danh sách annotation thành DataFrame mức-đối-tượng.

    Mỗi dòng = 1 bounding box, kèm sẵn các cột dẫn xuất phục vụ thống kê:
    diện tích tuyệt đối (px^2), diện tích tương đối, tỉ lệ khung, nhóm kích thước.
    """
    rows = []
    for rec in records:
        W, H = rec["width"], rec["height"]
        for obj in rec["objects"]:
            w = obj["xmax"] - obj["xmin"]
            h = obj["ymax"] - obj["ymin"]
            area = w * h
            rows.append(
                {
                    "stem": rec["stem"],
                    "img_w": W,
                    "img_h": H,
                    "class": obj["name"],
                    "xmin": obj["xmin"],
                    "ymin": obj["ymin"],
                    "xmax": obj["xmax"],
                    "ymax": obj["ymax"],
                    "box_w": w,
                    "box_h": h,
                    "area_px": area,
                    "area_frac": area / (W * H),
                    "aspect": w / h if h > 0 else np.nan,
                    "cx_norm": (obj["xmin"] + obj["xmax"]) / 2 / W,
                    "cy_norm": (obj["ymin"] + obj["ymax"]) / 2 / H,
                    "size_bucket": size_bucket(area),
                }
            )
    df = pd.DataFrame(rows)
    df["size_bucket"] = pd.Categorical(
        df["size_bucket"], categories=["small", "medium", "large"], ordered=True
    )
    return df


# --------------------------------------------------------------------------
# 2. Chuyển VOC -> YOLO
# --------------------------------------------------------------------------


def voc_to_yolo_lines(rec: dict, drop_degenerate_px: float = 1.0) -> tuple[list[str], int]:
    """Chuyển một annotation VOC thành các dòng nhãn YOLO.

    Định dạng YOLO: `<class_id> <cx> <cy> <w> <h>`, toàn bộ toạ độ đã chuẩn hóa
    về [0, 1] theo kích thước ảnh.

    Trả về (lines, n_dropped). Hộp có cạnh < `drop_degenerate_px` bị loại.
    """
    W, H = rec["width"], rec["height"]
    lines, dropped = [], 0
    for obj in rec["objects"]:
        if obj["name"] not in CLASS_TO_ID:
            dropped += 1
            continue
        w = obj["xmax"] - obj["xmin"]
        h = obj["ymax"] - obj["ymin"]
        if w < drop_degenerate_px or h < drop_degenerate_px:
            dropped += 1
            continue
        cx = (obj["xmin"] + obj["xmax"]) / 2 / W
        cy = (obj["ymin"] + obj["ymax"]) / 2 / H
        nw, nh = w / W, h / H
        # chốt chặn cuối: mọi giá trị phải nằm trong (0, 1]
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        nw, nh = min(nw, 1.0), min(nh, 1.0)
        lines.append(
            f"{CLASS_TO_ID[obj['name']]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
        )
    return lines, dropped


# --------------------------------------------------------------------------
# 3. Chia tập có phân tầng
# --------------------------------------------------------------------------


def stratification_key(rec: dict) -> str:
    """Khóa phân tầng cho một ảnh.

    Chia tập ngẫu nhiên thuần túy dễ làm lệch phân bố của lớp hiếm (`person`).
    Ta phân tầng theo: (lớp hiếm nhất xuất hiện trong ảnh, nhóm số lượng đối tượng)
    để 3 tập train/val/test có phân bố lớp và mật độ đối tượng tương đương nhau.
    """
    names = {o["name"] for o in rec["objects"] if o["name"] in CLASS_TO_ID}
    if "person" in names:
        tag = "person"
    elif "head" in names:
        tag = "head"
    elif "helmet" in names:
        tag = "helmet"
    else:
        tag = "empty"

    n = len(rec["objects"])
    if n == 0:
        bucket = "n0"
    elif n <= 2:
        bucket = "n1-2"
    elif n <= 5:
        bucket = "n3-5"
    elif n <= 10:
        bucket = "n6-10"
    else:
        bucket = "n11+"
    return f"{tag}|{bucket}"


def stratified_split(
    records: list[dict], ratios: dict = SPLIT_RATIOS, seed: int = SEED
) -> dict[str, list[str]]:
    """Chia danh sách ảnh thành train/val/test, phân tầng theo `stratification_key`."""
    rng = np.random.default_rng(seed)
    groups: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        groups[stratification_key(rec)].append(rec["stem"])

    split_stems: dict[str, list[str]] = {s: [] for s in SPLITS}
    for key in sorted(groups):                     # sort -> thứ tự tất định
        stems = sorted(groups[key])
        idx = rng.permutation(len(stems))
        stems = [stems[i] for i in idx]
        n = len(stems)
        n_train = int(round(n * ratios["train"]))
        n_val = int(round(n * ratios["val"]))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        split_stems["train"] += stems[:n_train]
        split_stems["val"] += stems[n_train : n_train + n_val]
        split_stems["test"] += stems[n_train + n_val :]

    for s in SPLITS:
        split_stems[s] = sorted(split_stems[s])
    return split_stems


# --------------------------------------------------------------------------
# 4. Ghi ra cây thư mục chuẩn của ultralytics
# --------------------------------------------------------------------------


def materialize_yolo_dataset(
    records_by_stem: dict[str, dict],
    split_stems: dict[str, list[str]],
    image_lookup: dict[str, Path],
    out_dir: Path,
    copy_mode: str = "symlink",
) -> pd.DataFrame:
    """Tạo cây thư mục `images/{split}` + `labels/{split}` mà ultralytics yêu cầu.

    copy_mode: 'symlink' (nhanh, tiết kiệm ổ đĩa) hoặc 'copy' (an toàn hơn trên Windows).
    Trả về DataFrame tóm tắt số ảnh / số hộp / số hộp bị loại theo từng tập.
    """
    summary = []
    for split, stems in split_stems.items():
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        for d in (img_dir, lbl_dir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

        n_boxes = n_dropped = n_empty = 0
        for stem in stems:
            src_img = image_lookup[stem]
            dst_img = img_dir / src_img.name
            if copy_mode == "symlink":
                try:
                    dst_img.symlink_to(src_img.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(src_img, dst_img)
            else:
                shutil.copy2(src_img, dst_img)

            lines, dropped = voc_to_yolo_lines(records_by_stem[stem])
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            n_boxes += len(lines)
            n_dropped += dropped
            n_empty += int(len(lines) == 0)

        summary.append(
            {
                "split": split,
                "n_images": len(stems),
                "n_boxes": n_boxes,
                "boxes_per_image": round(n_boxes / max(len(stems), 1), 2),
                "n_empty_images": n_empty,
                "n_boxes_dropped": n_dropped,
            }
        )
    return pd.DataFrame(summary).set_index("split").loc[list(SPLITS)]


def write_data_yaml(path: Path, dataset_root: Path, class_names: list[str] = CLASS_NAMES) -> Path:
    """Sinh file data.yaml cho ultralytics (dùng đường dẫn tuyệt đối cho chắc chắn)."""
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    content = (
        "# Sinh tự động bởi notebooks/01_data_prep.ipynb — không sửa tay\n"
        f"path: {dataset_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        f"nc: {len(class_names)}\n"
        "names:\n"
        f"{names_block}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# 5. Kiểm tra tính toàn vẹn
# --------------------------------------------------------------------------


def verify_dataset(out_dir: Path) -> pd.DataFrame:
    """Đối chiếu ảnh <-> nhãn và xác nhận mọi giá trị nằm trong [0, 1]."""
    rows = []
    for split in SPLITS:
        img_dir, lbl_dir = out_dir / "images" / split, out_dir / "labels" / split
        img_stems = {p.stem for p in img_dir.iterdir() if p.is_file() or p.is_symlink()}
        lbl_stems = {p.stem for p in lbl_dir.glob("*.txt")}

        bad_values = bad_class = 0
        for p in lbl_dir.glob("*.txt"):
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                cid = int(parts[0])
                vals = [float(v) for v in parts[1:]]
                if not (0 <= cid < len(CLASS_NAMES)):
                    bad_class += 1
                if any(v < 0 or v > 1 for v in vals) or len(vals) != 4:
                    bad_values += 1

        rows.append(
            {
                "split": split,
                "n_images": len(img_stems),
                "n_labels": len(lbl_stems),
                "img_without_label": len(img_stems - lbl_stems),
                "label_without_img": len(lbl_stems - img_stems),
                "bad_coord_lines": bad_values,
                "bad_class_ids": bad_class,
            }
        )
    return pd.DataFrame(rows).set_index("split")


# --------------------------------------------------------------------------
# 6. Bảng thống kê cho báo cáo
# --------------------------------------------------------------------------


def class_distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    """Bảng số hộp theo lớp × tập, kèm tỉ lệ phần trăm trong từng tập."""
    tab = pd.crosstab(df["class"], df["split"])
    tab = tab.reindex(index=CLASS_NAMES, columns=list(SPLITS), fill_value=0)
    tab["total"] = tab.sum(axis=1)
    pct = (tab[list(SPLITS)] / tab[list(SPLITS)].sum(axis=0) * 100).round(2)
    pct.columns = [f"{c}_%" for c in pct.columns]
    return pd.concat([tab, pct], axis=1)


def size_distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    """Bảng số hộp theo lớp × nhóm kích thước (small/medium/large, quy ước COCO)."""
    buckets = ["small", "medium", "large"]
    tab = pd.crosstab(df["class"], df["size_bucket"])
    # crosstab có thể bỏ qua nhóm không xuất hiện -> ép đủ 3 cột và đủ 3 lớp
    tab = tab.reindex(index=CLASS_NAMES, columns=buckets, fill_value=0)
    tab["total"] = tab[buckets].sum(axis=1)
    for c in ["small", "medium", "large"]:
        tab[f"{c}_%"] = (tab[c] / tab["total"].replace(0, np.nan) * 100).round(1)
    return tab
