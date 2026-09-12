"""
Sinh ảnh và video demo suy luận — sản phẩm cuối theo yêu cầu đề tài 4.

Khác với các hình phân tích trong notebook (vốn để mổ xẻ lỗi), file này dựng
demo theo đúng góc nhìn NGƯỜI DÙNG hệ thống giám sát an toàn: mỗi khung hình
hiện số người có mũ / không mũ, và bật cảnh báo khi phát hiện đầu trần.

Dùng đúng cấu hình triển khai đã chọn ở notebook 02 (conf/iou từ best_config.json),
KHÔNG dùng conf=0.001 như lúc đo mAP — vì đây là mô phỏng vận hành thật.

Cách chạy:
    python demo_inference.py                    # 20 ảnh test -> video + ảnh rời
    python demo_inference.py --n 40             # nhiều khung hơn
    python demo_inference.py --video clip.mp4   # xử lý một video có sẵn
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from config import ARTIFACT_DIR, CLASS_NAMES, SEED, YOLO_DIR  # noqa: E402

DEMO_DIR = ARTIFACT_DIR / "demo"

# Màu theo nghĩa nghiệp vụ, không phải màu trang trí:
#   xanh lá = an toàn (có mũ) · đỏ = vi phạm (đầu trần) · xanh dương = người
COLORS = {
    "helmet": (46, 204, 113),
    "head": (231, 76, 60),
    "person": (52, 152, 219),
}
LABEL_VI = {"helmet": "Co mu", "head": "KHONG MU", "person": "Nguoi"}

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """Tìm một font TTF có sẵn. Cần TTF (không dùng font mặc định của PIL)
    vì font bitmap mặc định không đổi được cỡ chữ."""
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_frame(img_bgr: np.ndarray, result, font_box, font_hud) -> tuple[np.ndarray, int, int]:
    """Vẽ hộp phát hiện + bảng thông tin. Trả về (ảnh, số mũ, số đầu trần)."""
    img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    W, H = img.size

    n_helmet = n_head = 0
    boxes = result.boxes
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        x0, y0, x1, y1 = (float(v) for v in boxes.xyxy[i].tolist())
        name = CLASS_NAMES[cls_id]
        color = COLORS[name]

        if name == "helmet":
            n_helmet += 1
        elif name == "head":
            n_head += 1

        # hộp vi phạm vẽ dày hơn để bắt mắt ngay
        width = 4 if name == "head" else 2
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)

        tag = f"{LABEL_VI[name]} {conf:.2f}"
        tw = draw.textlength(tag, font=font_box)
        th = font_box.size + 4
        ty = max(0, y0 - th)
        draw.rectangle([x0, ty, x0 + tw + 6, ty + th], fill=color)
        draw.text((x0 + 3, ty + 1), tag, fill=(255, 255, 255), font=font_box)

    # ---- bảng thông tin trên cùng ----
    bar_h = font_hud.size + 14
    draw.rectangle([0, 0, W, bar_h], fill=(20, 20, 20))
    hud = f"Co mu: {n_helmet}    Khong mu: {n_head}"
    draw.text((10, 6), hud, fill=(255, 255, 255), font=font_hud)

    if n_head > 0:
        warn = f"!!! CANH BAO: {n_head} nguoi khong doi mu !!!"
        wl = draw.textlength(warn, font=font_hud)
        draw.rectangle([W - wl - 20, 0, W, bar_h], fill=COLORS["head"])
        draw.text((W - wl - 10, 6), warn, fill=(255, 255, 255), font=font_hud)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), n_helmet, n_head


def pick_demo_images(n: int) -> list[Path]:
    """Chọn ảnh demo có chủ đích: ưu tiên ảnh CÓ đầu trần (đó mới là tình huống
    hệ thống sinh ra để bắt), trộn thêm ảnh chỉ có mũ để thấy cả hai trạng thái."""
    import pandas as pd

    test_dir = YOLO_DIR / "images" / "test"
    all_imgs = {p.stem: p for p in sorted(test_dir.iterdir()) if p.is_file()}

    ann_path = ARTIFACT_DIR / "annotations.parquet"
    if not ann_path.exists():
        return list(all_imgs.values())[:n]

    ann = pd.read_parquet(ann_path)
    test = ann[ann["split"] == "test"]
    per_img = test.groupby(["stem", "class"]).size().unstack(fill_value=0)
    if "head" not in per_img:
        per_img["head"] = 0

    rng = np.random.default_rng(SEED)
    # 70% ảnh có đầu trần, 30% ảnh toàn mũ
    n_head = int(n * 0.7)
    has_head = per_img[per_img["head"] >= 1].index.tolist()
    no_head = per_img[per_img["head"] == 0].index.tolist()

    pick = []
    if has_head:
        pick += list(rng.choice(has_head, size=min(n_head, len(has_head)), replace=False))
    remain = n - len(pick)
    if remain > 0 and no_head:
        pick += list(rng.choice(no_head, size=min(remain, len(no_head)), replace=False))

    return [all_imgs[s] for s in pick if s in all_imgs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="số ảnh demo (chế độ ảnh)")
    ap.add_argument("--video", type=str, default=None, help="đường dẫn video có sẵn")
    ap.add_argument("--hold", type=float, default=1.5, help="giây hiển thị mỗi ảnh")
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    from ultralytics import YOLO

    cfg = json.load(open(ARTIFACT_DIR / "best_config.json", encoding="utf-8"))
    model = YOLO(cfg["weights"])
    conf, iou, imgsz = cfg["conf_deploy"], cfg["iou_deploy"], cfg["imgsz"]
    print(f"Cấu hình triển khai: conf={conf}  iou={iou}  imgsz={imgsz}")

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    font_box = load_font(16)
    font_hud = load_font(22)
    out_path = DEMO_DIR / "demo_suy_luan.mp4"
    writer = None
    frame_size: tuple[int, int] | None = None
    total_head = 0

    if args.video:
        # ---- chế độ video có sẵn ----
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"Không mở được video: {args.video}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            r = model.predict(frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
            drawn, _, nh = draw_frame(frame, r, font_box, font_hud)
            total_head += nh
            if writer is None:
                h, w = drawn.shape[:2]
                writer = cv2.VideoWriter(str(out_path),
                                         cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
            writer.write(drawn)
            idx += 1
            if idx % 50 == 0:
                print(f"  đã xử lý {idx} khung")
        cap.release()
        print(f"Tổng {idx} khung hình")
    else:
        # ---- chế độ ghép ảnh test thành video ----
        imgs = pick_demo_images(args.n)
        print(f"Đã chọn {len(imgs)} ảnh demo từ tập test")
        hold_frames = max(1, int(args.hold * args.fps))

        for k, p in enumerate(imgs, 1):
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            r = model.predict(frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
            drawn, nhel, nhead = draw_frame(frame, r, font_box, font_hud)
            total_head += nhead

            # lưu ảnh rời để chèn vào slide/báo cáo
            cv2.imwrite(str(DEMO_DIR / f"demo_{k:02d}_{p.stem}.jpg"), drawn)

            if writer is None:
                h, w = drawn.shape[:2]
                frame_size = (w, h)   # mọi khung sau phải khớp kích thước này
                writer = cv2.VideoWriter(str(out_path),
                                         cv2.VideoWriter_fourcc(*"mp4v"), args.fps, frame_size)
            # ảnh trong bộ dữ liệu lệch nhau vài pixel (415 vs 416) -> phải ép về
            # cùng kích thước, nếu không VideoWriter âm thầm bỏ qua khung sai cỡ
            if (drawn.shape[1], drawn.shape[0]) != frame_size:
                drawn = cv2.resize(drawn, frame_size)
            for _ in range(hold_frames):
                writer.write(drawn)
            print(f"  [{k:>2}/{len(imgs)}] {p.stem}: {nhel} có mũ, {nhead} không mũ")

    if writer is not None:
        writer.release()
        print(f"\nVideo : {out_path}")
    print(f"Ảnh rời: {DEMO_DIR}/demo_*.jpg")
    print(f"Tổng số lượt phát hiện đầu trần: {total_head}")


if __name__ == "__main__":
    main()
