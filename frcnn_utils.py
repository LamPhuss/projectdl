"""
Tiện ích cho Faster R-CNN (torchvision) — dùng lại đúng cây thư mục
`data/yolo/{images,labels}/{split}` đã tạo ở notebook 01, không tải/ghi dữ liệu mới.

Quy ước nhãn của torchvision detection models: id 0 dành riêng cho "background",
nên lớp thật đánh số từ 1. Ba lớp helmet/head/person (id 0/1/2 trong YOLO) được
dịch thành 1/2/3 khi đưa vào Faster R-CNN — hàm `yolo_id_to_frcnn_id` xử lý việc này.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from torchvision.transforms import functional as TF


def yolo_id_to_frcnn_id(yolo_id: int) -> int:
    return yolo_id + 1


# --------------------------------------------------------------------------
# 1. Dataset
# --------------------------------------------------------------------------


class YoloDetectionDataset(Dataset):
    """Đọc ảnh + nhãn YOLO (.txt, toạ độ chuẩn hóa) và trả về định dạng
    torchvision: (ảnh tensor [0,1], dict{boxes: xyxy tuyệt đối, labels}).

    `augment=True` bật tăng cường dữ liệu cho tập TRAIN. Mục đích: thu hẹp
    khoảng cách với pipeline của YOLOv8n ở NB02 — nếu Faster R-CNN train "trần"
    (không augment) trong khi YOLO có cả bộ augment đầy đủ, phép so sánh
    one-stage vs two-stage không còn công bằng.

    ĐỐI CHIẾU với khối AUGMENT của YOLOv8n (NB02 §0) — cái gì khớp, cái gì không:

      | Phép        | YOLOv8n     | Ở đây     | Ghi chú                          |
      |-------------|-------------|-----------|----------------------------------|
      | fliplr      | 0.5         | 0.5  ✔    | khớp; hộp được lật theo ảnh      |
      | hsv jitter  | h/s/v jitter| ColorJitter ~✔ | tương đương về ý, khác cài đặt |
      | scale       | 0.5         | ✘         | chưa cài (cần biến đổi affine hộp)|
      | translate   | 0.1         | ✘         | chưa cài                          |
      | mosaic      | 1.0         | ✘         | đặc thù YOLO, không port sang     |

    Vậy đây là **thu hẹp một phần**, KHÔNG phải đồng bộ hoàn toàn — vẫn phải nêu
    rõ trong báo cáo rằng hai pipeline chưa tương đương tuyệt đối.
    """

    def __init__(self, yolo_dir: Path, split: str, augment: bool = False):
        self.img_dir = Path(yolo_dir) / "images" / split
        self.lbl_dir = Path(yolo_dir) / "labels" / split
        self.augment = augment
        self.img_paths = sorted(
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        # ColorJitter chỉ đổi màu pixel, KHÔNG đụng tới toạ độ hộp -> an toàn,
        # không cần biến đổi nhãn kèm theo.
        self._jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.7, hue=0.015)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        lbl_path = self.lbl_dir / f"{img_path.stem}.txt"
        boxes, labels = [], []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                if not line.strip():
                    continue
                cid, cx, cy, w, h = line.split()
                cid = int(cid)
                cx, cy, w, h = (float(v) for v in (cx, cy, w, h))
                x0 = (cx - w / 2) * W
                y0 = (cy - h / 2) * H
                x1 = (cx + w / 2) * W
                y1 = (cy + h / 2) * H
                # torchvision yêu cầu box hợp lệ (x1>x0, y1>y0); dữ liệu đã được
                # data_utils.voc_to_yolo_lines() lọc hộp suy biến từ NB01, nhưng
                # vẫn chốt chặn lại ở đây cho chắc.
                if x1 <= x0 or y1 <= y0:
                    continue
                boxes.append([x0, y0, x1, y1])
                labels.append(yolo_id_to_frcnn_id(cid))

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        if self.augment:
            # (a) Lật ngang — tương đương fliplr=0.5 của YOLOv8n.
            #     Lật ảnh thì PHẢI lật hộp theo, nếu không nhãn sẽ trỏ sai chỗ:
            #     điểm x=x0 sau khi lật nằm ở W-x0, và vì lật làm đảo trái/phải
            #     nên cạnh trái mới = W - cạnh phải cũ.
            if torch.rand(1).item() < 0.5:
                img = TF.hflip(img)
                if boxes_t.numel() > 0:
                    x0 = boxes_t[:, 0].clone()
                    x1 = boxes_t[:, 2].clone()
                    boxes_t[:, 0] = W - x1
                    boxes_t[:, 2] = W - x0
            # (b) Nhiễu màu — tương đương hsv_h/s/v của YOLOv8n. Không đụng toạ độ.
            img = self._jitter(img)

        target = {
            "boxes": boxes_t,
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "stem": img_path.stem,
        }
        img_tensor = TF.to_tensor(img)  # [0,1], CHW — Faster R-CNN tự chuẩn hoá bên trong
        return img_tensor, target


def collate_fn(batch):
    """Mỗi ảnh có số hộp khác nhau -> không stack được thành 1 tensor, giữ dạng list/tuple."""
    return tuple(zip(*batch))


# --------------------------------------------------------------------------
# 2. Dựng mô hình
# --------------------------------------------------------------------------


def build_faster_rcnn(num_real_classes: int, imgsz: int = 640):
    """Nạp Faster R-CNN tiền huấn luyện COCO, thay đầu ra cho đúng số lớp của ta.

    num_real_classes: số lớp THẬT (3, không tính background) — hàm tự +1.

    `imgsz`: Faster R-CNN có bước resize NỘI BỘ (min_size/max_size trong
    GeneralizedRCNNTransform), mặc định 800/1333 — khác hẳn imgsz=640 dùng cho
    YOLOv8n. Ép cả hai về min_size=max_size=imgsz để phép so sánh công bằng
    (cùng độ phân giải đầu vào thật sự, không phải chỉ cùng con số truyền vào).
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights, min_size=imgsz, max_size=imgsz,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_real_classes + 1)
    return model


# --------------------------------------------------------------------------
# 3. Vòng lặp train / eval
# --------------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, device) -> dict[str, float]:
    model.train()
    totals = {}
    n_batches = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items() if k != "stem"} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        loss.backward()
        optimizer.step()

        for k, v in loss_dict.items():
            totals[k] = totals.get(k, 0.0) + float(v.item())
        totals["total"] = totals.get("total", 0.0) + float(loss.item())
        n_batches += 1
    return {k: v / n_batches for k, v in totals.items()}


@torch.no_grad()
def collect_predictions(model, loader, device, score_thresh: float = 0.0):
    """Chạy suy luận trên cả loader, trả về list các dict để đưa vào torchmetrics
    MeanAveragePrecision, cùng dict tra theo stem để dùng lại ở notebook 04-style
    phân tích nếu cần.
    """
    model.eval()
    preds_all, targets_all, by_stem = [], [], {}
    for images, targets in loader:
        images_dev = [img.to(device) for img in images]
        outputs = model(images_dev)
        for out, tgt, img_t in zip(outputs, targets, images):
            keep = out["scores"] >= score_thresh
            p = {
                "boxes": out["boxes"][keep].cpu(),
                "scores": out["scores"][keep].cpu(),
                "labels": out["labels"][keep].cpu(),
            }
            t = {"boxes": tgt["boxes"], "labels": tgt["labels"]}
            preds_all.append(p)
            targets_all.append(t)
            by_stem[tgt["stem"]] = p
    return preds_all, targets_all, by_stem


@torch.no_grad()
def measure_fps_frcnn(model, device, imgsz: int = 640, n_warmup: int = 20, n_iter: int = 100) -> dict:
    """Đo tốc độ suy luận Faster R-CNN — cùng phương pháp measure_fps() của
    02_train_eval.ipynb (batch=1, warm-up, đồng bộ CUDA) để so sánh công bằng.
    """
    model.eval()
    dummy = torch.rand(3, imgsz, imgsz, device=device)
    for _ in range(n_warmup):
        model([dummy])
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        model([dummy])
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n_iter * 1000
    return {"ms_per_img": round(ms, 3), "fps": round(1000 / ms, 1)}
