# Phát hiện đội mũ bảo hộ lao động Fine-tune YOLOv8n và phân tích lỗi

Bài tập lớn môn Học sâ

Fine-tune YOLOv8n cho bài toán phát hiện ba lớp đối tượng trên công trường xây dựng `helmet` (có mũ), `head` (đầu trần), `person` (người toàn thân), kèm phân tích lỗi định lượng và so sánh chéo với Faster R-CNN (two-stage) để kiểm chứng nguyên nhân gốc của các lỗi quan sát được.

## Kết quả chính

| Mô hình | mAP@0.5 | mAP@0.5:0.95 | Số tham số | ms/ảnh @640 |
|---|---|---|---|---|
| YOLOv8n (fine-tune COCO) | **0,6289** | **0,4132** | 3,0 triệu | 3,57 |
| Faster R-CNN (ResNet-50+FPN) | 0,6167 | 0,3866 | 41,3 triệu | 9,74 |

- Hai lớp phổ biến (`helmet`, `head`) đạt AP@0.5 > 0,91. Lớp `person` gần như sụp đổ (AP 0,0149) — xác nhận bằng bằng chứng chéo từ hai kiến trúc độc lập rằng nguyên nhân nằm ở **dữ liệu** (562 hộp train), không phải kiến trúc mô hình.
- Loại lỗi chi phối là **bỏ sót** (454 trường hợp), không phải nhầm lớp (chỉ 20 trường hợp, nhầm `head`↔`helmet` chỉ 3%).
- Chi tiết đầy đủ, biểu đồ và số liệu từng bước nằm trong 4 notebook (xem Mục 3).

## Dữ liệu

[Safety Helmet Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection) — 5.000 ảnh công trường, nhãn PASCAL VOC XML, giấy phép CC0.

## 1. Cài đặt

```bash
git clone <repo-url> && cd projectdl
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# PyTorch đúng bản CUDA của máy (ví dụ CUDA 12.1) — cài trước
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

## 2. Tải dữ liệu

**Cách A — tự động.** Đặt `kaggle.json` vào `~/.kaggle/`; notebook `01_data_prep.ipynb` tự tải qua `kagglehub`.

**Cách B — tải tay.** Giải nén vào đúng cấu trúc:

```
data/raw/
├── images/        # 5000 file .png
└── annotations/   # 5000 file .xml
```

`data/raw/` chỉ đọc theo quy ước — mọi biến đổi (chia tập, chuyển định dạng) ghi ra `data/yolo/`.

## 3. Chạy lại toàn bộ pipeline

Chạy 4 notebook theo đúng thứ tự:

| # | Notebook | Nội dung | Thời gian (GPU tầm trung) |
|---|---|---|---|
| 01 | `01_data_prep.ipynb` | VOC → YOLO, chia tập phân tầng, thống kê phân bố lớp/kích thước | ~5 phút |
| 02 | `02_train_eval.ipynb` | Fine-tune YOLOv8n, mAP, PR curve, quét ngưỡng conf/NMS, khảo sát độ phân giải 416/640/960 | ~1,5 giờ |
| 03 | `03_faster_rcnn.ipynb` | So sánh one-stage (YOLOv8n) vs two-stage (Faster R-CNN) | ~30 phút |
| 04 | `04_error_analysis.ipynb` | Phân loại lỗi định lượng (đúng/nhầm lớp/định vị lệch/bỏ sót/phát hiện thừa) + minh hoạ trên ảnh thật | ~5 phút |

Notebook 02 và 03 có cờ `FORCE_RETRAIN`. Đặt `False` để dùng lại checkpoint có sẵn trong `runs/` (vài phút thay vì hàng giờ) — đây là cách khuyến nghị nếu chỉ muốn đọc lại kết quả.

## 4. Cấu trúc thư mục

```
projectdl/
├── 01_data_prep.ipynb
├── 02_train_eval.ipynb
├── 03_faster_rcnn.ipynb
├── 04_error_analysis.ipynb
├── config.py            # đường dẫn, seed, tên lớp, ngưỡng kích thước COCO
├── data_utils.py         # đọc VOC, chuyển sang YOLO, chia tập phân tầng, bảng thống kê
├── error_analysis.py     # ghép greedy dự đoán↔nhãn thật, phân loại 5 nhóm lỗi
├── frcnn_utils.py        # dataset/augmentation/huấn luyện cho Faster R-CNN
├── benchmark_fps.py      # đo tốc độ suy luận độc lập (CUDA Events, có warm-up)
├── demo_inference.py     # sinh ảnh/video demo ở cấu hình triển khai thực tế
├── exp_log.py            # ghi nhật ký mọi lượt chạy ra CSV
├── configs/helmet.yaml   # sinh tự động bởi notebook 01
├── artifacts/            # bảng CSV + hình PNG dùng cho báo cáo (kết quả đã tính sẵn)
├── logs/experiment_log.csv
├── data/                 # không commit lên git — xem Mục 2
└── runs/                 # output huấn luyện của ultralytics — không commit lên git
```

## 5. Tái lập kết quả

- `SEED = 42` cố định cho `random`, `numpy`, `torch`, `cudnn.deterministic = True`.
- Chia tập dùng `np.random.default_rng(42)` với khoá phân tầng tất định → chạy lại ra đúng cùng một tập test; danh sách lưu ở `artifacts/splits.json`.
- Mọi lượt chạy được ghi vào `logs/experiment_log.csv`, mọi con số trong báo cáo đều truy được về một dòng cụ thể trong file này.

## 6. Nguồn tham khảo và phần tự viết

**Thư viện sử dụng:**
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — kiến trúc, vòng huấn luyện, `val()` tính mAP, các hình do thư viện tự sinh.
- `torchvision`: Faster R-CNN cho notebook 03.
- Trọng số tiền huấn luyện `yolov8n.pt` (COCO) do Ultralytics phát hành.

**Phần tự viết:** toàn bộ `data_utils.py`, `error_analysis.py`, `frcnn_utils.py`, `benchmark_fps.py`, `demo_inference.py`, thiết kế thí nghiệm và nội dung phân tích trong cả 4 notebook.
