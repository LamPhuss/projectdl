"""
Đo lại tốc độ suy luận một cách đáng tin cậy — thay thế phép đo cũ ở NB02/NB03.

VÌ SAO PHẢI VIẾT LẠI. Phép đo cũ cho hai kết quả không thể tin được:
  - YOLOv8n đo hai lần trong cùng NB02: 98,6 FPS rồi 256,1 FPS (chênh 2,6 lần)
  - YOLOv8n (3,0M tham số) và Faster R-CNN (41,3M tham số) cùng ra đúng 98,6 FPS

Hai dấu hiệu đó gợi ý phép đo đang bị chi phối bởi chi phí cố định (overhead
Python/CUDA launch) chứ không đo được thời gian tính toán thật.

BA CẢI TIẾN SO VỚI PHÉP ĐO CŨ:
  1. Chạy trong TIẾN TRÌNH RIÊNG, không xen giữa/ngay sau lúc train.
  2. Lặp lại NHIỀU LƯỢT ĐỘC LẬP -> báo cáo trung bình ± độ lệch chuẩn, thay vì
     một con số trần trụi không biết dao động bao nhiêu.
  3. PHÉP CHẨN ĐOÁN QUYẾT ĐỊNH: đo ở nhiều độ phân giải (416/640/960).
     - Nếu thời gian TĂNG theo độ phân giải -> đang đo được phép tính thật -> tin được.
     - Nếu thời gian GẦN NHƯ KHÔNG ĐỔI -> bị overhead chi phối -> phép đo vô nghĩa,
       và điều này giải thích luôn vì sao hai mô hình khác nhau lại ra cùng con số.

Cách chạy:
    python benchmark_fps.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from config import ARTIFACT_DIR, NUM_CLASSES, RUNS_DIR  # noqa: E402

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Bật tối ưu tốc độ — khác lúc train (train cần deterministic để tái lập,
# nhưng deterministic tắt cudnn.benchmark và làm méo phép đo tốc độ).
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

N_WARMUP = 50
N_ITER = 200
N_REPEATS = 3          # số lượt đo độc lập -> tính độ lệch chuẩn
IMGSZ_GRID = [416, 640, 960]


def _time_loop(fn, n_warmup: int, n_iter: int) -> float:
    """Trả về thời gian trung bình mỗi lần gọi (ms), đo bằng CUDA Event.

    CUDA Event chính xác hơn time.perf_counter() cho việc đo GPU vì nó ghi mốc
    thời gian NGAY TRÊN dòng lệnh GPU, không bị nhiễu bởi độ trễ đồng bộ hoá
    giữa CPU và GPU.
    """
    for _ in range(n_warmup):
        fn()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iter):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / n_iter
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    return (time.perf_counter() - t0) / n_iter * 1000


def bench(fn, label: str) -> dict:
    """Đo `n_repeats` lượt độc lập, trả về trung bình ± độ lệch chuẩn."""
    samples = [_time_loop(fn, N_WARMUP, N_ITER) for _ in range(N_REPEATS)]
    mean_ms = statistics.mean(samples)
    std_ms = statistics.stdev(samples) if len(samples) > 1 else 0.0
    print(f"  {label:<34} {mean_ms:7.3f} ± {std_ms:5.3f} ms   "
          f"({1000/mean_ms:6.1f} FPS)")
    return {"ms_mean": round(mean_ms, 3), "ms_std": round(std_ms, 3),
            "fps": round(1000 / mean_ms, 1), "samples_ms": [round(s, 3) for s in samples]}


def _global_warmup() -> None:
    """Làm nóng toàn cục TRƯỚC mọi phép đo.

    Lượt chạy đầu tiên cho thấy phép đo ĐẦU TIÊN trong cả phiên bị nhiễu nặng:
    imgsz=416 .predict() ra 5,660 ± 2,743 ms, trong khi các phép đo sau đó có
    độ lệch chuẩn chỉ ~0,01 ms. Nguyên nhân: lần gọi CUDA đầu tiên của cả tiến
    trình phải khởi tạo context, nạp kernel, cấp phát bộ nhớ — chi phí một lần
    này bị tính nhầm vào phép đo đầu tiên.

    Đây chính là thứ đã tạo ra "bí ẩn 640 nhanh hơn 416" ở NB02: mô hình nào
    được đo TRƯỚC sẽ gánh chi phí khởi tạo, nên trông chậm hơn một cách giả tạo.
    """
    if DEVICE.type != "cuda":
        return
    x = torch.rand(1, 3, 640, 640, device=DEVICE)
    conv = torch.nn.Conv2d(3, 32, 3, padding=1).to(DEVICE)
    with torch.no_grad():
        for _ in range(30):
            conv(x)
    torch.cuda.synchronize()


def main() -> None:
    print(f"Thiết bị: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Cấu hình đo: warmup={N_WARMUP}, iter={N_ITER}, số lượt lặp={N_REPEATS}")
    _global_warmup()
    print("Đã làm nóng toàn cục (khử chi phí khởi tạo CUDA khỏi phép đo đầu tiên)\n")

    results: dict[str, dict] = {}

    # ---------------- YOLOv8n ----------------
    from ultralytics import YOLO

    print("=" * 78)
    print("YOLOv8n — đo ở 3 độ phân giải để kiểm tra thời gian có tỉ lệ theo ảnh không")
    print("=" * 78)
    for imgsz in IMGSZ_GRID:
        ckpt = RUNS_DIR / f"yolov8n_{imgsz}_e80" / "weights" / "best.pt"
        if not ckpt.exists():
            print(f"  [bỏ qua] không thấy {ckpt}")
            continue
        m = YOLO(str(ckpt))
        dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        # (a) toàn bộ pipeline .predict() — gồm tiền xử lý, forward, NMS, hậu xử lý
        yolo_dev = 0 if DEVICE.type == "cuda" else "cpu"
        results[f"yolo{imgsz}_predict"] = bench(
            lambda: m.predict(dummy, imgsz=imgsz, device=yolo_dev, verbose=False),
            f"imgsz={imgsz}  .predict() đầy đủ")

        # (b) CHỈ forward của mạng — bóc hết phần Python bao quanh, để lộ ra
        #     phần tính toán thuần tuý trên GPU
        raw = m.model.to(DEVICE).eval()
        x = torch.rand(1, 3, imgsz, imgsz, device=DEVICE)
        with torch.no_grad():
            results[f"yolo{imgsz}_forward"] = bench(
                lambda: raw(x), f"imgsz={imgsz}  chỉ forward mạng")
        print()

    # ---------------- Faster R-CNN ----------------
    import frcnn_utils as fu

    print("=" * 78)
    print("Faster R-CNN — cùng phương pháp đo, cùng dải độ phân giải")
    print("=" * 78)
    for ckpt_name in ["best_aug.pt", "best.pt"]:
        ckpt = RUNS_DIR / "faster_rcnn" / ckpt_name
        if ckpt.exists():
            break
    else:
        print("  [bỏ qua] không thấy checkpoint Faster R-CNN nào")
        ckpt = None

    if ckpt is not None:
        print(f"  (dùng checkpoint: {ckpt.name})")
        for imgsz in IMGSZ_GRID:
            model = fu.build_faster_rcnn(NUM_CLASSES, imgsz=imgsz).to(DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            model.eval()
            img = torch.rand(3, imgsz, imgsz, device=DEVICE)
            with torch.no_grad():
                results[f"frcnn{imgsz}_forward"] = bench(
                    lambda: model([img]), f"imgsz={imgsz}  forward + NMS")
            del model
            torch.cuda.empty_cache()
        print()

    # ---------------- Chẩn đoán ----------------
    print("=" * 78)
    print("CHẨN ĐOÁN: thời gian có tỉ lệ theo độ phân giải không?")
    print("=" * 78)
    for prefix, name in [("yolo", "YOLOv8n (forward)"), ("frcnn", "Faster R-CNN (forward)")]:
        key_lo = f"{prefix}416_forward"
        key_hi = f"{prefix}960_forward"
        if key_lo in results and key_hi in results:
            lo, hi = results[key_lo]["ms_mean"], results[key_hi]["ms_mean"]
            ratio = hi / lo if lo > 0 else float("nan")
            # 960^2 / 416^2 = 5,33 lần số pixel. Ba mức thay vì hai, vì thực tế
            # không nhị phân "dùng được / không dùng được":
            if ratio < 1.5:
                verdict = "overhead ÁP ĐẢO (gần như không đo được phép tính)"
            elif ratio < 3.0:
                verdict = "overhead + tính toán lẫn lộn (một phần đo được)"
            else:
                verdict = "tính toán chiếm ưu thế (số đo phản ánh khối lượng tính)"
            print(f"  {name:<26} 416px={lo:6.3f}ms  960px={hi:6.3f}ms  "
                  f"tỉ lệ={ratio:4.2f}×  -> {verdict}")
    print("\n  Số pixel tăng 5,33 lần khi đi từ 416 lên 960.")
    print("  Tỉ lệ thời gian càng xa 5,33 thì phần chi phí cố định (khởi chạy CUDA")
    print("  kernel, xử lý Python) càng lấn át phần tính toán thật.")
    print()
    print("  LƯU Ý DIỄN GIẢI: overhead áp đảo KHÔNG có nghĩa số đo vô dụng.")
    print("  Nó vẫn là ĐỘ TRỄ THẬT mà hệ thống chịu khi chạy batch=1 — đúng kịch bản")
    print("  camera giám sát xử lý từng khung hình. Chỉ là nó KHÔNG dùng được để so")
    print("  sánh 'kiến trúc nào tính toán nhiều hơn'. Muốn so sánh khối lượng tính,")
    print("  hãy dùng số tham số / GFLOPs, hoặc đo lại với batch lớn để chi phí cố")
    print("  định được chia đều cho nhiều ảnh.")

    out = ARTIFACT_DIR / "fps_benchmark.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nĐã lưu: {out}")


if __name__ == "__main__":
    main()
