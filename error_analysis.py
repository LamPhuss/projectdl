"""
Phân tích lỗi định lượng cho bài toán phát hiện đối tượng.

Thuật toán ghép: GREEDY theo confidence giảm dần (không dùng Hungarian —
xem lý do trong note của 02_train_eval.ipynb §9). Đây cũng là cách tiếp cận
theo tinh thần TIDE (Bolya et al., 2020) — dùng 2 ngưỡng IoU để tách rạch
4 loại lỗi mà đề bài yêu cầu: nhầm lớp / định vị lệch / bỏ sót / phát hiện thừa.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 1. IoU
# --------------------------------------------------------------------------


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU giữa hai hộp dạng [x1,y1,x2,y2]. Giống hệt hàm đã dùng ở 02_train_eval.ipynb."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def iou_matrix(preds_xyxy: np.ndarray, gts_xyxy: np.ndarray) -> np.ndarray:
    """Ma trận IoU [n_pred x n_gt]. Rỗng thì trả về mảng đúng shape (0,0)/(n,0)/(0,n)."""
    n_p, n_g = len(preds_xyxy), len(gts_xyxy)
    M = np.zeros((n_p, n_g))
    for i in range(n_p):
        for j in range(n_g):
            M[i, j] = iou_xyxy(preds_xyxy[i], gts_xyxy[j])
    return M


# --------------------------------------------------------------------------
# 2. Ghép greedy 1 ảnh, phân loại lỗi
# --------------------------------------------------------------------------

# Hai ngưỡng IoU (tinh thần TIDE): dưới ngưỡng thấp coi như "không liên quan gì
# tới vật thể thật nào" (phát hiện thừa); giữa hai ngưỡng là "có nhìn thấy vật
# thể nhưng vẽ hộp không đủ khít" (định vị lệch); trên ngưỡng cao mới xét nhãn.
IOU_LOW = 0.10
IOU_HIGH = 0.50

ERROR_TYPES = ["dung", "nham_lop", "dinh_vi_lech", "bo_sot", "phat_hien_thua"]


def match_one_image(
    pred_boxes: np.ndarray,   # [n_pred, 4] xyxy, pixel tuyệt đối
    pred_scores: np.ndarray,  # [n_pred]
    pred_classes: np.ndarray, # [n_pred] int
    gt_boxes: np.ndarray,     # [n_gt, 4] xyxy
    gt_classes: np.ndarray,   # [n_gt] int
    gt_size_bucket: list[str],# [n_gt] "small"/"medium"/"large"
    iou_low: float = IOU_LOW,
    iou_high: float = IOU_HIGH,
) -> pd.DataFrame:
    """Ghép greedy trên MỘT ảnh, trả về bảng chi tiết từng dòng lỗi/đúng.

    Quy trình (đúng thứ tự, greedy theo confidence giảm dần):
      1. Duyệt dự đoán theo điểm tin cậy giảm dần.
      2. Với mỗi dự đoán, tìm hộp thật CHƯA BỊ CHIẾM có IoU cao nhất (bất kể lớp).
      3. IoU tốt nhất >= iou_high:
           - đúng lớp  -> "dung" (TP), hộp thật đó bị chiếm
           - sai lớp   -> "nham_lop", hộp thật đó VẪN bị chiếm (đây là lời giải
             thích hợp lý nhất cho dự đoán này, dù sai nhãn)
      4. iou_low <= IoU tốt nhất < iou_high -> "dinh_vi_lech"; hộp thật đó CŨNG
         bị chiếm (dự đoán này đã là lời giải thích cho hộp thật đó, dù chưa
         khít — nếu không chiếm, hộp thật sẽ bị đếm trùng thêm một lần thành
         "bỏ sót" ở bước 6, dù thực chất đã có một dự đoán "nhận" nó rồi)
      5. IoU tốt nhất < iou_low (hoặc không còn hộp thật nào) -> "phat_hien_thua"
      6. Sau khi duyệt hết dự đoán: hộp thật nào chưa từng bị chiếm -> "bo_sot"
    """
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    rows = []
    claimed = np.zeros(n_gt, dtype=bool)

    order = np.argsort(-pred_scores)
    for pi in order:
        if n_gt == 0:
            rows.append(dict(kind="phat_hien_thua", pred_idx=int(pi),
                              gt_idx=-1, iou=0.0, pred_class=int(pred_classes[pi]),
                              gt_class=-1, score=float(pred_scores[pi]),
                              size_bucket=None))
            continue

        ious = np.array([iou_xyxy(pred_boxes[pi], gt_boxes[j]) if not claimed[j] else -1.0
                         for j in range(n_gt)])
        best_j = int(np.argmax(ious))
        best_iou = float(ious[best_j])

        if best_iou < 0:  # mọi hộp thật đều đã bị chiếm hết
            rows.append(dict(kind="phat_hien_thua", pred_idx=int(pi),
                              gt_idx=-1, iou=0.0, pred_class=int(pred_classes[pi]),
                              gt_class=-1, score=float(pred_scores[pi]),
                              size_bucket=None))
        elif best_iou >= iou_high:
            same_class = int(pred_classes[pi]) == int(gt_classes[best_j])
            claimed[best_j] = True
            rows.append(dict(
                kind="dung" if same_class else "nham_lop",
                pred_idx=int(pi), gt_idx=best_j, iou=best_iou,
                pred_class=int(pred_classes[pi]), gt_class=int(gt_classes[best_j]),
                score=float(pred_scores[pi]), size_bucket=gt_size_bucket[best_j],
            ))
        elif best_iou >= iou_low:
            # Cũng phải "chiếm" hộp thật — nếu không, hộp này sẽ bị đếm lần nữa
            # thành "bỏ sót" ở bước 6, dù đã có một dự đoán giải thích cho nó rồi.
            claimed[best_j] = True
            rows.append(dict(
                kind="dinh_vi_lech", pred_idx=int(pi), gt_idx=best_j, iou=best_iou,
                pred_class=int(pred_classes[pi]), gt_class=int(gt_classes[best_j]),
                score=float(pred_scores[pi]), size_bucket=gt_size_bucket[best_j],
            ))
        else:
            rows.append(dict(kind="phat_hien_thua", pred_idx=int(pi),
                              gt_idx=-1, iou=best_iou, pred_class=int(pred_classes[pi]),
                              gt_class=-1, score=float(pred_scores[pi]), size_bucket=None))

    for j in range(n_gt):
        if not claimed[j]:
            rows.append(dict(kind="bo_sot", pred_idx=-1, gt_idx=j, iou=0.0,
                              pred_class=-1, gt_class=int(gt_classes[j]),
                              score=None, size_bucket=gt_size_bucket[j]))

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. Tổng hợp & bảng thống kê
# --------------------------------------------------------------------------


def summarize_errors(all_rows: pd.DataFrame) -> pd.DataFrame:
    """Bảng đếm số lượng mỗi loại lỗi, theo tổng và theo size_bucket."""
    tab = pd.crosstab(
        all_rows["size_bucket"].fillna("(không áp dụng)"),
        all_rows["kind"],
    )
    for k in ERROR_TYPES:
        if k not in tab.columns:
            tab[k] = 0
    tab = tab[ERROR_TYPES]
    tab.loc["TỔNG"] = tab.sum(axis=0)
    return tab


def error_rate_by_class(all_rows: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    """Với mỗi lớp thật, tỉ lệ bị bỏ sót / định vị lệch / nhầm lớp là bao nhiêu."""
    rows = []
    for ci, name in enumerate(class_names):
        sub = all_rows[(all_rows["gt_class"] == ci)]
        n = len(sub)
        if n == 0:
            continue
        counts = sub["kind"].value_counts()
        rows.append({
            "class": name,
            "n_gt_involved": n,
            **{k: int(counts.get(k, 0)) for k in ["dung", "nham_lop", "dinh_vi_lech", "bo_sot"]},
        })
    df = pd.DataFrame(rows).set_index("class")
    for k in ["dung", "nham_lop", "dinh_vi_lech", "bo_sot"]:
        df[f"{k}_%"] = (df[k] / df["n_gt_involved"] * 100).round(1)
    return df
