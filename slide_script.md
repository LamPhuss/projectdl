# Kịch bản thuyết trình — Đề tài 4: Phát hiện đội mũ bảo hộ

**Thời lượng:** ~16 phút · **16 slide chính + 5 phụ lục**
**Quy ước:** ✂️ = cắt được nếu thiếu giờ · ⭐ = slide trọng tâm, không cắt

**Mạch truyện:** đặt câu hỏi từ dữ liệu (slide 2) → trả lời bằng số (slide 8) → đào sâu nguyên nhân (slide 13). Kể theo logic *"dữ liệu báo trước điều gì, kết quả xác nhận ra sao"*, không theo trình tự *"đã làm những gì"*.

**Về hình:** mọi slide đều ghi rõ tên file hình cần chèn — bạn tự chèn. **Về bảng:** mọi bảng đã ghi đầy đủ số liệu thật, copy thẳng được.

---

## CHƯƠNG 1 — VẤN ĐỀ & DỮ LIỆU

### Slide 1 — Bài toán

**NỘI DUNG SLIDE**

Tiêu đề: *Phát hiện đội mũ bảo hộ lao động — Fine-tune YOLO và phân tích lỗi*

📷 `artifacts/fig04_label_sanity_check.png` — cắt lấy 1-2 ảnh mẫu có hộp màu

| id | Lớp | Ý nghĩa |
|---|---|---|
| 0 | `helmet` | Đầu người **có** đội mũ |
| 1 | `head` | Đầu người **không** đội mũ |
| 2 | `person` | Người toàn thân |

Dữ liệu: Safety Helmet Detection — 5.000 ảnh · PASCAL VOC XML · giấy phép CC0

**NOTE NÓI** *(~40 giây)*

> Bài toán của em là giám sát an toàn lao động: phát hiện công nhân không đội mũ bảo hộ qua camera công trường. Ba lớp: có mũ, đầu trần, và người toàn thân.
>
> Có một điểm cần lưu ý ngay từ đầu vì nó xuyên suốt cả bài: lớp quan trọng nhất về mặt nghiệp vụ là `head` — đầu trần — vì đó chính là thứ hệ thống sinh ra để cảnh báo. Bỏ sót một cái đầu trần nguy hiểm hơn nhiều so với báo nhầm một lần.

---

### Slide 2 — Hai con số định hình cả đồ án ⭐

**NỘI DUNG SLIDE**

📷 `artifacts/fig01_class_distribution.png` (trái) · `artifacts/fig02_box_size_distribution.png` (phải)

Hai con số in cỡ lớn: **25,3×** và **53,9%**

**Bảng 1 — Phân bố lớp theo tập** *(số hộp)*

| Lớp | Train | Val | Test | Tổng | %Train | %Val | %Test |
|---|---|---|---|---|---|---|---|
| `helmet` | 13.196 | 2.860 | 2.910 | **18.966** | 74,21 | 73,96 | 75,55 |
| `head` | 4.025 | 901 | 859 | **5.785** | 22,63 | 23,30 | 22,30 |
| `person` | 562 | 106 | 83 | **751** | 3,16 | 2,74 | 2,15 |

**Bảng 2 — Phân bố kích thước hộp** *(quy ước COCO)*

| Lớp | small | medium | large | Tổng | %small | %medium | %large |
|---|---|---|---|---|---|---|---|
| `helmet` | 9.795 | 8.439 | 732 | 18.966 | 51,6 | 44,5 | 3,9 |
| `head` | 3.901 | 1.865 | 19 | 5.785 | **67,4** | 32,2 | 0,3 |
| `person` | 61 | 386 | 304 | 751 | 8,1 | 51,4 | **40,5** |
| **Toàn bộ** | | | | 25.502 | **53,9** | 41,9 | 4,1 |

Khối cuối slide, in đậm — **Ba dự đoán trước khi huấn luyện:**

| # | Từ đặc điểm | Dự đoán | Kiểm chứng ở |
|---|---|---|---|
| 1 | Mất cân bằng 25,3× | `person` sẽ có AP rất thấp | slide 8 |
| 2 | `head` 67,4% nhỏ | `head` sẽ khó hơn `helmet` | slide 8 |
| 3 | 53,9% vật nhỏ | Tăng độ phân giải sẽ cải thiện nhiều | slide 12 |

**NOTE NÓI** *(~70 giây)*

> Đây là slide quan trọng nhất của phần dữ liệu, vì hai con số ở đây cho phép em **dự đoán trước kết quả** ngay khi chưa huấn luyện dòng nào.
>
> Thứ nhất, **mất cân bằng lớp 25,3 lần**: `helmet` có 18.966 hộp, `person` chỉ 751. Mô hình sẽ thấy `helmet` nhiều gấp 25 lần, nên **dự đoán thứ nhất: `person` sẽ có AP rất thấp**. Đây là hệ quả gần như tất yếu của việc thiếu dữ liệu, không phải lỗi mô hình.
>
> Thứ hai, **53,9% vật thể thuộc nhóm nhỏ** theo quy ước COCO — cạnh dưới 32 pixel. Vật càng nhỏ càng ít pixel để mạng trích đặc trưng. Mà riêng lớp `head` thì tới **67,4% là nhỏ**, cao nhất trong ba lớp — nên **dự đoán thứ hai: `head` sẽ khó hơn `helmet`**, dù cả hai đều là "cái đầu". Và **dự đoán thứ ba: tăng độ phân giải đầu vào sẽ cải thiện đáng kể**, vì cho vật nhỏ thêm pixel.
>
> Còn một chi tiết ở bảng thứ hai cần chỉ ra: `person` có **40,5% thuộc nhóm lớn**, trong khi `helmet` chỉ 3,9%. Nghĩa là nhóm "vật thể lớn" trong bộ dữ liệu này gần như **đồng nghĩa với lớp `person`**. Chi tiết tưởng vụn vặt này chính là chìa khoá của phát hiện quan trọng nhất ở phần phân tích lỗi.
>
> Ba dự đoán trên sẽ được đối chiếu với số liệu thật ở các slide sau — và em xin nói trước: **có một dự đoán sai**.

---

### Slide 3 — Chuẩn hoá dữ liệu và kiểm chứng ✂️

**NỘI DUNG SLIDE**

📷 `artifacts/fig04_label_sanity_check.png`

Công thức VOC → YOLO:
`cx = (xmin+xmax)/2W` · `cy = (ymin+ymax)/2H` · `w = (xmax−xmin)/W` · `h = (ymax−ymin)/H`

Ví dụ: tâm ở pixel `380,5` → ảnh 416px hỏng khi resize · `380,5/416 = 0,9147` → đúng ở mọi cỡ ảnh

Khoá phân tầng = `lớp hiếm nhất trong ảnh` **|** `nhóm số vật thể`
→ `person|n3-5`, `head|n6-10`, `helmet|n1-2`, … — **mỗi nhóm chia riêng theo 70/15/15**

**Bảng 3 — Chia tập có phân tầng** *(seed 42)*

| Tập | Số ảnh | Số hộp | Hộp/ảnh | Ảnh trống | Hộp bị loại |
|---|---|---|---|---|---|
| train | 3.501 | 17.783 | 5,08 | 0 | 0 |
| val | 750 | 3.867 | 5,16 | 0 | 0 |
| test | 749 | 3.852 | 5,14 | 0 | 0 |

Dòng bằng chứng: **0 hộp bị mất · 0 nhãn lỗi · 3 tập rời nhau**

**NOTE NÓI** *(~50 giây)*

> VOC ghi hộp bằng **pixel tuyệt đối** và hai góc; YOLO cần **tâm và kích thước, chia cho cỡ ảnh** để nằm trong khoảng 0 tới 1.
>
> Vì sao phải chia? Vì ảnh gốc là 416×416, nhưng khi huấn luyện ta phóng lên **640×640**. Lấy một hộp thật làm ví dụ: tâm của nó nằm ở pixel 380,5. Nếu nhãn ghi thẳng số 380,5, thì sau khi phóng ảnh, tâm thật đã dịch tới pixel 585,4 — **nhãn cũ trở thành sai**, và mỗi lần đổi kích thước lại phải sửa toàn bộ nhãn. Còn nếu ghi 380,5 chia 416 bằng **0,9147**, con số này **không đổi** dù ảnh 416, 640 hay 960: nó nói "tâm nằm ở 91,47% chiều ngang", đúng với mọi cỡ ảnh. Nhân ngược lại với cỡ ảnh bất kỳ là ra pixel đúng.
>
> Về chia tập: nếu bốc ngẫu nhiên thuần, `person` chỉ có 751 hộp có thể rơi lệch hẳn — chẳng hạn tập test gần như không có `person` nào, thì kết quả đo trên lớp đó vô nghĩa. Nên em chia **có phân tầng**, làm như sau.
>
> Mỗi ảnh được gán một **nhãn nhóm** ghép từ hai thông tin. Thứ nhất là **lớp hiếm nhất có trong ảnh**: ảnh nào chứa `person` thì gắn thẻ `person`, không có `person` nhưng có `head` thì gắn `head`, còn lại gắn `helmet`. Thứ hai là **số vật thể trong ảnh**, chia thành các khoảng 1–2, 3–5, 6–10, và 11 trở lên. Ghép hai thứ lại được nhãn kiểu `person|n3-5` hay `helmet|n6-10`.
>
> Sau đó, thay vì chia toàn bộ 5.000 ảnh một lượt, em **chia riêng từng nhóm** theo đúng tỉ lệ 70–15–15. Nhóm `person|n3-5` có bao nhiêu ảnh thì cũng cắt 70% cho train, 15% val, 15% test. Nhờ vậy mỗi nhóm đều **góp mặt đúng tỉ lệ ở cả ba tập**, không nhóm nào bị dồn hết vào một chỗ.
>
> Hai bằng chứng cho thấy cách này hiệu quả: tỉ lệ `helmet` ở ba tập lần lượt 74,2 — 74,0 — 75,6%, lệch chưa tới 1,6 điểm phần trăm; và số hộp trên mỗi ảnh là 5,08 — 5,16 — 5,14, gần như bằng nhau.
>
> Hình là bước kiểm chứng: em vẽ lại hộp **đọc ngược từ file nhãn đã ghi ra**, không phải từ XML gốc. Nếu chuyển đổi sai hệ toạ độ, hộp sẽ lệch và phát hiện ngay tại đây — thay vì sau 40 phút huấn luyện.

---

## CHƯƠNG 2 — PHƯƠNG PHÁP

### Slide 4 — Fine-tune từ COCO + siêu tham số tường minh ⭐

*(gộp từ 2 slide: lý do fine-tune + cách khai báo siêu tham số)*

**NỘI DUNG SLIDE**

📷 `artifacts/fig04b_augmented_batch.png`

Sơ đồ nhỏ: `COCO (118k ảnh, 80 lớp)` → `thay tầng đầu ra` → `3 lớp`
Log thật: `Overriding model.yaml nc=80 with nc=3`

**Bảng 4 — Siêu tham số khai báo tường minh**

| Nhóm | Tham số | Giá trị | Ghi chú |
|---|---|---|---|
| Tối ưu | `optimizer` | **SGD** | *không dùng `"auto"`* |
| | `lr0` / `lrf` | 0,01 / 0,01 | |
| | `momentum` | 0,937 | |
| | `weight_decay` | 5e-4 | chính quy hoá L2 |
| | `warmup_epochs` | 3 | |
| | `cos_lr` | True | cosine annealing |
| Tăng cường | `mosaic` | 1,0 | ghép 4 ảnh |
| | `close_mosaic` | 10 | tắt mosaic 10 epoch cuối |
| | `fliplr` | 0,5 | lật ngang |
| | `scale` / `translate` | 0,5 / 0,1 | |
| | `hsv_h/s/v` | 0,015 / 0,7 / 0,4 | nhiễu màu |
| | `flipud`, `degrees`, `shear` | **0,0** | *tắt có chủ ý* |
| | `mixup`, `copy_paste` | **0,0** | *tắt có chủ ý* |
| Chung | `epochs` / `batch` / `imgsz` | 80 / 32 / 640 | seed 42 |

**NOTE NÓI** *(~75 giây)*

> Với 3.501 ảnh, huấn luyện lại đặc trưng thị giác cơ bản từ số 0 gần như chắc chắn thua. Trọng số COCO đã học sẵn cạnh, kết cấu, hình dáng người từ 118 nghìn ảnh. Việc duy nhất cần làm là thay tầng đầu ra từ 80 lớp xuống 3 — trong log có đúng một dòng xác nhận, em trích ở đây làm bằng chứng.
>
> Phần thứ hai là điểm em chủ động làm khác. Ultralytics mặc định đặt `optimizer` bằng `"auto"` — nó tự chọn AdamW và learning rate riêng mà **không in ra**. Đồng thời bật sẵn cả loạt phép tăng cường dữ liệu kể cả khi mình không viết dòng nào.
>
> Em khai báo lại tường minh toàn bộ, dùng SGD với momentum — đúng thuật toán đã học. Và khai cả những phép **cố ý tắt** kèm lý do: ví dụ tắt lật dọc, vì mũ bảo hộ luôn nằm trên đầu, lật dọc tạo ra ảnh không bao giờ xảy ra ngoài đời.
>
> Ảnh bên là ảnh **thật sự đi vào mạng** — không phải ảnh gốc, mà là mosaic ghép 4 ảnh, có lật, có nhiễu màu. Nếu không mở ra xem, em sẽ không trả lời được câu hỏi "ảnh huấn luyện của em trông thế nào".

---

### Slide 5 — YOLOv8 là anchor-free ✂️

**NỘI DUNG SLIDE**

📷 `artifacts/fig10_anchor_kmeans.png`

**Bảng 5 — Anchor-based vs anchor-free**

| | Anchor-based *(YOLOv5, Faster R-CNN)* | Anchor-free *(YOLOv8)* |
|---|---|---|
| Mẫu đặt sẵn | Có — k hộp mỗi ô lưới | Không |
| Mạng đoán gì | Độ lệch so với anchor | Khoảng cách tới 4 cạnh |
| Bước chuẩn bị | Chạy k-means chọn anchor | Không cần |

**Bảng 6 — Kiểm chứng từ chính checkpoint**

| Thuộc tính | Giá trị |
|---|---|
| Kiểu tầng cuối | `Detect` |
| Số lớp `nc` | 3 |
| `stride` | [8, 16, 32] |
| `reg_max` (DFL) | 16 |
| Thuộc tính anchor | **không có → anchor-free** |
| Lưới stride 8 | 80×80 = 6.400 vị trí |
| Lưới stride 16 | 40×40 = 1.600 vị trí |
| Lưới stride 32 | 20×20 = 400 vị trí |
| **Tổng hộp ứng viên** | **8.400** |

**NOTE NÓI** *(~55 giây)*

> Bài giảng dạy anchor-based đến YOLOv3. YOLOv8 mà đề bài yêu cầu thì đã bỏ hẳn anchor — mỗi điểm trên lưới dự đoán trực tiếp khoảng cách tới 4 cạnh hộp.
>
> Em kiểm chứng điều này **từ chính checkpoint**, không tin tài liệu suông: mô hình không có thuộc tính anchor nào. Và vẫn chạy k-means trên dữ liệu để minh hoạ anchor **sẽ** trông thế nào nếu dùng YOLOv5 — hình bên. Bảy trên chín anchor nằm sát đường chéo, tức gần vuông, khớp với việc đối tượng chủ yếu là cái đầu.
>
> Con số 8.400 hộp ứng viên cho một ảnh giải thích luôn vì sao bắt buộc phải có NMS — phần lớn trong số đó là bản sao chồng lấn của cùng một vật thể.

---

## CHƯƠNG 3 — KẾT QUẢ

### Slide 6 — Các phép đo dùng trong bài ⭐

**NỘI DUNG SLIDE**

**Bảng 7 — Công thức các chỉ số**

| Chỉ số | Công thức | Trả lời câu hỏi |
|---|---|---|
| **IoU** | `diện tích giao / diện tích hợp` | Hộp đoán khớp hộp thật đến đâu? |
| **Precision** | `TP / (TP + FP)` | Trong những cái mô hình báo, bao nhiêu % đúng? |
| **Recall** | `TP / (TP + FN)` | Trong những hộp thật, tìm được bao nhiêu %? |
| **F1** | `2PR / (P + R)` | Cân bằng cả hai — chỉ cao khi cả hai cùng cao |
| **AP** | diện tích dưới đường Precision–Recall | Tốt đến đâu, **không phụ thuộc ngưỡng**? |
| **mAP@0.5** | trung bình AP của 3 lớp, IoU ≥ 0,5 | Tìm **đúng chỗ** không? |
| **mAP@0.5:0.95** | như trên, trung bình 10 ngưỡng IoU 0,50→0,95 | Hộp **bám sát biên** không? |

Ba loại kết quả: **TP** báo đúng · **FP** báo thừa · **FN** **bỏ sót**

Sơ đồ phụ thuộc:
`IoU → TP/FP/FN → Precision & Recall → F1`   ·   `Precision & Recall → AP → mAP`

**NOTE NÓI** *(~70 giây)*

> Trước khi vào kết quả, em xin thống nhất cách đo, vì mọi con số phía sau đều dựa trên bảy chỉ số này.
>
> **IoU** so sánh độ khớp của hộp dự đoán với hộp gốc. Bằng 1 nếu trùng khít, bằng 0 nếu rời nhau. Cần nó vì phải có một quy tắc khách quan để nói dự đoán nào tính là đúng.
>
> Từ IoU ta chia kết quả thành ba loại: **TP** là báo đúng, **FP** là báo thừa, **FN** là bỏ sót.
>
> **Precision** là: trong những cái mô hình báo, bao nhiêu phần trăm đúng? **Recall** là: mô hình tìm được bao nhiêu phần trăm hộp so với số hộp thật? Hai chỉ số này khác nhau ở mẫu số — precision chia cho số mô hình báo, recall chia cho số thật sự có.
>
> Với bài toán này, **recall quan trọng hơn**: bỏ sót một đầu trần là bỏ lọt một người gặp nguy hiểm, còn báo nhầm chỉ tốn công kiểm tra. Đó là lý do em không chọn ngưỡng theo F1 mà ưu tiên recall của lớp `head`.
>
> **F1** là trung bình điều hoà của hai cái trên. Cần nó vì mỗi chỉ số riêng lẻ đều đánh lừa được: chỉ báo đúng một hộp duy nhất thì precision bằng 1, còn báo hộp khắp nơi thì recall bằng 1. F1 chỉ cao khi cả hai cùng cao.
>
> Nhưng cả ba đều có một điểm yếu chung: **chúng phụ thuộc vào ngưỡng tin cậy đã chọn**. Đổi ngưỡng là đổi số. **AP** giải quyết bằng cách không chọn ngưỡng nào cả — quét toàn bộ rồi lấy diện tích dưới đường precision-recall. **mAP** là trung bình AP của ba lớp.
>
> Cuối cùng, hai biến thể của mAP đo hai thứ khác nhau. **mAP@0.5** chấp nhận khi IoU đạt 0,5 — chỉ cần tìm **đúng chỗ**. **mAP@0.5:0.95** lấy trung bình qua 10 ngưỡng đến tận 0,95 — đòi hỏi hộp **bám sát biên**. Khoảng cách giữa hai số này sẽ cho biết mô hình yếu ở khâu tìm hay khâu vẽ hộp.

---

### Slide 7 — Đường cong huấn luyện + kết quả tổng

*(gộp từ 2 slide: đường cong + bảng val/test)*

**NỘI DUNG SLIDE**

📷 `artifacts/fig05_training_curves.png`

**Bảng 8 — Kết quả tổng** *(đo ở `conf = 0,001`)*

| Chỉ số | Val | Test |
|---|---|---|
| mAP@0.5 | 0,6270 | **0,6289** |
| mAP@0.5:0.95 | 0,4117 | **0,4132** |
| Precision | 0,6215 | 0,6307 |
| Recall | 0,5821 | 0,6157 |
| F1 | 0,6011 | 0,6231 |

Ghi chú: 80 epoch · 17,7 phút · epoch tốt nhất **69/80** · tập test chỉ chạm **một lần duy nhất**

**NOTE NÓI** *(~65 giây)*

> Điểm đáng chú ý ở panel bên trái: `train/box_loss` giảm đều suốt 80 epoch, nhưng `val/box_loss` — đường đứt nét đỏ — chạm đáy từ khoảng epoch 15 rồi đi ngang. Đó là dấu hiệu **quá khớp nhẹ**: mô hình vẫn học tốt hơn trên tập huấn luyện nhưng không còn cải thiện khả năng định vị trên dữ liệu chưa từng thấy. mAP ở panel giữa bão hoà từ khoảng epoch 50, epoch tốt nhất là 69.
>
> Về kết quả, val và test rất gần nhau, chênh dưới 0,02 ở mọi chỉ số. Đây là bằng chứng tốt cho thấy không có rò rỉ dữ liệu, và việc chọn checkpoint theo val không làm test trông tệ hơn thực tế.
>
> Một điểm về phương pháp: mAP được đo ở `conf` bằng 0,001 — rất thấp. Vì mAP theo định nghĩa là diện tích dưới toàn bộ đường precision-recall; cắt ở ngưỡng cao sẽ cụt đuôi đường cong và làm mAP giảm giả tạo.

**PHỤ LỤC NOTE 7A — Ba hàm mất mát** *(dùng khi bị hỏi, không đọc trong bài)*

> Bảy chỉ số ở slide trước dùng để **đánh giá** mô hình. Ba hàm mất mát ở đây khác hẳn: chúng dùng để **huấn luyện** — là thứ mạng thực sự tối ưu trong lúc học.
>
> Mỗi hộp phải trả lời ba câu hỏi độc lập, nên có ba loss riêng.
>
> `cls_loss` trả lời *"vật này thuộc lớp nào?"* — phạt khi mạng đoán sai lớp. Đây là loss tụt nhanh nhất, từ 2,22 xuống 1,21 chỉ sau một epoch, vì đầu phát hiện là phần bị thay mới nên phải học lại từ đầu.
>
> `box_loss` trả lời *"hộp có khớp vật thể không?"* — đo bằng 1 trừ IoU giữa hộp đoán và hộp thật. Nó nhìn hộp như **một khối tổng thể**.
>
> `dfl_loss` cũng về hộp, nhưng chi tiết hơn: thay vì bắt mạng đoán một con số cho mỗi cạnh, nó bắt mạng đưa ra **phân bố xác suất** trên các khoảng cách có thể. Cần thêm nó vì `box_loss` chỉ nói *"lệch bao nhiêu"* mà không nói *"lệch ở cạnh nào"* — hai hộp cùng IoU có thể sai theo hai kiểu hoàn toàn khác nhau. `dfl_loss` cho tín hiệu học rõ ràng cho từng cạnh, nên hộp bám sát biên vật thể hơn.

**PHỤ LỤC NOTE 7B — Vì sao đồ thị giật** *(câu hỏi rất dễ bị hỏi)*

> **Panel (b) — mAP giật, và giật mạnh nhất ở 10 epoch đầu.** Biên độ không đều theo thời gian, giảm dần rõ rệt:
>
> | Giai đoạn | Biên độ TB | Lớn nhất |
> |---|---|---|
> | epoch 1–10 | **0,0214** | **0,0562** |
> | epoch 11–30 | 0,0072 | 0,0140 |
> | epoch 31–60 | 0,0023 | 0,0090 |
> | epoch 61–80 | 0,0010 | 0,0030 |
>
> Giai đoạn đầu dao động **gấp 21 lần** giai đoạn cuối. Có hai nguyên nhân khác nhau cho hai giai đoạn.
>
> **10 epoch đầu — vì mạng ghép từ hai phần tuổi đời khác nhau.** Backbone lấy từ COCO nên đã tốt sẵn, gần như đứng yên. Đầu phát hiện thì mới khởi tạo ngẫu nhiên (đổi từ 80 lớp sang 3), phải học lại từ số 0 — `cls_loss` tụt 45% chỉ sau một epoch (2,22 → 1,21). Phần lớn thay đổi của cả mạng dồn hết vào đầu phát hiện đang "lột xác" liên tục, nên mAP nhảy theo. Đây **không phải nhiễu đo** mà là quá trình học thật — dù giật, xu hướng chung vẫn đi lên: 0,513 lên 0,593 sau 10 epoch. Cũng chính vì backbone gánh phần lớn việc ngay từ đầu mà mAP epoch 1 đã đạt 0,513 — cao bất thường cho một mạng vừa mới học.
>
> Cần nói rõ một điểm: **không thể quy hiện tượng này cho learning rate.** Epoch 11–30 có learning rate gần bằng epoch 1–10 (0,00853 so với 0,00888) nhưng dao động chỉ bằng một phần ba. Hệ số tương quan giữa learning rate và biên độ dao động chỉ đạt 0,531 — mức trung bình, không đủ để kết luận nhân quả.
>
> **Từ epoch 30 trở đi — learning rate giảm mới là yếu tố chính.** Cosine annealing kéo learning rate từ 0,0042 xuống 0,00063, tức mỗi bước cập nhật đi ngắn hơn, trọng số cuối mỗi epoch gần nhau hơn, đường cong mượt dần. Đây cũng là lý do chọn `cos_lr=True`: nó giúp giai đoạn cuối **ổn định để chọn checkpoint**, thay vì phải bốc giữa các epoch nhảy loạn.
>
> Phần dao động còn sót lại đến cuối (0,001) là nhiễu tự nhiên do tăng cường dữ liệu ngẫu nhiên khác nhau mỗi epoch — mosaic ghép 4 ảnh khác, lật khác, nhiễu màu khác.
>
> **Panel (c) — precision giật rất mạnh mà recall thì mượt.** Đây mới là điều đáng nói, và em đã truy ra nguyên nhân chính xác.
>
> Precision nhảy qua lại giữa hai mức: khoảng **0,94** và khoảng **0,61**. Có 9 cú nhảy như vậy trong 80 epoch. Điều quyết định: **cả 9 cú nhảy đều có biên độ đúng bằng 1/3** — trung bình đo được là 0,3331 so với 1/3 là 0,3333, lệch 0,1%.
>
> Con số 1/3 nói lên tất cả. Precision báo cáo là **trung bình của 3 lớp**. Nếu hai lớp đứng yên và **một lớp nhảy từ 1,0 xuống gần 0**, thì trung bình đổi đúng 1/3. Lớp đó là `person`.
>
> Vì sao `person` nhảy được như vậy? Vì tập val chỉ có **106 hộp `person`**, và mô hình gần như không phát hiện được lớp này. Ở một ngưỡng nào đó nó có thể chỉ đưa ra **một hoặc hai dự đoán** `person`. Nếu dự đoán đó trúng, precision của lớp bằng 1,0. Nếu trượt, bằng 0. Không có mức trung gian — mẫu quá ít.
>
> Còn recall thì mượt vì recall của `person` **luôn gần 0 dù thế nào**: tìm được 1 trên 106 hộp hay 0 trên 106 thì cũng đều xấp xỉ 0, nên nó đóng góp gần như không đổi vào trung bình.
>
> Bằng chứng cuối cùng: trong suốt các epoch precision nhảy loạn, **mAP@0.5 vẫn rất phẳng, chỉ dao động 0,613 đến 0,625**. Mô hình hoàn toàn không thay đổi. Cái nhảy là **cách báo cáo**, không phải chất lượng mô hình — chính xác là lý do vì sao mAP mới là chỉ số đáng tin để chọn checkpoint, còn precision/recall theo epoch chỉ để tham khảo.
>
> Đây cũng là một minh hoạ nữa cho vấn đề trung tâm của cả đồ án: lớp `person` với 562 hộp huấn luyện không chỉ cho kết quả kém, mà còn **làm nhiễu cả các chỉ số tổng hợp**.

---

### Slide 8 — Kết quả theo lớp: trả lời câu hỏi slide 2 ⭐⭐

**NỘI DUNG SLIDE**

📷 `artifacts/fig06_pr_curve_test.png`

**Bảng 9 — Chỉ số theo từng lớp trên tập test**

| Lớp | AP@0.5 | AP@0.5:0.95 | Precision | Recall | F1 | Số hộp test |
|---|---|---|---|---|---|---|
| `helmet` | **0,9558** | 0,6393 | 0,9157 | 0,9165 | 0,9161 | 2.910 |
| `head` | **0,9160** | 0,5925 | 0,8812 | 0,8824 | 0,8818 | 859 |
| `person` | **0,0149** | 0,0078 | 0,0953 | 0,0482 | 0,0640 | 83 |

Mũi tên nối ngược về slide 2

**NOTE NÓI** *(~70 giây)*

> Quay lại ba dự đoán ở slide 2. Slide này kiểm chứng hai dự đoán đầu.
>
> **Dự đoán 1 — `person` sẽ có AP rất thấp: ĐÚNG.** Chỉ 0,0149, thấp hơn hai lớp kia **60 lần**. Nhìn hình, đường màu xanh lá của `person` là một vệt dính sát trục hoành, gần như không nhìn thấy.
>
> **Dự đoán 2 — `head` khó hơn `helmet`: ĐÚNG, nhưng ít hơn em nghĩ.** 0,9160 so với 0,9558, chỉ chênh 4 phần trăm điểm. Dù `head` có tới 67,4% vật nhỏ so với 51,6% của `helmet`, khoảng cách thực tế nhỏ hơn dự kiến — mô hình xử lý vật nhỏ tốt hơn em lo ngại.
>
> Và đây là lý do mAP tổng chỉ 0,629 dù hai lớp kia gần 0,95: **mAP là trung bình cộng không trọng số trên ba lớp**. `person` chỉ chiếm 2,2% số hộp trong tập test nhưng vẫn gánh một phần ba trọng số, kéo tụt điểm rất mạnh.
>
> Nếu không giải thích điểm này, người đọc dễ hiểu nhầm rằng mô hình chỉ đạt 63% — trong khi thực chất hai trong ba lớp đều trên 91%.

---

### Slide 9 — Ma trận nhầm lẫn: vấn đề KHÔNG phải nhầm lớp ⭐

**NỘI DUNG SLIDE**

📷 `runs/yolov8n_640_e80_eval_test/confusion_matrix_normalized.png`

**Bảng 10 — Ma trận nhầm lẫn chuẩn hoá theo cột**

| Nhãn thật ↓ | → `helmet` | → `head` | → `person` | → bỏ sót |
|---|---|---|---|---|
| `helmet` | **0,93** | ~0 | — | 0,07 |
| `head` | **0,03** | **0,89** | — | 0,08 |
| `person` | ~0 | ~0 | **0,05** | **0,95** |
| *nền (báo động giả)* | **0,58** | 0,30 | 0,12 | — |

Ba ô cần khoanh đỏ: `head→helmet` = 3% · `person→bỏ sót` = 95% · nền→`helmet` = 58%

**NOTE NÓI** *(~60 giây)*

> Em từng dự đoán mô hình sẽ hay nhầm `head` thành `helmet`, vì `helmet` là lớp đa số nên khi lưỡng lự mô hình sẽ ngả về nó. **Số liệu bác bỏ dự đoán này**: chỉ 3%.
>
> Vấn đề thật nằm ở chỗ khác. `person` bị bỏ sót tới 95% — tức mô hình **không nhìn thấy**, chứ không phải nhìn nhầm. Hai chuyện hoàn toàn khác nhau, và hướng khắc phục cũng khác nhau.
>
> Còn hàng cuối cho thấy: khi mô hình bịa ra một hộp ở chỗ không có gì, 58% trường hợp nó gọi là `helmet`. Vậy thiên lệch về `helmet` là có thật — nhưng biểu hiện qua **báo động giả**, không phải qua việc nhầm một cái đầu trần thật thành có mũ.

---

### Slide 10 — Quét ngưỡng confidence ✂️

**NỘI DUNG SLIDE**

📷 `artifacts/fig07_conf_sweep.png`

**Bảng 11 — Quét 7 giá trị confidence** *(trên tập val)*

| conf | Precision | Recall | F1 | mAP@0.5 | R_helmet | R_head | **R_person** |
|---|---|---|---|---|---|---|---|
| 0,001 | 0,6215 | 0,5821 | 0,6011 | 0,6270 | 0,8949 | 0,8513 | **0,0** |
| 0,05 | 0,6215 | 0,5821 | 0,6011 | 0,6270 | 0,8949 | 0,8513 | **0,0** |
| 0,10 | 0,6215 | 0,5821 | 0,6011 | 0,6266 | 0,8949 | 0,8513 | **0,0** |
| 0,25 | 0,6215 | 0,5821 | 0,6011 | 0,6195 | 0,8949 | 0,8513 | **0,0** |
| **0,40** | 0,6214 | 0,5827 | **0,6014** | 0,6145 | 0,8959 | 0,8523 | **0,0** |
| 0,55 | 0,6325 | 0,5603 | 0,5942 | 0,6038 | 0,8650 | 0,8158 | **0,0** |
| 0,70 | 0,6483 | 0,4989 | 0,5639 | 0,5758 | 0,7853 | 0,7114 | **0,0** |

**NOTE NÓI** *(~50 giây)*

> Em quét 7 giá trị confidence trên tập val — không phải test, vì chọn ngưỡng là quyết định dựa trên dữ liệu, không được làm trên tập test.
>
> Điều đáng chú ý nhất không phải xu hướng chung, mà là cột cuối: **recall của `person` bằng đúng 0 ở cả 7 ngưỡng**. Recall là số hộp đúng tìm được chia cho tổng số hộp thật — bằng 0 nghĩa là mô hình tìm đúng **0 trên 106 hộp `person`** ở tập val, dù đo ở ngưỡng nào.
>
> Điểm quan trọng nằm ở ngưỡng thấp nhất: `conf = 0,001` gần như giữ lại mọi hộp mô hình xuất ra — gần như không lọc gì cả. Vậy mà recall vẫn là 0. Nghĩa là **không phải ngưỡng đang chặn mất các hộp `person` đúng** — mô hình đơn giản là **không sinh ra** hộp `person` đúng nào để mà giữ. Nếu đúng là do ngưỡng chặn, hạ ngưỡng xuống 0,001 phải vớt lại được ít nhất vài hộp; thực tế không có hộp nào.
>
> Nhìn kỹ hai cột `precision` và `recall`: từ conf 0,001 đến 0,40, cả hai gần như đứng yên — vùng này chưa cắt trúng dự đoán thật nào. Nhưng từ 0,40 trở đi, **recall giảm nhanh hơn precision tăng rất nhiều**: lên đến 0,70, precision chỉ tăng 4,3% trong khi recall giảm 14,4% — nhanh gấp hơn 3 lần. Đó là lý do F1 đạt đỉnh ngay tại 0,40 rồi đi xuống, dù precision vẫn tiếp tục tăng.
>
> Vì sao recall rơi nhanh vậy? Nhìn hai cột cuối: `R_helmet` và `R_head` đều giảm hai chữ số phần trăm trong cùng khoảng đó — 12,3% và 16,5%. Ngưỡng cao không chỉ chặn được các dự đoán sai, nó **cắt luôn nhiều dự đoán đúng** mà mô hình chỉ hơi thiếu tự tin. Vì hai lớp này chiếm 97% số hộp thật, mất recall ở đó kéo tụt recall tổng ngay lập tức, còn precision chỉ nhích lên chậm vì trong 3.852 hộp thật, số dự đoán sai bị loại không nhiều bằng số dự đoán đúng bị mất theo.
>
> Một điểm tinh tế cần nêu: mAP **giảm** khi tăng conf. Điều này không có nghĩa ngưỡng cao làm mô hình tệ đi — mAP là diện tích dưới đường PR, nâng ngưỡng cắt cụt đuôi đường cong nên diện tích nhỏ đi. Đây là hai việc khác nhau, không được lẫn lộn.

---

### Slide 11 — IoU và NMS: minh hoạ từ chính mô hình ⭐

**NỘI DUNG SLIDE**

📷 `artifacts/fig11_nms_effect.png` (chính) · `artifacts/fig09_conf_iou_heatmap.png` (nhỏ ở góc)

Ba con số lớn dưới mỗi ảnh: **77 hộp** · **27 hộp** · **23 hộp**
Ghi chú: *ảnh này có **26 người thật***

**Bảng 12 — Lưới 2D `conf` × `iou_nms`, giá trị F1 trên val**

| conf \ iou_nms | 0,40 | 0,55 | 0,70 | 0,85 |
|---|---|---|---|---|
| 0,05 | 0,601 | 0,601 | 0,601 | 0,598 |
| 0,15 | 0,601 | 0,601 | 0,601 | 0,598 |
| 0,25 | 0,601 | 0,601 | 0,601 | 0,595 |
| **0,40** | **0,602** | 0,601 | 0,601 | 0,598 |
| 0,55 | 0,594 | 0,594 | 0,594 | 0,592 |

→ Cấu hình triển khai chọn: **`conf = 0,4`, `iou_nms = 0,4`**

**NOTE NÓI** *(~65 giây)*

> Đề bài yêu cầu giải thích NMS bằng ví dụ **từ chính mô hình của nhóm**, nên em không lấy hình minh hoạ trên mạng mà dựng từ đầu ra thật.
>
> Cùng một ảnh có 26 người, chỉ thay đổi ngưỡng NMS. Gần như tắt NMS thì giữ 77 hộp — nhìn kỹ góc trên trái sẽ thấy nhiều hộp chồng khít lên cùng một cái đầu. Ở ngưỡng mặc định 0,7 còn 27 hộp. Ngưỡng rất hung hăng 0,3 còn 23 hộp.
>
> Con số 27 rất sát với 26 người thật — cho thấy NMS đang làm đúng việc: gộp các bản sao trùng lặp của cùng một vật thể.
>
> Bảng bên là lưới 2D quét 20 tổ hợp ngưỡng. Điều quan trọng không phải cấu hình tối ưu, mà là **vùng tốt rất rộng** — gần như phẳng hoàn toàn quanh mức 0,601. Điểm 0,602 được chọn không phải vì vượt trội, mà vì nó là số lớn nhất trong một cao nguyên bằng phẳng. Kết luận vì thế ổn định, không mong manh.

---

### Slide 12 — Ablation: ảnh hưởng của độ phân giải ✂️

**NỘI DUNG SLIDE**

📷 `artifacts/fig12_resolution_tradeoff.png`

Nhãn góc slide: **ABLATION STUDY** — *cùng seed, cùng epoch, cùng cấu hình; chỉ đổi `imgsz`*

**Bảng 13 — Huấn luyện lại đầy đủ ở 3 độ phân giải**

| imgsz | batch | mAP@0.5 | mAP@0.5:0.95 | AP50 `helmet` | AP50 `head` | AP50 `person` | ms/ảnh | Thời gian train |
|---|---|---|---|---|---|---|---|---|
| 416 | 32 | 0,6232 | 0,4035 | 0,9502 | 0,9059 | 0,0134 | 3,127 | 16,1 phút |
| **640** | 32 | **0,6289** | **0,4132** | 0,9558 | **0,9160** | 0,0149 | 3,566 | — |
| 960 | 16 | 0,6280 | 0,4132 | **0,9581** | 0,9152 | 0,0106 | 6,227 | 32,8 phút |

*(cột ms/ảnh lấy từ `benchmark_fps.py` — phép đo độc lập, không phải số đo trong vòng huấn luyện)*

**NOTE NÓI** *(~50 giây)*

> Đây là phép so sánh có kiểm soát của đồ án. Em **huấn luyện lại đầy đủ** ở từng độ phân giải, chứ không lấy mô hình 640 đem đánh giá ở 416 và 960 — cách sau đo lẫn hai thứ: ảnh hưởng thật của độ phân giải, và sự lệch phân bố giữa lúc huấn luyện với lúc kiểm thử.
>
> **Đây là dự đoán số 3 ở slide 2 — và nó SAI.** Dựa trên việc 53,9% vật thể là nhỏ, em dự đoán tăng độ phân giải sẽ cải thiện đáng kể. Thực tế hiệu ứng **có thật nhưng rất nhỏ**: `head` chỉ tăng 1 điểm phần trăm từ 416 lên 640, và **bão hoà ngay từ đó** — lên 960 không lợi thêm mà còn chậm gần gấp đôi.
>
> Vì sao dự đoán sai? Vì em suy luận đúng về cơ chế — vật nhỏ cần nhiều pixel hơn — nhưng nhầm về **độ lớn** của hiệu ứng. Ảnh gốc vốn đã là 416×416; phóng to lên 960 chỉ là nội suy, **không tạo ra chi tiết thật mới**. Cái tăng lên chỉ là số ô lưới đặc trưng, và lợi ích đó bão hoà nhanh.
>
> Đây là ví dụ cho thấy một lý luận cơ chế đúng chưa đảm bảo hiệu ứng lớn trong thực tế — phải đối chiếu với số liệu.

---

## CHƯƠNG 4 — PHÂN TÍCH LỖI

### Slide 13 — Phân loại lỗi + phát hiện quan trọng nhất ⭐⭐⭐

**NỘI DUNG SLIDE**

📷 `artifacts/fig13_error_by_size.png`

Sơ đồ nhỏ — 4 loại lỗi theo 2 ngưỡng IoU:

| Loại | Điều kiện |
|---|---|
| Đúng | IoU ≥ 0,5 **và** đúng lớp |
| Nhầm lớp | IoU ≥ 0,5 **nhưng** sai lớp |
| Định vị lệch | 0,1 ≤ IoU < 0,5 |
| Phát hiện thừa | IoU < 0,1 |
| Bỏ sót | Hộp thật không được dự đoán nào chiếm |

**Bảng 14 — Phân loại lỗi theo kích thước vật thể**

| Kích thước | Đúng | Nhầm lớp | Định vị lệch | **Bỏ sót** | Phát hiện thừa | Tổng hộp thật | **Tỉ lệ bỏ sót** |
|---|---|---|---|---|---|---|---|
| small | 1.786 | 13 | 19 | 286 | — | 2.104 | 13,6% |
| medium | 1.467 | 6 | 8 | 125 | — | 1.606 | 7,8% |
| **large** | 98 | 1 | 0 | 43 | — | 142 | **30,3%** |
| *(không áp dụng)* | — | — | — | — | 239 | — | — |
| **TỔNG** | **3.351** | **20** | **27** | **454** | **239** | 3.852 | |

Chữ lớn dưới cùng: **"large ≠ do kích thước"**

**NOTE NÓI** *(~90 giây)*

> Em ghép từng dự đoán với hộp thật bằng thuật toán **tham lam** — duyệt theo confidence giảm dần, mỗi dự đoán chiếm hộp thật gần nhất chưa ai chiếm. Chọn tham lam chứ không phải Hungarian tối ưu toàn cục, vì đây đúng là cách chuẩn COCO tính TP và FP để ra mAP. Dùng Hungarian sẽ cho bảng lỗi không khớp với con số mAP đã báo cáo.
>
> Kết quả: **bỏ sót áp đảo** với 454 trường hợp, nhiều hơn ba loại lỗi kia cộng lại. Vấn đề của hệ thống là "không nhìn thấy", không phải "nhìn nhầm".
>
> Nhưng phát hiện đáng chú ý nhất nằm ở cột cuối. Nhóm `large` bị bỏ sót nhiều nhất — 30,3%, gấp hơn hai lần nhóm `small`. **Ngược hẳn dự đoán "vật nhỏ khó hơn"**.
>
> Nếu dừng ở đây và kết luận "vật to khó phát hiện hơn" thì **sai**. Xin quay lại bảng ở slide 2: lớp `person` có 40,5% số hộp thuộc nhóm `large`, trong khi `helmet` chỉ 3,9%. Nghĩa là nhóm `large` gần như **đồng nghĩa với lớp `person`** — mà `person` thì thiếu dữ liệu trầm trọng.
>
> Vậy nguyên nhân thật là **thiếu dữ liệu**, không phải kích thước. Đây là một biến gây nhiễu, và em phát hiện ra nhờ đối chiếu chéo giữa hai notebook.

---

### Slide 14 — Minh hoạ lỗi trên ảnh thật

**NỘI DUNG SLIDE**

📷 `artifacts/fig14_error_examples.png` — phóng to ảnh có nhãn `helmet→head`

**Bảng 15 — Tỉ lệ lỗi theo từng lớp**

| Lớp | Số hộp thật | Đúng | Nhầm lớp | Định vị lệch | Bỏ sót | %Đúng | %Nhầm lớp | %Bỏ sót |
|---|---|---|---|---|---|---|---|---|
| `helmet` | 2.910 | 2.623 | 2 | 18 | 267 | 90,1 | 0,1 | 9,2 |
| `head` | 859 | 726 | 18 | 9 | 106 | 84,5 | **2,1** | 12,3 |
| `person` | 83 | 2 | 0 | 0 | 81 | 2,4 | 0,0 | **97,6** |

**NOTE NÓI** *(~60 giây)*

> Tám ảnh minh hoạ, mỗi loại lỗi hai ảnh, mỗi hộp có nhãn chữ ghi rõ lớp liên quan.
>
> Ảnh phóng to là trường hợp **nhầm lớp nguy hiểm nhất về nghiệp vụ**: người này đang đội một chiếc mũ vải tối màu, và mô hình đoán là `helmet` — có mũ bảo hộ. Thật ra nhãn đúng là `head`, đầu trần. Với hệ thống cảnh báo an toàn, nhầm theo chiều này nghĩa là **im lặng khi đáng lẽ phải cảnh báo**.
>
> May mắn là loại nhầm này chỉ chiếm 2,1% số hộp `head`. Vấn đề lớn hơn nhiều vẫn là bỏ sót: `head` bị bỏ sót 12,3%, còn `person` thì 97,6%.
>
> Một quan sát định tính nữa: bỏ sót tập trung ở cảnh thiếu sáng và nền phức tạp — như ảnh trong đường hầm. Đây là hướng phân tích em chưa kịp định lượng, xin nêu ở phần hạn chế.

---

## CHƯƠNG 5 — SO SÁNH & KẾT LUẬN

### Slide 15 — YOLOv8n vs Faster R-CNN + bằng chứng chéo ⭐⭐

*(gộp từ 2 slide: bảng so sánh + bằng chứng chéo về `person`)*

**NỘI DUNG SLIDE**

**Bảng 16 — So sánh hai kiến trúc** *(cùng tập test, cùng 640px, cùng điều kiện augmentation)*

| | YOLOv8n *(one-stage)* | Faster R-CNN *(two-stage)* | Chênh lệch |
|---|---|---|---|
| mAP@0.5 | **0,6289** | 0,6167 | +0,0122 |
| mAP@0.5:0.95 | **0,4132** | 0,3866 | +0,0266 |
| Số tham số | **3,0M** | 41,3M | **nhẹ hơn 13,7×** |
| ms/ảnh @640 | **3,57** | 9,74 | **nhanh hơn 2,73×** |
| Số thành phần loss | 3 | 4 | *(one vs two-stage)* |

**Bảng 17 — Lớp `person` ở hai kiến trúc độc lập** *(AP@0.5:0.95 — cùng một thang cho cả hai)*

| | YOLOv8n | Faster R-CNN |
|---|---|---|
| Số tham số | 3,0M | 41,3M *(gấp 13,7×)* |
| AP `helmet` | 0,6393 | 0,5996 |
| AP `head` | 0,5925 | 0,5373 |
| **AP `person`** | **0,0078** | **0,0230** |

Chữ lớn: **"Hai kiến trúc khác hẳn nhau — cùng thất bại như nhau"**

**NOTE NÓI** *(~85 giây)*

> Cả hai mô hình được đánh giá trên **đúng cùng một tập test**, cùng độ phân giải 640, cùng điều kiện tăng cường dữ liệu.
>
> Về độ chính xác, hai kiến trúc **xấp xỉ nhau** — chênh lệch chỉ 0,0122 mAP@0.5, quá nhỏ để kết luận kiến trúc nào ưu việt hơn. Nhất là khi augmentation giữa hai bên vẫn chưa tương đương tuyệt đối: Faster R-CNN có lật ngang và nhiễu màu, còn thiếu mosaic mà YOLOv8n đang dùng.
>
> Khác biệt thật sự nằm ở **chi phí**: nặng hơn 13,7 lần và chậm hơn 2,73 lần. Hai con số này đo trực tiếp từ kiến trúc, không phụ thuộc điều kiện huấn luyện, nên không gây tranh cãi như mAP.
>
> Nhưng bảng dưới mới là bằng chứng mạnh nhất của cả đồ án. Lưu ý bảng này đo cả hai ở **cùng một ngưỡng AP@0,5:0,95** — ngưỡng khó, đòi hỏi hộp bám sát biên — để so sánh công bằng, không lệch thang như phần mAP tổng ở trên vốn đã tách riêng hai ngưỡng.
>
> Hai kiến trúc hoàn toàn khác nhau — chênh nhau gần 14 lần về số tham số — nhưng **cùng thất bại thảm hại** với lớp `person`: 0,0078 và 0,0230, đều dưới 4% so với mức ~0,60 của hai lớp còn lại. Faster R-CNN nhỉnh hơn một chút, nhưng chênh lệch đó là 0,015 — không đáng kể so với khoảng cách 0,58 giữa `person` và hai lớp kia.
>
> Nếu đây là hạn chế của kiến trúc, thì đổi sang two-stage — vốn có mạng đề xuất vùng chuyên biệt — phải cải thiện đáng kể. Thực tế không hề. Vậy nguyên nhân nằm ở **dữ liệu**: chỉ 562 hộp `person` trong tập huấn luyện so với hơn 13 nghìn hộp `helmet`. Hướng cải thiện đúng là thu thập thêm dữ liệu, không phải đổi mô hình — kết luận mà nếu chỉ chạy một mô hình thì không rút ra được.

---

### Slide 16 — Kết luận, hạn chế, bài học ⭐

**NỘI DUNG SLIDE**

**Ba kết luận**
1. Hai lớp chính đạt AP@0.5 > 0,91 — hệ thống dùng được
2. Lỗi chi phối là **bỏ sót** (454/740 ≈ 61% tổng lỗi), không phải nhầm lớp (chỉ 3%)
3. Vấn đề lớp `person` nằm ở **dữ liệu** — xác nhận bằng 2 kiến trúc độc lập

**Ba hạn chế**
1. Augmentation giữa hai mô hình chưa đồng bộ hoàn toàn
2. Đo tốc độ ở batch=1 bị chi phí cố định chi phối — nút thắt ở CPU, không phải GPU
3. Chưa định lượng ảnh hưởng của điều kiện ánh sáng

**Bài học phương pháp** *(in đậm, chiếm nửa dưới slide)*
> *Ba dự đoán ban đầu đều bị chính số liệu bác bỏ: "vật lớn khó hơn", "hay nhầm head thành helmet", "tăng độ phân giải sẽ cải thiện nhiều". Đối chiếu chéo nhiều nguồn độc lập là thứ giúp phát hiện ra.*

**NOTE NÓI** *(~75 giây)*

> Ba kết luận. Một, hệ thống hoạt động tốt trên hai lớp chính, đều trên 0,91. Hai, loại lỗi chi phối là bỏ sót — chiếm khoảng 61% tổng số lỗi — chứ không phải nhầm lớp, vốn chỉ 3%. Ba, lớp `person` thất bại như nhau ở cả hai kiến trúc độc lập, chứng tỏ nguyên nhân nằm ở dữ liệu.
>
> Về hạn chế, em xin nêu thẳng ba điểm. Augmentation giữa hai mô hình chưa đồng bộ hoàn toàn nên phép so sánh kiến trúc chưa tuyệt đối công bằng. Phép đo tốc độ ở batch bằng 1 trên RTX 4090 bị chi phí cố định chi phối — thực tế phần chạy trên GPU gần như không đổi khi ảnh lớn hơn 5 lần, nút thắt nằm ở CPU. Và em quan sát thấy bỏ sót tập trung ở cảnh thiếu sáng nhưng chưa kịp định lượng.
>
> Cuối cùng là bài học lớn nhất. Ba dự đoán ban đầu của em đều bị chính số liệu bác bỏ — chuyện "vật lớn khó hơn", chuyện "hay nhầm head thành helmet", chuyện "tăng độ phân giải sẽ cải thiện nhiều". Thói quen đối chiếu chéo nhiều nguồn độc lập là thứ giúp em phát hiện ra những chỗ đó, thay vì báo cáo một kết luận sai về nguyên nhân.
>
> Em xin hết. Em sẵn sàng trả lời câu hỏi.

---

## PHỤ LỤC *(không trình bày — để sẵn trả lời câu hỏi)*

### PL1 — Đặc điểm hình học của dữ liệu

📷 `artifacts/fig03_aspect_and_center.png`

**Dùng khi được hỏi:** vì sao anchor k-means ra hình gần vuông · dữ liệu có thiên lệch không gian không

**Điểm nói:** tỉ lệ khung `helmet`/`head` đạt đỉnh ở 0,8–0,9 (gần vuông); `person` có **hai đỉnh** — đỉnh chính 0,3–0,4 (người đứng) và một đỉnh phụ ở vùng rộng hơn cao, có thể là lỗi gán nhãn cần kiểm tra. Bản đồ mật độ tâm hộp cho thấy vật thể tập trung ở dải `cy ≈ 0,3–0,55`, tức hơi dưới giữa khung hình.

### PL2 — Quét ngưỡng NMS đầy đủ

📷 `artifacts/fig08_nms_sweep.png`

**Bảng 18 — Quét 7 giá trị `iou_nms`** *(trên tập val, `conf = 0,001`)*

| iou_nms | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| 0,30 | 0,6228 | 0,5811 | 0,6012 | 0,6279 | **0,4155** |
| 0,45 | 0,6229 | 0,5816 | **0,6015** | 0,6285 | 0,4144 |
| 0,50 | 0,6225 | 0,5816 | 0,6014 | **0,6286** | 0,4139 |
| 0,60 | 0,6224 | 0,5818 | 0,6014 | 0,6283 | 0,4130 |
| 0,70 | 0,6215 | 0,5821 | 0,6011 | 0,6270 | 0,4117 |
| 0,80 | 0,6189 | 0,5822 | 0,6000 | 0,6236 | 0,4107 |
| 0,90 | 0,6033 | 0,5616 | 0,5817 | 0,6106 | 0,4058 |

**Điểm nói:** từ 0,3 đến 0,8 mọi chỉ số gần như đứng yên (mAP@0.5 dao động 0,624–0,629), chỉ tới 0,9 mới sụt rõ. Vùng an toàn rất rộng nên lựa chọn không mong manh.

### PL3 — Đo tốc độ đầy đủ

**Bảng 19 — `benchmark_fps.py`** *(tiến trình sạch, 3 lượt độc lập, RTX 4090)*

| Cấu hình | ms trung bình | ± độ lệch chuẩn | FPS |
|---|---|---|---|
| YOLOv8n 416 — `.predict()` | 3,127 | 0,005 | 319,8 |
| YOLOv8n 416 — chỉ forward | 2,208 | 0,003 | 452,8 |
| YOLOv8n 640 — `.predict()` | 3,566 | 0,013 | 280,4 |
| YOLOv8n 640 — chỉ forward | 2,244 | 0,003 | 445,6 |
| YOLOv8n 960 — `.predict()` | 6,227 | 0,075 | 160,6 |
| YOLOv8n 960 — chỉ forward | 2,288 | 0,001 | 437,0 |
| Faster R-CNN 416 | 9,342 | 0,007 | 107,0 |
| Faster R-CNN 640 | 9,740 | 0,041 | 102,7 |
| Faster R-CNN 960 | 15,143 | 0,037 | 66,0 |

**Điểm nói — phát hiện về nút thắt cổ chai:**

| | 416px | 960px | Tỉ lệ |
|---|---|---|---|
| Forward mạng (GPU) | 2,208 | 2,288 | **1,04×** |
| Tiền/hậu xử lý (CPU) | 0,919 | 3,939 | **4,29×** |
| *(số pixel tăng)* | | | *5,33×* |

Phần chạy trên GPU gần như không đổi dù ảnh lớn hơn 5,33 lần — vì tính toán thật chỉ mất ~0,1 ms, còn ~2,2 ms là chi phí khởi chạy khoảng 200 CUDA kernel. **Nút thắt nằm ở CPU (resize, NMS), không phải GPU.**

### PL4 — Kiểm chứng augmentation của Faster R-CNN

📷 `artifacts/fig16_frcnn_augment_check.png`

**Dùng khi được hỏi:** làm sao biết lật ảnh không làm hỏng nhãn

**Điểm nói:** lật ảnh mà quên lật hộp là lỗi âm thầm — không báo lỗi, chỉ lộ ra qua mAP thấp bất thường sau nhiều giờ train. Em kiểm chứng bằng số trước (chiều rộng hộp không đổi, hộp không lật ngược, lật hai lần trả về chính nó) rồi mới vẽ ra xem tận mắt.

### PL5 — Đường cong huấn luyện Faster R-CNN

📷 `artifacts/fig15_frcnn_training_curve.png`

**Dùng khi được hỏi:** vì sao two-stage có 4 loss còn YOLO chỉ 3

**Điểm nói:**

| Giai đoạn | Loss | Nhiệm vụ |
|---|---|---|
| 1 — RPN | `loss_objectness` | có vật thể hay là nền? *(không phân biệt lớp)* |
| 1 — RPN | `loss_rpn_box_reg` | chỉnh sơ bộ vùng đề xuất |
| 2 — RoI | `loss_classifier` | phân loại 4 lớp *(gồm background)* |
| 2 — RoI | `loss_box_reg` | tinh chỉnh hộp cuối, **riêng cho từng lớp** |

Hai loss giai đoạn 1 gần như bằng 0 ngay từ epoch đầu — khoanh vùng thô rất dễ. `loss_box_reg` là thành phần lớn nhất suốt 15 epoch — vẽ hộp khít mới là phần khó.

### PL6 — Các đường cong bổ trợ của ultralytics

📷 `runs/yolov8n_640_e80_eval_test/`: `F1_curve.png` · `P_curve.png` · `R_curve.png` · `confusion_matrix.png` *(chưa chuẩn hoá)*

**Dùng khi được hỏi:** số liệu tuyệt đối của ma trận nhầm lẫn · điểm F1 tối ưu theo ngưỡng

---

## Ghi chú khi dựng slide

**Hình cần copy từ server về máy dựng slide**

| File | Nguồn |
|---|---|
| `fig13_error_by_size.png` | `artifacts/` — notebook 04 |
| `fig14_error_examples.png` | `artifacts/` — notebook 04 |
| `fig15_frcnn_training_curve.png` | `artifacts/` — notebook 03 |
| `fig16_frcnn_augment_check.png` | `artifacts/` — notebook 03 |
| `confusion_matrix_normalized.png` | `runs/yolov8n_640_e80_eval_test/` |

**Còn thiếu theo yêu cầu đề bài:** *"video/ảnh demo suy luận"*. Hai hình `fig11_nms_effect.png` và `fig14_error_examples.png` đã là ảnh suy luận thật nên có thể tính là đạt, nhưng làm thêm một video ngắn hoặc vài ảnh demo riêng sẽ chắc chắn hơn.

**Nguyên tắc trình bày số:** trong văn bản tiếng Việt dùng dấu phẩy thập phân (0,6289); khi trích nguyên văn output máy in ra thì giữ dấu chấm.

**Nếu thiếu giờ:** cắt slide 3, 5, 10, 12 *(đánh dấu ✂️)* — còn 12 slide, mạch truyện vẫn liền vì đây là chi tiết bổ trợ, không phải mắt xích lập luận.

**Bốn slide không được cắt:** 2 *(đặt câu hỏi)* → 7 *(trả lời)* → 12 *(phát hiện chính)* → 14 *(bằng chứng chéo)*. Đây là bộ khung lập luận; mất một cái là gãy mạch.
