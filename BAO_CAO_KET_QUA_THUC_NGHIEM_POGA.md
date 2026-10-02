# BÁO CÁO KẾT QUẢ THỰC NGHIỆM RÚT GỌN DỮ LIỆU HYBRID PO–GA
**Mã đề tài:** CNTT-KLCN140  
**Tên đề tài:** Ứng dụng thuật toán Parrot Optimizer kết hợp Genetic Algorithm trong rút gọn tập dữ liệu ảnh cho bài toán phân loại cảnh tự nhiên  
**Phiên chạy thực nghiệm:** [`outputs/run_poga_all_20261001_235502`](file:///d:/Code/KhoaLuanDatasetReduction/outputs/run_poga_all_20261001_235502)  

---

## 1. Cấu hình Thực nghiệm và Thiết lập Đánh giá

| Hạng mục | Thông số kỹ thuật trong phiên chạy `run_poga_all_20261001_235502` |
| :--- | :--- |
| **Bộ dữ liệu** | Intel Image Classification ($17.034$ ảnh RGB kích thước $224 \times 224$, $C = 6$ lớp cảnh tự nhiên) |
| **Phân hoạch dữ liệu chuẩn** | Tập Train gốc $D_{\text{train}}$: **$11.228$ ảnh** ($80\%$ `seg_train`) $\cdot$ Tập Validation $D_{\text{val}}$: **$2.806$ ảnh** ($20\%$ `seg_train`) $\cdot$ Tập Test độc lập $D_{\text{test}}$: **$3.000$ ảnh** (`seg_test`) |
| **Đặc trưng & Khoảng cách** | Vector đặc trưng `ResNet-18` tiền huấn luyện ($512$ chiều), ma trận khoảng cách Cosine chuẩn hóa theo $\text{dist}_{\max} \in [0, 1]$ |
| **Hàm mục tiêu 4 thành phần** | $F(X) = 0,10\,\text{Div}(X) + 0,30\,\text{Cov}(X) + 0,20\,\text{Bal}(X) + 0,40\,\text{Com}(X)$ (tự cân bằng kích thước $M$) |
| **Ngân sách tối ưu hóa** | Kích thước quần thể $P = 20$, số vòng lặp $T = 100 \Rightarrow$ cố định **$2.000\text{ FEs}$** cho mọi thuật toán |
| **Tham số lai ghép PO–GA** | Độ dài pha xen kẽ (S1, S2): $p = 10$ vòng $\cdot$ Tỷ lệ chia khối (S3, S4, P1): $50\% / 50\%$ $\cdot$ Trao đổi ưu tú (C1): $k = 3$ cá thể / chu kỳ $5$ vòng |
| **Hạt giống thực nghiệm (Seeds)** | $3$ hạt giống tối ưu hóa độc lập `seeds` $= [42, 123, 456]$ (cùng quần thể khởi tạo tại mỗi seed); cố định `seed = 42` khi huấn luyện CNN |
| **Mô hình CNN đánh giá hạ nguồn** | **ResNet-18** ($11,18\text{M}$ tham số), **MobileNetV3-Small** ($1,52\text{M}$ tham số), **DenseNet-121** ($6,96\text{M}$ tham số) — huấn luyện $5$ epochs, `Adam` ($\text{lr}=10^{-3}$), `batch_size = 64` |

---

## 2. Kết quả Mốc chuẩn Toàn Tập Dữ liệu (Full Dataset Upper Bound - B0)

Mốc chuẩn trên (`B0 — Full Dataset`) được xác lập khi huấn luyện trực tiếp 3 mô hình CNN trên toàn bộ $11.228$ ảnh huấn luyện (không rút gọn, $\text{DR} = 0\%$) và đánh giá trên $3.000$ ảnh kiểm thử độc lập `seg_test`.

### Bảng 2.1: Hiệu năng tổng thể của 3 mô hình CNN trên toàn bộ tập dữ liệu gốc (B0)

| Mô hình CNN | Số ảnh Train ($N$) | Tỷ lệ rút gọn $\text{DR}$ (%) | Val Acc (%) | Test Acc (%) | Macro Precision (%) | Macro Recall (%) | Macro F1-Score (%) | Thời gian Train (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18** | $11.228$ | $0,00\%$ | $90,34\%$ | $90,27\%$ | $90,38\%$ | $90,46\%$ | $90,41\%$ | $265,16\text{ s}$ |
| **MobileNetV3-Small** | $11.228$ | $0,00\%$ | $90,34\%$ | $88,93\%$ | $89,26\%$ | $89,28\%$ | $89,25\%$ | $174,44\text{ s}$ |
| **DenseNet-121** | $11.228$ | $0,00\%$ | **$91,80\%$** | **$90,57\%$** | **$90,71\%$** | **$90,76\%$** | **$90,72\%$** | $301,34\text{ s}$ |

### Bảng 2.2: Chỉ số F1-Score (%) theo từng lớp cảnh tự nhiên của Mốc chuẩn B0 trên tập Test độc lập (`seg_test`)

| Lớp cảnh tự nhiên | Số mẫu Test | ResNet-18 F1 (%) | MobileNetV3-Small F1 (%) | DenseNet-121 F1 (%) | Đặc điểm phân lớp |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `buildings` (Tòa nhà) | $437$ | $90,35\%$ | $89,14\%$ | $91,20\%$ | Cấu trúc hình học rõ ràng, chia sẻ đặc trưng đô thị với `street`. |
| `forest` (Rừng cây) | $474$ | **$98,21\%$** | **$97,58\%$** | **$98,10\%$** | Kết cấu tán lá đặc trưng, đạt F1-score cao nhất trên cả 3 mô hình. |
| `glacier` (Sông băng) | $553$ | $85,20\%$ | $83,61\%$ | $85,37\%$ | Chồng lấp ngữ nghĩa cao với `mountain` do cùng có tuyết và sườn núi. |
| `mountain` (Núi) | $525$ | $85,49\%$ | $83,84\%$ | $85,63\%$ | Dễ nhầm lẫn với `glacier` tại vùng biên quyết định. |
| `sea` (Biển) | $510$ | $93,65\%$ | $93,05\%$ | $94,42\%$ | Bố cục đường chân trời và mặt nước tách biệt rõ. |
| `street` (Đường phố) | $501$ | $89,58\%$ | $88,28\%$ | $89,60\%$ | Có sự giao thoa cảnh quan kiến trúc với lớp `buildings`. |
| **Trung bình vĩ mô (Macro Avg)** | **$3.000$** | **$90,41\%$** | **$89,25\%$** | **$90,72\%$** | Phân loại đồng đều, không xảy ra hiện tượng lệch lớp. |

---

## 3. Kết quả Đánh giá Tầng 1: Chất lượng Tối ưu hóa Tổ hợp (PO–GA Optimization Metrics)

### Bảng 3.1: Thống kê chất lượng tối ưu hóa hàm mục tiêu 4 thành phần qua 3 Seeds (`42, 123, 456`)

| Phương pháp | Kiến trúc | Số ảnh giữ lại $M$ ($\text{Mean} \pm \text{Std}$) | Tỷ lệ rút gọn $\text{DR}$ (%) | Fitness ($\text{Mean} \pm \text{Std}$) | Fitness ($\text{Median} \text{ [IQR]}$) | $\text{Div}$ | $\text{Cov}$ | $\text{Bal}$ | $\text{Com}$ | Thời gian Tối ưu (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1 — Random** | Ngẫu nhiên | $604,0 \pm 0,0$ | $94,62\%$ | $0,8765 \pm 0,0010$ | $0,8759 \text{ [0,0011]}$ | $0,5037$ | $0,8351$ | $0,9855$ | $0,9462$ | $< 0,01\text{ s}$ |
| **B2 — Stratified** | Phân tầng | $604,0 \pm 0,0$ | $94,62\%$ | $0,8796 \pm 0,0000$ | $0,8796 \text{ [0,0000]}$ | $0,5060$ | $0,8355$ | **$0,9996$** | $0,9462$ | $< 0,01\text{ s}$ |
| **B3 — GA-only** | Đơn lẻ | $3.229,0 \pm 36,9$ | $71,24\%$ | $0,8026 \pm 0,0012$ | $0,8028 \text{ [0,0014]}$ | $0,5024$ | **$0,8926$** | $0,9979$ | $0,7124$ | $337,48 \pm 2,05\text{ s}$ |
| **B4 — PO-only** | Đơn lẻ | $223,7 \pm 2,5$ | $98,01\%$ | $0,8869 \pm 0,0006$ | **$0,8873 \text{ [0,0006]}$** | $0,5130$ | $0,8127$ | $0,9988$ | $0,9801$ | $139,24 \pm 6,37\text{ s}$ |
| **S1 — Alt GA→PO** | Sequential | $243,0 \pm 15,9$ | $97,84\%$ | $0,8862 \pm 0,0004$ | $0,8862 \text{ [0,0006]}$ | $0,5076$ | $0,8155$ | $0,9973$ | $0,9784$ | $123,29 \pm 5,75\text{ s}$ |
| **S2 — Alt PO→GA** | Sequential | $272,7 \pm 36,9$ | $97,57\%$ | $0,8863 \pm 0,0008$ | $0,8862 \text{ [0,0010]}$ | $0,5112$ | $0,8173$ | $0,9986$ | $0,9757$ | **$105,81 \pm 2,99\text{ s}$** |
| **S3 — Block GA→PO**| Sequential | $213,7 \pm 14,7$ | $98,10\%$ | **$0,8875 \pm 0,0007$** | $0,8872 \text{ [0,0008]}$ | **$0,5144$** | $0,8129$ | $0,9987$ | $0,9810$ | $215,95 \pm 2,74\text{ s}$ |
| **S4 — Block PO→GA**| Sequential | $227,3 \pm 7,4$ | $97,98\%$ | $0,8866 \pm 0,0004$ | $0,8869 \text{ [0,0004]}$ | $0,5126$ | $0,8132$ | $0,9977$ | $0,9798$ | $114,35 \pm 5,02\text{ s}$ |
| **P1 — Parallel** | Parallel | $221,7 \pm 21,0$ | $98,03\%$ | $0,8865 \pm 0,0004$ | $0,8865 \text{ [0,0006]}$ | $0,5100$ | $0,8127$ | $0,9982$ | $0,9803$ | $241,50 \pm 4,80\text{ s}$ |
| **C1 — Cooperative**| Cooperative| $204,0 \pm 6,2$ | **$98,18\%$** | $0,8870 \pm 0,0004$ | $0,8870 \text{ [0,0005]}$ | $0,5110$ | $0,8114$ | $0,9984$ | **$0,9818$** | $113,64 \pm 2,12\text{ s}$ |

### Bảng 3.2: Chi tiết nghiệm tối ưu tìm được tại từng Seed (`42, 123, 456`)

| Phương pháp | Seed 42: $M$ ($\text{DR}\%$) — Fitness | Seed 123: $M$ ($\text{DR}\%$) — Fitness | Seed 456: $M$ ($\text{DR}\%$) — Fitness | Nhận xét tính ổn định giữa các Seeds |
| :--- | :---: | :---: | :---: | :--- |
| **S1 (Alt GA→PO)** | $228$ ($97,97\%$) — $0,8868$ | $236$ ($97,90\%$) — $0,8857$ | $265$ ($97,64\%$) — $0,8862$ | Giữ lại $228 - 265$ ảnh, độ lệch chuẩn Fitness rất nhỏ ($0,0004$). |
| **S2 (Alt PO→GA)** | $309$ ($97,25\%$) — $0,8854$ | $287$ ($97,44\%$) — $0,8862$ | $222$ ($98,02\%$) — $0,8874$ | Giữ lại nhiều ảnh nhất trong nhóm lai ghép ($272,7$ ảnh trung bình). |
| **S3 (Block GA→PO)** | $226$ ($97,99\%$) — **$0,8884$** | $222$ ($98,02\%$) — $0,8868$ | $193$ ($98,28\%$) — $0,8872$ | Đạt đỉnh Fitness cao nhất toàn cục ($0,8884$ tại Seed 42). |
| **S4 (Block PO→GA)** | $219$ ($98,05\%$) — $0,8869$ | $237$ ($97,89\%$) — $0,8861$ | $226$ ($97,99\%$) — $0,8869$ | Kích thước tập con ổn định quanh mức $219 - 237$ ảnh. |
| **P1 (Parallel Merge)**| $238$ ($97,88\%$) — $0,8865$ | $192$ ($98,29\%$) — $0,8860$ | $235$ ($97,91\%$) — $0,8871$ | Hội tụ đồng nhất về điểm Fitness $0,8865 \pm 0,0004$. |
| **C1 (Coop Exchange)** | $197$ ($98,25\%$) — $0,8865$ | $212$ ($98,11\%$) — $0,8874$ | $203$ ($98,19\%$) — $0,8870$ | Nén mạnh và đồng đều nhất ($197 - 212$ ảnh, $\text{DR} = 98,18\%$). |
| **GA-only (B3)** | $3.213$ ($71,38\%$) — $0,8028$ | $3.280$ ($70,79\%$) — $0,8010$ | $3.194$ ($71,55\%$) — $0,8039$ | Duy trì tập con quy mô vừa ($3.194 - 3.280$ ảnh, $\text{Cov} > 0,892$). |
| **PO-only (B4)** | $221$ ($98,03\%$) — $0,8874$ | $223$ ($98,01\%$) — $0,8861$ | $227$ ($97,98\%$) — $0,8873$ | Hội tụ nhanh về vùng $221 - 227$ ảnh ($\text{DR} = 98,01\%$). |

---

## 4. Kết quả Đánh giá Tầng 2: Hiệu năng Huấn luyện Hạ nguồn trên 3 Kiến trúc CNN

### Bảng 4.1: Bảng tổng hợp toàn diện trên cả 3 mô hình CNN (`descriptive_statistics.csv`)

| Phương pháp | Mô hình CNN | Tỷ lệ rút gọn $\text{DR}$ (%) | Test Accuracy ($\text{Mean} \pm \text{Std} \%$) | Test Accuracy ($\text{Median} \text{ [IQR]} \%$) | Macro F1-Score ($\text{Mean} \pm \text{Std} \%$) | Tỷ lệ duy trì $\text{ARR}$ (%) | Tăng tốc $\text{Speedup}$ ($\times$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Dataset (B0)** | `resnet18` | $0,00\%$ | $90,27 \pm 0,00\%$ | $90,27 \text{ [0,00]}\%$ | $90,41 \pm 0,00\%$ | $100,00\%$ | $1,00\times$ |
| **Full Dataset (B0)** | `mobilenet_v3` | $0,00\%$ | $88,93 \pm 0,00\%$ | $88,93 \text{ [0,00]}\%$ | $89,25 \pm 0,00\%$ | $100,00\%$ | $1,00\times$ |
| **Full Dataset (B0)** | `densenet121` | $0,00\%$ | $90,57 \pm 0,00\%$ | $90,57 \text{ [0,00]}\%$ | $90,72 \pm 0,00\%$ | $100,00\%$ | $1,00\times$ |
| **B3 — GA-only** | `resnet18` | $71,24\%$ | **$89,35 \pm 0,39\%$** | **$89,13 \text{ [0,44]}\%$** | **$89,55 \pm 0,40\%$** | **$98,99\%$** | $1,57\times$ |
| **B3 — GA-only** | `mobilenet_v3` | $71,24\%$ | **$87,79 \pm 0,24\%$** | **$87,87 \text{ [0,28]}\%$** | **$88,13 \pm 0,24\%$** | **$98,72\%$** | $1,36\times$ |
| **B3 — GA-only** | `densenet121` | $71,24\%$ | **$88,99 \pm 0,09\%$** | **$89,03 \text{ [0,10]}\%$** | **$89,22 \pm 0,12\%$** | **$98,25\%$** | $1,44\times$ |
| **S2 — Alt PO→GA** | `resnet18` | $97,57\%$ | $75,49 \pm 1,80\%$ | $76,03 \text{ [2,15]}\%$ | $75,80 \pm 1,90\%$ | $83,63\%$ | $3,03\times$ |
| **S2 — Alt PO→GA** | `mobilenet_v3` | $97,57\%$ | **$79,81 \pm 0,67\%$** | $79,93 \text{ [0,82]}\%$ | **$79,72 \pm 0,76\%$** | **$89,75\%$** | $2,38\times$ |
| **S2 — Alt PO→GA** | `densenet121` | $97,57\%$ | **$79,10 \pm 2,57\%$** | $78,07 \text{ [3,02]}\%$ | **$79,35 \pm 2,61\%$** | **$87,34\%$** | $2,63\times$ |
| **S1 — Alt GA→PO** | `resnet18` | $97,84\%$ | $74,97 \pm 1,46\%$ | $75,97 \text{ [1,56]}\%$ | $75,22 \pm 1,54\%$ | $83,05\%$ | $2,41\times$ |
| **S1 — Alt GA→PO** | `mobilenet_v3` | $97,84\%$ | $79,59 \pm 1,85\%$ | **$80,57 \text{ [2,10]}\%$** | $79,66 \pm 2,05\%$ | $89,50\%$ | $2,28\times$ |
| **S1 — Alt GA→PO** | `densenet121` | $97,84\%$ | $78,93 \pm 0,27\%$ | **$78,93 \text{ [0,33]}\%$** | $79,10 \pm 0,33\%$ | $87,15\%$ | $2,45\times$ |
| **P1 — Parallel** | `resnet18` | $98,03\%$ | $72,02 \pm 4,20\%$ | $73,90 \text{ [4,88]}\%$ | $72,39 \pm 4,26\%$ | $79,79\%$ | $3,52\times$ |
| **P1 — Parallel** | `mobilenet_v3` | $98,03\%$ | $78,21 \pm 2,14\%$ | $78,97 \text{ [2,54]}\%$ | $78,17 \pm 2,22\%$ | $87,95\%$ | $2,80\times$ |
| **P1 — Parallel** | `densenet121` | $98,03\%$ | $77,02 \pm 1,44\%$ | $77,80 \text{ [1,63]}\%$ | $77,29 \pm 1,54\%$ | $85,04\%$ | $2,76\times$ |
| **S4 — Block PO→GA**| `resnet18` | $97,98\%$ | $74,55 \pm 1,20\%$ | $74,17 \text{ [1,44]}\%$ | $74,54 \pm 1,29\%$ | $82,58\%$ | $3,57\times$ |
| **S4 — Block PO→GA**| `mobilenet_v3` | $97,98\%$ | $77,99 \pm 0,53\%$ | $78,17 \text{ [0,63]}\%$ | $77,91 \pm 0,73\%$ | $87,70\%$ | $2,66\times$ |
| **S4 — Block PO→GA**| `densenet121` | $97,98\%$ | $76,71 \pm 0,48\%$ | $77,03 \text{ [0,52]}\%$ | $76,78 \pm 0,81\%$ | $84,70\%$ | $2,80\times$ |
| **B4 — PO-only** | `resnet18` | $98,01\%$ | $75,03 \pm 1,12\%$ | $75,60 \text{ [1,28]}\%$ | $75,17 \pm 0,97\%$ | $83,12\%$ | $2,00\times$ |
| **B4 — PO-only** | `mobilenet_v3` | $98,01\%$ | $77,94 \pm 0,39\%$ | $77,77 \text{ [0,45]}\%$ | $77,92 \pm 0,50\%$ | $87,64\%$ | $1,46\times$ |
| **B4 — PO-only** | `densenet121` | $98,01\%$ | $77,42 \pm 0,56\%$ | $77,50 \text{ [0,69]}\%$ | $77,81 \pm 0,59\%$ | $85,49\%$ | $1,71\times$ |
| **S3 — Block GA→PO**| `resnet18` | $98,10\%$ | $60,04 \pm 18,56\%$ | $73,03 \text{ [19,75]}\%$ | $58,93 \pm 20,03\%$ | $66,51\%$ | **$3,86\times$** |
| **S3 — Block GA→PO**| `mobilenet_v3` | $98,10\%$ | $77,65 \pm 1,50\%$ | $77,13 \text{ [1,78]}\%$ | $77,57 \pm 1,56\%$ | $87,32\%$ | $2,83\times$ |
| **S3 — Block GA→PO**| `densenet121` | $98,10\%$ | $72,27 \pm 10,53\%$ | $79,03 \text{ [11,48]}\%$ | $71,74 \pm 11,63\%$ | $79,79\%$ | $2,93\times$ |
| **C1 — Cooperative**| `resnet18` | $98,18\%$ | $68,06 \pm 5,70\%$ | $70,90 \text{ [6,53]}\%$ | $67,22 \pm 6,18\%$ | $75,39\%$ | $3,71\times$ |
| **C1 — Cooperative**| `mobilenet_v3` | $98,18\%$ | $75,62 \pm 1,66\%$ | $75,07 \text{ [1,97]}\%$ | $75,46 \pm 1,74\%$ | $85,03\%$ | $2,71\times$ |
| **C1 — Cooperative**| `densenet121` | $98,18\%$ | $74,16 \pm 2,34\%$ | $75,50 \text{ [2,61]}\%$ | $73,93 \pm 2,13\%$ | $81,88\%$ | $2,86\times$ |
| **B1 — Random** | `resnet18` | $94,62\%$ | $84,29 \pm 0,58\%$ | $84,00 \text{ [0,67]}\%$ | $84,52 \pm 0,52\%$ | $93,37\%$ | $2,16\times$ |
| **B1 — Random** | `mobilenet_v3` | $94,62\%$ | $84,60 \pm 0,74\%$ | $84,37 \text{ [0,89]}\%$ | $84,86 \pm 0,77\%$ | $95,13\%$ | $1,85\times$ |
| **B1 — Random** | `densenet121` | $94,62\%$ | $86,13 \pm 0,32\%$ | $86,23 \text{ [0,38]}\%$ | $86,35 \pm 0,40\%$ | $95,10\%$ | $1,92\times$ |
| **B2 — Stratified** | `resnet18` | $94,62\%$ | $85,20 \pm 0,12\%$ | $85,23 \text{ [0,15]}\%$ | $85,40 \pm 0,16\%$ | $94,38\%$ | $3,10\times$ |
| **B2 — Stratified** | `mobilenet_v3` | $94,62\%$ | $84,61 \pm 0,59\%$ | $84,53 \text{ [0,72]}\%$ | $84,87 \pm 0,57\%$ | $95,14\%$ | $2,17\times$ |
| **B2 — Stratified** | `densenet121` | $94,62\%$ | $86,28 \pm 0,54\%$ | $86,40 \text{ [0,65]}\%$ | $86,52 \pm 0,56\%$ | $95,26\%$ | $2,30\times$ |

---

### Bảng 4.2: Các Phiên chạy Đạt Hiệu năng Đỉnh (Peak Runs) trong `experiment_results.csv`

| Kỷ lục ghi nhận | Phương pháp & Seed | Mô hình CNN | Số ảnh giữ lại ($M / 11.228$) | Tỷ lệ rút gọn ($\text{DR}\%$) | Test Accuracy (%) | Macro F1 (%) | Tỷ lệ duy trì ($\text{ARR}\%$) | Tăng tốc ($\text{Speedup}$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Giữ nguyên hiệu năng gốc cao nhất** | `GA-only` (Seed 123) | `resnet18` | $3.280$ ảnh | $70,79\%$ | **$89,90\%$** | **$90,11\%$** | **$99,59\%$** | $1,78\times$ |
| **Nén $>97\%$ có Test Acc cao nhất (DenseNet)** | `S2 (Alt PO→GA)` (Seed 123) | `densenet121` | **$287$ ảnh** | **$97,44\%$** | **$82,63\%$** | **$82,98\%$** | **$91,23\%$** | **$2,65\times$** |
| **Nén $>97,5\%$ có ARR cao nhất (MobileNet)**| `S1 (Alt GA→PO)` (Seed 123) | `mobilenet_v3`| **$236$ ảnh** | **$97,90\%$** | **$81,20\%$** | **$81,50\%$** | **$91,31\%$** | $1,93\times$ |
| **Nén $>98\%$ có Test Acc cao nhất** | `S2 (Alt PO→GA)` (Seed 456) | `mobilenet_v3`| **$222$ ảnh** | **$98,02\%$** | **$80,57\%$** | **$80,38\%$** | **$90,60\%$** | $2,39\times$ |
| **Điểm Fitness 4 thành phần cao nhất** | `S3 (Block GA→PO)` (Seed 42) | `mobilenet_v3`| **$226$ ảnh** | **$97,99\%$** | **$79,70\%$** | **$79,65\%$** | **$89,62\%$** | $2,78\times$ |

---

## 5. Kiểm định Thống kê Phi tham số Wilcoxon Signed-Rank Test

| Phương pháp đối sánh (so với `C1 — Coop Exchange`) | `resnet18` ($p$-value) | `mobilenet_v3` ($p$-value) | `densenet121` ($p$-value) | Ghi chú thống kê ($n = 3$ seeds) |
| :--- | :---: | :---: | :---: | :--- |
| **S1 (Alt GA→PO)** | $0,2500$ | $0,2500$ | $0,2500$ | Đạt ngưỡng $p_{\min} = 2/2^3 = 0,2500$ (vượt C1 trên cả 3/3 seeds). |
| **S2 (Alt PO→GA)** | $0,2500$ | $0,2500$ | $0,2500$ | Đạt ngưỡng $p_{\min} = 0,2500$ (vượt C1 trên cả 3/3 seeds). |
| **S3 (Block GA→PO)** | $0,7500$ | $0,5000$ | $1,0000$ | Tương đương C1 do cùng nằm ở vùng nén sâu $> 98,1\%$. |
| **S4 (Block PO→GA)** | $0,2500$ | $0,2500$ | $0,5000$ | Ổn định hơn C1 trên `resnet18` và `mobilenet_v3`. |
| **P1 (Parallel Merge)** | $0,7500$ | $0,5000$ | $0,5000$ | Dao động tương đương ở mức rút gọn $98,03\%$. |
| **GA-only (B3)** | $0,2500$ | $0,2500$ | $0,2500$ | Đạt ngưỡng $p_{\min} = 0,2500$ (vượt C1 trên cả 3/3 seeds). |
| **PO-only (B4)** | $0,2500$ | $0,5000$ | $0,2500$ | Tương đương các biến thể nén $> 98\%$. |
| **Random Selection (B1)** | $0,2500$ | $0,2500$ | $0,2500$ | Cấp phát $M = 604$ ảnh (gấp $2,96$ lần số ảnh của C1). |
| **Stratified Random (B2)** | $0,2500$ | $0,2500$ | $0,2500$ | Cấp phát $M = 604$ ảnh (gấp $2,96$ lần số ảnh của C1). |

---

## 6. Tổng kết các Phát hiện Cốt lõi từ Thực nghiệm

1. **Khả năng tối ưu hóa hàm mục tiêu 4 thành phần và tốc độ thực thi**:
   Tất cả sáu cấu hình lai ghép PO–GA (`S1–S4, P1, C1`) và `PO-only` đều đạt điểm Fitness tổng hợp từ **$0,8862$ đến $0,8875$**, vượt qua `GA-only` ($0,8026$) và hai phương pháp lấy mẫu ngẫu nhiên ($0,8765 - 0,8796$). Đặc biệt, cấu hình lai xen kẽ **`S2 (Alt PO→GA)`** hoàn thành $100$ vòng lặp chỉ trong **$105,81\text{ s}$** (nhanh gấp **$3,19$ lần** so với `GA-only` mất $337,48\text{ s}$).

2. **Hai chế độ vận hành bổ trợ nhau rõ rệt**:
   * **Chế độ Bảo toàn Độ chính xác (`GA-only`, $\text{DR} = 71,24\%$ — giữ lại $3.229$ ảnh)**: Loại bỏ hơn $71\%$ tập dữ liệu huấn luyện nhưng vẫn giữ được **$98,99\%$** hiệu năng gốc trên `ResNet-18` ($89,35 \pm 0,39\%$, đạt đỉnh $89,90\%$), **$98,72\%$** trên `MobileNetV3-Small` ($87,79 \pm 0,24\%$) và **$98,25\%$** trên `DenseNet-121` ($88,99 \pm 0,09\%$).
   * **Chế độ Nén Cực đại (`S2 — Alt PO→GA` và `S1 — Alt GA→PO`, $\text{DR} = 97,57\% - 97,84\%$ — chỉ giữ lại $243 - 273$ ảnh)**: Giảm quy mô tập dữ liệu đi hơn **$40$ lần** (chỉ còn khoảng $40 - 45$ ảnh mỗi lớp), nhưng vẫn duy trì độ chính xác trung bình **$79,81 \pm 0,67\%$** trên `MobileNetV3-Small` ($\text{ARR} = 89,75\%$) và đạt đỉnh **$82,63\%$** ($\text{ARR} = 91,23\%$) trên `DenseNet-121`, đồng thời rút ngắn thời gian huấn luyện từ **$2,38\times$ đến $3,03\times$**.

3. **Kiến trúc Lai ghép Tuần tự Xen kẽ (`S2` và `S1`) hiệu quả hơn Chia khối (`S3, S4`) và Hợp tác nén sâu (`C1`)**:
   Khi không đặt ngưỡng chặn dưới cho kích thước tập con $M$, việc trao đổi cá thể ưu tú liên tục trong `C1` hoặc chạy `PO` kéo dài $50$ vòng liên tiếp ở cuối `S3` khiến tập con bị nén quá sâu xuống mức $193 - 204$ ảnh ($\text{DR} = 98,18\%$), làm suy giảm số lượng batch huấn luyện cho mạng lớn như `ResNet-18`. Ngược lại, cơ chế chuyển pha ngắn $10$ vòng lặp của **`S2 (Alt PO→GA)`** và **`S1 (Alt GA→PO)`** giữ tập con ở ngưỡng an toàn hơn ($243 - 273$ ảnh), mang lại độ chính xác kiểm thử cao nhất trong toàn bộ các kiến trúc lai ghép PO–GA trên cả ba mô hình CNN.
