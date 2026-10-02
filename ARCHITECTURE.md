# KIẾN TRÚC HỆ THỐNG RÚT GỌN DỮ LIỆU ẢNH HYBRID PO–GA (`CNTT-KLCN140`)

Tài liệu này mô tả chi tiết kiến trúc tổng thể, luồng dữ liệu 5 giai đoạn (End-to-End Dataflow), sơ đồ tương tác giữa các mô-đun và cơ chế vận hành của 3 nhóm kiến trúc lai hóa **Parrot Optimizer – Genetic Algorithm (Hybrid PO–GA)** trong dự án.

---

## 1. Sơ đồ Luồng Xử lý Tổng thể 5 Giai đoạn (End-to-End Pipeline)

Hệ thống được thiết kế theo mô hình **Filter-based Coreset Selection** tách biệt hoàn toàn giữa Tầng Tối ưu hóa Tổ hợp (Giai đoạn 3) và Tầng Huấn luyện Đánh giá Học sâu (Giai đoạn 4), giúp tìm tập ảnh đại diện chỉ trong vài chục giây mà không cần huấn luyện lại mạng CNN ở từng vòng lặp tiến hóa.

```mermaid
flowchart TD
    subgraph G1["Giai đoạn 1: Phân hoạch Dữ liệu Chuẩn (data/dataset.py)"]
        A["Intel Image Dataset<br/>seg_train (14,034 ảnh) & seg_test (3,000 ảnh)"] --> B["Tập Train gốc D_train<br/>11,228 ảnh (80% seg_train, seed=42)"]
        A --> C["Tập Validation D_val<br/>2,806 ảnh (20% seg_train)"]
        A --> D["Tập Test Độc lập D_test<br/>3,000 ảnh (seg_test)"]
    end

    subgraph G2["Giai đoạn 2: Trích xuất Đặc trưng & Ma trận Khoảng cách"]
        B --> E["ResNet-18 Pretrained Backbone<br/>(Loại bỏ tầng FC -> Vector 512 chiều)"]
        E --> F["Bộ nhớ đệm Đặc trưng<br/>features/train_features_train80.npy (11,228 × 512)"]
        F --> G["Ma trận Khoảng cách Cosine Chuẩn hóa GPU<br/>D_norm = dist(f_i, f_j) / dist_max ∈ [0, 1]"]
    end

    subgraph G3["Giai đoạn 3: Tối ưu hóa Tập con Hybrid PO–GA (optimization/)"]
        G --> H["Hàm Mục tiêu 4 Thành phần (fitness.py)<br/>F(X) = 0.1·Div + 0.3·Cov + 0.2·Bal + 0.4·Com"]
        H --> I["Khung Lai ghép PO–GA (2,000 FEs)<br/>Sequential (S1–S4) | Parallel (P1) | Cooperative (C1)"]
        I --> J["Tập con Tối ưu S (M ảnh đại diện)<br/>subsets/subset_&lt;method&gt;_seed&lt;seed&gt;.npy"]
    end

    subgraph G4["Giai đoạn 4: Huấn luyện & Đánh giá Hạ nguồn (models/ & training/)"]
        J --> K["Subsampled Train DataLoader<br/>(Chỉ giữ lại M ảnh có x_i = 1)"]
        K --> L1["ResNet-18<br/>(11.18M params)"]
        K --> L2["MobileNetV3-Small<br/>(1.52M params)"]
        K --> L3["DenseNet-121<br/>(6.96M params)"]
        C --> L1 & L2 & L3
        D --> M["Đánh giá trên Tập Test Độc lập (3,000 ảnh)<br/>Test Acc, Macro F1, DR%, ARR%, Speedup"]
        L1 & L2 & L3 --> M
    end

    subgraph G5["Giai đoạn 5: Thống kê Đa Hạt giống & Kiểm định (run_poga.py)"]
        M --> N["Tổng hợp Mean ± Std, Median [IQR]<br/>Kiểm định Wilcoxon Signed-Rank Test & Biểu đồ Hội tụ"]
    end
```

---

## 2. Cấu trúc Thư mục và Phân tầng Mô-đun (Module Architecture)

```text
KhoaLuanDatasetReduction/
├── configs/
│   ├── __init__.py
│   └── config.py                  # Quản lý tập trung BaselineConfig, POGAConfig, GAConfig, POConfig, FitnessConfig
│
├── data/
│   ├── __init__.py
│   └── dataset.py                 # Phân hoạch Train/Val/Test, lọc tập con theo mask .npy & trích xuất đặc trưng ResNet-18
│
├── features/
│   ├── train_features_train80.npy # Vector đặc trưng 512 chiều tiền trích xuất của 11,228 ảnh Train (~21 MB)
│   └── train_labels_train80.npy   # Nhãn phân lớp (0..5) tương ứng của 11,228 ảnh Train
│
├── optimization/
│   ├── __init__.py
│   ├── fitness.py                 # Ma trận khoảng cách Cosine chuẩn hóa trên GPU & Hàm Fitness 4 thành phần (Div, Cov, Bal, Com)
│   ├── genetic_algorithm.py       # Giải thuật Di truyền (Tournament Selection, Uniform Crossover, Adaptive Mutation, Elitism)
│   ├── parrot_optimizer.py        # Thuật toán Đàn vẹt truyền thống (4 hành vi St=1..4, Lévy Flight) + Ánh xạ nhị phân |tanh(Δx)|
│   └── hybrid_poga.py             # 6 cấu hình lai hóa (S1, S2, S3, S4, P1, C1) & 2 phương pháp đối chuẩn (Random, Stratified)
│
├── models/
│   ├── __init__.py
│   └── model.py                   # Khởi tạo 3 kiến trúc CNN: ResNet-18, MobileNetV3-Small, DenseNet-121
│
├── training/
│   ├── __init__.py
│   └── trainer.py                 # Vòng lặp huấn luyện Adam + StepLR, lưu best_model_weights theo Val Acc & chấm điểm Test Set
│
├── utils/
│   ├── __init__.py
│   └── utils.py                   # Cố định seed tái lập, vẽ biểu đồ hội tụ, learning curves & confusion matrix
│
├── train.py                       # Kịch bản huấn luyện mốc chuẩn Full Dataset (B0) hoặc huấn luyện đơn lẻ 1 tập con
├── run_poga.py                    # Kịch bản điều phối thực nghiệm toàn diện (Tầng 1 + Tầng 2 + Thống kê Wilcoxon)
├── kaggle_poga_notebook.ipynb     # Notebook tự động hóa thực thi trên đám mây Kaggle GPU
├── BAO_CAO_KHOA_LUAN_CNTT_KLCN140.md        # Báo cáo toàn văn Khóa luận Tốt nghiệp (Chương 1 -> Chương 8)
└── BAO_CAO_KET_QUA_THUC_NGHIEM_POGA.md      # Báo cáo chuyên biệt phân tích kết quả thực nghiệm Kaggle
```

---

## 3. Kiến trúc Chi tiết Mô-đun Tối ưu hóa (`optimization/`)

### 3.1. Cơ chế Ánh xạ Nhị phân trong Parrot Optimizer (`parrot_optimizer.py`)

Thuật toán Parrot Optimizer truyền thống (Lian et al., 2024) cập nhật vị trí cá thể trên miền liên tục $X_i^{\text{cont}} \in \mathbb{R}^N$ thông qua 4 phương trình hành vi ($S_t \in \{1, 2, 3, 4\}$). Để biểu diễn quyết định chọn ($1$) hoặc loại bỏ ($0$) từng bức ảnh, hệ thống sử dụng hàm truyền chữ V ($V(\Delta x) = |\tanh(\Delta x)|$) ở bước cuối của mỗi vòng lặp:

```mermaid
flowchart LR
    A["Vị trí hiện tại<br/>X_current ∈ {0, 1}^N"] --> B["Cập nhật PO Truyền thống<br/>(St = 1: Kiếm ăn | St = 2: Đậu<br/>St = 3: Giao tiếp | St = 4: Sợ người lạ)"]
    B --> C["Độ dịch chuyển liên tục<br/>ΔX = X_cont - X_current"]
    C --> D["Xác suất lật bit<br/>P_flip = |tanh(ΔX)| ∈ [0, 1)"]
    D --> E["Đảo bit (0 ↔ 1) nếu rand() < P_flip<br/>Giữ nguyên nếu rand() ≥ P_flip"]
    E --> F["Chọn lọc Tham lam (Greedy Selection)<br/>Chỉ cập nhật nếu F(X_new) > F(X_current)"]
```

### 3.2. Ba Nhóm Kiến trúc Lai hóa PO–GA (`hybrid_poga.py`)

Tất cả 6 cấu hình lai hóa đều nhận cùng một quần thể khởi tạo $P_0$ tại mỗi `seed` và chia sẻ cùng tổng ngân sách đánh giá hàm mục tiêu là **$\text{Total FEs} = 2.000$** ($P = 20, T = 100$):

```mermaid
flowchart TD
    P0["Quần thể Khởi tạo Chung P_0<br/>(P = 20 cá thể, N = 11,228 chiều, init_ratio = 0.30)"] --> SEQ & PAR & COOP

    subgraph SEQ["1. Kiến trúc Tuần tự (Sequential Hybrid)"]
        S1["S1 (Alt GA→PO):<br/>5 chu kỳ × [GA(10 vòng) → PO(10 vòng)]"]
        S2["S2 (Alt PO→GA):<br/>5 chu kỳ × [PO(10 vòng) → GA(10 vòng)]"]
        S3["S3 (Block GA→PO):<br/>GA chạy 50 vòng đầu → PO chạy 50 vòng cuối"]
        S4["S4 (Block PO→GA):<br/>PO chạy 50 vòng đầu → GA chạy 50 vòng cuối"]
    end

    subgraph PAR["2. Kiến trúc Song song Độc lập (Parallel Hybrid)"]
        P1["P1 (Parallel Merge):<br/>Nhánh GA (50 vòng) || Nhánh PO (50 vòng)<br/>──► Hợp nhất (40 cá thể) & Chọn Top-20"]
    end

    subgraph COOP["3. Kiến trúc Hợp tác Đồng tiến hóa (Cooperative Hybrid)"]
        C1["C1 (Bidirectional Elite Exchange):<br/>GA (50 vòng) ↔ PO (50 vòng)<br/>──► Trao đổi k = 3 cá thể ưu tú mỗi 5 vòng<br/>(Kèm kiểm soát chống trùng lặp Hamming > 0)"]
    end
```

---

## 4. Cơ chế Tối ưu Hiệu năng Tính toán (Performance Engineering)

1. **Tiền trích xuất và Lưu đệm Đặc trưng (`features/train_features_train80.npy`)**:
   - Toàn bộ $11.228$ ảnh của tập huấn luyện $D_{\text{train}}$ được trích xuất sẵn một lần duy nhất qua mạng `ResNet-18` thành ma trận `float32` kích thước $11.228 \times 512$ ($\approx 21\text{ MB}$).
   - Các phiên chạy tối ưu hóa nạp trực tiếp ma trận này từ ổ đĩa vào RAM trong $< 0,05$ giây, loại bỏ hoàn toàn nút thắt đọc/giải mã file ảnh JPEG.
2. **Tiền tính toán Khoảng cách bằng Nhân Ma trận BLAS (`precompute_distance_matrix`)**:
   - Ma trận khoảng cách Cosine $11.228 \times 11.228$ được tính bằng phép nhân ma trận chuẩn hóa $\hat{F}\hat{F}^\top$ (`float32` BLAS) và chia trực tiếp cho $\text{dist}_{\max}$ chỉ trong $\approx 1,2$ giây.
3. **Lưu đệm Tensor Khoảng cách trên Bộ nhớ GPU (`_get_dist_tensor`)**:
   - Ma trận khoảng cách $\hat{D}$ ($\approx 504\text{ MB}$) được giữ cố định trên VRAM GPU (`cuda`), cho phép hai phép toán tốn kém nhất là tính Độ đa dạng ($\text{Div}$ — cắt ma trận con $M \times M$) và Độ bao phủ ($\text{Cov}$ — tìm $\min$ trên ma trận $N \times M$) thực thi song song trên nhân CUDA.
