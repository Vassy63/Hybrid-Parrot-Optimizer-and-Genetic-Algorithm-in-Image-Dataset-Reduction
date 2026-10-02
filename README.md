# Báo cáo Tiến độ Tuần: Kiến trúc Module hóa Đa Tầng Huấn luyện Mô hình Phân loại Ảnh (Baseline)

Thư mục này được tổ chức theo chuẩn **kiến trúc phần mềm phân tầng rõ ràng (Multi-tier Modular Architecture)** dành riêng cho bài toán huấn luyện Baseline trực tiếp trên tập ảnh gốc (**Intel Image Dataset**).

Mỗi lần chạy sẽ tự động sinh ra một thư mục riêng biệt kèm mã nhận diện duy nhất (`Run ID` theo `timestamp` và tên tùy chọn), giúp bạn dễ dàng theo dõi, so sánh và nộp kết quả thực nghiệm mà không lo bị ghi đè.

---

## 📌 1. Cấu trúc Thư mục Phân tầng (`baseline_train/`)

```text
baseline_train/
├── configs/               # 📁 Quản lý cấu hình & siêu tham số
│   ├── __init__.py
│   └── config.py          # Dataclass BaselineConfig (paths, hyperparams, run_id, run_dir)
├── data/                  # 📁 Xử lý & nạp dữ liệu ảnh
│   ├── __init__.py
│   └── dataset.py         # Transforms, ImageFolder loader, chia batch, Fast Mode
├── models/                # 📁 Định nghĩa kiến trúc mạng nơ-ron
│   ├── __init__.py
│   └── model.py           # ResNet-18 Transfer Learning & Custom Classifier Head
├── training/              # 📁 Động cơ huấn luyện & đánh giá (Training Engine)
│   ├── __init__.py
│   └── trainer.py         # Class BaselineTrainer: train loop, val loop, checkpointing
├── utils/                 # 📁 Các hàm tiện ích
│   ├── __init__.py
│   └── utils.py           # Cố định seed, xuất run_metadata.json, training_log.csv, vẽ training_curve.png
├── outputs/               # 📁 Thư mục lưu kết quả thực nghiệm
│   └── run_<name>_<timestamp>/            # 🎯 DUY NHẤT 1 THƯ MỤC CHO MỖI PHIÊN CHẠY
│       ├── benchmark_summary.csv          # 📊 Bảng tổng quan (Model, Params, Val Acc, F1, Latency, Time)
│       ├── per_class_f1.csv               # 📊 Bảng điểm F1 từng lớp của 6 nhãn theo từng mô hình
│       ├── plots/                         # 🖼️ Thư mục gom toàn bộ ảnh biểu đồ và ma trận
│       │   ├── comparison_curves.png      # Đồ thị so sánh trực tiếp Acc & Loss giữa 3 mô hình
│       │   ├── confusion_matrices.png     # 3 ma trận nhầm lẫn xếp ngang đối chiếu trực diện
│       │   ├── class_distribution.png     # Phân bố số lượng ảnh 6 lớp tập gốc
│       │   ├── <model>_curve.png          # Biểu đồ Train vs Val riêng của từng mô hình
│       │   └── <model>_confusion_matrix.png # Ma trận nhầm lẫn riêng của từng mô hình
│       ├── weights/                       # 💾 Thư mục chứa các file trọng số (.pth)
│       │   ├── resnet18_best.pth
│       │   ├── mobilenet_v3_best.pth
│       │   └── densenet121_best.pth
│       └── logs/                          # 📝 Thư mục chứa log chi tiết từng epoch & metadata
├── train.py               # 🚀 Entrypoint CLI kết nối toàn bộ hệ thống
└── README.md              # 📄 Báo cáo & hướng dẫn chi tiết
```

---

## 🏷️ 2. Hệ thống Báo cáo Gom Gọn Ngăn Nắp

Sau khi chạy xong, toàn bộ sản phẩm của phiên chạy được gom gọn vào **1 thư mục duy nhất**:
1. **`benchmark_summary.csv`:** Bảng tính Excel chuẩn hình chữ nhật tóm lược các chỉ số cốt lõi.
2. **`per_class_f1.csv`:** Bảng điểm F1 từng lớp đối sánh giữa các mô hình.
3. **`plots/`:** Gom toàn bộ 9 file ảnh (cả biểu đồ đối sánh chung lẫn biểu đồ riêng từng mô hình), không để ảnh vứt bừa bãi ở ngoài.
4. **`weights/`:** Lưu riêng các file checkpoint trọng số nhị phân (`.pth`).
5. **`logs/`:** Lưu toàn bộ nhật ký chi tiết và metadata.

---

## 🚀 3. Hướng dẫn Thực thi

### Chạy kiểm tra nhanh cả 3 mô hình (Fast Mode):
```bash
python train.py --model all --num_samples 100 --epochs 1
```

### Chạy huấn luyện 1 mô hình cụ thể:
```bash
python train.py --model resnet18 --epochs 5 --batch_size 64
python train.py --model mobilenet_v3 --epochs 5 --batch_size 64
python train.py --model densenet121 --epochs 5 --batch_size 64
```

### Chạy đối sánh đầy đủ cả 3 mô hình (Xuất bảng tổng hợp benchmark_summary.csv):
```bash
python train.py --model all --epochs 5 --batch_size 64
```

### Tùy chỉnh các siêu tham số dòng lệnh:
* `--model`: Mô hình (`resnet18`, `mobilenet_v3`, `densenet121`, hoặc `all` để chạy cả 3).
* `--run_name`: Tên định danh tùy chọn cho lần chạy (ví dụ: `exp1`, `freeze_backbone`).
* `--train_dir`: Đường dẫn thư mục ảnh train (mặc định: `data/seg_train/seg_train`).
* `--test_dir`: Đường dẫn thư mục ảnh test độc lập (mặc định: `data/seg_test/seg_test`).
* `--val_split`: Tỷ lệ tách tập validation từ `seg_train` (mặc định: `0.2`, tức 80% Train / 20% Val).
* `--epochs`: Số lượt huấn luyện (mặc định: `5`).
* `--batch_size`: Kích thước batch (mặc định: `64`).
* `--lr`: Tốc độ học (mặc định: `0.001`).
* `--fine_tune`: Mở khóa toàn bộ các tầng mạng để fine-tune (mặc định: `False`).
* `--num_samples`: Giới hạn số ảnh train để kiểm tra nhanh.
