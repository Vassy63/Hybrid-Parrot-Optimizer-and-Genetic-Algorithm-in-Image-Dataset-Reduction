---
name: full-dataset-workflow
description: >-
  Standard operating procedure for training and evaluating baseline deep learning models (ResNet-18, MobileNetV3, DenseNet-121) on the full 100% original dataset (Intel Image Dataset). Use when running baseline experiments, benchmarking full dataset performance, or generating ground-truth comparative metrics.
---

# Full Dataset Baseline Workflow

Hệ thống quy chuẩn chạy thực nghiệm và xuất mốc đối chuẩn trên toàn bộ tập ảnh gốc.
Phân chia chuẩn khoa học: Train (80% seg_train ≈ 11.228 ảnh) / Val (20% seg_train ≈ 2.806 ảnh) / Test (seg_test = 3.000 ảnh độc lập).

## 1. Môi trường Thực thi Bắt buộc
- **Python / Conda:** Luôn sử dụng môi trường `gamesearcher` để kích hoạt GPU CUDA:
  `& "D:\Program Files\Miniconda\envs\gamesearcher\python.exe" train.py ...`
- **Phần cứng:** NVIDIA GeForce RTX 4060 Laptop GPU.
- **Batch Size Khuyến nghị:** `64` (tối ưu VRAM và tốc độ đạt ~8-10 it/s).
- **Epochs Khuyến nghị:** `5` đến `10` epochs để mô hình hội tụ ổn định.

## 2. Các Lệnh Huấn luyện Chuẩn
- **Chạy toàn bộ 3 mô hình đối sánh liên tiếp (Khuyên dùng):**
  ```bash
  & "D:\Program Files\Miniconda\envs\gamesearcher\python.exe" train.py --model all --epochs 5 --batch_size 64
  ```
- **Chạy từng mô hình riêng lẻ:**
  ```bash
  & "D:\Program Files\Miniconda\envs\gamesearcher\python.exe" train.py --model resnet18 --epochs 5 --batch_size 64
  & "D:\Program Files\Miniconda\envs\gamesearcher\python.exe" train.py --model mobilenet_v3 --epochs 5 --batch_size 64
  & "D:\Program Files\Miniconda\envs\gamesearcher\python.exe" train.py --model densenet121 --epochs 5 --batch_size 64
  ```
- **Kiểm tra nhanh (Fast / Dry-run):** Thêm cờ `--num_samples 100 --epochs 1`.

## 3. Quy tắc Quản lý Thư mục Kết quả (Bắt buộc)
Mọi kết quả của một phiên chạy **phải nằm gọn trong đúng 1 thư mục duy nhất**:
`outputs/run_<model_or_name>_<YYYYMMDD_HHMMSS>/`

Cấu trúc bắt buộc bên trong thư mục phiên chạy:
- `benchmark_summary.csv`: Bảng tổng quan (Model, Params, Test Acc, Val Acc, Macro F1, Train Acc, Latency, Time).
- `per_class_f1.csv`: Bảng F1-Score của 6 lớp theo từng mô hình (tính trên Test Set).
- `plots/`: Toàn bộ file ảnh, tuyệt đối không để ảnh nằm ngoài:
  - `comparison_curves.png`: Đồ thị so sánh trực tiếp Acc & Loss giữa 3 mô hình.
  - `confusion_matrices.png`: 3 Ma trận nhầm lẫn xếp ngang đối chiếu (Independent Test Set).
  - `class_distribution.png`: Biểu đồ phân bố 6 lớp tập train.
  - `<model>_curve.png`: Biểu đồ Train vs Val riêng từng mô hình.
  - `<model>_confusion_matrix.png`: Ma trận nhầm lẫn riêng từng mô hình (Test Set).
- `weights/`: Chứa các file checkpoint trọng số `.pth` (`resnet18_best.pth`, v.v.).
- `logs/`: Chứa log chi tiết từng epoch và metadata phần cứng/siêu tham số.
