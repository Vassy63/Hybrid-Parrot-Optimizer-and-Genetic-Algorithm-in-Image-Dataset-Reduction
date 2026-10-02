---
name: feature-distance-pipeline
description: >-
  Standard procedure for extracting 512-dimensional image feature representations using deep learning backbones (ResNet-18) and computing pairwise distance matrices (Euclidean, Cosine). Use when preprocessing image data for optimization algorithms (PO-GA, clustering, subset selection) and managing .npy feature caches.
---

# Feature & Distance Matrix Pipeline

Quy chuẩn trích xuất vector đặc trưng và tính toán ma trận khoảng cách hình học phục vụ các thuật toán tối ưu hóa rút gọn dữ liệu (PO-GA, K-Means, Coreset).

## 1. Trích xuất Đặc trưng (Feature Extraction)
- **Backbone mặc định:** ResNet-18 tiền huấn luyện trên ImageNet (`ResNet18_Weights.DEFAULT`), bỏ tầng `fc` cuối cùng để thu được vector đặc trưng $D = 512$ chiều.
- **Thiết bị chạy:** GPU RTX 4060 với `torch.no_grad()` và chế độ `eval()`.
- **Đầu ra:** Mảng NumPy `train_features.npy` kích thước $(N, 512)$ và `train_labels.npy` kích thước $(N,)$.

## 2. Tính Ma trận Khoảng cách (Pairwise Distance Matrix)
- **Khoảng cách hình học:** Sử dụng Euclidean Distance hoặc Cosine Distance:
  $$D_{ij} = ||\mathbf{f}_i - \mathbf{f}_j||_2$$
- **Lưu trữ ma trận:** `dist_matrix.npy` kích thước $(N, N)$ kiểu `float32`.
- **Cơ chế Caching:** Kiểm tra sự tồn tại của file cache trước khi tính toán. Tuyệt đối không tính toán lại nếu file `.npy` đã tồn tại để tiết kiệm thời gian GPU/CPU.

## 3. Quản lý Bộ nhớ & Tối ưu trên Windows
- Khống chế số luồng C-level để tránh lỗi cạn kiệt tài nguyên (Windows Error #1450):
  ```python
  import os
  os.environ["OMP_NUM_THREADS"] = "1"
  os.environ["MKL_NUM_THREADS"] = "1"
  ```
- Định dạng lưu trữ: `numpy.save(..., allow_pickle=False)` để đảm bảo tính an toàn và tương thích.
