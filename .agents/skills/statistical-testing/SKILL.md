---
name: statistical-testing
description: >-
  Standard operating procedure for conducting multi-seed experimental runs and statistical significance testing (Wilcoxon signed-rank test, Paired t-test, Mean +/- Std) for academic thesis reporting. Use when proving that a proposed dataset reduction method is statistically superior to baseline methods.
---

# Statistical Testing & Multi-Seed Benchmark Workflow

Quy chuẩn đánh giá độ tin cậy khoa học và kiểm định ý nghĩa thống kê cho kết quả khóa luận tốt nghiệp.

## 1. Nguyên tắc Chạy Đa Hạt giống (Multi-Seed Protocol)
Không bao giờ lấy kết quả của 1 lần chạy đơn lẻ ngẫu nhiên để kết luận. Bắt buộc chạy qua tối thiểu **3 đến 5 hạt giống ngẫu nhiên**:
- **Danh sách hạt giống chuẩn:** `[42, 123, 456, 789, 2026]`
- **Công thức tính toán:**
  - Giá trị trung bình: $\mu = \frac{1}{K} \sum_{k=1}^K x_k$
  - Độ lệch chuẩn: $\sigma = \sqrt{\frac{1}{K-1} \sum_{k=1}^K (x_k - \mu)^2}$
  - Báo cáo kết quả dưới dạng: $\mu \pm \sigma$ (ví dụ: $87.45 \pm 0.32\%$).

## 2. Kiểm định Ý nghĩa Thống kê (Wilcoxon Signed-Rank Test)
Để chứng minh thuật toán đề xuất (ví dụ PO-GA) vượt trội hơn thuật toán đối chứng (Random Selection / K-Means) có ý nghĩa khoa học chứ không phải do may mắn:
- **Giả thuyết không ($H_0$):** Không có sự khác biệt có ý nghĩa thống kê giữa phương pháp đề xuất và baseline.
- **Mức ý nghĩa ($\alpha$):** Thường chọn $\alpha = 0.05$.
- **Quy tắc bác bỏ:**
  - Nếu $p\text{-value} < 0.05$: Bác bỏ $H_0$, kết luận phương pháp đề xuất vượt trội có ý nghĩa thống kê ($*$).
  - Nếu $p\text{-value} < 0.01$: Vượt trội rất mạnh ($**$).
  - Nếu $p\text{-value} \ge 0.05$: Khác biệt không có ý nghĩa thống kê ($ns$).

## 3. Trực quan hóa & Định dạng Bảng biểu Khóa luận
- **Biểu đồ Hộp (Boxplot - `statistical_boxplot.png`):** So sánh trung vị (median), tứ phân vị (IQR) và độ phân tán của từng phương pháp qua các seeds.
- **Bảng số liệu tổng hợp:**
  | Phương pháp | Tỷ lệ ảnh | ResNet-18 (Mean $\pm$ Std) | MobileNetV3 (Mean $\pm$ Std) | DenseNet-121 (Mean $\pm$ Std) | $p$-value (vs. Random) |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | Full Dataset | 100% | $89.20 \pm 0.15$ | $85.40 \pm 0.22$ | $88.90 \pm 0.18$ | - |
  | Random Selection | 30% | $82.10 \pm 0.85$ | $78.30 \pm 0.92$ | $81.50 \pm 0.78$ | Baseline |
  | **PO-GA (Ours)** | 30% | **$86.85 \pm 0.28$** | **$83.10 \pm 0.35$** | **$86.20 \pm 0.25$** | **$< 0.01$ ($**$)** |
