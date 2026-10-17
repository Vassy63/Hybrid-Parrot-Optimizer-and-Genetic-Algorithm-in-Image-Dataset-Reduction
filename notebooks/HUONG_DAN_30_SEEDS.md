# Hướng dẫn chia 30 seed cho 3 người trên Kaggle

## 1. Phân công

| Người | Notebook | Seed | Số lượt huấn luyện CNN |
|---|---|---|---:|
| 1 | [kaggle_30seeds_person1.ipynb](kaggle_30seeds_person1.ipynb) | 1–10 | 330 |
| 2 | [kaggle_30seeds_person2.ipynb](kaggle_30seeds_person2.ipynb) | 11–20 | 330 |
| 3 | [kaggle_30seeds_person3.ipynb](kaggle_30seeds_person3.ipynb) | 21–30 | 330 |

Mỗi seed chạy **Full Dataset B0 và 10 phương pháp rút gọn**: Random, Stratified,
GA, PO, S1, S2, S3, S4, P1, C1. Mỗi phương pháp được đánh giá trên
ResNet-18, MobileNetV3-Small và DenseNet-121: 33 lượt/seed, tổng cộng **990 lượt**.
Chia đều số công việc; thời gian thực tế phụ thuộc GPU và phiên Kaggle của mỗi người.

**Trước mắt chạy đợt 1, mỗi người 5 seed:**

| Đợt | Người 1 | Người 2 | Người 3 | Tổng seed | Lượt CNN/người |
|---|---|---|---|---:|---:|
| 1 (mặc định) | 1–5 | 11–15 | 21–25 | 15 | 165 |
| 2 (chạy sau) | 6–10 | 16–20 | 26–30 | 15 | 165 |

Đợt 1 dùng `SEEDS_TO_RUN = SEEDS[:5]`; đợt 2 đổi thành `SEEDS_TO_RUN = SEEDS[5:]`.
Mỗi đợt có thể cần nhiều phiên và vượt sang tuần sau nếu quota không đủ.
Công cụ gộp kết quả cuối vẫn yêu cầu đủ cả 30 seed sau hai đợt.

## 2. Cấu hình thống nhất

- 20 epoch, batch size 64, Adam, learning rate 0.001, weight decay 0.0001.
- ImageNet pretrained, đóng băng backbone (`fine_tune=False`), ảnh 224×224.
- Train/validation chia 80/20 từ `seg_train`, **split seed cố định 42**; `seg_test` độc lập.
- Seed tối ưu hóa và seed CNN thay đổi theo seed 1–30; các phương pháp trong cùng seed dùng cùng seed CNN.
- Population 20, total iterations 100, phase size 10, exchange k 3.
- Fitness theo snapshot: Div 0.15, Cov 0.45, Bal 0.20, Com 0.20.
- B0 được huấn luyện thực tế cho từng seed. ARR, Delta Acc và Speedup dùng B0 cùng model và seed.
- Cả ba người upload cùng file `KhoaLuanDatasetReduction_30seeds_source.zip` làm Kaggle Input; notebook không chứa `CODE_B64`.
- Notebook ghi SHA256 nội dung mã nguồn vào `DONE.json` để kiểm tra khi khôi phục/gộp.

Không sửa các giá trị trên giữa ba người. Đổi `SEEDS_TO_RUN` để chia nhỏ phiên
không làm thay đổi cấu hình thực nghiệm. Split cố định nghĩa là 30 lần đo biến thiên
do thuật toán chọn tập con và huấn luyện, không phải 30 cách chia dữ liệu khác nhau.

## 3. Chạy lần đầu

### Kiểm tra trước bằng CPU

Import [kaggle_cpu_smoke_test.ipynb](kaggle_cpu_smoke_test.ipynb), chọn **Accelerator: None**,
bật Internet và thêm cùng ZIP source + Intel dataset. Chạy **Run All**.
Notebook dùng 120 ảnh train, 1 seed, 1 epoch và 2 vòng tối ưu để chạy thật
đủ 11 phương pháp × 3 CNN. Không dùng mock model hoặc ảnh giả.

Cuối quá trình phải in **`CPU_SMOKE_PASS`** và tạo:

- `/kaggle/working/cpu_smoke/CPU_SMOKE_REPORT.json`.
- `/kaggle/working/cpu_smoke_seed1.zip`: đầy đủ CSV, log, biểu đồ, mask, metadata, 33 checkpoint.

Kiểm tra gồm nạp ZIP/thư mục source, nhận diện dataset, thư viện, trích đặc trưng,
thuật toán, huấn luyện/đánh giá CPU, đọc lại checkpoint, số dòng kết quả,
ZIP CRC và giải nén khôi phục. Chạy lại ô benchmark để kiểm tra bỏ qua seed đã xong và tạo lại ZIP.
Internet cần để tải trọng số pretrained nếu chưa cache; CPU không cần bật GPU.
Kết quả smoke test nằm riêng và **không được gộp vào benchmark 20 epoch**.
PASS trên CPU không xác nhận CUDA/VRAM, thời gian chạy toàn dataset, quota hay độ chính xác cuối cùng.

### Chạy benchmark GPU

1. Tải đúng notebook được phân công từ nhánh `jaxy` của repository.
2. Trên Kaggle chọn **New Notebook → File → Import Notebook**, chọn file `.ipynb`.
3. Bật **GPU** (T4 hoặc P100) và **Internet** để tải trọng số ImageNet. Nên dùng cùng loại GPU cho cả ba người khi so sánh thời gian.
4. Add Input: `puneet6060/intel-image-classification`. Chỉ thêm một bản dataset ảnh.
   Tạo thêm một Kaggle Dataset chứa [KhoaLuanDatasetReduction_30seeds_source.zip](KhoaLuanDatasetReduction_30seeds_source.zip),
   rồi thêm Dataset đó vào Input của cả ba notebook. Nếu Kaggle tự giải nén ZIP, notebook cũng nhận diện được source đã giải nén.
   Ô `SOURCE_ZIP = ''` tự tìm ZIP chứa `run_poga.py` và `train.py`, kể cả khi ZIP bị đổi tên hoặc có thư mục bọc ngoài.
   Nếu có nhiều bộ source, điền `SOURCE_ZIP` là đường dẫn cụ thể tới ZIP hoặc thư mục đã giải nén trong `/kaggle/input/...`.
   Nếu không tìm thấy, notebook liệt kê các Input hiện có để kiểm tra dataset source đã được Add Input hay chưa.
   Bộ tìm source, dataset ảnh và ZIP khôi phục dùng `os.walk(..., followlinks=True)` để nhận cả thư mục liên kết bên dưới `/kaggle/input/datasets`.
5. Ô cấu hình có sẵn `PERSON` và `SEEDS`. Mặc định chọn 5 seed đầu cho đợt 1.
   Nếu muốn chạy từng seed, đặt `SEEDS_TO_RUN = [1]` cho người 1, hoặc seed thuộc phần của mình.
6. Chạy ô cấu hình, ô chuẩn bị, rồi ô benchmark. Source ZIP bắt buộc có `run_poga.py` hỗ trợ `--benchmark_seeds`
   và `scripts/merge_seed_results.py`; dùng file ZIP mới kèm bộ notebook này. Không cần clone Git.
7. Theo dõi thông báo định kỳ; log chi tiết nằm ở `/kaggle/working/personN/seedS/console.log`.

Chương trình chạy tuần tự trên một GPU; bật T4 x2 không tự động chia đôi công việc.
Chưa đo thời gian toàn bộ benchmark này, vì vậy không giả định 10 seed sẽ vừa một phiên Kaggle.
Hãy đo seed đầu, sau đó chọn số seed phù hợp với thời gian phiên và quota đang có.

## 4. Lưu kết quả và chạy tiếp

Sau mỗi seed hoàn tất và đủ 33 dòng kết quả, notebook tạo:

```text
/kaggle/working/personN_seedS.zip
  seedS/
    DONE.json
    console.log
    run_poga_all_<timestamp>/
      experiment_results.csv
      convergence_histories.csv
      model_runs/<method>_seedS/<model>/logs/
      model_runs/<method>_seedS/<model>/plots/
      model_runs/<method>_seedS/<model>/weights/<model>_best.pth
      subsets/
      plots/
```

- **Tải từng ZIP sau khi hoàn tất seed, trước khi kết thúc phiên.** ZIP nằm trong Output/Files của Kaggle.
- ZIP chứa **toàn bộ file trong thư mục seed**, bao gồm checkpoint `.pth`, log huấn luyện từng epoch,
  CSV phân loại, metadata từng mô hình, biểu đồ, mask `.npy`, lịch sử hội tụ và báo cáo thống kê.
  Không lọc bỏ trọng số. `DONE.json` và `console.log` cũng có trong ZIP.
- `/kaggle/working/personN/seedS/` giữ bản kết quả đầy đủ chưa nén; ZIP `personN_seedS.zip` nằm cùng vùng Output.
  Mã nguồn và cache đặc trưng nằm trong `/kaggle/temp`, nên không lẫn trong vùng Output này.
  Dataset ảnh vẫn ở Input; source ZIP là đầu vào, không phải kết quả.
- Phiên mới: thêm các ZIP `personN_seedS.zip` đã tải vào Kaggle Input, import lại **cùng notebook** và chạy.
  Nếu Kaggle tự giải nén file khi tạo Dataset, hãy upload ZIP nằm trong một thư mục con và giữ file ZIP trong Input.
- Notebook khôi phục ZIP, kiểm tra snapshot/cấu hình/33 dòng rồi bỏ qua seed đã xong.
- Trước khi tạo ZIP đầy đủ, notebook kiểm tra có đủ 33 checkpoint. ZIP cũ từng bỏ `.pth`
  không đủ để khôi phục output đầy đủ; cần chạy lại seed đó trong phiên sạch, không thêm ZIP thiếu trọng số vào Input.
- Seed đang dở khi mất phiên sẽ phải chạy lại; chưa hỗ trợ tiếp tục ở giữa một seed.
- Nếu source SHA khác hoặc CSV thiếu/sai dòng, notebook dừng để tránh gộp sai dữ liệu.
- Không dùng `descriptive_statistics.csv` của một seed làm thống kê cuối cho 30 seed.

## 5. Gộp kết quả của cả ba người

Giải nén mỗi ZIP đúng một lần vào cùng thư mục. Sau khi giải nén cần có
`seed1/DONE.json` … `seed30/DONE.json`; có thể nằm trong thư mục con của từng người.
Không để nhiều bản sao của cùng seed trong thư mục đầu vào.

Từ thư mục gốc repository, dùng Python có `scipy` (môi trường `gamesearcher` hiện đã có):

```powershell
& "D:\Program Files\Miniconda\envs\gamesearcher\python.exe" scripts/merge_seed_results.py outputs/collected_30seeds --output outputs/merged_30seeds
```

Script kiểm tra đủ 30 seed, mỗi seed đủ 11 phương pháp × 3 model, cùng snapshot,
cùng 20 epoch/batch 64/split 42; từ chối seed trùng hoặc kết quả thiếu. Đầu ra:

- `experiment_results_30seeds.csv`: 990 dòng kết quả gốc, khóa theo Method + Model + Seed.
- `summary_30seeds.csv`: số lần đo, mean và **sample std (`ddof=1`)** cho từng phương pháp/model.
- `wilcoxon_C1_30seeds.csv`: Wilcoxon hai phía giữa C1 và từng đối chứng, ghép cặp đúng theo Seed.
  `Mean_Delta_Acc > 0` nghĩa là C1 tốt hơn về trung bình. Các p-value này chưa hiệu chỉnh cho nhiều phép so sánh;
  p nhỏ chỉ cho biết có khác biệt, không tự chứng minh C1 tốt hơn nếu chênh lệch âm.

Speedup chỉ tính thời gian huấn luyện CNN, không gồm tối ưu hóa; thời gian tối ưu
có cột riêng `Opt_Time_Sec`. Khác GPU có thể làm so sánh thời gian giữa người chạy bị nhiễu.

## 6. Dành cho người duy trì mã nguồn

Sau khi sửa mã, tạo lại ZIP source và cả ba notebook cùng lúc:

```powershell
python scripts/build_30seed_notebooks.py
& "D:\Program Files\Miniconda\envs\gamesearcher\python.exe" -m unittest discover -s tests
```

Không trộn kết quả trước và sau khi thay snapshot. Không dùng chế độ `run_poga.py`
cũ không có `--benchmark_seeds` cho bộ 30 seed này: chế độ cũ dùng B0 tham chiếu cố định và CNN seed 42.
File source ZIP chỉ chứa các file Python cần thiết, không đóng gói `.git`, dataset ảnh,
output của lần chạy cũ hay checkpoint. Gửi cùng một ZIP mới cho cả ba người trước khi bắt đầu.

## 7. Kế hoạch quota 30 giờ/người/tuần

Kaggle công bố quota GPU theo tuần, thông thường 30 giờ (có thể thay đổi theo tài nguyên)
và giới hạn phiên CPU/GPU tối đa 12 giờ. Xem quota thực tế trong Notebook settings trước mỗi phiên:
[GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage),
[Notebook specifications](https://www.kaggle.com/docs/notebooks).

Ước tính từ các log đã có trong dự án:

- `run_poga_all_20261002_173334`: 30 lượt CNN tập con/seed ở 5 epoch mất trung bình 4.295 giây;
  8 thuật toán tối ưu cộng khoảng 79 giây/seed (chỉ cộng một lần, không nhân theo 3 CNN).
- `run_all_20261017_030433`: B0 cả 3 CNN ở 20 epoch mất 3.710 giây.
- Ngoại suy thô: `4.295 × 4 + 79 + 3.710 ≈ 20.969 giây ≈ 5,82 giờ/seed`.
  Chi phí đánh giá không tăng tuyến tính theo epoch, và mã tối ưu đã thay đổi, nên đây là ước tính lập lịch,
  không phải số đo của benchmark mới. Cần đo lại seed đầu.

| Phạm vi | Ước tính |
|---|---:|
| 1 seed (33 CNN) | 5,82 giờ |
| 10 seed/người | 58,2 giờ |
| 30 seed/toàn nhóm | 174,7 giờ |
| Quota nhóm nếu mỗi người còn đủ 30 giờ | 90 giờ/tuần |

**Không nên hứa hoàn tất trong một tuần với 30 giờ/người.** Kế hoạch có dự phòng:

| Tuần | Người 1 | Người 2 | Người 3 | Thời gian/người dự kiến |
|---|---|---|---|---:|
| 1 | 1–4 | 11–14 | 21–24 | 23,3 giờ |
| 2 | 5–8 | 15–18 | 25–28 | 23,3 giờ |
| 3 | 9–10 | 19–20 | 29–30 | 11,6 giờ |

Hai đợt phân công 5 seed/người không đồng nghĩa mỗi đợt chắc chắn vừa một tuần.
Theo ước tính này, 5 seed đã khoảng 29,1 giờ, không đủ dự phòng tải model, trích đặc trưng,
đánh giá, nén ZIP và thời gian GPU bật nhưng nhàn rỗi. Nếu seed đầu thực tế nhanh hơn đáng kể,
có thể điều chỉnh và hoàn tất trong hai tuần; không giảm epoch hay bỏ seed ở một người để ép quota.

Notebook có `MAX_SESSION_HOURS=10.5`, `AVAILABLE_GPU_HOURS=30.0` và ước lượng khởi đầu
6,5 giờ/seed. **Điền số giờ quota còn lại thực tế trước mỗi phiên**; notebook không đọc được tài khoản Kaggle.
Chương trình kiểm tra ngân sách trước khi bắt đầu seed mới, cập nhật ước lượng theo thời gian thực tế
và dành 0,5 giờ so với quota bạn nhập. Đây là kiểm tra dự báo, không bảo đảm seed đang chạy sẽ xong
trước giới hạn nếu thời gian thực tế vượt dự báo. Có thể chỉ định một seed mỗi phiên bằng `SEEDS_TO_RUN`.
Hết công việc thì tắt GPU/session; chờ reset quota cho các seed còn lại.
