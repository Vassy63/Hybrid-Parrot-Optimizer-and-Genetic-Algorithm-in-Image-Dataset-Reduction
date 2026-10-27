# Demo PO-GA

Giao diện Streamlit tiếng Việt để rút gọn tập dữ liệu ảnh bằng PO–GA, huấn luyện CNN và so sánh với B0. Mã giao diện và bộ điều phối nằm trong **`src/`**; các module thuật toán đi kèm nằm ở thư mục gốc của demo.

Sau khi clone repository, mở thư mục **`POGA_Streamlit/`** rồi thực hiện hướng dẫn dưới đây. Cần Python 3.10 trở lên và dữ liệu Intel Image Classification; GPU NVIDIA có CUDA giúp chạy huấn luyện nhanh hơn.

## Chạy trên Windows

1. Mở folder này, chạy **`setup_demo.bat`** một lần để tạo `.venv` và cài thư viện. Cần Internet; lần cài PyTorch có thể lâu và không tính vào thời gian demo.
2. Chạy **`start_demo.bat`**. Mở **http://127.0.0.1:8501** nếu trình duyệt chưa tự mở.
3. Bấm **Kiểm tra môi trường**. Nếu CUDA chưa sẵn sàng, chọn bản cài PyTorch phù hợp ở <https://pytorch.org/get-started/locally/> rồi cài bằng `.venv\Scripts\python.exe -m pip ...` theo lệnh của trang đó. Máy đã được nhận diện có RTX 3060 12 GB; điều này không có nghĩa Python đã cài PyTorch CUDA.
4. Chọn phương pháp/CNN, số ảnh, epochs và bấm **Chạy demo**. App tự dùng dữ liệu trong `datasets/intel/` và giới hạn mỗi lượt tối đa 15 phút.

Nếu môi trường được cài thành PyTorch bản CPU: đóng cửa sổ server, chạy **`install_cuda.bat`**, rồi mở lại `start_demo.bat`. File này cài cặp PyTorch 2.13.0 / torchvision 0.28.0 từ kho CUDA 12.6 chính thức. Máy hiện có driver NVIDIA 560.94; bản CUDA 12.6 phù hợp hơn việc tự chọn CUDA 12.8/13.x. Lần cài CUDA cần Internet và tải nhiều GB, không phải thời gian huấn luyện demo.

Nếu dùng Python riêng, đặt biến `POGA_PYTHON` trỏ tới `python.exe` trước khi chạy setup. Ví dụ PowerShell:

```powershell
$env:POGA_PYTHON = 'C:\duong_dan\python.exe'
.\setup_demo.ps1
```

## Dữ liệu

Bản làm việc hiện tại đã giải nén đúng **14.034 ảnh seg_train và 3.000 ảnh seg_test** từ ZIP Intel của người dùng vào `datasets/intel/`. Không dùng `seg_pred` để đánh giá.

- Train: `datasets/intel/seg_train/seg_train`
- Test: `datasets/intel/seg_test/seg_test`

Hai đường dẫn trên đều phải chứa sáu folder lớp ảnh. Khi chuyển project sang máy khác, sao chép dataset và giữ cấu trúc thư mục trên. Dataset không được đưa vào ZIP mã nguồn.

Để giải nén lại dataset sau khi tải bản ZIP mã nguồn:

```powershell
.venv\Scripts\python.exe scripts\prepare_dataset.py "D:\New folder\Code\DoAn\Intel_Image_Classification_Dataset.zip"
```

## Cấu hình và thời gian

Sidebar cho chỉnh trực tiếp số ảnh và epochs, mặc định 600 ảnh / 3 epochs. Các mức dưới đây là gợi ý tham khảo:

| Cấu hình | Ảnh train trước rút gọn | Epochs |
|---|---:|---:|
| Kiểm tra nhanh | 600 | 3 |
| Trình bày | 3.000 | 5 |
| Mở rộng | 6.000 | 5 |

Mỗi lượt do giao diện khởi tạo có giới hạn cố định **15 phút**; hết thời gian, tiến trình dừng và đánh dấu chưa hoàn tất. Cấu hình nhỏ có thể hoàn tất sớm hơn. Không có bước cố tình chờ cho đủ thời gian.

Lần đầu có thể tải trọng số ImageNet và trích đặc trưng; các lượt sau dùng cache theo đúng danh sách ảnh, split và cấu hình trích xuất. Nên chạy thử một lần trước buổi demo. Sau khi có số đo trên máy, chọn số ảnh/epochs phù hợp.

## Tỷ lệ rút gọn

- Slider là **mức rút gọn tối đa**, từ 0% đến 70%.
- Bộ tối ưu hướng đến tỷ lệ mục tiêu mềm của source; phần demo thêm điều kiện giữ ít nhất `ceil(N × (1 − DR/100))` ảnh và giữ đủ các lớp.
- Ví dụ N=600, chọn 70%: giữ **ít nhất 180 ảnh**, DR thực tế **không vượt 70%**. Thuật toán có thể giữ nhiều hơn 180 ảnh.
- Chọn 0% giữ toàn bộ tập train demo. Nếu B0 đã chạy, sử dụng cùng kết quả đo cho tập con trùng B0.
- Com trong hàm fitness lúc bật mục tiêu là điểm bám mục tiêu, không phải DR thực tế. Adapter đã truyền target_ratio vào lịch sử GA/PO để biểu đồ thống nhất.

## Đánh giá và kết quả

B0 được huấn luyện trên tập train demo trước rút gọn, cùng CNN, seed, epochs, validation và test với tập con. Không dùng các số B0 cố định trong runner benchmark cũ. Khi bỏ chọn B0, không hiển thị ARR hay speedup thiếu đối chứng.

Mỗi phiên nằm trong một folder riêng:

```text
outputs/demo_.../
  config.json
  environment.json
  status.json
  events.jsonl
  console.log
  image_manifest.json
  convergence_histories.csv
  experiment_results.csv
  summary.json
  subsets/
  model_runs/b0/{logs,plots,weights}/
  model_runs/s1/{logs,plots,weights}/
```

Giao diện có ba tab: **Chạy demo**, **Kết quả 30 seeds**, **Lịch sử**. Kết quả 30 seeds tự đọc dữ liệu trong `assets/results_30seeds/`, tách khỏi kết quả demo một seed. Chọn CNN để xem biểu đồ Accuracy, Macro F1, tỷ lệ rút gọn và speedup; Accuracy và Macro F1 có thêm độ lệch chuẩn mẫu qua các seed. Biểu đồ lớp của demo đặt cột trước/sau rút gọn cạnh nhau, có số lượng trên từng cột. Một lượt demo không dùng để kết luận vượt trội thống kê.

Nút **Dừng** chỉ dừng tiến trình do app hiện tại quản lý. Không chạy nhiều instance app trên cùng GPU. Giữ cửa sổ `start_demo.bat` mở trong lúc sử dụng.

## Kiểm tra mã

```powershell
.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Các test kiểm tra giới hạn DR/làm tròn, bảo toàn lớp, seed, thống kê mẫu, đối chứng đo thực tế và điều phối tiến trình. Kiểm tra huấn luyện CUDA và đo thời gian yêu cầu cài đầy đủ thư viện.
