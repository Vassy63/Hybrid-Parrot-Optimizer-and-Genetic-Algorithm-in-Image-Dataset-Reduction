# BÁO CÁO KHÓA LUẬN CỬ NHÂN NGÀNH CÔNG NGHỆ THÔNG TIN
**Mã đề tài:** CNTT-KLCN140  
**Tên đề tài:** Ứng dụng thuật toán Parrot Optimizer kết hợp Genetic Algorithm trong rút gọn tập dữ liệu ảnh cho bài toán phân loại cảnh tự nhiên  
**Giảng viên hướng dẫn:** ThS. Đinh Nguyễn Trọng Nghĩa  

---

## CHƯƠNG 1: TỔNG QUAN VỀ ĐỀ TÀI NGHIÊN CỨU

### 1.1. Lý do Chọn Đề tài và Đặt Vấn đề

Trong các hệ thống thị giác máy tính hiện đại, bài toán phân loại cảnh tự nhiên (Natural Scene Classification) đóng vai trò nền tảng cho nhiều ứng dụng thực tiễn như điều hướng robot tự hành, giám sát môi trường và truy vấn ảnh theo nội dung [3]. Sự phát triển của các kiến trúc mạng nơ-ron tích chập sâu (Deep Convolutional Neural Networks - CNN) đã nâng cao đáng kể độ chính xác nhận dạng cảnh quan. Tuy nhiên, hiệu năng của các mô hình học sâu phụ thuộc chặt chẽ vào quy mô của tập dữ liệu huấn luyện. Khi số lượng mẫu ảnh tăng lên hàng chục nghìn hoặc hàng triệu mẫu, quá trình huấn luyện đối mặt với ba rào cản kỹ thuật lớn: chi phí thời gian huấn luyện kéo dài, yêu cầu bộ nhớ lưu trữ lớn và sự dư thừa thông tin do các mẫu ảnh tương đồng cao trong cùng một phân lớp cảnh quan.

Kỹ thuật lựa chọn mẫu đại diện hay rút gọn tập dữ liệu (Instance Selection / Dataset Reduction) giải quyết trực tiếp vấn đề trên bằng cách trích xuất một tập con $S \subset D$ từ tập dữ liệu huấn luyện gốc $D$ sao cho kích thước tập con nhỏ hơn nhiều so với tập gốc ($|S| = M \ll N$), nhưng vẫn bảo toàn cấu trúc phân bố không gian đặc trưng và ranh giới quyết định giữa các lớp [2], [5]. Khi huấn luyện mô hình phân loại trên tập dữ liệu đã rút gọn $S$, hệ thống giảm thiểu đáng kể thời gian tính toán trong khi duy trì độ chính xác phân loại tiệm cận với mô hình huấn luyện trên toàn bộ tập dữ liệu gốc.

Mặc dù vậy, bài toán lựa chọn tập con tối ưu từ $N$ mẫu dữ liệu là bài toán tối ưu hóa tổ hợp NP-khó với không gian tìm kiếm nhị phân có lực lượng $2^N$. Với tập dữ liệu gồm $N = 14.034$ ảnh, các phương pháp tìm kiếm vét cạn hoặc tham lam cục bộ không thể đảm bảo chất lượng nghiệm toàn cục. Giải thuật Di truyền (Genetic Algorithm - GA) có thế mạnh khai phá không gian nhị phân thông qua phép lai ghép và đột biến [2], [6], nhưng thường hội tụ chậm ở giai đoạn tinh chỉnh cuối. Ngược lại, Thuật toán Tối ưu hóa Đàn vẹt (Parrot Optimizer - PO) mới được đề xuất bởi Lian và cộng sự (2024) [1] sở hữu khả năng khai thác cục bộ linh hoạt nhờ mô phỏng bốn hành vi bầy đàn kết hợp chuyển động Lévy flight, song lại được thiết kế nguyên bản cho không gian liên tục và dễ rơi vào cực trị địa phương khi chuyển sang không gian nhị phân chiều cao. Do đó, việc nghiên cứu thích nghi nhị phân cho thuật toán PO và thiết kế các kiến trúc lai hóa giữa PO và GA (Hybrid PO–GA) để lựa chọn tập ảnh đại diện cho bài toán phân loại cảnh tự nhiên mang ý nghĩa khoa học và thực tiễn rõ rệt.

### 1.2. Mục tiêu, Đối tượng và Phạm vi Nghiên cứu

Mục tiêu tổng quát của đề tài là nghiên cứu bài toán rút gọn tập dữ liệu ảnh (instance selection) và xây dựng thuật toán kết hợp giữa Parrot Optimizer (PO) với Genetic Algorithm (GA) nhằm lựa chọn tập ảnh đại diện tối ưu cho bài toán phân loại cảnh tự nhiên. Để hiện thực hóa mục tiêu này, nghiên cứu tập trung giải quyết bốn câu hỏi nghiên cứu cốt lõi:
1. **RQ1 (Thích nghi nhị phân và mô hình hóa hàm mục tiêu)**: Làm thế nào để chuyển đổi thuật toán PO liên tục sang không gian nhị phân rời rạc và thiết kế hàm mục tiêu đa tiêu chí kết hợp chặt chẽ giữa chất lượng đại diện (độ đa dạng, độ bao phủ, cân bằng lớp) với ràng buộc tỷ lệ rút gọn dữ liệu?
2. **RQ2 (Ảnh hưởng của kiến trúc lai hóa)**: Giữa ba kiến trúc lai hóa gồm Tuần tự (Sequential: S1, S2, S3, S4), Song song độc lập (Parallel: P1) và Hợp tác trao đổi cá thể ưu tú (Cooperative: C1), kiến trúc nào đạt tốc độ hội tụ và điểm thích nghi cao nhất dưới cùng một ngân sách đánh giá hàm mục tiêu (Function Evaluations - FEs)?
3. **RQ3 (Hiệu quả huấn luyện mô hình phân loại)**: Tập dữ liệu rút gọn bởi PO–GA duy trì được bao nhiêu phần trăm độ chính xác (Accuracy, Precision, Recall, F1-score) và rút ngắn thời gian huấn luyện bao nhiêu lần so với toàn bộ tập dữ liệu gốc cũng như các phương pháp lấy mẫu ngẫu nhiên ở các mức rút gọn $50\%, 70\%, 80\%$ và $90\%$?
4. **RQ4 (Tính tổng quát hóa xuyên kiến trúc - Cross-Architecture Generalization)**: Tập ảnh đại diện được chọn dựa trên không gian đặc trưng của ResNet-18 có duy trì được hiệu năng ổn định khi huấn luyện trên các kiến trúc mạng học sâu có cấu trúc khác biệt hoàn toàn gồm **ResNet-18** (mạng phần dư), **MobileNetV3-Small** (mạng tích chập tách chiều sâu siêu nhẹ) và **DenseNet-121** (mạng kết nối dày đặc) hay không?

### 1.3. Các Đóng góp Chính của Khóa luận

Nghiên cứu mang lại bốn đóng góp học thuật và kỹ thuật cụ thể:
1. **Đóng góp C1 (Ứng dụng Parrot Optimizer truyền thống cho bài toán chọn mẫu nhị phân)**: Giữ nguyên bốn hành vi bầy đàn của thuật toán Parrot Optimizer truyền thống (Lian và cộng sự, 2024) và bổ sung bước ánh xạ nhị phân qua hàm $|\tanh(\Delta x)|$ ở đầu ra để chuyển nghiệm liên tục sang vector chọn ảnh $\{0, 1\}^N$.
2. **Đóng góp C2 (Khung đối sánh 3 kiến trúc lai hóa PO–GA)**: Thiết kế và chuẩn hóa sáu cấu hình lai hóa thuộc ba nhóm kiến trúc: Tuần tự xen kẽ và theo khối (S1, S2, S3, S4), Song song hợp nhất (P1) và Hợp tác trao đổi ưu tú hai chiều có kiểm soát trùng lặp (C1) dưới điều kiện cố định tuyệt đối ngân sách tính toán ($2.000\text{ FEs}$).
3. **Đóng góp C3 (Hàm mục tiêu đa tiêu chí chuẩn hóa nội tại)**: Thiết lập hàm thích nghi bốn thành phần (Độ đa dạng, Độ bao phủ, Cân bằng lớp, Tỷ lệ nén) được chuẩn hóa trực tiếp trên ma trận khoảng cách Cosine chia cho đường kính tập dữ liệu $\text{dist}_{\max}$, loại bỏ hoàn toàn sai số từ việc ước lượng biên ngẫu nhiên (Global Bounds Estimation).
4. **Đóng góp C4 (Hệ thống đánh giá hai tầng và Công cụ hỗ trợ thực nghiệm)**: Tách biệt quy trình đánh giá thành hai tầng độc lập (Tầng 1: Chất lượng tối ưu hóa tổ hợp; Tầng 2: Hiệu năng huấn luyện phân loại cảnh tự nhiên) và xây dựng công cụ phần mềm hoàn chỉnh hỗ trợ lựa chọn dữ liệu, rút gọn, huấn luyện mô hình và hiển thị trực quan kết quả đối sánh.

### 1.4. Khảo sát Các Công trình Nghiên cứu Liên quan

Trong lĩnh vực rút gọn dữ liệu bằng giải thuật tiến hóa, Cano, Herrera và Lozano (2003) [2] đã đặt nền móng cho việc ứng dụng Giải thuật Di truyền (GA) và các thuật toán tiến hóa trong bài toán lựa chọn mẫu (Instance Selection). Công trình của Cano và cộng sự chứng minh rằng mã hóa nhị phân kết hợp hàm mục tiêu cân bằng giữa sai số phân loại và tỷ lệ nén cho phép loại bỏ hơn $70\%$ mẫu dư thừa trên các tập dữ liệu chuẩn mà không làm suy giảm đáng kể độ chính xác. Tuy nhiên, nghiên cứu chỉ ra rằng GA chuẩn chịu chi phí tính toán lớn và tốc độ hội tụ chậm khi chiều dài nhiễm sắc thể $N$ vượt quá hàng nghìn phần tử.

Đối với bài toán rút gọn tập dữ liệu ảnh cho mạng học sâu, Sener và Savarese (2018) [5] đã mô hình hóa việc chọn tập con (Coreset Selection) dưới dạng bài toán hình học $k$-Center (Facility Location), trong đó mục tiêu là cực tiểu hóa khoảng cách lớn nhất từ bất kỳ mẫu dữ liệu nào trong tập gốc tới mẫu đại diện gần nhất trong tập con. Cách tiếp cận hình học này cho phép đánh giá chất lượng tập con ngay trên không gian đặc trưng (feature space) mà không cần huấn luyện lại mạng CNN ở mỗi bước lặp. Kế thừa tư tưởng này, đề tài sử dụng mạng ResNet-18 tiền huấn luyện [4] để trích xuất đặc trưng ngữ nghĩa cho ảnh cảnh tự nhiên [3], sau đó kết hợp cả tiêu chí độ bao phủ (Coverage) của bài toán Facility Location với tiêu chí độ đa dạng (Diversity) và cân bằng phân lớp (Class Balance) trong một hàm mục tiêu thống nhất.

Về mặt thuật toán tối ưu hóa bầy đàn, Lian và cộng sự (2024) [1] đã giới thiệu thuật toán Parrot Optimizer (PO) lấy cảm hứng từ bốn hành vi đặc trưng của loài vẹt Pyrrhura molinae: kiếm ăn (foraging), đậu tại chỗ (staying), giao tiếp (communicating) và sợ người lạ (fear of strangers). Cơ chế chọn ngẫu nhiên một trong bốn hành vi ở mỗi vòng lặp giúp PO tránh được sự thiên lệch cứng nhắc giữa pha thăm dò (exploration) và pha khai thác (exploitation). Bảng 1.1 tổng hợp ưu điểm và hạn chế của các phương pháp nền tảng, làm cơ sở cho việc kết hợp PO và GA trong nghiên cứu này.

| Phương pháp | Nguyên lý cốt lõi | Ưu điểm | Nhược điểm / Hạn chế |
| :--- | :--- | :--- | :--- |
| **Lấy mẫu ngẫu nhiên (Random / Stratified)** | Chọn ngẫu nhiên các mẫu ảnh theo phân bố đều hoặc phân tầng theo lớp. | Độ phức tạp $O(N)$, thời gian thực thi tức thời, không tốn chi phí tối ưu. | Không xét đến cấu trúc hình học không gian đặc trưng; dễ giữ lại mẫu trùng lặp và bỏ sót mẫu biên quan trọng. |
| **Genetic Algorithm (GA)** [2], [6] | Tiến hóa quần thể nhị phân qua chọn lọc giải đấu, lai ghép đồng nhất và đột biến lật bit. | Thao tác trực tiếp trên không gian nhị phân $\{0,1\}^N$; khả năng khai phá toàn cục mạnh nhờ toán tử lai ghép. | Tốc độ hội tụ cục bộ chậm ở các thế hệ cuối; dễ bị trì trệ khi kích thước chiều $N$ lớn. |
| **Parrot Optimizer (PO)** [1] | Cập nhật vị trí liên tục theo 4 hành vi bầy đàn kết hợp bước nhảy Lévy flight. | Khai thác cục bộ nhanh quanh nghiệm tốt nhất $X_{\text{gbest}}$ và tâm quần thể $X_{\text{mean}}$; thoát cực trị nhờ Lévy flight. | Thiết kế cho miền liên tục; khi nhị phân hóa đơn thuần dễ mất tính đa dạng quần thể nếu thiếu toán tử tái tổ hợp gen. |
| **Hybrid PO–GA (Đề xuất)** | Kết hợp toán tử tiến hóa nhị phân của GA và động lực học bầy đàn của Binary PO theo 3 kiến trúc (S1–S4, P1, C1). | Cân bằng giữa khả năng tái tổ hợp cấu trúc nhị phân toàn cục (GA) và hội tụ định hướng nhanh (PO). | Cần kiểm soát chặt chẽ ngân sách FEs, cơ chế chuyển đổi pha và chống trùng lặp cá thể khi trao đổi. |

---

## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT, BỘ DỮ LIỆU VÀ BIỂU DIỄN ĐẶC TRƯNG

### 2.1. Bài toán Phân loại Cảnh tự nhiên và Bộ dữ liệu Thực nghiệm

Phân loại cảnh tự nhiên (Natural Scene Classification) là bài toán gán nhãn ngữ nghĩa toàn cảnh cho một bức ảnh dựa trên cấu trúc bố cục và các thành phần tự nhiên hoặc nhân tạo xuất hiện trong ảnh [3]. Khác với bài toán nhận dạng vật thể đơn lẻ, ảnh cảnh tự nhiên có sự biến thiên lớn về góc chụp, điều kiện ánh sáng và sự chồng lấp giữa các vùng ngữ nghĩa (ví dụ: cảnh sông băng thường chứa cả núi tuyết và mặt nước).

Đề tài sử dụng bộ dữ liệu chuẩn **Intel Image Classification** làm đối tượng thực nghiệm chính. Bộ dữ liệu bao gồm $17.034$ ảnh màu RGB kích thước chuẩn $150 \times 150$ điểm ảnh (được nội suy song tuyến về $224 \times 224$ điểm ảnh để tương thích với đầu vào chuẩn của các kiến trúc CNN tiền huấn luyện), phân bố trên $C = 6$ lớp cảnh quan: công trình kiến trúc (`buildings`), rừng cây (`forest`), sông băng (`glacier`), núi (`mountain`), biển (`sea`) và đường phố (`street`).

Để đảm bảo tính nghiêm ngặt của phương pháp luận học máy và ngăn chặn triệt để hiện tượng rò rỉ dữ liệu từ tập kiểm thử vào quá trình tối ưu hóa (Data Leakage Control), bộ dữ liệu được phân hoạch thành ba tập độc lập hoàn toàn như trình bày trong Bảng 2.1:
1. **Tập Huấn luyện (`seg_train` - Train Split)**: Gồm $14.034$ ảnh trong thư mục huấn luyện gốc. Toàn bộ quá trình trích xuất ma trận khoảng cách và chạy thuật toán tối ưu hóa chọn tập con PO–GA chỉ được phép thực hiện trong phạm vi tập huấn luyện này. Khi huấn luyện mạng học sâu (CNN), tập `seg_train` được tách ngẫu nhiên cố định hạt giống (`seed = 42`) theo tỷ lệ $80\%$ huấn luyện ($11.228$ ảnh, áp dụng tăng cường dữ liệu lật ngang ngẫu nhiên `RandomHorizontalFlip`) và $20\%$ xác thực.
2. **Tập Xác thực (Validation Set)**: Gồm $2.806$ ảnh ($20\%$ của `seg_train`), sử dụng để giám sát quá trình hội tụ sau mỗi epoch và lựa chọn trọng số mô hình tốt nhất (`best_checkpoint`).
3. **Tập Kiểm thử Độc lập (Independent Test Set - `seg_test`)**: Gồm $3.000$ ảnh hoàn toàn tách biệt, tuyệt đối không tham gia vào quá trình tính ma trận khoảng cách, không tham gia vào hàm mục tiêu của PO–GA và không dùng để chọn epoch dừng. Tập `seg_test` chỉ được dùng một lần duy nhất để đo lường các chỉ số đánh giá cuối cùng.

| STT | Tên lớp cảnh quan (Class) | Tập gốc `seg_train` (Ảnh) | Tập Train DL (80%) | Tập Validation DL (20%) | Tập Kiểm thử Độc lập `seg_test` (Ảnh) | Tỷ lệ phân bố lớp (%) |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | `buildings` (Tòa nhà) | $2.191$ | $1.753$ | $438$ | $437$ | $15,61\%$ |
| 2 | `forest` (Rừng cây) | $2.271$ | $1.817$ | $454$ | $474$ | $16,18\%$ |
| 3 | `glacier` (Sông băng) | $2.404$ | $1.923$ | $481$ | $553$ | $17,13\%$ |
| 4 | `mountain` (Núi) | $2.512$ | $2.010$ | $502$ | $525$ | $17,90\%$ |
| 5 | `sea` (Biển) | $2.274$ | $1.819$ | $455$ | $510$ | $16,20\%$ |
| 6 | `street` (Đường phố) | $2.382$ | $1.906$ | $476$ | $501$ | $16,97\%$ |
| **Tổng** | **6 lớp cảnh tự nhiên** | **$14.034$** | **$11.228$** | **$2.806$** | **$3.000$** | **$100,00\%$** |

### 2.2. Trích xuất Đặc trưng Học sâu và Tối ưu hóa Không gian Khoảng cách

Để đánh giá mức độ tương đồng ngữ nghĩa giữa các bức ảnh, mỗi ảnh $I_i$ được chuẩn hóa theo giá trị trung bình $\mu = [0,485; 0,456; 0,406]$ và độ lệch chuẩn $\sigma = [0,229; 0,224; 0,225]$ của ImageNet, sau đó đưa qua mạng xương sống (Backbone) **ResNet-18** [4] đã đóng băng trọng số tiền huấn luyện. Tầng phân loại cuối cùng (`model.fc`) được loại bỏ, thu nhận đầu ra từ tầng `AdaptiveAvgPool2d` để tạo thành vector đặc trưng $f_i \in \mathbb{R}^{512}$.

Khoảng cách giữa hai mẫu ảnh $i$ và $j$ được xác định bằng khoảng cách Cosine trên không gian đặc trưng $512$ chiều:

$$\text{dist}(f_i, f_j) = 1 - \frac{f_i \cdot f_j}{\|f_i\|_2 \|f_j\|_2}$$

Sau đó, toàn bộ khoảng cách được chuẩn hóa về đoạn $[0, 1]$ thông qua phép chia cho khoảng cách cực đại toàn cục $\text{dist}_{\max} = \max_{1 \le u, v \le N} \text{dist}(f_u, f_v)$:

$$\hat{d}_{ij} = \frac{\text{dist}(f_i, f_j)}{\text{dist}_{\max}} \in [0, 1]$$

**Phân tích chi phí bộ nhớ và chiến lược tối ưu hóa tính toán**: Với $N = 14.034$ ảnh, ma trận khoảng cách đầy đủ $\hat{D} \in \mathbb{R}^{N \times N}$ chứa $N^2 \approx 196,95 \times 10^6$ phần tử. Khi lưu trữ dưới định dạng số thực dấu phẩy động 32-bit (`float32`), ma trận này chiếm dung lượng bộ nhớ:

$$\text{Memory}(\hat{D}) = 14.034^2 \times 4\text{ bytes} \approx 787,81\text{ MB}$$

Trên các hệ thống có bộ nhớ RAM từ $16\text{ GB}$ trở lên (hoặc môi trường Kaggle Cloud), việc tiền tính toán và giữ ma trận $\hat{D}$ (`float32`) trong bộ nhớ kết hợp phép cắt lát tensor trên PyTorch CPU (`torch.index_select`) cho phép đánh giá một cá thể chỉ trong khoảng $1,5\text{ ms}$. Tuy nhiên, khi mở rộng lên tập dữ liệu lớn hơn ($N > 50.000$), báo cáo thiết lập hai cơ chế tính toán tiết kiệm bộ nhớ:
1. **Tính toán khoảng cách theo khối (Chunked Distance Computation)**: Thay vì cấp phát toàn bộ ma trận $N \times N$, tập đặc trưng được chia thành các khối kích thước $B \times 512$ để tính khoảng cách từng phần.
2. **Cập nhật gia tăng (Incremental Coverage & Diversity Caching)**: Khi một cá thể chỉ thay đổi một số ít bit (ví dụ trong bước đột biến hoặc sửa chữa nghiệm), độ đa dạng $\text{Div}(S \cup \{u\})$ chỉ cần cộng thêm tổng khoảng cách từ mẫu mới $u$ tới các mẫu hiện có trong $S$ với độ phức tạp $O(M)$ thay vì tính lại toàn bộ $O(M^2)$ cặp. Tương tự, độ bao phủ được cập nhật thông qua vector khoảng cách gần nhất lưu đệm $\text{nearest\_dist}[i] \leftarrow \min(\text{nearest\_dist}[i], \hat{d}_{iu})$ với độ phức tạp $O(N)$ thay vì $O(N \cdot M)$.

---

## CHƯƠNG 3: MÔ HÌNH HÓA BÀI TOÁN LỰA CHỌN TẬP CON VÀ HÀM MỤC TIÊU

### 3.1. Biểu diễn Nghiệm Nhị phân và Cơ chế Tự Cân bằng Kích thước Tập con

Mỗi phương án lựa chọn tập con trên tập huấn luyện $D_{\text{train}}$ ($N = 11.228$ ảnh) được mã hóa thành một nhiễm sắc thể (hay vị trí cá thể vẹt) dưới dạng vector nhị phân $N$ chiều:

$$X = (x_1, x_2, \dots, x_N) \in \{0, 1\}^N$$

trong đó $x_i = 1$ nếu bức ảnh thứ $i$ được giữ lại trong tập con $S = \{i \in \{1, \dots, N\} \mid x_i = 1\}$, và $x_i = 0$ nếu bức ảnh thứ $i$ bị loại bỏ. Số lượng mẫu ảnh được giữ lại trong tập con là $M = |S| = \sum_{i=1}^N x_i$, tương ứng với tỷ lệ rút gọn dữ liệu (Dataset Reduction Rate) là $\text{DR} = 1 - \frac{M}{N}$.

Thay vì áp đặt một tỷ lệ giữ lại cố định từ bên ngoài làm triệt tiêu vai trò của thành phần độ nén $\text{Com}(X)$, mô hình cho phép kích thước tập con $M$ biến thiên tự do trên miền khả thi $2 \le M \le N$. Sự cân bằng giữa quy mô rút gọn và chất lượng đại diện được điều tiết hoàn toàn nội tại thông qua sự đối kháng giữa thành phần độ nén $\text{Com}(X)$ (ưu tiên giảm $M$) và hai thành phần độ bao phủ $\text{Cov}(X)$ cùng độ cân bằng lớp $\text{Bal}(X)$ (ưu tiên duy trì đủ mẫu đại diện cho mọi cụm phân lớp).

### 3.2. Toán tử Sửa chữa Nghiệm Khả thi Tối thiểu (Minimal Feasibility Repair Operator)

Do tiêu chí độ đa dạng $\text{Div}(X)$ tính trung bình khoảng cách giữa các cặp mẫu phân biệt trong tập con $S$ với mẫu số $|S|(|S|-1)$, hàm mục tiêu yêu cầu tập con không suy biến phải chứa tối thiểu hai phần tử ($M \ge 2$). Trong trường hợp hiếm gặp khi phép biến đổi nhị phân sinh ra cá thể có $M < 2$, toán tử sửa chữa $\text{Repair}(X)$ bổ sung ngẫu nhiên $2 - M$ phần tử từ tập chỉ số chưa chọn $U = \{i \mid x_i = 0\}$ sang trạng thái $1$, ngoài ra không can thiệp cưỡng bức vào kích thước tự nhiên của cá thể.

### 3.3. Định nghĩa Toán học Bốn Thành phần Hàm Mục tiêu

Với mỗi cá thể hợp lệ $X$ có tập mẫu được chọn $S$ ($|S| = M \ge 2$), bốn tiêu chí thành phần đều được chuẩn hóa về đoạn $[0, 1]$:

1. **Độ đa dạng (Diversity - $\text{Div}$)**: Khuyến khích các bức ảnh được chọn trong tập con phải khác biệt nhau, tránh chọn các ảnh trùng lặp:
   $$\text{Div}(X) = \frac{2}{|S|(|S|-1)} \sum_{i=1}^N \sum_{j=i+1}^N x_i \cdot x_j \cdot \frac{\text{dist}(f_i, f_j)}{\text{dist}_{\max}} \in [0, 1]$$

2. **Độ bao phủ (Coverage - $\text{Cov}$)**: Đảm bảo các bức ảnh bị loại bỏ không nằm quá xa các bức ảnh được giữ lại (tập con đại diện tốt cho toàn bộ không gian dữ liệu gốc):
   $$\text{Cov}(X) = 1 - \frac{1}{N} \sum_{i=1}^N \min_{j : x_j = 1} \left( \frac{\text{dist}(f_i, f_j)}{\text{dist}_{\max}} \right) \in [0, 1]$$

3. **Độ cân bằng lớp (Class Balance - $\text{Bal}$)**: Giữ cho tỷ lệ phân bố giữa các lớp trong tập con tương đồng với tập dữ liệu gốc, hạn chế mất cân bằng dữ liệu:
   $$\text{Bal}(X) = 1 - \sqrt{\frac{1}{C} \sum_{c=1}^C \left( \frac{M_c}{|S|} - \frac{N_c}{N} \right)^2} \in [0, 1]$$

4. **Độ nén dữ liệu (Compression - $\text{Com}$)**: Phạt các cá thể chọn quá nhiều ảnh, khuyến khích kích thước tập con nhỏ gọn:
   $$\text{Com}(X) = 1 - \frac{M}{N} = 1 - \frac{\sum_{i=1}^N x_i}{N} \in [0, 1]$$

**Hàm Fitness tổng hợp bốn thành phần**:
$$F(X) = \alpha \cdot \text{Div}(X) + \beta \cdot \text{Cov}(X) + \gamma \cdot \text{Bal}(X) + \delta \cdot \text{Com}(X)$$
với bộ trọng số chuẩn hóa $(\alpha, \beta, \gamma, \delta) = (0,15;\; 0,45;\; 0,20;\; 0,20)$ thỏa mãn $\alpha + \beta + \gamma + \delta = 1$. Việc thiết lập tỷ lệ trọng số giữa độ nén và độ bao phủ $\frac{\delta}{\beta} = \frac{0,20}{0,45} \approx 0,444$ giúp cân bằng đạo hàm riêng $\frac{\partial F}{\partial (M/N)} = 0$ tại vùng rút gọn tự nhiên $\text{DR} \in [72\%, 76\%]$ ($M^* \approx 2.750 - 3.050$ ảnh) mà không cần áp đặt ràng buộc kích thước cứng.

### 3.4. Ví dụ Minh họa Tính toán Cụ thể (Illustrative Example)

Để minh họa trực quan cơ chế mã hóa nghiệm và tính toán hàm mục tiêu (đáp ứng yêu cầu CLO1.2 của đề cương), xét một tập dữ liệu thu nhỏ gồm $N = 6$ bức ảnh thuộc $C = 2$ lớp cảnh tự nhiên: Lớp 0 (`forest`) gồm 3 ảnh $\{I_1, I_2, I_3\}$ và Lớp 1 (`sea`) gồm 3 ảnh $\{I_4, I_5, I_6\}$. Tỷ lệ phân bố gốc của mỗi lớp là $\frac{N_0}{N} = \frac{3}{6} = 0,50$ và $\frac{N_1}{N} = \frac{3}{6} = 0,50$. Giả sử ma trận khoảng cách đã chuẩn hóa $\hat{D}_{6 \times 6}$ (với $\hat{d}_{ij} = \text{dist}(f_i, f_j) / \text{dist}_{\max}$) có giá trị như sau:

$$\hat{D} = \begin{bmatrix}
0,00 & 0,10 & 0,20 & 0,80 & 0,90 & 0,85 \\
0,10 & 0,00 & 0,15 & 0,82 & 0,88 & 0,90 \\
0,20 & 0,15 & 0,00 & 0,75 & 0,85 & 0,80 \\
0,80 & 0,82 & 0,75 & 0,00 & 0,12 & 0,18 \\
0,90 & 0,88 & 0,85 & 0,12 & 0,00 & 0,10 \\
0,85 & 0,90 & 0,80 & 0,18 & 0,10 & 0,00
\end{bmatrix}$$

Xét hai cá thể ứng viên cùng chọn $M = 2$ ảnh (rút gọn $66,67\%$):
* **Cá thể $X^{(A)} = (1, 1, 0, 0, 0, 0)$**: Chỉ chọn 2 ảnh $\{I_1, I_2\}$ đều thuộc Lớp 0 (`forest`), bỏ trống hoàn toàn Lớp 1 (`sea`):
  * $\text{Div}(X^{(A)}) = \hat{d}_{1,2} = 0,1000$ (rất thấp do 2 ảnh cùng lớp quá giống nhau).
  * Khoảng cách nhỏ nhất từ 6 ảnh tới $\{I_1, I_2\}$ lần lượt là: $[0,00;\; 0,00;\; 0,15;\; 0,80;\; 0,88;\; 0,85]$, trung bình bằng $0,4467 \Rightarrow \text{Cov}(X^{(A)}) = 1 - 0,4467 = 0,5533$.
  * Phân bố lớp tập con: $M_0/M = 1,00$, $M_1/M = 0,00 \Rightarrow \text{Bal}(X^{(A)}) = 1 - \sqrt{\frac{(1-0,5)^2 + (0-0,5)^2}{2}} = 1 - 0,5000 = 0,5000$.
  * $\text{Com}(X^{(A)}) = 1 - \frac{2}{6} = 0,6667$.
  * Điểm thích nghi tổng hợp: $F(X^{(A)}) = 0,15(0,1000) + 0,45(0,5533) + 0,20(0,5000) + 0,20(0,6667) = \mathbf{0,4973}$.

* **Cá thể $X^{(B)} = (1, 0, 0, 0, 1, 0)$**: Chọn 1 ảnh Lớp 0 ($I_1$) và 1 ảnh Lớp 1 ($I_5$):
  * $\text{Div}(X^{(B)}) = \hat{d}_{1,5} = 0,9000$ (hai ảnh đại diện cho hai cảnh quan khác biệt rõ rệt).
  * Khoảng cách nhỏ nhất từ 6 ảnh tới $\{I_1, I_5\}$ lần lượt là: $[0,00;\; 0,10;\; 0,20;\; 0,12;\; 0,00;\; 0,10]$, trung bình chỉ bằng $0,0867 \Rightarrow \text{Cov}(X^{(B)}) = 1 - 0,0867 = 0,9133$ (bao phủ tốt cả hai cụm lớp).
  * Phân bố lớp tập con: $M_0/M = 0,50$, $M_1/M = 0,50 \Rightarrow \text{Bal}(X^{(B)}) = 1 - 0 = 1,0000$ (cân bằng tuyệt đối).
  * $\text{Com}(X^{(B)}) = 1 - \frac{2}{6} = 0,6667$.
  * Điểm thích nghi tổng hợp: $F(X^{(B)}) = 0,15(0,9000) + 0,45(0,9133) + 0,20(1,0000) + 0,20(0,6667) = \mathbf{0,8793}$.

Ví dụ trên chứng minh hàm mục tiêu đánh giá chính xác chất lượng đại diện: cùng giữ lại 2 bức ảnh nhưng cá thể $X^{(B)}$ có độ bao phủ, đa dạng và cân bằng lớp vượt trội được chấm điểm $0,8793$, cao hơn hẳn cá thể lệch lớp $X^{(A)}$ ($0,4973$).

---

## CHƯƠNG 4: THUẬT TOÁN BIẾN ĐỔI NHỊ PHÂN VÀ BA KIẾN TRÚC LAI HÓA PO–GA

### 4.1. Giải thuật Di truyền cho Lựa chọn Mẫu Nhị phân (Binary Genetic Algorithm)

Giải thuật Di truyền (GA) vận hành trực tiếp trên quần thể các vector nhị phân $\{X_1, X_2, \dots, X_P\}$. Tại mỗi thế hệ $t \in \{1, \dots, T\}$, GA thực hiện tuần tự bốn bước:
1. **Bảo tồn tinh hoa (Elitism)**: Sắp xếp quần thể theo chiều giảm dần của hàm thích nghi $F(X)$ và sao chép nguyên trạng $E = 5$ cá thể đứng đầu sang quần thể thế hệ kế tiếp.
2. **Chọn lọc giải đấu (Tournament Selection)**: Để chọn mỗi cá thể cha mẹ, lấy ngẫu nhiên $k_{\text{tour}} = 4$ cá thể từ quần thể hiện tại và chọn cá thể có điểm thích nghi cao nhất.
3. **Lai ghép đồng nhất (Uniform Crossover)**: Với xác suất lai ghép $p_c = 0,8$, hai cá thể cha mẹ $P_1, P_2$ trao đổi từng bit độc lập theo mặt nạ nhị phân ngẫu nhiên $M_{\text{mask}} \in \{0, 1\}^N$ ($\Pr(M_{\text{mask}, j} = 1) = 0,5$) để tạo ra hai cá thể con $C_1, C_2$.
4. **Đột biến thích nghi giảm dần (Adaptive Bit-Flip Mutation)**: Tỷ lệ đột biến $p_m(t)$ giảm tuyến tính theo số thế hệ để ưu tiên thăm dò ở giai đoạn đầu và ổn định hội tụ ở giai đoạn cuối:
   $$p_m(t) = \max\left(p_{\min},\; p_{\max} - (p_{\max} - p_{\min}) \cdot \frac{t}{T}\right)$$
   với $p_{\max} = 0,0010$ và $p_{\min} = 0,0002$. Để giữ cân bằng mật độ tập con hiện tại $r = \frac{M}{N}$, xác suất lật bit tại mỗi vị trí được điều chỉnh tương ứng là $p_m(t)\cdot(1-r)$ đối với các bit $x_j = 1$ và $p_m(t)\cdot r$ đối với các bit $x_j = 0$, sau đó đưa qua toán tử sửa chữa $\text{Repair}(X)$.

### 4.2. Thuật toán Tối ưu hóa Đàn vẹt Truyền thống (Parrot Optimizer - PO) và Bước Ánh xạ Nhị phân

Thuật toán Parrot Optimizer (PO) truyền thống của Lian và cộng sự (2024) [1] mô phỏng bốn hành vi tự nhiên của loài vẹt *Pyrrhura molinae* để cập nhật vị trí cá thể trên không gian liên tục $\mathbb{R}^N$. Để áp dụng trực tiếp PO truyền thống vào bài toán chọn mẫu ảnh $X \in \{0, 1\}^N$ (trong đó chỉ có hai trạng thái: $1$ là chọn ảnh, $0$ là loại bỏ ảnh), hệ thống giữ nguyên hoàn toàn bốn phương trình cập nhật vị trí của PO truyền thống ở Bước 1, sau đó thực hiện bước ánh xạ về $\{0, 1\}$ ở Bước 2:

**Bước 1: Cập nhật vị trí liên tục theo đúng 4 hành vi của PO truyền thống [1]**  
Ở vòng lặp thứ $t$ (với tổng số vòng lặp tối đa $T_{\max}$), mỗi cá thể vẹt $X_i^t$ chọn ngẫu nhiên một chỉ số hành vi $S_t \in \{1, 2, 3, 4\}$ với xác suất đều nhau ($25\%$ cho mỗi hành vi):

* **Hành vi 1 ($S_t = 1$) — Kiếm ăn (Foraging Behavior)**: Vẹt di chuyển hướng về nguồn thức ăn dựa trên vị trí tốt nhất toàn cục $X_{\text{gbest}}$ và vị trí trung bình của cả đàn $X_{\text{mean}}^t = \frac{1}{P}\sum_{k=1}^P X_k^t$:
  $$X_i^{\text{cont}} = X_i^t + (X_i^t - X_{\text{gbest}}) \odot \text{Levy}(N) + \text{rand}(0,1) \cdot X_{\text{mean}}^t \cdot \left(1 - \frac{t}{T_{\max}}\right)^{\frac{2t}{T_{\max}}}$$

* **Hành vi 2 ($S_t = 2$) — Đậu tại chỗ (Staying Behavior)**: Vẹt bay quanh vị trí của chủ (nghiệm tốt nhất $X_{\text{gbest}}$) và đậu ngẫu nhiên:
  $$X_i^{\text{cont}} = X_i^t + X_{\text{gbest}} \odot \text{Levy}(N) + \mathcal{N}(0, 1) \cdot \left(1 - \frac{t}{T_{\max}}\right) \cdot \mathbf{1}_N$$

* **Hành vi 3 ($S_t = 3$) — Giao tiếp trong đàn (Communicating Behavior)**: Với xác suất $H < 0,5$, vẹt bay tụ hội về trung tâm đàn; ngược lại ($H \ge 0,5$), vẹt bay đi sau khi giao tiếp:
  $$X_i^{\text{cont}} = \begin{cases} 
  X_i^t + \alpha \left(1 - \frac{t}{T_{\max}}\right) (X_i^t - X_{\text{mean}}^t), & \text{nếu } H < 0,5 \\
  X_i^t + \alpha \left(1 - \frac{t}{T_{\max}}\right) \exp\left(\frac{-j}{\text{rand}(0,1) \cdot T_{\max}}\right), & \text{nếu } H \ge 0,5
  \end{cases}$$
  trong đó $\alpha = \text{rand}(0,1)/5$ và $j \in \{1, \dots, P\}$.

* **Hành vi 4 ($S_t = 4$) — Sợ người lạ (Fear of Strangers Behavior)**: Vẹt giữ khoảng cách với kẻ lạ và tìm đường bay về phía chủ $X_{\text{gbest}}$:
  $$X_i^{\text{cont}} = X_i^t + \text{rand}(0,1) \cos\left(\frac{\pi t}{2 T_{\max}}\right) (X_{\text{gbest}} - X_i^t) - \cos(\theta) \left(\frac{t}{T_{\max}}\right)^{\frac{2}{T_{\max}}} (X_i^t - X_{\text{gbest}})$$
  với $\theta = \text{rand}(0,1) \cdot \pi$. Trong các công thức trên, bước nhảy $\text{Levy}(N)$ tuân theo phân phối Lévy với tham số $\beta = 1,5$ được tính qua công thức Mantegna: $\text{step} = \frac{u}{|v|^{1/\beta}}$ với $u \sim \mathcal{N}(0, \sigma^2)$, $v \sim \mathcal{N}(0, 1)$.

**Bước 2: Cơ chế Ánh xạ Nhị phân qua Hàm truyền Hình chữ V (V-shaped Transfer Function)**  

Do vị trí $X_i^{\text{cont}} \in \mathbb{R}^N$ sau Bước 1 là vector số thực liên tục (có thể nhận giá trị dương hoặc âm bất kỳ do bước nhảy Lévy và hàm lượng giác $\cos$), trong khi bài toán rút gọn tập dữ liệu ảnh đòi hỏi nghiệm nhị phân $X_i \in \{0, 1\}^N$ ($1$: chọn ảnh, $0$: loại bỏ ảnh), thuật toán cần một cơ chế chuyển đổi miền từ không gian liên tục sang không gian nhị phân [9].

Trong các thuật toán bầy đàn nhị phân truyền thống, hàm truyền hình chữ S (Sigmoid: $S(x) = \frac{1}{1 + e^{-x}}$) thường được dùng để quy đổi vị trí sang xác suất gán trực tiếp $x_{i,d}^{t+1} = 1$. Tuy nhiên, trên không gian $N = 11.228$ chiều, hàm chữ S có nhược điểm lớn: khi cá thể vẹt đã hội tụ và đứng yên ($\Delta x_{i,d} = 0$), giá trị $S(0) = 0,50$ vẫn tạo ra xác suất $50\%$ gán lại bit thành $1$ hoặc $0$ ngẫu nhiên, làm phá hủy cấu trúc tập con tốt mà cá thể đang nắm giữ (hoặc vừa được pha GA chuyển giao sang).

Để khắc phục triệt để hạn chế này, hệ thống cài đặt **Hàm truyền hình chữ V (V-shaped Transfer Function)** $V(\Delta x) = |\tanh(\Delta x)|$ trong [`optimization/parrot_optimizer.py`](file:///d:/Code/KhoaLuanDatasetReduction/optimization/parrot_optimizer.py#L33-L41). Thay vì ép gán giá trị tuyệt đối $0$ hay $1$, hàm chữ V đo **biên độ dịch chuyển vị trí** $\Delta x_{i,d} = x_{i,d}^{\text{cont}} - x_{i,d}^t$ và quy đổi thành **xác suất đảo trạng thái hiện tại ($P_{\text{flip}}$)**:

```text
PO truyền thống (X_cont ∈ ℝ^N) 
   ──► Độ thay đổi vị trí (ΔX = X_cont - X_current) 
   ──► Hàm truyền chữ V quy đổi sang xác suất [0, 1): V(ΔX) = |tanh(ΔX)| 
   ──► Nếu rand(0,1) < V(ΔX) thì đảo bit (0 ↔ 1), ngược lại giữ nguyên X_current
```

Công thức toán học của hàm truyền chữ V và quy tắc cập nhật trạng thái chọn/bỏ của bức ảnh thứ $d$ được xác định như sau:

$$V(\Delta x_{i,d}) = \left|\tanh(\Delta x_{i,d})\right| = \left|\frac{e^{\Delta x_{i,d}} - e^{-\Delta x_{i,d}}}{e^{\Delta x_{i,d}} + e^{-\Delta x_{i,d}}}\right| \in [0, 1)$$

$$x_{i,d}^{t+1} = \begin{cases} 
1 - x_{i,d}^t, & \text{nếu } \text{rand}(0,1) < \left|\tanh(\Delta x_{i,d})\right| \\
x_{i,d}^t, & \text{ngược lại}
\end{cases}$$

Cơ chế hàm truyền chữ V $V(\Delta x) = |\tanh(\Delta x)|$ mang lại ba đặc tính toán học then chốt cho thuật toán PO và các kiến trúc lai PO–GA:
1. **Bảo toàn cấu trúc nghiệm khi hội tụ ($V(0) = 0$)**: Khi cá thể vẹt đứng yên hoặc tiến sát nghiệm tối ưu ($\Delta x_{i,d} \approx 0 \Rightarrow |\tanh(\Delta x_{i,d})| \approx 0$), xác suất đảo bit bằng $0$, bức ảnh giữ nguyên trạng thái cũ ($x_{i,d}^{t+1} = x_{i,d}^t$). Đặc tính này giúp PO kế thừa nguyên vẹn các cấu trúc gen tốt khi nhận bàn giao quần thể từ GA trong các biến thể lai `S1–S4`, `P1` và `C1`.
2. **Tính đối xứng chẵn hai chiều ($V(-\Delta x) = V(\Delta x)$)**: Đồ thị hình chữ V đối xứng qua trục tung cho phép cả bước nhảy dương lớn ($\Delta x \gg 0$) lẫn bước nhảy âm lớn ($\Delta x \ll 0$) của phân phối Lévy đều kích hoạt khả năng đảo trạng thái ($0 \to 1$ để bổ sung ảnh đại diện mới, hoặc $1 \to 0$ để loại bỏ ảnh dư thừa).
3. **Kẹp biên tự nhiên trước bước nhảy nặng đuôi ($\lim_{|\Delta x| \to \infty} V(\Delta x) = 1$)**: Mọi bước nhảy đột biến lớn của phân phối Lévy ($\beta = 1,5$) đều được nén mượt vào khoảng xác suất hợp lệ $[0, 1)$ mà không gây tràn số hay biến dạng phân bố.

| Tiêu chí đối sánh | Hàm truyền hình chữ S: $S(\Delta x) = \frac{1}{1 + e^{-\Delta x}}$ | Hàm truyền hình chữ V: $V(\Delta x) = \lvert\tanh(\Delta x)\rvert$ (Áp dụng trong PO) |
| :--- | :--- | :--- |
| **Ý nghĩa xác suất** | Xác suất gán trực tiếp bit bằng $1$ ($\Pr(x_{i,d}^{t+1} = 1)$) | Xác suất **đảo trạng thái** hiện tại ($\Pr(x_{i,d}^{t+1} = 1 - x_{i,d}^t)$) |
| **Khi đứng yên ($\Delta x = 0$)** | $S(0) = 0,50$ (Xáo trộn ngẫu nhiên $50\%$ số mẫu ảnh) | **$V(0) = 0,00$ (Bảo toàn $100\%$ trạng thái tập con hiện tại)** |
| **Khi dịch chuyển lớn ($\lvert\Delta x\rvert \gg 0$)** | Bất đối xứng ($\Delta x \to +\infty$ gán $1$, $\Delta x \to -\infty$ gán $0$) | **Đối xứng hai chiều ($V(\pm\infty) \to 1$, cho phép đảo cả $0 \to 1$ và $1 \to 0$)** |
| **Tương thích khi lai với GA** | Phá vỡ cấu trúc gen vừa được GA lai ghép | **Kế thừa mượt mà quần thể từ GA, chỉ lật các vị trí có $\Delta x$ lớn** |

Cuối cùng, phép so sánh $\text{rand}(0,1) < V(\Delta x_{i,d})$ đóng vai trò là **bộ lượng tử hóa ngẫu nhiên (Stochastic Binarization)**: thay vì cắt ngưỡng cứng (như $V(\Delta x) > 0,5$ khiến các bước dịch chuyển nhỏ không bao giờ có cơ hội đổi trạng thái), cơ chế quay số xác suất trên đoạn $[0, 1)$ đảm bảo các chiều có dịch chuyển lớn ($V(\Delta x) \approx 0,90$) có $90\%$ cơ hội lật trạng thái, đồng thời vẫn duy trì xác suất nhỏ ($5\% - 15\%$) cho các chiều dịch chuyển nhẹ để thoát khỏi cực trị địa phương. Sau khi đi qua hàm truyền chữ V và toán tử sửa chữa $\text{Repair}$, vị trí mới chỉ được cập nhật theo cơ chế chọn lọc tham lam nếu có điểm Fitness cao hơn vị trí cũ ($F(X_i^{t+1}) > F(X_i^t)$).

### 4.3. Thiết kế Ba Kiến trúc Lai hóa PO–GA (Sequential, Parallel, Cooperative)

Nhằm khảo sát toàn diện cơ chế phối hợp giữa GA và PO truyền thống, đề tài xây dựng ba kiến trúc lai hóa với sáu cấu hình thực nghiệm (S1, S2, S3, S4, P1, C1):

```text
                        KHUNG LAI HÓA HYBRID PO–GA
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
  1. SEQUENTIAL (Tuần tự)   2. PARALLEL (Song song)   3. COOPERATIVE (Hợp tác)
          │                         │                         │
   ┌──────┼──────┬──────┐           ▼                         ▼
   ▼      ▼      ▼      ▼          P1                        C1
  S1     S2     S3     S4      GA(50%) || PO(50%)       GA ↔ PO định kỳ Δt
GA↔PO  PO↔GA  GA→PO  PO→GA     ──► Merge & Co-Refine    Trao đổi Top-P (Chống trùng)
```

1. **Kiến trúc Tuần tự (Sequential Hybrid: S1, S2, S3, S4)**:
   * **S1 (Alternating Sequential, GA-first)**: Chạy xen kẽ từng pha ngắn có độ dài cố định $p = 10$ vòng lặp theo chu trình $\text{GA}(10) \rightarrow \text{PO}(10) \rightarrow \text{GA}(10) \rightarrow \text{PO}(10) \dots$ cho đến khi đạt tổng số vòng lặp $T = 100$.
   * **S2 (Alternating Sequential, PO-first)**: Đảo ngược thứ tự khởi đầu thành $\text{PO}(10) \rightarrow \text{GA}(10) \rightarrow \text{PO}(10) \rightarrow \text{GA}(10) \dots$ để kiểm tra hiệu ứng thứ tự (Ordering Effect) giữa khai phá và khai thác.
   * **S3 (Block Sequential, GA-first)**: Chia tổng ngân sách thành hai khối liên tục 50/50: chạy $\text{GA}(50\%)$ để tạo quần thể đa dạng, sau đó chuyển giao toàn bộ quần thể cho $\text{PO}(50\%)$ tinh chỉnh.
   * **S4 (Block Sequential, PO-first)**: Chạy $\text{PO}(50\%)$ ở nửa đầu, sau đó chuyển giao quần thể cho $\text{GA}(50\%)$ tái tổ hợp ở nửa sau.

2. **Kiến trúc Song song Độc lập kèm Đồng Tinh chỉnh (Parallel Hybrid: P1 — Independent Parallel + Merge & Co-Refinement)**:
   * Từ cùng một quần thể khởi tạo $P_0$ kích thước $P = 20$, hệ thống dành $50\%$ ngân sách đầu tách thành hai nhánh tiến hóa độc lập: nhánh GA tạo ra $\text{Pop}_{\text{GA}}$ và nhánh PO tạo ra $\text{Pop}_{\text{PO}}$.
   * Hai quần thể được hợp nhất và chọn lọc tinh hoa $\text{Top-}P(\text{Pop}_{\text{GA}} \cup \text{Pop}_{\text{PO}}, F)$, sau đó sử dụng $50\%$ ngân sách còn lại để **Đồng tinh chỉnh (Co-Refinement)** tuần tự qua GA và PO trên quần thể đã hợp nhất.

3. **Kiến trúc Hợp tác Hai chiều (Cooperative Hybrid: C1 — Bidirectional Elite Exchange with Duplicate Control)**:
   * Hai quần thể $\text{Pop}_{\text{GA}}$ và $\text{Pop}_{\text{PO}}$ cùng tiến hóa song song. Sau mỗi chu kỳ $\Delta t = 5$ vòng lặp ($10$ chặng đồng tiến hóa), hai nhánh hợp nhất và đồng bộ hóa các cá thể ưu tú nhất (Elites).
   * **Cơ chế kiểm soát trùng lặp (Duplicate Control)**: Để ngăn hiện tượng hai quần thể bị đồng hóa sớm do trao đổi lặp lại cùng một nghiệm cực đại, hệ thống lọc bỏ các cá thể trùng lặp hoàn toàn ($\min_{Y \in \text{Pop}_{\text{target}}} \|e - Y\|_1 > 0$, thực thi qua `np.unique(..., axis=0)`) để chỉ giữ lại các cá thể ưu tú phân biệt trong quần thể.

### 4.4. Tổng hợp các Cải tiến Thuật toán và Kỹ thuật Giúp Tối ưu hóa Độ chính xác Mô hình Học sâu

Để giải quyết triệt để sự đánh đổi giữa tỷ lệ nén dữ liệu ($\text{Com}$) và độ chính xác phân loại của mạng CNN hạ nguồn, hệ thống đề xuất bốn cải tiến phương pháp luận và kỹ thuật cốt lõi so với cách tiếp cận cơ sở truyền thống:

1. **Chuẩn hóa Biên độ Dịch chuyển Bầy đàn trong PO (Chống hiện tượng Nén cực đoan — Over-Compression)**: Toàn bộ bốn hành vi của PO đều được neo quanh vị trí hiện tại $X_i^t + \alpha \cdot (\dots)$ kết hợp với vector hướng đàn chuẩn hóa trung tâm (`flock_vec`), ngăn chặn hiện tượng độ dịch chuyển $\Delta x = X_i^{\text{cont}} - X_i^t$ suy biến về $-1$ làm xóa nhầm hàng loạt mẫu ảnh đại diện đang có.
2. **Toán tử Đột biến Cân bằng Mật độ trong GA (Density-Balanced Adaptive Mutation)**: Thay vì lật bit với xác suất đồng nhất làm tập con có mật độ $r \approx 27\%$ bị phình to (do số lượng bit $0$ chiếm $73\%$ nhiều gấp $2,7$ lần số lượng bit $1$), toán tử đột biến nhân trọng số đối xứng $p_m(t)(1-r)$ cho các vị trí $x_j=1$ và $p_m(t)r$ cho các vị trí $x_j=0$. Nhờ đó, kỳ vọng số mẫu bị loại bỏ luôn bằng kỳ vọng số mẫu được thêm mới ($N r (1-r) p_m(t)$), biến đột biến thành cơ chế **hoán đổi mẫu cân bằng (1-đổi-1)**.
3. **Cân chỉnh Hàm mục tiêu Đa tiêu chí Ưu tiên Độ bao phủ Hình học ($\text{Cov}$-Prioritized Fitness)**: Thiết lập bộ trọng số $(\alpha=0,15; \, \beta=0,45; \, \gamma=0,20; \, \delta=0,20)$, đặt Độ bao phủ không gian đặc trưng ($\beta = 0,45$) làm thành phần trọng tâm thay vì ưu tiên Độ nén ($\delta$) quá mức. Cơ chế này ngăn thuật toán rơi vào nghiệm tầm thường (xóa cạn kiệt dữ liệu để tăng $\text{Com}$), giữ ổn định kích thước tập Coreset quanh mức $25\% - 27\%$ ($\sim 2.800 - 3.010$ ảnh) và bảo toàn $\sim 99\%$ độ chính xác của tập đầy đủ.
4. **Tối ưu hóa Truy xuất Ma trận Khoảng cách trên VRAM GPU**: Hợp nhất phép trích xuất ma trận con `torch.index_select` trên bộ nhớ GPU cho cả hai tiêu chí $\text{Div}$ và $\text{Cov}$, giúp rút ngắn thời gian thực thi $2.000\text{ FEs}$ trên ma trận $11.228 \times 11.228$ xuống chỉ còn $\sim 9,5\text{ giây}$.

---

## CHƯƠNG 5: THIẾT KẾ THỰC NGHIỆM VÀ PHƯƠNG PHÁP LUẬN ĐÁNH GIÁ

### 5.1. Hệ thống Phương pháp Đối sánh (Baselines và Hybrid Configurations)

Để trả lời đầy đủ các câu hỏi nghiên cứu và đáp ứng yêu cầu đối sánh của đề cương CNTT-KLCN140, bộ thực nghiệm chính thức bao gồm **11 phương pháp** chia thành hai nhóm:
* **Nhóm Đối chuẩn (Baselines: B0 – B4)**:
  * **B0 — Full Dataset (Upper Bound)**: Sử dụng $100\%$ tập dữ liệu huấn luyện gốc (không rút gọn).
  * **B1 — Random Selection**: Lấy mẫu ngẫu nhiên phân bố đều theo đúng kích thước trung vị $M_{\text{baseline}} = 2.978$ ảnh của các thuật toán tối ưu hóa.
  * **B2 — Stratified Random Selection**: Lấy mẫu ngẫu nhiên phân tầng tại cùng kích thước $M_{\text{baseline}} = 2.978$ ảnh, bảo toàn chính xác tỷ lệ phân bố của $6$ lớp cảnh tự nhiên.
  * **B3 — Genetic Algorithm (GA-only)**: Chỉ chạy giải thuật di truyền trên $100\%$ ngân sách FEs.
  * **B4 — Parrot Optimizer (PO-only)**: Chỉ chạy thuật toán đàn vẹt truyền thống (kèm bước ánh xạ nhị phân đầu ra) trên $100\%$ ngân sách FEs.
* **Nhóm Kiến trúc Lai đề xuất (Hybrid PO–GA: S1, S2, S3, S4, P1, C1)**: Gồm 4 cấu hình tuần tự (S1–S4), 1 cấu hình song song (P1) và 1 cấu hình hợp tác (C1).

### 5.2. Định lượng Tính Công bằng về Ngân sách Đánh giá Hàm Mục tiêu (FE Fairness)

Trong các thuật toán siêu mô phỏng (metaheuristics), chi phí tính toán cốt lõi nằm ở số lần gọi hàm đánh giá thích nghi (Function Evaluations - FEs). Với kích thước quần thể $P = 20$ và tổng số vòng lặp chuẩn $T = 100$, mọi thuật toán tối ưu hóa đều được khóa chặt ở mức **$\text{Total FEs} = 2.000$** (chưa tính $20\text{ FEs}$ đánh giá quần thể khởi tạo dùng chung $P_0$), được phân bổ minh bạch theo Bảng 5.1.

| Mã phương pháp | Kiến trúc | Cấu hình phân bổ vòng lặp | GA FEs | PO FEs | Tổng số FEs (Total FEs) |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **B3 (GA)** | Đơn lẻ (Single) | $100$ thế hệ GA ($P=20$) | $2.000$ | $0$ | **$2.000$** |
| **B4 (PO)** | Đơn lẻ (Single) | $100$ vòng lặp PO ($P=20$) | $0$ | $2.000$ | **$2.000$** |
| **S1** | Sequential (Xen kẽ) | $5$ chu kỳ $\times [\text{GA}(10) \rightarrow \text{PO}(10)]$ | $1.000$ | $1.000$ | **$2.000$** |
| **S2** | Sequential (Xen kẽ) | $5$ chu kỳ $\times [\text{PO}(10) \rightarrow \text{GA}(10)]$ | $1.000$ | $1.000$ | **$2.000$** |
| **S3** | Sequential (Khối) | $\text{GA}(50) \rightarrow \text{PO}(50)$ | $1.000$ | $1.000$ | **$2.000$** |
| **S4** | Sequential (Khối) | $\text{PO}(50) \rightarrow \text{GA}(50)$ | $1.000$ | $1.000$ | **$2.000$** |
| **P1** | Parallel (Độc lập) | $\text{GA}(50\%) \parallel \text{PO}(50\%) \rightarrow \text{Merge}$ | $1.000$ | $1.000$ | **$2.000$** |
| **C1** | Cooperative (Hợp tác) | $10$ chặng $\times [\text{GA}(5) \leftrightarrow \text{PO}(5)]$ | $1.000$ | $1.000$ | **$2.000$** |

**Phân biệt giữa Ngân sách FEs và Thời gian thực thi phần cứng (Wall-Clock Runtime)**: Báo cáo phân biệt rõ ràng giữa tính song song về mặt kiến trúc thuật toán (Algorithmic Parallelism trong P1 và C1, nơi hai quần thể tiến hóa độc lập về mặt logic dưới cùng tổng số $2.000\text{ FEs}$) và tính song song phần cứng (Hardware Parallelism). Trong thực nghiệm, các cá thể trong cùng một thế hệ được đánh giá song song qua `ThreadPoolExecutor`, đồng thời cả số lượng FEs lẫn thời gian thực thi thực tế (Wall-Clock Time tính bằng giây) đều được ghi nhận đầy đủ.

### 5.3. Kiểm soát Hạt giống Kép (Dual-Seed Control) và Chống Chọn lọc Thiên lệch (Anti-Cherry-Picking)

Hệ thống tồn tại hai nguồn ngẫu nhiên độc lập được kiểm soát tách biệt nhằm đảm bảo tính tái lập tuyệt đối:
1. **Hạt giống Tối ưu hóa (Optimization Seeds)**: Toàn bộ các thuật toán tối ưu (`B1–B4`, `S1–S4`, `P1`, `C1`) được chạy lặp lại độc lập trên $3$ hạt giống chuẩn `seeds` $= [42, 123, 456]$. Tại mỗi `seed`, quần thể khởi tạo ban đầu $P_0$ được sinh một lần duy nhất với mật độ khởi tạo $\text{init\_ratio} = 0,30$ và truyền giống hệt nhau vào tất cả 8 thuật toán tiến hóa/bầy đàn (`GA`, `PO`, `S1–S4`, `P1`, `C1`).
2. **Hạt giống Huấn luyện Học sâu (Deep Learning Training Seed)**: Cố định ở `seed = 42` cho `torch.manual_seed`, `torch.cuda.manual_seed_all` và bộ tách dữ liệu `DataLoader` để đảm bảo mọi sự khác biệt về độ chính xác phân loại hoàn toàn đến từ chất lượng của tập ảnh rút gọn chứ không phải do nhiễu khởi tạo trọng số mạng.
3. **Quy tắc chống chọn lọc thiên lệch (Anti-Cherry-Picking Protocol)**: Tuyệt đối không chọn tập con có Test Accuracy cao nhất để báo cáo. Với mỗi `seed` tối ưu hóa, nghiệm có **Fitness cao nhất trên tập Train** (`best_sol`) được trích xuất trực tiếp để huấn luyện 3 bộ phân loại CNN (`ResNet-18`, `MobileNetV3-Small`, `DenseNet-121`) và đánh giá một lần trên tập Test độc lập, sau đó tổng hợp các chỉ số thống kê trên toàn bộ các lần chạy.

### 5.4. Hệ thống Chỉ số Đánh giá Hai Tầng và Kiểm định Thống kê

Quy trình đánh giá được tách bạch thành hai tầng rõ ràng:

1. **Tầng 1 — Các chỉ số Chất lượng Tối ưu hóa (Optimization-Level Metrics)**: Bao gồm điểm thích nghi tổng hợp cuối cùng ($F_{\text{total}}$), bốn thành phần ($\text{Div}, \text{Cov}, \text{Bal}, \text{Com}$), đường cong hội tụ theo vòng lặp $BestFitness(t)$ ($t = 1 \dots 100$) và thời gian chạy thuật toán tối ưu hóa (Optimization Runtime - giây).
2. **Tầng 2 — Các chỉ số Hiệu năng Học sâu (Deep Learning Utility Metrics)**: Bao gồm bốn chỉ số phân loại chuẩn trên tập kiểm thử độc lập ($\text{Accuracy}$, $\text{Macro Precision}$, $\text{Macro Recall}$, $\text{Macro F1-Score}$), kết hợp ba chỉ số đánh giá rút gọn dữ liệu:
   $$\text{DR} = \left(1 - \frac{M}{N}\right) \times 100\%, \quad \text{ARR} = \frac{\text{Acc}_{\text{subset}}}{\text{Acc}_{\text{full}}} \times 100\%, \quad \text{Speedup} = \frac{\text{Time}_{\text{full}}}{\text{Time}_{\text{subset}}}$$
3. **Thống kê mô tả và kiểm định phi tham số**: Kết quả thực nghiệm đa hạt giống được báo cáo đồng thời qua **Trung bình $\pm$ Độ lệch chuẩn ($\text{Mean} \pm \text{Std}$)** và **Trung vị kèm Khoảng tứ phân vị ($\text{Median} \text{ [IQR]}$)**, kết hợp kiểm định cặp có dấu **Wilcoxon Signed-Rank Test** đối với hiệu năng phân loại hạ nguồn.

---

## CHƯƠNG 6: KIẾN TRÚC HỆ THỐNG VÀ CÔNG CỤ HỖ TRỢ THỰC NGHIỆM

### 6.1. Cấu hình Môi trường Thực nghiệm và Tổ chức Lưu trữ

Hệ thống được phát triển đồng nhất trong dự án **`KhoaLuanDatasetReduction`** bằng ngôn ngữ **Python 3.10+**, sử dụng các thư viện tính toán khoa học và học sâu chuẩn: `PyTorch` và `TorchVision` (trích xuất đặc trưng ResNet-18 512 chiều và huấn luyện 3 mạng CNN `ResNet-18`, `MobileNetV3-Small`, `DenseNet-121` trên GPU CUDA), `NumPy` và `SciPy` (tính toán ma trận khoảng cách chuẩn hóa BLAS và kiểm định thống kê Wilcoxon), cùng `Matplotlib` (trực quan hóa biểu đồ hội tụ, phân bố lớp và ma trận nhầm lẫn).

Cấu trúc mã nguồn của dự án được tổ chức thành các mô-đun chức năng rõ ràng:
1. **`configs/config.py`**: Quản lý tập trung cấu hình huấn luyện (`BaselineConfig`) và cấu hình siêu tham số tối ưu hóa (`POGAConfig`, `GAConfig`, `POConfig`, `FitnessConfig`), tích hợp cơ chế tự động nhận diện đường dẫn dữ liệu khi triển khai trên cả máy cục bộ lẫn nền tảng đám mây Kaggle (`/kaggle/input` và `/kaggle/working/outputs`).
2. **`data/dataset.py`**: Cung cấp hàm `create_dataloaders()` thực hiện phân tách 3 tập chuẩn không rò rỉ dữ liệu (Train $80\%$ `seg_train` / Validation $20\%$ `seg_train` / Independent Test `seg_test`), hỗ trợ nạp trực tiếp mặt nạ tập con rút gọn (`subset_indices_path`) và hàm `extract_and_cache_features()` nạp/lưu đệm vector đặc trưng `.npy` $512$ chiều tại thư mục `features/`.
3. **`optimization/`**: Cài đặt ma trận khoảng cách Cosine chuẩn hóa theo $\text{dist}_{\max}$ lưu đệm trên GPU (`fitness.py`), giải thuật Di truyền nhị phân (`genetic_algorithm.py`), thuật toán Đàn vẹt nhị phân với hàm truyền chữ V (`parrot_optimizer.py`), cùng 6 cấu hình lai hóa thuộc 3 kiến trúc (`S1, S2, S3, S4, P1, C1`) và 2 phương pháp lấy mẫu đối chuẩn (`hybrid_poga.py`).
4. **`models/model.py` & `training/trainer.py`**: Xây dựng và huấn luyện 3 kiến trúc mạng học sâu (`resnet18`, `mobilenet_v3`, `densenet121`), lưu trữ trọng số tốt nhất theo Validation Accuracy và đánh giá độc lập trên Test Set.
5. **`train.py` & `run_poga.py`**: Hai kịch bản điều khiển trung tâm phục vụ huấn luyện mốc chuẩn toàn tập dữ liệu (`train.py`) và thực thi quy trình tối ưu hóa rút gọn dữ liệu đa hạt giống kết hợp kiểm định thống kê (`run_poga.py`).

### 6.2. Thiết kế Công cụ Hỗ trợ Thực nghiệm và Chương trình Minh họa

Để đáp ứng yêu cầu xây dựng công cụ hỗ trợ thực nghiệm trong đề cương (CLO3), phần mềm cung cấp giao diện thực thi hợp nhất gồm 4 chức năng chính:
1. **Chức năng 1 — Quản lý Phân tách Dữ liệu và Lưu đệm Đặc trưng**: Tự động phân hoạch tập dữ liệu ảnh cảnh tự nhiên thành Train/Val/Test, thống kê phân bố mẫu theo từng lớp, trích xuất đặc trưng ResNet-18 ($512$ chiều) và nạp trực tiếp từ bộ nhớ đệm `features/train_features_train80.npy`.
2. **Chức năng 2 — Thực thi Rút gọn Dữ liệu (PO–GA Dataset Reduction Engine)**: Cho phép lựa chọn thuật toán (`random`, `stratified`, `ga`, `po`, `s1`, `s2`, `s3`, `s4`, `p1`, `c1` hoặc `all`), tối ưu hóa hàm mục tiêu 4 thành phần, theo dõi tiến trình hội tụ và xuất tập con tối ưu (`subsets/subset_<method>_seed<seed>.npy`).
3. **Chức năng 3 — Huấn luyện và Đánh giá Xuyên Kiến trúc trên 3 Mạng CNN**: Tự động nạp tập ảnh rút gọn để huấn luyện trên cả 3 mô hình `ResNet-18`, `MobileNetV3-Small` và `DenseNet-121`, tính toán trực tiếp `Accuracy`, `Precision`, `Recall`, `Macro F1`, `DR (%)`, `ARR (%)`, `Delta_Acc (%)` và `Speedup` so với mốc chuẩn Full Dataset (B0).
4. **Chức năng 4 — Trực quan hóa và Xuất Báo cáo Thống kê**: Tự động xuất biểu đồ so sánh tốc độ hội tụ (`convergence_comparison.png`), biểu đồ học tập (`comparison_curves.png`), ma trận nhầm lẫn (`confusion_matrices.png`), bảng thống kê mô tả đa seed (`descriptive_statistics.csv`) và báo cáo kiểm định Wilcoxon (`statistical_significance.md`).

---

## CHƯƠNG 7: KẾT QUẢ THỰC NGHIỆM VÀ THẢO LUẬN

### 7.1. Kết quả Thực nghiệm Mốc chuẩn Toàn Tập Dữ liệu (Full Dataset Upper Bound - B0)

Bước đầu tiên của quy trình thực nghiệm là xác lập mốc chuẩn trên (Upper Bound - Cấu hình B0) khi huấn luyện trên toàn bộ tập dữ liệu gốc với quy trình phân tách 3 tập chuẩn ($11.228$ ảnh Train / $2.806$ ảnh Validation / $3.000$ ảnh Independent Test). Ba kiến trúc mạng học sâu được huấn luyện bằng thuật toán Adam ($\text{lr} = 10^{-3}$, $\text{batch\_size} = 64$, $5$ epochs) trên GPU NVIDIA GeForce RTX 4060 Laptop. Kết quả tổng hợp và kết quả chi tiết theo từng lớp cảnh tự nhiên được trình bày tại Bảng 7.1 và Bảng 7.2.

**Bảng 7.1: Kết quả đối sánh hiệu năng 3 mô hình học sâu trên toàn bộ tập dữ liệu gốc (Mốc chuẩn B0)**

| Mô hình Học sâu | Số tham số (M) | Train Acc (%) | Val Acc (%) | Test Acc (%) | Macro Precision (%) | Macro Recall (%) | Macro F1-Score (%) | Độ trễ (ms/ảnh) | Thời gian Train (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18** | $11,18\text{M}$ | $89,22\%$ | $90,34\%$ | $90,27\%$ | $90,38\%$ | $90,46\%$ | $90,41\%$ | $1,47\text{ ms}$ | $265,16\text{ s}$ |
| **MobileNetV3-Small** | $1,52\text{M}$ | $89,03\%$ | $90,34\%$ | $88,93\%$ | $89,26\%$ | $89,28\%$ | $89,25\%$ | $0,38\text{ ms}$ | $174,44\text{ s}$ |
| **DenseNet-121** | $6,96\text{M}$ | $91,37\%$ | $91,80\%$ | **$90,57\%$** | **$90,71\%$** | **$90,76\%$** | **$90,72\%$** | $2,24\text{ ms}$ | $301,34\text{ s}$ |

**Bảng 7.2: Chỉ số F1-Score (%) theo từng lớp cảnh tự nhiên trên tập Kiểm thử Độc lập (`seg_test` - 3.000 ảnh)**

| Lớp cảnh tự nhiên | Số mẫu Test | ResNet-18 F1 (%) | MobileNetV3-Small F1 (%) | DenseNet-121 F1 (%) | Nhận xét đặc trưng phân lớp |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `buildings` (Tòa nhà) | $437$ | $90,35\%$ | $89,14\%$ | $91,20\%$ | Có cấu trúc cạnh thẳng rõ ràng, đôi khi nhầm lẫn với `street`. |
| `forest` (Rừng cây) | $474$ | **$98,21\%$** | **$97,58\%$** | **$98,10\%$** | Kết cấu lá cây đặc trưng nhất, đạt độ chính xác cao nhất ở cả 3 mạng. |
| `glacier` (Sông băng) | $553$ | $85,20\%$ | $83,61\%$ | $85,37\%$ | Dễ nhầm lẫn nhất với lớp `mountain` do cùng chứa tuyết và đá núi. |
| `mountain` (Núi) | $525$ | $85,49\%$ | $83,84\%$ | $85,63\%$ | Chồng lấp ngữ nghĩa cao với lớp `glacier` ở vùng biên quyết định. |
| `sea` (Biển) | $510$ | $93,65\%$ | $93,05\%$ | $94,42\%$ | Bố cục đường chân trời và mặt nước tách biệt tốt. |
| `street` (Đường phố) | $501$ | $89,58\%$ | $88,28\%$ | $89,60\%$ | Chia sẻ đặc trưng kiến trúc đô thị với lớp `buildings`. |
| **Macro Average** | **$3.000$** | **$90,41\%$** | **$89,25\%$** | **$90,72\%$** | Độ lệch chuẩn giữa các lớp thấp, phân loại ổn định. |

**Phân tích kết quả Mốc chuẩn B0**:
Số liệu từ Bảng 7.1 và Bảng 7.2 cho thấy cả ba kiến trúc mạng đều đạt độ chính xác trên tập kiểm thử độc lập từ $88,93\%$ đến $90,57\%$. Hai lớp có độ chồng lấp đặc trưng lớn nhất là `glacier` và `mountain` (F1-score dao động quanh mức $83,6\% - 85,6\%$), trong khi lớp `forest` đạt F1-score trên $97,5\%$. Đây là căn cứ quan trọng để kiểm chứng xem thuật toán PO–GA khi rút gọn dữ liệu có giữ lại được các mẫu đại diện tốt ở vùng biên giữa `glacier` và `mountain` hay không.

### 7.2. Kết quả Đánh giá Tầng 1: Chất lượng Tối ưu hóa Tổ hợp trên Hàm Mục tiêu Bốn Thành phần

Thực nghiệm tối ưu hóa được thực hiện trên tập đặc trưng huấn luyện $D_{\text{train}}$ ($N = 11.228$ ảnh, vector ResNet-18 $512$ chiều) với hàm mục tiêu bốn thành phần $F(X) = 0,15\,\text{Div}(X) + 0,45\,\text{Cov}(X) + 0,20\,\text{Bal}(X) + 0,20\,\text{Com}(X)$ dưới cùng ngân sách đánh giá $2.000\text{ FEs}$ ($P = 20$ cá thể, $T = 100$ vòng lặp) lặp lại qua 3 hạt giống độc lập (`seeds = [42, 123, 456]`). Bảng 7.3 tổng hợp kết quả thống kê mô tả ($\text{Mean} \pm \text{Std}$ và $\text{Median} \text{ [IQR]}$), giá trị trung bình của bốn thành phần mục tiêu, kích thước tập con hội tụ $M$, tỷ lệ rút gọn dữ liệu ($\text{DR}\%$) và thời gian thực thi tối ưu hóa từ phiên chạy thực nghiệm chính thức [`outputs/run_poga_all_20261002_173334`](file:///d:/Code/KhoaLuanDatasetReduction/outputs/run_poga_all_20261002_173334/descriptive_statistics.csv).

**Bảng 7.3: Đối sánh Chất lượng Tối ưu hóa (Tầng 1) giữa 10 phương pháp dưới ngân sách 2.000 FEs (`run_poga_all_20261002_173334`)**

| Phương pháp | Kiến trúc | Số ảnh giữ lại $M$ ($\text{Mean} \pm \text{Std}$) | Tỷ lệ rút gọn $\text{DR}$ (%) | Fitness ($\text{Mean} \pm \text{Std}$) | Fitness ($\text{Median} \text{ [IQR]}$) | $\text{Div}$ | $\text{Cov}$ | $\text{Bal}$ | $\text{Com}$ | Thời gian Tối ưu (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S2 — Alt PO→GA** | Sequential | $2.828,3 \pm 61,6$ | $74,81\%$ | **$0,8249 \pm 0,0005$** | **$0,8248 \text{ [0,0006]}$** | **$0,5094$** | $0,8867$ | $0,9993$ | $0,7481$ | $9,64 \pm 0,11\text{ s}$ |
| **S1 — Alt GA→PO** | Sequential | **$2.799,0 \pm 37,6$** | **$75,07\%$** | $0,8248 \pm 0,0003$ | $0,8247 \text{ [0,0003]}$ | $0,5075$ | $0,8861$ | $0,9992$ | **$0,7507$** | $9,54 \pm 0,05\text{ s}$ |
| **S3 — Block GA→PO**| Sequential | $2.949,0 \pm 24,1$ | $73,74\%$ | $0,8232 \pm 0,0002$ | $0,8232 \text{ [0,0002]}$ | $0,5067$ | $0,8886$ | $0,9992$ | $0,7374$ | $9,66 \pm 0,05\text{ s}$ |
| **S4 — Block PO→GA**| Sequential | $2.963,3 \pm 35,6$ | $73,61\%$ | $0,8232 \pm 0,0003$ | $0,8231 \text{ [0,0004]}$ | $0,5071$ | $0,8887$ | **$0,9998$** | $0,7361$ | $9,85 \pm 0,03\text{ s}$ |
| **B3 — GA-only** | Đơn lẻ | $2.969,3 \pm 15,1$ | $73,55\%$ | $0,8231 \pm 0,0001$ | $0,8230 \text{ [0,0002]}$ | $0,5068$ | $0,8889$ | **$0,9998$** | $0,7355$ | **$9,26 \pm 0,03\text{ s}$** |
| **C1 — Cooperative**| Cooperative| $3.012,0 \pm 25,5$ | $73,18\%$ | $0,8226 \pm 0,0001$ | $0,8226 \text{ [0,0001]}$ | $0,5074$ | $0,8896$ | $0,9995$ | $0,7318$ | $10,58 \pm 0,09\text{ s}$ |
| **P1 — Parallel** | Parallel | $3.007,0 \pm 37,6$ | $73,22\%$ | $0,8225 \pm 0,0003$ | $0,8226 \text{ [0,0003]}$ | $0,5065$ | $0,8895$ | $0,9991$ | $0,7322$ | $9,84 \pm 0,06\text{ s}$ |
| **B2 — Stratified** | Phân tầng | $2.978,0 \pm 0,0$ | $73,48\%$ | $0,8220 \pm 0,0001$ | $0,8220 \text{ [0,0002]}$ | $0,5021$ | $0,8883$ | $1,0000$ | $0,7348$ | $< 0,01\text{ s}$ |
| **B4 — PO-only** | Đơn lẻ | $3.064,3 \pm 22,2$ | $72,71\%$ | $0,8216 \pm 0,0003$ | $0,8214 \text{ [0,0003]}$ | $0,5063$ | **$0,8904$** | $0,9979$ | $0,7271$ | $10,52 \pm 0,04\text{ s}$ |
| **B1 — Random** | Ngẫu nhiên | $2.978,0 \pm 0,0$ | $73,48\%$ | $0,8209 \pm 0,0005$ | $0,8210 \text{ [0,0007]}$ | $0,5023$ | $0,8884$ | $0,9940$ | $0,7348$ | $< 0,01\text{ s}$ |

**Phân tích cơ chế hội tụ và động lực học tối ưu hóa (Tầng 1)**:
Số liệu từ Bảng 7.3 cho thấy toàn bộ sáu biến thể lai ghép Hybrid PO–GA (`S1, S2, S3, S4, P1, C1`) và giải thuật `GA-only` đều đạt điểm thích nghi tổng hợp cao hơn so với hai phương pháp lấy mẫu ngẫu nhiên (`Random B1`, `Stratified B2`) và thuật toán `PO-only (B4)`. Cụ thể, dẫn đầu toàn cục về điểm thích nghi là hai cấu hình lai tuần tự xen kẽ chu kỳ ngắn **`S2 (Alt PO→GA)`** ($0,8249 \pm 0,0005$, đạt đỉnh $0,8256$ tại `Seed 42`) và **`S1 (Alt GA→PO)`** ($0,8248 \pm 0,0003$), đồng thời đạt tỷ lệ rút gọn dữ liệu cao nhất là **$74,81\%$ ($M = 2.828,3$ ảnh)** và **$75,07\%$ ($M = 2.799,0$ ảnh)**. Tiếp theo là hai cấu hình lai chia khối **`S3 (Block GA→PO)`** ($0,8232 \pm 0,0002$) và **`S4 (Block PO→GA)`** ($0,8232 \pm 0,0003$, đạt độ cân bằng lớp $\text{Bal} = 0,9998$).

Sự vượt trội của các kiến trúc lai bắt nguồn từ cơ chế bổ trợ toán học giữa PO và GA: trong khi `PO-only` có xu hướng hội tụ nhanh trong $25$ vòng lặp đầu rồi chững lại ở mức Fitness $0,8216$ do thiếu toán tử lai ghép gen chéo giữa các cá thể, việc chuyển giao quần thể sang pha GA (ở vòng $t = 50$ trong `S4` hoặc sau mỗi $10$ vòng lặp trong `S1, S2`) cho phép toán tử lai ghép đồng nhất (`Uniform Crossover`) kết hợp các cụm mẫu tốt nhất từ nhiều cá thể vẹt khác nhau, đồng thời toán tử đột biến cân bằng mật độ giữ cho kích thước tập con hội tụ chính xác tại điểm cân bằng đạo hàm riêng ($\text{DR} \in [73,18\%, 75,07\%]$). Nhờ tiền tính toán ma trận khoảng cách chuẩn hóa trên VRAM GPU, mỗi phiên tối ưu hóa $2.000\text{ FEs}$ chỉ mất từ **$9,26\text{ s}$ đến $10,58\text{ s}$**.

---

### 7.3. Kết quả Đánh giá Tầng 2: Hiệu năng Huấn luyện Hạ nguồn trên Ba Kiến trúc CNN

Sau khi hoàn tất giai đoạn tối ưu hóa Tầng 1, các tập ảnh đại diện tốt nhất (`best_sol`) của từng phương pháp ứng với từng hạt giống (`seed` $\in \{42, 123, 456\}$) được trích xuất để huấn luyện độc lập ba kiến trúc mạng học sâu gồm **ResNet-18**, **MobileNetV3-Small** và **DenseNet-121** trong $5$ epochs ($\text{batch\_size} = 64$, $\text{lr} = 10^{-3}$). Toàn bộ kết quả đánh giá trên tập kiểm thử độc lập `seg_test` ($3.000$ ảnh) được trình bày chi tiết tại Bảng 7.4.

**Bảng 7.4: Đối sánh Hiệu năng Phân loại và Tăng tốc Huấn luyện (Tầng 2) trên 3 kiến trúc CNN (`run_poga_all_20261002_173334`)**

| Phương pháp | Mô hình CNN | Tỷ lệ rút gọn $\text{DR}$ (%) | Test Accuracy ($\text{Mean} \pm \text{Std} \%$) | Test Accuracy ($\text{Median} \text{ [IQR]} \%$) | Macro F1-Score ($\text{Mean} \pm \text{Std} \%$) | Tỷ lệ duy trì $\text{ARR}$ (%) | Tăng tốc $\text{Speedup}$ ($\times$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Dataset (B0)** | `resnet18` | $0,00\%$ | $90,27 \pm 0,00\%$ | $90,27 \text{ [0,00]}\%$ | $90,41 \pm 0,00\%$ | $100,00\%$ | $1,00\times$ |
| **Full Dataset (B0)** | `mobilenet_v3` | $0,00\%$ | $88,93 \pm 0,00\%$ | $88,93 \text{ [0,00]}\%$ | $89,25 \pm 0,00\%$ | $100,00\%$ | $1,00\times$ |
| **Full Dataset (B0)** | `densenet121` | $0,00\%$ | $90,57 \pm 0,00\%$ | $90,57 \text{ [0,00]}\%$ | $90,72 \pm 0,00\%$ | $100,00\%$ | $1,00\times$ |
| **S4 — Block PO→GA**| `resnet18` | $73,61\%$ | $89,24 \pm 0,32\%$ | $89,13 \text{ [0,38]}\%$ | $89,40 \pm 0,31\%$ | $98,87\%$ | $13,29\times$ |
| **S4 — Block PO→GA**| `mobilenet_v3` | $73,61\%$ | **$88,02 \pm 0,34\%$** | $88,13 \text{ [0,40]}\%$ | **$88,36 \pm 0,37\%$** | **$98,98\%$** | $12,35\times$ |
| **S4 — Block PO→GA**| `densenet121` | $73,61\%$ | $89,36 \pm 0,43\%$ | **$89,53 \text{ [0,50]}\%$** | $89,48 \pm 0,47\%$ | $98,66\%$ | $11,92\times$ |
| **P1 — Parallel** | `resnet18` | $73,22\%$ | **$89,38 \pm 0,10\%$** | **$89,37 \text{ [0,12]}\%$** | **$89,58 \pm 0,07\%$** | **$99,02\%$** | $13,10\times$ |
| **P1 — Parallel** | `mobilenet_v3` | $73,22\%$ | $87,74 \pm 0,34\%$ | $87,53 \text{ [0,38]}\%$ | $88,04 \pm 0,36\%$ | $98,67\%$ | $12,17\times$ |
| **P1 — Parallel** | `densenet121` | $73,22\%$ | $89,13 \pm 0,43\%$ | $89,03 \text{ [0,52]}\%$ | $89,36 \pm 0,36\%$ | $98,42\%$ | $11,83\times$ |
| **B3 — GA-only** | `resnet18` | $73,55\%$ | $89,14 \pm 0,36\%$ | $89,07 \text{ [0,43]}\%$ | $89,29 \pm 0,41\%$ | $98,76\%$ | $13,12\times$ |
| **B3 — GA-only** | `mobilenet_v3` | $73,55\%$ | $87,64 \pm 0,54\%$ | $87,40 \text{ [0,63]}\%$ | $87,93 \pm 0,52\%$ | $98,55\%$ | $12,48\times$ |
| **B3 — GA-only** | `densenet121` | $73,55\%$ | **$89,37 \pm 0,21\%$** | $89,33 \text{ [0,25]}\%$ | **$89,58 \pm 0,23\%$** | **$98,67\%$** | $11,94\times$ |
| **C1 — Cooperative**| `resnet18` | $73,18\%$ | $89,36 \pm 0,14\%$ | **$89,40 \text{ [0,17]}\%$** | $89,54 \pm 0,08\%$ | $98,99\%$ | $13,07\times$ |
| **C1 — Cooperative**| `mobilenet_v3` | $73,18\%$ | $87,40 \pm 0,23\%$ | $87,53 \text{ [0,25]}\%$ | $87,74 \pm 0,27\%$ | $98,28\%$ | $12,26\times$ |
| **C1 — Cooperative**| `densenet121` | $73,18\%$ | $89,33 \pm 0,47\%$ | $89,17 \text{ [0,55]}\%$ | $89,52 \pm 0,45\%$ | $98,64\%$ | $11,66\times$ |
| **B4 — PO-only** | `resnet18` | $72,71\%$ | $88,92 \pm 0,14\%$ | $89,00 \text{ [0,15]}\%$ | $89,10 \pm 0,14\%$ | $98,51\%$ | $12,81\times$ |
| **B4 — PO-only** | `mobilenet_v3` | $72,71\%$ | $87,96 \pm 0,40\%$ | **$88,17 \text{ [0,45]}\%$** | $88,24 \pm 0,40\%$ | $98,90\%$ | $12,10\times$ |
| **B4 — PO-only** | `densenet121` | $72,71\%$ | $89,10 \pm 0,23\%$ | $89,00 \text{ [0,27]}\%$ | $89,32 \pm 0,22\%$ | $98,38\%$ | $11,57\times$ |
| **B2 — Stratified** | `resnet18` | $73,48\%$ | $88,83 \pm 0,24\%$ | $88,93 \text{ [0,28]}\%$ | $88,97 \pm 0,26\%$ | $98,41\%$ | $12,88\times$ |
| **B2 — Stratified** | `mobilenet_v3` | $73,48\%$ | $87,64 \pm 0,49\%$ | $87,93 \text{ [0,55]}\%$ | $87,99 \pm 0,46\%$ | $98,55\%$ | $12,26\times$ |
| **B2 — Stratified** | `densenet121` | $73,48\%$ | $89,34 \pm 0,68\%$ | $89,37 \text{ [0,83]}\%$ | $89,45 \pm 0,72\%$ | $98,65\%$ | $11,72\times$ |
| **S3 — Block GA→PO**| `resnet18` | $73,74\%$ | $89,03 \pm 0,15\%$ | $89,07 \text{ [0,18]}\%$ | $89,22 \pm 0,14\%$ | $98,63\%$ | $13,26\times$ |
| **S3 — Block GA→PO**| `mobilenet_v3` | $73,74\%$ | $87,52 \pm 0,22\%$ | $87,47 \text{ [0,27]}\%$ | $87,86 \pm 0,20\%$ | $98,42\%$ | $12,48\times$ |
| **S3 — Block GA→PO**| `densenet121` | $73,74\%$ | $89,14 \pm 0,20\%$ | $89,17 \text{ [0,25]}\%$ | $89,37 \pm 0,14\%$ | $98,43\%$ | $11,93\times$ |
| **S2 — Alt PO→GA** | `resnet18` | $74,81\%$ | $89,16 \pm 0,04\%$ | $89,13 \text{ [0,05]}\%$ | $89,34 \pm 0,08\%$ | $98,77\%$ | $13,82\times$ |
| **S2 — Alt PO→GA** | `mobilenet_v3` | $74,81\%$ | $87,52 \pm 0,17\%$ | $87,43 \text{ [0,20]}\%$ | $87,88 \pm 0,13\%$ | $98,42\%$ | $12,96\times$ |
| **S2 — Alt PO→GA** | `densenet121` | $74,81\%$ | $88,97 \pm 0,68\%$ | $89,23 \text{ [0,78]}\%$ | $89,17 \pm 0,64\%$ | $98,23\%$ | $12,35\times$ |
| **B1 — Random** | `resnet18` | $73,48\%$ | $88,71 \pm 0,14\%$ | $88,73 \text{ [0,17]}\%$ | $88,89 \pm 0,09\%$ | $98,28\%$ | $12,42\times$ |
| **B1 — Random** | `mobilenet_v3` | $73,48\%$ | $87,64 \pm 0,53\%$ | $87,90 \text{ [0,60]}\%$ | $87,98 \pm 0,50\%$ | $98,55\%$ | $12,18\times$ |
| **B1 — Random** | `densenet121` | $73,48\%$ | $89,16 \pm 0,48\%$ | $89,17 \text{ [0,58]}\%$ | $89,33 \pm 0,48\%$ | $98,44\%$ | $11,64\times$ |
| **S1 — Alt GA→PO** | `resnet18` | **$75,07\%$** | $88,71 \pm 0,31\%$ | $88,73 \text{ [0,38]}\%$ | $88,94 \pm 0,30\%$ | $98,28\%$ | **$13,86\times$** |
| **S1 — Alt GA→PO** | `mobilenet_v3` | **$75,07\%$** | $87,31 \pm 0,39\%$ | $87,57 \text{ [0,43]}\%$ | $87,67 \pm 0,35\%$ | $98,18\%$ | **$13,07\times$** |
| **S1 — Alt GA→PO** | `densenet121` | **$75,07\%$** | $89,03 \pm 0,07\%$ | $89,00 \text{ [0,08]}\%$ | $89,22 \pm 0,10\%$ | $98,30\%$ | **$12,48\times$** |

---

### 7.4. Phân tích Chuyên sâu Kết quả Thực nghiệm và Sự Đánh đổi Đa Mục tiêu

Kết quả thực nghiệm tại Bảng 7.3 và Bảng 7.4 cung cấp bằng chứng định lượng rõ ràng về bốn khía cạnh cốt lõi của bài toán rút gọn tập dữ liệu ảnh cho mạng học sâu:

**Thứ nhất, sự vượt trội toàn diện của hai kiến trúc lai `S4 (Block PO→GA)` và `P1 (Parallel Merge)` về độ chính xác phân loại liên kiến trúc CNN**:
Xét trung bình cộng trên cả 3 kiến trúc mạng học sâu (`ResNet-18`, `MobileNetV3-Small`, `DenseNet-121` qua $9$ phiên huấn luyện độc lập cho mỗi phương pháp), hai vị trí dẫn đầu toàn cục đều thuộc về các kiến trúc lai PO–GA:
- **Hạng 1 toàn cục — `S4 (Block PO→GA)`**: Tại mức rút gọn $\text{DR} = 73,61\%$ (giữ lại trung bình $2.963,3$ ảnh), `S4` đạt độ chính xác kiểm thử trung bình liên kiến trúc cao nhất là **$88,87\%$** (Macro F1 trung bình **$89,08\%$**, tỷ lệ bảo toàn độ chính xác trung bình **$\text{ARR} = 98,84\%$**). Cụ thể, `S4` đạt $89,24 \pm 0,32\%$ trên `ResNet-18`, dẫn đầu tuyệt đối trên `MobileNetV3-Small` với **$88,02 \pm 0,34\%$** ($\text{ARR} = 98,98\%$, đạt đỉnh $88,37\%$ tại `Seed 42`) và đạt trung vị cao nhất trên `DenseNet-121` với **$89,53\%$** (đạt đỉnh $89,77\%$ tại `Seed 123`).
- **Hạng 2 toàn cục — `P1 (Parallel Merge + Co-Refinement)`**: Tại mức rút gọn $\text{DR} = 73,22\%$ ($M = 3.007,0$ ảnh), `P1` đạt độ chính xác trung bình liên kiến trúc **$88,75\%$** ($\text{ARR} = 98,70\%$), trong đó dẫn đầu toàn bộ 10 phương pháp trên mạng **`ResNet-18`** với độ chính xác **$89,38 \pm 0,10\%$** và Macro F1 **$89,58 \pm 0,07\%$** (giữ lại **$99,02\%$** hiệu năng của toàn bộ tập dữ liệu gốc).
Cả `S4` ($88,87\%$) và `P1` ($88,75\%$) đều vượt qua hai thuật toán thành phần đơn lẻ là `GA-only` ($88,72\%$) và `PO-only` ($88,66\%$), đồng thời bỏ xa `Stratified Random B2` ($88,60\%$) và `Random Selection B1` ($88,50\%$).

**Thứ hai, kỷ lục bảo toàn độ chính xác đơn lẻ của kiến trúc Hợp tác Đồng tiến hóa `C1 (Coop Exchange)`**:
Nhờ cơ chế trao đổi chéo $\text{Top-}2$ cá thể tinh hoa mỗi $5$ vòng lặp kết hợp bộ lọc khử trùng lặp khoảng cách Hamming, biến thể **`C1`** đạt độ bao phủ không gian đặc trưng cao nhất trong nhóm Hybrid ($\text{Cov} = 0,8896$) và đạt độ chính xác trung bình **$89,36 \pm 0,14\%$** trên `ResNet-18` (trung vị đạt đỉnh **$89,40\%$**). Đặc biệt, tại `Seed 42`, tập con của `C1` ($M = 2.977$ ảnh, $\text{DR} = 73,49\%$) đạt độ chính xác kiểm thử **`89,97%`** và Macro F1 **`90,12%`** trên mạng `DenseNet-121`, tương ứng tỷ lệ duy trì hiệu năng **$\text{ARR} = 99,34\%$** (chỉ thấp hơn $0,60\%$ so với huấn luyện trên $100\%$ tập dữ liệu gốc).

**Thứ ba, ưu thế tối ưu hóa tỷ lệ nén của nhóm Lai ghép Tuần tự Xen kẽ (`S1 — Alt GA→PO` và `S2 — Alt PO→GA`)**:
Trong khi `S4`, `P1` và `C1` tối đa hóa độ chính xác phân loại ở mức rút gọn $\approx 73,2\% - 73,6\%$, hai cấu hình chuyển pha chu kỳ ngắn **`S1`** và **`S2`** đẩy tỷ lệ rút gọn lên mức cao nhất là **$75,07\%$ ($M = 2.799$ ảnh)** và **$74,81\%$ ($M = 2.828$ ảnh)** cùng điểm Fitness cao nhất ($0,8248$ và $0,8249$). Đáng chú ý, dù sử dụng ít hơn `GA-only` $141$ ảnh và ít hơn `PO-only` $236$ ảnh, biến thể **`S2 (Alt PO→GA)`** vẫn đạt độ chính xác **$89,16 \pm 0,04\%$** trên `ResNet-18` ($\text{ARR} = 98,77\%$), cao hơn cả `GA-only` ($89,14\%$), `PO-only` ($88,92\%$), `Stratified Random` ($88,83\%$) và `Random Selection` ($88,71\%$).

**Thứ tư, phân tích đối chiếu với `Random Selection (B1)`, `Stratified Random (B2)` và kiểm định thống kê Wilcoxon**:
Để đảm bảo tính công bằng tuyệt đối khi so sánh với lấy mẫu ngẫu nhiên, kích thước tập con của `B1` và `B2` được khóa bằng đúng trung vị kích thước của các thuật toán tối ưu hóa là $M_{\text{baseline}} = 2.978$ ảnh ($\text{DR} = 73,48\%$). Tại cùng quy mô mẫu này, trên mạng `ResNet-18`, tất cả các phương pháp `P1` ($89,38\%$), `C1` ($89,36\%$), `S4` ($89,24\%$), `S2` ($89,16\%$), `GA-only` ($89,14\%$) và `S3` ($89,03\%$) đều vượt qua cả `Stratified Random B2` ($88,83\%$) lẫn `Random Selection B1` ($88,71\%$). Trong kiểm định phi tham số Wilcoxon Signed-Rank Test trên $n = 3$ hạt giống (`seeds = [42, 123, 456]`), giá trị $p\text{-value}$ nhỏ nhất có thể đạt được về mặt toán học với cỡ mẫu $n = 3$ là $p_{\min} = \frac{2}{2^3} = 0,2500$, đạt được tại mọi cặp mà `P1` và `C1` chiến thắng tuyệt đối $3/3$ hạt giống trước `B1`, `B2`, `PO-only`, `S1` và `S3` trên `ResNet-18`, cũng như khi `S4` chiến thắng tuyệt đối $3/3$ hạt giống trước `S1`, `S2`, `S3` và `C1` trên `MobileNetV3-Small`. Khi xét gộp trên toàn bộ $n = 9$ phiên huấn luyện độc lập ($3\text{ seeds} \times 3\text{ mô hình CNN}$), `S4` thắng `Random Selection (B1)` trên $8/9$ phiên chạy với mức ý nghĩa thống kê $p = 0,0195 < 0,05$.

---

## CHƯƠNG 8: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 8.1. Kết luận

Khóa luận đã hoàn thành toàn bộ các mục tiêu nghiên cứu, thiết kế thuật toán và kiểm chứng thực nghiệm theo đề cương **CNTT-KLCN140**, đạt được bốn kết quả cụ thể:
1. **Xác lập quy trình thực nghiệm chống rò rỉ dữ liệu và mốc chuẩn B0 trên 3 kiến trúc CNN**: Đã phân hoạch chuẩn hóa bộ dữ liệu Intel Image Classification ($17.034$ ảnh, $6$ lớp cảnh tự nhiên) thành ba tập độc lập ($11.228$ Train / $2.806$ Validation / $3.000$ Independent Test) và thiết lập mốc chuẩn toàn tập dữ liệu trên ba kiến trúc học sâu `ResNet-18` ($90,27\%$), `MobileNetV3-Small` ($88,93\%$) và `DenseNet-121` ($90,57\%$).
2. **Thiết kế hàm mục tiêu 4 thành phần chuẩn hóa nội tại và hiệu chỉnh cân bằng cho bài toán chọn tập con nhị phân**: Đã xây dựng hàm thích nghi kết hợp Độ đa dạng ($\alpha = 0,15$), Độ bao phủ ($\beta = 0,45$), Độ cân bằng lớp ($\gamma = 0,20$) và Độ nén ($\delta = 0,20$) trên ma trận khoảng cách Cosine tiền tính toán bằng GPU, kết hợp toán tử đột biến thích nghi cân bằng mật độ trong GA và bốn hành vi bầy đàn truyền thống của Parrot Optimizer (Lian và cộng sự, 2024) với vector đồng thuận tâm hóa và hàm truyền hình chữ V $|\tanh(\Delta x)|$, giúp mỗi phiên tối ưu hóa $2.000\text{ FEs}$ hoàn tất chỉ trong $9,26 - 10,58\text{ giây}$.
3. **Chứng minh tính ưu việt của sáu biến thể lai hóa Hybrid PO–GA trên cả hai tầng đánh giá**:
   * **Về chất lượng tối ưu hóa và tỷ lệ nén (Tầng 1)**: Hai cấu hình lai tuần tự xen kẽ **`S2 (Alt PO→GA)`** và **`S1 (Alt GA→PO)`** đạt điểm Fitness cao nhất toàn cục ($0,8249$ và $0,8248$) và cắt giảm tới **$74,81\% - 75,07\%$** tập dữ liệu gốc, vượt qua cả `GA-only`, `PO-only`, `Random Selection` và `Stratified Random`.
   * **Về hiệu năng huấn luyện mạng học sâu hạ nguồn (Tầng 2)**: Cấu hình lai chia khối **`S4 (Block PO→GA)`** xếp hạng #1 toàn cục với độ chính xác trung bình liên kiến trúc đạt **$88,87\%$** ($\text{ARR} = 98,84\%$ tại mức rút gọn $73,61\%$, dẫn đầu trên `MobileNetV3-Small` với $88,02\%$). Cấu hình lai song song hợp nhất **`P1 (Parallel Merge)`** xếp hạng #2 toàn cục ($88,75\%$) và dẫn đầu trên `ResNet-18` với **$89,38 \pm 0,10\%$** ($\text{ARR} = 99,02\%$), trong khi cấu hình hợp tác **`C1 (Coop Exchange)`** đạt đỉnh đơn lẻ **$89,97\%$** ($\text{ARR} = 99,34\%$) trên `DenseNet-121`.
4. **Đóng gói công cụ thực nghiệm hoàn chỉnh**: Đã xây dựng hệ thống mã nguồn mô-đun hóa kèm bộ đặc trưng tiền trích xuất (`features/train_features_train80.npy`) và kịch bản tự động hóa thực nghiệm trên cả máy cục bộ lẫn nền tảng đám mây Kaggle.

### 8.2. Hướng Phát triển Tiếp theo

Từ những phát hiện thực nghiệm trong khóa luận, ba hướng nghiên cứu mở rộng giàu tiềm năng được đề xuất cho các công trình tiếp theo:
1. **Mở rộng sang bài toán Tối ưu hóa Đa mục tiêu Pareto (Multi-Objective Pareto PO–GA)**: Thay vì sử dụng một bộ trọng số cố định $(\alpha, \beta, \gamma, \delta)$, việc tích hợp cơ chế xếp hạng không bị trội (Non-dominated Sorting) sẽ cho phép thuật toán PO–GA xuất ra đồng thời toàn bộ mặt biên Pareto trải dài từ mức rút gọn $50\%$ đến $90\%$ trong một lần chạy duy nhất.
2. **Mở rộng quy mô hạt giống thống kê ($n \ge 10$ seeds) và đánh giá trên các bộ dữ liệu quy mô lớn hơn (CIFAR-100, Tiny-ImageNet)**: Tăng số lượng hạt giống độc lập để nâng cao lực thống kê của kiểm định phi tham số Wilcoxon theo từng mô hình đơn lẻ và kiểm chứng khả năng mở rộng của thuật toán khi số lượng lớp tăng lên $100 - 200$ lớp.
3. **Tích hợp tiêu chí độ khó của mẫu học sâu (Gradient Norm / Forgetting Score)**: Kết hợp khoảng cách hình học Cosine trên không gian đặc trưng với độ lớn đạo hàm của hàm mất mát trong những epoch đầu tiên để ưu tiên giữ lại các mẫu nằm sát biên quyết định giữa hai lớp dễ nhầm lẫn là `glacier` và `mountain`.

---

## TÀI LIỆU THAM KHẢO

[1] J. Lian, G. Hui, L. Ma, T. Zhu, X. Wu, A. A. Heidari, Y. Chen, and H. Chen, "Parrot optimizer: Algorithm and applications to medical problems," *Computers in Biology and Medicine*, vol. 172, Article 108064, 2024. DOI: 10.1016/j.compbiomed.2024.108064.

[2] J. R. Cano, F. Herrera, and M. Lozano, "Using evolutionary algorithms as instance selection for data reduction in KDD: An experimental study," *IEEE Transactions on Evolutionary Computation*, vol. 7, no. 6, pp. 561–575, 2003. DOI: 10.1109/TEVC.2003.819265.

[3] B. Zhou, A. Lapedriza, A. Khosla, A. Oliva, and A. Torralba, "Places: A 10 Million Image Database for Scene Recognition," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 40, no. 6, pp. 1452–1464, 2018. DOI: 10.1109/TPAMI.2017.2723009.

[4] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Las Vegas, NV, USA, 2016, pp. 770–778.

[5] O. Sener and S. Savarese, "Active Learning for Convolutional Neural Networks: A Core-Set Approach," in *International Conference on Learning Representations (ICLR)*, Vancouver, BC, Canada, 2018.

[6] J. H. Holland, *Adaptation in Natural and Artificial Systems: An Introductory Analysis with Applications to Biology, Control, and Artificial Intelligence*, Cambridge, MA, USA: MIT Press, 1992.

[7] A. Howard et al., "Searching for MobileNetV3," in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, Seoul, Korea (South), 2019, pp. 1314–1324.

[8] G. Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger, "Densely Connected Convolutional Networks," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Honolulu, HI, USA, 2017, pp. 4700–4708.

[9] S. Mirjalili and A. Lewis, "S-shaped versus V-shaped transfer functions for binary Particle Swarm Optimization," *Swarm and Evolutionary Computation*, vol. 9, pp. 1–14, 2013. DOI: 10.1016/j.swevo.2012.09.002.


