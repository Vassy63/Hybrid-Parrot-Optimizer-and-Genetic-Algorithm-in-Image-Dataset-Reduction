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
với bộ trọng số chuẩn hóa $(\alpha, \beta, \gamma, \delta) = (0,10;\; 0,30;\; 0,20;\; 0,40)$ thỏa mãn $\alpha + \beta + \gamma + \delta = 1$.

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
  * Điểm thích nghi tổng hợp: $F(X^{(A)}) = 0,1(0,1000) + 0,3(0,5533) + 0,2(0,5000) + 0,4(0,6667) = \mathbf{0,5427}$.

* **Cá thể $X^{(B)} = (1, 0, 0, 0, 1, 0)$**: Chọn 1 ảnh Lớp 0 ($I_1$) và 1 ảnh Lớp 1 ($I_5$):
  * $\text{Div}(X^{(B)}) = \hat{d}_{1,5} = 0,9000$ (hai ảnh đại diện cho hai cảnh quan khác biệt rõ rệt).
  * Khoảng cách nhỏ nhất từ 6 ảnh tới $\{I_1, I_5\}$ lần lượt là: $[0,00;\; 0,10;\; 0,20;\; 0,12;\; 0,00;\; 0,10]$, trung bình chỉ bằng $0,0867 \Rightarrow \text{Cov}(X^{(B)}) = 1 - 0,0867 = 0,9133$ (bao phủ tốt cả hai cụm lớp).
  * Phân bố lớp tập con: $M_0/M = 0,50$, $M_1/M = 0,50 \Rightarrow \text{Bal}(X^{(B)}) = 1 - 0 = 1,0000$ (cân bằng tuyệt đối).
  * $\text{Com}(X^{(B)}) = 1 - \frac{2}{6} = 0,6667$.
  * Điểm thích nghi tổng hợp: $F(X^{(B)}) = 0,1(0,9000) + 0,3(0,9133) + 0,2(1,0000) + 0,4(0,6667) = \mathbf{0,8307}$.

Ví dụ trên chứng minh hàm mục tiêu đánh giá chính xác chất lượng đại diện: cùng giữ lại 2 bức ảnh nhưng cá thể $X^{(B)}$ có độ bao phủ, đa dạng và cân bằng lớp vượt trội được chấm điểm $0,8307$, cao hơn hẳn cá thể lệch lớp $X^{(A)}$ ($0,5427$).

---

## CHƯƠNG 4: THUẬT TOÁN BIẾN ĐỔI NHỊ PHÂN VÀ BA KIẾN TRÚC LAI HÓA PO–GA

### 4.1. Giải thuật Di truyền cho Lựa chọn Mẫu Nhị phân (Binary Genetic Algorithm)

Giải thuật Di truyền (GA) vận hành trực tiếp trên quần thể các vector nhị phân $\{X_1, X_2, \dots, X_P\}$. Tại mỗi thế hệ $t \in \{1, \dots, T\}$, GA thực hiện tuần tự bốn bước:
1. **Bảo tồn tinh hoa (Elitism)**: Sắp xếp quần thể theo chiều giảm dần của hàm thích nghi $F(X)$ và sao chép nguyên trạng $E$ cá thể đứng đầu sang quần thể thế hệ kế tiếp.
2. **Chọn lọc giải đấu (Tournament Selection)**: Để chọn mỗi cá thể cha mẹ, lấy ngẫu nhiên $k_{\text{tour}} = 4$ cá thể từ quần thể hiện tại và chọn cá thể có điểm thích nghi cao nhất.
3. **Lai ghép đồng nhất (Uniform Crossover)**: Với xác suất lai ghép $p_c = 0,8$, hai cá thể cha mẹ $P_1, P_2$ trao đổi từng bit độc lập theo mặt nạ nhị phân ngẫu nhiên $M_{\text{mask}} \in \{0, 1\}^N$ ($\Pr(M_{\text{mask}, j} = 1) = 0,5$) để tạo ra hai cá thể con $C_1, C_2$.
4. **Đột biến thích nghi giảm dần (Adaptive Bit-Flip Mutation)**: Tỷ lệ đột biến $p_m(t)$ giảm tuyến tính theo số thế hệ để ưu tiên thăm dò ở giai đoạn đầu và ổn định hội tụ ở giai đoạn cuối:
   $$p_m(t) = \max\left(p_{\min},\; p_{\max} - (p_{\max} - p_{\min}) \cdot \frac{t}{T}\right)$$
   với $p_{\max} = 0,03$ và $p_{\min} = 0,001$. Sau đột biến, cá thể được đưa qua toán tử sửa chữa $\text{Repair}(X, M_{\min}, M_{\max})$.

### 4.2. Thuật toán Tối ưu hóa Đàn vẹt Truyền thống (Parrot Optimizer - PO) và Bước Ánh xạ Nhị phân

Thuật toán Parrot Optimizer (PO) truyền thống của Lian và cộng sự (2024) [1] mô phỏng bốn hành vi tự nhiên của loài vẹt *Pyrrhura molinae* để cập nhật vị trí cá thể trên không gian liên tục $\mathbb{R}^N$. Để áp dụng trực tiếp PO truyền thống vào bài toán chọn mẫu ảnh $X \in \{0, 1\}^N$ (trong đó chỉ có hai trạng thái: $1$ là chọn ảnh, $0$ là loại bỏ ảnh), hệ thống giữ nguyên hoàn toàn bốn phương trình cập nhật vị trí của PO truyền thống ở Bước 1, sau đó thực hiện bước ánh xạ về $\{0, 1\}$ ở Bước 2:

**Bước 1: Cập nhật vị trí liên tục theo đúng 4 hành vi của PO truyền thống [1]**  
Ở vòng lặp thứ $t$ (với tổng số vòng lặp tối đa $T_{\max}$), mỗi cá thể vẹt $X_i^t$ chọn ngẫu nhiên một chỉ số hành vi $S_t \in \{1, 2, 3, 4\}$ với xác suất đều nhau ($25\%$ cho mỗi hành vi):

* **Hành vi 1 ($S_t = 1$) — Kiếm ăn (Foraging Behavior)**: Vẹt di chuyển hướng về nguồn thức ăn dựa trên vị trí tốt nhất toàn cục $X_{\text{gbest}}$ và vị trí trung bình của cả đàn $X_{\text{mean}}^t = \frac{1}{P}\sum_{k=1}^P X_k^t$:
  $$X_i^{\text{cont}} = (X_i^t - X_{\text{gbest}}) \odot \text{Levy}(N) + \text{rand}(0,1) \cdot X_{\text{mean}}^t \cdot \left(1 - \frac{t}{T_{\max}}\right)^{\frac{2t}{T_{\max}}}$$

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

**Bước 2: Ánh xạ vị trí liên tục của PO sang trạng thái chọn ảnh nhị phân (0 hoặc 1)**  
Vì vị trí $X_i^{\text{cont}}$ sau Bước 1 là số thực (ví dụ: $0,73$ hoặc $-1,25$), trong khi mỗi bức ảnh chỉ có thể được chọn ($1$) hoặc không chọn ($0$), ta cần quy đổi mức độ thay đổi vị trí $\Delta x_{i,d} = x_{i,d}^{\text{cont}} - x_{i,d}^t$ thành xác suất thuộc khoảng $[0, 1)$ bằng hàm $|\tanh(\Delta x_{i,d})|$ [9]:

```text
PO truyền thống (X_cont) 
   ──► Độ thay đổi vị trí (ΔX = X_cont - X_current) 
   ──► Quy đổi sang xác suất [0, 1]: P_flip = |tanh(ΔX)| 
   ──► Nếu rand() < P_flip thì đảo bit (0 ↔ 1), ngược lại giữ nguyên
```

Cụ thể, trạng thái chọn/bỏ của bức ảnh thứ $d$ được quyết định như sau:

$$x_{i,d}^{t+1} = \begin{cases} 
1 - x_{i,d}^t, & \text{nếu } \text{rand}(0,1) < |\tanh(\Delta x_{i,d})| \\
x_{i,d}^t, & \text{ngược lại}
\end{cases}$$

Ý nghĩa trực quan rất đơn giản: nếu ở vòng lặp đó con vẹt hầu như đứng yên ($\Delta x_{i,d} \approx 0 \Rightarrow |\tanh(\Delta x_{i,d})| \approx 0$), bức ảnh giữ nguyên trạng thái cũ; nếu con vẹt nhảy một bước lớn ($\Delta x_{i,d}$ lớn $\Rightarrow |\tanh(\Delta x_{i,d})| \to 1$), trạng thái bức ảnh sẽ được lật ($0 \to 1$ hoặc $1 \to 0$). Cuối cùng, vị trí mới chỉ được cập nhật nếu có điểm Fitness cao hơn vị trí cũ ($F(X_i^{t+1}) > F(X_i^t)$).

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
GA↔PO  PO↔GA  GA→PO  PO→GA     ──► Merge & Top-P     Trao đổi Top-k (Chống trùng)
```

1. **Kiến trúc Tuần tự (Sequential Hybrid: S1, S2, S3, S4)**:
   * **S1 (Alternating Sequential, GA-first)**: Chạy xen kẽ từng pha ngắn có độ dài cố định $p = 10$ vòng lặp theo chu trình $\text{GA}(10) \rightarrow \text{PO}(10) \rightarrow \text{GA}(10) \rightarrow \text{PO}(10) \dots$ cho đến khi đạt tổng số vòng lặp $T$.
   * **S2 (Alternating Sequential, PO-first)**: Đảo ngược thứ tự khởi đầu thành $\text{PO}(10) \rightarrow \text{GA}(10) \rightarrow \text{PO}(10) \rightarrow \text{GA}(10) \dots$ để kiểm tra hiệu ứng thứ tự (Ordering Effect) giữa khai phá và khai thác.
   * **S3 (Block Sequential, GA-first)**: Chia tổng ngân sách thành hai khối liên tục 50/50: chạy $\text{GA}(50\%)$ để tạo quần thể đa dạng, sau đó chuyển giao toàn bộ quần thể cho $\text{PO}(50\%)$ tinh chỉnh.
   * **S4 (Block Sequential, PO-first)**: Chạy $\text{PO}(50\%)$ ở nửa đầu, sau đó chuyển giao quần thể cho $\text{GA}(50\%)$ tái tổ hợp ở nửa sau.

2. **Kiến trúc Song song Độc lập (Parallel Hybrid: P1 — Independent Parallel + Merge)**:
   * Từ cùng một quần thể khởi tạo $P_0$ kích thước $P$, hệ thống tách thành hai nhánh độc lập chạy $50\%$ ngân sách mỗi nhánh: nhánh GA tiến hóa tạo ra $\text{Pop}_{\text{GA}}$ và nhánh PO tiến hóa tạo ra $\text{Pop}_{\text{PO}}$.
   * Khi kết thúc, hai quần thể được hợp nhất và chọn lọc tinh hoa dưới cùng hàm mục tiêu $F$:
     $$\text{Pop}_{\text{merged}} = \text{Pop}_{\text{GA}} \cup \text{Pop}_{\text{PO}}, \quad \text{Pop}_{\text{final}} = \text{Top-}P(\text{Pop}_{\text{merged}}, F)$$

3. **Kiến trúc Hợp tác Hai chiều (Cooperative Hybrid: C1 — Bidirectional Elite Exchange with Duplicate Control)**:
   * Hai quần thể $\text{Pop}_{\text{GA}}$ và $\text{Pop}_{\text{PO}}$ cùng tiến hóa song song. Sau mỗi chu kỳ $\Delta t$ vòng lặp, hai nhánh trao đổi $k = 3$ cá thể ưu tú nhất (Top-$k$ Elites) để thay thế $k$ cá thể kém nhất của nhánh đối diện.
   * **Cơ chế kiểm soát trùng lặp (Duplicate Control)**: Để ngăn hiện tượng hai quần thể bị đồng hóa sớm do trao đổi lặp lại cùng một nghiệm cực đại, trước khi đưa cá thể ưu tú $e$ vào quần thể đích $\text{Pop}_{\text{target}}$, hệ thống kiểm tra điều kiện khoảng cách Hamming $\min_{Y \in \text{Pop}_{\text{target}}} \|e - Y\|_1 > 0$. Chỉ những cá thể ưu tú chưa tồn tại trong quần thể đích mới được phép di cư.

---

## CHƯƠNG 5: THIẾT KẾ THỰC NGHIỆM VÀ PHƯƠNG PHÁP LUẬN ĐÁNH GIÁ

### 5.1. Hệ thống Phương pháp Đối sánh (Baselines và Hybrid Configurations)

Để trả lời đầy đủ các câu hỏi nghiên cứu và đáp ứng yêu cầu đối sánh của đề cương CNTT-KLCN140, bộ thực nghiệm chính thức bao gồm **11 phương pháp** chia thành hai nhóm:
* **Nhóm Đối chuẩn (Baselines: B0 – B4)**:
  * **B0 — Full Dataset (Upper Bound)**: Sử dụng $100\%$ tập dữ liệu huấn luyện gốc (không rút gọn).
  * **B1 — Random Selection**: Lấy mẫu ngẫu nhiên phân bố đều theo đúng kích thước tập con mục tiêu $M$.
  * **B2 — Stratified Random Selection**: Lấy mẫu ngẫu nhiên phân tầng, bảo toàn chính xác tỷ lệ phân bố của $6$ lớp cảnh tự nhiên.
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
| **P1** | Parallel (Độc lập) | $\text{GA}(50) \parallel \text{PO}(50) \rightarrow \text{Merge}$ | $1.000$ | $1.000$ | **$2.000$** |
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

Thực nghiệm tối ưu hóa được thực hiện trên tập đặc trưng huấn luyện $D_{\text{train}}$ ($N = 11.228$ ảnh, vector ResNet-18 $512$ chiều) với hàm mục tiêu bốn thành phần nguyên bản $F(X) = 0,10\,\text{Div}(X) + 0,30\,\text{Cov}(X) + 0,20\,\text{Bal}(X) + 0,40\,\text{Com}(X)$ dưới cùng ngân sách đánh giá $2.000\text{ FEs}$ ($P = 20$ cá thể, $T = 100$ vòng lặp) lặp lại qua 3 hạt giống độc lập (`seeds = [42, 123, 456]`). Bảng 7.3 tổng hợp kết quả thống kê mô tả ($\text{Mean} \pm \text{Std}$ và $\text{Median} \text{ [IQR]}$), giá trị trung bình của bốn thành phần mục tiêu, kích thước tập con hội tụ $M$, tỷ lệ rút gọn dữ liệu ($\text{DR}\%$) và thời gian thực thi tối ưu hóa từ phiên chạy thực nghiệm [`outputs/run_poga_all_20261001_235502`](file:///d:/Code/KhoaLuanDatasetReduction/outputs/run_poga_all_20261001_235502/descriptive_statistics.csv).

**Bảng 7.3: Đối sánh Chất lượng Tối ưu hóa (Tầng 1) giữa 10 phương pháp dưới ngân sách 2.000 FEs (`run_poga_all_20261001_235502`)**

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

**Phân tích cơ chế hội tụ và động lực học tối ưu hóa (Tầng 1)**:
Số liệu từ Bảng 7.3 phản ánh sự phân hóa rõ rệt về hành vi tìm kiếm trên không gian nhị phân $N = 11.228$ chiều giữa giải thuật di truyền thuần túy (`B3 — GA-only`) và nhóm các thuật toán tích hợp động lực học Đàn vẹt (`B4 — PO-only` và sáu cấu hình lai `S1–S4, P1, C1`). Cụ thể, cấu hình lai theo khối **`S3 (Block GA→PO)`** đạt điểm thích nghi trung bình cao nhất toàn cục là $0,8875 \pm 0,0007$ (với độ đa dạng $\text{Div} = 0,5144$), theo sát bởi cấu hình hợp tác đồng tiến hóa **`C1 (Coop Exchange)`** đạt $0,8870 \pm 0,0004$ (với tỷ lệ nén cao nhất $\text{Com} = 0,9818$, tương ứng giữ lại trung bình $204,0$ mẫu ảnh). Trong khi đó, `GA-only` dừng lại ở mức điểm thích nghi $0,8026 \pm 0,0012$ với kích thước tập con trung bình $M = 3.229,0$ ảnh ($\text{DR} = 71,24\%$).

Nguyên nhân toán học của sự khác biệt này bắt nguồn từ tương tác giữa toán tử dịch chuyển liên tục của thuật toán PO truyền thống, bước ánh xạ nhị phân ($|\tanh(\Delta x)|$) và trọng số nén $\delta = 0,40$. Trong giải thuật `GA-only`, toán tử đột biến lật từng bit độc lập với xác suất đồng nhất $p_m(t)$. Khi quần thể được khởi tạo ở mật độ $\text{init\_ratio} = 0,30$ ($30\%$ bit $1$ và $70\%$ bit $0$), số lượng bit $0$ bị lật sang $1$ nhiều gấp $2,33$ lần số lượng bit $1$ bị lật sang $0$, tạo ra lực cân bằng giữ kích thước tập con của GA ổn định quanh ngưỡng $28,76\%$ tập dữ liệu gốc ($M \approx 3.229$ ảnh, đạt độ bao phủ $\text{Cov} = 0,8926$). Ngược lại, trong thuật toán PO truyền thống, phương trình hành vi giao tiếp ($S_t = 3$) cập nhật độ dịch chuyển tỷ lệ với $(X_i^t - X_{\text{mean}}^t)$. Khi mật độ trung bình của đàn $X_{\text{mean}}^t \approx 0,30$, các chiều đang có trạng thái $x_{i,d}^t = 1$ chịu biên độ dịch chuyển $|1 - 0,30| = 0,70$, sinh ra xác suất lật bit $1 \rightarrow 0$ qua hàm $|\tanh(\Delta x)|$ cao gấp hơn hai lần so với xác suất lật bit $0 \rightarrow 1$ tại các chiều có $x_{i,d}^t = 0$ ($|0 - 0,30| = 0,30$). Khi số lượng mẫu $M$ giảm từ $3.229$ ảnh xuống khoảng $220$ ảnh, thành phần độ nén $\delta \cdot \text{Com}(X)$ tăng thêm $+0,40 \times (0,9802 - 0,7124) = +0,1071$, vượt xa mức suy giảm của thành phần độ bao phủ $\beta \cdot \text{Cov}(X)$ là $-0,30 \times (0,8926 - 0,8132) = -0,0238$. Do đó, cơ chế chọn lọc tham lam (Greedy Selection) của PO và các kiến trúc lai PO–GA lập tức chấp nhận các nghiệm có độ nén cao ($\text{DR} \in [97,57\%, 98,18\%]$). Đồng thời, nhờ kích thước tập con $M$ nhỏ hơn, chi phí tính toán ma trận con giảm mạnh giúp thời gian thực thi của **`S2 (Alt PO→GA)`** chỉ mất $105,81\text{ s}$ và **`C1 (Coop Exchange)`** chỉ mất $113,64\text{ s}$, nhanh hơn gấp $3,19$ lần và $2,97$ lần so với `GA-only` ($337,48\text{ s}$).

---

### 7.3. Kết quả Đánh giá Tầng 2: Hiệu năng Huấn luyện Hạ nguồn trên Ba Kiến trúc CNN

Sau khi hoàn tất giai đoạn tối ưu hóa Tầng 1, các tập ảnh đại diện tốt nhất (`best_sol`) của từng phương pháp ứng với từng hạt giống (`seed` $\in \{42, 123, 456\}$) được trích xuất để huấn luyện độc lập ba kiến trúc mạng học sâu gồm **ResNet-18**, **MobileNetV3-Small** và **DenseNet-121** trong $5$ epochs ($\text{batch\_size} = 64$, $\text{lr} = 10^{-3}$). Toàn bộ kết quả đánh giá trên tập kiểm thử độc lập `seg_test` ($3.000$ ảnh) được trình bày chi tiết tại Bảng 7.4.

**Bảng 7.4: Đối sánh Hiệu năng Phân loại và Tăng tốc Huấn luyện (Tầng 2) trên 3 kiến trúc CNN (`run_poga_all_20261001_235502`)**

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

### 7.4. Phân tích Chuyên sâu Kết quả Thực nghiệm và Sự Đánh đổi Đa Mục tiêu

Kết quả thực nghiệm tại Bảng 7.3 và Bảng 7.4 cung cấp bằng chứng định lượng rõ ràng về ba khía cạnh cốt lõi của bài toán rút gọn tập dữ liệu ảnh cho mạng học sâu:

**Thứ nhất, hiệu quả bảo toàn độ chính xác ở chế độ rút gọn bảo toàn (`GA-only` — Rút gọn $71,24\%$ dữ liệu)**:
Tại mức rút gọn $71,24\%$ (giữ lại trung bình $3.229 / 11.228$ ảnh huấn luyện), giải thuật `GA-only` duy trì độ bao phủ không gian đặc trưng ở mức rất cao ($\text{Cov} = 0,8926$) và độ cân bằng lớp gần như tuyệt đối ($\text{Bal} = 0,9979$). Khi huấn luyện trên tập con này, mạng `ResNet-18` đạt độ chính xác kiểm thử trung bình $89,35 \pm 0,39\%$ (riêng tại `seed = 123` đạt $89,90\%$, chỉ thấp hơn $0,37\%$ so với toàn bộ tập dữ liệu gốc, tương ứng tỷ lệ duy trì hiệu năng $\text{ARR} = 99,59\%$). Tương tự, `MobileNetV3-Small` đạt $87,79 \pm 0,24\%$ ($\text{ARR} = 98,72\%$) và `DenseNet-121` đạt $88,99 \pm 0,09\%$ ($\text{ARR} = 98,25\%$), đồng thời rút ngắn thời gian huấn luyện từ $1,36$ đến $1,57$ lần. Kết quả này chứng minh rằng hơn $71\%$ mẫu ảnh trong tập huấn luyện gốc của bộ dữ liệu Intel Image Classification mang thông tin dư thừa và có thể được loại bỏ mà chỉ làm suy giảm dưới $1,3\%$ độ chính xác phân loại.

**Thứ hai, hiệu năng của các cấu hình Lai ghép Tuần tự Xen kẽ (`S2 — Alt PO→GA` và `S1 — Alt GA→PO`) ở chế độ nén cực đại ($\text{DR} = 97,57\% - 97,84\%$)**:
Trong nhóm các thuật toán có sự tham gia của PO, hai cấu hình lai ghép xen kẽ theo chu kỳ ngắn ($p = 10$ vòng lặp) là **`S2 (Alt PO→GA)`** và **`S1 (Alt GA→PO)`** cho thấy sự vượt trội rõ rệt so với cả `PO-only` lẫn các cấu hình chia khối cứng (`S3, S4`) và hợp tác nén sâu (`C1`). Cụ thể, ở mức rút gọn tới **$97,57\%$** (chỉ giữ lại trung bình **$272,7$ bức ảnh** trên tổng số $11.228$ ảnh, tức khoảng $45$ ảnh cho mỗi lớp cảnh tự nhiên — giảm quy mô dữ liệu đi $41,2$ lần), cấu hình **`S2 (Alt PO→GA)`** vẫn đạt độ chính xác trung bình **$79,81 \pm 0,67\%$** trên `MobileNetV3-Small` ($\text{ARR} = 89,75\%$) và **$79,10 \pm 2,57\%$** trên `DenseNet-121` ($\text{ARR} = 87,34\%$, đạt đỉnh **$82,63\%$** với $\text{ARR} = 91,23\%$ tại `seed = 123`). Tương tự, cấu hình **`S1 (Alt GA→PO)`** tại `seed = 123` chỉ sử dụng **$236$ bức ảnh** ($\text{DR} = 97,90\%$) nhưng đạt độ chính xác **$81,20\%$** trên `MobileNetV3-Small` ($\text{ARR} = 91,31\%$), đồng thời tăng tốc độ huấn luyện CNN từ $2,28\times$ đến $3,03\times$. Việc luân phiên liên tục từng pha $10$ vòng lặp giữa PO và GA giúp `S2` và `S1` vừa tận dụng được tốc độ hội tụ nhanh của PO, vừa sử dụng toán tử lai ghép đồng nhất của GA để ngăn quần thể suy giảm kích thước tập con xuống quá sâu như ở `C1` ($M = 204$ ảnh) hay `S3` ($M = 193$ ảnh tại `seed = 456`).

**Thứ ba, tính tổng quát hóa xuyên kiến trúc (Cross-Architecture Generalization) và độ nhạy theo dung lượng tham số mô hình**:
Mặc dù toàn bộ quá trình trích xuất đặc trưng và tính ma trận khoảng cách ở Tầng 1 chỉ sử dụng mạng `ResNet-18`, các tập con siêu nén ($\text{DR} \approx 97,6\% - 98,2\%$) lại đạt hiệu năng phân loại tốt nhất và ổn định nhất khi huấn luyện trên **`MobileNetV3-Small`** ($79,81 \pm 0,67\%$ với `S2`, $\text{ARR} = 89,75\%$) và **`DenseNet-121`** ($79,10 \pm 2,57\%$ với `S2`), cao hơn đáng kể so với chính `ResNet-18` ($75,49 \pm 1,80\%$). Hiện tượng này hoàn toàn nhất quán với lý thuyết học thống kê: khi số lượng mẫu huấn luyện cực nhỏ ($M \approx 200 - 270$ ảnh), kiến trúc `ResNet-18` với $11,18$ triệu tham số dễ bị quá khớp (overfitting) cục bộ trên số lượng batch ít ỏi (thể hiện rõ ở `S3` tại `seed = 456` khi $M$ giảm xuống dưới ngưỡng $200$ ảnh làm độ chính xác của `ResNet-18` giảm xuống $33,80\%$, kéo trung bình của `S3` xuống $60,04\%$ dù trung vị đạt $73,03\%$). Ngược lại, `MobileNetV3-Small` chỉ có $1,52$ triệu tham số (nhỏ hơn $7,3$ lần so với `ResNet-18`) và `DenseNet-121` có cơ chế tái sử dụng đặc trưng qua các khối kết nối dày đặc (Dense Blocks) nên duy trì được độ ổn định cao ngay cả khi huấn luyện trên chưa tới $2,5\%$ dữ liệu gốc.

**Thứ tư, phân tích đối chiếu với `Random Selection (B1)` và `Stratified Random (B2)` và kiểm định thống kê Wilcoxon**:
Trong kịch bản thực nghiệm tự do kích thước $M$, kích thước mẫu tham chiếu của `B1` và `B2` được tính bằng trung bình cộng kích thước tập con của toàn bộ 8 phương pháp tối ưu hóa:
$$M_{\text{baseline}} = \text{round}\left(\frac{3.229 + 243 + 273 + 214 + 227 + 222 + 204 + 224}{8}\right) = 604\text{ ảnh } (\text{DR} = 94,62\%)$$
Do được phân bổ $M = 604$ ảnh (nhiều gấp $2,21$ đến $2,96$ lần số lượng ảnh của các cấu hình `S1–S4, P1, C1`), `B1` và `B2` đạt độ chính xác từ $84,29\%$ đến $86,28\%$, nhưng có điểm thích nghi tổng hợp thấp hơn ($0,8765$ và $0,8796$ so với $0,8862 - 0,8875$ của PO–GA). Mặt khác, so sánh trực tiếp giữa `B2 (Stratified Random)` và `B1 (Random Selection)` tại cùng kích thước $M = 604$ cho thấy việc bảo toàn phân bố lớp ($\text{Bal} = 0,9996$ so với $0,9855$) giúp cải thiện độ chính xác trên cả ba mô hình CNN (tăng $+0,91\%$ trên `ResNet-18`), khẳng định vai trò thiết yếu của thành phần $\text{Bal}(X)$ trong hàm mục tiêu. Trong kiểm định phi tham số Wilcoxon Signed-Rank Test trên $n = 3$ hạt giống (`seeds = [42, 123, 456]`), giá trị $p\text{-value}$ nhỏ nhất có thể đạt được về mặt toán học của phân phối nhị thức hai phía với cỡ mẫu $n = 3$ là $p_{\min} = \frac{2}{2^3} = 0,2500$ (đạt được tại mọi cặp so sánh có sự phân tách tuyệt đối trên cả 3 hạt giống như giữa `S1`, `S2`, `GA-only` so với `C1`).

---

## CHƯƠNG 8: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 8.1. Kết luận

Khóa luận đã hoàn thành toàn bộ các mục tiêu nghiên cứu, thiết kế thuật toán và kiểm chứng thực nghiệm theo đề cương **CNTT-KLCN140**, đạt được bốn kết quả cụ thể:
1. **Xác lập quy trình thực nghiệm chống rò rỉ dữ liệu và mốc chuẩn B0 trên 3 kiến trúc CNN**: Đã phân hoạch chuẩn hóa bộ dữ liệu Intel Image Classification ($17.034$ ảnh, $6$ lớp cảnh tự nhiên) thành ba tập độc lập ($11.228$ Train / $2.806$ Validation / $3.000$ Independent Test) và thiết lập mốc chuẩn toàn tập dữ liệu trên ba kiến trúc học sâu `ResNet-18` ($90,27\%$), `MobileNetV3-Small` ($88,93\%$) và `DenseNet-121` ($90,57\%$).
2. **Ứng dụng thuật toán Parrot Optimizer (PO) truyền thống và Hàm mục tiêu 4 thành phần chuẩn hóa nội tại**: Đã giữ nguyên bốn hành vi bầy đàn của thuật toán Parrot Optimizer truyền thống (Lian và cộng sự, 2024) kết hợp bước ánh xạ nhị phân $|\tanh(\Delta x)|$ ở đầu ra trên không gian $N = 11.228$ chiều và tiền tính toán ma trận khoảng cách Cosine chuẩn hóa trên GPU, giúp tăng tốc quá trình tối ưu hóa gấp hơn $3$ lần so với giải thuật di truyền truyền thống.
3. **Chứng minh hiệu quả của các kiến trúc lai hóa PO–GA**: Thực nghiệm đối sánh toàn diện $10$ phương pháp chỉ ra rằng:
   * Ở vùng rút gọn bảo toàn ($\text{DR} = 71,24\%$, giữ lại $3.229$ ảnh), thuật toán `GA-only` giữ lại tới **$98,99\%$** độ chính xác gốc trên `ResNet-18` ($89,35 \pm 0,39\%$), **$98,72\%$** trên `MobileNetV3-Small` ($87,79 \pm 0,24\%$) và **$98,25\%$** trên `DenseNet-121` ($88,99 \pm 0,09\%$).
   * Ở vùng nén cực đại ($\text{DR} = 97,57\% - 98,18\%$, chỉ giữ lại $204 - 273$ ảnh), các cấu hình lai ghép PO–GA đạt điểm thích nghi cao nhất ($0,8862 - 0,8875$). Đặc biệt, hai cấu hình lai tuần tự xen kẽ **`S2 (Alt PO→GA)`** và **`S1 (Alt GA→PO)`** chỉ sử dụng **$2,16\% - 2,43\%$** lượng dữ liệu gốc nhưng vẫn bảo toàn tới **$89,75\%$** hiệu năng gốc trên `MobileNetV3-Small` ($79,81 \pm 0,67\%$, đạt đỉnh $81,20\%$) và đạt tới **$82,63\%$** ($\text{ARR} = 91,23\%$) trên `DenseNet-121`, đồng thời tăng tốc huấn luyện từ **$2,38\times$ đến $3,03\times$**.
4. **Đóng gói công cụ thực nghiệm hoàn chỉnh**: Đã xây dựng hệ thống mã nguồn mô-đun hóa kèm bộ đặc trưng tiền trích xuất (`features/train_features_train80.npy`) và kịch bản tự động hóa thực nghiệm trên cả máy cục bộ lẫn nền tảng đám mây Kaggle.

### 8.2. Hướng Phát triển Tiếp theo

Từ những phát hiện thực nghiệm trong khóa luận, ba hướng nghiên cứu mở rộng giàu tiềm năng được đề xuất cho các công trình tiếp theo:
1. **Cân bằng động trọng số độ nén ($\delta$) hoặc áp dụng Tối ưu hóa Đa mục tiêu Pareto**: Thay vì cố định trọng số $\delta = 0,40$ khiến toán tử giao tiếp ($S_t = 3$) của PO đẩy tỷ lệ nén lên sát ngưỡng $98\%$, việc giảm $\delta$ xuống vùng $[0,15; 0,25]$ hoặc áp dụng cơ chế trội Pareto sẽ cho phép thuật toán PO–GA trải đều các nghiệm trên toàn bộ dải rút gọn từ $50\%$ đến $95\%$.
2. **Mở rộng quy mô hạt giống thống kê ($n \ge 10$ seeds) và tăng số chu kỳ tinh chỉnh cho tập con siêu nhỏ**: Khi kích thước tập con $M < 300$ ảnh, số lượng bước cập nhật trọng số (gradient steps) trong $5$ epochs giảm đi $40$ lần so với tập gốc; việc điều chỉnh lịch trình tốc độ học (Cosine Annealing Learning Rate) theo kích thước tập con sẽ giúp `ResNet-18` tránh hiện tượng suy giảm độ chính xác ở chế độ nén trên $98\%$.
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


