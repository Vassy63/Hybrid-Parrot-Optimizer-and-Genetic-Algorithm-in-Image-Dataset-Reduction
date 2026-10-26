# TÀI LIỆU KỸ THUẬT VÀ BÁO CÁO KẾT QUẢ THỰC NGHIỆM: PHƯƠNG PHÁP THỰC HIỆN, CHI TIẾT CÁC KIẾN TRÚC LAI HÓA HYBRID PO–GA VÀ ĐỐI SÁNH ĐA KIẾN TRÚC CNN

- **Mã đề tài khóa luận**: CNTT-KLCN140
- **Tên đề tài**: Kết hợp giải thuật di truyền và tối ưu hóa đàn vẹt trong bài toán rút gọn tập dữ liệu ảnh (*Hybrid Parrot Optimizer and Genetic Algorithm in Image Dataset Reduction*)
- **Tập dữ liệu thực nghiệm**: Intel Image Classification ($17.034$ ảnh, $6$ lớp cảnh tự nhiên)
- **Phiên thực nghiệm chính thức**: [`outputs/run_poga_all_20261002_173334`](file:///d:/Code/KhoaLuanDatasetReduction/outputs/run_poga_all_20261002_173334)
- **Cấu hình ngân sách đánh giá**: Kích thước quần thể $P = 20$, số vòng lặp chuẩn $T = 100$, tổng ngân sách cố định $\text{Total FEs} = 2.000\text{ FEs}$ cho mọi thuật toán tối ưu hóa
- **Hệ thống hạt giống kiểm chứng**: Đa hạt giống độc lập `Optimization Seeds` $\in \{42, 123, 456\}$ kết hợp cố định hạt giống huấn luyện học sâu `DL Seed` $= 42$
- **Mô hình học sâu đánh giá hạ nguồn**: `ResNet-18`, `MobileNetV3-Small`, `DenseNet-121`

---

## PHẦN 1: PHƯƠNG PHÁP THỰC HIỆN VÀ QUY TRÌNH THỰC NGHIỆM ("CÁCH LÀM")

### 1.1. Quy trình Phân hoạch Dữ liệu Chống Rò rỉ (Zero Data Leakage) và Trích xuất Đặc trưng

Để đảm bảo tính khách quan tuyệt đối của quy trình đánh giá tập ảnh rút gọn (Coreset Selection), bộ dữ liệu **Intel Image Classification** được phân hoạch nghiêm ngặt thành ba tập tách biệt trong mô-đun [`data/dataset.py`](file:///d:/Code/KhoaLuanDatasetReduction/data/dataset.py):

1. **Tập Huấn luyện gốc ($D_{\text{train}}$ — $80\%$ thư mục `seg_train`)**: Bao gồm **$N = 11.228$ bức ảnh** thuộc $K = 6$ lớp cảnh tự nhiên (`buildings`: $1.753$ ảnh, `forest`: $1.817$ ảnh, `glacier`: $1.924$ ảnh, `mountain`: $2.010$ ảnh, `sea`: $1.820$ ảnh, `street`: $1.904$ ảnh). Đây là không gian tìm kiếm duy nhất mà các thuật toán tối ưu hóa (`GA`, `PO`, `S1–S4`, `P1`, `C1`) được phép truy cập để lựa chọn tập con $S \subset D_{\text{train}}$.
2. **Tập Kiểm định ($D_{\text{val}}$ — $20\%$ thư mục `seg_train`)**: Bao gồm **$2.806$ bức ảnh**, chỉ sử dụng để theo dõi quá trình hội tụ và lưu lại bộ trọng số mô hình tốt nhất (`best_model_state`) theo Validation Accuracy trong suốt các chu kỳ huấn luyện mạng CNN.
3. **Tập Kiểm thử Độc lập ($D_{\text{test}}$ — $100\%$ thư mục `seg_test`)**: Bao gồm **$3.000$ bức ảnh** hoàn toàn độc lập, tuyệt đối không xuất hiện trong giai đoạn trích xuất ma trận khoảng cách, đánh giá hàm mục tiêu hay lựa chọn trọng số.

**Trích xuất vector đặc trưng và tiền tính toán ma trận khoảng cách trên GPU**:
Mỗi bức ảnh $I_i \in D_{\text{train}}$ được chuẩn hóa kích thước về $224 \times 224$ điểm ảnh và đi qua mạng trục **ResNet-18** tiền huấn luyện trên ImageNet (đã loại bỏ tầng phân loại tuyến tính cuối cùng) để trích xuất vector đặc trưng tại tầng gộp trung bình toàn cục (`Global Average Pooling`) $f_i \in \mathbb{R}^{512}$, lưu đệm tại [`features/train_features_train80.npy`](file:///d:/Code/KhoaLuanDatasetReduction/features/train_features_train80.npy). Để loại bỏ sự suy biến hình học của khoảng cách Euclid trong không gian $512$ chiều, toàn bộ các vector đặc trưng được chuẩn hóa độ dài đơn vị ($\hat{f}_i = f_i / \|f_i\|_2$) và tính ma trận khoảng cách Cosine đối xứng $\mathbf{D} \in \mathbb{R}^{11.228 \times 11.228}$ trực tiếp trên bộ nhớ VRAM của GPU trong mô-đun [`optimization/fitness.py`](file:///d:/Code/KhoaLuanDatasetReduction/optimization/fitness.py):

$$D(i, j) = \frac{1 - \hat{f}_i^\top \hat{f}_j}{\max_{u, v \in D_{\text{train}}} \left(1 - \hat{f}_u^\top \hat{f}_v\right)} \in [0, 1]$$

Việc chia cho giá trị cực đại thực nghiệm $\text{dist}_{\max}$ trên toàn tập $D_{\text{train}}$ đảm bảo mọi phần tử trong ma trận khoảng cách $\mathbf{D}$ đều nằm trong đoạn $[0, 1]$ mà không cần sử dụng bất kỳ hằng số gán cứng nào.

---

### 1.2. Biểu diễn Nghiệm Nhị phân và Hàm Mục tiêu Bốn Thành phần

Mỗi phương án rút gọn tập dữ liệu được mã hóa thành một vector nhị phân $X = (x_1, x_2, \dots, x_N) \in \{0, 1\}^N$ với $N = 11.228$, trong đó $x_i = 1$ biểu thị ảnh thứ $i$ được giữ lại trong tập con rút gọn $S = \{i \mid x_i = 1\}$ (với kích thước tập con $M = |S| = \sum_{i=1}^N x_i$) và $x_i = 0$ biểu thị ảnh thứ $i$ bị loại bỏ.

Chất lượng của mỗi cá thể hợp lệ $X$ ($|S| = M \ge 2$) được đánh giá qua bốn thành phần chuẩn hóa trong đoạn $[0, 1]$ tại [`optimization/fitness.py`](file:///d:/Code/KhoaLuanDatasetReduction/optimization/fitness.py):

1. **Độ đa dạng (Diversity - $\text{Div}$)**: Khuyến khích các bức ảnh được chọn phải khác biệt nhau, tránh chọn các ảnh trùng lặp:
   $$\text{Div}(X) = \frac{2}{|S|(|S|-1)} \sum_{i=1}^{N} \sum_{j=i+1}^{N} x_i \cdot x_j \cdot \frac{\text{dist}(f_i, f_j)}{\text{dist}_{\max}} \in [0, 1]$$

2. **Độ bao phủ (Coverage - $\text{Cov}$)**: Đảm bảo các bức ảnh bị loại bỏ không nằm quá xa các bức ảnh được giữ lại (tập con vẫn đại diện được cho toàn bộ không gian dữ liệu):
   $$\text{Cov}(X) = 1 - \frac{1}{N} \sum_{i=1}^{N} \min_{j : x_j = 1} \left( \frac{\text{dist}(f_i, f_j)}{\text{dist}_{\max}} \right) \in [0, 1]$$

3. **Cân bằng lớp (Class Balance - $\text{Bal}$)**: Giữ cho tỷ lệ phân bố giữa các lớp trong tập con phải tương đồng với tập dữ liệu gốc:
   $$\text{Bal}(X) = 1 - \sqrt{\frac{1}{C} \sum_{c=1}^{C} \left( \frac{M_c}{|S|} - \frac{N_c}{N} \right)^2} \in [0, 1]$$

4. **Độ nén (Compression - $\text{Com}$)**: Phạt các cá thể chọn quá nhiều ảnh, khuyến khích kích thước tập con càng nhỏ càng tốt:
   $$\text{Com}(X) = 1 - \frac{M}{N} = 1 - \frac{\sum_{i=1}^{N} x_i}{N} \in [0, 1]$$

**Hàm Fitness tổng hợp**:
$$F(X) = \alpha \cdot \text{Div}(X) + \beta \cdot \text{Cov}(X) + \gamma \cdot \text{Bal}(X) + \delta \cdot \text{Com}(X)$$
với bộ trọng số chuẩn hóa $(\alpha, \beta, \gamma, \delta) = (0,15; \, 0,45; \, 0,20; \, 0,20)$ thỏa mãn $\alpha + \beta + \gamma + \delta = 1$.

---

### 1.3. Cơ chế Toán học của Hai Thuật toán Thành phần (GA và PO)

#### 1.3.1. Giải thuật Di truyền cho Lựa chọn Mẫu Nhị phân (`optimization/genetic_algorithm.py`)
Giải thuật Di truyền (GA) vận hành trực tiếp trên quần thể các vector nhị phân $\{X_1, X_2, \dots, X_P\}$ ($P = 20$). Tại mỗi thế hệ $t \in \{1, \dots, T\}$, GA thực hiện tuần tự bốn bước:
1. **Bảo tồn tinh hoa (Elitism)**: Sắp xếp quần thể theo chiều giảm dần của hàm thích nghi $F(X)$ và sao chép nguyên trạng $E = 5$ cá thể đứng đầu sang quần thể thế hệ kế tiếp.
2. **Chọn lọc giải đấu (Tournament Selection)**: Để chọn mỗi cá thể cha mẹ, lấy ngẫu nhiên $k_{\text{tour}} = 4$ cá thể từ quần thể hiện tại và chọn cá thể có điểm thích nghi cao nhất.
3. **Lai ghép đồng nhất (Uniform Crossover)**: Với xác suất lai ghép $p_c = 0,8$, hai cá thể cha mẹ $P_1, P_2$ trao đổi từng bit độc lập theo mặt nạ nhị phân ngẫu nhiên $M_{\text{mask}} \in \{0, 1\}^N$ ($\Pr(M_{\text{mask}, j} = 1) = 0,5$) để tạo ra hai cá thể con $C_1, C_2$.
4. **Đột biến thích nghi giảm dần (Adaptive Bit-Flip Mutation)**: Tỷ lệ đột biến $p_m(t)$ giảm tuyến tính theo số thế hệ để ưu tiên thăm dò ở giai đoạn đầu và ổn định hội tụ ở giai đoạn cuối:
   $$p_m(t) = \max\left(p_{\min},\; p_{\max} - (p_{\max} - p_{\min}) \cdot \frac{t}{T}\right)$$
   với $p_{\max} = 0,0010$ và $p_{\min} = 0,0002$. Để giữ cân bằng mật độ tập con hiện tại $r = \frac{M}{N}$, xác suất lật bit tại mỗi vị trí được điều chỉnh tương ứng là $p_m(t)\cdot(1-r)$ đối với các bit $x_j = 1$ và $p_m(t)\cdot r$ đối với các bit $x_j = 0$, sau đó đưa qua toán tử sửa chữa $\text{Repair}(X)$.

#### 1.3.2. Thuật toán Tối ưu hóa Đàn vẹt Truyền thống Kèm Bước Ánh xạ Nhị phân Đầu ra (`optimization/parrot_optimizer.py`)
Thuật toán Parrot Optimizer (PO) truyền thống của Lian và cộng sự (2024) mô phỏng bốn hành vi tự nhiên của loài vẹt *Pyrrhura molinae* để cập nhật vị trí cá thể trên không gian liên tục $\mathbb{R}^N$. Để áp dụng trực tiếp PO truyền thống vào bài toán chọn mẫu ảnh $X \in \{0, 1\}^N$, hệ thống giữ nguyên bốn phương trình cập nhật vị trí của PO truyền thống ở Bước 1, sau đó thực hiện bước ánh xạ về $\{0, 1\}$ ở Bước 2:

**Bước 1: Cập nhật vị trí liên tục theo đúng 4 hành vi của PO truyền thống**  
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

1. **Bài toán bất tương thích miền biểu diễn (Domain Mismatch)**:
   Thuật toán Parrot Optimizer nguyên bản được thiết kế cho các bài toán tối ưu hóa liên tục ($X_i^{\text{cont}} \in \mathbb{R}^N$), nơi vị trí cá thể sau Bước 1 có thể nhận giá trị thực dương hoặc âm bất kỳ (chẳng hạn $0,73$ hoặc $-1,42$) do tác động của phân phối nặng đuôi $\text{Levy}(N)$ và hàm lượng giác $\cos(\theta)$. Ngược lại, trong bài toán rút gọn tập dữ liệu ảnh, nghiệm bắt buộc phải nằm trên không gian tổ hợp nhị phân $\{0, 1\}^N$ ($1$: giữ lại ảnh, $0$: loại bỏ ảnh).

2. **Hạn chế của phương pháp cắt ngưỡng cứng và hàm truyền hình chữ S (S-shaped / Sigmoid)**:
   Trong các thuật toán bầy đàn nhị phân thế hệ đầu, vị trí liên tục thường được quy đổi qua hàm Sigmoid hình chữ S: $S(x) = \frac{1}{1 + e^{-x}}$, sau đó gán trực tiếp $x_{i,d}^{t+1} = 1$ nếu $\text{rand}(0,1) < S(x)$. Tuy nhiên, cơ chế chữ S bộc lộ nhược điểm nghiêm trọng trên không gian $N = 11.228$ chiều: khi cá thể vẹt đã hội tụ về gần nghiệm tối ưu và hầu như đứng yên ($\Delta x_{i,d} = 0$), giá trị $S(0) = 0,5$ vẫn tạo ra xác suất $50\%$ gán lại bit thành $1$ hoặc $0$ một cách ngẫu nhiên. Điều này làm phá vỡ cấu trúc tập con tốt đang có của cá thể hiện tại $X_i^t$ và đẩy mật độ tập con trôi cưỡng bức về $50\%$.

3. **Nguyên lý hoạt động của Hàm truyền Hình chữ V ($V\text{-shaped Transfer Function}$)**:
   Để khắc phục triệt để hạn chế trên, mô-đun [`optimization/parrot_optimizer.py`](file:///d:/Code/KhoaLuanDatasetReduction/optimization/parrot_optimizer.py#L33-L41) cài đặt cơ chế **Hàm truyền hình chữ V** (Mirjalili & Lewis, 2013). Thay vì dùng vị trí tuyệt đối để gán cứng bit $0$ hay $1$, thuật toán trích xuất **vector độ dịch chuyển (Positional Displacement)** $\Delta X_i = X_i^{\text{cont}} - X_i^t$ và đưa qua hàm truyền chữ V $V(\Delta x) = |\tanh(\Delta x)|$ để tính **xác suất đảo trạng thái bit hiện tại ($P_{\text{flip}}$)**:

```text
PO truyền thống (X_cont ∈ ℝ^N) 
   ──► Tính độ dịch chuyển: ΔX = X_cont - X_current
   ──► Hàm truyền chữ V:    V(ΔX) = |tanh(ΔX)| ∈ [0, 1)
   ──► Đảo bit có điều kiện: Nếu rand(0,1) < V(ΔX) thì lật bit (0 ↔ 1), ngược lại giữ nguyên X_current
```

   Công thức toán học của hàm truyền chữ V và quy tắc cập nhật trạng thái nhị phân tại mỗi chiều $d \in \{1, \dots, N\}$ được xác định như sau:

   $$V(\Delta x_{i,d}) = \left|\tanh(\Delta x_{i,d})\right| = \left|\frac{e^{\Delta x_{i,d}} - e^{-\Delta x_{i,d}}}{e^{\Delta x_{i,d}} + e^{-\Delta x_{i,d}}}\right| \in [0, 1)$$

   $$x_{i,d}^{t+1} = \begin{cases} 
   1 - x_{i,d}^t, & \text{nếu } \text{rand}(0,1) < \left|\tanh(\Delta x_{i,d})\right| \\
   x_{i,d}^t, & \text{ngược lại}
   \end{cases}$$

4. **Ba đặc tính toán học ưu việt của hàm $V(\Delta x) = |\tanh(\Delta x)|$ trong bài toán rút gọn dữ liệu**:
   - **Bảo toàn cấu trúc nghiệm khi hội tụ ($V(0) = 0$)**: Khi cá thể vẹt đã tiến sát về vị trí tốt nhất toàn cục $X_{\text{gbest}}$ hoặc ở giai đoạn cuối quá trình tìm kiếm khi biên độ dịch chuyển $\Delta x_{i,d} \to 0$, xác suất đảo bit $|\tanh(\Delta x_{i,d})| \to 0$. Nhờ đó, bức ảnh thứ $d$ giữ nguyên trạng thái đã chọn/bỏ từ vòng lặp trước ($x_{i,d}^{t+1} = x_{i,d}^t$), giúp bảo toàn trọn vẹn cấu trúc tập con ưu tú mà không bị nhiễu loạn ngẫu nhiên.
   - **Tính đối xứng chẵn hai chiều ($V(-\Delta x) = V(\Delta x)$)**: Do đồ thị hàm $|\tanh(\Delta x)|$ có hình chữ V đối xứng qua trục tung tại gốc tọa độ, cả bước nhảy dương lớn ($\Delta x \gg 0$) lẫn bước nhảy âm lớn ($\Delta x \ll 0$) sinh ra bởi chuyển động Lévy đều tạo ra xác suất đảo bit cao như nhau. Khi $|\Delta x_{i,d}|$ lớn, hàm chữ V cho phép linh hoạt chuyển đổi hai chiều: từ $0 \to 1$ (bổ sung một bức ảnh mới giàu tính đại diện vào tập con) hoặc từ $1 \to 0$ (loại bỏ một bức ảnh dư thừa để tăng độ nén $\text{Com}$).
   - **Khả năng kẹp biên mượt trước các bước nhảy cực đại của phân phối Lévy ($\lim_{|\Delta x| \to \infty} V(\Delta x) = 1$)**: Phân phối nặng đuôi Lévy với $\beta = 1,5$ thỉnh thoảng sinh ra các bước nhảy có biên độ rất lớn. Hàm $\tanh$ tự động nén mượt mọi giá trị $|\Delta x| \in [0, +\infty)$ vào nửa khoảng xác suất hợp lệ $[0, 1)$ mà không cần cắt ngưỡng thô bạo (clipping), giúp duy trì độ nhạy đạo hàm ở vùng lân cận $0$ ($\text{d}V/\text{d}|\Delta x|\big|_{0} = 1$) và bão hòa dần khi $|\Delta x| > 2$.

**Bảng đối sánh trực quan giữa Hàm truyền chữ S (Sigmoid) và Hàm truyền chữ V ($\lvert\tanh(\Delta x)\rvert$)**:

| Tiêu chí kỹ thuật | Hàm truyền hình chữ S: $S(\Delta x) = \frac{1}{1 + e^{-\Delta x}}$ | Hàm truyền hình chữ V: $V(\Delta x) = \lvert\tanh(\Delta x)\rvert$ (Sử dụng trong PO) |
| :--- | :--- | :--- |
| **Ý nghĩa xác suất đầu ra** | Xác suất gán cứng bit thành $1$ ($\Pr(x_{i,d}^{t+1} = 1)$) | Xác suất **đảo trạng thái** hiện tại ($\Pr(x_{i,d}^{t+1} = 1 - x_{i,d}^t)$) |
| **Hành vi khi đứng yên ($\Delta x = 0$)** | $S(0) = 0,50$ (Xáo trộn ngẫu nhiên $50\%$ số mẫu ảnh) | **$V(0) = 0,00$ (Giữ nguyên $100\%$ trạng thái tập con hiện tại)** |
| **Hành vi khi dịch chuyển mạnh ($\lvert\Delta x\rvert \to \infty$)** | Bất đối xứng ($\Delta x \to +\infty$ ép về $1$, $\Delta x \to -\infty$ ép về $0$) | **Đối xứng hai chiều ($V(\pm\infty) \to 1$, cho phép đảo cả $0 \to 1$ và $1 \to 0$)** |
| **Tính tương thích với kế thừa gen GA** | Kém (Xóa bỏ cấu trúc tập con mà pha GA vừa lai ghép được) | **Rất cao (Kế thừa trực tiếp vector nhị phân từ GA và chỉ lật các bit có $\Delta x$ lớn)** |

Cuối cùng, sau khi tạo ra vector nhị phân ứng viên $X_i^{\text{cand}}$ qua hàm truyền chữ V và toán tử sửa chữa $\text{Repair}$, vị trí mới chỉ được cập nhật theo cơ chế chọn lọc tham lam (Greedy Selection) nếu có điểm Fitness cao hơn vị trí cũ ($F(X_i^{\text{cand}}) > F(X_i^t)$).

> [!TIP]
> **Ghi chú Trực quan cho Thành viên Nhóm (Cách hiểu nhanh cơ chế V-Transfer & `rand() < P_lật` khi bảo vệ Khóa luận)**:
> Toàn bộ cơ chế nhị phân hóa trong [`optimization/parrot_optimizer.py`](file:///d:/Code/KhoaLuanDatasetReduction/optimization/parrot_optimizer.py#L33-L41) thực chất chỉ gói gọn trong 3 bước cực kỳ đơn giản:
> ```python
> def v_shaped_transfer(self, delta_x):
>     return np.abs(np.tanh(delta_x))
>     
> def binarize(self, x_current, x_new):
>     delta_x = x_new - x_current                           # Bước 1: Lấy vị trí mới (số thực) trừ vị trí cũ (0 hoặc 1)
>     v_prob = self.v_shaped_transfer(delta_x)              # Bước 2: Tính xác suất lật bit P_lật = |tanh(delta_x)| ∈ [0, 1)
>     mask = (np.random.rand(self.N) < v_prob).astype(int)  # Bước 3: Lượng tử hóa ngẫu nhiên (Stochastic Binarization)
>     x_binary = np.where(mask, 1 - x_current, x_current)   # Nếu trúng xác suất thì đảo bit (0 ↔ 1), ngược lại giữ nguyên
>     return repair_solution(x_binary)
> ```
> - **Tại sao lấy `vị trí mới - vị trí cũ` ($\Delta x$)?** Vì trong không gian nhị phân, ta không quan tâm con vẹt đang bay tới tọa độ thực bao nhiêu, mà chỉ quan tâm **bước này nó muốn dịch chuyển mạnh hay yếu** so với trạng thái hiện tại. Nếu $\Delta x \approx 0$ (vẹt đứng yên), $P_{\text{lật}} = |\tanh(0)| = 0\% \rightarrow$ giữ nguyên trạng thái ảnh. Nếu $|\Delta x|$ càng lớn (dù âm hay dương), $P_{\text{lật}} \to 100\% \rightarrow$ bức ảnh đó đang rất cần đổi trạng thái ($0 \to 1$ hoặc $1 \to 0$).
> - **Tại sao dùng `rand() < P_lật` mà không dùng ngưỡng cố định `P_lật > 0.5`?** Phép so sánh `rand() < P_lật` (với `rand()` quay số ngẫu nhiên đều từ $0.00$ đến $1.00$) chính là **vòng quay xác suất (Stochastic Binarization)**:
>   - Ví dụ bức ảnh có $P_{\text{lật}} = 0,80$ ($80\%$ cơ hội lật): Khi quay một số ngẫu nhiên từ $0$ đến $1$, đoạn từ $[0,00; \, 0,80)$ chiếm đúng $80\%$ diện tích. Nếu quay ra `0.35 < 0.80` $\rightarrow$ **Lật bit**; nếu quay ra `0.92 >= 0.80` $\rightarrow$ **Giữ nguyên**.
>   - Nếu dùng ngưỡng cứng `P_lật > 0.5`, mọi bức ảnh có bước dịch chuyển nhỏ ($P_{\text{lật}} = 0,15$ hay $15\%$) sẽ **không bao giờ** có cơ hội được thử nghiệm đổi trạng thái ($0\%$ thay vì $15\%$). Nhờ dùng `rand() < P_lật`, các bit có bước nhảy lớn ($P_{\text{lật}} = 0,95$) gần như chắc chắn được lật ($95\%$), trong khi các bit có bước nhảy nhỏ ($P_{\text{lật}} = 0,10$) vẫn có $10\%$ cơ hội đổi trạng thái để giúp thuật toán thoát khỏi cực trị địa phương.

---

### 1.4. Các Cải tiến Thuật toán & Kỹ thuật Giúp Nhảy vọt Độ chính xác (Đối sánh với Phiên bản Cơ sở Ban đầu)

Trong quá trình nghiên cứu và hoàn thiện hệ thống, nhóm đã phát hiện và giải quyết các nút thắt toán học khiến phiên bản sơ khởi ban đầu gặp hiện tượng suy giảm độ chính xác hoặc nén dữ liệu mất kiểm soát. Những thay đổi dưới đây (được ghi nhận trong lịch sử mã nguồn của dự án) vừa bao gồm việc **hiệu chỉnh công thức vật lý của PO**, vừa mang lại **3 đóng góp phương pháp luận (Methodological Contributions)** quan trọng giúp Test Accuracy trên tập Coreset nhảy vọt lên mức **$89,03\% - 89,51\%$ trên ResNet-18** và **$89,42\%$ trên DenseNet-121** (bảo toàn $\approx 99\%$ độ chính xác của tập gốc dù đã cắt bỏ $\sim 74\%$ dữ liệu):

#### 1.4.1. Khắc phục hiện tượng "Nén cực đoan" (Over-Compression) trong Cập nhật Vị trí của PO
* **Vấn đề ở phiên bản cũ**: Trong công thức Kiếm ăn ($S_t = 1$), việc thiếu số hạng vị trí gốc `x_current` (`x_new = (x_current - gbest_pos) * levy + term2`) làm cho khi cá thể hiện tại và cá thể tốt nhất cùng chọn một bức ảnh (`x_current = 1`, `gbest_pos = 1`), hiệu số bằng $0$ kéo `x_new` sụt về sát $0$. Khi đó độ dịch chuyển $\Delta x = x_{\text{new}} - x_{\text{current}} \approx -1$ sinh ra xác suất lật bit rất lớn, vô tình xóa hàng loạt ảnh tốt đang có (lật từ $1 \to 0$). Ngoài ra, bước nhảy ở các hành vi $S_t \in \{2, 3\}$ chưa được điều tiết bởi hệ số co giãn $\alpha \in [0; 0,2]$ và vector hướng đàn chuẩn hóa, làm $|\Delta x|$ quá lớn khiến hàm truyền chữ V lật bit gần như ngẫu nhiên.
* **Giải pháp cải tiến ([`optimization/parrot_optimizer.py`](file:///d:/Code/KhoaLuanDatasetReduction/optimization/parrot_optimizer.py#L43-L84))**: Chuẩn hóa toàn bộ 4 hành vi xuất phát từ vị trí nền `x_current + alpha * (...)` kết hợp vector hướng đàn trung tâm hóa `flock_vec = 0.22 * (x_current - mean_pos) + 0.85 * v`. Nhờ đó, độ dịch chuyển $\Delta x = x_{\text{new}} - x_{\text{current}}$ phản ánh chính xác lực kéo của bầy đàn, chỉ lật các bit thực sự cần thiết.

#### 1.4.2. Đóng góp Học thuật 1 — Toán tử Đột biến Cân bằng Mật độ (Density-Balanced Adaptive Mutation) trong GA
* **Vấn đề của GA truyền thống**: Đột biến lật bit cổ điển sử dụng chung một xác suất $p_m$ cho cả bit `1` và bit `0`. Tuy nhiên, trong bài toán rút gọn dữ liệu với tỷ lệ giữ mẫu $r = \frac{M}{N} \approx 27\%$, số lượng bit `0` (ảnh bị loại, chiếm $\sim 73\%$) nhiều gấp $2,7$ lần số lượng bit `1` (ảnh được giữ, chiếm $\sim 27\%$). Nếu áp dụng cùng xác suất đột biến $p_m$, số lượng ảnh bị lật từ `0 -> 1` trong mỗi thế hệ sẽ nhiều gấp $2,7$ lần số ảnh bị lật từ `1 -> 0`, gây ra hiện tượng **trôi dạt kích thước tập con (Subset Size Drift)** làm tập Coreset phình to mất kiểm soát.
* **Giải pháp cải tiến ([`optimization/genetic_algorithm.py`](file:///d:/Code/KhoaLuanDatasetReduction/optimization/genetic_algorithm.py#L45-L56))**: Thiết kế toán tử đột biến bất đối xứng có trọng số phụ thuộc mật độ hiện tại $r = \text{mean}(X)$:
  $$\Pr(\text{lật bit } j) = \begin{cases} p_m(t) \cdot (1 - r), & \text{nếu } x_j = 1 \\ p_m(t) \cdot r, & \text{nếu } x_j = 0 \end{cases}$$
  Với công thức này, kỳ vọng toán học của số lượng bit bị loại bỏ ($1 \to 0$) là $N \cdot r \cdot p_m(t)(1-r)$, **bằng tuyệt đối** với kỳ vọng số lượng bit được thêm mới ($0 \to 1$) là $N \cdot (1-r) \cdot p_m(t) \cdot r$. Toán tử đột biến từ chỗ gây nhiễu kích thước đã trở thành một **cơ chế hoán đổi mẫu cân bằng (Balanced Sample Swapping)**: cứ loại bỏ 1 bức ảnh dư thừa thì đồng thời bổ sung 1 bức ảnh tiềm năng mới, giữ ổn định tỷ lệ nén.

#### 1.4.3. Đóng góp Học thuật 2 — Tái cân bằng Trọng số Hàm Mục tiêu Ưu tiên Độ bao phủ Hình học ($\text{Cov}$-Prioritized Fitness)
* **Vấn đề đánh đổi (Trade-off) trong Coreset Selection**: Trong 4 thành phần mục tiêu, Độ nén $\text{Com}(X) = 1 - \frac{M}{N}$ là đại lượng tuyến tính dễ tăng điểm nhất (chỉ cần xóa bớt ảnh là $\text{Com}$ tăng ngay lập tức). Ở cấu hình thử nghiệm ban đầu, khi đặt trọng số nén quá cao ($\delta = 0,40$ và $\beta = 0,30$), các thuật toán tiến hóa sẽ đi theo "đường tắt" là xóa sạch dữ liệu để ăn điểm $\text{Com}$, dẫn đến tập con quá nhỏ và làm mạng CNN bị thiếu hụt dữ liệu nghiêm trọng (Underfitting).
* **Giải pháp cải tiến ([`configs/config.py`](file:///d:/Code/KhoaLuanDatasetReduction/configs/config.py#L40-L46))**: Xác lập bộ trọng số thực nghiệm tối ưu $(\alpha = 0,15; \, \beta = 0,45; \, \gamma = 0,20; \, \delta = 0,20)$, đặt **Độ bao phủ không gian đặc trưng ($\beta = 0,45$)** làm trụ cột lớn nhất (chiếm $45\%$ tổng điểm thích nghi). Cấu hình này tạo ra một điểm cân bằng Pareto tự nhiên quanh ngưỡng rút gọn $\text{DR} \approx 73\% - 75\%$ ($M \approx 2.800 - 3.010$ ảnh): thuật toán chỉ được phép cắt bỏ những bức ảnh trùng lặp nằm sâu trong tâm cụm, nhưng bắt buộc phải giữ lại đầy đủ các mẫu đại diện ở vùng biên phân lớp để bảo toàn $\text{Cov} \approx 0,889$, giúp mạng CNN đạt độ chính xác sát mức tập đầy đủ.

#### 1.4.4. Đóng góp Học thuật 3 — Pha Đồng tinh chỉnh sau Hợp nhất (`P1`) và Đồng bộ hóa Tinh hoa Khử trùng lặp (`C1`)
* **Ở kiến trúc Song song `P1` ([`run_p1_parallel_merge`](file:///d:/Code/KhoaLuanDatasetReduction/optimization/hybrid_poga.py#L94-L166))**: Thay vì dành toàn bộ 100% ngân sách cho hai nhánh chạy độc lập rồi chỉ lấy cá thể tốt nhất ở bước cuối (khiến cá thể của GA và PO chưa từng được kết hợp với nhau), kiến trúc `P1` mới chia quá trình làm 2 giai đoạn: $50\%$ ngân sách đầu chạy song song độc lập ($\text{GA} \parallel \text{PO}$), sau đó hợp nhất chọn $\text{Top-}20$ cá thể và dùng $50\%$ ngân sách còn lại để thực hiện **Đồng tinh chỉnh (Co-Refinement)** qua cả GA và PO trên quần thể đã hợp nhất.
* **Ở kiến trúc Hợp tác `C1` ([`run_c1_cooperative_exchange`](file:///d:/Code/KhoaLuanDatasetReduction/optimization/hybrid_poga.py#L168-L229))**: Sau mỗi chặng 5 vòng lặp, hai quần thể $\text{Pop}_{\text{GA}}$ và $\text{Pop}_{\text{PO}}$ được gộp chung, lọc bỏ hoàn toàn các cá thể trùng lặp bằng phép kiểm tra vector phân biệt `np.unique(..., axis=0)` và chọn ra $\text{Top-}20$ cá thể ưu tú độc nhất để đồng bộ hóa cho chu kỳ tiếp theo. Cơ chế này ngăn chặn triệt để hiện tượng một cá thể siêu hạng nhân bản tràn lan làm suy thoái đa dạng di truyền.

#### 1.4.5. Cải tiến Kỹ thuật Hệ thống — Tách biệt Tập Test Độc lập (`seg_test`) và Tăng tốc Tensor trên GPU
* **Quy trình đánh giá 3 tập chuẩn khoa học ([`data/dataset.py`](file:///d:/Code/KhoaLuanDatasetReduction/data/dataset.py#L26-L105))**: Tách bạch hoàn toàn tập huấn luyện gốc `seg_train` ($14.034$ ảnh) thành $80\%$ `Train` ($11.228$ ảnh — chỉ rút gọn trên tập này) và $20\%$ `Val` ($2.806$ ảnh — dùng để chọn checkpoint tốt nhất qua các epoch), sau đó đánh giá cuối cùng trên $100\%$ tập `seg_test` độc lập ($3.000$ ảnh) chưa từng xuất hiện trong quá trình chọn mẫu hay huấn luyện.
* **Tối ưu hóa tính toán ma trận trên VRAM GPU ([`optimization/fitness.py`](file:///d:/Code/KhoaLuanDatasetReduction/optimization/fitness.py#L78-L92))**: Gộp chung phép trích xuất ma trận con `cols = torch.index_select(dist_pt, 1, idx_pt)` trên GPU để tính đồng thời cả $\text{Div}$ và $\text{Cov}$ trong một lần truy xuất bộ nhớ duy nhất, đồng thời loại bỏ `ThreadPoolExecutor` gây tranh chấp khóa GIL của Python. Cải tiến này giúp thời gian chạy trọn vẹn 2.000 lần đánh giá hàm mục tiêu (2.000 FEs) giảm xuống chỉ còn **$\sim 9,5\text{ giây}$ cho mỗi thuật toán**.

**Bảng 1.2: Tóm tắt đối sánh giữa Phiên bản Cơ sở ban đầu và Phiên bản Cải tiến Hoàn chỉnh**

| Thành phần hệ thống | Phiên bản sơ khởi (Trước cải tiến) | Phiên bản hoàn chỉnh (Sau cải tiến) | Tác động trực tiếp lên kết quả thực nghiệm |
| :--- | :--- | :--- | :--- |
| **Cập nhật vị trí PO (`St=1..4`)** | Thiếu số hạng gốc `x_current` ở $S_t=1$; bước nhảy chưa chuẩn hóa theo $\alpha$ và `flock_vec` | Xuất phát chuẩn từ `x_current + alpha * (...)` kèm vector hướng đàn trung tâm hóa | Chấm dứt hiện tượng xóa nhầm hàng loạt bit `1` (nén cực đoan), giữ ổn định $\text{DR} \approx 73\% - 75\%$ |
| **Đột biến trong GA (`mutate`)** | Đột biến đồng nhất `rand(N) < p_mut` với $p_{\max} = 0,03$ quá lớn | **Đột biến cân bằng mật độ** $p_m \cdot (1-r)$ cho bit `1` và $p_m \cdot r$ cho bit `0` ($p_{\max}=0,001$) | Biến đột biến thành phép hoán đổi mẫu 1-đổi-1, chống phình to kích thước tập con |
| **Trọng số Fitness $(\alpha, \beta, \gamma, \delta)$** | Ưu tiên nén quá mức $(0,10; \, 0,30; \, 0,20; \, 0,40)$ | **Ưu tiên Độ bao phủ đặc trưng** $(0,15; \, \mathbf{0,45}; \, 0,20; \, 0,20)$ | Bảo toàn trọn vẹn biên phân lớp ($\text{Cov} \approx 0,889$), giúp Test Accuracy đạt tới $89,51\%$ |
| **Kiến trúc Lai `P1` & `C1`** | `P1` chỉ gộp cuối kỳ; `C1` chép cá thể đơn lẻ | `P1` bổ sung pha **Co-Refinement**; `C1` gộp và khử trùng lặp toàn cục `np.unique` | `P1` đạt $\text{ARR} = 99,01\%$ trên ResNet-18; `C1` đạt độ ổn định cao nhất ($\text{Std} = \pm 0,0001$) |
| **Đánh giá Fitness trên GPU** | Gọi tách rời `compute_diversity` & `compute_coverage` qua `ThreadPoolExecutor` | Tái sử dụng chung tensor `cols` trên GPU VRAM, thực thi tuần tự không nghẽn GIL | Tăng tốc độ tối ưu hóa gấp $\sim 5\times$ (chỉ mất $\sim 9,5\text{ s}$ cho $2.000\text{ FEs}$) |

---

## PHẦN 2: THIẾT KẾ CHI TIẾT SÁU BIẾN THỂ LAI HÓA HYBRID PO–GA ("CHI TIẾT VỀ CÁC HYBRID")

Mặc dù cả GA và PO đều có khả năng giải bài toán tối ưu hóa tổ hợp, hai thuật toán này sở hữu cơ chế tìm kiếm đối lập và bổ trợ lẫn nhau:
- **Ưu thế và hạn chế của GA**: Toán tử lai ghép đồng nhất (`Uniform Crossover`) của GA rất mạnh trong việc kết hợp các nhóm mẫu ảnh con đã tốt từ hai cá thể cha mẹ khác nhau và duy trì độ cân bằng lớp ($\text{Bal}$), nhưng lại thiếu khả năng dẫn hướng toàn cục theo tâm đàn nên tốc độ hội tụ ở giai đoạn đầu chậm hơn.
- **Ưu thế và hạn chế của PO**: Nhờ bốn hành vi bầy đàn định hướng trực tiếp bởi $X_{\text{gbest}}$, $X_{\text{mean}}^t$ và bước nhảy phân phối nặng đuôi $\text{Levy}(N)$, PO có khả năng thăm dò không gian rộng và thoát khỏi cực trị địa phương rất nhanh, nhưng do cập nhật độc lập từng cá thể quanh tâm đàn mà không có toán tử lai ghép gen chéo giữa hai cá thể nên khả năng tinh chỉnh tổ hợp vi mô ở cuối quá trình kém tinh tế hơn GA.

Để khảo sát toàn diện cách thức phối hợp hai cơ chế này dưới cùng ngân sách cố định **$\text{Total FEs} = 2.000$** ($P = 20, T = 100$), mô-đun [`optimization/hybrid_poga.py`](file:///d:/Code/KhoaLuanDatasetReduction/optimization/hybrid_poga.py) thiết kế **6 cấu hình lai hóa** chia thành **3 nhóm kiến trúc** chính.

```
====================================================================================================
 TỔNG QUAN 6 BIẾN THỂ LAI HÓA HYBRID PO-GA DƯỚI CÙNG NGÂN SÁCH 2.000 FEs (P = 20, T = 100)
====================================================================================================

 [NHÓM 1: TUẦN TỰ XEN KẼ CHU KỲ NGẮN (Alternating Sequential - p = 10 vòng/pha, 5 chu kỳ)]
   • S1 (Alt GA→PO):  P0 ──► [GA(10) ──► PO(10)] × 5 chu kỳ liên tiếp  (1.000 GA FEs + 1.000 PO FEs)
   • S2 (Alt PO→GA):  P0 ──► [PO(10) ──► GA(10)] × 5 chu kỳ liên tiếp  (1.000 PO FEs + 1.000 GA FEs)

 [NHÓM 2: TUẦN TỰ CHIA KHỐI LỚN (Block Sequential - 50 vòng/khối, 1 lần chuyển giao)]
   • S3 (Block GA→PO): P0 ──► [Khối GA: 50 thế hệ] ──(Handoff)──► [Khối PO: 50 vòng lặp] (2.000 FEs)
   • S4 (Block PO→GA): P0 ──► [Khối PO: 50 vòng lặp] ──(Handoff)──► [Khối GA: 50 thế hệ] (2.000 FEs)

 [NHÓM 3: SONG SONG HỢP NHẤT (P1) & HỢP TÁC ĐỒNG TIẾN HÓA TRAO ĐỔI TINH HOA (C1)]
   • P1 (Parallel Merge):
        P0 ──┬──► Nhánh GA (50% = 1.000 FEs) ──┐
             └──► Nhánh PO (50% = 1.000 FEs) ──┴──► Merge & Chọn lọc Tinh hoa Top-20
   • C1 (Cooperative Bidirectional Elite Exchange):
        P0 ──┬──► Quần thể GA (P=20) ──[5 vòng]──┐  Trao đổi chéo Elites
             └──► Quần thể PO (P=20) ──[5 vòng]──┴──(Khử trùng lặp Hamming)──► Lặp lại 10 chặng
====================================================================================================
```

### 2.1. Nhóm Kiến trúc 1: Lai hóa Tuần tự Xen kẽ Chu kỳ Ngắn (`S1: Alt GA→PO` và `S2: Alt PO→GA`)

Trong kiến trúc Tuần tự Xen kẽ, một quần thể duy nhất có kích thước $P = 20$ liên tục luân phiên đi qua hai pha điều khiển của GA và PO với độ dài mỗi pha là $p = 10$ vòng lặp. Với tổng số vòng lặp $T = 100$, thuật toán thực hiện tổng cộng $C = \frac{T}{2p} = 5$ chu kỳ khép kín ($9$ lần chuyển giao trạng thái quần thể giữa hai bộ tối ưu hóa):

- **Biến thể `S1 — Alternating GA→PO` (`run_s1_alternating_ga_po`)**:
  - Trong mỗi chu kỳ $c \in \{1, 2, 3, 4, 5\}$:
    1. **Pha GA ($10$ thế hệ = $200\text{ FEs}$)**: Quần thể hiện tại thực hiện chọn lọc giải đấu, lai ghép đồng nhất và đột biến thích nghi để tái tổ hợp các đoạn gen tốt và ổn định phân bố lớp.
    2. **Chuyển giao trạng thái (State Handoff)**: Toàn bộ $20$ cá thể nhị phân cùng điểm Fitness vừa đánh giá xong ở thế hệ thứ $10$ của GA được truyền nguyên vẹn sang làm quần thể đầu vào cho PO mà không tốn thêm bất kỳ lần gọi hàm mục tiêu dư thừa nào.
    3. **Pha PO ($10$ vòng lặp = $200\text{ FEs}$)**: Quần thể tiếp tục thực hiện bốn hành vi đàn vẹt với bước nhảy Lévy và hàm truyền chữ V quanh cá thể dẫn đầu $X_{\text{gbest}}$ để đẩy nhanh tốc độ nén dữ liệu và thoát khỏi vùng hội tụ cục bộ của GA.
- **Biến thể `S2 — Alternating PO→GA` (`run_s2_alternating_po_ga`)**:
  - Đảo ngược trình tự khởi phát trong mỗi chu kỳ: bắt đầu bằng **Pha PO ($10$ vòng lặp)** để nhanh chóng phát hiện các vùng không gian có mật độ tập con tốt và kéo cả đàn về vùng tiềm năng, sau đó chuyển giao sang **Pha GA ($10$ thế hệ)** để lai ghép chéo các cá thể vẹt dẫn đầu và khôi phục độ đa dạng cấu trúc gen.
- **Đặc trưng động lực học của `S1` và `S2`**:
  - Nhờ việc chuyển pha liên tục mỗi $10$ vòng lặp, bộ tham số động của cả hai thuật toán được đồng bộ theo tiến độ toàn cục $t \in [1, 100]$.
  - Cơ chế xen kẽ chu kỳ ngắn tạo ra hiệu ứng **"rung lắc — tái tổ hợp" (Perturbation–Recombination Synergy)**: ngay khi GA bắt đầu chững lại sau $10$ thế hệ lai ghép, PO lập tức tạo các bước nhảy hướng tâm đàn để cắt giảm các mẫu dư thừa; ngược lại, khi PO có xu hướng làm các cá thể hội tụ sát về $X_{\text{gbest}}$, pha GA kế tiếp lập tức trộn chéo các đặc trưng tốt giữa các cá thể. Đây là lý do thực nghiệm cho thấy **`S2` và `S1` đạt điểm Fitness cao nhất toàn cục ($0,8249$ và $0,8248$) cùng tỷ lệ nén cao nhất ($74,81\%$ và $75,07\%$)**.

---

### 2.2. Nhóm Kiến trúc 2: Lai hóa Tuần tự Chia Khối Lớn (`S3: Block GA→PO` và `S4: Block PO→GA`)

Khác với cơ chế chuyển pha tần suất cao của `S1/S2`, kiến trúc Tuần tự Chia Khối phân tách toàn bộ ngân sách $T = 100$ vòng lặp thành hai giai đoạn độc lập nối tiếp nhau, mỗi giai đoạn chiếm đúng $50\%$ ngân sách ($50$ vòng lặp $= 1.000\text{ FEs}$) và chỉ thực hiện **duy nhất 1 lần chuyển giao quần thể** tại vòng lặp $t = 50$:

- **Biến thể `S3 — Block GA→PO` (`run_s3_block_ga_po`)**:
  1. **Giai đoạn 1 — Khối GA ($t = 1 \dots 50$, $1.000\text{ FEs}$)**: Giải thuật di truyền vận hành liên tục trong $50$ thế hệ đầu tiên để sàng lọc quần thể khởi tạo $P_0$, hình thành các khung tập con có độ bao phủ không gian đặc trưng ($\text{Cov}$) và độ cân bằng lớp ($\text{Bal}$) cao.
  2. **Giai đoạn 2 — Khối PO ($t = 51 \dots 100$, $1.000\text{ FEs}$)**: Nhận quần thể đã hội tụ sơ bộ từ GA tại $t = 50$, thuật toán Đàn vẹt thực hiện $50$ vòng lặp tinh chỉnh cục bộ quanh cá thể tốt nhất $X_{\text{gbest}}$ để loại bỏ thêm các mẫu ảnh dư thừa.
- **Biến thể `S4 — Block PO→GA` (`run_s4_block_po_ga`)**:
  1. **Giai đoạn 1 — Khối PO ($t = 1 \dots 50$, $1.000\text{ FEs}$)**: Tận dụng khả năng thăm dò toàn cục (Global Exploration) mạnh mẽ của bước nhảy Lévy và bốn hành vi bầy đàn trong $50$ vòng lặp đầu để nhanh chóng quét qua không gian $11.228$ chiều và đưa cả $20$ cá thể trong đàn về vùng có tỷ lệ nén và độ bao phủ tối ưu.
  2. **Giai đoạn 2 — Khối GA ($t = 51 \dots 100$, $1.000\text{ FEs}$)**: Nhận quần thể vẹt đã định vị đúng vùng mục tiêu tại $t = 50$, giải thuật GA thực hiện $50$ thế hệ lai ghép đồng nhất và đột biến thích nghi để tinh chỉnh tổ hợp mẫu giữa các cá thể.
- **Đặc trưng động lực học của `S3` và `S4`**:
  - Biến thể **`S4 (Block PO→GA)`** tuân thủ triết lý kinh điển trong thiết kế siêu mô phỏng lai (*"Thăm dò toàn cục trước bằng bầy đàn, Khai thác tổ hợp sau bằng di truyền"*). Khi GA nắm quyền điều khiển ở $50$ vòng lặp cuối, toán tử lai ghép đồng nhất và chọn lọc giải đấu giúp `S4` đạt độ cân bằng phân bố lớp cao nhất trong tất cả các thuật toán tối ưu hóa ($\text{Bal} = 0,9998$), qua đó giúp **`S4` đạt độ chính xác phân loại trung bình liên kiến trúc CNN cao nhất toàn cục ($88,87\%$, bảo toàn $98,84\%$ hiệu năng gốc)**.

---

### 2.3. Nhóm Kiến trúc 3a: Lai hóa Song song Độc lập và Hợp nhất (`P1: Parallel Merge`)

Trong các kiến trúc tuần tự (`S1–S4`), cả GA và PO đều tác động nối tiếp lên cùng một quần thể duy nhất nên quỹ đạo tìm kiếm ở giai đoạn sau phụ thuộc chặt chẽ vào trạng thái hội tụ của giai đoạn trước. Để khắc phục sự phụ thuộc đường đi này (Path Dependency), biến thể **`P1 (`run_p1_parallel_merge`)`** triển khai hai nhánh tiến hóa độc lập rồi hợp nhất và đồng tinh chỉnh dưới cùng ngân sách $2.000\text{ FEs}$:

1. **Giai đoạn 1 — Tiến hóa Song song Độc lập**:
   - Từ cùng quần thể khởi tạo dùng chung $P_0$ ($20$ cá thể), hệ thống tách thành hai nhánh riêng biệt: nhánh GA tiến hóa tạo ra $\text{Pop}_{\text{GA}}$ và nhánh PO tiến hóa tạo ra $\text{Pop}_{\text{PO}}$.
2. **Giai đoạn 2 — Hợp nhất và Sàng lọc Tinh hoa (Elite Pool Merging)**:
   - Gộp toàn bộ $20$ cá thể của nhánh GA và $20$ cá thể của nhánh PO thành một bể chứa chung gồm $40$ cá thể: $\text{Pop}_{\text{merged}} = \text{Pop}_{\text{GA}} \cup \text{Pop}_{\text{PO}}$.
   - Sắp xếp $40$ cá thể theo điểm Fitness giảm dần và trích xuất **$\text{Top-}20$ cá thể xuất sắc nhất** ($\text{Top-}P(\text{Pop}_{\text{merged}}, F)$) để tạo thành quần thể hợp nhất chứa đựng đồng thời các nghiệm tốt nhất từ cả hai cơ chế tiến hóa và bầy đàn, sau đó đồng tinh chỉnh trên quần thể hợp nhất.
   - Nhờ sự giao thoa giữa hai quỹ đạo độc lập, **`P1` tạo ra các tập con có độ bao phủ đặc trưng rất cao ($\text{Cov} = 0,8895$) và đạt độ chính xác kiểm thử trung bình cao nhất toàn cục trên mạng `ResNet-18` ($89,38 \pm 0,10\%$, giữ lại $99,02\%$ độ chính xác của tập dữ liệu gốc)**.

---

### 2.4. Nhóm Kiến trúc 3b: Lai hóa Hợp tác Đồng tiến hóa Trao đổi Tinh hoa Song phương (`C1: Cooperative Bidirectional Elite Exchange`)

Biến thể **`C1 (`run_c1_cooperative_exchange`)`** mô hình hóa cơ chế đồng tiến hóa đa quần thể, trong đó quần thể GA ($\text{Pop}_{\text{GA}}, P = 20$) và quần thể PO ($\text{Pop}_{\text{PO}}, P = 20$) cùng tiến hóa song hành và định kỳ trao đổi các cá thể tinh hoa qua lại sau mỗi chu kỳ $\Delta t = 5$ vòng lặp:

1. **Phân bổ Ngân sách Đồng tiến hóa theo Chặng (Interval Budget Allocation)**:
   - Tổng ngân sách $2.000\text{ FEs}$ (tương ứng $50$ vòng lặp trên mỗi quần thể $20$ cá thể) được chia thành $10$ chặng đồng tiến hóa ($\Delta t = 5$ vòng lặp/chặng).
   - Trong mỗi chặng, quần thể GA chạy $5$ thế hệ ($100\text{ FEs}$) và quần thể PO chạy $5$ vòng lặp ($100\text{ FEs}$).
2. **Cơ chế Trao đổi Tinh hoa Song phương và Kiểm soát Trùng lặp (Duplicate Control)**:
   - Kết thúc mỗi chặng $\Delta t = 5$ vòng lặp, hai nhánh trao đổi các cá thể ưu tú nhất (Elites) sang quần thể đối diện.
   - Để ngăn hiện tượng hai quần thể bị đồng hóa sớm do trao đổi lặp lại cùng một nghiệm cực đại, trước khi đưa cá thể ưu tú $e$ vào quần thể đích $\text{Pop}_{\text{target}}$, hệ thống kiểm tra điều kiện khoảng cách Hamming $\min_{Y \in \text{Pop}_{\text{target}}} \|e - Y\|_1 > 0$ (thực thi qua phép lọc nghiệm phân biệt `np.unique(..., axis=0)`). Chỉ những cá thể ưu tú chưa tồn tại trong quần thể đích mới được phép di cư. Nhờ cơ chế này, **`C1` đạt độ ổn định thống kê cao nhất trong toàn bộ 6 biến thể lai với độ lệch chuẩn Fitness gần như bằng không ($\pm 0,0001$) và độ bao phủ không gian đặc trưng cao nhất nhóm Hybrid ($\text{Cov} = 0,8896$)**.

### 2.5. Bảng Tổng hợp Đặc tả Kỹ thuật của 6 Biến thể Hybrid PO–GA và 4 Phương pháp Đối chuẩn

| Mã phương pháp | Nhóm kiến trúc | Quy mô quần thể | Phân bổ vòng lặp / Ngân sách FEs | Số lần chuyển giao / trao đổi | Cơ chế cốt lõi và Đặc trưng tìm kiếm |
| :---: | :--- | :---: | :--- | :---: | :--- |
| **B0 — Full Dataset** | Mốc chuẩn trên | — | $0\text{ FEs}$ ($100\%$ dữ liệu gốc, $N=11.228$) | $0$ | Huấn luyện trực tiếp trên toàn bộ $11.228$ ảnh; làm mốc tham chiếu $\text{ARR}=100\%$ và $\text{Speedup}=1,00\times$. |
| **B1 — Random** | Đối chuẩn ngẫu nhiên | — | $0\text{ FEs}$ (Lấy mẫu phân bố đều tại $M=2.978$) | $0$ | Lấy ngẫu nhiên $2.978$ ảnh (bằng trung vị kích thước của 8 thuật toán tối ưu hóa) không phân biệt lớp. |
| **B2 — Stratified** | Đối chuẩn phân tầng | — | $0\text{ FEs}$ (Lấy mẫu phân tầng tại $M=2.978$) | $0$ | Lấy ngẫu nhiên $2.978$ ảnh nhưng giữ nguyên tỷ lệ phân bố của $6$ lớp ($\text{Bal} = 1,0000$). |
| **B3 — GA-only** | Tiến hóa đơn lẻ | $P = 20$ | $100$ thế hệ GA $= 2.000\text{ FEs}$ | $0$ | Elitism ($E=5$), chọn lọc giải đấu ($k=4$), lai ghép đồng nhất ($p_c=0,8$), đột biến thích nghi giảm dần. |
| **B4 — PO-only** | Bầy đàn đơn lẻ | $P = 20$ | $100$ vòng lặp PO $= 2.000\text{ FEs}$ | $0$ | $4$ hành vi đàn vẹt truyền thống, ánh xạ nhị phân $\lvert\tanh(\Delta x)\rvert$ + chọn lọc tham lam. |
| **S1 — Alt GA→PO** | Tuần tự xen kẽ | $P = 20$ | $5 \times [\text{GA}(10) \to \text{PO}(10)] = 2.000\text{ FEs}$ | $9$ lần | Khởi phát bằng tái tổ hợp GA ($10$ vòng), nối tiếp bằng tinh chỉnh PO ($10$ vòng) lặp lại $5$ chu kỳ. |
| **S2 — Alt PO→GA** | Tuần tự xen kẽ | $P = 20$ | $5 \times [\text{PO}(10) \to \text{GA}(10)] = 2.000\text{ FEs}$ | $9$ lần | Khởi phát bằng thăm dò bầy đàn PO ($10$ vòng), nối tiếp bằng lai ghép tái tổ hợp GA ($10$ vòng) lặp lại $5$ chu kỳ. |
| **S3 — Block GA→PO** | Tuần tự chia khối | $P = 20$ | $\text{GA}(50) \to \text{PO}(50) = 2.000\text{ FEs}$ | $1$ lần | Dùng GA định hình khung tập con trong $50$ thế hệ đầu, sau đó dùng PO tinh chỉnh cục bộ trong $50$ vòng cuối. |
| **S4 — Block PO→GA** | Tuần tự chia khối | $P = 20$ | $\text{PO}(50) \to \text{GA}(50) = 2.000\text{ FEs}$ | $1$ lần | Dùng PO thăm dò không gian rộng trong $50$ vòng đầu, sau đó dùng GA lai ghép và cân bằng lớp trong $50$ vòng cuối. |
| **P1 — Parallel** | Song song hợp nhất | $2 \times 20 \to 20$ | $\text{GA}(50\%) \parallel \text{PO}(50\%) \to \text{Merge Top-}20$ | $1$ lần hợp nhất | Hai nhánh tiến hóa độc lập, hợp nhất chọn lọc tinh hoa $\text{Top-}20$ cá thể và đồng tinh chỉnh. |
| **C1 — Cooperative** | Hợp tác đồng tiến hóa | $2 \times 20$ | $10 \text{ chặng} \times [\text{GA}(5) \leftrightarrow \text{PO}(5)] = 2.000\text{ FEs}$ | $10$ đợt di cư song phương | Trao đổi chéo các cá thể ưu tú sau mỗi $5$ vòng lặp kèm bộ lọc chống trùng lặp khoảng cách Hamming. |

---

## PHẦN 3: KẾT QUẢ THỰC NGHIỆM TẦNG 1 — CHẤT LƯỢNG TỐI ƯU HÓA VÀ ĐỘNG LỰC HỘI TỤ

### 3.1. Thống kê Mô tả Đa Hạt giống (`Seeds = [42, 123, 456]`) trên Không gian $N = 11.228$ Ảnh

Bảng 3.1 trình bày đầy đủ kết quả thống kê mô tả của 10 phương pháp rút gọn dữ liệu từ tệp [`outputs/run_poga_all_20261002_173334/descriptive_statistics.csv`](file:///d:/Code/KhoaLuanDatasetReduction/outputs/run_poga_all_20261002_173334/descriptive_statistics.csv), bao gồm số lượng ảnh giữ lại $M$, tỷ lệ rút gọn dữ liệu ($\text{DR}\%$), điểm Fitness trung bình kèm độ lệch chuẩn ($\text{Mean} \pm \text{Std}$), trung vị kèm khoảng tứ phân vị ($\text{Median} \text{ [IQR]}$), bốn thành phần mục tiêu ($\text{Div}, \text{Cov}, \text{Bal}, \text{Com}$) và thời gian chạy thuật toán tối ưu hóa trên GPU.

**Bảng 3.1: Tổng hợp Chất lượng Tối ưu hóa (Tầng 1) qua 3 hạt giống độc lập dưới ngân sách 2.000 FEs (`run_poga_all_20261002_173334`)**

| Hạng Fitness | Phương pháp | Kiến trúc | Số ảnh giữ lại $M$ ($\text{Mean} \pm \text{Std}$) | Tỷ lệ rút gọn $\text{DR}$ (%) | Fitness ($\text{Mean} \pm \text{Std}$) | Fitness ($\text{Median} \text{ [IQR]}$) | $\text{Div}$ | $\text{Cov}$ | $\text{Bal}$ | $\text{Com}$ | Thời gian Tối ưu (s) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **S2 — Alt PO→GA** | Sequential (Xen kẽ) | $2.828,3 \pm 61,6$ | $74,81\%$ | **$0,8249 \pm 0,0005$** | **$0,8248 \text{ [0,0006]}$** | **$0,5094$** | $0,8867$ | $0,9993$ | $0,7481$ | $9,64 \pm 0,11\text{ s}$ |
| **2** | **S1 — Alt GA→PO** | Sequential (Xen kẽ) | **$2.799,0 \pm 37,6$** | **$75,07\%$** | $0,8248 \pm 0,0003$ | $0,8247 \text{ [0,0003]}$ | $0,5075$ | $0,8861$ | $0,9992$ | **$0,7507$** | $9,54 \pm 0,05\text{ s}$ |
| **3** | **S3 — Block GA→PO** | Sequential (Khối) | $2.949,0 \pm 24,1$ | $73,74\%$ | $0,8232 \pm 0,0002$ | $0,8232 \text{ [0,0002]}$ | $0,5067$ | $0,8886$ | $0,9992$ | $0,7374$ | $9,66 \pm 0,05\text{ s}$ |
| **4** | **S4 — Block PO→GA** | Sequential (Khối) | $2.963,3 \pm 35,6$ | $73,61\%$ | $0,8232 \pm 0,0003$ | $0,8231 \text{ [0,0004]}$ | $0,5071$ | $0,8887$ | **$0,9998$** | $0,7361$ | $9,85 \pm 0,03\text{ s}$ |
| **5** | **B3 — GA-only** | Tiến hóa đơn lẻ | $2.969,3 \pm 15,1$ | $73,55\%$ | $0,8231 \pm 0,0001$ | $0,8230 \text{ [0,0002]}$ | $0,5068$ | $0,8889$ | **$0,9998$** | $0,7355$ | **$9,26 \pm 0,03\text{ s}$** |
| **6** | **C1 — Coop Exchange** | Hợp tác đồng tiến hóa | $3.012,0 \pm 25,5$ | $73,18\%$ | $0,8226 \pm 0,0001$ | $0,8226 \text{ [0,0001]}$ | $0,5074$ | **$0,8896$** | $0,9995$ | $0,7318$ | $10,58 \pm 0,09\text{ s}$ |
| **7** | **P1 — Parallel Merge**| Song song hợp nhất | $3.007,0 \pm 37,6$ | $73,22\%$ | $0,8225 \pm 0,0003$ | $0,8226 \text{ [0,0003]}$ | $0,5065$ | $0,8895$ | $0,9991$ | $0,7322$ | $9,84 \pm 0,06\text{ s}$ |
| **8** | **B2 — Stratified** | Ngẫu nhiên phân tầng | $2.978,0 \pm 0,0$ | $73,48\%$ | $0,8220 \pm 0,0001$ | $0,8220 \text{ [0,0002]}$ | $0,5021$ | $0,8883$ | $1,0000$ | $0,7348$ | $< 0,01\text{ s}$ |
| **9** | **B4 — PO-only** | Bầy đàn đơn lẻ | $3.064,3 \pm 22,2$ | $72,71\%$ | $0,8216 \pm 0,0003$ | $0,8214 \text{ [0,0003]}$ | $0,5063$ | **$0,8904$** | $0,9979$ | $0,7271$ | $10,52 \pm 0,04\text{ s}$ |
| **10** | **B1 — Random** | Ngẫu nhiên đều | $2.978,0 \pm 0,0$ | $73,48\%$ | $0,8209 \pm 0,0005$ | $0,8210 \text{ [0,0007]}$ | $0,5023$ | $0,8884$ | $0,9940$ | $0,7348$ | $< 0,01\text{ s}$ |

---

### 3.2. Chi tiết Kết quả Tối ưu hóa theo Từng Hạt giống (`Seed 42, 123, 456`)

Để kiểm chứng tính ổn định của các thuật toán giữa các lần khởi tạo ngẫu nhiên khác nhau, Bảng 3.2 liệt kê kết quả từng lần chạy độc lập của cả 10 phương pháp.

**Bảng 3.2: Bảng phân rã chi tiết kết quả tối ưu hóa theo từng hạt giống (`Seed 42`, `Seed 123`, `Seed 456`)**

| Phương pháp | Seed | Số ảnh chọn ($M$) | Tỷ lệ rút gọn $\text{DR}$ (%) | Tổng Fitness ($F$) | $\text{Div}$ | $\text{Cov}$ | $\text{Bal}$ | $\text{Com}$ | Thời gian tối ưu (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1 — Alt GA→PO** | `42` | $2.801$ | $75,05\%$ | $0,8247$ | $0,5072$ | $0,8862$ | $0,9989$ | $0,7505$ | $9,59\text{ s}$ |
| **S1 — Alt GA→PO** | `123` | $2.752$ | $75,49\%$ | $0,8252$ | $0,5081$ | $0,8850$ | $0,9991$ | $0,7549$ | $9,48\text{ s}$ |
| **S1 — Alt GA→PO** | `456` | $2.844$ | $74,67\%$ | $0,8246$ | $0,5073$ | $0,8872$ | $0,9996$ | $0,7467$ | $9,56\text{ s}$ |
| **S2 — Alt PO→GA** | `42` | **$2.746$** | **$75,54\%$** | **$0,8256$** | $0,5085$ | $0,8852$ | $0,9993$ | **$0,7554$** | $9,50\text{ s}$ |
| **S2 — Alt PO→GA** | `123` | $2.894$ | $74,23\%$ | $0,8243$ | **$0,5105$** | $0,8878$ | $0,9990$ | $0,7423$ | $9,77\text{ s}$ |
| **S2 — Alt PO→GA** | `456` | $2.845$ | $74,66\%$ | $0,8248$ | $0,5093$ | $0,8871$ | $0,9997$ | $0,7466$ | $9,64\text{ s}$ |
| **S3 — Block GA→PO** | `42` | $2.923$ | $73,97\%$ | $0,8234$ | $0,5066$ | $0,8880$ | $0,9991$ | $0,7397$ | $9,60\text{ s}$ |
| **S3 — Block GA→PO** | `123` | $2.943$ | $73,79\%$ | $0,8232$ | $0,5057$ | $0,8885$ | $0,9996$ | $0,7379$ | $9,66\text{ s}$ |
| **S3 — Block GA→PO** | `456` | $2.981$ | $73,45\%$ | $0,8230$ | $0,5078$ | $0,8892$ | $0,9988$ | $0,7345$ | $9,73\text{ s}$ |
| **S4 — Block PO→GA** | `42` | $2.913$ | $74,06\%$ | $0,8236$ | $0,5066$ | $0,8878$ | **$0,9998$** | $0,7406$ | $9,89\text{ s}$ |
| **S4 — Block PO→GA** | `123` | $2.987$ | $73,40\%$ | $0,8231$ | $0,5088$ | $0,8890$ | **$0,9998$** | $0,7340$ | $9,81\text{ s}$ |
| **S4 — Block PO→GA** | `456` | $2.990$ | $73,37\%$ | $0,8228$ | $0,5060$ | $0,8893$ | **$0,9998$** | $0,7337$ | $9,84\text{ s}$ |
| **P1 — Parallel Merge**| `42` | $2.961$ | $73,63\%$ | $0,8226$ | $0,5038$ | $0,8886$ | $0,9996$ | $0,7363$ | $9,78\text{ s}$ |
| **P1 — Parallel Merge**| `123` | $3.007$ | $73,22\%$ | $0,8227$ | $0,5084$ | $0,8894$ | $0,9989$ | $0,7322$ | $9,82\text{ s}$ |
| **P1 — Parallel Merge**| `456` | $3.053$ | $72,81\%$ | $0,8221$ | $0,5072$ | **$0,8905$** | $0,9987$ | $0,7281$ | $9,92\text{ s}$ |
| **C1 — Coop Exchange** | `42` | $2.977$ | $73,49\%$ | $0,8228$ | $0,5059$ | $0,8890$ | $0,9997$ | $0,7349$ | $10,71\text{ s}$ |
| **C1 — Coop Exchange** | `123` | $3.037$ | $72,95\%$ | $0,8226$ | $0,5087$ | $0,8900$ | $0,9994$ | $0,7295$ | $10,53\text{ s}$ |
| **C1 — Coop Exchange** | `456` | $3.022$ | $73,09\%$ | $0,8225$ | $0,5076$ | $0,8897$ | $0,9993$ | $0,7309$ | $10,51\text{ s}$ |
| **B3 — GA-only** | `42` | $2.948$ | $73,74\%$ | $0,8233$ | $0,5070$ | $0,8884$ | $0,9999$ | $0,7374$ | $9,28\text{ s}$ |
| **B3 — GA-only** | `123` | $2.979$ | $73,47\%$ | $0,8230$ | $0,5067$ | $0,8891$ | $0,9996$ | $0,7347$ | $9,28\text{ s}$ |
| **B3 — GA-only** | `456` | $2.981$ | $73,45\%$ | $0,8230$ | $0,5067$ | $0,8891$ | $0,9999$ | $0,7345$ | $9,22\text{ s}$ |
| **B4 — PO-only** | `42` | $3.082$ | $72,55\%$ | $0,8214$ | $0,5055$ | $0,8906$ | $0,9985$ | $0,7255$ | $10,54\text{ s}$ |
| **B4 — PO-only** | `123` | $3.078$ | $72,59\%$ | $0,8214$ | $0,5057$ | $0,8908$ | $0,9976$ | $0,7259$ | $10,46\text{ s}$ |
| **B4 — PO-only** | `456` | $3.033$ | $72,99\%$ | $0,8220$ | $0,5077$ | $0,8898$ | $0,9975$ | $0,7299$ | $10,55\text{ s}$ |
| **B1 — Random** | `42 / 123 / 456` | $2.978$ | $73,48\%$ | $0,8210 / 0,8202 / 0,8215$ | $0,5023$ | $0,8884$ | $0,9940$ | $0,7348$ | $< 0,01\text{ s}$ |
| **B2 — Stratified** | `42 / 123 / 456` | $2.978$ | $73,48\%$ | $0,8220 / 0,8218 / 0,8221$ | $0,5021$ | $0,8883$ | $1,0000$ | $0,7348$ | $< 0,01\text{ s}$ |

---

### 3.3. Phân tích Động lực Hội tụ và Đối sánh Tầng 1

Từ số liệu thực nghiệm tại Bảng 3.1, Bảng 3.2 và biểu đồ hội tụ [`outputs/run_poga_all_20261002_173334/plots/convergence_comparison.png`](file:///d:/Code/KhoaLuanDatasetReduction/outputs/run_poga_all_20261002_173334/plots/convergence_comparison.png), có thể rút ra bốn nhận xét kỹ thuật quan trọng:

1. **Tất cả 6 biến thể lai Hybrid PO–GA và thuật toán GA-only đều vượt qua toàn bộ các phương pháp đối chuẩn (`Random B1`, `Stratified B2`, `PO-only B4`) về điểm mục tiêu tổng hợp**:
   - Đứng đầu bảng xếp hạng Fitness là hai cấu hình lai tuần tự xen kẽ **`S2 (Alt PO→GA)`** ($0,8249 \pm 0,0005$, đạt đỉnh $0,8256$ tại `Seed 42`) và **`S1 (Alt GA→PO)`** ($0,8248 \pm 0,0003$), tiếp theo là hai cấu hình lai chia khối **`S3 (Block GA→PO)`** ($0,8232 \pm 0,0002$) và **`S4 (Block PO→GA)`** ($0,8232 \pm 0,0003$), 모두 vượt trên `GA-only` ($0,8231$), `C1` ($0,8226$), `P1` ($0,8225$), `Stratified Random` ($0,8220$), `PO-only` ($0,8216$) và `Random Selection` ($0,8209$).
2. **Ưu thế vượt trội về độ đa dạng nội tập con ($\text{Div}$) của các thuật toán tối ưu hóa so với lấy mẫu ngẫu nhiên**:
   - Tại cùng vùng kích thước tập con $M \approx 2.800 - 3.000$ ảnh, cả hai phương pháp lấy mẫu ngẫu nhiên `B1` và `B2` chỉ đạt độ đa dạng $\text{Div} = 0,5021 - 0,5023$ do chọn phải nhiều bức ảnh có góc chụp tương đồng nhau trong cùng một lớp. Ngược lại, mọi thuật toán tối ưu hóa đều nâng $\text{Div}$ lên vùng $0,5065 - 0,5094$ (cao nhất ở `S2` với $\text{Div} = 0,5094$, đạt $0,5105$ tại `Seed 123`), đồng thời tăng độ bao phủ $\text{Cov}$ lên $0,8861 - 0,8904$.
3. **Hiệu ứng cộng hưởng của cơ chế chuyển pha trong `S1`, `S2`, `S3`, `S4`**:
   - Biểu đồ hội tụ `convergence_comparison.png` cho thấy: trong khi đường cong của `PO-only` tăng nhanh ở $25$ vòng đầu rồi đi ngang quanh mức $0,8216$ (do thiếu toán tử lai ghép gen chéo giữa các cá thể), thì tại `S4 (Block PO→GA)`, ngay tại mốc vòng lặp $t = 50$ khi quần thể PO được bàn giao sang cho GA, đường cong Fitness lập tức bứt phá mạnh từ $0,8213$ vọt lên $0,8232$. Tương tự, tại `S1` và `S2`, mỗi chu kỳ chuyển giao $10$ vòng lặp tạo ra các nấc thang tăng trưởng liên tục giúp đẩy tỷ lệ rút gọn lên tới **$75,07\%$ ($M = 2.799$ ảnh)** và **$74,81\%$ ($M = 2.828$ ảnh)**.
4. **Tốc độ thực thi tối ưu hóa cực nhanh trên GPU**:
   - Nhờ tiền tính toán ma trận khoảng cách chuẩn hóa trên VRAM và vector hóa toàn bộ phép đánh giá ma trận con, mỗi lần chạy tối ưu hóa $2.000\text{ FEs}$ trên không gian $11.228$ chiều chỉ mất từ **$9,26\text{ giây}$ đến $10,58\text{ giây}$** với độ lệch chuẩn thời gian giữa các hạt giống dưới $0,11\text{ giây}$.

---

## PHẦN 4: KẾT QUẢ THỰC NGHIỆM TẦNG 2 — HIỆU NĂNG PHÂN LOẠI TRÊN 3 KIẾN TRÚC CNN

### 4.1. Bảng Đối sánh Toàn diện Hiệu năng Phân loại Hạ nguồn (`ResNet-18`, `MobileNetV3-Small`, `DenseNet-121`)

Sau khi kết thúc giai đoạn tối ưu hóa Tầng 1, mỗi tập con tốt nhất (`best_sol`) từ 3 hạt giống (`Seeds = 42, 123, 456`) của 10 phương pháp được đưa vào huấn luyện độc lập 3 mạng học sâu trong $5$ epochs ($\text{Adam}, \text{lr} = 10^{-3}, \text{batch\_size} = 64$) và đánh giá trên tập kiểm thử độc lập `seg_test` ($3.000$ ảnh). Bảng 4.1 tổng hợp toàn bộ kết quả từ [`outputs/run_poga_all_20261002_173334/descriptive_statistics.csv`](file:///d:/Code/KhoaLuanDatasetReduction/outputs/run_poga_all_20261002_173334/descriptive_statistics.csv).

**Bảng 4.1: Kết quả Đánh giá Tầng 2 trên Tập Kiểm thử Độc lập (`seg_test` — 3.000 ảnh) qua 3 kiến trúc CNN**

| Phương pháp | Mô hình CNN | Số ảnh $M$ | Tỷ lệ rút gọn $\text{DR}$ (%) | Val Acc (%) | Test Accuracy ($\text{Mean} \pm \text{Std} \%$) | Test Accuracy ($\text{Median} \text{ [IQR]} \%$) | Macro F1-Score ($\text{Mean} \pm \text{Std} \%$) | Duy trì $\text{ARR}$ (%) | Độ lệch $\Delta\text{Acc}$ (%) | Tăng tốc $\text{Speedup}$ ($\times$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Dataset (B0)** | `resnet18` | $11.228$ | $0,00\%$ | $90,34\%$ | $90,27 \pm 0,00\%$ | $90,27 \text{ [0,00]}\%$ | $90,41 \pm 0,00\%$ | $100,00\%$ | $+0,00\%$ | $1,00\times$ |
| **Full Dataset (B0)** | `mobilenet_v3` | $11.228$ | $0,00\%$ | $90,34\%$ | $88,93 \pm 0,00\%$ | $88,93 \text{ [0,00]}\%$ | $89,25 \pm 0,00\%$ | $100,00\%$ | $+0,00\%$ | $1,00\times$ |
| **Full Dataset (B0)** | `densenet121` | $11.228$ | $0,00\%$ | $91,80\%$ | $90,57 \pm 0,00\%$ | $90,57 \text{ [0,00]}\%$ | $90,72 \pm 0,00\%$ | $100,00\%$ | $+0,00\%$ | $1,00\times$ |
| **S4 — Block PO→GA**| `resnet18` | $2.963$ | $73,61\%$ | $90,32\%$ | $89,24 \pm 0,32\%$ | $89,13 \text{ [0,38]}\%$ | $89,40 \pm 0,31\%$ | $98,87\%$ | $-1,03\%$ | $13,29\times$ |
| **S4 — Block PO→GA**| `mobilenet_v3` | $2.963$ | $73,61\%$ | $88,75\%$ | **$88,02 \pm 0,34\%$** | $88,13 \text{ [0,40]}\%$ | **$88,36 \pm 0,37\%$** | **$98,98\%$** | **$-0,91\%$** | $12,35\times$ |
| **S4 — Block PO→GA**| `densenet121` | $2.963$ | $73,61\%$ | $90,35\%$ | $89,36 \pm 0,43\%$ | **$89,53 \text{ [0,50]}\%$** | $89,48 \pm 0,47\%$ | $98,66\%$ | $-1,21\%$ | $11,92\times$ |
| **P1 — Parallel Merge**| `resnet18` | $3.007$ | $73,22\%$ | **$90,46\%$** | **$89,38 \pm 0,10\%$** | **$89,37 \text{ [0,12]}\%$** | **$89,58 \pm 0,07\%$** | **$99,02\%$** | **$-0,89\%$** | $13,10\times$ |
| **P1 — Parallel Merge**| `mobilenet_v3` | $3.007$ | $73,22\%$ | $88,43\%$ | $87,74 \pm 0,34\%$ | $87,53 \text{ [0,38]}\%$ | $88,04 \pm 0,36\%$ | $98,67\%$ | $-1,19\%$ | $12,17\times$ |
| **P1 — Parallel Merge**| `densenet121` | $3.007$ | $73,22\%$ | $90,47\%$ | $89,13 \pm 0,43\%$ | $89,03 \text{ [0,52]}\%$ | $89,36 \pm 0,36\%$ | $98,42\%$ | $-1,44\%$ | $11,83\times$ |
| **B3 — GA-only** | `resnet18` | $2.969$ | $73,55\%$ | $90,18\%$ | $89,14 \pm 0,36\%$ | $89,07 \text{ [0,43]}\%$ | $89,29 \pm 0,41\%$ | $98,76\%$ | $-1,13\%$ | $13,12\times$ |
| **B3 — GA-only** | `mobilenet_v3` | $2.969$ | $73,55\%$ | $88,52\%$ | $87,64 \pm 0,54\%$ | $87,40 \text{ [0,63]}\%$ | $87,93 \pm 0,52\%$ | $98,55\%$ | $-1,29\%$ | $12,48\times$ |
| **B3 — GA-only** | `densenet121` | $2.969$ | $73,55\%$ | $90,64\%$ | **$89,37 \pm 0,21\%$** | $89,33 \text{ [0,25]}\%$ | **$89,58 \pm 0,23\%$** | **$98,67\%$** | **$-1,20\%$** | $11,94\times$ |
| **C1 — Coop Exchange**| `resnet18` | $3.012$ | $73,18\%$ | $90,06\%$ | $89,36 \pm 0,14\%$ | **$89,40 \text{ [0,17]}\%$** | $89,54 \pm 0,08\%$ | $98,99\%$ | $-0,91\%$ | $13,07\times$ |
| **C1 — Coop Exchange**| `mobilenet_v3` | $3.012$ | $73,18\%$ | $88,30\%$ | $87,40 \pm 0,23\%$ | $87,53 \text{ [0,25]}\%$ | $87,74 \pm 0,27\%$ | $98,28\%$ | $-1,53\%$ | $12,26\times$ |
| **C1 — Coop Exchange**| `densenet121` | $3.012$ | $73,18\%$ | $90,46\%$ | $89,33 \pm 0,47\%$ | $89,17 \text{ [0,55]}\%$ | $89,52 \pm 0,45\%$ | $98,64\%$ | $-1,24\%$ | $11,66\times$ |
| **B4 — PO-only** | `resnet18` | $3.064$ | $72,71\%$ | $89,77\%$ | $88,92 \pm 0,14\%$ | $89,00 \text{ [0,15]}\%$ | $89,10 \pm 0,14\%$ | $98,51\%$ | $-1,35\%$ | $12,81\times$ |
| **B4 — PO-only** | `mobilenet_v3` | $3.064$ | $72,71\%$ | $88,43\%$ | $87,96 \pm 0,40\%$ | **$88,17 \text{ [0,45]}\%$** | $88,24 \pm 0,40\%$ | $98,90\%$ | $-0,97\%$ | $12,10\times$ |
| **B4 — PO-only** | `densenet121` | $3.064$ | $72,71\%$ | $90,34\%$ | $89,10 \pm 0,23\%$ | $89,00 \text{ [0,27]}\%$ | $89,32 \pm 0,22\%$ | $98,38\%$ | $-1,47\%$ | $11,57\times$ |
| **B2 — Stratified** | `resnet18` | $2.978$ | $73,48\%$ | $90,02\%$ | $88,83 \pm 0,24\%$ | $88,93 \text{ [0,28]}\%$ | $88,97 \pm 0,26\%$ | $98,41\%$ | $-1,44\%$ | $12,88\times$ |
| **B2 — Stratified** | `mobilenet_v3` | $2.978$ | $73,48\%$ | $88,52\%$ | $87,64 \pm 0,49\%$ | $87,93 \text{ [0,55]}\%$ | $87,99 \pm 0,46\%$ | $98,55\%$ | $-1,29\%$ | $12,26\times$ |
| **B2 — Stratified** | `densenet121` | $2.978$ | $73,48\%$ | $90,31\%$ | $89,34 \pm 0,68\%$ | $89,37 \text{ [0,83]}\%$ | $89,45 \pm 0,72\%$ | $98,65\%$ | $-1,23\%$ | $11,72\times$ |
| **S3 — Block GA→PO**| `resnet18` | $2.949$ | $73,74\%$ | $89,80\%$ | $89,03 \pm 0,15\%$ | $89,07 \text{ [0,18]}\%$ | $89,22 \pm 0,14\%$ | $98,63\%$ | $-1,24\%$ | $13,26\times$ |
| **S3 — Block GA→PO**| `mobilenet_v3` | $2.949$ | $73,74\%$ | $88,41\%$ | $87,52 \pm 0,22\%$ | $87,47 \text{ [0,27]}\%$ | $87,86 \pm 0,20\%$ | $98,42\%$ | $-1,41\%$ | $12,48\times$ |
| **S3 — Block GA→PO**| `densenet121` | $2.949$ | $73,74\%$ | $90,05\%$ | $89,14 \pm 0,20\%$ | $89,17 \text{ [0,25]}\%$ | $89,37 \pm 0,14\%$ | $98,43\%$ | $-1,43\%$ | $11,93\times$ |
| **S2 — Alt PO→GA** | `resnet18` | $2.828$ | $74,81\%$ | $89,80\%$ | $89,16 \pm 0,04\%$ | $89,13 \text{ [0,05]}\%$ | $89,34 \pm 0,08\%$ | $98,77\%$ | $-1,11\%$ | $13,82\times$ |
| **S2 — Alt PO→GA** | `mobilenet_v3` | $2.828$ | $74,81\%$ | $88,37\%$ | $87,52 \pm 0,17\%$ | $87,43 \text{ [0,20]}\%$ | $87,88 \pm 0,13\%$ | $98,42\%$ | $-1,41\%$ | $12,96\times$ |
| **S2 — Alt PO→GA** | `densenet121` | $2.828$ | $74,81\%$ | $90,15\%$ | $88,97 \pm 0,68\%$ | $89,23 \text{ [0,78]}\%$ | $89,17 \pm 0,64\%$ | $98,23\%$ | $-1,60\%$ | $12,35\times$ |
| **B1 — Random** | `resnet18` | $2.978$ | $73,48\%$ | $89,65\%$ | $88,71 \pm 0,14\%$ | $88,73 \text{ [0,17]}\%$ | $88,89 \pm 0,09\%$ | $98,28\%$ | $-1,56\%$ | $12,42\times$ |
| **B1 — Random** | `mobilenet_v3` | $2.978$ | $73,48\%$ | $88,18\%$ | $87,64 \pm 0,53\%$ | $87,90 \text{ [0,60]}\%$ | $87,98 \pm 0,50\%$ | $98,55\%$ | $-1,29\%$ | $12,18\times$ |
| **B1 — Random** | `densenet121` | $2.978$ | $73,48\%$ | $90,34\%$ | $89,16 \pm 0,48\%$ | $89,17 \text{ [0,58]}\%$ | $89,33 \pm 0,48\%$ | $98,44\%$ | $-1,41\%$ | $11,64\times$ |
| **S1 — Alt GA→PO** | `resnet18` | **$2.799$** | **$75,07\%$** | $89,58\%$ | $88,71 \pm 0,31\%$ | $88,73 \text{ [0,38]}\%$ | $88,94 \pm 0,30\%$ | $98,28\%$ | $-1,56\%$ | **$13,86\times$** |
| **S1 — Alt GA→PO** | `mobilenet_v3` | **$2.799$** | **$75,07\%$** | $88,52\%$ | $87,31 \pm 0,39\%$ | $87,57 \text{ [0,43]}\%$ | $87,67 \pm 0,35\%$ | $98,18\%$ | $-1,62\%$ | **$13,07\times$** |
| **S1 — Alt GA→PO** | `densenet121` | **$2.799$** | **$75,07\%$** | $90,14\%$ | $89,03 \pm 0,07\%$ | $89,00 \text{ [0,08]}\%$ | $89,22 \pm 0,10\%$ | $98,30\%$ | $-1,54\%$ | **$12,48\times$** |

---

### 4.2. Bảng Xếp hạng Tổng hợp Liên Kiến trúc (Cross-Architecture Leaderboard)

Để đánh giá năng lực tổng quát hóa xuyên kiến trúc (Cross-Architecture Generalization) khi tập con được chọn một lần duy nhất nhưng phục vụ huấn luyện nhiều họ mạng CNN khác nhau, Bảng 4.2 tổng hợp giá trị trung bình cộng trên cả 3 mô hình (`ResNet-18`, `MobileNetV3-Small`, `DenseNet-121` — tương đương $3\text{ seeds} \times 3\text{ models} = 9$ phiên huấn luyện độc lập cho mỗi phương pháp).

**Bảng 4.2: Bảng xếp hạng tổng hợp hiệu năng phân loại trung bình trên cả 3 kiến trúc CNN (9 lần huấn luyện/phương pháp)**

| Hạng Chung cuộc | Phương pháp | Nhóm Kiến trúc | Số ảnh giữ lại $M$ | Tỷ lệ rút gọn $\text{DR}$ (%) | ResNet-18 Test Acc (%) | MobileNetV3 Test Acc (%) | DenseNet-121 Test Acc (%) | Trung bình 3 Mạng CNN (%) | Trung bình Macro F1 (%) | Duy trì $\text{ARR}$ Trung bình (%) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mốc B0** | **Full Dataset** | Mốc chuẩn trên | $11.228$ | $0,00\%$ | $90,27\%$ | $88,93\%$ | $90,57\%$ | **$89,92\%$** | **$90,13\%$** | **$100,00\%$** |
| **1** | **S4 — Block PO→GA** | **Hybrid (Khối)** | $2.963,3$ | $73,61\%$ | $89,24\%$ | **$88,02\%$** | $89,36\%$ | **$88,87\%$** | **$89,08\%$** | **$98,84\%$** |
| **2** | **P1 — Parallel Merge** | **Hybrid (Song song)** | $3.007,0$ | $73,22\%$ | **$89,38\%$** | $87,74\%$ | $89,13\%$ | **$88,75\%$** | **$88,99\%$** | **$98,70\%$** |
| **3** | **B3 — GA-only** | Đơn lẻ (Tiến hóa) | $2.969,3$ | $73,55\%$ | $89,14\%$ | $87,64\%$ | **$89,37\%$** | $88,72\%$ | $88,93\%$ | $98,66\%$ |
| **4** | **C1 — Coop Exchange** | **Hybrid (Hợp tác)** | $3.012,0$ | $73,18\%$ | $89,36\%$ | $87,40\%$ | $89,33\%$ | $88,70\%$ | $88,93\%$ | $98,64\%$ |
| **5** | **B4 — PO-only** | Đơn lẻ (Bầy đàn) | $3.064,3$ | $72,71\%$ | $88,92\%$ | $87,96\%$ | $89,10\%$ | $88,66\%$ | $88,89\%$ | $98,60\%$ |
| **6** | **B2 — Stratified** | Ngẫu nhiên phân tầng | $2.978,0$ | $73,48\%$ | $88,83\%$ | $87,64\%$ | $89,34\%$ | $88,60\%$ | $88,80\%$ | $98,54\%$ |
| **7** | **S3 — Block GA→PO** | **Hybrid (Khối)** | $2.949,0$ | $73,74\%$ | $89,03\%$ | $87,52\%$ | $89,14\%$ | $88,56\%$ | $88,82\%$ | $98,49\%$ |
| **8** | **S2 — Alt PO→GA** | **Hybrid (Xen kẽ)** | $2.828,3$ | $74,81\%$ | $89,16\%$ | $87,52\%$ | $88,97\%$ | $88,55\%$ | $88,80\%$ | $98,47\%$ |
| **9** | **B1 — Random** | Ngẫu nhiên đều | $2.978,0$ | $73,48\%$ | $88,71\%$ | $87,64\%$ | $89,16\%$ | $88,50\%$ | $88,73\%$ | $98,42\%$ |
| **10** | **S1 — Alt GA→PO** | **Hybrid (Xen kẽ)** | **$2.799,0$** | **$75,07\%$** | $88,71\%$ | $87,31\%$ | $89,03\%$ | $88,35\%$ | $88,61\%$ | $98,25\%$ |

---

### 4.3. Các Kỷ lục Thực nghiệm Đơn lẻ (Peak Seed Runs) và Phân tích Chuyên sâu

Bên cạnh các chỉ số trung bình đa hạt giống, tệp [`outputs/run_poga_all_20261002_173334/experiment_results.csv`](file:///d:/Code/KhoaLuanDatasetReduction/outputs/run_poga_all_20261002_173334/experiment_results.csv) ghi nhận những kỷ lục đáng chú ý ở từng kiến trúc mạng:

1. **Kỷ lục trên mạng `DenseNet-121`**:
   - Tại `Seed 42`, tập con của **`C1 (Coop Exchange)`** ($M = 2.977$ ảnh, $\text{DR} = 73,49\%$) đạt Test Accuracy **$89,97\%$** và Macro F1-Score **$90,12\%$**, giữ lại tới **$\text{ARR} = 99,34\%$** hiệu năng gốc (chỉ thấp hơn $0,60\%$ so với huấn luyện trên toàn bộ $11.228$ ảnh).
   - Tại `Seed 123`, tập con của **`S4 (Block PO→GA)`** ($M = 2.987$ ảnh, $\text{DR} = 73,40\%$) đạt Test Accuracy **$89,77\%$** và Macro F1-Score **$89,91\%$** ($\text{ARR} = 99,12\%$).
   - Tại `Seed 42`, tập con của **`P1 (Parallel Merge)`** ($M = 2.961$ ảnh, $\text{DR} = 73,63\%$) đạt Test Accuracy **$89,70\%$** và Macro F1-Score **$89,84\%$** ($\text{ARR} = 99,04\%$).
2. **Kỷ lục trên mạng `ResNet-18`**:
   - Tập con của **`S4 (Block PO→GA)`** tại `Seed 123` đạt Test Accuracy **$89,67\%$** ($\text{ARR} = 99,34\%$, chỉ kém tập dữ liệu gốc $0,60\%$).
   - Tập con của **`P1 (Parallel Merge)`** đạt độ ổn định cao nhất trên `ResNet-18` qua cả 3 hạt giống: `Seed 42` đạt **$89,27\%$**, `Seed 123` đạt **$89,37\%$**, `Seed 456` đạt **$89,50\%$** (trung bình **$89,38 \pm 0,10\%$**, dẫn đầu toàn bộ 10 phương pháp trên `ResNet-18` và vượt `Stratified Random B2` $+0,55\%$ cùng `Random Selection B1` $+0,67\%$).
3. **Kỷ lục trên mạng nhẹ `MobileNetV3-Small`**:
   - Tập con của **`S4 (Block PO→GA)`** dẫn đầu toàn cục với trung bình **$88,02 \pm 0,34\%$** (đạt đỉnh **$88,37\%$** tại `Seed 42`, tương ứng tỷ lệ duy trì độ chính xác **$\text{ARR} = 99,37\%$**, chỉ thấp hơn tập gốc $0,56\%$).
4. **Sự phân công vai trò rõ rệt giữa các nhóm kiến trúc Hybrid PO–GA trên mặt phẳng Pareto (Accuracy vs. Reduction Ratio)**:
   - **Nếu ưu tiên tối đa hóa Tỷ lệ nén ($\text{DR}$) và Điểm mục tiêu ($F$)**: Hai cấu hình lai tuần tự xen kẽ **`S1 (Alt GA→PO)`** và **`S2 (Alt PO→GA)`** là lựa chọn tối ưu nhất, cắt giảm tới **$75,07\%$** và **$74,81\%$** dữ liệu (chỉ giữ lại $2.799 - 2.828$ ảnh, tiết kiệm thêm $140 - 180$ ảnh so với các phương pháp khác) trong khi vẫn duy trì độ chính xác $89,16\%$ trên `ResNet-18` (với `S2`, cao hơn cả `GA-only` $89,14\%$, `PO-only` $88,92\%$, `Stratified B2` $88,83\%$ và `Random B1` $88,71\%$ dù dùng ít hơn từ $141$ đến $236$ bức ảnh).
   - **Nếu ưu tiên tối đa hóa Độ chính xác phân loại CNN hạ nguồn trên mọi kiến trúc mạng**: Hai cấu hình **`S4 (Block PO→GA)`** ($88,87\%$ trung bình 3 mạng, hạng #1 toàn cục) và **`P1 (Parallel Merge)`** ($88,75\%$ trung bình 3 mạng, hạng #2 toàn cục và hạng #1 trên `ResNet-18` với $89,38\%$) là lựa chọn xuất sắc nhất, vượt qua cả hai thuật toán đơn lẻ (`GA-only` $88,72\%$, `PO-only` $88,66\%$) lẫn cả hai phương pháp lấy mẫu ngẫu nhiên (`Stratified B2` $88,60\%$, `Random B1` $88,50\%$).

---

### 4.4. Kiểm định Ý nghĩa Thống kê Phi tham số Wilcoxon Signed-Rank Test

Theo báo cáo kiểm định tự động tại [`outputs/run_poga_all_20261002_173334/statistical_significance.md`](file:///d:/Code/KhoaLuanDatasetReduction/outputs/run_poga_all_20261002_173334/statistical_significance.md):
- Với cỡ mẫu $n = 3$ hạt giống độc lập (`Seeds = [42, 123, 456]`), giá trị $p\text{-value}$ nhỏ nhất có thể đạt được về mặt toán học của kiểm định hai phía Wilcoxon Signed-Rank Test là $p_{\min} = \frac{2}{2^3} = 0,2500$ — xảy ra khi một phương pháp chiến thắng tuyệt đối phương pháp đối thủ trên cả $3/3$ hạt giống ($\text{Win Rate} = 100\%$).
- Các cặp đối sánh đạt mức thống trị tuyệt đối $3/3$ hạt giống ($p = 0,2500$, chênh lệch hoàn toàn một chiều trên mọi `seed`) bao gồm:
  - Trên **`ResNet-18`**: **`P1 (Parallel Merge)`** và **`C1 (Coop Exchange)`** thắng tuyệt đối $3/3$ hạt giống trước cả `Random Selection (B1)` ($+0,67\%$ và $+0,64\%$), `Stratified Random (B2)` ($+0,54\%$ và $+0,52\%$), `PO-only (B4)` ($+0,46\%$ và $+0,43\%$), `S3` ($+0,34\%$ và $+0,32\%$) và `S1` ($+0,67\%$ và $+0,64\%$); biến thể **`S2 (Alt PO→GA)`** thắng tuyệt đối $3/3$ hạt giống trước `Random Selection (B1)` ($+0,44\%$) và `S1` ($+0,44\%$) dù sử dụng ít hơn $150$ bức ảnh so với `B1`.
  - Trên **`MobileNetV3-Small`**: **`S4 (Block PO→GA)`** thắng tuyệt đối $3/3$ hạt giống trước `S1`, `S2`, `S3` và `C1` (chênh lệch từ $+0,50\%$ đến $+0,71\%$ trên mọi hạt giống).
- Bên cạnh kiểm định theo từng mạng với $n = 3$, nếu xét gộp cặp đối sánh trên toàn bộ $n = 9$ phiên huấn luyện độc lập ($3\text{ seeds} \times 3\text{ kiến trúc CNN}$), biến thể **`S4 (Block PO→GA)`** thắng `Random Selection (B1)` trên $8/9$ phiên chạy (trung bình $+0,37\%$, $p = 0,0195 < 0,05$ theo kiểm định Wilcoxon 9 cặp) và thắng `PO-only (B4)` trên $7/9$ phiên chạy, khẳng định tính ưu việt có ý nghĩa thống kê của kiến trúc lai hóa PO–GA.
