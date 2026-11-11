# Giải thích chi tiết từng lớp trong LPRNet

Tài liệu này giải thích file `LPRNet.py` của bạn theo ba góc nhìn cho mỗi lớp: **khái niệm**, **vai trò trong model** và **công thức toán học**. Công thức viết bằng LaTeX (`$$...$$`), xem đẹp nhất trong Markdown Preview của VS Code, GitHub hoặc Typora.

**Quy ước:**

- $N$: batch size, $C$: số kênh, $H$: chiều cao, $W$: chiều rộng. Shape ghi dạng `C×H×W` (bỏ $N$).
- Ảnh đầu vào cố định `3×24×94` (3 kênh màu, cao 24, rộng 94).
- $K$ = `class_num` = 37 (36 ký tự + 1 ký tự blank của CTC). Mọi con số trong tài liệu đã được tính với $K = 37$.
- Số liệu shape và số tham số được tính bằng công thức, khớp với cấu trúc code của bạn.

---

## 1. Bức tranh tổng thể

LPRNet nhận dạng biển số **không cần tách từng ký tự**. Nó coi chiều rộng của ảnh như một "trục thời gian": ảnh được nén thành **18 cột**, mỗi cột dự đoán một phân phối xác suất trên 37 lớp (36 ký tự + blank). Sau đó hàm loss **CTC** tự gộp 18 dự đoán này thành chuỗi biển số, nên nhãn chỉ cần là chuỗi ký tự mà không cần vị trí từng ký tự.

```
Ảnh 3×24×94
   │
   ▼
┌──────────────── BACKBONE (nn.Sequential, 23 lớp: chỉ số 0..22) ────────────────┐
│ Conv → BN → ReLU ──────────────────────────────► [f0: 64×22×92]  (chỉ số 2)    │
│ MaxPool3d → small_basic_block → BN → ReLU ─────► [f1: 128×20×90] (chỉ số 6)    │
│ MaxPool3d → small_basic_block → BN → ReLU                                       │
│           → small_basic_block → BN → ReLU ─────► [f2: 256×18×44] (chỉ số 13)   │
│ MaxPool3d → Dropout → Conv(1×4) → BN → ReLU → Dropout                           │
│           → Conv(13×1) → BN → ReLU ────────────► [f3: 37×4×18]   (chỉ số 22)    │
└─────────────────────────────────────────────────────────────────────────────────┘
   │  4 đặc trưng ở 4 độ sâu khác nhau
   ▼
AvgPool2d đưa f0, f1, f2 về cùng cỡ 4×18        (f3 đã là 4×18)
   ▼
Chia mỗi đặc trưng cho năng lượng trung bình của nó   (chuẩn hóa "global context")
   ▼
Nối theo kênh (torch.cat): 64 + 128 + 256 + 37 = 485 kênh × 4 × 18
   ▼
container: Conv 1×1  (485 → 37)
   ▼
Trung bình theo chiều cao (4 hàng → 1)
   ▼
logits: 37 × 18   →  log_softmax → CTC loss (nằm trong file train)
```

Hai ý tưởng chính:

1. **Backbone** trích đặc trưng và thu nhỏ dần chiều cao ảnh (24 → 22 → 20 → 18 → 16 → 4) cùng chiều rộng (94 → 18).
2. **Global context**: không chỉ dùng đầu ra cuối của backbone mà còn lấy các đặc trưng ở tầng nông và tầng giữa, trộn lại để dự đoán cuối cùng. Tầng nông giữ chi tiết nét chữ, tầng sâu giữ thông tin ngữ nghĩa.

---

## 2. Bảng shape và số tham số từng lớp

Chỉ số là vị trí của lớp trong `self.backbone` (khớp với `i` trong vòng lặp `forward`). Các dòng có 🔸 là lớp mà đầu ra được giữ lại trong `keep_features`.

| Chỉ số | Lớp | Shape đầu ra | Số tham số |
|---|---|---|---|
| input | ảnh | `3×24×94` | |
| 0 | `Conv2d(3→64, k=3)` | `64×22×92` | 1,792 |
| 1 | `BatchNorm2d(64)` | `64×22×92` | 128 |
| 2 🔸 | `ReLU` → **f0** | `64×22×92` | 0 |
| 3 | `MaxPool3d(k=(1,3,3), s=(1,1,1))` | `64×20×90` | 0 |
| 4 | `small_basic_block(64→128)` | `128×20×90` | 12,512 |
| 5 | `BatchNorm2d(128)` | `128×20×90` | 256 |
| 6 🔸 | `ReLU` → **f1** | `128×20×90` | 0 |
| 7 | `MaxPool3d(k=(1,3,3), s=(2,1,2))` | `64×18×44` | 0 |
| 8 | `small_basic_block(64→256)` | `256×18×44` | 45,504 |
| 9 | `BatchNorm2d(256)` | `256×18×44` | 512 |
| 10 | `ReLU` | `256×18×44` | 0 |
| 11 | `small_basic_block(256→256)` | `256×18×44` | 57,792 |
| 12 | `BatchNorm2d(256)` | `256×18×44` | 512 |
| 13 🔸 | `ReLU` → **f2** | `256×18×44` | 0 |
| 14 | `MaxPool3d(k=(1,3,3), s=(4,1,2))` | `64×16×21` | 0 |
| 15 | `Dropout(p)` | `64×16×21` | 0 |
| 16 | `Conv2d(64→256, k=(1,4))` | `256×16×18` | 65,792 |
| 17 | `BatchNorm2d(256)` | `256×16×18` | 512 |
| 18 | `ReLU` | `256×16×18` | 0 |
| 19 | `Dropout(p)` | `256×16×18` | 0 |
| 20 | `Conv2d(256→37, k=(13,1))` | `37×4×18` | 123,173 |
| 21 | `BatchNorm2d(37)` | `37×4×18` | 74 |
| 22 🔸 | `ReLU` → **f3** | `37×4×18` | 0 |
| (forward) | `AvgPool2d(5, 5)` trên f0 | `64×4×18` | 0 |
| (forward) | `AvgPool2d(5, 5)` trên f1 | `128×4×18` | 0 |
| (forward) | `AvgPool2d((4,10), (4,2))` trên f2 | `256×4×18` | 0 |
| (forward) | chuẩn hóa + `torch.cat` | `485×4×18` | 0 |
| container | `Conv2d(485→37, k=1)` | `37×4×18` | 17,982 |
| (forward) | `torch.mean(dim=2)` | **`37×18`** | 0 |
| | **Tổng tham số huấn luyện được** | | **326,541** |

Con số 326,541 chỉ tính các tham số học được. `BatchNorm` còn có thêm `running_mean` và `running_var` (không học bằng gradient, nên không nằm trong tổng này).

---

## 3. Các loại lớp: khái niệm, vai trò, công thức

### 3.1 `Conv2d` (tích chập 2D)

**Khái niệm.** Một ma trận trọng số nhỏ (kernel) trượt trên ảnh. Tại mỗi vị trí, nó nhân từng phần tử với vùng ảnh bên dưới rồi cộng tổng lại, tạo ra một điểm của bản đồ đặc trưng. Mỗi kernel học để phát hiện một mẫu cục bộ (cạnh, nét ngang, nét dọc, góc...).

**Vai trò.** Là khối trích đặc trưng chính. Các lớp đầu học nét đơn giản, các lớp sau học mẫu phức tạp hơn (nét cong của chữ `S`, `8`, `B`...). Dùng chung một kernel cho mọi vị trí nên số tham số ít và model nhận ra mẫu ở bất kỳ chỗ nào trên ảnh.

**Công thức.** Với $s$ là stride, $P$ là padding:

$$
y_{n,o,i,j} \;=\; b_o \;+\; \sum_{c=0}^{C_{in}-1}\;\sum_{p=0}^{K_h-1}\;\sum_{q=0}^{K_w-1} W_{o,c,p,q}\; x_{n,\,c,\;i\cdot s_h + p - P_h,\;\; j\cdot s_w + q - P_w}
$$

Kích thước đầu ra và số tham số:

$$
H_{out}=\left\lfloor \frac{H_{in}+2P_h-K_h}{s_h}\right\rfloor+1,\qquad
W_{out}=\left\lfloor \frac{W_{in}+2P_w-K_w}{s_w}\right\rfloor+1
$$

$$
\#\text{params}=C_{out}\cdot C_{in}\cdot K_h\cdot K_w + C_{out}
$$

**Các conv trong model của bạn:**

- **Chỉ số 0**, `Conv2d(3→64, k=3)`: nhìn thẳng vào pixel RGB, học các bộ lọc cạnh và màu cơ bản. Không có padding nên $24\times94 \to 22\times92$.
- **Chỉ số 16**, `Conv2d(64→256, k=(1,4))`: kernel cao 1 và rộng 4, chỉ gộp thông tin theo **chiều ngang** (4 cột liền nhau). Giúp mỗi cột nhìn được ngữ cảnh các cột bên cạnh, rất hợp với việc đọc một chuỗi ký tự. Chiều rộng $21 \to 18$.
- **Chỉ số 20**, `Conv2d(256→37, k=(13,1))`: kernel cao 13 và rộng 1, gộp thông tin theo **chiều dọc**. Chiều cao $16 \to 4$ (trượt 4 vị trí), đồng thời đổi số kênh từ 256 sang 37, tức mỗi điểm bắt đầu mang "điểm số cho 37 lớp". Đây là lớp nặng nhất model (123,173 tham số).

**Conv 1×1 (kernel 1×1)** là trường hợp đặc biệt: không nhìn hàng xóm mà chỉ trộn các kênh tại cùng một vị trí.

$$
y_{o,i,j} = b_o + \sum_{c} W_{o,c}\; x_{c,i,j}
$$

Nó hoạt động như một lớp fully-connected áp dụng độc lập lên từng điểm ảnh, dùng để **đổi số kênh** (nén hoặc mở rộng) với chi phí thấp.

---

### 3.2 `BatchNorm2d` (chuẩn hóa theo batch)

**Khái niệm.** Với mỗi kênh, chuẩn hóa các giá trị trong batch về trung bình 0, phương sai 1, rồi cho model tự học lại độ co giãn $\gamma$ và độ dịch $\beta$.

**Vai trò.** Giữ phân phối đầu vào của mỗi lớp ổn định trong lúc train, nên train nhanh hơn, cho phép dùng learning rate lớn hơn và đỡ nhạy với khởi tạo trọng số. Mỗi kênh có hai tham số học được ($\gamma_c$, $\beta_c$), nên `BatchNorm2d(C)` có $2C$ tham số.

**Công thức (chế độ train).** Thống kê tính trên cả batch và các vị trí không gian cho từng kênh $c$:

$$
\mu_c=\frac{1}{NHW}\sum_{n,i,j}x_{n,c,i,j},\qquad
\sigma_c^2=\frac{1}{NHW}\sum_{n,i,j}\left(x_{n,c,i,j}-\mu_c\right)^2
$$

$$
\hat x_{n,c,i,j}=\frac{x_{n,c,i,j}-\mu_c}{\sqrt{\sigma_c^2+\varepsilon}},\qquad
y_{n,c,i,j}=\gamma_c\,\hat x_{n,c,i,j}+\beta_c
$$

với $\varepsilon = 10^{-5}$ để tránh chia cho 0.

**Chế độ eval.** Không dùng thống kê của batch mà dùng trung bình trượt đã tích lũy trong lúc train (momentum $m = 0.1$):

$$
\mu_{run}\leftarrow(1-m)\,\mu_{run}+m\,\mu_c,\qquad
\sigma^2_{run}\leftarrow(1-m)\,\sigma^2_{run}+m\,\sigma_c^2
$$

(PyTorch dùng phương sai không chệch khi cập nhật `running_var`.) Đây là lý do bắt buộc phải gọi `model.eval()` khi đánh giá hoặc nhận dạng. Nếu quên, kết quả phụ thuộc vào các ảnh khác trong batch và sẽ sai.

**Trong model:** đứng ngay sau mỗi conv (chỉ số 1, 5, 9, 12, 17, 21), trước `ReLU`.

---

### 3.3 `ReLU` (hàm kích hoạt)

**Khái niệm.** Giữ nguyên giá trị dương và đưa giá trị âm về 0.

**Vai trò.** Thêm tính **phi tuyến**. Nếu thiếu nó, chồng nhiều conv liên tiếp chỉ tương đương một conv duy nhất. ReLU đơn giản, tính nhanh và ít bị triệt tiêu gradient so với sigmoid/tanh.

**Công thức:**

$$
\mathrm{ReLU}(x)=\max(0,x),\qquad
\frac{d\,\mathrm{ReLU}}{dx}=\begin{cases}1 & x>0\\ 0 & x\le 0\end{cases}
$$

**Lưu ý trong model:** `ReLU` ở chỉ số 22 (cuối backbone) làm cho f3 không âm. Đầu ra cuối của model (logits) vẫn có thể âm vì phía sau còn `container` (một conv không có ReLU).

---

### 3.4 `MaxPool3d` (gộp cực đại) và mẹo input 4D

**Khái niệm.** Chia dữ liệu thành các cửa sổ nhỏ rồi chỉ giữ giá trị **lớn nhất** trong mỗi cửa sổ.

**Vai trò thông thường.** Giảm kích thước không gian, giữ lại phản hồi mạnh nhất của đặc trưng, và giúp model chịu được lệch vị trí nhỏ.

**Công thức** với cửa sổ $(k_D,k_H,k_W)$ và stride $(s_D,s_H,s_W)$:

$$
y_{d,i,j}=\max_{0\le a<k_D,\;0\le p<k_H,\;0\le q<k_W} x_{\,d\cdot s_D+a,\;\; i\cdot s_H+p,\;\; j\cdot s_W+q}
$$

Kích thước từng chiều: $\left\lfloor\frac{\text{in}-k}{s}\right\rfloor+1$.

**Điểm đặc biệt (quan trọng).** Tensor trong model có dạng 4D $(N, C, H, W)$, nhưng `MaxPool3d` đọc input 4D theo dạng không có batch $(C', D, H, W)$. Do đó:

- $N$ (batch) bị đọc là "kênh", nên mỗi ảnh được pool độc lập, không trộn lẫn nhau.
- Số **kênh $C$ bị đọc là chiều sâu $D$**.

Cửa sổ có $k_D=1$ nên không có phép max nào giữa các kênh. Còn $s_D$ lớn hơn 1 nghĩa là **chỉ lấy một phần các kênh**:

| Chỉ số | Cửa sổ / stride | Tác dụng lên kênh | Tác dụng lên không gian |
|---|---|---|---|
| 3 | `(1,3,3)` / `(1,1,1)` | giữ nguyên 64 kênh | max pool 3×3: `22×92 → 20×90` |
| 7 | `(1,3,3)` / `(2,1,2)` | **chọn kênh 0,2,4,… : 128 → 64** | `20×90 → 18×44` (rộng giảm một nửa) |
| 14 | `(1,3,3)` / `(4,1,2)` | **chọn kênh 0,4,8,… : 256 → 64** | `18×44 → 16×21` (rộng giảm một nửa) |

Vì vậy hai khối `small_basic_block` sau đó mới có `ch_in=64` (chứ không phải 128 hay 256) và `Conv2d` chỉ số 16 có `in_channels=64`. Chiều cao chỉ giảm theo kernel 3 (stride 1), còn chiều rộng giảm theo stride 2.

Đây là hành vi của bản cài đặt PyTorch này. Lưu ý rằng nó **bỏ hẳn** một nửa hoặc ba phần tư số kênh ở hai chỗ đó chứ không "nén" chúng, và mình không khẳng định nó giống hệt cách thiết kế trong bài báo gốc. Model vẫn học tốt vì khối phía trước đã được huấn luyện để các kênh được giữ lại là những kênh hữu ích.

Yêu cầu: cần PyTorch đủ mới để `MaxPool3d` nhận input 4D.

---

### 3.5 `Dropout`

**Khái niệm.** Trong lúc train, ngẫu nhiên "tắt" (đặt về 0) một tỷ lệ $p$ các phần tử, phần còn lại được nhân lên để giữ nguyên kỳ vọng.

**Vai trò.** Chống **overfitting**: buộc model không dựa vào một vài nơ-ron cụ thể mà phải học các đặc trưng dư thừa, đáng tin cậy hơn. Có hai lớp Dropout (chỉ số 15 và 19) với $p$ = `dropout_rate` = 0.5, đặt trước hai conv cuối.

**Công thức (train):**

$$
y=\frac{m}{1-p}\odot x,\qquad m_k\sim\mathrm{Bernoulli}(1-p)
$$

**Eval:** $y = x$ (không làm gì). Đây là lý do thứ hai cần `model.eval()` khi nhận dạng.

---

### 3.6 `small_basic_block` (khối bottleneck phân rã)

**Khái niệm.** Một khối gồm 4 conv: **nén kênh → conv dọc → conv ngang → mở rộng kênh**.

```
x (C_in) ─ Conv1×1 → ReLU ─ Conv(3×1) → ReLU ─ Conv(1×3) → ReLU ─ Conv1×1 ─► y (C_out)
           C_in → C_out/4     C_out/4          C_out/4            C_out/4 → C_out
```

**Vai trò.** Thay một conv 3×3 đắt tiền bằng cách rẻ hơn mà vẫn nhìn được vùng 3×3:

- Conv 1×1 đầu **nén** số kênh xuống $C_{out}/4$ (xem giải thích `ch_out // 4` ở trên).
- Conv $(3,1)$ rồi conv $(1,3)$ tạo vùng nhìn (receptive field) 3×3, mỗi cặp kênh chỉ cần $3+3=6$ trọng số thay vì $9$. Chúng cũng chuyên biệt hóa: một cái bắt cấu trúc dọc, một cái bắt cấu trúc ngang. Hai `ReLU` xen kẽ thêm phi tuyến.
- Conv 1×1 cuối **mở rộng** về $C_{out}$ kênh. Lớp này không có `ReLU` ngay sau nó, vì `BatchNorm2d` và `ReLU` nằm bên ngoài khối.

**Công thức** ($\sigma$ là ReLU, $*$ là tích chập):

$$
y=W_4 * \sigma\!\Big(W_3 * \sigma\big(W_2 * \sigma(W_1 * x)\big)\Big)
$$

với $W_1: 1{\times}1$ $(C_{in}\to C_{out}/4)$, $W_2: 3{\times}1$, $W_3: 1{\times}3$ (cùng $C_{out}/4\to C_{out}/4$, có padding để giữ nguyên $H$, $W$), $W_4: 1{\times}1$ $(C_{out}/4\to C_{out})$.

Số tham số (cộng thêm bias):

$$
C_{in}\tfrac{C_{out}}{4}+2\cdot3\left(\tfrac{C_{out}}{4}\right)^2+\tfrac{C_{out}}{4}C_{out}
$$

Ví dụ khối `256→256` có **57,792** tham số, trong khi một conv 3×3 `256→256` thông thường cần **590,080** (gấp khoảng 10 lần).

**Trong model:** có 3 khối (chỉ số 4: `64→128`, chỉ số 8: `64→256`, chỉ số 11: `256→256`). Mỗi khối giữ nguyên $H\times W$, chỉ đổi số kênh.

---

### 3.7 `AvgPool2d` (gộp trung bình)

**Khái niệm.** Giống max pooling nhưng lấy **trung bình** cửa sổ thay vì cực đại.

**Vai trò.** Trong model, nó **thu nhỏ f0, f1, f2 về cùng cỡ `4×18`** để có thể ghép với f3. Dùng trung bình (thay vì max) để tóm tắt cả vùng rộng một cách mượt mà.

**Công thức:**

$$
y_{c,i,j}=\frac{1}{k_hk_w}\sum_{p=0}^{k_h-1}\sum_{q=0}^{k_w-1}x_{c,\;i\cdot s_h+p,\;j\cdot s_w+q}
$$

| Đặc trưng | Cửa sổ / stride | Shape vào → ra |
|---|---|---|
| f0 | `5×5` / `5` | `64×22×92 → 64×4×18` |
| f1 | `5×5` / `5` | `128×20×90 → 128×4×18` |
| f2 | `(4,10)` / `(4,2)` | `256×18×44 → 256×4×18` |
| f3 | không pool | `37×4×18` |

---

### 3.8 Chuẩn hóa "global context" (`f / mean(f²)`)

**Khái niệm.** Với mỗi trong 4 đặc trưng, tính **năng lượng trung bình** (trung bình bình phương) trên toàn bộ tensor, rồi chia đặc trưng cho giá trị đó:

$$
\bar f=\frac{1}{N\,C\,H\,W}\sum_{n,c,i,j}f_{n,c,i,j}^{\,2},\qquad
f'=\frac{f}{\bar f}
$$

Trong code: `f_pow = torch.pow(f, 2)`, `f_mean = torch.mean(f_pow)` (một số vô hướng duy nhất cho cả tensor), `f = torch.div(f, f_mean)`.

**Vai trò.** Bốn nhánh có biên độ giá trị rất khác nhau (f0 là đặc trưng nông, f3 là điểm số cuối). Nếu ghép thẳng, nhánh có giá trị lớn sẽ lấn át phần còn lại. Phép chia này **cân bằng biên độ** giữa các nhánh để `container` trộn chúng hợp lý. Phép này không có tham số học.

**Lưu ý:**

- Chia cho $\bar f$ (trung bình bình phương) chứ không phải căn bậc hai của nó, nên đây **không** phải chuẩn hóa về đơn vị năng lượng chính xác, chỉ là cách cân bằng biên độ tương đối.
- Vì $\bar f$ tính trên cả batch, kết quả của một ảnh phụ thuộc nhẹ vào các ảnh còn lại trong batch (kể cả khi `eval`). Khi nhận dạng một ảnh đơn lẻ, kết quả có thể lệch chút so với khi đánh giá theo batch. Đây là đặc điểm có sẵn của bản cài đặt này.
- Nếu một đặc trưng toàn bằng 0 (hiếm), $\bar f = 0$ và phép chia cho ra `nan`.

---

### 3.9 `torch.cat` (ghép kênh)

Nối 4 đặc trưng cùng cỡ `4×18` theo chiều kênh:

$$
X=[\,f_0';\,f_1';\,f_2';\,f_3'\,]\in\mathbb{R}^{(64+128+256+K)\times4\times18}=\mathbb{R}^{485\times4\times18}
$$

**Vai trò:** gom thông tin từ nhiều mức trừu tượng (nét chữ ở tầng nông, cấu trúc ở tầng giữa, điểm số lớp ở tầng sâu) vào một tensor. Con số `448 + class_num` trong `container` chính là $64+128+256+K$. Nếu bạn đổi `class_num`, nhớ rằng số kênh vào `container` tự đổi theo vì code đã viết `448 + self.class_num`.

---

### 3.10 `container`: Conv 1×1 trộn đặc trưng

`Conv2d(485 → 37, kernel 1×1)`:

$$
z_{k,i,j}=b_k+\sum_{c=1}^{485}W_{k,c}\,X_{c,i,j},\qquad k=1,\dots,37
$$

**Vai trò.** Học cách **kết hợp** 485 kênh (từ cả 4 mức đặc trưng) thành 37 điểm số cho mỗi vị trí `(hàng, cột)`. Đây là lớp quyết định cuối cùng trước khi lấy trung bình. Số tham số: $485\times37+37=17{,}982$.

---

### 3.11 `torch.mean(x, dim=2)` (trung bình theo chiều cao)

$$
\mathrm{logits}_{n,k,j}=\frac{1}{4}\sum_{i=1}^{4}z_{n,k,i,j}
$$

**Vai trò.** Gộp 4 hàng dọc thành 1, nên mỗi **cột** $j\in\{1,\dots,18\}$ có đúng một vector 37 điểm số. Kết quả là chuỗi dài $T=18$ bước, mỗi bước 37 lớp, đúng định dạng CTC cần. Mỗi cột tương ứng xấp xỉ một dải ngang rộng khoảng $94/18\approx5$ pixel của ảnh.

---

### 3.12 Ngoài model nhưng cần để hiểu đầu ra: `log_softmax` và CTC loss

Model trả về **logits** (chưa qua softmax). Trong file train, logits được đổi chiều thành `T×N×K` rồi qua `log_softmax`.

**Softmax tại mỗi bước thời gian:**

$$
p_t(k)=\frac{e^{\,\mathrm{logits}_{k,t}}}{\sum_{k'=1}^{K}e^{\,\mathrm{logits}_{k',t}}}
$$

**CTC (Connectionist Temporal Classification).** Một đường đi (path) $\pi=(\pi_1,\dots,\pi_{18})$ là một chuỗi nhãn tại từng bước. Hàm $\mathcal{B}$ biến đường đi thành chuỗi cuối bằng cách **gộp ký tự lặp liên tiếp, rồi bỏ blank**. Ví dụ: `3 3 - 0 A A - -` → `30A`. Xác suất của chuỗi đích $\mathbf{y}$ là tổng xác suất mọi đường đi dẫn đến nó:

$$
P(\mathbf{y}\mid x)=\sum_{\pi\in\mathcal{B}^{-1}(\mathbf{y})}\;\prod_{t=1}^{18}p_t(\pi_t),\qquad
\mathcal{L}=-\ln P(\mathbf{y}\mid x)
$$

Tổng này được tính hiệu quả bằng quy hoạch động (thuật toán forward–backward). Với `reduction='mean'`, PyTorch chia loss của mỗi mẫu cho độ dài nhãn của nó rồi lấy trung bình theo batch.

**Vai trò của blank.** Blank (index 36, ký tự `-` trong `CHARS`) cho phép model "không ghi gì" ở một cột, và tách hai ký tự giống nhau đứng liền nhau (ví dụ `11` cần `1 - 1` giữa). Vì vậy cần $T\ge L+(\text{số cặp ký tự giống nhau liền kề})$. Với $T=18$ và biển số 8–9 ký tự thì rất thoải mái.

**Giải mã (greedy):** lấy `argmax` mỗi cột, gộp ký tự lặp, bỏ blank:

$$
\hat{\mathbf{y}}=\mathcal{B}\big(\arg\max_k\,p_1(k),\;\dots,\;\arg\max_k\,p_{18}(k)\big)
$$

---

## 4. Đi qua model theo thứ tự thực thi

1. **Chỉ số 0–2, `Conv(3→64)` + `BN` + `ReLU`**: học 64 bộ lọc cạnh/nét cơ bản trực tiếp từ pixel. Đầu ra được giữ làm **f0** (đặc trưng nông, nhiều chi tiết nhất).
2. **Chỉ số 3, `MaxPool3d`**: giảm nhiễu và thu nhỏ nhẹ `22×92 → 20×90`.
3. **Chỉ số 4–6, `small_basic_block(64→128)` + `BN` + `ReLU`**: mở rộng lên 128 kênh, học mẫu phức tạp hơn. Đầu ra là **f1**.
4. **Chỉ số 7, `MaxPool3d` stride `(2,1,2)`**: chọn một nửa số kênh (128 → 64) và giảm chiều rộng một nửa (`90 → 44`), nhờ đó rút ngắn trục "thời gian".
5. **Chỉ số 8–10, `small_basic_block(64→256)` + `BN` + `ReLU`**: nâng lên 256 kênh để có năng lực biểu diễn lớn.
6. **Chỉ số 11–13, `small_basic_block(256→256)` + `BN` + `ReLU`**: tinh chỉnh thêm đặc trưng. Đầu ra là **f2**.
7. **Chỉ số 14, `MaxPool3d` stride `(4,1,2)`**: chọn 1/4 số kênh (256 → 64), rộng `44 → 21`.
8. **Chỉ số 15, `Dropout`**: điều chuẩn để chống overfit.
9. **Chỉ số 16–18, `Conv(1×4)` + `BN` + `ReLU`**: gộp ngữ cảnh 4 cột liền kề (`21 → 18`), lên 256 kênh.
10. **Chỉ số 19, `Dropout`**: điều chuẩn lần hai.
11. **Chỉ số 20–22, `Conv(13×1)` + `BN` + `ReLU`**: gộp toàn bộ chiều dọc (`16 → 4`) và đổi sang 37 kênh. Đầu ra là **f3**.
12. **Global context**: AvgPool f0, f1, f2 về `4×18`, chia mỗi nhánh cho năng lượng trung bình, rồi nối cả bốn thành `485×4×18`.
13. **`container` (Conv 1×1)**: trộn 485 kênh thành 37 điểm số.
14. **`torch.mean(dim=2)`**: gộp 4 hàng thành 1, ra **logits `37×18`**.
15. **(file train) `log_softmax` + `CTCLoss`**: so sánh 18 dự đoán với chuỗi biển số thật để tính loss, hoặc `argmax` + gộp lặp + bỏ blank để ra biển số khi nhận dạng.

---

## 5. Phân bố tham số

| Thành phần | Số tham số | Tỷ lệ |
|---|---|---|
| `Conv2d(256→37, 13×1)` (chỉ số 20) | 123,173 | 37.7% |
| `Conv2d(64→256, 1×4)` (chỉ số 16) | 65,792 | 20.1% |
| `small_basic_block(256→256)` | 57,792 | 17.7% |
| `small_basic_block(64→256)` | 45,504 | 13.9% |
| `container` (Conv 1×1) | 17,982 | 5.5% |
| `small_basic_block(64→128)` | 12,512 | 3.8% |
| `Conv2d(3→64)` | 1,792 | 0.5% |
| 6 lớp `BatchNorm2d` | 2,494 | 0.8% |
| **Tổng** | **326,541** | 100% |

Model chỉ khoảng **0.33 triệu** tham số, rất nhẹ, phù hợp chạy trên thiết bị yếu.

---

## 6. Những điều cần nhớ khi dùng model

- **Kích thước ảnh phải đúng `24×94`.** Conv `(13,1)` cần chiều cao 16 tại chỉ số 20, và các `AvgPool2d` cần đúng `22×92`, `20×90`, `18×44` để ra `4×18`. Sai kích thước thì `torch.cat` lỗi.
- **`class_num` = `len(CHARS)` = 37** (gồm blank ở index 36). Default trong `build_lprnet` của bạn hiện là 36, nên luôn truyền `class_num` tường minh.
- **Luôn gọi `model.eval()`** khi đánh giá hoặc nhận dạng, vì `BatchNorm` và `Dropout` hoạt động khác nhau giữa train và eval.
- **Đầu ra là logits `N×37×18`**, chưa qua softmax. Khi giải mã thì chỉ cần `argmax` theo chiều lớp, không cần softmax.
- **Biển 2 dòng** (xe máy) phải được ghép thành một dải ngang trước khi đưa vào model (hàm `merge_two_lines`), vì model đọc từ trái sang phải theo chiều rộng.
