# Giải thích chi tiết từng lớp trong LPRNet

Tài liệu giải thích file `LPRNet.py` và phần liên quan trong `train_LPRNet.py`. Mỗi lớp có bốn phần: **khái niệm**, **vai trò trong model**, **công thức** và **ghi chú theo tài liệu PyTorch**.

Về cách hiển thị: file này **không dùng LaTeX**. Mọi công thức nằm trong khối code (ba dấu backtick) nên hiển thị giống nhau ở VS Code, GitHub, Typora, Notepad hay bất kỳ trình xem Markdown nào. Sơ đồ chỉ dùng ký tự ASCII để không bị lệch cột.

Nguồn đối chiếu: tài liệu chính thức của PyTorch (`docs.pytorch.org`) cho `Conv2d`, `BatchNorm2d`, `MaxPool3d`, `Dropout`, `AvgPool2d`, `CTCLoss`; xem danh sách ở cuối file.

---

## 0. Quy ước

- `N` là batch size, `C` là số kênh, `H` là chiều cao, `W` là chiều rộng. Shape ghi dạng `C×H×W` (bỏ `N`).
- Ảnh đầu vào cố định `3×24×94`: 3 kênh màu, cao 24, rộng 94.
- `K` là `class_num`, bằng `len(CHARS)`. Theo tài liệu cũ của bạn, `K = 37` (36 ký tự + 1 ký tự blank của CTC). **Mọi con số trong file này tính với K = 37.** Nếu `CHARS` của bạn có độ dài khác, hãy thay `K` tương ứng (các công thức theo `K` có ghi ở từng mục).
- Chỉ số lớp là vị trí trong `self.backbone` (khớp biến `i` trong vòng lặp của `forward`).

---

## 1. Bức tranh tổng thể

LPRNet đọc biển số **không cần tách từng ký tự**. Chiều rộng của ảnh được coi như một "trục thời gian": ảnh được nén thành **18 cột**, mỗi cột cho một vector 37 điểm số (36 ký tự + blank). Hàm loss **CTC** tự gộp 18 dự đoán đó thành chuỗi biển số, nên nhãn chỉ cần là chuỗi ký tự, không cần vị trí từng ký tự.

```
Anh 3x24x94
   |
   v
BACKBONE (nn.Sequential, 23 lop, chi so 0..22)
   [0-2]    Conv -> BN -> ReLU .................. lay lam f0 (64x22x92)
   [3]      MaxPool3d
   [4-6]    small_basic_block -> BN -> ReLU ..... lay lam f1 (128x20x90)
   [7]      MaxPool3d
   [8-10]   small_basic_block -> BN -> ReLU
   [11-13]  small_basic_block -> BN -> ReLU ..... lay lam f2 (256x18x44)
   [14]     MaxPool3d
   [15-18]  Dropout -> Conv(1x4) -> BN -> ReLU
   [19-22]  Dropout -> Conv(13x1) -> BN -> ReLU . lay lam f3 (37x4x18)
   |
   v
GLOBAL CONTEXT (trong forward)
   AvgPool2d dua f0, f1, f2 ve 4x18  (f3 da la 4x18)
   moi nhanh chia cho mean(f^2) cua chinh no
   torch.cat theo kenh: 64 + 128 + 256 + 37 = 485 kenh x 4 x 18
   |
   v
container: Conv2d 1x1 (485 -> 37)
   |
   v
torch.mean(dim=2): 4 hang thanh 1  ->  logits 37 x 18
   |
   v
(file train) permute -> log_softmax -> CTCLoss
```

Hai ý tưởng chính:

1. **Backbone** trích đặc trưng và thu nhỏ dần chiều cao (24 → 22 → 20 → 18 → 16 → 4) và chiều rộng (94 → 18).
2. **Global context**: không chỉ dùng đầu ra cuối của backbone mà lấy thêm đặc trưng ở tầng nông (f0, f1) và tầng giữa (f2), trộn lại để dự đoán cuối cùng. Tầng nông giữ chi tiết nét chữ, tầng sâu giữ thông tin về lớp ký tự.

---

## 2. Bảng shape và số tham số từng lớp

Dòng có ghi "lấy làm f0..f3" là các lớp có đầu ra được giữ trong `keep_features` (các chỉ số 2, 6, 13, 22).

| Chỉ số | Lớp | Shape đầu ra | Số tham số học được |
|---|---|---|---|
| input | `Ảnh đầu vào` | `3×24×94` |  |
| 0 | `Conv2d(3→64, k=3×3)` | `64×22×92` | 1,792 |
| 1 | `BatchNorm2d(64)` | `64×22×92` | 128 |
| 2 | `ReLU`  (lấy làm f0) | `64×22×92` | 0 |
| 3 | `MaxPool3d(k=(1,3,3), s=(1,1,1))` | `64×20×90` | 0 |
| 4 | `small_basic_block(64→128)` | `128×20×90` | 12,512 |
| 5 | `BatchNorm2d(128)` | `128×20×90` | 256 |
| 6 | `ReLU`  (lấy làm f1) | `128×20×90` | 0 |
| 7 | `MaxPool3d(k=(1,3,3), s=(2,1,2))` | `64×18×44` | 0 |
| 8 | `small_basic_block(64→256)` | `256×18×44` | 45,504 |
| 9 | `BatchNorm2d(256)` | `256×18×44` | 512 |
| 10 | `ReLU` | `256×18×44` | 0 |
| 11 | `small_basic_block(256→256)` | `256×18×44` | 57,792 |
| 12 | `BatchNorm2d(256)` | `256×18×44` | 512 |
| 13 | `ReLU`  (lấy làm f2) | `256×18×44` | 0 |
| 14 | `MaxPool3d(k=(1,3,3), s=(4,1,2))` | `64×16×21` | 0 |
| 15 | `Dropout(p)` | `64×16×21` | 0 |
| 16 | `Conv2d(64→256, k=1×4)` | `256×16×18` | 65,792 |
| 17 | `BatchNorm2d(256)` | `256×16×18` | 512 |
| 18 | `ReLU` | `256×16×18` | 0 |
| 19 | `Dropout(p)` | `256×16×18` | 0 |
| 20 | `Conv2d(256→K, k=13×1)` | `37×4×18` | 123,173 |
| 21 | `BatchNorm2d(K)` | `37×4×18` | 74 |
| 22 | `ReLU`  (lấy làm f3) | `37×4×18` | 0 |
| forward | `AvgPool2d(k=5, s=5)` trên f0 | `64×4×18` | 0 |
| forward | `AvgPool2d(k=5, s=5)` trên f1 | `128×4×18` | 0 |
| forward | `AvgPool2d(k=(4,10), s=(4,2))` trên f2 | `256×4×18` | 0 |
| forward | chuẩn hóa f/mean(f²) rồi `torch.cat(dim=1)` | `485×4×18` | 0 |
| container | `Conv2d(485→K, k=1×1)` | `37×4×18` | 17,982 |
| forward | `torch.mean(dim=2)` | `37×18` | 0 |
| | **Tổng tham số học được** | | **326,541** |

Chỉ đếm tham số học được bằng gradient. `BatchNorm2d` còn có các buffer `running_mean`, `running_var`, `num_batches_tracked` (không học bằng gradient nên không nằm trong tổng).

Công thức tổng quát theo `K`: lớp chỉ số 20 có `256·13·K + K` tham số, mỗi `BatchNorm2d(K)` có `2K`, `container` có `(448+K)·K + K`.

---

## 3. Từng loại lớp

### 3.1 `Conv2d` (tích chập 2D)

**Khái niệm.** Một khối trọng số nhỏ (kernel) trượt trên ảnh. Tại mỗi vị trí, nó nhân từng phần tử với vùng ảnh bên dưới rồi cộng tổng lại thành một điểm của bản đồ đặc trưng. Mỗi kernel học để phát hiện một mẫu cục bộ như cạnh, nét ngang, nét dọc, góc.

**Vai trò.** Là khối trích đặc trưng chính. Các lớp đầu học nét đơn giản, các lớp sau học mẫu phức tạp hơn. Một kernel dùng chung cho mọi vị trí nên số tham số ít và model nhận ra mẫu ở bất kỳ chỗ nào trên ảnh.

**Công thức.**

```
out[n, o, i, j] = bias[o]
                + sum_{c=0..Cin-1} sum_{p=0..kH-1} sum_{q=0..kW-1}
                  W[o, c, p, q] * x_pad[n, c, i*sH + p, j*sW + q]

x_pad = x sau khi them padding bang 0 (dilation = 1 trong model nay)

Hout = floor( (Hin + 2*padH - dilH*(kH - 1) - 1) / sH + 1 )
Wout = floor( (Win + 2*padW - dilW*(kW - 1) - 1) / sW + 1 )

So tham so = Cout * Cin * kH * kW + Cout
```

**Theo tài liệu PyTorch.** `Conv2d` thực chất tính **cross-correlation** (không lật kernel như tích chập toán học). Đây là cách mọi framework deep learning gọi "convolution".

**Các conv trong model của bạn.**

- Chỉ số 0, `Conv2d(3→64, k=3×3)`: nhìn thẳng vào pixel RGB, học bộ lọc cạnh và màu cơ bản. Không padding nên `24×94` thành `22×92`.
- Chỉ số 16, `Conv2d(64→256, k=1×4)`: kernel cao 1, rộng 4, chỉ gộp thông tin theo **chiều ngang** (4 cột liền nhau). Mỗi cột nhìn được ngữ cảnh các cột bên cạnh, hợp với việc đọc một chuỗi ký tự. Chiều rộng `21 → 18`.
- Chỉ số 20, `Conv2d(256→K, k=13×1)`: kernel cao 13, rộng 1, gộp thông tin theo **chiều dọc**. Chiều cao `16 → 4`, đồng thời đổi số kênh từ 256 sang K, tức mỗi điểm bắt đầu mang "điểm số cho K lớp". Đây là lớp nhiều tham số nhất model.
- Bốn conv bên trong mỗi `small_basic_block` và `container` (xem 3.6 và 3.10).

**Conv 1×1** là trường hợp đặc biệt: không nhìn các điểm lân cận mà chỉ trộn các kênh tại cùng một vị trí.

```
y[o, i, j] = b[o] + sum_c W[o, c] * x[c, i, j]
```

Nó hoạt động như một lớp fully-connected áp dụng độc lập lên từng điểm ảnh, dùng để đổi số kênh (nén hoặc mở rộng) với chi phí thấp.

---

### 3.2 `BatchNorm2d` (chuẩn hóa theo batch)

**Khái niệm.** Với mỗi kênh, chuẩn hóa các giá trị trong batch về trung bình 0 và phương sai 1, rồi cho model tự học lại độ co giãn `gamma` và độ dịch `beta`.

**Vai trò.** Giữ phân phối đầu vào của mỗi lớp ổn định khi train, nên train nhanh hơn, chịu được learning rate lớn hơn và đỡ nhạy với khởi tạo trọng số. `BatchNorm2d(C)` có `2C` tham số học được.

**Công thức (chế độ train).** Thống kê tính theo từng kênh `c`, trên các chiều `(N, H, W)`:

```
mu_c  = (1 / (N*H*W)) * sum_{n,i,j} x[n, c, i, j]
var_c = (1 / (N*H*W)) * sum_{n,i,j} (x[n, c, i, j] - mu_c)^2        <- phuong sai LECH (biased)

y[n, c, i, j] = (x[n, c, i, j] - mu_c) / sqrt(var_c + eps) * gamma_c + beta_c

eps = 1e-5 (mac dinh)
```

**Cập nhật thống kê chạy** (momentum `m = 0.1` mặc định):

```
running_mean <- (1 - m) * running_mean + m * mu_c
running_var  <- (1 - m) * running_var  + m * var_unbiased_c

var_unbiased_c = var_c * (N*H*W) / (N*H*W - 1)        <- phuong sai KHONG LECH
```

**Chế độ eval.** Không dùng thống kê của batch mà dùng thống kê chạy đã tích lũy lúc train:

```
y = (x - running_mean) / sqrt(running_var + eps) * gamma + beta
```

**Theo tài liệu PyTorch.** Lúc train, độ lệch chuẩn dùng để chuẩn hóa tính bằng ước lượng **có chệch** (`torch.var(input, correction=0)`), còn giá trị lưu vào thống kê chạy tính bằng ước lượng **không chệch** (`correction=1`). Đây là lý do phải gọi `model.eval()` khi đánh giá hoặc nhận dạng: nếu quên, kết quả phụ thuộc vào các ảnh khác trong batch và sẽ sai.

**Trong model:** đứng sau mỗi conv hoặc `small_basic_block` và trước `ReLU` (chỉ số 1, 5, 9, 12, 17, 21).

---

### 3.3 `ReLU` (hàm kích hoạt)

**Khái niệm.** Giữ nguyên giá trị dương, đưa giá trị âm về 0.

**Vai trò.** Thêm tính **phi tuyến**. Nếu thiếu nó, chồng nhiều conv liên tiếp chỉ tương đương một conv duy nhất. ReLU tính nhanh và ít bị triệt tiêu gradient hơn sigmoid hoặc tanh.

**Công thức.**

```
ReLU(x) = max(0, x)

d ReLU / dx = 1 neu x > 0, nguoc lai bang 0
```

**Lưu ý trong model.** `ReLU` ở chỉ số 22 (cuối backbone) làm cho f3 không âm. Logits cuối cùng vẫn có thể âm vì phía sau còn `container` là một conv không có ReLU.

---

### 3.4 `MaxPool3d` (gộp cực đại) và điểm đặc biệt với input 4D

**Khái niệm.** Chia dữ liệu thành các cửa sổ nhỏ rồi chỉ giữ giá trị **lớn nhất** trong mỗi cửa sổ.

**Vai trò thông thường.** Giảm kích thước không gian, giữ lại phản hồi mạnh nhất của đặc trưng, giúp model chịu được lệch vị trí nhỏ.

**Công thức (theo tài liệu PyTorch).** Với input `(N, C, D, H, W)`, cửa sổ `(kD, kH, kW)` và stride `(sD, sH, sW)`:

```
out[n, c, d, h, w] = max_{a=0..kD-1, b=0..kH-1, e=0..kW-1}
                     input[n, c, sD*d + a, sH*h + b, sW*w + e]

Dout = floor( (Din + 2*pad[0] - dil[0]*(k[0] - 1) - 1) / s[0] + 1 )
Hout = floor( (Hin + 2*pad[1] - dil[1]*(k[1] - 1) - 1) / s[1] + 1 )
Wout = floor( (Win + 2*pad[2] - dil[2]*(k[2] - 1) - 1) / s[2] + 1 )
```

Trong model `padding = 0` và `dilation = 1` (mặc định), nên mỗi chiều rút gọn thành `floor((in - k) / s) + 1`.

**Điểm đặc biệt (quan trọng).** Tài liệu PyTorch ghi input của `MaxPool3d` là `(N, C, D, H, W)` **hoặc** `(C, D, H, W)`. Tensor trong model có dạng 4D `(N, C, H, W)`, nên PyTorch hiểu nó là dạng không có batch `(C', D, H, W)`:

- Chiều batch `N` bị đọc là "kênh" `C'`, nên mỗi ảnh được pool độc lập, không trộn lẫn nhau.
- Số **kênh `C` của model bị đọc là chiều sâu `D`**.

Cửa sổ có `kD = 1` nên không có phép max nào giữa các kênh. Còn `sD` lớn hơn 1 nghĩa là **chỉ lấy một phần các kênh** (kênh `0, sD, 2*sD, ...`):

| Chỉ số | Cửa sổ / stride | Tác dụng lên số kênh | Tác dụng lên H×W |
|---|---|---|---|
| 3 | `(1,3,3)` / `(1,1,1)` | giữ nguyên 64 kênh | `22×92 → 20×90` (max pool 3×3) |
| 7 | `(1,3,3)` / `(2,1,2)` | lấy kênh 0, 2, 4, ... : `128 → 64` | `20×90 → 18×44` |
| 14 | `(1,3,3)` / `(4,1,2)` | lấy kênh 0, 4, 8, ... : `256 → 64` | `18×44 → 16×21` |

Số kênh sau pool tính bằng `floor((C - 1) / sD) + 1`, ví dụ `floor(127/2)+1 = 64` và `floor(255/4)+1 = 64`.

Đó là lý do hai khối `small_basic_block` phía sau mới có `ch_in = 64` (không phải 128 hay 256) và `Conv2d` ở chỉ số 16 có `in_channels = 64`.

Hai điều cần lưu ý:

- Đây là hành vi của bản cài đặt này. Nó **bỏ hẳn** một nửa hoặc ba phần tư số kênh chứ không "nén" chúng. Tài liệu này không khẳng định nó giống thiết kế trong bài báo gốc.
- Cần PyTorch đủ mới để `MaxPool3d` nhận input 4D. Tài liệu của các bản cũ (ví dụ 1.6) chỉ ghi input 5D.

---

### 3.5 `Dropout`

**Khái niệm.** Khi train, ngẫu nhiên "tắt" (đặt về 0) mỗi phần tử với xác suất `p`; các phần tử còn lại được nhân lên để giữ nguyên kỳ vọng.

**Vai trò.** Chống **overfitting**: buộc model không dựa vào một vài nơ-ron cụ thể mà học đặc trưng dư thừa, đáng tin cậy hơn. Model có hai lớp Dropout (chỉ số 15 và 19) đặt trước hai conv cuối, với `p = dropout_rate`. Mặc định của `train_LPRNet.py` là `0.5`.

**Công thức.**

```
Train: y = x * m / (1 - p),     m[k] ~ Bernoulli(1 - p)  (m = 0 voi xac suat p)
Eval : y = x                    (khong lam gi)
```

**Theo tài liệu PyTorch.** Các phần tử được nhân với `1/(1-p)` khi train; khi đánh giá module chỉ là phép đồng nhất. Đây là lý do thứ hai cần `model.eval()` khi nhận dạng.

---

### 3.6 `small_basic_block` (khối bottleneck phân rã)

**Khái niệm.** Một khối gồm 4 conv: **nén kênh → conv dọc → conv ngang → mở rộng kênh**.

```
x (Cin)
  -> Conv 1x1  (Cin -> Cout/4) -> ReLU
  -> Conv 3x1  (Cout/4 -> Cout/4, padding (1,0)) -> ReLU
  -> Conv 1x3  (Cout/4 -> Cout/4, padding (0,1)) -> ReLU
  -> Conv 1x1  (Cout/4 -> Cout)
  -> y (Cout)
```

**Vai trò.** Thay một conv 3×3 đắt tiền bằng cách rẻ hơn mà vẫn nhìn được vùng 3×3:

- Conv 1×1 đầu **nén** số kênh xuống `Cout // 4` (phép chia nguyên trong code).
- Conv 3×1 rồi conv 1×3 cho vùng nhìn (receptive field) 3×3, mỗi cặp kênh chỉ cần `3 + 3 = 6` trọng số thay vì 9. Một cái bắt cấu trúc dọc, một cái bắt cấu trúc ngang. Hai `ReLU` xen kẽ thêm phi tuyến.
- Conv 1×1 cuối **mở rộng** về `Cout` kênh. Ngay sau nó **không có** `ReLU` trong khối, vì `BatchNorm2d` và `ReLU` nằm bên ngoài khối.
- Padding `(1,0)` và `(0,1)` giữ nguyên `H` và `W`, nên cả khối chỉ đổi số kênh.

**Công thức.** Gọi `m = Cout // 4`:

```
y = W4 * ReLU( W3 * ReLU( W2 * ReLU( W1 * x ) ) )      (dau * la conv)

So tham so = (Cin*m + m) + 2*(3*m*m + m) + (m*Cout + Cout)
```

**Ví dụ** (cả ba khối trong model, kiểm tra bằng công thức trên):

| Chỉ số | Khối | `m` | Số tham số |
|---|---|---|---|
| 4 | `small_basic_block(64→128)` | 32 | 12,512 |
| 8 | `small_basic_block(64→256)` | 64 | 45,504 |
| 11 | `small_basic_block(256→256)` | 64 | 57,792 |

Một conv 3×3 thường `256→256` cần `256·256·9 + 256 = 590,080` tham số, nhiều hơn khối `256→256` khoảng 10 lần.

---

### 3.7 `AvgPool2d` (gộp trung bình)

**Khái niệm.** Giống max pooling nhưng lấy **trung bình** của cửa sổ thay vì cực đại.

**Vai trò.** Trong model, nó **thu nhỏ f0, f1, f2 về cùng cỡ `4×18`** để ghép được với f3. Dùng trung bình (thay vì max) để tóm tắt cả vùng rộng một cách mượt.

**Công thức.**

```
out[c, i, j] = (1 / (kH*kW)) * sum_{p=0..kH-1} sum_{q=0..kW-1} x[c, i*sH + p, j*sW + q]

Hout = floor( (Hin - kH) / sH + 1 )      (padding = 0, dilation khong co)
Wout = floor( (Win - kW) / sW + 1 )
```

`AvgPool2d` không đổi số kênh.

| Đặc trưng | Cửa sổ / stride | Shape vào → ra |
|---|---|---|
| f0 | `5×5` / `5` | `64×22×92 → 64×4×18` |
| f1 | `5×5` / `5` | `128×20×90 → 128×4×18` |
| f2 | `(4,10)` / `(4,2)` | `256×18×44 → 256×4×18` |
| f3 | không pool | `37×4×18` |

Với f0: `floor((22-5)/5)+1 = 4` và `floor((92-5)/5)+1 = 18`. Vì `floor`, một phần rìa không đủ cửa sổ bị bỏ qua.

---

### 3.8 Chuẩn hóa "global context" (`f / mean(f²)`)

**Khái niệm.** Với mỗi trong 4 đặc trưng, tính **năng lượng trung bình** (trung bình của bình phương) trên toàn bộ tensor, rồi chia đặc trưng cho giá trị đó.

**Trong code:**

```python
f_pow  = torch.pow(f, 2)
f_mean = torch.mean(f_pow)     # khong co dim: mot so vo huong cho CA tensor (ke ca chieu batch)
f      = torch.div(f, f_mean)
```

**Công thức:**

```
fbar = (1 / (N*C*H*W)) * sum_{n,c,i,j} f[n, c, i, j]^2
f'   = f / fbar
```

**Vai trò.** Bốn nhánh có biên độ giá trị rất khác nhau (f0 là đặc trưng nông, f3 gần với điểm số cuối). Nếu ghép thẳng, nhánh có giá trị lớn sẽ lấn át phần còn lại. Phép chia này **cân bằng biên độ** giữa các nhánh để `container` trộn chúng hợp lý. Phép này **không có tham số học**.

**Lưu ý:**

- Chia cho `fbar` (trung bình bình phương) chứ không phải căn bậc hai của nó, nên đây **không** phải chuẩn hóa về đơn vị năng lượng chính xác, chỉ là cách cân bằng biên độ tương đối.
- `fbar` tính trên cả batch, nên kết quả của một ảnh phụ thuộc nhẹ vào các ảnh còn lại trong batch, **kể cả khi `eval()`**. Nhận dạng một ảnh đơn lẻ có thể lệch chút so với khi đánh giá theo batch.
- Nếu một đặc trưng toàn bằng 0 (hiếm), `fbar = 0` và phép chia cho ra `nan`.

---

### 3.9 `torch.cat` (ghép kênh)

Nối 4 đặc trưng cùng cỡ `4×18` theo chiều kênh (`dim=1`):

```
X = cat([f0', f1', f2', f3'], dim=1)  ->  (64 + 128 + 256 + K) x 4 x 18 = 485 x 4 x 18
```

**Vai trò.** Gom thông tin từ nhiều mức trừu tượng vào một tensor. `torch.cat` yêu cầu các chiều còn lại (`4×18`) phải trùng nhau, nếu ảnh sai kích thước thì ở đây sẽ báo lỗi. Con số `448 + class_num` trong `container` chính là `64 + 128 + 256 + K`; vì code viết `448 + self.class_num` nên đổi `class_num` thì số kênh vào `container` tự đổi theo.

---

### 3.10 `container`: Conv 1×1 trộn đặc trưng

`Conv2d(485 → K, kernel 1×1)`:

```
z[k, i, j] = b[k] + sum_{c=1..485} W[k, c] * X[c, i, j],     k = 1..K
```

**Vai trò.** Học cách **kết hợp** 485 kênh (từ cả 4 mức đặc trưng) thành K điểm số cho mỗi vị trí `(hàng, cột)`. Số tham số: `485·37 + 37 = 17,982`.

---

### 3.11 `torch.mean(x, dim=2)` (trung bình theo chiều cao)

```
logits[n, k, j] = (1/4) * sum_{i=1..4} z[n, k, i, j]
```

**Vai trò.** Gộp 4 hàng thành 1, nên mỗi **cột** `j` (từ 1 đến 18) có đúng một vector K điểm số. Kết quả là chuỗi dài `T = 18` bước, mỗi bước K lớp, đúng định dạng CTC cần. Đầu ra của model có shape `N×K×18`.

---

### 3.12 Ngoài model nhưng cần để hiểu đầu ra: `log_softmax` và `CTCLoss`

Model trả về **logits** (chưa qua softmax). Trong `train_LPRNet.py`:

```python
log_probs = logits.permute(2, 0, 1)            # N x K x T  ->  T x N x K
log_probs = log_probs.log_softmax(2)           # log_softmax theo chieu lop
ctc_loss  = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean')
loss = ctc_loss(log_probs, labels, input_lengths, target_lengths)   # T_length = 18
```

**Theo tài liệu PyTorch**, `CTCLoss` nhận `log_probs` có shape `(T, N, C)` (hoặc `(T, C)`), trong đó `C` gồm cả blank; `targets` không được chứa blank; mỗi `input_lengths` phải nhỏ hơn hoặc bằng `T`. Code của bạn đặt `blank = len(CHARS) - 1`, tức chỉ số 36 khi `K = 37`.

**Softmax tại mỗi bước thời gian:**

```
p_t(k) = exp(logits[k, t]) / sum_{k'=1..K} exp(logits[k', t])
```

**CTC (Connectionist Temporal Classification).** Một đường đi (path) `pi = (pi_1, ..., pi_18)` là một chuỗi nhãn tại từng bước. Hàm `B` biến đường đi thành chuỗi cuối bằng cách **gộp ký tự lặp liên tiếp, rồi bỏ blank**. Ví dụ với `-` là blank: `3 3 - 0 A A - -` thành `30A`. Xác suất của chuỗi đích `y` là tổng xác suất mọi đường đi dẫn đến nó:

```
P(y | x) = sum_{pi in B^-1(y)}  prod_{t=1..18} p_t(pi_t)

Loss = - ln P(y | x)
```

Tổng này được tính hiệu quả bằng quy hoạch động (thuật toán forward-backward).

**`reduction='mean'`.** Theo tài liệu PyTorch, loss của từng mẫu được **chia cho độ dài nhãn của mẫu đó**, rồi lấy trung bình theo batch.

**Vai trò của blank.** Blank cho phép model "không ghi gì" ở một cột, và tách hai ký tự giống nhau đứng liền nhau (ví dụ `11` cần `1 - 1`). Vì vậy cần:

```
T >= L + (so cap ky tu giong nhau lien ke)
```

Với `T = 18` và biển số 8-9 ký tự thì rất thoải mái. Nếu đầu vào quá ngắn so với nhãn, tài liệu PyTorch cho biết loss có thể là vô cùng; code của bạn đã bỏ qua batch có loss `nan` hoặc `inf`.

**Giải mã greedy (khi đánh giá):** lấy `argmax` mỗi cột, gộp ký tự lặp, bỏ blank:

```
y_hat = B( argmax_k p_1(k), ..., argmax_k p_18(k) )
```

---

## 4. Đi qua model theo thứ tự thực thi

1. **Chỉ số 0-2, Conv(3→64) + BN + ReLU**: học 64 bộ lọc cạnh và nét cơ bản trực tiếp từ pixel. Đầu ra giữ làm **f0** (đặc trưng nông, nhiều chi tiết nhất).
2. **Chỉ số 3, MaxPool3d**: thu nhỏ nhẹ `22×92 → 20×90`, giữ nguyên 64 kênh.
3. **Chỉ số 4-6, small_basic_block(64→128) + BN + ReLU**: mở rộng lên 128 kênh. Đầu ra là **f1**.
4. **Chỉ số 7, MaxPool3d stride (2,1,2)**: lấy một nửa số kênh (`128 → 64`) và giảm chiều rộng một nửa (`90 → 44`), rút ngắn trục "thời gian".
5. **Chỉ số 8-10, small_basic_block(64→256) + BN + ReLU**: nâng lên 256 kênh để tăng năng lực biểu diễn.
6. **Chỉ số 11-13, small_basic_block(256→256) + BN + ReLU**: tinh chỉnh đặc trưng. Đầu ra là **f2**.
7. **Chỉ số 14, MaxPool3d stride (4,1,2)**: lấy một phần tư số kênh (`256 → 64`), rộng `44 → 21`.
8. **Chỉ số 15, Dropout**: điều chuẩn, chống overfit.
9. **Chỉ số 16-18, Conv(1×4) + BN + ReLU**: gộp ngữ cảnh 4 cột liền kề (`21 → 18`), lên 256 kênh.
10. **Chỉ số 19, Dropout**: điều chuẩn lần hai.
11. **Chỉ số 20-22, Conv(13×1) + BN + ReLU**: gộp toàn bộ chiều dọc (`16 → 4`) và đổi sang K kênh. Đầu ra là **f3**.
12. **Global context**: AvgPool f0, f1, f2 về `4×18`, chia mỗi nhánh cho `mean(f²)`, rồi nối cả bốn thành `485×4×18`.
13. **container (Conv 1×1)**: trộn 485 kênh thành K điểm số.
14. **torch.mean(dim=2)**: gộp 4 hàng thành 1, ra **logits `K×18`**.
15. **(file train) log_softmax + CTCLoss**: so sánh 18 dự đoán với chuỗi biển số thật để tính loss. Khi nhận dạng thì dùng `argmax`, gộp lặp, bỏ blank.

---

## 5. Phân bố tham số

| Thành phần | Số tham số | Tỷ lệ |
|---|---|---|
| `Conv2d(256→K, 13×1)` (chỉ số 20) | 123,173 | 37.7% |
| `Conv2d(64→256, 1×4)` (chỉ số 16) | 65,792 | 20.1% |
| `small_basic_block(256→256)` (chỉ số 11) | 57,792 | 17.7% |
| `small_basic_block(64→256)` (chỉ số 8) | 45,504 | 13.9% |
| `container` (Conv 1×1) | 17,982 | 5.5% |
| `small_basic_block(64→128)` (chỉ số 4) | 12,512 | 3.8% |
| `Conv2d(3→64, 3×3)` (chỉ số 0) | 1,792 | 0.5% |
| 6 lớp `BatchNorm2d` | 1,994 | 0.6% |
| **Tổng** | **326,541** | 100% |

Model chỉ khoảng **0.33 triệu** tham số, rất nhẹ, phù hợp chạy trên thiết bị yếu.

---

## 6. Những điều cần nhớ khi dùng model

- **Kích thước ảnh phải đúng `24×94`.** Conv `(13,1)` cần chiều cao 16 tại chỉ số 20, và các `AvgPool2d` cần đúng `22×92`, `20×90`, `18×44` để ra `4×18`. Sai kích thước thì `torch.cat` báo lỗi.
- **`class_num` phải bằng `len(CHARS)`** (gồm cả blank). Giá trị mặc định trong `build_lprnet` là 36, nên luôn truyền `class_num` tường minh như `train_LPRNet.py` đang làm.
- **Luôn gọi `model.eval()`** khi đánh giá hoặc nhận dạng, vì `BatchNorm2d` và `Dropout` hoạt động khác nhau giữa train và eval.
- **`lpr_max_len` không được dùng trong bất kỳ lớp nào** của model, nên không ảnh hưởng shape của trọng số. Số 18 bước thời gian do kích thước ảnh quyết định, không phải do tham số này.
- **Đầu ra là logits `N×K×18`**, chưa qua softmax. Khi giải mã chỉ cần `argmax` theo chiều lớp.
- **Kết quả phụ thuộc nhẹ vào batch** do phép chia `f / mean(f²)` (mục 3.8), kể cả khi `eval()`.

---

## 7. Tài liệu PyTorch đã đối chiếu

- Conv2d: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- BatchNorm2d: https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
- MaxPool3d: https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool3d.html
- AvgPool2d: https://docs.pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
- Dropout: https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html
- CTCLoss: https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
