= Phương pháp phát hiện ảnh giả

Để xác định ảnh tạo bởi Gen-AI, cần xác định các dấu vết mà quá trình tạo sinh để lại. Giả sử ảnh thật và ảnh fake là hai phân phối khác nhau. Detector không cần biết chính xác ảnh đến từ đâu, mà đo xem ảnh đó có tuân theo các quy luật thống kê của ảnh thật hay không.

Xem ảnh thật được lấy từ $P_"real"$, còn ảnh sinh ra được lấy từ $P_"fake"$. Quan trọng là $P_"fake"$ không cố định. Hôm nay có GAN, diffusion; ngày mai có LoRA mới, hoặc ảnh đã qua nén và resize. Detector nhận ảnh $x$ và trả về điểm fake $S(x) in [0, 1]$. Vì vậy detector không nên tìm dấu vết cụ thể của một generator. Nó cần dựa vào các thông tin ổn định hơn, ví dụ phổ ảnh, nhiễu, độ mịn, entropy và cả đặc trưng ngữ nghĩa sâu.

Báo cáo kết hợp hai hướng bổ sung nhau:
- SPAI @spai2025 tập trung vào miền phổ và xem ảnh fake như mẫu lệch khỏi phân phối ảnh thật.
- AIDE @aide2025 thiên về deep learning, kết hợp nhiều loại dấu vết hơn: tần số, noise pattern và đặc trưng ngữ nghĩa từ OpenCLIP.

== SPAI: học phân phối phổ của ảnh thật

Với ảnh $x$, ta có thể viết biến đổi Fourier là:

$
  chi = cal(F)(x)
$

Trong miền tần số, các thành phần thông thấp thường chứa bố cục và vùng màu lớn, còn thành phần thông cao chứa cạnh, texture nhỏ, nhiễu và các chi tiết mịn. SPAI dùng một mask $M$ để tách hai phần này:

$
  x^h = cal(F)^(-1)(chi dot.o M)
$

$
  x^l = cal(F)^(-1)(chi dot.o (bold(1) - M))
$

Cách làm này có liên hệ khá tự nhiên với mô hình phát tán. Diffusion học quá trình khử nhiễu để đi từ $X_t$ về $X_0$. Còn SPAI học một dạng bài toán tái tạo trong miền phổ: từ một phần phổ của ảnh thật, mô hình phải phục hồi lại quan hệ phổ hợp lý. Nếu ảnh thuộc phân phối thật, việc tái tạo và so sánh phổ sẽ ổn định hơn. Nếu ảnh là ảnh mô hình tạo ra, các quan hệ này có thể bị lệch.

Có thể viết đơn giản quá trình học của SPAI như sau:

$
  hat(x) = cal(H)(cal(G)(cal(B)(x^l, x^h)))
$

với loss tái tạo phổ:

$
  cal(L)_"rec" = cal(D)(cal(F)(x), cal(F)(hat(x)))
$

Sau quá trình huấn luyện, phần head tái tạo $cal(H)$ không còn được dùng. Backbone $cal(G)$ được dùng như mô hình đã học đặc trưng phổ của ảnh thật. Vì vậy score của SPAI có thể hiểu nôm na là:

$
  S_"SPAI" (x) approx 1 - "sim"("spectral pattern of " x, "real spectral pattern")
$

Nếu $S$ cao, ảnh đó không còn giống ảnh thật trong miền phổ. Điểm này rất hợp với mục tiêu tổng quát hóa, vì SPAI không cần biết ảnh fake đến từ đâu. Nó chỉ cần đo xem phổ của ảnh có tuân theo quy luật của ảnh thật không.

Tuy nhiên, anh thật bây giờ cũng bị xử lý rất nhiều bởi điện thoại: HDR, denoise, sharpen, compression. Ngược lại ảnh Gen-AI nếu được chụp lại, hoặc bị chỉnh sửa nhiều lần, có thể nhận thêm dấu vết giống ảnh thật. Vì vậy SPAI là một hướng rất hợp lý về mặt invariant, nhưng vẫn không nên xem là lời giải duy nhất.

== AIDE: không tin một dấu vết duy nhất

Triết lý của AIDE bổ sung cho SPAI: ảnh AI không nhất thiết sai ở cùng một chỗ. Có ảnh sai ở texture, có ảnh sai ở noise, có ảnh sai ở ngữ nghĩa, cũng có ảnh nhìn rất thật nhưng vùng chi tiết lại quá mượt. Vì vậy nếu chỉ nhìn một loại feature thì dễ bị lừa.

=== Trích xuất đặc trưng cục bộ

AIDE cũng dùng miền tần số như SPAI, nhưng xử lý ở mức vi mô bằng cách chia ảnh thành các patch nhỏ:

$
  bold(I) = {x_1, x_2, ..., x_n}, quad x_i in RR^(N times N times 3)
$

Sau đó dùng phép biến đổi cosin rời rạc (DCT) cho từng patch:

$
  cal(X)_f = {x_1^"dct", x_2^"dct", ..., x_n^"dct"}
$

Những patch có tần số rất cao có thể chứa cạnh, texture, noise hoặc artifact; còn những patch có tần số rất thấp có thể là vùng quá phẳng, quá mịn. AIDE chọn cả hai nhóm cực trị này:

$
  X_"max" = "TopK-high-DCT"(x), quad
  X_"min" = "TopK-low-DCT"(x)
$

Nhờ các patch cực trị này, mô hình AIDE dễ tập trung vào những vùng có bất thường. Các patch này đi qua SRM để lấy noise residual, rồi qua ResNet để tạo đặc trưng low-level:

$
  z_"low" = f_"SRM/ResNet" (X_"max", X_"min")
$

Nhánh này tương tự các đặc trưng thống kê đơn giản như entropy, Laplacian variance, edge density và DCT high-frequency ratio. Chúng không đủ để kết luận ảnh thật/giả một mình, nhưng giúp giải thích dấu vết fake: ảnh có thể quá mượt, thiếu nhiễu tự nhiên, hoặc có phân bố tần số cao bất thường.

=== Trích xuất đặc trưng ngữ nghĩa

Bên cạnh nhánh low-level, AIDE còn dùng OpenCLIP để lấy đặc trưng ngữ nghĩa:

$
  z_"sem" = f_"OpenCLIP" (x)
$

Nhánh này nhìn ảnh ở mức cao hơn, ví dụ bố cục, object, scene và tính hợp lý về mặt nội dung.


=== Kết hợp đặc trưng

Các mô hình diffusion hiện đại sinh ảnh rất đẹp và sát với prompt. Vì vậy AIDE nối hai loại thông tin cục bộ và ngữ nghĩa lại:

$
  S_"AIDE" (x) =
  sigma("MLP"([z_"low"; z_"sem"]))
$

Điều này hợp lý với ảnh sinh hiện đại: phần ngữ nghĩa có thể rất thuyết phục, nhưng các đặc trưng mức thấp như nhiễu, texture hoặc phân bố tần số vẫn có khả năng để lại dấu vết.

Tuy nhiên, AIDE vẫn là mô hình học có giám sát, nên không tránh khỏi rủi ro bị lệ thuộc vào dữ liệu huấn luyện. Nếu checkpoint được train chủ yếu trên một số generator nhất định, kết quả trên generator mới có thể giảm. Các đặc trưng dựa trên DCT/patch cũng dễ bị ảnh hưởng bởi JPEG, resize, blur hoặc các bước hậu xử lý khác. Vì vậy, báo cáo xem AIDE là một detector lai có giá trị tham khảo mạnh, nhưng không dùng để kết luận tuyệt đối.

== Kết hợp 2 phương pháp

Điểm tổng quát là:

$
  S(x) = alpha S_"SPAI" (x) + beta S_"AIDE" (x)
$

- Trường hợp $S_"AIDE" (x)$ và $S_"SPAI" (x)$ đều cao, thì kết luận fake có thêm bằng chứng giải thích được.
- Trường hai detector mâu thuẫn, ví dụ như AIDE cho điểm fake thấp nhưng SPAI cho điểm fake cao (ảnh không sai nhiều về ngữ nghĩa/noise local, nhưng lại lệch trong miền phổ). Do có sự mâu thuẫn, ta tính độ bất định:

$
  U(x) = abs(S_"AIDE" (x) - S_"SPAI" (x))
$

Nếu $U(x)$ lớn, ảnh đó nên được xem là mẫu khó, và chuyển sang các bước hậu kiểm hoặc đánh giá thêm.

== Test-Time Adaptation

Trong thực tế, ảnh có thể bị chỉnh sửa rất nhiều: crop, resize, nén JPEG, blur,... Một biến đổi nhẹ không nên làm điểm số thay đổi quá mạnh. Vì vậy có thể đưa cùng ảnh $x$ qua vài phép biến đổi nhẹ $a_1, a_2, ..., a_K$, rồi lấy trung bình:

$
  mu(x) = 1 / K sum_(i=1)^K S(a_i(x))
$

Độ dao động của các điểm này được xem như uncertainty:

$
  U_"tta" (x) = "Var"_i (S(a_i(x)))
$

$U_"tta" (x)$ cao chứng tỏ ảnh rất nhạy với các phép biến đổi. Khi đó dùng threshold cứng là không an toàn.

Vì vậy quyết định cuối nên có ba vùng:

$
  hat(y) =
  cases(
    1"," S(x) > tau_h " và uncertainty thấp",
    0"," S(x) < tau_l " và uncertainty thấp",
    "review, " "ngược lại",
  )
$

Tóm lại, hai phương pháp này trả lời câu hỏi "Ảnh này có giống một ảnh thật về mặt thống kê không?" SPAI mạnh ở phần phổ của ảnh thật, còn AIDE mạnh ở việc gom nhiều dấu vết lại, từ forensic cue đến semantic cue. Khi thêm uncertainty hoặc TTA, hệ thống cho biết ảnh nào mơ hồ thì nên giữ lại để kiểm tra thêm.
