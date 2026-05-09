= Cơ sở toán học

== Gen-AI != Pred-AI

Pred-AI thường học một hàm gần như xác định $y = f_theta (x)$ bằng cách cực tiểu hóa sai số dự đoán, ví dụ $min_theta bb(E)[ell(f_theta (x), y)]$

Gen-AI học phân phối dữ liệu và sinh mẫu theo dạng $y = f_theta (x, z)$, trong đó $z$ là nhiễu. Để có thể dùng lan truyền ngược, cần dùng phép tái tham số hóa: $y = mu_theta (x) + sigma_theta (x) z$. Vì vậy ảnh sinh ra không có nhãn, mà là một mẫu từ một phân phối học được. Bài toán phát hiện fake data vì thế được nhìn như bài toán phân biệt hai phân phối $P_"real"$ và $P_"fake"$, chứ không phải kiểm tra watermark hay tên generator.

== Diffusion khử nhiễu để tạo ảnh, còn RL giúp ảnh giống thật

Với mô hình phát tán, quá trình forward thêm nhiễu vào ảnh thật:
$
  X_t = sqrt(overline(alpha)_t) X_0 + sqrt(1 - overline(alpha)_t) overline(epsilon)_t
$
sau đó mô hình học mạng $epsilon_theta approx overline(epsilon)_t$ để đi ngược lại:
$
  X_(t-1) = 1/sqrt(alpha_t) ( X_t - (1 - alpha_t)/sqrt(1 - overline(alpha)_t) epsilon_theta ) + sigma_q (t)z.
$

Theo bài toán tối ưu, phần học khử nhiễu là cực tiểu hóa loss kiểu dự đoán như $theta^* = min_theta cal(L)(theta)$. Nhưng với các mô hình mới như Z-Image-Turbo @zimage2025, quá trình này còn bị kéo bởi mục tiêu cực đại hóa phần thưởng: $E[sum_t gamma^t R(s_t, a_t)]$, trong đó reward là độ đẹp, độ giống thật, độ khớp prompt. Mô hình vừa học để khử nhiễu đúng, vừa học để "lừa mắt người" tốt hơn.

== Hướng phát hiện: tìm dấu vết ổn định của ảnh thật
Ảnh thật và ảnh AI không hoàn toàn giống nhau về mặt thống kê. Vì vậy detector sẽ gán cho mỗi ảnh một điểm số $S(x)$: điểm càng cao thì ảnh có khả năng fake. Mỗi detector chịu trách nhiệm cho một nhóm đặc điểm khó giả của ảnh thật: phổ/tần số, nhiễu cảm biến, độ mịn, hoặc đặc trưng ngữ nghĩa sâu.

Khi các detector cho kết quả trái nhau thì ảnh đó nằm trong vùng mơ hồ. Khi đó, đưa một ảnh qua vài biến đổi nhẹ như crop, resize, nén JPEG, rồi chạy detector nhiều lần. Nếu kết quả vẫn ổn định thì đáng tin hơn; nếu điểm số vẫn dao động sau các biến đổi nhỏ thì cần hậu kiểm.
