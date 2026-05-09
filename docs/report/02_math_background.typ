= Cơ sở toán học

== Bản chất xác suất của quá trình tạo sinh
- Phân tích sự khác biệt giữa Pred-AI ($y = f(x)$) và Gen-AI ($y = f(x, z)$).
- Sử dụng công thức tái tham số hóa: $y = mu + sigma z$ để giải thích cách nhiễu được đưa vào hệ thống.

== Cơ chế khuếch tán và sự can thiệp của Reinforcement Learning
- Trình bày công thức Forward/Backward Process của Diffusion.
- Luận điểm: Z-Image-Turbo dùng RL để tối ưu hóa quỹ đạo khử nhiễu (Denosing Trajectory). Việc ép mô hình hội tụ nhanh (ít bước sampling) vô tình tạo ra các artifacts trong miền tần số mà mắt người không thấy nhưng toán học có thể bóc tách.

== Không gian đặc trưng và sự lệch pha phân phối
Giải thích bài toán dưới dạng Out-of-Distribution (OOD) detection. Nếu ảnh thực thuộc phân phối $P_"real"$, ảnh giả thuộc $P_"fake"$, thì mục tiêu là tìm một metric $d(P_"real", P_"fake")$ đủ nhạy.
