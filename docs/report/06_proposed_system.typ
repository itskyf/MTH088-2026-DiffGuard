= Đề xuất hệ thống ngăn chặn dữ liệu giả mạo (Phần quan trọng nhất của Task)

== Kiến trúc hệ thống đa tầng (Multi-layered Defense)
- Tầng 1: Sanity check (Entropy, Laplacian) để lọc nhanh các mẫu fake thô sơ.
- Tầng 2: Hybrid Detection (SPAI + AIDE) để phân tích sâu.

== Cơ chế lọc dựa trên Uncertainty
Đề xuất một ngưỡng (threshold) $tau$. Nếu $U(x) > tau$, hệ thống sẽ không đưa ra kết luận tự động mà chuyển sang hậu kiểm (human-in-the-loop).

== Triết lý "Phòng thủ chiều sâu"
Khẳng định phòng chống fake data là một cuộc đua vũ trang (arms race). Giải pháp không phải là một mô hình tĩnh, mà là một quy trình cập nhật liên tục dựa trên các bất biến toán học.
