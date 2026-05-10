= Đề xuất giải pháp: Passive Detection và TTA

Từ thực nghiệm trên, điểm rõ nhất là không có một dấu hiệu đơn lẻ nào đủ ổn định. Vì vậy, giải pháp phòng chống dữ liệu giả hợp lý là xây dựng một hệ thống kết hợp nhiều phương pháp với nhau (ensemble).
$
  S(x) = alpha S_"SPAI" (x) + beta S_"AIDE" (x)
$
Kết quả thực nghiệm cho thấy SPAI và AIDE này bổ sung nhau: SPAI tốt hơn trên dữ liệu từ môi trường thực, còn AIDE nhảy cảm hơn với tập ảnh từ mô hình phát tán.

Khi hai phương pháp cho điểm trái nhau, ảnh đó nên được kiểm tra thêm. Báo cáo dùng Hậu kiểm tại thời điểm suy luận (Test-Time Adaptation - TTA). Vì cả AIDE và SPAI đều có những vùng không tự tin (uncertainty), ta lấy độ lệch điểm giữa chúng làm tín hiệu: $U(x) = abs(S_"AIDE" - S_"SPAI")$. Chủ động hơn, đưa ảnh $x$ qua $K$ phép biến đổi nhiễu nhỏ $a_i$ (crop, nén, thêm noise) và đo phương sai của phán đoán:
$
  U_"tta" (x) = "Var"_i (S(a_i (x)))
$

#figure(
  image("figures/uncertainty_proxy.svg", width: 100%),
  caption: [Độ bất định ước lượng bằng bất đồng giữa AIDE/SPAI và khoảng cách tới ngưỡng quyết định.],
) <fig:uncertainty>

Trong @fig:uncertainty, biểu đồ bên trái cho thấy độ bất đồng $abs(S_"AIDE" - S_"SPAI")$. Có nhiều mẫu có độ bất đồng lớn, nghĩa là hai detector đưa ra đánh giá rất khác nhau trên cùng một ảnh. Đây là tín hiệu rủi ro rõ nhất trong thực nghiệm.

Biểu đồ bên phải đo khoảng cách từ điểm trung bình $(S_"AIDE" + S_"SPAI") / 2$ đến ngưỡng $0.5$. Phân phối này cho thấy nhiều mẫu có điểm trung bình nằm khá xa ngưỡng, nhưng điều đó chưa đủ để kết luận an toàn nếu hai detector đang mâu thuẫn mạnh. Việc xác định ảnh giả nên dựa trên cả khoảng cách tới ngưỡng và theo độ bất đồng giữa các detector. Các mẫu có bất đồng lớn nên được chuyển sang hậu kiểm.

Khi $U_"tta"(x)$ cao, có nghĩa là đường biên quyết định của 2 mô hình đang cực kỳ mỏng manh đối với mẫu $x$ này. Quyết định cuối cùng của hệ thống nên là:
$
  hat(y) =
  cases(
    1"," S(x) > tau_h " và " U(x) < epsilon,
    0"," S(x) < tau_l " và " U(x) < epsilon,
    "human review," "ngược lại",
  )
$
