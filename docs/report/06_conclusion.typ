= Kết luận

Báo cáo đã tìm hiểu bài toán phát hiện ảnh giả từ các mô hình phát tán hiện đại, đặc biệt trong điều kiện có sự dịch chuyển phân phối (distribution shift) do các phép biến đổi nhiễu và sự xuất hiện của các mô hình sinh ảnh mới. Qua các thực nghiệm trên tập dữ liệu NTIRE 2026 và ảnh tự sinh từ Z-Image-Turbo, báo cáo rút ra một số kết luận:

- Sự bù trừ giữa các đặc trưng: Phương pháp tập trung vào phổ tần số (SPAI) ổn định trước các nhiễu như nén và mờ, nhưng có thể hoạt động kém khi cấu trúc phổ bị biến dạng do chụp màn hình hoặc nén sâu. Trong khi đó, phương pháp kết hợp ngữ nghĩa và điểm ảnh (AIDE) phát hiện tốt ảnh từ các generator mới chưa từng thấy, nhưng lại dễ bỏ lọt các bức ảnh giả được bố cục quá hợp lý về mặt logic.
- Thách thức trong thực tế: Cả hai mô hình đều có tỷ lệ bỏ lọt cao (Recall thấp) khi thắt chặt tỷ lệ cảnh báo nhầm (FPR $<= 0.05$).

Để khắc phục, báo cáo đề xuất xây dựng một hệ thống kết hợp (ensemble) giữa SPAI và AIDE. Với các trường hợp biên quyết định mỏng manh, hệ thống khai thác độ bất định (uncertainty) dựa trên sự bất đồng giữa các detector và ứng dụng phương pháp Hậu kiểm tại thời điểm suy luận (Test-Time Adaptation - TTA). Những mẫu có độ bất định cao sẽ được chuyển sang quy trình kiểm tra thủ công (human review), giúp đảm bảo an toàn.
