= Thiết kế thực nghiệm và Kết quả

== Thiết lập hệ thống
Mô tả pipeline từ bước sinh ảnh bằng Z-Image-Turbo (GGUF) đến bước đánh giá trên tập NTIRE Robust AI.

== Phân tích định lượng
Trình bày kết quả thông qua ROC-AUC.
Luận điểm: So sánh xem mô hình AIDE (hybrid) hay SPAI (spectral) chống chịu tốt hơn với các biến đổi (nén JPEG, resize).

== Phân tích định tính (Case Study)
Chọn ra các mẫu "uncertainty" cao (điểm số nằm gần biên quyết định $0.5$) để phân tích lý do tại sao detector bị đánh lừa (có thể do ánh sáng quá phức tạp hoặc prompt quá lạ).
