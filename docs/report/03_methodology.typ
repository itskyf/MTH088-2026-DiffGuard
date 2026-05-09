= Các phương pháp phát hiện dựa trên các đặc trưng bất biến

== Phân tích miền phổ (Spectral Analysis) và SPAI
- Tại sao lại là miền phổ? Ảnh thực có phân phối năng lượng tần số cao tuân theo quy luật tự nhiên của cảm biến (sensor noise), trong khi ảnh AI thường bị "lỗi" ở các dải tần cao do quá trình upsampling/convolution.
- Dùng toán học miền tần số (DCT) để giải thích cách SPAI học "reconstruction error" trên phổ ảnh thật.

== Kết hợp giữa đặc trưng hình thái và ngữ nghĩa (AIDE)
- Luận điểm: Chỉ dùng deep learning là chưa đủ vì dễ bị overfit vào một generator cụ thể. Cần kết hợp các chỉ số cơ bản như Smoothness (biến phân Laplacian) và Entropy.
- Công thức inline về entropy: $H(x) = -sum p_i log p_i$ để đánh giá độ phức tạp của texture. Ảnh AI đôi khi quá "mượt" (smooth) ở những vùng chi tiết, dẫn đến entropy thấp bất thường.

== Thích nghi tại thời điểm kiểm thử (Test-Time Adaptation - TTA)
Giải thích tại sao việc biến đổi ảnh đầu vào (augmentation) rồi lấy trung bình kết quả lại giúp giảm uncertainty.
