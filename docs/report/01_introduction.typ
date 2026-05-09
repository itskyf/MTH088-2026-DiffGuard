= Giới thiệu

== Sự bùng nổ của Generative AI và bài toán tin cậy dữ liệu
Đặt vấn đề về việc các mô hình như Diffusion hay GAN (cụ thể là Z-Image-Turbo) đã xóa nhòa ranh giới thực-ảo.

== Tại sao Watermarking không phải là "chìa khóa vạn năng"?
Luận điểm chính là tính thực tế. Watermarking yêu cầu sự hợp tác từ phía generator (active approach), điều này bất khả thi với các model mã nguồn mở hoặc generator độc hại. Ngoài ra, các phép biến đổi ảnh (nén, crop) dễ dàng phá hủy watermark.

== Hướng tiếp cận Passive Detection
Khẳng định đây là giải pháp thực dụng nhất. Thay vì tìm dấu vết được "cài cắm", ta đi tìm những "bất biến thống kê" mà các mô hình toán học hiện nay chưa mô phỏng hoàn hảo được so với ảnh thực.
