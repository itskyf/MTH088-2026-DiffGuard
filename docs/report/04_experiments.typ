= Kết quả thực nghiệm

Báo cáo thiết lập kịch bản để đánh giá bài toán dịch chuyển phân phối (distribution shift).

== Thiết lập hệ thống

*Dataset:*
1. Tập test NTIRE 2026 @ntire2026 ($2500$ ảnh) đại diện cho môi trường thực tế. Trong dataset này, ảnh thật $P_"real"$ và ảnh fake $P_"fake"$ đều đã bị áp dụng ngẫu nhiên các hàm biến đổi nhiễu $cal(T)(x)$ như nén JPEG, blur, hoặc crop.
2. Tự sinh một tập $100$ ảnh bằng Z-Image-Turbo @zimage2025. Tập này đóng vai trò như một phân phối hoàn toàn chưa từng xuất hiện (unseen generator) $P_"unseen"$.

*Thang đo:*
Khi phân phối bị biến đổi, điểm số $S(x)$ của các fake detector cũng sẽ thay đổi. Do đó, báo cáo dùng ROC AUC làm thước đo cốt lõi để đánh giá xác suất mô hình xếp hạng đúng thứ tự $P(S(x_"fake") > S(x_"real"))$, bất chấp ngưỡng quyết định bị trôi dạt bao nhiêu.


== So sánh kết quả của SPAI và AIDE

Trên tập NTIRE chứa nhiều biến đổi nhiễu, SPAI tỏ ra nhỉnh hơn với ROC AUC đạt $0.6428$, so với $0.6127$ của AIDE. Như đã trình bày, SPAI học phân phối phổ của tập ảnh thật $hat(x) = cal(H)(cal(G)(cal(B)(x^l, x^h)))$, giúp giữ được tính bất biến (invariance) tốt hơn khi ảnh bị crop hay nén.

#figure(
  image("figures/roc_comparison.svg", width: 80%),
  caption: [Đường cong ROC của SPAI và AIDE trên tập NTIRE.],
) <fig:roc>

#figure(
  image("figures/ntire_score_distribution.svg", width: 100%),
  caption: [Phân phối điểm số của AIDE và SPAI trên tập NTIRE (real/fake).],
) <fig:ntire_score_dist>

#figure(
  image("figures/zit_score_distribution.svg", width: 100%),
  caption: [Phân phối điểm số của AIDE và SPAI trên tập Z-Image-Turbo (chỉ có ảnh fake).],
) <fig:zit_score_dist>

Tuy nhiên, khi đối diện với tập Z-Image-Turbo (@fig:zit_score_dist), AIDE lại nhạy bén hơn hẳn với điểm trung bình $0.771$, trong khi SPAI chỉ ở $0.720$. Điều này cho thấy giới hạn của việc chỉ nhìn vào phổ. Các mô hình phát tán có thể tái tạo quan hệ tần số khá tự nhiên nhưng lại để sót các cụm nhiễu (noise residuals) hoặc sai lệch nội dung. AIDE kết hợp thông tin low-level từ các patch nhỏ $X_"max", X_"min"$ và thông tin semantic $z_"sem"$, tạo ra màng lọc kép xác định các ảnh fake từ generator mới tốt hơn.

Dù vậy, 2 phương pháp đều có điểm yếu trong trường hợp yêu cầu hệ thống gần như không được báo nhầm ảnh thật là fake, tức $"FPR" = "FP" / ("FP" + "TN") <= 0.05$. Khi đó số ảnh fake được bắt đúng (TP) còn thấp; đồng nghĩa số ảnh fake bị bỏ lọt (FN) vẫn nhiều. Cụ thể, tại $"FPR" <= 0.05$, SPAI chỉ đạt $"Recall" = "TP" / ("TP" + "FN") = 0.1315$, còn AIDE đạt $0.0992$.

== Các trường hợp thất bại

Đặc trưng ngữ nghĩa từ các mô hình như CLIP trong AIDE rất mạnh để hiểu bối cảnh, nhưng lại dễ bỏ qua các chi tiết mức pixel. Nếu một ảnh sinh ra quá hợp lý về mặt logic, nhánh CLIP gần như bị vô hiệu hóa.

Ngược lại, phương pháp SPAI nhạy cảm với miền tần số, nhưng nếu một ảnh fake bị tải lên mạng xã hội, bị nén lại, hoặc chụp màn hình (derivative images), cấu trúc phổ nguyên bản sẽ bị thay đổi. Quá trình hậu xử lý này ép phổ của ảnh fake về gần giống phổ của ảnh thật (từ cảm biến camera chụp màn hình), khiến $S_"SPAI" approx 0$.

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("figures/spai_attn_tp.png", width: 100%),
    image("figures/spai_attn_miss.png", width: 100%),
  ),
  caption: [Ảnh trái: SPAI tự tin phát hiện ảnh giả nhờ tập trung vào các vùng sai lệch phổ. Ảnh phải: SPAI thất bại hoàn toàn trên một ảnh fake khác do phổ bị bóp méo bởi nén JPEG.],
) <fig:spai_attn>


#figure(
  image("figures/failure_cases_contact_sheet.png", width: 100%),
  caption: [Các trường hợp 2 phương pháp thất bại hoặc cho điểm số mâu thuẫn.],
) <fig:failure>
