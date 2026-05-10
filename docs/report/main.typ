#set text(lang: "vi")
#set heading(numbering: "1.")
#set par(
  justify: true,
)
#include "00_cover.typ"

// Bắt đầu đánh số trang từ đây
#set page(numbering: "1")

#counter(page).update(1)

#outline()

#align(center)[
  _Tôi cam đoan đây là báo cáo công việc của tôi, không phải lấy từ thông minh nhân tạo._
]
#include "01_introduction.typ"
#include "02_math_background.typ"
#include "03_methodology.typ"
#include "04_experiments.typ"
#include "05_discussion.typ"
#include "06_conclusion.typ"

#bibliography("references.yaml")
