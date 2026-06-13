# DiffGuard

Thí nghiệm phát hiện ảnh sinh bởi AI trong điều kiện dịch chuyển phân phối. Repo dùng hai detector chính:

- [AIDE](https://github.com/shilinyan99/AIDE): kết hợp đặc trưng low-level/noise/tần số với đặc trưng ngữ nghĩa OpenCLIP.
- [SPAI](https://github.com/kartyg23/spai): tập trung vào bất thường trong miền phổ của ảnh.

Dữ liệu chính gồm NTIRE 2026 test-public (`data/bronze/ntire/`) và 100 ảnh tự sinh bằng Z-Image-Turbo (`data/bronze/z_image_turbo/`). Kết quả suy luận nằm trong `data/silver/`; phân tích tổng hợp và hình cho báo cáo nằm trong `data/gold/` và `docs/report/figures/`.

## Cài môi trường

Dùng [uv](https://docs.astral.sh/uv/getting-started/installation/) package and project manager.

```shell
uv sync
```

## Notebooks

Các notebook trong `notebooks/experiment/` chủ yếu dùng để đọc artifact đã sinh sẵn, vẽ hình và tổng hợp kết quả cho báo cáo. Không notebook nào mặc định chạy lại toàn bộ inference nặng.

### `01_aide_inference.ipynb`

- Đọc `data/silver/aide/ntire_scores.csv`, `z_image_turbo_scores.csv`, `ntire_metrics.json`.
- Tạo/hiển thị histogram score, ROC và PR cho AIDE.
- Cell inference để `RUN_INFERENCE = False`; chỉ bật khi muốn tái sinh score từ checkpoint `data/artifacts/checkpoints/GenImage_train.pth`.

### `02_spai_inference.ipynb`

- Đọc `data/silver/spai/ntire_scores.csv`, `z_image_turbo_scores.csv`, `ntire_metrics.json`, `experiment_config.json`.
- Tạo/hiển thị histogram score, ROC và PR cho SPAI.
- Inference đầy đủ được tách ra CLI/background job; notebook chỉ giữ cell tái chạy khi cần.
- Phần cuối chọn một số case đúng/sai/gần ngưỡng và xuất attention mask/overlay vào `data/gold/spai_attention/`.

### `03_handcrafted_features_and_failure_analysis.ipynb`

- Đọc handcrafted features từ `data/silver/handcrafted/` và kết quả AIDE/SPAI đã lưu.
- Merge thành bảng gold: `merged_ntire.csv`, `merged_z_image_turbo.csv`, `metrics_comparison.csv`, `failure_cases.csv`.
- Vẽ hình dùng trong báo cáo: phân phối score, ROC comparison, boxplot handcrafted features, uncertainty proxy và contact sheet failure cases.
- Bước tính feature nặng nằm ngoài notebook: `uv run python -m diffguard.cli.compute_handcrafted_features`.

## Build báo cáo PDF

Báo cáo chính là `docs/report/main.typ`, include các chương `00_cover.typ` đến `06_conclusion.typ` và bibliography `references.yaml`.

```shell
typst compile docs/report/main.typ docs/report/main.pdf
```

File PDF đầu ra: `docs/report/main.pdf`.
