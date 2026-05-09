# AGENTS.md

## Folder structure

- src/diffguard/...: repo-specific modules.
- data/bronze/ntire/: raw NTIRE test-public dataset (2,500 JPGs + test_labels.csv, label=1=fake).
- data/bronze/z_image_turbo/: 100 Z-Image-Turbo generated images (PNGs + metadata.csv).
- data/silver/aide/: AIDE inference outputs (per-image scores, metrics JSON, plots).
- data/silver/spai/: SPAI inference outputs (per-image scores, metrics JSON, config snapshot, plots).
- data/artifacts/checkpoints/: model weights (GenImage_train.pth for AIDE, spai.pth for SPAI).
- notebooks/experiment/: AIDE and SPAI inference notebooks.

## Technical

- Evaluate folder for generated data carefully: bronze, silver, or gold. Do not add another tier.
- Use the appropriate agent skills for the task.
- Python: use `uv run script.py`, `uv run -m ...`. Ask me if you need to modify the python packages.
- To show figure in Jupyter, save the figure in SVG (prefered) or PNG, then display it. Don't use inline visualization like `.show()`.

## Code Convention

- Docstrings: Apply Google-style to all new or modified code. Clearly state the overarching context, purpose, and contract (the "What"), strictly avoiding internal step-by-step implementation details.
- Code Comments: Keep them extremely concise. Explain only the "Why" behind non-obvious logic or business decisions (the "Intent").
- Comment Mapping: Directly relate inline comments back to the overarching context defined in the docstring to maintain readability.
- Logging: Add necessary debug logs using lazy formatting (`%` style). Avoid log flooding in loops.
