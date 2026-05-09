# AGENTS.md

## Folder structure

- src/diffguard/...: repo-specific modules.
- data/bronze/ntire/: raw NTIRE dataset.
- data/artifacts/checkpoints/: checkpoints of AIDE and SPAI.

## Technical

- Evaluate folder for generated data carefully: bronze, silver, or gold. Do not add another tier.
- Use the appropriate agent skills for the task.
- Python: use `uv run script.py`, `uv run -m ...`. Ask me if you need to modify the python packages.

## Code Convention

- Docstrings: Apply Google-style to all new or modified code. Clearly state the overarching context, purpose, and contract (the "What"), strictly avoiding internal step-by-step implementation details.
- Code Comments: Keep them extremely concise. Explain only the "Why" behind non-obvious logic or business decisions (the "Intent").
- Comment Mapping: Directly relate inline comments back to the overarching context defined in the docstring to maintain readability.
- Logging: Add necessary debug logs using lazy formatting (`%` style). Avoid log flooding in loops.
