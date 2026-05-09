# AGENTS.md

## Context

- NTIRE dataset in `/mnt/c/Users/itsky/Downloads/Math/data/`
- All checkpoints in `/mnt/c/Users/itsky/Downloads/Math/checkpoints/`

## Code Convention

- Docstrings: Apply Google-style to all new or modified code. Clearly state the overarching context, purpose, and contract (the "What"), strictly avoiding internal step-by-step implementation details.
- Code Comments: Keep them extremely concise. Explain only the "Why" behind non-obvious logic or business decisions (the "Intent").
- Comment Mapping: Directly relate inline comments back to the overarching context defined in the docstring to maintain readability.
- Logging: Add necessary debug logs using lazy formatting (`%` style). Avoid log flooding in loops.
