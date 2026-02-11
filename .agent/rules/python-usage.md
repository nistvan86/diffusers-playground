---
trigger: always_on
glob:
description: How to use Python in this project
---

1. My build system is uv.
2. Always run commands using `uv run` or via the `.venv` environment.
3. When introducing new Python dependencies, use `uv add` and then `uv sync`.