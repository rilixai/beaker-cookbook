# Contributing

Each recipe is a standalone uv project and carries its own lint, type-check, and test config. CI runs the same steps in every recipe folder:

```bash
cd <recipe>
uv sync --group dev --locked     # fails if uv.lock is out of date; run `uv lock` after changing pyproject.toml
uv run ruff check && uv run ruff format --check
uv run python -m mypy
uv run python -m pytest -q
```

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.12+.
