# rilixai-cookbook

Reproducible GEPA benchmark recipes for the
[rilixai](https://github.com/rilixai/rilixai) prompt optimizer.

Each top-level folder is a **self-contained worked benchmark** — its
own README, code, tests, and reproduce scripts. They all consume
rilixai as an external dependency through the public
`rilixai.prompt_optimization` API, so what runs here is what a real
external user of the optimizer would write.

## Layout

```
hotpotqa/         # GEPA paper's HotpotQA benchmark (workflow + PydanticAI agent)
apex_agents/      # (planned follow-up)
cl_bench/         # (planned follow-up)
```

## Quick start

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

The cookbook is a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/);
each benchmark is a workspace member with its own `pyproject.toml`.
Install all members at once:

```bash
uv sync --all-packages --group dev
```

Run that benchmark's tests:

```bash
uv run python -m pytest hotpotqa/tests -q
```

Each benchmark folder has its own `README.md` with reproduce commands.

## rilixai dependency

Pinned via git+url at a specific main-branch SHA until a fresh
rilixai is published to PyPI. See `pyproject.toml` for the current pin.

## Development

```bash
uv sync --all-packages --group dev
uv run python -m pytest -q
uv run ruff check
uv run ruff format --check
uv run python -m mypy
```
