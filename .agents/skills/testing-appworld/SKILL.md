---
name: testing-appworld
description: How to set up and smoke-test the appworld recipe (AppWorld benchmark, OpenAI Agents SDK) in rilixai-cookbook, including known environment pitfalls.
---

# Testing the appworld recipe

All commands from `appworld/` in rilixai-cookbook.

## Setup
1. `export UV_GIT_LFS=1 && uv sync --group dev`
2. `uv run appworld install`
3. `uv run appworld download data` (~few hundred MB, cached in `./data`)

## Known pitfalls (fixed in-recipe on PR #51; verify before assuming broken)
- **Git LFS**: the `appworld` git-pinned dependency ships `.source/*.bundle` via Git LFS. Always `export UV_GIT_LFS=1` before `uv sync`/`uv run` (uv skips LFS smudge by default), otherwise `appworld install` fails with "is a Git LFS pointer" or `ModuleNotFoundError: No module named 'appworld.apps.admin'`. If a pointer-only clone got cached (symptom: `smudge filter lfs failed` / "remote missing object" even though the object exists on GitHub): `uv cache clean appworld` is NOT enough — uv keeps git checkouts separately; run `rm -rf ~/.cache/uv/git-v0` and re-sync.
- **jinja2** may be missing (appworld needs it for prompt templates): declared in the recipe's pyproject.
- **Sync tools hang**: AppWorld's interpreter (`world.execute`) hangs when called from a worker thread; the recipe's `execute_python` tool is `async def` so the SDK runs it on the event-loop thread. Don't convert it to a sync tool.

## Smoke run
- `uv run --no-sync appworld-openai-agents run --config configs/model.toml --split dev --max-tasks 2` (~2.5 min)
- `uv run --no-sync appworld-openai-agents evaluate --config configs/model.toml --split dev --max-tasks 2` → prints TGC/SGC table.
- For a standard model: `--model gpt-4.1 --temperature 0` (family is inferred from the name).
- Healthy runs show alternating `agent (step #N)` / `environment (step #N)` panels; grep for `Error code: 400`.

## Devin Secrets Needed
- `OPENAI_API_KEY` (session/org secret) — bind via exec env for run/evaluate.
