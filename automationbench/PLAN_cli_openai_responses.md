# Plan: make the AutomationBench CLI work out of the box with OpenAI reasoning models

Status: proposal, nothing implemented. Written so another coding agent can implement and verify it without
re-deriving the analysis. **Hard constraint: nothing on the Beaker side may change behaviour** (single-model
optimization and multi-model / model-swap optimization). See "Beaker invariants".

## 1. The problem

Running the README quick start with the shipped defaults (`--model gpt-5.6-luna`, `--reasoning-effort medium`,
`--skills-dir skills`) makes **every task error**, and the CLI reports it as a score of **0.000**.

Chain of events (all verified except where marked):

1. `ModelSpec.resolved_api()` (`runner.py`) calls the vendored `resolve_api` -> upstream
   `automationbench.scripts.eval._resolve_api`. Upstream's `auto` sends only first-party `claude-*` and `gemini-*`
   to their native APIs; **everything else, including OpenAI models, stays on Chat Completions** (deliberate:
   gateways/proxies/alpha endpoints speak Chat Completions).
2. `ModelSpec.sampling_args()` adds the reasoning effort (`reasoning_effort: "medium"`) to every request.
3. The agent always has function tools (the benchmark's tools plus `list_skills` / `read_skill`).
4. OpenAI rejects tools + `reasoning_effort` on `/v1/chat/completions` for this model with HTTP 400:
   `Function tools with reasoning_effort are not supported for gpt-5.6-luna in /v1/chat/completions. To use
   function tools, use /v1/responses ...` (message truncated in my capture; the tail may name a second
   workaround, check before writing the hint text).
5. verifiers swallows the exception into `output["error"]`; `_to_result` still builds a `RunResult` with
   `partial_credit=0.0`. `cli.py` prints the table with 0.000 and exits 0. A person or coding agent then reads
   "the agent scores zero" and starts tuning prompts.

Workaround used so far: pass `--api responses`. With it a 2-task canary scored 1.0 and 0.5 and a 54-task run had
0 errors.

Secondary: on the responses path `cost_usd` is not populated (cost tracking lives only in
`CostTrackingChatCompletionsClient`), so `evaluate` prints `-` for cost.

Why it is a trap: the failure is silent (exit 0, plausible-looking table), the fix is an undiscoverable flag, and
the obvious "fix" (change the default API) breaks Beaker.

## 2. Beaker invariants (must hold before and after the change)

Source: `.beaker/beaker_integration.py`. Beaker never imports `cli.py`; it imports `ModelSpec` and
`run_rollout_raw` from `runner.py`.

| # | Invariant | Where |
|---|-----------|-------|
| B1 | `ModelSpec()` default still resolves to `"chat_completions"` | `_client_for` raises `RuntimeError` at line ~219 otherwise. Beaker's tracing is a subclass of the Chat Completions client (`_TracedChatCompletionsClient`) |
| B2 | Model-swap branch (`runtime.model` set; multi-model optimization) unchanged: `ModelSpec(name=target.model, base_url=target.base_url, api="chat_completions", reasoning_effort=None)` via Beaker's inference gateway | `_client_for` |
| B3 | No-selection branch (single-model optimization, app defaults, `OPENAI_API_KEY` = Beaker provider proxy) unchanged | `_client_for` |
| B4 | `run_rollout_raw` signature and behaviour unchanged | `runner.py` |
| B5 | `ModelSpec.sampling_args()` / `resolved_api()` / `effective_*()` semantics unchanged for every input | `runner.py` |
| B6 | `.beaker/*` not edited | |

Note (my reading, not run): Beaker already treats `vf.ModelError` as `RetryableCaseError`, so the 400 is loud
there, not scored 0. The silent-zero problem is CLI-only. Whether Beaker's hosted proxy absorbs the
tools+effort 400 in branch B3 is **untested**; that is out of scope here and listed in section 6.

Rule for the implementer: every change below lives in `cli.py`, `evaluation/summary.py`, `README.md`, and
tests, plus a comment on `ModelSpec`. If a change needs an edit to `runner.py` logic or `.beaker/`, stop and
re-plan.

## 3. Changes

### C1. CLI-only default that works (`cli.py`, `_cmd_run`)
When the CLI builds the `ModelSpec`: if `--api` is `auto` **and** `--base-url` is not given **and** the model
name is a plain OpenAI name (no `/`, not `claude-*`, not `gemini-*`) **and** a reasoning effort is set (not
`None`/`""`/`default`) -> pass `api="responses"`. Print one line: `note: using --api responses (OpenAI rejects
tools + reasoning_effort on chat completions); pass --api chat_completions to override`.
Any explicit `--api` or any `--base-url` leaves behaviour exactly as today (gateway and OpenRouter users
unaffected). Extract the rule into a small pure function (e.g. `_default_api(model, base_url, api, effort)`) so it
is unit-testable.

### C2. Loud failures (`cli.py`, `evaluation/summary.py`)
- After a run, count results whose `error` is truthy. Print `N/M tasks errored; most common: <first 200 chars>`.
- If the most common error matches the known 400 (`reasoning_effort` and `/v1/responses` in the text), print the
  hint: `rerun with --api responses`.
- Exit code 1 when more than half the tasks errored (keep results on disk anyway). Otherwise 0.
- `summarize()` keeps `pass_rate`/`partial_credit` semantics for existing callers but adds `errored` per
  domain/overall; `format_summary` prints an `errored` column. Do **not** silently drop errored tasks from the
  existing averages (that would change published numbers); report them next to the averages.

### C3. Guard against the wrong "fix" (`runner.py` comment only, plus test)
Add a comment above `ModelSpec` (no code change): the default must resolve to Chat Completions because
`.beaker/beaker_integration.py` traces that client and raises otherwise; CLI-specific routing lives in
`cli.py::_default_api`. This is for future coding agents who see the 400 and reach for the default.

### C4. README (`automationbench/README.md`)
A short "OpenAI reasoning models" note under Models: the exact error text, why it happens, that the CLI now
picks `--api responses` automatically, and how to override. Mention that cost is not tracked on this path
until C5.

### C5. (optional, separate PR) cost on the responses path
Derive `cost_usd` from `usage` and a price table when the client reports none. Out of scope for the first PR.

## 4. Tests (no network, fake client; `uv run pytest -q`)

Add to `tests/test_harness.py` (or a new `tests/test_cli_routing.py`):

1. `_default_api`: returns `"responses"` for (`gpt-5.6-luna`, no base_url, `auto`, `medium`); returns `"auto"` /
   unchanged for: explicit `--api chat_completions`; any `--base-url`; `claude-*`; `gemini-*`; `vendor/model`
   (OpenRouter); effort `None`/`default`.
2. **Beaker guard:** `ModelSpec().resolved_api() == "chat_completions"` and `ModelSpec(name="gpt-5.6-luna",
   base_url="https://gw.example/v1", api="chat_completions", reasoning_effort=None).sampling_args() == {}`.
   Assertion message must say: "Beaker requires Chat Completions; see .beaker/beaker_integration.py `_client_for`".
3. Errored run: scripted fake client that raises -> `_cmd_run` exits 1, prints `N/N tasks errored` and the
   hint; result JSONs still written.
4. Mixed run (some errors, fewer than half) -> exit 0, `errored` count shown, averages unchanged vs today.
5. `summarize` backward compatible: existing `test_summarize_and_format` passes unmodified except for the new
   column if it asserts exact text (update only that).

## 5. Verification checklist for the reviewing agent

Run from `automationbench/`:

```bash
uv sync --group dev
uv run ruff check && uv run ruff format --check
uv run python -m mypy
uv run pytest -q                         # all green, incl. new tests

# Beaker invariants: these must produce an EMPTY diff
git diff -- .beaker/ src/automationbench_skills/runner.py   # only a comment added above ModelSpec is allowed
git diff --stat | grep -E 'beaker_integration|world_diff|upload_splits|beaker.yaml' && echo VIOLATION

# Beaker's module still imports and its guard still passes
uv run python -c "from automationbench_skills.runner import ModelSpec; assert ModelSpec().resolved_api()=='chat_completions'"
```

Live checks (need a valid `OPENAI_API_KEY`; the temporary key used earlier is dead):

```bash
# 1. default command now works: expect scores > 0 and "using --api responses" note
uv run automationbench-skills run --split train --limit 2 --skills-dir skills --prompts-dir prompts
# 2. old behavior reproducible and now LOUD: expect "2/2 tasks errored", the hint, exit code 1
uv run automationbench-skills run --split train --limit 2 --skills-dir skills --prompts-dir prompts --api chat_completions; echo "exit=$?"
```

Acceptance: all offline checks pass; diff touches only `cli.py`, `evaluation/summary.py`, `README.md`, tests, and
one comment in `runner.py`; live check 1 scores > 0 with no errors; live check 2 exits 1.

## 6. Open items (not part of this PR)

- Confirm in a real Beaker run that branch B3 (no model selected, hosted provider proxy, effort `medium`) does
  not hit the same 400. If it does, the fix there is to send no reasoning effort (as B2 already does) and needs a
  Beaker-side test; do not guess.
- Confirm the full text of the OpenAI 400 and whether it offers a second workaround.
- C5 cost tracking.
