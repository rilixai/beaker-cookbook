# Attribution / NOTICE

Parts of this recipe are vendored from **AppWorld** by the Stony Brook NLP
group (StonyBrookNLP), licensed under the **Apache License 2.0**.

- Upstream repository: https://github.com/StonyBrookNLP/appworld
- Vendored at commit: `a072b7a86e7c1d5b1d7175659d750ebb9b79f10a`
- Upstream license: Apache-2.0 (a copy is included in this folder as
  [`LICENSE`](LICENSE), verbatim from that commit)
- Paper: *AppWorld: A Controllable World of Apps and People for Benchmarking
  Interactive Coding Agents* — Trivedi et al., ACL 2024 (Best Resource Paper),
  [arXiv:2407.18901](https://arxiv.org/abs/2407.18901)

**License scope:** the `rilixai-cookbook` repository's top-level licensing does
not apply to the vendored subtree below — those files remain **Apache-2.0**,
as marked by the provenance header at the top of each vendored source file.

## Vendored files

All from the upstream repository at commit
`a072b7a86e7c1d5b1d7175659d750ebb9b79f10a`:

| File in this recipe | Upstream path | Modified |
|---|---|---|
| `src/appworld_openai_agents_sdk/vendored/common/logger.py` | `experiments/code/common/logger.py` | imports rewritten to absolute `appworld_openai_agents_sdk.*` imports |
| `src/appworld_openai_agents_sdk/vendored/common/usage_tracker.py` | `experiments/code/common/usage_tracker.py` | unmodified |
| `src/appworld_openai_agents_sdk/prompts/react_code_agent/instructions.txt` | `experiments/prompts/react_code_agent/instructions.txt` | adapted: code is submitted via the `execute_python` tool instead of fenced ```python blocks in the message text; wording updated to match |

The prompt file carries no in-file provenance header because a header would
change the prompt content itself; its provenance is recorded here instead.

## Modifications (Apache-2.0 §4(b) statement of changes)

1. Internal `appworld_agents.code.*` imports rewritten to absolute
   `appworld_openai_agents_sdk.*` imports so the recipe is self-contained.
2. The ReAct instructions were adapted from upstream's
   `react_code_agent/instructions.txt`: upstream's agent emits fenced
   ```python blocks in plain messages, this recipe's agent submits code
   through an `execute_python` function tool, so the submission instructions
   and demo formatting were updated. The environment semantics (persistent
   variables, `apis.<app>.<api>` calls, `api_docs` discovery,
   `apis.supervisor.complete_task`) and the worked demo task are upstream's.

The agent loop itself (`src/appworld_openai_agents_sdk/code_agent.py`) is this
recipe's own code, written against the OpenAI Agents SDK; it mirrors the
semantics of upstream's ReAct baseline (`experiments/code/simplified/`):
max_steps=50, random_seed=100, one code chunk per step executed with
`world.execute`, episode ends on `world.task_completed()`.

The AppWorld *environment* (apps, tasks, servers, evaluator) is **not**
vendored; it is consumed as the pinned external `appworld` package (installed
from the same upstream commit).
