# Attribution

This recipe vendors the **OpenAI Agents SDK baseline agent** from
[AppWorld](https://github.com/StonyBrookNLP/appworld) by **Stony Brook NLP**.

- Upstream repo: <https://github.com/StonyBrookNLP/appworld>
- License: **Apache-2.0** (verified at the vendored commit; copy in
  [`LICENSE`](LICENSE))
- Vendored commit: **`a072b7a86e7c1d5b1d7175659d750ebb9b79f10a`**
- Paper: Trivedi et al., *AppWorld: A Controllable World of Apps and People
  for Benchmarking Interactive Coding Agents*, ACL 2024 (Best Resource Paper),
  [arXiv:2407.18901](https://arxiv.org/abs/2407.18901)

The AppWorld **environment** (apps, tasks, servers, evaluator) is NOT
vendored — it is consumed as a pinned pip dependency on the same commit.

## Vendored files

All upstream paths are relative to the repo root at the commit above; all
local paths are relative to `src/appworld_openai_agents/`.

| Local path | Upstream path | Modifications |
|---|---|---|
| `vendored/common/api_predictor.py` | `experiments/code/common/api_predictor.py` | none (verbatim) |
| `vendored/common/usage_tracker.py` | `experiments/code/common/usage_tracker.py` | none (verbatim) |
| `vendored/common/logger.py` | `experiments/code/common/logger.py` | imports rewritten¹ |
| `vendored/common/utils.py` | `experiments/code/common/utils.py` | imports rewritten¹ |
| `vendored/openai_agents/run.py` | `experiments/code/openai_agents/run.py` | imports rewritten¹; removed module-level `set_default_openai_api("chat_completions")` (the API type is set per-run from the config's `api_type`, which this recipe defaults to `responses`) |
| `vendored/openai_agents/api_predictor.py` | `experiments/code/openai_agents/api_predictor.py` | imports rewritten¹ |
| `vendored/openai_agents/language_model.py` | `experiments/code/openai_agents/language_model.py` | imports rewritten¹; added an `api_type` option so `type: openai` routes to the SDK's native `OpenAIResponsesModel` by default; output-text extraction made robust to leading non-message items (reasoning items) |
| `vendored/openai_agents/mcp.py` | `experiments/code/openai_agents/mcp.py` | imports rewritten¹ |
| `prompts/api_predictor.txt` | `experiments/prompts/api_predictor.txt` | none (byte-identical; prompt files carry no headers so the prompts stay unmodified) |
| `prompts/function_calling_agent/instructions.txt` | `experiments/prompts/function_calling_agent/instructions.txt` | none (byte-identical) |
| `prompts/function_calling_agent/demos.json` | `experiments/prompts/function_calling_agent/demos.json` | none (byte-identical) |

¹ `appworld_agents.code.*` imports rewritten to local
`appworld_openai_agents.vendored.*` imports so the folder is self-contained.

The reference experiment config
`experiments/configs/openai_agents_mcp_agent/openai/gpt-4o-2024-05-13/*.jsonnet`
was not vendored; its semantics were translated into
`src/appworld_openai_agents/config.py` (see that module's docstring).
