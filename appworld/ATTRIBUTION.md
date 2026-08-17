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
| `src/appworld_openai_agents/vendored/openai_agents/run.py` | `experiments/code/openai_agents/run.py` | imports rewritten to package-relative; LitellmModel import made lazy |
| `src/appworld_openai_agents/vendored/openai_agents/api_predictor.py` | `experiments/code/openai_agents/api_predictor.py` | imports rewritten to package-relative |
| `src/appworld_openai_agents/vendored/openai_agents/language_model.py` | `experiments/code/openai_agents/language_model.py` | LitellmModel import made lazy (and the `_model` annotation widened accordingly) |
| `src/appworld_openai_agents/vendored/openai_agents/mcp.py` | `experiments/code/openai_agents/mcp.py` | imports rewritten to package-relative; MCP URL normalized to a trailing slash; log streamer handles list-form tool outputs |
| `src/appworld_openai_agents/vendored/common/api_predictor.py` | `experiments/code/common/api_predictor.py` | unmodified |
| `src/appworld_openai_agents/vendored/common/logger.py` | `experiments/code/common/logger.py` | imports rewritten to package-relative |
| `src/appworld_openai_agents/vendored/common/usage_tracker.py` | `experiments/code/common/usage_tracker.py` | unmodified |
| `src/appworld_openai_agents/vendored/common/utils.py` | `experiments/code/common/utils.py` | imports rewritten to package-relative |
| `src/appworld_openai_agents/prompts/api_predictor.txt` | `experiments/prompts/api_predictor.txt` | unmodified |
| `src/appworld_openai_agents/prompts/function_calling_agent/instructions.txt` | `experiments/prompts/function_calling_agent/instructions.txt` | unmodified |
| `src/appworld_openai_agents/prompts/function_calling_agent/demos.json` | `experiments/prompts/function_calling_agent/demos.json` | unmodified |

The prompt/demo files carry no in-file provenance header because a header
would change the prompt content itself (and JSON has no comments); their
provenance is recorded here instead.

## Modifications (Apache-2.0 §4(b) statement of changes)

Changes to vendored `.py` files (each file's header states its own):

1. Internal `appworld_agents.code.*` imports rewritten to package-relative
   imports so the recipe is self-contained.
2. In `run.py` and `language_model.py`, the `LitellmModel` import was moved
   inside the `type == "litellm"` branch, because `litellm` is an optional
   extra this recipe does not install (it targets native OpenAI models).
3. In `mcp.py`, the MCP server URL is normalized to end with a trailing
   slash: the mcp 1.x streamable-http client otherwise fails on the server's
   307 redirect from `/mcp` to `/mcp/`.
4. In `mcp.py`, the log streamer handles tool outputs that arrive as a list
   of content blocks rather than a JSON string (observed with gpt-4o), which
   otherwise crashed the run in `json.loads`.

Agent behavior is unchanged.
The upstream jsonnet experiment config
(`experiments/configs/openai_agents_mcp_agent/openai/.../*.jsonnet`) was
translated — semantics preserved — into Python/TOML in this recipe's
`src/appworld_openai_agents/runner.py` + `configs/*.toml`.

The AppWorld *environment* (apps, tasks, servers, evaluator) is **not**
vendored; it is consumed as the pinned external `appworld` package (installed
from the same upstream commit).
