# Attribution

This recipe builds on **τ²-bench** by Sierra Research.

- Repository: https://github.com/sierra-research/tau2-bench
- Paper: https://arxiv.org/abs/2506.07982
- License: MIT (copied verbatim to [LICENSE](LICENSE))
- Pinned release: `v2.0.0`, commit `989d3285ae669e49503324bdb9981bf14fccedeb` (package `tau2==2.0.0`)

## What is used vs. copied

- The benchmark package source from the pinned commit is copied into
  `vendor/tau2`. Its Python source is unchanged. Its package metadata pins
  LiteLLM to `1.104.0` so Beaker can install it alongside its runtime.
- Upstream data files (airline and retail databases, policies, tasks, splits, and the
  English language pack) are **downloaded at run time** from the same commit and
  checked against SHA-256 digests. They are not stored in this repository.
- `prompts/system.md` starts from upstream's `AGENT_INSTRUCTION` text in
  `src/tau2/agent/llm_agent.py` (same commit).

Everything else in this recipe (agent, chat layer, simulator tracing, splits, scoring
map, CLI, tests) is original to this repository.
