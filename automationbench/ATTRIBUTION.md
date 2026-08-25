# Attribution

This recipe builds on **AutomationBench** by Zapier, Inc.

- Repository: https://github.com/zapier/AutomationBench
- Paper: https://arxiv.org/abs/2604.18934
- License: MIT (copied verbatim to [LICENSE](LICENSE))
- Pinned commit: `4a8e1061254004d9dac807054eed33fad7d1ff14` (package version `automation-bench==1.0.6`)

## What is used vs. copied

- The benchmark itself (environment, simulated tools, tasks, assertion rubric) is
  **used as an unmodified dependency**, installed straight from the pinned Git
  commit — nothing from it is vendored or altered.
- `src/automationbench_skills/vendored/model_setup.py` **adapts a small portion**
  of upstream `automationbench/scripts/eval.py` (same commit): the model→API
  routing, sampling-args construction (reasoning-effort handling), and client
  construction. The file carries a provenance header; batch-API code paths were
  dropped because this recipe runs single rollouts.

Everything else in this recipe (runner, skills tools, splits, CLI, tests) is
original to this repository.
