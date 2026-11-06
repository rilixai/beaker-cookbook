# Attribution

This recipe builds on **AutomationBench** by Zapier, Inc. and the **AutomationBench-PI** fork by xeophon.

- Dataset, tools, services and graders: https://github.com/xeophon/AutomationBench-PI
- Pinned commit: `858718b19e5d66c74451432783d34ee64a550a15` (`automation-bench==1.1.0`)
- Original repository: https://github.com/zapier/AutomationBench
- Paper: https://arxiv.org/abs/2604.18934
- License: MIT (copied to [LICENSE](LICENSE))

The PI benchmark is an unmodified Git dependency. The recipe drives its MCP toolset and scores with its rubric.

`src/automationbench_skills/vendored/clients.py` preserves Zapier's provider clients from `automationbench/clients.py` at `4a8e1061254004d9dac807054eed33fad7d1ff14`. Imports now refer to the recipe's model protocol because PI removed the legacy client API. This retains native Responses, Anthropic streaming and Gemini interactions behavior.

`src/automationbench_skills/vendored/model_setup.py` extracts model/API routing, sampling arguments and client construction from the original `automationbench/scripts/eval.py` at the same commit. Routing helpers are now included locally; batch paths remain omitted.

The recipe-owned rollout loop, model protocol, skills tools, splits, CLI and tests are maintained here. They do not replace PI's simulated service implementations or grading rules.
