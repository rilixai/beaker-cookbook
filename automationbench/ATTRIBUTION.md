# Attribution

This recipe builds on **AutomationBench** by Zapier, Inc. and the **AutomationBench-PI** fork by xeophon.

- Dataset, tools, services and graders: https://github.com/xeophon/AutomationBench-PI
- Pinned commit: `858718b19e5d66c74451432783d34ee64a550a15` (`automation-bench==1.1.0`)
- Original repository: https://github.com/zapier/AutomationBench
- Paper: https://arxiv.org/abs/2604.18934
- License: MIT (copied to [LICENSE](LICENSE))

The PI benchmark is an unmodified Git dependency. The recipe drives its MCP toolset and scores with its rubric.

The recipe uses the native PI task, MCP toolset, and Verifiers V1 `NullHarness`.
Its local extensions add Beaker tracing, skill tools, frozen inputs, and scoring
state transport. They do not replace PI's service implementations or grading rules.
