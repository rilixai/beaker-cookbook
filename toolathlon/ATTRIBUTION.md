# Attribution

The agent scaffold, tasks, MCP configurations, and evaluators are provided by
[HKUST NLP's Toolathlon](https://github.com/hkust-nlp/Toolathlon), pinned to commit
`9be8d8fe07a497b18ee61e3f2ae694e9797f39eb`.

This recipe downloads upstream into an ignored local cache. Task data and
ground-truth files are not vendored into the cookbook. A versioned source patch
migrates the original TaskRunner, TaskAgent, provider and context runner to
OpenAI Agents 0.20.0, together with the upstream dependency manifests. Both the
standalone runner and Beaker onboarding execute this migrated original agent;
there is no replacement agent. The task assets, prompts, tool implementations
and official evaluator logic remain upstream. Execution and native tracing
boundaries are described in `.beaker/README.md`. Cite the original work
when reporting benchmark results:

Junlong Li et al. *The Tool Decathlon: Benchmarking Language Agents for Diverse,
Realistic, and Long-Horizon Task Execution*.
[arXiv:2510.25726](https://arxiv.org/abs/2510.25726).
