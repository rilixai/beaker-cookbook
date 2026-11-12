# Attribution

The agent scaffold, tasks, MCP configurations, and evaluators are provided by
[HKUST NLP's Toolathlon](https://github.com/hkust-nlp/Toolathlon), pinned to commit
`9be8d8fe07a497b18ee61e3f2ae694e9797f39eb`.

This recipe downloads upstream into an ignored local cache. No Toolathlon source,
task data, or ground-truth files are vendored into the cookbook. The recipe's
standalone runner uses the upstream agent. The Beaker onboarding entrypoint uses
a native OpenAI Agents 0.20 implementation with Toolathlon's prompts, local tools,
MCP servers and history format. Its execution boundaries are described in
`.beaker/README.md`. Cite the original work
when reporting benchmark results:

Junlong Li et al. *The Tool Decathlon: Benchmarking Language Agents for Diverse,
Realistic, and Long-Horizon Task Execution*.
[arXiv:2510.25726](https://arxiv.org/abs/2510.25726).
