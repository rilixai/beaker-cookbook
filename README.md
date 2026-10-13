<p align="center">
  <img src="assets/logo.png" width="120" alt="Beaker">
</p>
<h3 align="center">Beaker, the autonomous AI engineer</h3>
<p align="center">
  Beaker autonomously experiments with your AI agents: finding failures,
  testing fixes, and compounding the improvements that work.
</p>
<p align="center">
  <a href="https://app.withbeaker.ai/?utm_source=cookbook&utm_medium=github">
    <img src="assets/try-for-free.svg" width="280" alt="Try Beaker for free">
  </a>
</p>
<p align="center">
  <a href="https://www.withbeaker.ai/docs/getting-started/getting-started">Tutorial</a> ·
  <a href="https://www.withbeaker.ai/docs">Docs</a> ·
  <a href="https://withbeaker.ai">Website</a> ·
  <a href="https://www.withbeaker.ai/blog/tune-the-agent">Blog: tuning 8 models</a>
</p>

Every recipe in this cookbook is a real agent on a real benchmark, including legal
research, knowledge work, app control, and enterprise automation.

## Run it with Beaker

1. Fork this repo.
2. [Sign up for Beaker](https://app.withbeaker.ai/?utm_source=cookbook&utm_medium=github) and connect your fork.
3. Follow the [tutorial](https://www.withbeaker.ai/docs/getting-started/getting-started).

## Recipes

| Recipe | What it is |
|---|---|
| [`appworld/`](appworld/) | AppWorld: an agent that drives simulated apps by writing code. Built on the OpenAI Agents SDK, scored with TGC/SGC. |
| [`harvey_lab/`](harvey_lab/) | A junior-lawyer agent on Harvey's Legal Agent Benchmark: reads a case folder, writes the deliverables, gets graded criterion by criterion by an LLM judge. |
| [`automationbench/`](automationbench/) | Zapier AutomationBench: a tool-calling agent on `verifiers` that can read a `skills/` folder — edit the skills, rerun, watch the score move. |

Each recipe is a standalone uv project with its own README covering setup, commands, and expected scores.

## Stay in the loop

[**Try Beaker for free →**](https://app.withbeaker.ai/?utm_source=cookbook&utm_medium=github), then star the repo and follow
[@withbeaker on X](https://x.com/withbeaker) for Beaker news.
