# Attribution

This recipe builds on **LiveNewsBench**.

- Repository: https://github.com/YunfanZhang42/LiveNewsBench
- Paper: https://arxiv.org/abs/2602.13543
- License: MIT (copied verbatim to [LICENSE](LICENSE))
- Pinned commit: `8a6b96e7904893977ac8a3ae3342ae3d39f89c2e`, release `jan_2026_release_2`

## What is used vs. copied

- **Data** is downloaded from the pinned commit at `prepare` time and checked against
  SHA-256 hashes. Nothing from it is committed here. Only each row's question, answer,
  event date and category are kept; the source articles are dropped.
- `src/livenewsbench/grade_simple_qa_prompt.txt` is upstream
  `evals/grade_simple_qa_prompt.txt`, byte for byte. Upstream adopted it from
  OpenAI's SimpleQA grader ([openai/simple-evals](https://github.com/openai/simple-evals), MIT).
- The seed agent's prompt in `src/livenewsbench/agent/prompts.py` **adapts** the
  prompt of upstream `evals/deep_research_single_agent.py` (same commit) from text
  action blocks to function tools. The rules it states (5 searches, 5 page visits,
  open only URLs from earlier searches, archive sites excluded) are upstream's.

Everything else in this recipe (agent loop, You.com client, rule enforcement, CLI,
Beaker integration, tests) is original to this repository.
