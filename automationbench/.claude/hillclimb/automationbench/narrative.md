| round | change | test score | train score | test pass | test PC | Mtok in | tool calls | skill calls |
|---|---|---|---|---|---|---|---|---|
| 0 | baseline (stub skills) | 0.716 ±0.098 | 0.644 ±0.095 | 0.500 | 0.770 | 19.6 | 24.6 | 4.7 |
| 1 | filled skills + general/workflow | 0.706 ±0.142 (Δ −0.010 ±0.123) | 0.793 ±0.091 (Δ +0.149 ±0.108) | 0.472 | 0.765 | 19.3 | 30.5 | 5.9 |

Baseline: seed skills are one-line stubs; the agent reads them (~4.7 skill calls/rollout) but gets no content. 3/108 rows hit OpenAI `invalid_prompt` policy flags and are excluded (pre-registered). Rep-to-rep variance is high (18/54 cases swing >0.3 between reps), so only gains of roughly ≥0.08 on test are distinguishable from noise.


Round 1 filled the skills from train failures (policy discovery, authority rules, record selection, write completeness, tool mechanics). Train jumped +0.149 (significant), test flat within a ±0.12 noise band; tool calls +25%. Best stays at baseline. Round 2 builds on v1 and calibrates the rules to fire only when warranted. Note: the round-2 hypothesis (over-application) was informed by the *direction* of test-score drops (no test transcripts read) — a mild use of held-out signal, disclosed here.
