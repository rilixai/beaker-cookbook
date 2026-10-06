| round | change | test score | train score | test pass | test PC | Mtok in | tool calls | skill calls |
|---|---|---|---|---|---|---|---|---|
| 0 | baseline (stub skills) | 0.716 ±0.098 | 0.644 ±0.095 | 0.500 | 0.770 | 19.6 | 24.6 | 4.7 |

Baseline: seed skills are one-line stubs; the agent reads them (~4.7 skill calls/rollout) but gets no content. 3/108 rows hit OpenAI `invalid_prompt` policy flags and are excluded (pre-registered). Rep-to-rep variance is high (18/54 cases swing >0.3 between reps), so only gains of roughly ≥0.08 on test are distinguishable from noise.
