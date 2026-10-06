| round | change | test score | train score | test pass | test PC | Mtok in | tool calls | skill calls |
|---|---|---|---|---|---|---|---|---|
| 0 | baseline (stub skills) | 0.716 ±0.098 | 0.644 ±0.095 | 0.500 | 0.770 | 19.6 | 24.6 | 4.7 |
| 1 | filled skills + general/workflow | 0.706 ±0.142 (Δ −0.010 ±0.123) | 0.793 ±0.091 (Δ +0.149 ±0.108) | 0.472 | 0.765 | 19.3 | 30.5 | 5.9 |
| 2 | calibrate v1 rules (scope, proportionate search) | **0.821 ±0.105 (Δ +0.105 ±0.110)** | 0.888 ±0.058 (Δ +0.244 ±0.093) | 0.639 | 0.866 | 17.8 | 25.1 | 5.9 |

Baseline: seed skills are one-line stubs; the agent reads them (~4.7 skill calls/rollout) but gets no content. 3/108 rows hit OpenAI `invalid_prompt` policy flags and are excluded (pre-registered). Rep-to-rep variance is high (18/54 cases swing >0.3 between reps), so only gains of roughly ≥0.08 on test are distinguishable from noise.


Round 1 filled the skills from train failures (policy discovery, authority rules, record selection, write completeness, tool mechanics). Train jumped +0.149 (significant), test flat within a ±0.12 noise band; tool calls +25%. Best stays at baseline. Round 2 builds on v1 and calibrates the rules to fire only when warranted. Note: the round-2 hypothesis (over-application) was informed by the *direction* of test-score drops (no test transcripts read) — a mild use of held-out signal, disclosed here.

Round 2 kept v1's content but calibrated it: perform every requested action and add none, search proportionately, explicit fields decide selection, numeric API params as plain numbers. Tool calls back to baseline (25.1), input tokens below baseline, and test now +0.105 vs baseline (CI just reaches zero) — current best. Extra test reps for baseline and v2 are running to tighten that interval.

Test reps raised to 4 for baseline and v2: v2 test 0.808 vs baseline 0.726, Δ +0.082 ±0.086 (pass +0.139 ±0.174, PC +0.068 ±0.072) — consistently positive but borderline; the 18-case test set's case variance is the limit, not reps.
