| round | change | test | train | tokens in (cached) |
|---|---|---|---|---|
| 0 | baseline (seed skills, gpt-5.6-luna medium, --api responses) | 0.665 | 0.660 | 4.9M (99.6%) / 8.7M (99.7%) |
| 1 | policy sweep: read Gmail/Slack/all sheet tabs before writing; apply rules literally | 0.746 (+0.080, se 0.057) | 0.789 (+0.128, se 0.039) | 4.9M / 8.5M (flat) |
| 2 | per-record screening block (REVERTED) | 0.662 (-0.084 vs v1, se 0.071) | 0.773 (-0.016 vs v1, se 0.041) | 4.1M / 8.2M |

Best: round 1 (kept). Round 2 (screening rules) did not beat v1: excluding 2 zero-scored cases where OpenAI rejected the prompt as flagged (1 train, 1 test, both only under v2 - possibly triggered by the new text), train delta -0.005, test -0.054; neither clears noise. Reverted. 1 flat round so far.
