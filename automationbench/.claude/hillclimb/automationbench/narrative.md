| round | change | test | train | tokens in (cached) |
|---|---|---|---|---|
| 0 | baseline (seed skills, gpt-5.6-luna medium, --api responses) | 0.665 | 0.660 | 4.9M (99.6%) / 8.7M (99.7%) |
| 1 | policy sweep: read Gmail/Slack/all sheet tabs before writing; apply rules literally | 0.746 (+0.080, se 0.057) | 0.789 (+0.128, se 0.039) | 4.9M / 8.5M (flat) |

Best: round 1. Change targets the dominant train failure (acting without reading rules that live in Gmail/Slack/other sheet tabs). Train gain is clearly above noise; test gain is positive but ~1.4 SE, so directional. Tokens and tool calls flat (28 -> 28 train), so no cost regression; test latency rose 52->67s.
