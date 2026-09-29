| round | change | test | train | tokens in (cached) |
|---|---|---|---|---|
| 0 | baseline (seed skills, gpt-5.6-luna medium, --api responses) | 0.665 | 0.660 | 4.9M (99.6%) / 8.7M (99.7%) |

Score = 0.8*partial_credit + 0.2*strict pass. Baseline: test pass 0.333 / partial 0.748 (n=18); train pass 0.306 / partial 0.749 (n=36). Per-task sd ~0.27, so test SE ~0.066: only changes worth >~0.13 are detectable on test at 1 rep. Train and test means agree.
