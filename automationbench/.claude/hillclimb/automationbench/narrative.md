| round | change | test | train | tokens in (cached) |
|---|---|---|---|---|
| 0 | baseline (seed skills, gpt-5.6-luna medium, --api responses) | 0.665 | 0.660 | 4.9M (99.6%) / 8.7M (99.7%) |
| 1 | policy sweep: read Gmail/Slack/all sheet tabs before writing; apply rules literally | 0.746 (+0.080, se 0.057) | 0.789 (+0.128, se 0.039) | 4.9M / 8.5M (flat) |
| 2 | per-record screening block (REVERTED) | 0.662 (-0.084 vs v1, se 0.071) | 0.773 (-0.016 vs v1, se 0.041) | 4.1M / 8.2M |
| 3 | no change proposed (remaining train loss <~0.02 from tool quirks) | - | - | - |

## Final summary
**Recommended change:** v1 policy sweep (already applied on this branch): read Gmail inbox broadly, Slack rules/channels and every sheet tab before writing; apply found rules literally (verbatim codes, exact vocabulary, one email per person). 23 lines across prompts/system.md and the gmail, slack, google_sheets skills.

**Versus baseline (score = 0.8*partial + 0.2*pass, gpt-5.6-luna medium, 1 rep):** train 0.660 -> 0.789 (paired +0.128, se 0.039, clears noise). Test 0.665 -> 0.746 (paired +0.080, se 0.057, ~1.4 se: directional, NOT significant on 18 tasks). Test pass rate 0.333 -> 0.444, partial 0.748 -> 0.821. Tokens and tool calls flat; test latency 52 -> 67s.

**Why trust / caveats:** n=54 tasks, 1 rep; test 95% CI on the delta is roughly -0.03..+0.19. Round-1 analyzer and later analyzers looked up train tasks in the installed package's task-definition file, which also contains test tasks (searched by train name only; no test files opened). Both variant and baseline share the same seeded subset of the frozen splits (6 train / 3 test per domain).

**What else was tried:** Round 2 (generic per-record screening rules): no gain (train -0.005, test -0.054 vs v1, ex 2 prompt-flagged errors), reverted. Round 3: analyzer found remaining loss is rule-application per task, tool-level fixes worth <=0.02, so no run.

**Next:** more reps (3x) and the full 150-task test split to confirm v1; worked-example domain playbooks; guard against OpenAI 'flagged prompt' rejections (2 of 108 v2 runs).
