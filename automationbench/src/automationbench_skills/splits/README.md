# Training and test tasks

Use `--split train` while developing skills and prompts. Reserve `--split test`
for evaluating a configuration after you finish editing it. Both splits cover
sales, marketing, operations, support, finance, and HR.

The recipe selects 478 of 600 public tasks. Its exclusion list comes from an audit
of the original AutomationBench task instructions, metadata, and graders. The
list also applies when running PI, but the audit does not establish which defects
remain in PI. [EXCLUSIONS.md](EXCLUSIONS.md) records the original findings and sources.

`load_split()` applies the same exclusion list to train and test before either
split is used or sampled for an upload:

| Split | Cases in file | Excluded | Cases loaded |
| --- | ---: | ---: | ---: |
| `train.txt` | 450 | 76 | 374 |
| `test.txt` | 150 | 46 | 104 |
| **Scored total** | **600** | **122** | **478** |

The original audit classified exclusions as **50 task/metadata issues** and **72 grading defects**.
See [EXCLUSIONS.md](EXCLUSIONS.md) for each case's reason and source evidence.
The split files record all 600 source cases and their train/test assignments.
The loader selects eligible cases in file order without moving them between splits.

- `train.txt` — 450 tasks (75 per scored domain); 374 loaded after exclusions.
  Available for training and development, including learning from trajectories
  and grading checks.
- `test.txt` — 150 tasks (25 per scored domain); 104 loaded after exclusions.
  Held out for final evaluation only. No validation split: model selection
  happens on train.

Tasks are keyed by `task_name` (`<domain>.<name>`), the benchmark's only
globally-unique stable id (`example_id` collides across domains).

## How the split files are generated

The generation script, `make_splits.py`, uses the pinned `automation-bench` dependency:
per scored domain, tasks are grouped into families (first underscore token of
the name, e.g. `docusign_contract_send` → `docusign`), each family's test quota
is set by largest-remainder allocation of 25 seats proportional to family size,
and members are drawn with a fixed seed (`random.Random(0)`). This represents
task families in both splits in proportion to their size.

## Regenerate the splits (maintainers)

```bash
uv run python -m automationbench_skills.data.make_splits
```

This generates the unfiltered split files; exclusions are applied only
when loading train or test. Generation is deterministic for a given dependency
pin. Changes to split membership or the exclusion list change the evaluation
set; record them when comparing results across runs.

## Keep test tasks held out

- Never read `test.txt` tasks (prompts, assertions, trajectories) while
  authoring or optimizing skills.
- Skills may encode anything learned from train tasks or general
  product knowledge, but must not reference specific test tasks.
- Report test metrics only for configurations frozen before touching test.
