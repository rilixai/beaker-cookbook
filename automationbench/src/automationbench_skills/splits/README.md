# Frozen splits

The source is [AutomationBench-PI](https://github.com/xeophon/AutomationBench-PI) at `858718b19e5d66c74451432783d34ee64a550a15`. We use all **800 published tasks**: 100 each in sales, marketing, operations, support, finance and HR, plus 200 in `simple`.

| Split | Cases in file | Excluded | Cases loaded |
| --- | ---: | ---: | ---: |
| `train.txt` | 600 | 16 | 584 |
| `test.txt` | 200 | 9 | 191 |
| **Total** | **800** | **25** | **775** |

The 25 retained exclusions are the original excluded tasks whose definitions PI did not directly change. The other 97 former exclusions are restored under the migration policy. See [EXCLUSIONS.md](EXCLUSIONS.md) for the retained reasons, evidence and limits of this policy.

## Assignment and regeneration

The original six domains retain every train/test assignment and their order: 75 train and 25 test tasks per domain. The added `simple` domain has 150 train and 50 test tasks. `make_splits.py` groups tasks by the first underscore token of their name, allocates each domain's test quota by largest remainder, and selects within each family using `random.Random(0)`. Source files include excluded IDs; `load_split()` filters them at read time.

```bash
uv run python -m automationbench_skills.data.make_splits
```

Tasks are keyed by globally unique `domain.task_name`, since upstream numeric `example_id` values collide across domains. The generator is deterministic for the dependency pin. Dataset changes affect score comparability: rebuild the hosted integration and upload a fresh PI dataset before evaluating. The uploader records the benchmark commit in row metadata and uses the distinct `automationbench-pi-skills-quickstart` dataset name.

## Balanced demo

`python -m automationbench_skills.data.demo --output-dir <local-directory>` exports
140 train and 70 test cases without uploading. It takes 20 train and 10 test
cases per domain after exclusions. Within each domain/split, largest-remainder
quotas preserve the eligible task-family proportions, and seed 0 selects members.
Every selected ID keeps its source split; no scores or test outcomes affect selection.
The verification manifest records domain/family counts and hashes of both files.

## Leakage policy

- Use train tasks for development and model selection.
- Keep test prompts, assertions and trajectories out of skill authoring and optimization.
- Report test metrics only for configurations frozen before test evaluation.
