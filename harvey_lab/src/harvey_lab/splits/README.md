# Frozen splits

`train.txt`, `val.txt`, `test.txt` are the fixed train / validation / test
partitions of Harvey LAB, one task ID per line (`<practice-area>/…/<slug>`,
the task's directory path under `harvey-labs/tasks/`). They are the source of
truth for reproducible runs — `cli.py --split {train,val,test}` loads exactly
these IDs.

- **Pinned commit:** `harveyai/harvey-labs@1da4750171bc5a534960b3d82d15ba7fd2cf653f`
  (also in `config.py` as `HARVEY_LABS_COMMIT`). Task IDs are paths into that
  exact tree, so clone the benchmark at this commit.
- **Sizes:** train 1560 / val 100 / test 100 (all 1760 tasks, three-way disjoint).
- **Sampling:** val and test are each capped at 100 (`config.VAL_CAP` /
  `TEST_CAP`) and drawn to **follow the natural practice-area distribution** of
  the pinned commit (e.g. contracts ~28% of val); train is everything else.
  Each list is ordered round-robin across practice areas, so any prefix stays
  distribution-representative — `--limit N` takes the first N.

## Regenerating

Not part of the package (run once). To reproduce: clone `harvey-labs` at the
pinned commit, then for each practice area take its tasks, shuffle with a fixed
seed (0), and allocate per-area counts for val/test by the largest-remainder
method so the two 100-caps are hit exactly while tracking each area's share;
train gets the remainder. Write each split round-robin across areas.
