# OfficeQA Pro splits

`train.txt` (93 uids) and `test.txt` (40 uids) partition all 133 rows of
`officeqa_pro.csv`. **This split is ours.** It is not a published partition and
no number measured on it is comparable to the OfficeQA Pro report, which
evaluates on all 133 questions.

- Dataset: `databricks/officeqa`, file `officeqa_pro.csv`
- Pinned revision: `763a8366abf2a3605c381d53586d844dc60fa756`
  (`officeqa.config.OFFICEQA_DATASET_REVISION`)
- Generator: `officeqa.splits.generate_split`, seed `0`

## Procedure

The report's capability tags (multi-bulletin, web-search-needed, visual,
data-analysis) are not CSV columns, so we stratify on two features derived from
`source_files`:

1. `n_source_files` — number of referenced bulletins, bucketed `1` / `2` / `3+`.
   In the shipped CSV the entries are **newline**-separated (the spec assumed
   `;`); the loader accepts both.
2. `decade` of the earliest referenced bulletin, parsed from the
   `treasury_bulletin_YYYY_MM` basename. The corpus switches from scanned
   physical documents to digital-native PDFs in 1996, and the report measures a
   5–15 pp accuracy gap across that boundary, so decade balance matters.

Then:

1. Form the cross-product strata (29 non-empty cells) and sort them.
2. Within each stratum, sort uids, then shuffle with a single `random.Random(0)`
   consumed in stratum order.
3. Allocate the 40 test slots across strata by the **largest-remainder method**
   (floor of the proportional quota, leftovers to the largest fractional
   remainders; ties by stratum size, then order). Totals are exactly 40.
4. Train gets the remainder of each stratum.
5. Write each file **round-robin across strata**, so any prefix stays
   distribution-representative and `--limit N` takes a meaningful sample.

## Stratum counts

| files | decade | train | test | total |
|---|---|---|---|---|
| 1 | 1930s | 1 | 1 | 2 |
| 1 | 1940s | 6 | 3 | 9 |
| 1 | 1950s | 5 | 2 | 7 |
| 1 | 1960s | 6 | 2 | 8 |
| 1 | 1970s | 4 | 1 | 5 |
| 1 | 1980s | 10 | 5 | 15 |
| 1 | 1990s | 2 | 1 | 3 |
| 1 | 2000s | 6 | 2 | 8 |
| 1 | 2010s | 5 | 2 | 7 |
| 1 | 2020s | 2 | 1 | 3 |
| 2 | 1930s | 2 | 1 | 3 |
| 2 | 1940s | 4 | 1 | 5 |
| 2 | 1950s | 3 | 1 | 4 |
| 2 | 1960s | 4 | 2 | 6 |
| 2 | 1970s | 1 | 1 | 2 |
| 2 | 1980s | 3 | 1 | 4 |
| 2 | 1990s | 2 | 1 | 3 |
| 2 | 2000s | 3 | 1 | 4 |
| 2 | 2010s | 6 | 2 | 8 |
| 3+ | 1930s | 1 | 1 | 2 |
| 3+ | 1940s | 2 | 1 | 3 |
| 3+ | 1950s | 2 | 1 | 3 |
| 3+ | 1960s | 2 | 1 | 3 |
| 3+ | 1970s | 4 | 1 | 5 |
| 3+ | 1980s | 1 | 1 | 2 |
| 3+ | 1990s | 2 | 1 | 3 |
| 3+ | 2000s | 1 | 1 | 2 |
| 3+ | 2010s | 2 | 1 | 3 |
| 3+ | 2020s | 1 | 0 | 1 |
| **all** | | **93** | **40** | **133** |

## Regeneration

One-off, deterministic; requires `HF_TOKEN` with access to the gated dataset
(or pass `--csv path/to/officeqa_pro.csv` to use a local copy):

```bash
cd officeqa
uv run python -m officeqa.splits
```

The output must be byte-identical to the committed files. `tests/test_splits.py`
checks sizes, disjointness, coverage, and determinism against a fixture CSV.

## `dev`

`--split dev` is the 113 `easy` rows of `officeqa_full.csv`, exposed for cheap
smoke runs only. Upstream defined "easy" as "two frontier agents already solved
it", so it saturates quickly and is not a useful diagnostic for a strong agent.
It is never scored as part of train/test.
