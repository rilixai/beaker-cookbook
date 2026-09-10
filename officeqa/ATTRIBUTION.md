# Attribution

This recipe builds on **OfficeQA Pro** by Databricks, Inc.

- Repository: https://github.com/databricks/officeqa
- Technical report: https://arxiv.org/abs/2603.08655 (OfficeQA Pro: An Enterprise Benchmark for End-to-End Grounded Reasoning, March 2026)
- Dataset: https://huggingface.co/datasets/databricks/officeqa (gated)
- Code license: Apache License 2.0 (copied verbatim to [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0))
- Data license: CC-BY-SA-4.0 — **no data is redistributed here**; the recipe downloads it into a local cache at run time, so only Apache 2.0 applies to this directory.
- Pinned repository commit: `7b9a3c154ef9fb40215bb67934afc43e6799de16` (`reward.py` last changed in `309dc01ad21113d240700eecb74d2f3e59f163b6`)
- Pinned dataset revision: `763a8366abf2a3605c381d53586d844dc60fa756`

## What is copied vs. cited vs. original

- `src/officeqa/evaluation/reward.py` is **copied verbatim** from upstream
  `reward.py` at the pinned commit, with a five-line provenance header
  prepended. It is excluded from this repo's lint/format/type checks so diffs
  against upstream stay reviewable. Do not modify it.
- `src/officeqa/agent/prompts.py` reproduces the agent system prompt from
  **Appendix E.5 of the technical report**. This is quoted paper text, cited as
  such; it is not code and no software license is claimed for it.
- Loop constants (200 steps, 30-message window, 25k-character tool-output
  truncation, 30 retries), the REPL sandbox restrictions, and the
  `fs_search` / `fs_read` capability mapping follow the report's §4.1 and
  Appendix D.1 (Table 3). Each carries a source comment at its definition.

Everything else in this recipe (agent loop, tools, data loading, corpus cache,
splits, CLI, tests) is original to this repository.

## Upstream NOTICE (reproduced as required by Apache 2.0 §4(d))

> Copyright (2025) Databricks, Inc.
>
> This repository includes materials developed by Databricks, Inc.
> (https://www.databricks.com/) and contains files provided under different
> licenses.
>
> The files reward.py, corpus_scripts/transform_scripts/transform_parsed_files.py,
> corpus_scripts/transform_scripts/transform_files_page_level.py,
> corpus_scripts/ocr_removal.ipynb, and corpus_scripts/render-officeqa-json.ipynb
> are licensed under the Apache 2.0 license. See the LICENSE-APACHE file in this
> repository for details.
>
> The files officeqa_pro.csv and officeqa_full.csv are licensed under the
> Creative Commons Attribution-ShareAlike 4.0 license and are available on
> Hugging Face at https://huggingface.co/datasets/databricks/officeqa. See the
> LICENSE-CC-BY-SA file in this repository for details.
