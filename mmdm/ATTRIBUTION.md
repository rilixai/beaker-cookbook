# Attribution

This recipe builds on **MMDM: Measuring Massive Multitask Decision Making**.

- Paper: Shen et al. (2026), [*MMDM: Measuring Massive Multitask Decision Making*](https://arxiv.org/abs/2611.14309)
- Data: [`shenjunhao/mmdm`](https://huggingface.co/datasets/shenjunhao/mmdm), file `mmdm.jsonl` at commit
  `1551e88c3b5290d6c5d3fb4c01b0ec00c57cfe7e` (SHA-256 `70fb1ba6a61406bbb25b3125e63c21f3e1fc2ac0dce5b57525ff1cbc0c61d2ed`)
- Evaluation toolkit: [`mmdm-benchmark/mmdm`](https://github.com/mmdm-benchmark/mmdm) (`mmdm_eval`) at commit
  `eedc865ef7303740a92e1abe73e13698e9f88091`, installed as an unmodified Git dependency
- License: original MMDM content and the toolkit are MIT (copied to [LICENSE](LICENSE))

Many MMDM questions are adapted from public datasets (the row's `source` field
names it) and keep their upstream terms. See "Sources and licensing" on the
[dataset card](https://huggingface.co/datasets/shenjunhao/mmdm). The pinned
dataset commit has no separate `SOURCE_NOTICES.md`.

The recipe does not modify MMDM's questions or labels. It makes its own
train/validation/held-out splits, leaves out the 94 questions with more than
255 options, and uses `mmdm_eval` for request building, state truncation and
scoring.
