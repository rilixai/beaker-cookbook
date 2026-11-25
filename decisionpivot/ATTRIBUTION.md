# Attribution

This recipe builds on **DecisionPivot: A Diagnostic Benchmark for Decision Models in LLM Systems** (formerly DecideBench).

- Paper: Zhuang et al. (2026), [*DecisionPivot: A Diagnostic Benchmark for Decision Models in LLM Systems*](https://arxiv.org/abs/2610.28522)
- Code: [`StartLuxLabs/DecisionPivot`](https://github.com/StartLuxLabs/DecisionPivot) at commit
  `fd57688f033864e64c452561ffdfcba9739d1811`, Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). `prepare`
  fetches this commit into a local cache and runs its generators and scorer there, unmodified. None of its code is copied
  into this recipe.
- Items: the 13 core families (release 1.1), written by the upstream generators with no third-party text, under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The recipe regenerates them at run time and commits none.
  The reference track is not used.
- Terms: DecisionPivot is for evaluation only. Its items must not be used as training data or put into the prompts of a
  model that is being trained.

The recipe does not change any item or answer. It draws two fresh sets (seeds 1 and 2) of the seven families whose
generator takes a seed, and makes its own train/validation/held-out splits by contrast group.

```bibtex
@misc{zhuang2026decisionpivot,
  title         = {DecisionPivot: A Diagnostic Benchmark for Decision Models in LLM Systems},
  author        = {Zhuang, Xialie and Lei, Mingcong and Su, Zunhai and Liu, Ri and Li, Jianjin and Ma, Peixian and Pan, Jia and Wong, Ngai and Chen, Guanhua and Kuo, Chuan-Wei},
  year          = {2026},
  eprint        = {2610.28522},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2610.28522}
}
```
