# Attribution

This recipe builds on **ALFWorld**, which pairs TextWorld text games with the
household tasks of ALFRED.

- ALFWorld: Shridhar et al. (ICLR 2021), [*ALFWorld: Aligning Text and Embodied Environments for Interactive Learning*](https://arxiv.org/abs/2010.03768).
  Code: [`alfworld/alfworld`](https://github.com/alfworld/alfworld), installed unmodified from PyPI as
  `alfworld==0.4.2` (text-only, no extras). MIT License, copied to [LICENSE](LICENSE).
- TextWorld: Côté et al. (2018), [*TextWorld: A Learning Environment for Text-based Games*](https://arxiv.org/abs/1806.11532).
  Code: [`microsoft/TextWorld`](https://github.com/microsoft/TextWorld), installed as `textworld[pddl]==1.7.0`.
  MIT License.
- ALFRED: Shridhar et al. (CVPR 2020), [*ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks*](https://arxiv.org/abs/1912.01734).
  The game files are generated from ALFRED trajectories. ALFRED's repository,
  [`askforalfred/alfred`](https://github.com/askforalfred/alfred), is released under the MIT License
  (`LICENSE`, "Copyright (c) 2019 ALFRED"; the README's License section says MIT License).
- Fast Downward: TextWorld's PDDL engine depends on
  [`fast-downward-textworld`](https://github.com/MarcCote/downward) (GPL-3.0), installed as a dependency and not
  redistributed here.

## Game files

`prepare` and the Beaker sandbox download these ALFWorld release files and
check their SHA-256. None are committed.

| File | SHA-256 |
|---|---|
| [`0.2.2/json_2.1.1_json.zip`](https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip) | `25171f16e20ad7b048c47275c45b0babf3aa1cbab29cec97387922350a9844bc` |
| [`0.2.2/json_2.1.1_pddl.zip`](https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_pddl.zip) | `913942ebed06659ea0da2f8122512d98bc6add30d84961ca803132d8fbcad585` |
| [`0.4.2/json_2.1.3_tw-pddl.zip`](https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip) | `5df77ea759f2211a4106082839ddbbb790f1ba4e7d097ed732cf453f72aa36cf` |

The game archive is the one upstream's `alfworld-download` has fetched since
July 2025. The `alfworld-download` script inside the 0.4.2 wheel still points to
`0.4.0/json_2.1.2_tw-pddl.zip`. The two archives have the same games: every
`pddl_domain`, `grammar`, `pddl_problem` and `solvable` value is identical. They
differ only in the expert `walkthrough` field (added for `valid_train`, updated
for `valid_unseen`), which this recipe never shows the agent.

The recipe does not modify any game. It uses ALFWorld's own splits and filtering
and makes no changes to upstream code.
