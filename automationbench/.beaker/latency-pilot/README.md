# H20 latency pilot

A frozen starting point for speed-only optimization with correctness measured
separately. This pilot requires the Beaker runtime PR linked from the companion
pull request; older runtimes reject `optimization_goal: latency`.

## Provenance

The starting candidate is H20 from run
[7c782c4f-3b76-4817-a826-94724420f290](https://app.withbeaker.ai/ai-engineer/runs/7c782c4f-3b76-4817-a826-94724420f290),
exported candidate `933fb80f-3db9-42c7-b854-1e36ef285fc8`. All 36 editable files
under `project/{src,skills,prompts}` are byte-for-byte copies verified against
the exported SHA-256 manifest. The source revision is
`9480b207e387d46384c417f18bc04c03b5865e9d`; its `pyproject.toml` and `uv.lock`
are preserved. The existing recipe outside this directory is unaffected.

The trusted integration keeps the source correctness formula:
`0.8 * partial_credit + 0.2 * task_completed_correctly`. Only `.beaker` timing
and the guard against selected-model overrides were added. Timing surrounds
`run_rollout_raw`, including model and tool work and excluding client cleanup,
rendering, scoring, and environment preparation. The optimizer cannot edit
`.beaker`, the dependency files, or the dataset.

The original production-system settings are pinned: `gpt-5.6-luna`, medium
reasoning, 50 steps, 1,200-second task timeout, concurrency 32 and two retries.
`provenance.json` records the exported file hashes, immutable source dataset
revision, dependency hashes, and selected task IDs.

## Frozen pilot data

The original immutable dataset has 120 TRAIN and 104 TEST cases. This pilot
takes the first six TRAIN and three TEST cases per domain in source-file order,
without consulting outcomes: 36 TRAIN / 18 TEST. Both splits retain their
original cases and assertions. The smaller TEST set supports preliminary
claims only. It was part of a previously evaluated benchmark, not a new unseen
benchmark.

## Validate and prepare

From this directory, run the offline checks:

```bash
python3 verify_pilot.py
python3 -m compileall -q project/.beaker
```

The opt-in config is `project/.beaker/beaker.yaml`. It pins concurrency and
retry settings and sets `harness_optimization.optimization_goal: latency`.
Use that config explicitly; do not change the normal recipe's default config.
Use a fresh dataset name when uploading `dataset/`; never replace the source
run's dataset revision. Pin the returned immutable revision for smoke and
launch. The source integration name is `automationbench-skills`; its agent
must be the Rilix agent associated with `rilixai/beaker-cookbook`.

Before a hosted launch, publish the runtime change and rebuild the integration
and optimization playbook from the companion commit. Keep the generated
playbook's objective instructions consistent with latency mode. Confirm the
resolved configuration retains this pilot's model, concurrency, source and
immutable dataset revision. Launching is a separate explicit action; these
files perform no upload or paid evaluation.

## Evaluation policy

Each speed proposal is evaluated on all 36 TRAIN cases three times. Fresh
parent and fixed-starting-candidate arms are interleaved in randomized complete
batches at identical concurrency; parent and starting candidate share an arm
when they are the same version. Reference evaluations count toward spend.
At the first round this costs 216 task executions; later rounds can cost 324.

The controller bootstraps paired case means 10,000 times, keeping all three
observations together, with the random seed recorded. A candidate needs a
positive lower one-sided 95% bound for mean time saved and no negative upper
one-sided 95% correctness bound against either reference. This detects clear
loss, not statistical equivalence, and does not provide an experiment-wide
multiple-comparison guarantee.

After TRAIN selects and freezes a winner, compare only it and H20 on all 18
TEST tasks, three times each (108 executions for distinct versions). TEST
results never influence promotion. Missing measurements cannot establish a
successful comparison. Do not tune on TEST or keep trying winners until TEST
passes.

Read separate correctness, strict task completion, mean latency, median, p95,
successful-task latency, error counts and costs in the latency verdict archive.
Final TEST evidence lives in the controller's `final-test-evidence` directory,
outside the proposer-readable archive. Record actual fresh measurements in
both PRs when a pilot has been explicitly launched; historical H20 numbers
(74.6% test score, 45.0 seconds) are context, not new validation results.
