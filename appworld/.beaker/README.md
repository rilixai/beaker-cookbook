# AppWorld Fresh

This integration optimizes **Scenario Goal Completion (SGC)**. Each case runs
all three variants of one scenario. Its objective is 1 only if AppWorld's
pinned evaluator passes every requirement in every variant; otherwise it is 0.
Task Goal Completion is reported separately and does not affect the objective.
Evaluation is deterministic; no judge model is used.

The quick-start dataset uses the first four scenarios in the official training
split: three for optimization and one held out. The benchmark test splits are
unused. `upload_dataset.py` validates real instructions and requirements, stages
JSONL in a temporary directory, and uploads it to AppWorld Fresh. Dataset
revisions are passed explicitly to smoke and launch rather than saved in YAML.

`uv run python .beaker/upload_dataset.py --full` uploads `appworld-sgc-full`
with all 90 official training tasks (30 scenarios) for optimization and all
57 dev tasks (19 scenarios) for evaluation. Each scenario remains one Beaker
case. The full dataset replaces the earlier training-only revision under that
name; existing runs retain their original immutable dataset revisions.

Editable scope: `code_agent.py` and the agent prompts. The integration imports
the candidate application's runner and uses its prompt directory. Scoring,
bootstrap code, fixed model configuration, and vendored code are outside scope.
The default remains `configs/model.toml`; native OpenAI Responses requests use
Beaker's hosted provider routing when available. Selected models from any provider
supported by Beaker's inference gateway use its Chat Completions endpoint,
canonical model name, and run-scoped token. The gateway controls reasoning and
sampling settings; the integration does not apply OpenAI model-family defaults
to selected models. This applies to both scenario and task goal completion.

AppWorld requires generated application modules and benchmark data beyond pip
installation. `appworld_setup.py` reuses local data when present, otherwise
prepares the pinned public assets in a temporary cache. It verifies the apps
bundle SHA-256 and dataset version. It does not unpack upstream tests.

Beaker uses the OpenAI Agents tracing adapter around candidate execution only.
The runner's optional `run_config` and `raise_provider_errors` arguments preserve
its normal defaults. Evaluation runs surface provider errors and retain model
and tool traces in Beaker. Scoring runs after the tracing scope closes.

Run commands from `appworld/` with `uv run beaker`, selecting integration
`appworld_openai_agents_sdk`. Strict smoke validates structure and labeled data;
it does not execute the agent or establish benchmark quality.

## Task Goal Completion agent

`AppWorld Task Goal Completion` uses `.beaker/tgc.yaml` and integration
`appworld_tgc`. Each case is one task. Its objective is 1 only when the official
AppWorld evaluator passes every requirement for that task, otherwise 0.
There is no partial credit. It reuses the SGC integration's runner and tracing.

`uv run python .beaker/upload_dataset.py --tgc` uploads `appworld-tgc-full`:
90 training cases and 57 evaluation cases from the official dev split.
Use `uv run beaker --config-file .beaker/tgc.yaml` for this agent's commands.
The earlier execution timeout remains undiagnosed; task-level grouping alone
is not evidence that it is fixed.
