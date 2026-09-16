# AppWorld SGC integration

Each dataset row contains one complete source scenario: all three tasks,
user-facing instructions, and the original benchmark requirements. Train uses
AppWorld's train split; held-out evaluation uses dev. Official test splits are
not used for optimization.

The objective is Scenario Goal Completion: 1 only when every task in the
scenario passes AppWorld's own evaluator. Task Goal Completion is diagnostic.
The scorer emits each benchmark requirement with its pass/fail result.

The integration calls the candidate's ordinary `run_code_agent_on_tasks`
workflow with its default model and 50-step limit. Beaker may edit application
source and configs; evaluation policy stays under `.beaker`. Each case gets a
temporary simulator root. Cases are serialized because AppWorld uses global
state. Setup stages real benchmark data as case files and downloads the public
benchmark when it is absent from the hosted checkout.

OpenAI Responses calls retain their native client and use Beaker's hosted
provider routing. `OPENAI_API_KEY` remains declared because the client reads
it; a backend with provider routing enabled can supply the runtime placeholder.
No personal key is stored here. The OpenAI Agents tracing adapter covers the
candidate workflow, excluding benchmark evaluation.

`prepare_dataset.py` converts existing local data in an OS temporary directory,
performs structural smoke validation, and uploads it. Pass the resulting
immutable dataset revision explicitly to remote smoke and launch. Generated
JSONL files and benchmark data must not be committed.

Structural smoke validates setup and files; it does not execute the candidate,
model, or scorer. Runtime tracing and hosted execution need separate evidence.
