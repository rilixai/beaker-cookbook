# TerminalBench 2.0 with Terminus 2

A small Beaker experiment with **GPT-6 Luna**, **8 train tasks**, and **4 held-out
test tasks**. The seed subclasses [Harbor's Terminus 2](https://github.com/harbor-framework/harbor)
reference agent. Beaker can edit `src/terminalbench_agent/`, including the agent
class and its prompt templates. Task selection, task files, container resources,
and grading stay fixed under `.beaker/` or in the pinned Harbor dependency.

## Prepare an experiment

Use Python 3.12 or 3.13 and uv. Select the API for the app where the experiment
should appear. A local frontend can use a preview API: use that API's full URL,
including its preview path, rather than the frontend's localhost URL. This also
keeps preview setup separate from production.

```bash
cd terminalbench
uv sync --group dev --locked

export BEAKER_API_BASE_URL='<API URL for your Beaker app>'
uv run beaker --base-url "$BEAKER_API_BASE_URL" login
uv run beaker --base-url "$BEAKER_API_BASE_URL" agent setup 'TerminalBench 2.0' \
  --integration-id terminalbench --repo '<owner>/beaker-cookbook'

# Use the agent key returned by setup if it differs from this one.
uv run python .beaker/upload_dataset.py --base-url "$BEAKER_API_BASE_URL" \
  --agent terminalbench-2-0

# Replace <revision> with the immutable revision returned by upload.
uv run beaker --base-url "$BEAKER_API_BASE_URL" run smoke --strict \
  --integration-id terminalbench --dataset 'terminalbench-small@<revision>'
```

The upload script creates JSONL files in a temporary directory and removes them
after upload. The rows contain real task instructions and IDs, plus the expected
binary success reward. It requires an explicit API URL.

Smoke validates all rows, the pinned instructions, staged task files, and the
integration contract. **It makes no model calls and does not test Docker or run
the verifier.** Docker starts on the first executed case, not during smoke.

Commit and push the recipe, then select that branch and dataset in Beaker's launch
dialog. To launch from the CLI when ready:

```bash
uv run beaker --base-url "$BEAKER_API_BASE_URL" run trigger \
  --integration-id terminalbench --ref '<pushed-branch>' \
  --dataset 'terminalbench-small@<revision>'
```

Preparing the agent, uploading data, and running smoke do not launch an experiment.

## Agent and costs

The seed uses `openai/gpt-6-luna` through Chat Completions, with reasoning `none`,
32 turns, and at most 8,192 output tokens per call. It retains Terminus 2's JSON
command parser, tmux terminal interaction, and context summarization. The editable
class can override the upstream loop and methods; the prompts are shipped as
editable package files. See [ATTRIBUTION.md](ATTRIBUTION.md).

Beaker's normal hosted provider routing supplies model access. A selected
comparison model uses `inference_target(runtime)` through the gateway. Ordinary
runs keep the Luna default. Local model execution needs `OPENAI_API_KEY`.
Model traces use Beaker's supported LiteLLM adapter, including summarization calls.

Concurrency defaults to 2: each selected task requests 1 CPU and 2 GiB of memory,
so two task containers need 4 GiB in addition to the worker and agent processes.
The turn limit and small dataset reduce cost; they are not a dollar budget.
Choose the experiment's spend limit in Beaker before launch. Container image
downloads and verification also take time. Baseline scores and costs are not yet
measured. This subset is for development, not an official leaderboard result.

## Task containers

Harbor uses its Docker backend. Local execution uses an existing Docker daemon
and requires Docker Compose v2. The recipe never starts Docker Desktop for you.

Hosted execution requires a **Beaker Linux VM worker** with root access. The
integration image installs `docker.io` through `apt_install`. If no daemon is
available, a rollout starts its own daemon with a private Unix socket and data
directory. It downloads Docker Compose v2.39.4 with a pinned SHA-256 checksum.
Docker networks use separate address ranges from Modal's usual control network.
Harbor creates and deletes a container for each task; the rollout closes its
daemon when it finishes. No separate Modal or Daytona credentials are needed.

This depends on Beaker's VM runtime; a gVisor-only deployment is not supported.
VM Docker startup is an execution prerequisite, not something structural smoke
proves. See [Modal's Docker documentation](https://modal.com/docs/guide/sandboxes#running-docker-in-a-sandbox).

The agent loop and model calls run outside the task container. Provider keys and
the Docker socket are not mounted into that container. Task tests are copied by
Harbor only after agent execution. Oracle solutions are not staged.

## Fixed small dataset

The source is the exact commit used by Harbor's `terminal-bench@2.0` registry:
`69671fbaac6d67a7ef0dfec016cc38a64ef7a77c`. The downloaded archive is checked against
a pinned SHA-256 digest before extraction. Setup stages only the chosen task's
instruction, configuration, environment, and tests for each case.

The subset was selected without agent scores. From the pinned task metadata:

1. Keep tasks with a prebuilt image, at most 1 CPU, 2 GiB memory, and an agent
   timeout of at most 900 seconds.
2. Sort by expert time estimate, then task ID; take the first 12.
3. Sort those IDs by SHA-256 of `terminalbench-small-v1:<task-id>`; assign the
   first 8 to train and the remaining 4 to test.

| Train | Test |
|---|---|
| kv-store-grpc | crack-7z-hash |
| pytorch-model-recovery | raman-fitting |
| nginx-request-logging | openssl-selfsigned-cert |
| prove-plus-comm | polyglot-c-py |
| hf-model-inference | |
| fix-git | |
| configure-git-webserver | |
| cobol-modernization | |

Beaker uses train tasks to improve the agent and keeps test tasks held out.
Four test tasks give a coarse score, so use a larger held-out set for firm claims.

## Scoring and failures

The objective is the upstream verifier's binary reward: 1 for success, 0 for
failure. `task_success` reports the same value. Checks include verifier output
when available. There is no LLM judge. Upstream task timeouts and resources stay
unchanged; the agent's 32-turn cap is an explicit quick-start limitation.

Provider failures, sandbox startup failures, and missing or malformed verifier
rewards raise errors. Retryable Harbor errors become `RetryableCaseError`.
An agent timeout may be scored using the upstream verifier's valid reward.
Each case has a unique workspace and trial name; only Docker startup is locked,
so cases can run concurrently. Harbor handles task cleanup on cancellation.

## Development checks

```bash
uv sync --group dev --locked
uv run ruff check
uv run ruff format --check
uv run python -m mypy
uv build --no-sources
```

Use strict smoke against the chosen dataset revision as shown above. A real
container run is a separate check and requires a running Docker daemon.
