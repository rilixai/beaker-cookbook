"""CLI entrypoint for the HotpotQA agent.

Run as ``python -m hotpotqa.cli <command> ...`` after installing this
project from its directory (``cd hotpotqa && uv sync --group dev``).

Subcommands:
* ``run`` — run the agent over a split and write its answers +
  retrieval traces to ``run_outputs.json`` (no scoring).
* ``evaluate`` — run the agent AND score every case, writing
  ``eval_summary.json`` (aggregate exact match / answer F1 /
  supporting-title recall) and ``eval_outputs.json`` (per case).

Both commands default to the 300-case test slice under the fullwiki /
k=7 setup. Data loading slices the HotpotQA *train* split into
``[0, 40%)`` test, ``[40%, 80%)`` validation, ``[80%, 100%)`` train, then
samples with ``random.Random(1)``; see
:func:`hotpotqa.data.dataset.load_hotpotqa_paper_split` for the full
provenance.

``--no-network`` is the guard for dry runs: instead of downloading the
HF dataset it raises ``RuntimeError``, so a misconfigured invocation
never accidentally hits HuggingFace or an LLM provider.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any

from .agent.agent import HotpotQAPydanticAgent
from .config import HotpotQAConfig, bare_openai_model
from .data.dataset import HotpotQARecord, load_hotpotqa_paper_split
from .evaluation import (
    ANSWER_F1_FIELD,
    ANSWER_FIELD,
    SUPPORTING_TITLES_RECALL_FIELD,
    eval_summary,
    evaluate_agent_on_records,
    run_agent_on_record,
    write_json,
)


logger = logging.getLogger("hotpotqa")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "HotpotQA agent: a PydanticAI tool-using agent (retrieve_k + summarize) "
            "that answers multi-hop Wikipedia questions. `run` executes it and dumps "
            "its answers; `evaluate` also scores them (exact match / answer F1 / "
            "supporting-title recall)."
        ),
    )
    parser.add_argument(
        "command",
        choices=("run", "evaluate"),
        help="`run` executes the agent and dumps answers; `evaluate` also scores them.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=300,
        help="Validation case count drawn from the [40%%, 80%%) slice. Default 300.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=300,
        help="Test case count drawn from the [0%%, 40%%) slice. Default 300.",
    )
    parser.add_argument(
        "--split",
        choices=("test", "validation"),
        default="test",
        help="Which split to run on. Default `test`.",
    )
    parser.add_argument(
        "--retrieval",
        choices=("fullwiki", "distractor"),
        default="fullwiki",
        help="Retrieval corpus. `fullwiki` (default, open-domain) or `distractor` (cheap, case-local).",
    )
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=7,
        help="Paragraphs returned per retrieval call. Default 7.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=8,
        help="Cap on the agent's tool-use loop length.",
    )
    parser.add_argument(
        "--pydantic-agent-model",
        type=str,
        default=None,
        help="PydanticAI model spec for the outer agent (e.g. 'openai:gpt-4.1-mini').",
    )
    parser.add_argument(
        "--task-model",
        type=str,
        default="openai/gpt-4.1-mini",
        help="Default model for the PydanticAI agent + summarize tool. Slash form is auto-translated.",
    )
    parser.add_argument(
        "--task-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the task LLM. Defaults to 0.0.",
    )
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hotpotqa_run"),
        help="Directory where results are written.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="HuggingFace datasets cache directory (defaults to the user cache).",
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Refuse to download the HF dataset (test / dry-run guard).",
    )
    return parser.parse_args(argv)


def _fmt_hms(seconds: float) -> str:
    total_seconds = int(round(max(0.0, seconds)))
    hours, rem = divmod(total_seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{sec:02d}"


def _load_records(args: argparse.Namespace) -> list[HotpotQARecord]:
    """Load the split both commands operate on."""
    if args.no_network:
        raise RuntimeError(
            "Refusing to download the HotpotQA dataset because --no-network was set. "
            "This guard is for dry runs / accidental-spend prevention. For offline "
            "structural checks run: uv run python -m pytest (scripted FunctionModel, "
            "zero network)."
        )
    config = "fullwiki" if args.retrieval == "fullwiki" else "distractor"
    cache_dir = str(args.cache_dir) if args.cache_dir else None
    if args.split == "test":
        if args.test_size <= 0:
            raise ValueError("--split test requires --test-size > 0.")
        return load_hotpotqa_paper_split("test", max_cases=args.test_size, config=config, cache_dir=cache_dir)
    if args.val_size <= 0:
        raise ValueError("--split validation requires --val-size > 0.")
    return load_hotpotqa_paper_split("validation", max_cases=args.val_size, config=config, cache_dir=cache_dir)


def _config_from_args(args: argparse.Namespace) -> HotpotQAConfig:
    # Fall back to the shared ``--task-model`` when no explicit agent spec is
    # given. The slash→colon rewrite is handled centrally by ``HotpotQAConfig``
    # (see ``to_pydantic_ai_model``), so no per-call normalization is needed here.
    return HotpotQAConfig(
        retrieval_mode=args.retrieval,
        retrieve_k=args.retrieve_k,
        max_iters=args.max_iters,
        pydantic_agent_model=args.pydantic_agent_model or args.task_model,
        pydantic_agent_temperature=args.task_temperature,
    )


def _build_agent(config: HotpotQAConfig) -> HotpotQAPydanticAgent:
    model = config.pydantic_agent_model
    if not model:
        raise ValueError("A model is required: pass --task-model or --pydantic-agent-model.")
    return HotpotQAPydanticAgent(
        model=model,
        top_k=config.retrieve_k,
        max_iters=config.max_iters,
        temperature=config.pydantic_agent_temperature,
        # The summarize tool calls the OpenAI API directly, which wants the
        # bare model name rather than PydanticAI's ``provider:model`` form.
        summarize_model=bare_openai_model(model),
    )


def _run_run(args: argparse.Namespace) -> int:
    records = _load_records(args)
    if not records:
        logger.error("run got no cases for --split %s.", args.split)
        return 2
    config = _config_from_args(args)
    agent = _build_agent(config)

    async def _run_all() -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(max(1, args.max_concurrency))

        async def _one(record: HotpotQARecord) -> dict[str, Any]:
            async with semaphore:
                # Contain per-case failures so one erroring case does not
                # cancel the batch (mirrors evaluate_agent_on_records).
                try:
                    output = await run_agent_on_record(agent=agent, record=record, config=config)
                except Exception as exc:  # noqa: BLE001 - report, don't abort
                    logger.warning("Case %s failed: %s", record.case_id, exc)
                    return {
                        "case_id": record.case_id,
                        "question": record.question,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                if output.error:
                    # The agent reports a failed run instead of raising.
                    return {
                        "case_id": record.case_id,
                        "question": record.question,
                        "error": output.error,
                    }
                return {
                    "case_id": record.case_id,
                    "question": record.question,
                    "answer": output.answer,
                    "gold_answer": record.answer,
                    "retrieved_titles": [p.title for p in output.retrieved_paragraphs],
                    "num_tool_calls": len(output.tool_calls),
                }

        return list(await asyncio.gather(*[_one(record) for record in records]))

    logger.info("Running the agent on %d case(s)...", len(records))
    started = time.monotonic()
    results = asyncio.run(_run_all())
    write_json(args.output_dir / "run_outputs.json", results)
    num_errored = sum(1 for result in results if "error" in result)
    logger.info(
        "Wrote answers for %d case(s) under %s in %s (%d succeeded, %d errored)",
        len(results),
        args.output_dir,
        _fmt_hms(time.monotonic() - started),
        len(results) - num_errored,
        num_errored,
    )
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    records = _load_records(args)
    if not records:
        logger.error("evaluate got no cases for --split %s.", args.split)
        return 2
    config = _config_from_args(args)
    agent = _build_agent(config)
    logger.info("Starting evaluate on split=%s (%d cases)...", args.split, len(records))
    started = time.monotonic()
    report = asyncio.run(
        evaluate_agent_on_records(
            agent=agent,
            records=records,
            config=config,
            max_concurrency=args.max_concurrency,
        )
    )
    write_json(args.output_dir / "eval_summary.json", eval_summary(report, split=args.split))
    write_json(args.output_dir / "eval_outputs.json", report.per_case)
    logger.info("evaluate complete in %s", _fmt_hms(time.monotonic() - started))
    logger.info(
        "Split=%s | %s=%.4f %s=%.4f %s=%.4f over %d case(s) (%d scored, %d errored, %d unscoreable)",
        args.split,
        ANSWER_FIELD,
        report.field_accuracies.get(ANSWER_FIELD, report.objective),
        ANSWER_F1_FIELD,
        report.field_accuracies.get(ANSWER_F1_FIELD, 0.0),
        SUPPORTING_TITLES_RECALL_FIELD,
        report.field_accuracies.get(SUPPORTING_TITLES_RECALL_FIELD, 0.0),
        report.num_cases,
        report.num_scored,
        report.num_errored,
        report.num_unscoreable,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    if args.command == "run":
        return _run_run(args)
    return _run_evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
