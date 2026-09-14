"""OfficeQA Pro recipe: baseline grounded-reasoning agent + local evaluation.

from officeqa import load_split, run_one

sample = load_split("train")[0]
result = run_one(sample, model="gpt-5.4", tools=["fs", "repl"], corpus="pdfs")
result.correct, result.trajectory, result.tool_calls, result.cost_usd, result.latency_s
"""

import importlib

from officeqa.config import RunConfig
from officeqa.data import AgentInput, CorpusManifest, EvalRecord, fetch_corpus, load_records, load_split
from officeqa.runner import RunResult, run_one, run_one_async, run_split, run_split_async


def reload() -> None:
    """Re-import the agent (prompts, tools, loop, runner) so edits under ``agent/`` apply without a restart."""
    from officeqa import runner as _runner
    from officeqa.agent import agent as _agent
    from officeqa.agent import prompts as _prompts
    from officeqa.agent import tools as _tools

    for module in (_prompts, _tools, _agent, _runner):
        importlib.reload(module)
    globals().update(
        RunResult=_runner.RunResult,
        run_one=_runner.run_one,
        run_one_async=_runner.run_one_async,
        run_split=_runner.run_split,
        run_split_async=_runner.run_split_async,
    )


__all__ = [
    "AgentInput",
    "CorpusManifest",
    "EvalRecord",
    "RunConfig",
    "RunResult",
    "fetch_corpus",
    "load_records",
    "load_split",
    "reload",
    "run_one",
    "run_one_async",
    "run_split",
    "run_split_async",
]
