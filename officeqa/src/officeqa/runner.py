"""Per-sample and per-split runners: the Beaker entry point.

    from officeqa import load_split, run_one
    sample = load_split("train")[0]
    result = run_one(sample, model="gpt-5.4", tools=["fs", "repl"], corpus="pdfs")
    result.correct, result.trajectory, result.tool_calls, result.cost_usd, result.latency_s

``run_one`` scores against the sample's ground truth but the agent only ever
receives ``sample.agent_input(manifest)`` (uid, question, corpus manifest).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from officeqa import config
from officeqa.agent.agent import Episode, LiteLLMClient, LLMClient, OfficeQAAgent, Usage
from officeqa.agent.tools import ToolSet, WebSearchBackend, build_toolset
from officeqa.config import RunConfig
from officeqa.data.corpus import Workspace, create_workspace, ensure_corpus
from officeqa.data.dataset import EvalRecord
from officeqa.data.manifest import CorpusManifest, build_manifest
from officeqa.evaluation.reward import normalize_text, score_answer


logger = logging.getLogger(__name__)

ClientFactory = Callable[[RunConfig], LLMClient]
ToolsetFactory = Callable[[Workspace, RunConfig], ToolSet]

CLEAN_STATUSES = frozenset({"answered", "no_answer"})


@dataclass
class RunResult:
    uid: str
    question: str
    final_answer: str | None
    status: str  # answered | no_answer | timeout | error
    scores: dict[str, float]  # tolerance (as str) -> 0/1
    trajectory: list[dict[str, Any]]
    tool_calls: int
    tool_call_counts: dict[str, int]
    steps: int
    usage: dict[str, int]
    cost_usd: float | None
    latency_s: float
    queue_wait_s: float
    attempts: int
    model: str
    error: str | None = None
    rollouts: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def correct(self) -> float:
        return self.scores.get(str(config.HEADLINE_TOLERANCE), 0.0)

    @property
    def clean(self) -> bool:
        return self.status in CLEAN_STATUSES

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["correct"] = self.correct
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> RunResult:
        d = dict(d)
        d.pop("correct", None)
        return cls(**d)


def score_all(ground_truth: str, final_answer: str | None) -> dict[str, float]:
    """0/1 at each tolerance; a missing ``<FINAL_ANSWER>`` scores 0 everywhere."""
    if final_answer is None or not final_answer.strip():
        return {str(t): 0.0 for t in config.TOLERANCES}
    return {str(t): float(score_answer(ground_truth, final_answer, t)) for t in config.TOLERANCES}


class RetryBudget:
    """Report §4.1 fn. 9: up to 30 restarts per run after crashes. Shared across a split."""

    def __init__(self, limit: int = config.MAX_RETRIES) -> None:
        self.limit = limit
        self.used = 0

    def take(self) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


def _default_client_factory(cfg: RunConfig) -> LLMClient:
    return LiteLLMClient(
        cfg.model,
        reasoning_effort=cfg.reasoning_effort,
        timeout_s=cfg.llm_timeout_s,
        max_output_tokens=cfg.max_output_tokens,
    )


# Module-level so the CLI (and tests) can swap the model client without threading it everywhere.
CLIENT_FACTORY: ClientFactory = _default_client_factory


def make_toolset_factory(web_backend: WebSearchBackend | None = None) -> ToolsetFactory:
    def factory(ws: Workspace, cfg: RunConfig) -> ToolSet:
        if cfg.search != "fs":
            raise NotImplementedError(
                f"search arm {cfg.search!r} (vector search, report App. D.4) is stubbed; only 'fs' is implemented"
            )
        return build_toolset(ws, cfg.tools, output_limit=cfg.tool_output_limit, web_backend=web_backend)

    return factory


def _vote_key(answer: str) -> str:
    return str(normalize_text(answer)).replace(",", "").replace("$", "").strip()


def plurality_vote(answers: Sequence[str | None], *, seed: str) -> int:
    """Index of the rollout whose answer wins the plurality; random tiebreak (report App. D.5)."""
    keyed = [(i, _vote_key(a)) for i, a in enumerate(answers) if a is not None and a.strip()]
    if not keyed:
        return 0
    counts = Counter(k for _, k in keyed)
    top = max(counts.values())
    winners = [k for k, c in counts.items() if c == top]
    chosen = random.Random(seed).choice(sorted(winners))
    return next(i for i, k in keyed if k == chosen)


async def _run_episode(
    sample: EvalRecord,
    manifest: CorpusManifest,
    ws: Workspace,
    cfg: RunConfig,
    client_factory: ClientFactory,
    toolset_factory: ToolsetFactory,
) -> tuple[Episode, str | None]:
    """One rollout. Returns (episode, error); ``episode.status`` is 'timeout' on timeout."""
    toolset = toolset_factory(ws, cfg)
    agent = OfficeQAAgent(
        client_factory(cfg),
        toolset,
        max_steps=cfg.max_steps,
        window_size=cfg.window_size,
        tool_output_limit=cfg.tool_output_limit,
    )
    ep = Episode()
    try:
        await asyncio.wait_for(agent.forward(sample.agent_input(manifest), ep), timeout=cfg.task_timeout_s)
        return ep, None
    except asyncio.TimeoutError:
        ep.status = "timeout"
        return ep, f"task timed out after {cfg.task_timeout_s:.0f}s"
    finally:
        toolset.close()


async def run_one_async(
    sample: EvalRecord,
    *,
    cfg: RunConfig | None = None,
    corpus_root: Path | None = None,
    work_dir: Path | None = None,
    client_factory: ClientFactory | None = None,
    toolset_factory: ToolsetFactory | None = None,
    retry_budget: RetryBudget | None = None,
    isolate: bool = config.ISOLATE_PER_QUESTION,
    queue_wait_s: float = 0.0,
    **overrides: Any,
) -> RunResult:
    if cfg is None:
        if "tools" in overrides:
            overrides["tools"] = tuple(overrides["tools"])
        cfg = RunConfig(**overrides)
    elif overrides:
        raise TypeError(f"pass either cfg or RunConfig overrides, not both: {sorted(overrides)}")
    client_factory = client_factory or CLIENT_FACTORY
    toolset_factory = toolset_factory or make_toolset_factory()
    budget = retry_budget or RetryBudget(cfg.max_retries)
    corpus = corpus_root or ensure_corpus(cfg.corpus)
    manifest = build_manifest(corpus, default=cfg.corpus)
    work = work_dir or (config.cache_dir() / "work")

    started = time.time()
    t0 = time.monotonic()
    attempts = 0
    episodes: list[Episode] = []
    error: str | None = None
    status = "error"

    for _rollout in range(cfg.n_rollouts):
        while True:
            attempts += 1
            ws = create_workspace(work, sample.uid, corpus, isolate=isolate, fresh=True)
            try:
                ep, err = await _run_episode(sample, manifest, ws, cfg, client_factory, toolset_factory)
            except Exception as exc:  # crash: provider error, tool infra failure, ...
                logger.warning("uid=%s attempt %d crashed: %s", sample.uid, attempts, exc)
                if budget.take():
                    continue
                error = f"{type(exc).__name__}: {exc} (retry budget exhausted after {budget.used} retries)"
                ep = Episode()
                ep.status = "error"
            else:
                error = err or error
            episodes.append(ep)
            break

    if len(episodes) == 1:
        chosen = episodes[0]
        rollouts: list[dict[str, Any]] = []
    else:
        idx = plurality_vote([e.final_answer for e in episodes], seed=f"{sample.uid}:{cfg.model}")
        chosen = episodes[idx]
        rollouts = [
            {
                "final_answer": ep_.final_answer,
                "status": ep_.status,
                "steps": ep_.n_steps,
                "tool_calls": ep_.tool_calls,
                "cost_usd": ep_.cost_usd if ep_.cost_known else None,
                "chosen": i == idx,
            }
            for i, ep_ in enumerate(episodes)
        ]

    status = chosen.status if chosen.status != "running" else "error"
    usage = Usage()
    total_cost = 0.0
    cost_known = True
    counts: Counter[str] = Counter()
    for ep_ in episodes:
        usage.add(ep_.usage)
        total_cost += ep_.cost_usd
        cost_known = cost_known and ep_.cost_known
        counts.update(ep_.tool_call_counts)

    final_answer = chosen.final_answer if status == "answered" else None
    return RunResult(
        uid=sample.uid,
        question=sample.question,
        final_answer=final_answer,
        status=status,
        scores=score_all(sample.answer, final_answer),
        trajectory=chosen.trajectory,
        tool_calls=sum(counts.values()),
        tool_call_counts=dict(counts),
        steps=sum(e.n_steps for e in episodes),
        usage=asdict(usage),
        cost_usd=total_cost if cost_known else None,
        latency_s=time.monotonic() - t0,
        queue_wait_s=queue_wait_s,
        attempts=attempts,
        model=cfg.model,
        error=error,
        rollouts=rollouts,
        started_at=started,
        finished_at=time.time(),
    )


def run_one(sample: EvalRecord, **kwargs: Any) -> RunResult:
    """Synchronous wrapper; see :func:`run_one_async`. Keyword args become :class:`RunConfig` fields."""
    return asyncio.run(run_one_async(sample, **kwargs))


ResultCallback = Callable[[RunResult], None]


async def run_split_async(
    samples: Sequence[EvalRecord],
    cfg: RunConfig,
    *,
    corpus_root: Path | None = None,
    work_dir: Path | None = None,
    client_factory: ClientFactory | None = None,
    toolset_factory: ToolsetFactory | None = None,
    on_result: ResultCallback | None = None,
    isolate: bool = config.ISOLATE_PER_QUESTION,
) -> list[RunResult]:
    """Run many samples with bounded concurrency; results arrive via ``on_result`` as they finish."""
    sem = asyncio.Semaphore(cfg.max_concurrent)
    budget = RetryBudget(cfg.max_retries)
    corpus = corpus_root or ensure_corpus(cfg.corpus)

    async def one(sample: EvalRecord) -> RunResult:
        queued = time.monotonic()
        async with sem:
            wait = time.monotonic() - queued
            result = await run_one_async(
                sample,
                cfg=cfg,
                corpus_root=corpus,
                work_dir=work_dir,
                client_factory=client_factory,
                toolset_factory=toolset_factory,
                retry_budget=budget,
                isolate=isolate,
                queue_wait_s=wait,
            )
        if on_result is not None:
            on_result(result)
        return result

    return list(await asyncio.gather(*(one(s) for s in samples)))


def run_split(samples: Sequence[EvalRecord], cfg: RunConfig, **kwargs: Any) -> list[RunResult]:
    return asyncio.run(run_split_async(samples, cfg, **kwargs))


def write_result(result: RunResult, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{result.uid}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def read_results(results_dir: Path) -> dict[str, RunResult]:
    out: dict[str, RunResult] = {}
    if not results_dir.is_dir():
        return out
    for path in sorted(results_dir.glob("*.json")):
        try:
            out[path.stem] = RunResult.from_json(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning("skipping unreadable result %s: %s", path, e)
    return out
