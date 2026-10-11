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
    config: dict[str, Any] = field(default_factory=dict)  # RunConfig.behavior() that produced this result
    corpus_manifest: dict[str, int] = field(default_factory=dict)  # CorpusManifest.fingerprint() the agent saw
    retried_attempts: list[dict[str, Any]] = field(default_factory=list)  # attempts superseded by a retry
    # Model calls cancelled in flight by the task timeout. Their tokens never reach us, so
    # ``usage``/``cost_usd`` are lower bounds whenever this is > 0.
    interrupted_requests: int = 0

    def compatible_with(self, cfg: RunConfig, manifest: CorpusManifest | None = None) -> bool:
        """Whether this result was produced by a run with the same behavior-affecting configuration
        and (when ``manifest`` is given) the same reachable corpus.

        Results written before per-result provenance existed carry no ``config`` /
        ``corpus_manifest`` and are trusted; the directory-level ``config.json`` check still applies.
        """
        if self.config and self.config != cfg.behavior():
            return False
        if manifest is not None and self.corpus_manifest and self.corpus_manifest != manifest.fingerprint():
            return False
        return True

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
    """Report §4.1 fn. 9: up to 30 restarts per run after crashes or timeouts. Shared across a split."""

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


def _same_answer(a: str, b: str) -> bool:
    """Scorer equivalence at 0% tolerance, so votes group exactly the way answers are graded."""
    try:
        return bool(float(score_answer(a, b, 0.0)) == 1.0 or float(score_answer(b, a, 0.0)) == 1.0)
    except ValueError:
        return bool(str(normalize_text(a)).casefold() == str(normalize_text(b)).casefold())


def plurality_vote(answers: Sequence[str | None], *, seed: str) -> int:
    """Index of the rollout whose answer wins the plurality; random tiebreak (report App. D.5).

    Answers are grouped by :func:`score_answer` equivalence (``1,000`` / ``$1000`` / ``1000.0``,
    ``Texas`` / ``texas``), not by string, so equivalent formats never split a majority.
    """
    given = [i for i, a in enumerate(answers) if a is not None and a.strip()]
    if not given:
        return 0
    clusters: list[list[int]] = []
    for i in given:
        for cluster in clusters:
            if _same_answer(str(answers[cluster[0]]), str(answers[i])):
                cluster.append(i)
                break
        else:
            clusters.append([i])
    top = max(len(c) for c in clusters)
    winners = [c[0] for c in clusters if len(c) == top]
    return random.Random(seed).choice(winners)


async def _run_episode(
    sample: EvalRecord,
    manifest: CorpusManifest,
    ws: Workspace,
    cfg: RunConfig,
    client_factory: ClientFactory,
    toolset_factory: ToolsetFactory,
    ep: Episode,
) -> tuple[Episode, str | None]:
    """One rollout into ``ep``. Returns (episode, error); ``episode.status`` is 'timeout' on timeout.

    ``ep`` belongs to the caller so that whatever was spent before a crash is still on it."""
    toolset = toolset_factory(ws, cfg)
    agent = OfficeQAAgent(
        client_factory(cfg),
        toolset,
        max_steps=cfg.max_steps,
        window_size=cfg.window_size,
        tool_output_limit=cfg.tool_output_limit,
    )
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
    corpus = corpus_root or ensure_corpus(cfg.corpus, expected_documents=config.EXPECTED_CORPUS_DOCUMENTS)
    manifest = build_manifest(corpus, default=cfg.corpus)
    work = work_dir or (config.cache_dir() / "work")

    started = time.time()
    t0 = time.monotonic()
    attempts = 0
    episodes: list[Episode] = []
    retried: list[Episode] = []  # timed-out / crashed attempts superseded by a retry; their spend still counts
    error: str | None = None
    status = "error"

    for _rollout in range(cfg.n_rollouts):
        while True:
            attempts += 1
            ws = create_workspace(work, sample.uid, corpus, isolate=isolate, fresh=True)
            ep = Episode()
            try:
                _, err = await _run_episode(sample, manifest, ws, cfg, client_factory, toolset_factory, ep)
            except Exception as exc:  # crash: provider error, tool infra failure, ...
                logger.warning("uid=%s attempt %d crashed: %s", sample.uid, attempts, exc)
                ep.status = "error"
                if budget.take():
                    retried.append(ep)
                    continue
                error = f"{type(exc).__name__}: {exc} (retry budget exhausted after {budget.used} retries)"
            else:
                if ep.status == "timeout":
                    logger.warning("uid=%s attempt %d %s", sample.uid, attempts, err)
                    if budget.take():
                        retried.append(ep)
                        continue
                    err = f"{err} (retry budget exhausted after {budget.used} retries)"
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
    interrupted = 0
    for ep_ in episodes + retried:
        usage.add(ep_.usage)
        total_cost += ep_.cost_usd
        cost_known = cost_known and ep_.cost_known
        counts.update(ep_.tool_call_counts)
        interrupted += ep_.interrupted_requests

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
        steps=sum(e.n_steps for e in episodes + retried),
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
        config=cfg.behavior(),
        corpus_manifest=manifest.fingerprint(),
        retried_attempts=[
            {
                "status": ep_.status,
                "steps": ep_.n_steps,
                "tool_calls": ep_.tool_calls,
                "cost_usd": ep_.cost_usd if ep_.cost_known else None,
            }
            for ep_ in retried
        ],
        interrupted_requests=interrupted,
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
    corpus = corpus_root or ensure_corpus(cfg.corpus, expected_documents=config.EXPECTED_CORPUS_DOCUMENTS)

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
