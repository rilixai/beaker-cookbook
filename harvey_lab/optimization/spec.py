"""Factory that assembles a rilixai :class:`~rilixai.Spec` for Harvey LAB.

The SDK :class:`~rilixai.Spec` binds four things the optimizer needs: the
seed :class:`~rilixai.OptimizationTargets` (``system_prompt`` +
``task_template``), a :class:`~harvey_lab.data.dataset.HarveyLabDataLoader`
that turns uploaded JSONL rows into cases, the async ``run_case`` that
drives the Stirrup agent + rubric judge, and the :class:`HarveyLabScorer`.

``task_source`` is plumbed through so tests inject a fixture-backed
workspace factory; ``judge`` lets tests inject a stub. In production the
defaults are the pinned-commit GitHub task source + the litellm-backed
per-criterion judge.

This module also exports the rilixai Modal-sandbox entry point
:func:`build_spec` — a ``@spec(name="harvey-lab")``-decorated factory
rilixai's sandbox dispatcher invokes once per remote optimization run.
"""

from __future__ import annotations

from typing import Any

from rilixai import CaseDataLoader, OptimizationContext, OptimizationTargets, Spec, spec

from ..agent.agent import HarveyLabAgent, ModelFactory
from ..agent.prompts import harvey_lab_seed_targets
from ..agent.workspace import TaskSource
from ..config import HARVEY_LABS_COMMIT, HARVEY_LABS_REPO, HarveyLabConfig
from ..data.dataset import HarveyLabDataLoader, HarveyLabRecord
from .runtime import build_harvey_lab_run_case
from .scoring import CriterionJudge, HarveyLabScorer


def build_harvey_lab_spec(
    *,
    seed_targets: OptimizationTargets | None = None,
    config: HarveyLabConfig | None = None,
    agent: HarveyLabAgent | None = None,
    task_source: TaskSource | None = None,
    model_factory: ModelFactory | None = None,
    judge: CriterionJudge | None = None,
    name: str = "harvey-lab",
    field_weights: dict[str, float] | None = None,
    data_loader: CaseDataLoader[HarveyLabRecord] | None = None,
) -> Spec:
    """Build a ready-to-run rilixai :class:`~rilixai.Spec` for Harvey LAB.

    ``task_source`` is required at run time unless a pre-built ``agent`` is
    supplied — production passes the pinned-commit GitHub task source; tests
    pass a fixture-backed one. ``judge`` defaults to the litellm per-criterion
    judge; tests inject a stub. The optimizer sources cases from
    ``data_loader`` (uploaded JSONL); :class:`HarveyLabDataLoader` maps each
    row to one :class:`~rilixai.Case`.
    """
    cfg = config or HarveyLabConfig()
    seed = seed_targets or harvey_lab_seed_targets()
    run_case = build_harvey_lab_run_case(
        config=cfg,
        agent=agent,
        task_source=task_source,
        model_factory=model_factory,
        judge=judge,
    )
    return Spec(
        name=name,
        seed_targets=seed,
        data_loader=data_loader or HarveyLabDataLoader(),
        run_case=run_case,
        scorer=HarveyLabScorer(field_weights=field_weights),
    )


# ─── rilixai Modal sandbox entry point ─────────────────────────────────


_DEFAULT_SANDBOX_CONFIG: dict[str, Any] = {
    "task_model": "openai/gpt-4.1-mini-2025-04-14",
    "task_temperature": 0.0,
    "judge_model": "gemini/gemini-2.5-flash",
    "max_turns": 40,
}


@spec(
    name="harvey-lab",
    description="Harvey LAB (legal agent benchmark) — Stirrup harness + all-pass rubric + GEPA",
    metadata={"benchmark": "harvey_lab", "agent_kind": "stirrup"},
    dataset_schema=HarveyLabDataLoader.dataset_schema,
)
def build_spec(ctx: OptimizationContext) -> Spec:
    """Spec factory for the rilixai sandbox path. See README for usage.

    Under the SDK-only shape the optimizer sources cases from the uploaded
    JSONL dataset via :class:`HarveyLabDataLoader`; the agent knobs come from
    ``ctx.config`` (merged over :data:`_DEFAULT_SANDBOX_CONFIG`). The task
    documents are fetched per case from the pinned ``harveyai/harvey-labs``
    commit — network is available inside the sandbox container.
    """
    cfg_in: dict[str, Any] = {**_DEFAULT_SANDBOX_CONFIG, **dict(ctx.config or {})}
    harvey_cfg = HarveyLabConfig(
        task_model=str(cfg_in["task_model"]),
        task_temperature=float(cfg_in["task_temperature"]),
        judge_model=str(cfg_in["judge_model"]),
        max_turns=int(cfg_in["max_turns"]),
    )
    from ..agent.workspace import build_github_task_source

    return build_harvey_lab_spec(
        config=harvey_cfg,
        task_source=build_github_task_source(
            repo=HARVEY_LABS_REPO,
            commit=HARVEY_LABS_COMMIT,
            max_document_chars=harvey_cfg.max_document_chars,
        ),
    )


__all__ = ["build_harvey_lab_spec", "build_spec"]
