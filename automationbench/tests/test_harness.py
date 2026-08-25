"""Hermetic end-to-end tests: real environment, real tools, real rubric,
scripted (network-free) model client."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fake_client import ScriptedClient

from automationbench_skills import runner as runner_mod
from automationbench_skills.data import PUBLIC_DOMAINS, load_samples, load_split, task_family
from automationbench_skills.evaluation.summary import format_summary, summarize
from automationbench_skills.runner import STATE_COLUMNS, _rollout_input, _to_result, get_env
from automationbench_skills.skills_tools import list_skills, read_skill, set_skills_dir


def _sample() -> Any:
    return load_split("test")[0]


async def _rollout(client: ScriptedClient, *, skills: bool, sample: Any = None) -> dict[str, Any]:
    env = get_env(skills=skills)
    return await env.run_rollout(
        _rollout_input(sample or _sample()),
        client,
        "scripted-model",
        {},
        state_columns=STATE_COLUMNS,
    )


class TestSplits:
    def test_frozen_split_shape(self) -> None:
        train, test = load_split("train"), load_split("test")
        assert len(train) == 450 and len(test) == 150
        assert not {s.task_name for s in train} & {s.task_name for s in test}
        for domain in PUBLIC_DOMAINS:
            assert sum(1 for s in test if s.domain == domain) == 25
            assert sum(1 for s in train if s.domain == domain) == 75

    def test_split_regeneration_is_deterministic(self) -> None:
        from automationbench_skills.data.make_splits import make_splits
        from automationbench_skills.data.tasks import read_split_names

        train, test, simple = make_splits()
        assert train == read_split_names("train")
        assert test == read_split_names("test")
        assert simple == read_split_names("simple")

    def test_task_names_globally_unique(self) -> None:
        samples = load_samples(include_simple=True)
        assert len({s.task_name for s in samples}) == len(samples) == 800

    def test_task_family(self) -> None:
        assert task_family("sales.docusign_contract_send") == "docusign"


class TestSkillsTools:
    def test_live_reload_and_listing(self, tmp_path: Path) -> None:
        set_skills_dir(tmp_path)
        assert list_skills() == "No skills available."
        (tmp_path / "alpha.md").write_text("# Alpha summary line\nbody text\n")
        listing = list_skills()
        assert "alpha: Alpha summary line" in listing
        assert "body text" in read_skill("alpha")
        (tmp_path / "alpha.md").write_text("# Changed\nnew body\n")
        assert "alpha: Changed" in list_skills()
        assert "new body" in read_skill("alpha")

    def test_read_skill_missing(self, tmp_path: Path) -> None:
        set_skills_dir(tmp_path)
        (tmp_path / "alpha.md").write_text("# A\n")
        message = read_skill("nope")
        assert "unknown skill" in message.lower() and "alpha" in message


class TestRunner:
    async def test_baseline_has_no_skill_tools(self) -> None:
        client = ScriptedClient()
        output = await _rollout(client, skills=False)
        tool_names = {t["name"] for t in client.calls[0]["tools"]}
        assert "list_skills" not in tool_names and "read_skill" not in tool_names
        result = _to_result(_sample(), output)
        assert result.task_completed_correctly == 0.0
        assert 0.0 <= result.partial_credit <= 1.0
        assert result.trajectory and result.end_state is not None

    async def test_skills_arm_reads_live_files(self, tmp_path: Path) -> None:
        (tmp_path / "howto.md").write_text("# How to do the thing\nSECRET-PROCEDURE\n")
        set_skills_dir(tmp_path)
        client = ScriptedClient(
            turns=[
                {"tool_calls": [{"name": "list_skills"}]},
                {"tool_calls": [{"name": "read_skill", "arguments": {"name": "howto"}}]},
                {"content": "done"},
            ]
        )
        output = await _rollout(client, skills=True)
        tool_names = {t["name"] for t in client.calls[0]["tools"]}
        assert {"list_skills", "read_skill"} <= tool_names
        dump = json.dumps([m if isinstance(m, dict) else m.model_dump(mode="json") for m in output["completion"]])
        assert "How to do the thing" in dump
        assert "SECRET-PROCEDURE" in dump

    async def test_state_resets_between_rollouts(self) -> None:
        sample = _sample()
        out1 = await _rollout(ScriptedClient(), skills=False, sample=sample)
        out2 = await _rollout(ScriptedClient(), skills=False, sample=sample)
        r1, r2 = _to_result(sample, out1), _to_result(sample, out2)
        assert r1.partial_credit == r2.partial_credit
        assert r1.task_completed_correctly == r2.task_completed_correctly
        # end-state entity ids are freshly generated per rollout, but the
        # world structure must be identical across resets
        assert r1.end_state is not None and r2.end_state is not None
        assert set(r1.end_state) == set(r2.end_state)

    def test_env_is_cached(self) -> None:
        assert get_env(skills=False) is get_env(skills=False)
        assert get_env(skills=True) is not get_env(skills=False)

    def test_limited_zapier_with_skills_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            get_env(toolset="limited_zapier", skills=True)

    async def test_run_split_concurrency(self, monkeypatch: Any) -> None:
        samples = load_split("test")[:3]
        client = ScriptedClient()
        monkeypatch.setattr(runner_mod, "get_client", lambda model: ScriptedClient())
        results = await runner_mod.run_split_async(samples, model="scripted-model", skills_dir=None, max_concurrent=2)
        assert [r.task_name for r in results] == [s.task_name for s in samples]
        assert all(r.task_completed_correctly in (0.0, 1.0) for r in results)
        del client


class TestSummary:
    def test_summarize_and_format(self) -> None:
        rows = [
            {"domain": "sales", "task_completed_correctly": 1.0, "partial_credit": 1.0},
            {"domain": "sales", "task_completed_correctly": 0.0, "partial_credit": 0.5},
            {"domain": "hr", "task_completed_correctly": 0.0, "partial_credit": 0.0},
        ]
        summary = summarize(rows)
        assert summary["domains"]["sales"] == {"tasks": 2, "pass_rate": 0.5, "partial_credit": 0.75}
        assert summary["overall"]["tasks"] == 3
        text = format_summary(summary)
        assert "overall" in text and "sales" in text
