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
from automationbench_skills.prompts import load_system_prompt, with_system_prompt
from automationbench_skills.runner import STATE_COLUMNS, _rollout_input, _to_result, get_env
from automationbench_skills.skills_tools import SkillUsage, list_skills, read_skill, set_skills_dir, skill_usage


RECIPE_ROOT = Path(__file__).parent.parent


def _sample() -> Any:
    return load_split("test")[0]


async def _rollout(
    client: ScriptedClient, *, skills: bool, sample: Any = None, system_prompt: str | None = None
) -> dict[str, Any]:
    env = get_env(skills=skills)
    return await env.run_rollout(
        _rollout_input(sample or _sample(), system_prompt),
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
    @staticmethod
    def _write(root: Path, skill_id: str, description: str, body: str) -> None:
        p = root / skill_id / "SKILL.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"---\nname: {skill_id.split('/')[-1]}\ndescription: {description}\n---\n{body}\n")

    def test_nested_discovery_and_live_reload(self, tmp_path: Path) -> None:
        set_skills_dir(tmp_path)
        assert list_skills() == "No skills available."
        self._write(tmp_path, "apps/gmail", "Gmail procedures", "body text")
        self._write(tmp_path, "domains/finance", "Finance playbooks", "finance body")
        listing = list_skills()
        assert "apps/gmail: Gmail procedures" in listing
        assert "domains/finance: Finance playbooks" in listing
        assert "body text" in read_skill("apps/gmail")
        assert "finance body" in read_skill("domains/finance")
        self._write(tmp_path, "apps/gmail", "Changed", "new body")
        assert "apps/gmail: Changed" in list_skills()
        assert "new body" in read_skill("apps/gmail")

    def test_read_skill_missing(self, tmp_path: Path) -> None:
        set_skills_dir(tmp_path)
        self._write(tmp_path, "apps/gmail", "Gmail", "body")
        message = read_skill("nope")
        assert "unknown skill" in message.lower() and "apps/gmail" in message

    def test_shipped_seed_stubs(self) -> None:
        shipped = RECIPE_ROOT / "skills"
        set_skills_dir(shipped)
        listing = list_skills()
        for domain in PUBLIC_DOMAINS:
            assert f"domains/{domain}: " in listing
        for app in ["gmail", "google_sheets", "google_drive", "slack", "salesforce"]:
            assert f"apps/{app}: " in listing
        assert "unknown skill" not in read_skill("apps/gmail").lower()


class TestSkillUsage:
    def test_reads_from_dict_and_openai_shaped_tool_calls(self) -> None:
        completion = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": ['{"id": "c0", "name": "list_skills", "arguments": "{}"}'],
            },
            {"role": "tool", "tool_call_id": "c0", "content": "..."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "function": {"name": "read_skill", "arguments": '{"skill_id": "domains/hr"}'}},
                    {"id": "c2", "function": {"name": "read_skill", "arguments": '{"skill_id": "/apps/gmail/"}'}},
                    {"id": "c3", "function": {"name": "search_tools", "arguments": '{"query": "gmail"}'}},
                ],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "read_skill", "arguments": {"skill_id": "domains/hr"}}],
            },
            {"role": "assistant", "content": "done"},
        ]
        usage = skill_usage(completion)
        assert usage == SkillUsage(listed=True, reads={"domains/hr": 2, "apps/gmail": 2})
        assert SkillUsage.from_json(usage.to_json()) == usage
        assert skill_usage([]) == SkillUsage(listed=False, reads={})
        assert SkillUsage.from_json(None) is None


class TestPrompts:
    def test_file_replaces_the_system_message_and_keeps_the_task(self) -> None:
        prompt = [{"role": "system", "content": "BENCHMARK PROMPT"}, {"role": "user", "content": "do the task"}]
        out = with_system_prompt(prompt, "OURS")
        assert out == [{"role": "system", "content": "OURS"}, prompt[1]]
        assert prompt[0]["content"] == "BENCHMARK PROMPT"
        assert with_system_prompt(prompt, None) is prompt
        assert with_system_prompt(prompt, "") is prompt
        assert with_system_prompt("plain", "OURS") == "plain"
        assert with_system_prompt([prompt[1]], "OURS") == [{"role": "system", "content": "OURS"}, prompt[1]]

    def test_load_reads_live_and_tolerates_absence(self, tmp_path: Path) -> None:
        assert load_system_prompt(None) is None
        assert load_system_prompt(tmp_path) is None
        (tmp_path / "system.md").write_text("  \n")
        assert load_system_prompt(tmp_path) is None
        (tmp_path / "system.md").write_text("first\n")
        assert load_system_prompt(tmp_path) == "first"
        (tmp_path / "system.md").write_text("second\n")
        assert load_system_prompt(tmp_path) == "second"

    def test_shipped_seed_starts_with_the_benchmark_prompt_verbatim(self) -> None:
        text = load_system_prompt(RECIPE_ROOT / "prompts")
        assert text is not None
        upstream = {s.prompt[0]["content"] for s in load_split("train") + load_split("test")}
        assert len(upstream) == 1
        assert text.startswith(upstream.pop())
        for domain in PUBLIC_DOMAINS:
            assert domain in text
        assert "list_skills" in text and "read_skill" in text


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
        howto = tmp_path / "apps" / "howto" / "SKILL.md"
        howto.parent.mkdir(parents=True)
        howto.write_text("---\nname: howto\ndescription: How to do the thing\n---\nSECRET-PROCEDURE\n")
        set_skills_dir(tmp_path)
        client = ScriptedClient(
            turns=[
                {"tool_calls": [{"name": "list_skills"}]},
                {"tool_calls": [{"name": "read_skill", "arguments": {"skill_id": "apps/howto"}}]},
                {"content": "done"},
            ]
        )
        output = await _rollout(client, skills=True)
        tool_names = {t["name"] for t in client.calls[0]["tools"]}
        assert {"list_skills", "read_skill"} <= tool_names
        dump = json.dumps([m if isinstance(m, dict) else m.model_dump(mode="json") for m in output["completion"]])
        assert "How to do the thing" in dump
        assert "SECRET-PROCEDURE" in dump
        assert skill_usage(output["completion"]) == SkillUsage(listed=True, reads={"apps/howto": 2})

    async def test_system_prompt_reaches_the_model(self) -> None:
        sample = _sample()
        baseline = ScriptedClient()
        await _rollout(baseline, skills=True, sample=sample)
        ours = ScriptedClient()
        await _rollout(ours, skills=True, sample=sample, system_prompt="READ YOUR SKILLS FIRST")
        base_system = baseline.calls[0]["prompt"][0]
        system = ours.calls[0]["prompt"][0]
        assert base_system.role == system.role == "system"
        assert base_system.content == sample.prompt[0]["content"]
        assert system.content == "READ YOUR SKILLS FIRST"
        assert ours.calls[0]["prompt"][1:] == baseline.calls[0]["prompt"][1:]

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

    async def test_task_timeout_returns_error_result(self, monkeypatch: Any) -> None:
        import asyncio

        class StallingClient(ScriptedClient):
            async def get_native_response(self, *args: Any, **kwargs: Any) -> Any:
                await asyncio.sleep(30)

        monkeypatch.setattr(runner_mod, "get_client", lambda model: StallingClient())
        result = await runner_mod.run_one_async(_sample(), skills_dir=None, timeout=0.2)
        assert result.error is not None and "timeout" in str(result.error)
        assert result.partial_credit == 0.0 and result.task_completed_correctly == 0.0

    def test_client_cache_is_per_event_loop(self, monkeypatch: Any) -> None:
        import asyncio

        from automationbench_skills.runner import ModelSpec, get_client

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        spec = ModelSpec(name="gpt-5-mini")

        async def grab() -> Any:
            return get_client(spec)

        async def grab_twice() -> tuple[Any, Any]:
            return get_client(spec), get_client(spec)

        first, second = asyncio.run(grab_twice())
        assert first is second  # same loop -> shared client
        assert asyncio.run(grab()) is not first  # new asyncio.run -> fresh client
        # closed-loop entries are evicted, so the cache doesn't grow across runs
        assert len([k for k in runner_mod._CLIENT_CACHE if k[0] == spec]) == 1

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
