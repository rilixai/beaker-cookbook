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
from automationbench_skills.prompts import load_system_prompt, task_clock, with_system_prompt
from automationbench_skills.runner import (
    DEFAULT_REASONING_EFFORT,
    OPENROUTER_BASE_URL,
    STATE_COLUMNS,
    ModelSpec,
    _rollout_input,
    _to_result,
    get_env,
)
from automationbench_skills.skills_tools import list_skills, read_skill, set_skills_dir


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
    def test_filtered_split_shape(self) -> None:
        train, test = load_split("train"), load_split("test")
        assert len(train) == 374 and len(test) == 104
        assert not {s.task_name for s in train} & {s.task_name for s in test}
        from automationbench_skills.data.tasks import read_split_names

        excluded = read_split_names("excluded_cases")
        retained = {s.task_name for s in train + test}
        assert len(excluded) == len(set(excluded)) == 122
        assert not retained & set(excluded)
        assert retained | set(excluded) == {s.task_name for s in load_samples()}
        for split, samples in (("train", train), ("test", test)):
            assert [s.task_name for s in samples] == [name for name in read_split_names(split) if name not in excluded]
        expected = {
            "sales": (64, 20),
            "marketing": (68, 19),
            "operations": (59, 11),
            "support": (48, 12),
            "finance": (69, 22),
            "hr": (66, 20),
        }
        for domain in PUBLIC_DOMAINS:
            assert (sum(s.domain == domain for s in train), sum(s.domain == domain for s in test)) == expected[domain]

    def test_split_regeneration_is_deterministic(self) -> None:
        from automationbench_skills.data.make_splits import make_splits
        from automationbench_skills.data.tasks import read_split_names

        train, test = make_splits()
        assert train == read_split_names("train")
        assert test == read_split_names("test")

    def test_task_names_globally_unique(self) -> None:
        samples = load_samples()
        assert len({s.task_name for s in samples}) == len(samples) == 600

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

    def test_clock_appends_to_the_system_message(self) -> None:
        prompt = [{"role": "system", "content": "BENCHMARK PROMPT"}, {"role": "user", "content": "do the task"}]
        out = with_system_prompt(prompt, "OURS", clock="2026-03-10T09:00:00Z")
        assert out[0] == {"role": "system", "content": "OURS\n\nCurrent date and time: 2026-03-10T09:00:00Z"}
        assert out[1:] == prompt[1:]

    def test_clock_without_a_system_prompt_keeps_the_rows_system_message(self) -> None:
        prompt = [{"role": "system", "content": "BENCHMARK PROMPT"}, {"role": "user", "content": "do the task"}]
        out = with_system_prompt(prompt, None, clock="2026-03-10T09:00:00Z")
        assert out[0] == {
            "role": "system",
            "content": "BENCHMARK PROMPT\n\nCurrent date and time: 2026-03-10T09:00:00Z",
        }
        assert out[1:] == prompt[1:]
        assert prompt[0]["content"] == "BENCHMARK PROMPT"
        out = with_system_prompt([prompt[1]], None, clock="2026-03-10T09:00:00Z")
        assert out[0] == {"role": "system", "content": "Current date and time: 2026-03-10T09:00:00Z"}

    def test_no_clock_is_unchanged_behaviour(self) -> None:
        prompt = [{"role": "system", "content": "BENCHMARK PROMPT"}, {"role": "user", "content": "do the task"}]
        assert with_system_prompt(prompt, "OURS", clock=None) == [{"role": "system", "content": "OURS"}, prompt[1]]
        assert with_system_prompt(prompt, None, clock=None) is prompt
        assert with_system_prompt(prompt, "OURS", clock="") == [{"role": "system", "content": "OURS"}, prompt[1]]

    def test_task_clock_reads_dict_or_json_initial_state(self) -> None:
        info = {"initial_state": {"meta": {"current_time": "2026-03-10T09:00:00Z"}}}
        assert task_clock(info) == "2026-03-10T09:00:00Z"
        assert task_clock({"initial_state": json.dumps(info["initial_state"])}) == "2026-03-10T09:00:00Z"
        assert task_clock({}) is None
        assert task_clock({"initial_state": {"meta": {}}}) is None
        assert task_clock({"initial_state": {}}) is None
        assert task_clock({"initial_state": "not json"}) is None

    def test_load_reads_live_and_tolerates_absence(self, tmp_path: Path) -> None:
        assert load_system_prompt(None) is None
        assert load_system_prompt(tmp_path) is None
        (tmp_path / "system.md").write_text("  \n")
        assert load_system_prompt(tmp_path) is None
        (tmp_path / "system.md").write_text("first\n")
        assert load_system_prompt(tmp_path) == "first"
        (tmp_path / "system.md").write_text("second\n")
        assert load_system_prompt(tmp_path) == "second"
        assert load_system_prompt(tmp_path, skills=False) is None
        (tmp_path / "system_no_skills.md").write_text("plain\n")
        assert load_system_prompt(tmp_path, skills=False) == "plain"
        assert load_system_prompt(tmp_path) == "second"

    def test_shipped_seeds_start_with_the_benchmark_prompt_verbatim(self) -> None:
        upstream = {s.prompt[0]["content"] for s in load_split("train") + load_split("test")}
        assert len(upstream) == 1
        benchmark = upstream.pop().strip()
        assert load_system_prompt(RECIPE_ROOT / "prompts", skills=False) == benchmark
        text = load_system_prompt(RECIPE_ROOT / "prompts")
        assert text is not None
        assert text.startswith(benchmark)
        for domain in PUBLIC_DOMAINS:
            assert domain in text
        assert "list_skills" in text and "read_skill" in text


class TestRunner:
    def test_openrouter_routing(self) -> None:
        openrouter = ModelSpec(name="z-ai/glm-5.3-flash", reasoning_effort="max")
        assert openrouter.is_openrouter()
        assert openrouter.resolved_api() == "chat_completions"
        assert openrouter.effective_base_url() == OPENROUTER_BASE_URL
        assert openrouter.effective_api_key_var() == "OPENROUTER_API_KEY"
        assert openrouter.sampling_args() == {
            "extra_body": {"reasoning": {"effort": "max"}},
            "max_completion_tokens": 32768,
        }

        qwen = ModelSpec(name="qwen/qwen3.8-flash", reasoning_effort="default", reasoning_enabled=True)
        assert qwen.sampling_args() == {"extra_body": {"reasoning": {"enabled": True}}, "max_completion_tokens": 32768}

        native = ModelSpec(name="gpt-6-astra")
        assert not native.is_openrouter()
        assert native.effective_api_key_var() == "OPENAI_API_KEY"
        assert native.sampling_args() == {"reasoning_effort": DEFAULT_REASONING_EFFORT, "max_completion_tokens": 32768}

        explicit = ModelSpec(name="z-ai/glm-5.3-flash", api_key_var="CUSTOM_API_KEY")
        assert explicit.effective_api_key_var() == "CUSTOM_API_KEY"

    def test_record_cost_reads_openrouter_usage(self) -> None:
        from types import SimpleNamespace

        import pytest
        from openai.types import CompletionUsage

        from automationbench_skills.clients import record_cost

        usage = CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2, cost=0.0123)
        response = SimpleNamespace(usage=usage)
        state: dict[str, Any] = {}
        record_cost(state, response)
        record_cost(state, response)
        assert state["_perf"]["cost_usd"] == pytest.approx(0.0246)
        record_cost(None, response)

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

    async def test_system_prompt_reaches_the_model(self) -> None:
        sample = _sample()
        baseline = ScriptedClient()
        await _rollout(baseline, skills=True, sample=sample)
        ours = ScriptedClient()
        await _rollout(ours, skills=True, sample=sample, system_prompt="READ YOUR SKILLS FIRST")
        base_system = baseline.calls[0]["prompt"][0]
        system = ours.calls[0]["prompt"][0]
        assert base_system.role == system.role == "system"
        clock = task_clock(sample.info)
        line = f"Current date and time: {clock}" if clock else None
        expected_base = f"{sample.prompt[0]['content']}\n\n{line}" if line else sample.prompt[0]["content"]
        expected_ours = f"READ YOUR SKILLS FIRST\n\n{line}" if line else "READ YOUR SKILLS FIRST"
        assert base_system.content == expected_base
        assert system.content == expected_ours
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

    async def test_task_timeout_scores_the_work_done_so_far(self, monkeypatch: Any) -> None:
        import asyncio

        class StallingClient(ScriptedClient):
            """Acts once, then hangs like a stuck API request."""

            async def get_native_response(self, *args: Any, **kwargs: Any) -> Any:
                response = await super().get_native_response(*args, **kwargs)
                if len(self.calls) > 1:
                    await asyncio.sleep(30)
                return response

        client = StallingClient(turns=[{"tool_calls": [{"name": "search_tools", "arguments": {"query": "email"}}]}])
        monkeypatch.setattr(runner_mod, "get_client", lambda model: client)
        result = await runner_mod.run_one_async(_sample(), skills_dir=None, timeout=0.5)
        # The env stops its loop and the rubric still grades the world, so the
        # rollout keeps its trajectory and whatever credit it earned.
        assert len(client.calls) == 2  # the second call is the one that hangs
        assert result.error is None
        assert result.end_state is not None
        assert [m["role"] for m in result.trajectory] == ["assistant"]
        assert 0.0 <= result.partial_credit <= 1.0

    async def test_result_records_latency_and_usage(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(runner_mod, "get_client", lambda model: ScriptedClient())
        result = await runner_mod.run_one_async(_sample(), skills_dir=None)
        assert isinstance(result.latency_s, float) and result.latency_s > 0
        assert result.cost_usd is None
        assert isinstance(result.usage, dict)
        assert isinstance(result.perf, dict)
        serialized = result.to_json()
        assert {"latency_s", "cost_usd", "usage", "perf"} <= serialized.keys()

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

    def test_beaker_single_model_spec_is_pinned(self) -> None:
        import sys

        sys.path.insert(0, str(RECIPE_ROOT / ".beaker"))
        from beaker_integration import default_model_spec

        # Independent of ModelSpec's defaults, so the CLI default can change freely.
        spec = default_model_spec()
        assert (spec.name, spec.resolved_api(), spec.sampling_args()) == (
            "claude-sonnet-5-5",
            "anthropic",
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}, "max_tokens": 64000},
        )

    async def test_beaker_no_model_uses_traced_anthropic_client(self, monkeypatch: Any) -> None:
        import sys
        from types import SimpleNamespace

        from automationbench.clients import StreamingAnthropicClient

        sys.path.insert(0, str(RECIPE_ROOT / ".beaker"))
        import beaker_integration
        from beaker_integration import _client_for, _TracedAnthropicClient

        monkeypatch.setattr(
            beaker_integration,
            "default_model_spec",
            lambda: ModelSpec(name="claude-sonnet-5-5", api="anthropic", reasoning_effort="max"),
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        client, model = _client_for(SimpleNamespace(model=None))
        try:
            assert isinstance(client, _TracedAnthropicClient)
            assert isinstance(client, StreamingAnthropicClient)
            assert model.name == "claude-sonnet-5-5"
        finally:
            await client.close()

    def test_cli_api_routing(self, monkeypatch: Any) -> None:
        from automationbench_skills.cli import _default_api

        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        assert _default_api("gpt-5.6-luna", None, "auto") == "responses"
        assert _default_api("gpt-5.6-luna", None, "chat_completions") == "chat_completions"
        assert _default_api("gpt-5.6-luna", "https://gw.example/v1", "auto") == "auto"
        assert _default_api("claude-opus-5-5", None, "auto") == "auto"
        assert _default_api("gemini-3-pro", None, "auto") == "auto"
        assert _default_api("openai/gpt-5.6-luna", None, "auto") == "auto"
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gw.example/v1")
        assert _default_api("gpt-5.6-luna", None, "auto") == "auto"

    def test_cli_reports_errors_and_fails_when_all_errored(self, capsys: Any) -> None:
        from automationbench_skills.cli import _report_errors
        from automationbench_skills.runner import RunResult

        def result(name: str, error: Any = None) -> RunResult:
            return RunResult(name, "crm", 0.0, 0.0, [], None, error=error)

        assert _report_errors([result("a"), result("b")]) == 0
        assert _report_errors([result("a"), result("b", {"error": "BadRequestError: 400"})]) == 0
        assert "1/2 task(s) errored" in capsys.readouterr().err
        assert _report_errors([result("a", "BadRequestError: 400"), result("b", "BadRequestError: 400")]) == 1

    def test_env_is_cached(self) -> None:
        assert get_env(skills=False) is get_env(skills=False)
        assert get_env(skills=True) is not get_env(skills=False)

    def test_limited_zapier_with_skills_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            get_env(toolset="limited_zapier", skills=True)

    async def test_tool_executions_become_tool_spans(self, tmp_path: Path) -> None:
        from beaker.tracing import local_capture
        from beaker.tracing.integrations import verifiers as beaker_verifiers
        from beaker.tracing.projection import parse_jsonl, project

        env = get_env(skills=True)
        assert beaker_verifiers.is_instrumented(env)
        client = ScriptedClient(
            turns=[
                {"tool_calls": [{"name": "list_skills"}]},
                {"tool_calls": [{"name": "search_tools", "arguments": {"query": "send email", "top_k": 1}}]},
                {"content": "done"},
            ]
        )
        with local_capture(tmp_path, case_id="case", candidate_id="cand", strict_evidence=False) as capture:
            await _rollout(client, skills=True)
        assert capture.receipt is not None
        projection = capture.receipt.to_dict()["projection"]
        assert projection["tool_counts"] == {"list_skills": 1, "search_tools": 1}
        by_name = {call["tool_name"]: call for call in projection["tool_calls"]}
        assert by_name["search_tools"]["args"] == {"query": "send email", "top_k": 1}
        assert by_name["search_tools"]["tool_call_id"]
        assert "world" not in by_name["search_tools"]["args"]
        assert by_name["list_skills"]["return_content"]
        # a second rollout on the same cached env is traced independently
        with local_capture(tmp_path / "second", case_id="case", candidate_id="cand", strict_evidence=False) as again:
            await _rollout(ScriptedClient(), skills=True)
        assert again.receipt is not None
        assert again.receipt.to_dict()["projection"]["tool_counts"] == {}
        captures = list((tmp_path / "captures").glob("*.otlp.jsonl"))
        assert len(captures) == 1
        assert project(parse_jsonl(captures[0].read_bytes()), artifacts=()).tool_counts == projection["tool_counts"]

    async def test_traced_anthropic_client_records_model_call(self, tmp_path: Path, monkeypatch: Any) -> None:
        import sys

        from anthropic import AsyncAnthropic
        from anthropic.types import Message, TextBlock, Usage
        from automationbench.clients import StreamingAnthropicClient
        from beaker.tracing import local_capture

        sys.path.insert(0, str(RECIPE_ROOT / ".beaker"))
        from beaker_integration import _TracedAnthropicClient

        async def fake_get_native_response(
            self: Any, prompt: Any, model: str, sampling_args: Any, tools: Any = None, **kwargs: Any
        ) -> Message:
            del self, prompt, model, sampling_args, tools, kwargs
            return Message(
                id="msg_test",
                type="message",
                role="assistant",
                model="claude-sonnet-5-5",
                content=[TextBlock(type="text", text="ok")],
                stop_reason="end_turn",
                stop_sequence=None,
                usage=Usage(input_tokens=11, output_tokens=7),
            )

        monkeypatch.setattr(StreamingAnthropicClient, "get_native_response", fake_get_native_response)
        client = _TracedAnthropicClient(AsyncAnthropic(api_key="test-key"))
        try:
            with local_capture(tmp_path, case_id="case", candidate_id="cand", strict_evidence=False) as capture:
                await client.get_native_response(
                    prompt=[{"role": "user", "content": "hi"}],
                    model="claude-sonnet-5-5",
                    sampling_args={},
                    system="SYS",
                )
        finally:
            await client.close()
        assert capture.receipt is not None
        projection = capture.receipt.to_dict()["projection"]
        assert len(projection["model_calls"]) == 1
        call = projection["model_calls"][0]
        assert call["provider"] == "anthropic"
        assert call["model"] == "claude-sonnet-5-5"
        assert call["usage"]["input_tokens"] == 11
        assert call["usage"]["output_tokens"] == 7
        assert call["usage"]["total_tokens"] == 18
        assert '"role": "system"' in json.dumps(call["messages"])
        assert "SYS" in json.dumps(call["messages"])

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
            {
                "domain": "sales",
                "task_completed_correctly": 1.0,
                "partial_credit": 1.0,
                "latency_s": 2.0,
                "cost_usd": 0.0123,
            },
            {
                "domain": "sales",
                "task_completed_correctly": 0.0,
                "partial_credit": 0.5,
                "latency_s": 4.0,
                "cost_usd": 0.0456,
            },
            {"domain": "hr", "task_completed_correctly": 0.0, "partial_credit": 0.0},
        ]
        summary = summarize(rows)
        assert summary["domains"]["sales"] == {
            "tasks": 2,
            "pass_rate": 0.5,
            "partial_credit": 0.75,
            "avg_latency_s": 3.0,
            "avg_cost_usd": 0.02895,
        }
        assert summary["domains"]["hr"]["avg_latency_s"] is None
        assert summary["domains"]["hr"]["avg_cost_usd"] is None
        assert summary["overall"]["tasks"] == 3
        text = format_summary(summary)
        assert "overall" in text and "sales" in text
        assert "3.0" in text and "0.0290" in text and "-" in text
