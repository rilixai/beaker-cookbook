"""Tests for the faithful APEX-Agents ReAct toolbelt agent.

Drives the loop with an injected scripted ``_FakeChatModel`` and the
package's :class:`FakeWorld` — no HF, no real LLM calls.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from apex_agents.agent.agent import ApexReActAgent
from apex_agents.agent.prompts import load_apex_agents_prompts
from apex_agents.data.dataset import ApexAgentsRecord, RubricCriterion
from tests.fake_world import FakeWorld


def _record() -> ApexAgentsRecord:
    return ApexAgentsRecord(
        task_id="ib-1",
        task_name="Valuation memo",
        domain="Investment Banking",
        prompt="Read the brief and state the enterprise value.",
        world_id="world-a",
        rubric=(RubricCriterion("output_llm", "States an enterprise value."),),
        task_input_files=(),
        raw_task={"task_id": "ib-1"},
    )


class _ScriptedModel:
    """A chat model whose ``complete`` returns canned responses in order.

    Each scripted entry is the dict the agent's loop expects:
    ``{"content": str, "tool_calls": [{"id","name","arguments"}],
    "cost": float}``. ``seen`` records the messages list passed on each
    call so a test can assert what the LLM actually saw.
    """

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self._script = script
        self._idx = 0
        self.seen: list[list[dict[str, Any]]] = []

    def complete(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        self.seen.append([dict(m) for m in messages])
        if self._idx < len(self._script):
            out = self._script[self._idx]
        else:
            out = {"content": "no-op", "tool_calls": [], "cost": 0.0}
        self._idx += 1
        return out


def _tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> dict[str, Any]:
    return {"id": call_id, "name": name, "arguments": args}


def _build_agent(*, model: Any, world: FakeWorld, max_steps: int = 20) -> ApexReActAgent:
    sys_p, task_t, resum_p = load_apex_agents_prompts()
    return ApexReActAgent(
        model_name="scripted/test",
        model_temperature=0.0,
        max_steps=max_steps,
        cost_limit=100.0,
        system_prompt=sys_p,
        task_template=task_t,
        resum_summary_prompt=resum_p,
        world_factory=lambda _record: world,
        model_factory=lambda _name, _temp: model,
    )


def test_agent_runs_end_to_end_add_tool_read_final_answer() -> None:
    """Scripted model: add read_file -> read it -> final_answer."""
    world = FakeWorld({"brief.txt": "Enterprise value: $42M."})
    model = _ScriptedModel(
        [
            {
                "content": "Adding read_file.",
                "tool_calls": [_tool_call("toolbelt_add_tool", {"name": "read_file"})],
                "cost": 0.01,
            },
            {
                "content": "Reading the brief.",
                "tool_calls": [_tool_call("read_file", {"path": "brief.txt"})],
                "cost": 0.01,
            },
            {
                "content": "Submitting.",
                "tool_calls": [
                    _tool_call("final_answer", {"answer": "The enterprise value is $42M.", "status": "completed"})
                ],
                "cost": 0.01,
            },
        ]
    )
    agent = _build_agent(model=model, world=world)
    out = asyncio.run(agent.forward(record=_record()))

    assert out.status == "completed"
    assert out.final_answer == "The enterprise value is $42M."
    assert out.total_steps == 3
    assert out.total_cost == 0.03
    tool_names = [m.tool_name for m in out.messages if m.role == "assistant" and m.tool_name]
    assert tool_names == ["toolbelt_add_tool", "read_file", "final_answer"]
    # The world's content was returned to the model as a tool message.
    tool_outputs = [m.output for m in out.messages if m.role == "tool"]
    assert any("Enterprise value: $42M." in (o or "") for o in tool_outputs)


def test_domain_tool_rejected_until_added_to_toolbelt() -> None:
    """The toolbelt starts EMPTY — read_file must be added before use."""
    world = FakeWorld({"f.txt": "data"})
    model = _ScriptedModel(
        [
            {
                "content": "Try reading without adding.",
                "tool_calls": [_tool_call("read_file", {"path": "f.txt"})],
                "cost": 0.0,
            },
            {
                "content": "Give up.",
                "tool_calls": [_tool_call("final_answer", {"answer": "done", "status": "completed"})],
                "cost": 0.0,
            },
        ]
    )
    agent = _build_agent(model=model, world=world)
    out = asyncio.run(agent.forward(record=_record()))
    tool_outputs = [m.output for m in out.messages if m.role == "tool"]
    assert any("not in the active toolbelt" in (o or "") for o in tool_outputs)


def test_final_answer_rejected_with_open_todos() -> None:
    """final_answer is rejected while a todo is still open (Archipelago-faithful)."""
    world = FakeWorld({})
    model = _ScriptedModel(
        [
            {
                "content": "Plan.",
                "tool_calls": [
                    _tool_call("todo_write", {"todos": [{"id": "1", "content": "do it", "status": "in_progress"}]})
                ],
                "cost": 0.0,
            },
            {
                "content": "Try to finish early.",
                "tool_calls": [_tool_call("final_answer", {"answer": "premature", "status": "completed"})],
                "cost": 0.0,
            },
            {
                "content": "Close the todo.",
                "tool_calls": [
                    _tool_call("todo_write", {"todos": [{"id": "1", "content": "do it", "status": "completed"}]})
                ],
                "cost": 0.0,
            },
            {
                "content": "Now finish.",
                "tool_calls": [_tool_call("final_answer", {"answer": "real answer", "status": "completed"})],
                "cost": 0.0,
            },
        ]
    )
    agent = _build_agent(model=model, world=world)
    out = asyncio.run(agent.forward(record=_record()))
    assert out.final_answer == "real answer"
    tool_outputs = [m.output for m in out.messages if m.role == "tool"]
    assert any("final_answer rejected" in (o or "") for o in tool_outputs)


def test_custom_prompts_reach_the_loop() -> None:
    """Prompts passed to the constructor are what the LLM actually sees."""
    world = FakeWorld({})
    model = _ScriptedModel(
        [
            {
                "content": "done",
                "tool_calls": [_tool_call("final_answer", {"answer": "x", "status": "completed"})],
                "cost": 0.0,
            }
        ]
    )
    custom_system = "CUSTOM_SYSTEM_PROMPT_42"
    custom_task = "CUSTOM_TASK_FRAMING_99 :: {{task}}"
    custom_resum = "CUSTOM_RESUM {conversation}"
    agent = ApexReActAgent(
        model_name="scripted/test",
        system_prompt=custom_system,
        task_template=custom_task,
        resum_summary_prompt=custom_resum,
        world_factory=lambda _record: world,
        model_factory=lambda _n, _t: model,
    )
    assert agent.system_prompt == custom_system
    assert agent.task_template == custom_task
    assert agent.resum_summary_prompt == custom_resum

    asyncio.run(agent.forward(record=_record()))
    seen_system = model.seen[-1][0]["content"]
    seen_user = model.seen[-1][1]["content"]
    assert custom_system in seen_system
    # The {{task}} substitution happened with the custom template.
    assert "CUSTOM_TASK_FRAMING_99" in seen_user
    assert "Read the brief" in seen_user


def test_task_template_substitutes_task_variable() -> None:
    world = FakeWorld({})
    model = _ScriptedModel(
        [{"content": "done", "tool_calls": [_tool_call("final_answer", {"answer": "x"})], "cost": 0.0}]
    )
    agent = _build_agent(model=model, world=world)
    asyncio.run(agent.forward(record=_record()))
    user_msg = model.seen[0][1]["content"]
    # Seed task_template is "{{task}}" → user message == raw task prompt.
    assert user_msg == "Read the brief and state the enterprise value."


def test_resum_triggers_and_keeps_recent_messages() -> None:
    """ReSum fires when the context grows past the trigger and keeps last N."""
    world = FakeWorld({"f.txt": "x"})
    # A model that keeps adding/removing a tool to grow the message
    # history, then finishes. Each turn emits a large content blob so
    # the token estimate crosses the (tiny) configured budget fast.
    big = "A" * 4000
    script: list[dict[str, Any]] = []
    for i in range(14):
        script.append(
            {
                "content": big + f" turn {i}",
                "tool_calls": [_tool_call("toolbelt_list_tools", {}, call_id=f"c{i}")],
                "cost": 0.0,
            }
        )
    script.append({"content": "finish", "tool_calls": [_tool_call("final_answer", {"answer": "done"})], "cost": 0.0})
    model = _ScriptedModel(script)
    sys_p, task_t, resum_p = load_apex_agents_prompts()
    agent = ApexReActAgent(
        model_name="scripted/test",
        model_temperature=0.0,
        max_steps=30,
        cost_limit=100.0,
        max_context_tokens=2_000,  # tiny so ReSum trigger fires quickly
        system_prompt=sys_p,
        task_template=task_t,
        resum_summary_prompt=resum_p,
        world_factory=lambda _r: world,
        model_factory=lambda _n, _t: model,
    )
    out = asyncio.run(agent.forward(record=_record()))

    assert out.resum_count >= 1, "ReSum never fired despite a tiny context budget"
    # After a compaction the message list begins with the system
    # message followed by the compacted-state user message.
    final_seen = model.seen[-1]
    assert final_seen[0]["role"] == "system"
    assert any("[Compacted reasoning state from earlier in the session]" in str(m.get("content")) for m in final_seen)
    assert out.final_answer == "done"


def test_model_failure_surfaces_as_error_status() -> None:
    world = FakeWorld({})

    class _BrokenModel:
        def complete(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            raise RuntimeError("simulated outage")

    agent = _build_agent(model=_BrokenModel(), world=world)
    out = asyncio.run(agent.forward(record=_record()))
    assert out.status == "RuntimeError"
    assert out.final_answer == ""
    assert "simulated outage" in str(out.extra.get("error", ""))


def test_world_factory_runs_off_the_event_loop() -> None:
    """Regression: the world factory may call asyncio.run (HF zip extraction).

    Routing world construction through asyncio.to_thread means a
    factory that itself calls asyncio.run does not raise
    'asyncio.run() cannot be called from a running event loop'.
    """
    import threading

    constructed_thread_ids: list[int] = []

    def _factory(_record: Any) -> FakeWorld:
        constructed_thread_ids.append(threading.get_ident())

        async def _noop() -> None:
            return None

        asyncio.run(_noop())  # would raise if not offloaded
        return FakeWorld({})

    sys_p, task_t, resum_p = load_apex_agents_prompts()
    model = _ScriptedModel(
        [{"content": "done", "tool_calls": [_tool_call("final_answer", {"answer": "ok"})], "cost": 0.0}]
    )
    agent = ApexReActAgent(
        model_name="scripted/test",
        system_prompt=sys_p,
        task_template=task_t,
        resum_summary_prompt=resum_p,
        world_factory=_factory,
        model_factory=lambda _n, _t: model,
    )
    out = asyncio.run(agent.forward(record=_record()))
    assert constructed_thread_ids, "world factory was never invoked"
    assert out.final_answer == "ok"
    main_thread_id = threading.main_thread().ident
    assert constructed_thread_ids[0] != main_thread_id, (
        "world factory ran on the main thread; asyncio.to_thread offload regressed"
    )


def test_to_openai_api_messages_reshapes_replayed_tool_calls() -> None:
    """Regression: flat internal tool_calls must be reshaped for the API.

    The loop stores assistant tool calls flat
    (``{"id","name","arguments"}``). Replaying that verbatim to the
    OpenAI / litellm chat API makes call #2 fail with BadRequestError —
    which previously killed every multi-step task right after its first
    tool round-trip (todo_write). The API requires the nested
    ``{"id","type":"function","function":{"name","arguments"}}`` shape
    and null (not "") assistant content alongside tool calls.
    """
    from apex_agents.agent.agent import _to_openai_api_messages

    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "name": "todo_write", "arguments": {"todos": [], "merge": False}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "todo_write", "content": "ok"},
        {"role": "assistant", "content": "thinking", "tool_calls": []},
    ]

    out = _to_openai_api_messages(history)

    # Plain turns: tool_calls key dropped, content preserved.
    assert out[0] == {"role": "system", "content": "sys"}
    assert "tool_calls" not in out[1]
    assert "tool_calls" not in out[4] and out[4]["content"] == "thinking"

    # Assistant tool call reshaped to the nested OpenAI schema.
    tc = out[2]["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["id"] == "call_1"
    assert tc["function"]["name"] == "todo_write"
    # arguments must be a JSON *string*, not a dict.
    assert isinstance(tc["function"]["arguments"], str)
    assert json.loads(tc["function"]["arguments"]) == {"todos": [], "merge": False}
    # content must be null (not "") next to tool_calls.
    assert out[2]["content"] is None
    # The malformed flat shape must NOT survive into the API payload.
    assert "name" not in tc and "arguments" not in tc
    # Tool result message passes through untouched.
    assert out[3]["role"] == "tool" and out[3]["tool_call_id"] == "call_1"


def test_todo_status_synonym_closes_the_final_answer_gate() -> None:
    """Regression: a todo marked 'done' must satisfy the final_answer gate.

    The gate originally accepted only {'completed','cancelled'}; models
    naturally emit 'done', so final_answer was rejected forever and the
    agent livelocked, burning its whole step budget. The IB/Law skyline
    numbers were measured under this bug.
    """
    world = FakeWorld({})
    model = _ScriptedModel(
        [
            {
                "content": "",
                "tool_calls": [
                    _tool_call(
                        "todo_write", {"todos": [{"id": "1", "content": "x", "status": "pending"}], "merge": False}
                    )
                ],
                "cost": 0.0,
            },
            {
                "content": "",
                "tool_calls": [
                    _tool_call(
                        "todo_write", {"todos": [{"id": "1", "content": "x", "status": "done"}], "merge": False}
                    )
                ],
                "cost": 0.0,
            },
            {
                "content": "",
                "tool_calls": [_tool_call("final_answer", {"answer": "EV is $42", "status": "completed"})],
                "cost": 0.0,
            },
        ]
    )
    agent = _build_agent(model=model, world=world, max_steps=10)
    out = asyncio.run(agent.forward(record=_record()))
    assert out.status == "completed"
    assert out.final_answer == "EV is $42"
    assert out.extra.get("forced_final_answer") is not True  # closed cleanly, not force-accepted


def test_final_answer_livelock_guard_terminates_with_answer() -> None:
    """A willing agent that can't satisfy the todo gate must still terminate.

    After _MAX_FINAL_ANSWER_REJECTIONS the answer is accepted (flagged
    forced) instead of the agent burning its step budget — a harness
    defect, not faithful Archipelago behavior.
    """
    from apex_agents.agent.agent import _MAX_FINAL_ANSWER_REJECTIONS

    world = FakeWorld({})
    open_todo = {"todos": [{"id": "1", "content": "never closes", "status": "pending"}], "merge": False}
    script = [{"content": "", "tool_calls": [_tool_call("todo_write", open_todo)], "cost": 0.0}]
    # Keep trying final_answer forever; the guard must cut it off.
    script += [
        {
            "content": "",
            "tool_calls": [_tool_call("final_answer", {"answer": "forced answer", "status": "completed"})],
            "cost": 0.0,
        }
        for _ in range(_MAX_FINAL_ANSWER_REJECTIONS + 3)
    ]
    model = _ScriptedModel(script)
    agent = _build_agent(model=model, world=world, max_steps=40)
    out = asyncio.run(agent.forward(record=_record()))
    assert out.final_answer == "forced answer"
    assert out.extra.get("forced_final_answer") is True
    assert out.status != "max_steps"  # did NOT burn the whole budget livelocking


def test_todo_write_merge_does_not_collapse_empty_ids() -> None:
    """Regression: empty-id todos must stay distinct on merge.

    Keying merge purely on id collapsed every ``id=''`` todo into one
    entry, so the agent could never address individual items to close
    them — half of the livelock.
    """
    from apex_agents.agent.agent import ApexReActAgent

    todos: list[dict[str, Any]] = []
    ApexReActAgent._tool_todo_write(
        args={
            "todos": [
                {"id": "", "content": "alpha", "status": "pending"},
                {"id": "", "content": "beta", "status": "pending"},
            ],
            "merge": False,
        },
        todos=todos,
    )
    assert len(todos) == 2
    # Merge an update to just 'beta' by content — must not collapse.
    ApexReActAgent._tool_todo_write(
        args={"todos": [{"id": "", "content": "beta", "status": "done"}], "merge": True},
        todos=todos,
    )
    assert len(todos) == 2
    by_content = {t["content"]: t["status"] for t in todos}
    assert by_content == {"alpha": "pending", "beta": "done"}


# ─── Fix 4: per-call LLM timeout + bounded retry ──────────────────────


def test_litellm_model_passes_timeout_and_retries(monkeypatch: Any) -> None:
    """The production model wrapper must bound every call (no infinite hang)."""
    import apex_agents.agent.agent as agentmod

    captured: dict[str, Any] = {}

    class _Msg:
        content = "ok"
        tool_calls: list[Any] = []

    class _Resp:
        choices = [type("C", (), {"message": _Msg()})()]

    fake_litellm = type(
        "L",
        (),
        {
            "completion": staticmethod(lambda **kw: (captured.update(kw), _Resp())[1]),
            "completion_cost": staticmethod(lambda **kw: 0.0),
        },
    )
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake_litellm)

    factory = agentmod.build_litellm_model_factory(timeout=7.0, num_retries=3)
    model = factory("openai/gpt-4.1-mini", 0.0)
    model.complete(messages=[{"role": "user", "content": "hi"}], tools=[])
    assert captured["timeout"] == 7.0
    assert captured["num_retries"] == 3


def test_timeout_exception_fails_case_fast_not_hang() -> None:
    """A timed-out LLM call must terminate the case (status set), not wedge."""
    world = FakeWorld({})

    class _TimeoutModel:
        def complete(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            raise TimeoutError("litellm request timed out")

    agent = _build_agent(model=_TimeoutModel(), world=world)
    out = asyncio.run(agent.forward(record=_record()))
    assert out.status == "TimeoutError"
    assert "timed out" in str(out.extra.get("error", ""))
