"""The Harvey LAB legal agent, driven through the Stirrup harness.

Stirrup (Artificial Analysis' agent framework) supplies the tool-use loop,
context management, and message plumbing; this module supplies the
*domain*: a per-task :class:`~harvey_lab.agent.workspace.TaskWorkspace` and
the file tools a legal knowledge worker uses over it (read documents,
search, write / edit deliverables). The two optimizable prompt components
(``system_prompt`` + ``task_template``) are the GEPA targets.

Why Stirrup rather than a hand-rolled ReAct loop (as ``apex_agents`` uses)?
It is the harness Artificial Analysis' Harvey LAB-AA leaderboard runs on,
so building the recipe on it keeps the agent's control flow aligned with
the benchmark's published numbers. The LLM client is pluggable: production
uses Stirrup's LiteLLM client (so the same ``provider/model`` strings as the
apex recipe work); tests inject a scripted client and never hit the network.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, Field

from ..config import HarveyLabConfig
from .prompts import (
    SYSTEM_PROMPT_COMPONENT,
    TASK_TEMPLATE_COMPONENT,
    load_harvey_lab_seed_prompts,
)
from .types import HarveyLabAgentOutput
from .workspace import TaskSource, TaskWorkspace


# Factory that builds a Stirrup ``LLMClient`` from ``(model, temperature,
# max_tokens)``. Kept behind a factory so tests inject a scripted client and
# the real (litellm-backed) client is imported lazily.
ModelFactory = Callable[[str, float, int], Any]


def _default_model_factory(model: str, temperature: float, max_tokens: int) -> Any:
    """Build Stirrup's LiteLLM client (imported lazily; needs ``stirrup[litellm]``)."""
    from stirrup.clients.litellm_client import LiteLLMClient

    return LiteLLMClient(model=model, max_tokens=max_tokens, kwargs={"temperature": temperature})


def _render_task_template(task_template: str, *, instructions: str, deliverables: str) -> str:
    """Substitute the ``{{instructions}}`` + ``{{deliverables}}`` Jinja2 vars.

    A candidate may reframe the template but both variables must survive; if a
    candidate drops one, the raw value is appended so the agent still receives
    it (mirrors the apex ``{{task}}`` fallback).
    """
    rendered = task_template
    for name, value in (("instructions", instructions), ("deliverables", deliverables)):
        placeholder = "{{%s}}" % name
        spaced = "{{ %s }}" % name
        if placeholder in rendered:
            rendered = rendered.replace(placeholder, value)
        elif spaced in rendered:
            rendered = rendered.replace(spaced, value)
        else:
            rendered = f"{rendered}\n\n{value}"
    return rendered


# ─── tool parameter models ────────────────────────────────────────────


class _ReadParams(BaseModel):
    path: str = Field(description="Path of the document to read, e.g. 'documents/contract.docx'.")


class _GrepParams(BaseModel):
    query: str = Field(description="Case-insensitive substring to search for across the documents.")


class _WriteParams(BaseModel):
    path: str = Field(description="Output filename to write, e.g. 'memo.docx' (rooted in output/).")
    content: str = Field(description="Full text content of the deliverable.")


class _EditParams(BaseModel):
    path: str = Field(description="Output filename to edit (rooted in output/).")
    old: str = Field(description="Exact snippet to replace.")
    new: str = Field(description="Replacement text.")


class _FinishParams(BaseModel):
    reason: str = Field(description="Why the task is complete (all requested deliverables written).")


def _build_finish_tool() -> Any:
    """A ``reason``-only finish tool.

    Stirrup's default ``SIMPLE_FINISH_TOOL`` requires a ``paths`` list it
    validates against a code-exec env; this recipe manages the ``output/``
    tree itself (deliverables are collected off disk), so a lighter finish
    tool that just records the agent's closing reason is a better fit.
    """
    from stirrup import Tool, ToolResult

    return Tool(
        name="finish",
        description="Signal the task is complete once every requested deliverable exists in output/.",
        parameters=_FinishParams,
        executor=lambda params: ToolResult(content=params.reason),
    )


def _build_workspace_tools(workspace: TaskWorkspace) -> list[Any]:
    """Build the Stirrup file tools bound to ``workspace``."""
    from stirrup import Tool, ToolResult

    def _list(_: BaseModel) -> Any:
        files = workspace.list_files()
        return ToolResult(content="\n".join(files) if files else "(workspace is empty)")

    def _read(params: _ReadParams) -> Any:
        try:
            return ToolResult(content=workspace.read_document(params.path))
        except FileNotFoundError:
            return ToolResult(content=f"No such document: {params.path!r}.", success=False)
        except Exception as exc:  # noqa: BLE001 - surface parse errors to the agent
            return ToolResult(content=f"Could not read {params.path!r}: {exc}", success=False)

    def _grep(params: _GrepParams) -> Any:
        hits = workspace.search_documents(params.query)
        if not hits:
            return ToolResult(content=f"No matches for {params.query!r}.")
        return ToolResult(content="\n".join(f"{h['file']}:{h['line']}: {h['text']}" for h in hits))

    def _write(params: _WriteParams) -> Any:
        rel = workspace.write_deliverable(params.path, params.content)
        return ToolResult(content=f"Wrote {rel} ({len(params.content)} chars).")

    def _edit(params: _EditParams) -> Any:
        try:
            rel = workspace.edit_deliverable(params.path, params.old, params.new)
            return ToolResult(content=f"Edited {rel}.")
        except (FileNotFoundError, ValueError) as exc:
            return ToolResult(content=str(exc), success=False)

    return [
        Tool(name="list_files", description="List the files in documents/ and output/.", executor=_list),
        Tool(
            name="read_document",
            description="Read a source document (.docx/.xlsx/.pdf/.eml/text) from documents/.",
            parameters=_ReadParams,
            executor=_read,
        ),
        Tool(
            name="grep_documents",
            description="Case-insensitive search across the task documents.",
            parameters=_GrepParams,
            executor=_grep,
        ),
        Tool(
            name="write_deliverable",
            description="Write the full text of an output deliverable (rooted in output/).",
            parameters=_WriteParams,
            executor=_write,
        ),
        Tool(
            name="edit_deliverable",
            description="Replace a snippet in a deliverable you already wrote.",
            parameters=_EditParams,
            executor=_edit,
        ),
    ]


class HarveyLabAgent:
    """A Stirrup-driven legal agent with a per-task file workspace.

    ``apply_candidate`` swaps the two optimizable prompt components; ``forward``
    materializes the task workspace, runs the Stirrup loop, and returns the
    produced deliverables for the rubric judge.
    """

    def __init__(
        self,
        *,
        config: HarveyLabConfig,
        task_source: TaskSource,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self._config = config
        self._task_source = task_source
        self._model_factory = model_factory or _default_model_factory
        seed_system, seed_task = load_harvey_lab_seed_prompts()
        self._system_prompt = seed_system
        self._task_template = seed_task
        # A single agent instance is reused across concurrent ``run_case``
        # calls on worker threads. Guard the stashed components so an
        # ``apply_candidate`` can't interleave with a ``forward`` snapshot and
        # mix prompts from two candidates (mirrors ``ApexReActAgent``).
        self._build_lock = threading.Lock()

    def apply_candidate(self, components: Mapping[str, str]) -> None:
        """Update the stashed prompt components from an optimizer candidate."""
        system = components.get(SYSTEM_PROMPT_COMPONENT)
        task = components.get(TASK_TEMPLATE_COMPONENT)
        with self._build_lock:
            if system is not None:
                self._system_prompt = system
            if task is not None:
                self._task_template = task

    def _snapshot_components(self) -> tuple[str, str]:
        with self._build_lock:
            return self._system_prompt, self._task_template

    @property
    def current_system_prompt(self) -> str:
        with self._build_lock:
            return self._system_prompt

    @property
    def current_task_template(self) -> str:
        with self._build_lock:
            return self._task_template

    async def forward(self, *, record: Any) -> HarveyLabAgentOutput:
        from stirrup import Agent

        # Snapshot the components under the build lock so a concurrent
        # ``apply_candidate`` can't swap them mid-run (mirrors the apex
        # agent's lock-guarded copy-then-run).
        system_prompt, task_template = self._snapshot_components()

        deliverable_lines = "\n".join(f"- `{name}`" for name in getattr(record, "deliverable_names", ()))
        user_prompt = _render_task_template(
            task_template,
            instructions=str(getattr(record, "instructions", "")),
            deliverables=deliverable_lines or "- `response.md`",
        )

        workspace = self._task_source(record)
        started = time.monotonic()
        try:
            client = self._model_factory(
                self._config.task_model,
                self._config.task_temperature,
                self._config.max_output_tokens,
            )
            agent: Any = Agent(
                client=client,
                name="harvey-lab",
                system_prompt=system_prompt,
                tools=_build_workspace_tools(workspace),
                finish_tool=_build_finish_tool(),
                max_turns=self._config.max_turns,
            )
            # cache_on_interrupt=False: the hosted optimizer runs cases in worker
            # threads, where Stirrup's default SIGINT handler raises
            # "signal only works in main thread of the main interpreter".
            async with agent.session(output_dir=workspace.output_dir, cache_on_interrupt=False) as session:
                finish_params, history, _metadata = await session.run(user_prompt)
            deliverables = workspace.collect_deliverables()
            return HarveyLabAgentOutput(
                final_answer=str(getattr(finish_params, "reason", "")),
                status="completed",
                deliverables=deliverables,
                total_turns=len(history),
                wall_seconds=time.monotonic() - started,
            )
        finally:
            workspace.close()


__all__ = ["HarveyLabAgent", "ModelFactory"]
