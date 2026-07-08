"""Runtime configuration for the HotpotQA agent benchmark.

``HotpotQAConfig`` is the single knob bag the CLI, the runtime adapter,
and the spec all share. Lives at the top level (as a peer of ``cli.py``)
because it's consumed by both the CLI and the optimization runtime —
keeping it here avoids a cycle between ``cli.py`` and
``optimization/runtime.py``.

(Renamed from ``HotpotQAPipelineConfig`` post-workflow-deletion — the
"Pipeline" qualifier was disambiguating against the deleted DSPy
workflow; with one runtime there's no ambiguity left.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


def to_pydantic_ai_model(spec: str) -> str:
    """Canonicalize a model spec to PydanticAI's ``provider:model`` form.

    PydanticAI model specs use a colon separator (``"openai:gpt-4.1-mini"``),
    while the ``--task-model`` CLI flag and the litellm ecosystem use a slash
    (``"openai/gpt-4.1-mini"``). Historically the slash→colon rewrite lived in
    the CLI only, so a slash-form model reaching the runtime via sandbox config
    or a direct spec build produced an invalid PydanticAI spec. Normalizing here
    — the one layer every path funnels through (``HotpotQAConfig``) — makes the
    rewrite source-independent. Only the first separator is rewritten so model
    names that legitimately contain a slash are preserved.
    """
    return spec.replace("/", ":", 1)


def bare_openai_model(spec: str) -> str:
    """Strip the provider prefix from a model spec.

    PydanticAI uses ``"openai:gpt-4.1-mini"``; the raw OpenAI
    ``chat.completions.create`` call inside the summarize tool wants the bare
    ``"gpt-4.1-mini"``. Accepts either separator and returns the original string
    unchanged when no provider prefix is present.
    """
    for separator in (":", "/"):
        _, found, model = spec.partition(separator)
        if found:
            return model
    return spec


# ``RetrievalMode`` is the config-option type, so it lives next to the
# config dataclass — not under ``agent/retrieval/``. Keeps ``config.py``
# from having to reach across into ``agent/`` for a one-line literal.
RetrievalMode = Literal["distractor", "fullwiki"]


@dataclass(frozen=True)
class HotpotQAConfig:
    """Knobs for the HotpotQA agent runtime.

    ``retrieval_mode`` chooses the retrieval corpus the agent sees.
    ``fullwiki`` mirrors the GEPA paper's setup (bm25s over the 2017
    Wikipedia abstracts); ``distractor`` is the HF
    ``hotpot_qa[distractor]`` 10-paragraph-per-case shape, useful as a
    test-friendly opt-out.

    ``retrieve_k`` is the number of paragraphs returned per retrieval
    call. The paper uses ``k=7``.

    ``max_iters`` caps the agent-loop length;
    ``pydantic_agent_model`` is the PydanticAI model spec when no
    pre-built agent is supplied.
    """

    retrieval_mode: RetrievalMode = "fullwiki"
    retrieve_k: int = 7
    max_paragraph_chars: int = 1_500
    max_iters: int = 8
    pydantic_agent_model: str | None = None
    # Pinned at ``0.0`` for reproducibility — used by the PydanticAI
    # agent's outer model settings and its raw summarize-tool call.
    pydantic_agent_temperature: float = 0.0

    def __post_init__(self) -> None:
        # Canonicalize the model spec to PydanticAI colon form regardless of
        # source (CLI slash form, sandbox config, or a direct build) so the
        # runtime never sees an invalid slash-form PydanticAI spec. ``frozen``
        # dataclass → assign through ``object.__setattr__``.
        if self.pydantic_agent_model is not None:
            object.__setattr__(self, "pydantic_agent_model", to_pydantic_ai_model(self.pydantic_agent_model))
        if self.retrieve_k < 1:
            raise ValueError(f"HotpotQAConfig.retrieve_k must be >= 1, got {self.retrieve_k}.")
        if self.max_iters < 1:
            raise ValueError(f"HotpotQAConfig.max_iters must be >= 1, got {self.max_iters}.")
        if self.retrieval_mode not in ("distractor", "fullwiki"):
            raise ValueError(
                f"HotpotQAConfig.retrieval_mode must be one of 'distractor' / 'fullwiki', got {self.retrieval_mode!r}."
            )
