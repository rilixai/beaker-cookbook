"""Runtime configuration for the HotpotQA agent benchmark.

``HotpotQAConfig`` is the single knob bag the runner reads. Lives at the top
level so ``rilixai_spec.py`` (and the agent / data subpackages) can import it
without a cycle.

(Renamed from ``HotpotQAPipelineConfig`` post-workflow-deletion — the
"Pipeline" qualifier was disambiguating against the deleted DSPy
workflow; with one runtime there's no ambiguity left.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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
        if self.retrieve_k < 1:
            raise ValueError(f"HotpotQAConfig.retrieve_k must be >= 1, got {self.retrieve_k}.")
        if self.max_iters < 1:
            raise ValueError(f"HotpotQAConfig.max_iters must be >= 1, got {self.max_iters}.")
        if self.retrieval_mode not in ("distractor", "fullwiki"):
            raise ValueError(
                f"HotpotQAConfig.retrieval_mode must be one of 'distractor' / 'fullwiki', got {self.retrieval_mode!r}."
            )
