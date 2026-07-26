"""BM25 retrieval — the agent's ``retrieve_k`` tool implementation.

Two BM25 variants share the same ``RetrieveKFn`` signature:

* :mod:`.bm25` — per-case BM25 over the distractor 10-paragraph context
  (plus the mode-dispatching ``build_retrieve_k_fn_for_case`` factory
  that injects a per-case retriever into the agent).
* :mod:`.fullwiki` — bm25s over the 2017 Wikipedia abstracts dump (the
  open-domain setting). Imported lazily by the dispatcher only when
  ``retrieval_mode="fullwiki"`` is selected, so distractor-mode tests
  never pay the ``bm25s`` startup cost.

Lives under ``agent/`` because retrieval is the agent's tool surface —
``retrieve_k(query)`` is one of two tools the agent calls. The
evaluation consumes ``build_retrieve_k_fn_for_case`` from here to
parameterize the agent's per-case retriever.

``RetrievalMode`` (the ``Literal["distractor", "fullwiki"]`` type) lives
in :mod:`hotpotqa.config` next to the config dataclass that uses it,
so this package doesn't have to re-export it.
"""

from __future__ import annotations

from .bm25 import RetrieveKFn, bm25_top_k, build_retrieve_k_fn_for_case, tokenize


__all__ = [
    "RetrieveKFn",
    "bm25_top_k",
    "build_retrieve_k_fn_for_case",
    "tokenize",
]
