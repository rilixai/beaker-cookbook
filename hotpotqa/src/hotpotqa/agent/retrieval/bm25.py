"""BM25 retrieval over the per-case HotpotQA distractor paragraphs.

Used by the PydanticAI agent's ``retrieve_k`` tool. HotpotQA distractor
exposes only ~10 paragraphs per case, so the BM25 index is rebuilt per
call rather than precomputed. The functions are intentionally
framework-agnostic — PydanticAI isn't imported here.

The ``build_retrieve_k_fn_for_case`` helper also dispatches between
distractor (per-case BM25) and fullwiki (bm25s over the 2017 Wikipedia
abstracts dump) so the agent runtime can be mode-agnostic.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

from ...data.dataset import HotpotQAParagraph


if TYPE_CHECKING:
    from ...config import HotpotQAConfig
    from ...data.dataset import HotpotQARecord


# Signature shared by both retrieval modes. ``(query, k) -> list of paragraphs``.
RetrieveKFn = Callable[[str, int], list[HotpotQAParagraph]]


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer shared across BM25 calls."""
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def bm25_top_k(
    *,
    query: str,
    paragraphs: Sequence[HotpotQAParagraph],
    top_k: int,
    already_retrieved_titles: set[str],
) -> list[HotpotQAParagraph]:
    """Rank candidate paragraphs with BM25 and return the top-k unseen ones."""
    candidates = [p for p in paragraphs if p.title not in already_retrieved_titles]
    if not candidates or top_k <= 0:
        return []
    tokenized_corpus = [tokenize(p.text) for p in candidates]
    # rank_bm25 errors on a corpus with no tokens at all; fall back to taking
    # the head of the candidate list rather than crashing the whole pass.
    if not any(tokenized_corpus):
        return list(candidates[:top_k])
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = tokenize(query)
    if not query_tokens:
        return list(candidates[:top_k])
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [paragraph for paragraph, _score in ranked[:top_k]]


def build_retrieve_k_fn_for_case(
    *,
    record: "HotpotQARecord",  # noqa: F821 - forward ref to avoid runtime dataset import
    cfg: "HotpotQAConfig",  # noqa: F821 - forward ref to avoid circular import
) -> RetrieveKFn:
    """Return a ``RetrieveKFn`` scoped to this case's retrieval corpus.

    The two retrieval modes are *intentionally asymmetric* on cross-call
    deduplication:

    * **Distractor mode** closes over the case's ~10-paragraph context
      and a running ``already_retrieved`` set so subsequent calls can
      never re-surface earlier hits. Without this, the small corpus
      would cause the BM25 ranking to trivially return the same top-k
      again, wasting every later slot on duplicates. The dedup is a
      small-corpus pragmatism — the open-domain reference pipelines
      don't have a distractor mode; we added this for cheap iteration
      and smoke tests.

    * **Fullwiki mode** does NOT dedupe across calls. It delegates to
      ``fullwiki_retrieve_k_fn`` which dedupes only *within* a single
      retrieve call. This matches ``dspy.Retrieve`` — the retrieval
      primitive the reference multi-hop HotpotQA programs call for hop
      2 — which doesn't filter earlier hits out of subsequent results.
      Reproducing this behavior keeps the agent's retrieval semantics
      comparable to published HotpotQA numbers.
    """
    if cfg.retrieval_mode == "distractor":
        already_retrieved: set[str] = set()

        def _retrieve(query: str, k: int) -> list[HotpotQAParagraph]:
            hits = bm25_top_k(
                query=query,
                paragraphs=record.paragraphs,
                top_k=k,
                already_retrieved_titles=already_retrieved,
            )
            for paragraph in hits:
                already_retrieved.add(paragraph.title)
            return hits

        return _retrieve

    if cfg.retrieval_mode == "fullwiki":
        from .fullwiki import fullwiki_retrieve_k_fn

        return fullwiki_retrieve_k_fn(k_default=cfg.retrieve_k)

    raise ValueError(f"Unsupported retrieval_mode: {cfg.retrieval_mode!r}")  # unreachable per __post_init__
