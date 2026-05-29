"""bm25s retrieval over the 2017 Wikipedia abstracts dump.

Mirrors the artifact's
[hover_program.py](https://github.com/gepa-ai/gepa-artifact/blob/cbefbc1aa0f43dd39874ec4bf42211365dbda42e/gepa_artifact/benchmarks/hover/hover_program.py)
``search`` function: download ``wiki.abstracts.2017.tar.gz`` once from
HuggingFace, build a BM25 index with ``bm25s`` (k1=0.9, b=0.4 — the
artifact's tuning), and serve top-k lookups. The index and corpus are
cached on disk under a directory the runtime owns, and held in memory as
module-level singletons after first initialization.

The artifact returns each retrieval result as ``"title | text"``; we
parse that back into :class:`HotpotQAParagraph` for type uniformity
with the distractor path. Per-call dedup is delegated to the program
(``HotpotQAProgram`` tracks already-retrieved titles across hops).

This file is imported lazily by :mod:`.bm25` only when
``retrieval_mode="fullwiki"`` is selected, so distractor-mode tests
never pay the ``bm25s`` startup cost.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...data.dataset import HotpotQAParagraph


logger = logging.getLogger(__name__)


_WIKI_TARBALL_URL = "https://huggingface.co/dspy/cache/resolve/main/wiki.abstracts.2017.tar.gz"
_WIKI_TARBALL_NAME = "wiki.abstracts.2017.tar.gz"
_WIKI_CORPUS_NAME = "wiki.abstracts.2017.jsonl"
_BM25S_INDEX_DIR = "bm25s_retriever"


_init_lock = threading.Lock()
_initialized = False
# The bm25s and PyStemmer libraries ship without stubs; we hold the
# already-initialized singletons as ``Any`` so attribute access against
# them isn't blocked by mypy.
_retriever: Any | None = None
_stemmer: Any | None = None
_corpus: list[str] | None = None


def _download_to(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` atomically using an absolute path.

    Earlier this module used ``dspy.utils.download``, which always
    writes to ``os.path.basename(url)`` in the *current working
    directory*. Pointing it at a custom cache dir required an
    ``os.chdir(directory)`` wrap — but ``os.chdir`` is process-wide,
    so any unrelated filesystem operation on another thread during the
    download window would resolve relative paths against the wrong
    directory. The download window is bounded (it only fires on the
    first-ever fullwiki invocation per cache dir), but the hazard is
    real, so we stream directly to ``dest`` via ``urllib`` instead.

    Atomic via ``dest.with_suffix(dest.suffix + ".part")`` then rename —
    a crashed mid-download leaves no half-written file that the
    ``not tarball_path.exists()`` guard would mistake for a finished
    one on the next run.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as f:  # noqa: S310 — trusted HF URL
        shutil.copyfileobj(resp, f)
    tmp.replace(dest)


def _default_cache_dir() -> Path:
    """Cache directory the bm25s index and the abstracts dump live under.

    Defaults to ``$XDG_CACHE_HOME/rilixai/hotpotqa/fullwiki`` (or
    ``~/.cache/...``). The cache is reused across runs once built, so the
    first run pays the ~few-minutes index cost and subsequent runs load
    from disk in seconds.
    """
    override = os.environ.get("RILIXAI_HOTPOTQA_FULLWIKI_CACHE_DIR")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "rilixai" / "hotpotqa" / "fullwiki"


def _ensure_initialized(cache_dir: Path | None = None) -> None:
    """Build the bm25s index on disk if missing, then load it into memory.

    Safe to call concurrently; only the first caller does the work.
    """
    global _initialized, _retriever, _stemmer, _corpus
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return

        import bm25s
        import Stemmer

        directory = cache_dir or _default_cache_dir()
        directory.mkdir(parents=True, exist_ok=True)

        corpus_path = directory / _WIKI_CORPUS_NAME
        index_path = directory / _BM25S_INDEX_DIR
        tarball_path = directory / _WIKI_TARBALL_NAME

        if not corpus_path.exists() or not index_path.exists():
            logger.info("Building HotpotQA fullwiki bm25s index under %s; this may take several minutes", directory)
            if not tarball_path.exists():
                logger.info("Downloading %s", _WIKI_TARBALL_URL)
                _download_to(_WIKI_TARBALL_URL, tarball_path)
            if not corpus_path.exists():
                import tarfile

                with tarfile.open(tarball_path, "r:gz") as tar:
                    # ``filter="data"`` (Python 3.12+) rejects entries that
                    # would write outside the target directory (path
                    # traversal) or set unsafe permissions. The tarball is
                    # served by HuggingFace, but a compromised /
                    # MITM'd download would otherwise be free to overwrite
                    # arbitrary files via ``../../`` entries — and as of
                    # 3.12 the default-no-filter call raises a
                    # ``DeprecationWarning`` for exactly this reason.
                    tar.extractall(path=directory, filter="data")
            assert corpus_path.exists(), f"Corpus file not found at {corpus_path} after extraction."

            # Cold path: read corpus once, reuse for both index
            # construction below AND the in-memory singleton at the
            # bottom of this function. ``corpus_data`` stays bound on
            # the warm path too (``None`` sentinel) so we re-read only
            # when we didn't just build.
            corpus_data: list[str] | None = _read_corpus(corpus_path)
            stemmer = Stemmer.Stemmer("english")
            corpus_tokens = bm25s.tokenize(corpus_data, stopwords="en", stemmer=stemmer)
            # k1=0.9, b=0.4 mirror the artifact's tuning.
            retriever = bm25s.BM25(k1=0.9, b=0.4)
            retriever.index(corpus_tokens)
            retriever.save(str(index_path))
            assert index_path.exists(), f"Index not saved at {index_path}."
        else:
            corpus_data = None

        # Load (or reload) the on-disk artifacts into the module-level
        # singletons. Re-load the bm25s index from disk even when we
        # just built it so warm + cold paths produce identical
        # in-memory state; the corpus list is reused from the cold-
        # path read when available (the multi-GB file is otherwise
        # parsed twice on first-ever init).
        retriever = bm25s.BM25.load(str(index_path))
        stemmer = Stemmer.Stemmer("english")
        if corpus_data is None:
            corpus_data = _read_corpus(corpus_path)

        _retriever = retriever
        _stemmer = stemmer
        _corpus = corpus_data
        _initialized = True


def _read_corpus(corpus_path: Path) -> list[str]:
    """Parse the Wikipedia abstracts JSONL into the artifact's flat ``"title | text"`` strings.

    Pulled into a module helper so the cold (build) and warm (reload)
    paths share one implementation and the multi-GB read happens at
    most once per ``_ensure_initialized`` invocation.
    """
    import ujson

    out: list[str] = []
    with corpus_path.open() as f:
        for line in f:
            record = ujson.loads(line)
            out.append(f"{record['title']} | {' '.join(record['text'])}")
    return out


def fullwiki_retrieve_k_fn(
    *,
    k_default: int = 7,
    cache_dir: Path | None = None,
) -> Callable[[str, int], list[HotpotQAParagraph]]:
    """Return a ``RetrieveKFn`` over the global fullwiki bm25s index.

    ``k_default`` is used when callers do not pass an explicit ``k`` (kept
    aligned with the artifact's ``k=7``). The returned function is
    process-safe; the underlying index is held as a module-level
    singleton.
    """

    def _retrieve(query: str, k: int) -> list[HotpotQAParagraph]:
        _ensure_initialized(cache_dir=cache_dir)
        assert _retriever is not None
        assert _stemmer is not None
        assert _corpus is not None
        import bm25s

        effective_k = k if k > 0 else k_default
        tokens = bm25s.tokenize(query, stopwords="en", stemmer=_stemmer, show_progress=False)
        results, _scores = _retriever.retrieve(tokens, k=effective_k, n_threads=1, show_progress=False)
        out: list[HotpotQAParagraph] = []
        seen_titles: set[str] = set()
        for doc_idx in results[0]:
            entry = _corpus[doc_idx]
            title, _, text = entry.partition(" | ")
            if title in seen_titles:
                continue
            seen_titles.add(title)
            out.append(HotpotQAParagraph(title=title, sentences=(text,)))
            if len(out) >= effective_k:
                break
        return out

    return _retrieve


def reset_for_tests() -> None:
    """Drop the module-level cached index. Tests use this to avoid bleed."""
    global _initialized, _retriever, _stemmer, _corpus
    with _init_lock:
        _initialized = False
        _retriever = None
        _stemmer = None
        _corpus = None
