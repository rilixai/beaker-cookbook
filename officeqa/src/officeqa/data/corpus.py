"""Corpus download, caching and per-question workspace layout.

Cache layout (``~/.cache/officeqa/`` or ``$OFFICEQA_CACHE_DIR``)::

    hf/                                   snapshot_download(local_dir=...) target
      treasury_bulletin_pdfs/*.pdf          ~4.2 GB, 697 files
      treasury_bulletins_parsed/transformed/*.txt   ~480 MB, 697 files
    officeqa_corpus/                      the agent-facing root
      pdfs/    -> ../hf/treasury_bulletin_pdfs
      parsed/  -> ../hf/treasury_bulletins_parsed/transformed

Per-question workspace (``<output_dir>/work/<uid>/``)::

    officeqa_corpus/  -> <cache>/officeqa_corpus          (symlink, never copied)
    cwd/              the agent's working directory; prompt says ``../officeqa_corpus/``
    .venv/            optional per-question virtualenv for the REPL

``huggingface_hub`` is imported lazily so this module stays importable
offline; tests monkeypatch :func:`_snapshot`.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from officeqa import config


logger = logging.getLogger(__name__)


def _snapshot(*, allow_patterns: list[str], local_dir: Path, revision: str) -> None:
    """Resumable download of a corpus subtree into ``local_dir`` (real files, progress bars)."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=config.OFFICEQA_HF_REPO,
        repo_type="dataset",
        revision=revision,
        allow_patterns=allow_patterns,
        local_dir=str(local_dir),
        max_workers=8,
    )


def hf_dir(cache: Path | None = None) -> Path:
    return (cache or config.cache_dir()) / "hf"


def corpus_root(cache: Path | None = None) -> Path:
    return (cache or config.cache_dir()) / config.CORPUS_ROOT_NAME


def _link(target: Path, link: Path) -> None:
    """Idempotent directory symlink ``link -> target``."""
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"{link} exists and is not a symlink")
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target.resolve(), link, target_is_directory=True)


def materialize(cache: Path | None = None) -> Path:
    """Create ``<cache>/officeqa_corpus/{pdfs,parsed}`` symlinks for whatever is downloaded."""
    root = corpus_root(cache)
    root.mkdir(parents=True, exist_ok=True)
    for name, (hf_subdir, _fmt) in config.CORPUS_REPRESENTATIONS.items():
        src = hf_dir(cache) / hf_subdir
        if src.is_dir():
            _link(src, root / name)
    return root


def fetch_corpus(
    representations: tuple[str, ...] = ("pdfs", "parsed"),
    *,
    cache: Path | None = None,
    revision: str = config.OFFICEQA_DATASET_REVISION,
) -> Path:
    """Download the requested representations (resumable) and materialize the corpus root.

    Both representations are fetched by default so the manifest can list
    both; ``--corpus`` only picks which one is the *default* for the agent.
    """
    patterns = {"pdfs": config.CORPUS_PDF_PATTERN, "parsed": config.CORPUS_PARSED_PATTERN}
    target = hf_dir(cache)
    target.mkdir(parents=True, exist_ok=True)
    for rep in representations:
        logger.info("fetching %s (%s) into %s", rep, patterns[rep], target)
        _snapshot(allow_patterns=[patterns[rep]], local_dir=target, revision=revision)
    root = materialize(cache)
    for rep in representations:
        n = sum(1 for _ in (root / rep).iterdir())
        if n != config.EXPECTED_CORPUS_DOCUMENTS:
            logger.warning("%s has %d documents; expected %d", rep, n, config.EXPECTED_CORPUS_DOCUMENTS)
    return root


def ensure_corpus(required: str, *, cache: Path | None = None) -> Path:
    """Return the corpus root, verifying the required representation is present."""
    root = materialize(cache)
    if not (root / required).is_dir():
        raise FileNotFoundError(f"corpus representation {required!r} missing under {root}; run `officeqa fetch` first")
    return root


# --- Per-question workspace -------------------------------------------------


@dataclass(frozen=True)
class Workspace:
    """One question's isolated directories."""

    root: Path
    cwd: Path
    corpus_link: Path
    python: str

    @property
    def allowed_roots(self) -> tuple[Path, ...]:
        """Directories the fs tools may touch: the cwd and each corpus representation."""
        roots = [self.cwd.resolve()]
        for name in config.CORPUS_REPRESENTATIONS:
            sub = self.corpus_link / name
            if sub.is_dir():
                roots.append(sub.resolve())
        return tuple(roots)


def _make_venv(venv: Path) -> str:
    """Per-question virtualenv (report §3.2). System site-packages stay visible so
    preinstalled OCR libraries work; anything the agent installs lands here."""
    if not venv.exists():
        uv = shutil.which("uv")
        if uv:
            subprocess.run(
                [uv, "venv", "--quiet", "--seed", "--system-site-packages", "--python", sys.executable, str(venv)],
                check=True,
                capture_output=True,
            )
        else:
            subprocess.run(
                [sys.executable, "-m", "venv", "--system-site-packages", str(venv)], check=True, capture_output=True
            )
    py = venv / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    return str(py)


def create_workspace(
    work_dir: Path, uid: str, corpus: Path, *, isolate: bool = config.ISOLATE_PER_QUESTION, fresh: bool = True
) -> Workspace:
    """Lay out ``work_dir/<uid>/{officeqa_corpus -> corpus, cwd/, .venv/}``."""
    root = work_dir / uid
    cwd = root / "cwd"
    if fresh and cwd.exists():
        shutil.rmtree(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    link = root / config.CORPUS_ROOT_NAME
    _link(corpus, link)
    python = _make_venv(root / ".venv") if isolate else sys.executable
    return Workspace(root=root, cwd=cwd, corpus_link=link, python=python)
