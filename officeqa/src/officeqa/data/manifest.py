"""Neutral corpus manifest handed to the agent.

The manifest states what is on disk under ``officeqa_corpus/`` — both
representations, with document counts — and which one is the default. It
puts the existence of the parsed corpus in the trace without steering the
agent toward it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from officeqa import config


@dataclass(frozen=True, slots=True)
class Representation:
    name: str
    path: str
    format: str
    documents: int

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "path": self.path, "format": self.format, "documents": self.documents}


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    """``root`` is the agent-relative path (``../officeqa_corpus/``)."""

    root: str
    representations: tuple[Representation, ...]
    default: str

    def to_json(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "representations": [r.to_json() for r in self.representations],
            "default": self.default,
        }

    def representation(self, name: str) -> Representation:
        for rep in self.representations:
            if rep.name == name:
                return rep
        raise KeyError(name)


def _count_documents(directory: Path, suffix: str) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.iterdir() if p.suffix == suffix)


def build_manifest(
    corpus_root: Path, *, default: str = config.DEFAULT_CORPUS, root_label: str = "../officeqa_corpus/"
) -> CorpusManifest:
    """Describe the representations present under ``corpus_root``.

    ``corpus_root`` is the on-disk ``officeqa_corpus/`` directory (or a
    symlink to it). Both ``pdfs/`` and ``parsed/`` are listed when present;
    ``default`` names the one the run is configured for.
    """
    reps: list[Representation] = []
    suffixes = {"pdf": ".pdf", "text": ".txt"}
    for name, (_hf_subdir, fmt) in config.CORPUS_REPRESENTATIONS.items():
        sub = corpus_root / name
        if sub.is_dir():
            reps.append(
                Representation(name=name, path=f"{name}/", format=fmt, documents=_count_documents(sub, suffixes[fmt]))
            )
    if default not in {r.name for r in reps}:
        raise FileNotFoundError(f"default corpus {default!r} is not present under {corpus_root}; run `officeqa fetch`")
    return CorpusManifest(root=root_label, representations=tuple(reps), default=default)
