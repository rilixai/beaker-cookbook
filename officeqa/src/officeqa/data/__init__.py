"""OfficeQA data: gated HF loading, the ground-truth boundary, corpus cache, manifest."""

from officeqa.data.corpus import Workspace, corpus_root, create_workspace, ensure_corpus, fetch_corpus
from officeqa.data.dataset import (
    KNOWN_SPLITS,
    AgentInput,
    EvalRecord,
    load_records,
    load_records_from_csv,
    load_split,
    read_split_ids,
    select_split,
)
from officeqa.data.manifest import CorpusManifest, Representation, build_manifest


__all__ = [
    "KNOWN_SPLITS",
    "AgentInput",
    "CorpusManifest",
    "EvalRecord",
    "Representation",
    "Workspace",
    "build_manifest",
    "corpus_root",
    "create_workspace",
    "ensure_corpus",
    "fetch_corpus",
    "load_records",
    "load_records_from_csv",
    "load_split",
    "read_split_ids",
    "select_split",
]
