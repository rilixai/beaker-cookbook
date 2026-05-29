"""In-memory fake world that mirrors :class:`WorldFiles`'s read surface.

The fake holds a ``{relative_path: text}`` mapping — no disk, no
network, no zip extraction. It implements the same methods the agent's
domain tools call so unit tests can drive the full ReAct loop
end-to-end without touching HuggingFace.

Spreadsheets / PDFs are faked as plain text: the test preloads the
already-rendered text under the file's path. ``read_spreadsheet`` /
``read_pdf`` just return that text so the agent code path is exercised
without ``openpyxl`` / ``pypdf``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


__all__ = ["FakeWorld", "fake_world_factory"]


class FakeWorld:
    """Scripted in-memory stand-in for :class:`WorldFiles`."""

    def __init__(self, files: Mapping[str, str] | None = None) -> None:
        # Preserve insertion order; ``list_files`` returns sorted keys
        # to match :class:`WorldFiles`.
        self._files: dict[str, str] = {str(k): str(v) for k, v in (files or {}).items()}

    # ─── construction parity with WorldFiles ──────────────────────────

    def close(self) -> None:
        """No-op — nothing to tear down for an in-memory world."""
        return None

    # ─── read surface ─────────────────────────────────────────────────

    def list_files(self, subdir: str = "") -> list[str]:
        prefix = subdir.strip("/")
        if not prefix:
            return sorted(self._files)
        prefix = prefix + "/"
        return sorted(p for p in self._files if p == subdir or p.startswith(prefix))

    def read_text(self, rel_path: str, *, max_bytes: int = 200_000) -> str:
        if rel_path not in self._files:
            raise FileNotFoundError(rel_path)
        return self._files[rel_path][:max_bytes]

    def read_spreadsheet(
        self,
        rel_path: str,
        *,
        sheet: str | None = None,
        max_rows: int = 200,
        max_cols: int = 40,
    ) -> str:
        if rel_path not in self._files:
            raise FileNotFoundError(rel_path)
        # The test preloads the rendered table text under this path. A
        # test may key by ``"<path>::<sheet>"`` to exercise sheet
        # targeting; fall back to the bare path otherwise.
        if sheet is not None and f"{rel_path}::{sheet}" in self._files:
            return self._files[f"{rel_path}::{sheet}"]
        return self._files[rel_path]

    def read_pdf(self, rel_path: str, *, max_pages: int = 50) -> str:
        if rel_path not in self._files:
            raise FileNotFoundError(rel_path)
        return self._files[rel_path]

    def read_docx(self, rel_path: str, *, max_bytes: int = 200_000) -> str:
        if rel_path not in self._files:
            raise FileNotFoundError(rel_path)
        return self._files[rel_path][:max_bytes]

    def search(self, query: str, *, max_results: int = 50) -> list[dict[str, Any]]:
        needle = query.casefold()
        if not needle:
            return []
        hits: list[dict[str, Any]] = []
        for rel in sorted(self._files):
            for line_no, line in enumerate(self._files[rel].splitlines(), start=1):
                if needle in line.casefold():
                    hits.append({"file": rel, "line": line_no, "text": line.strip()[:300]})
                    if len(hits) >= max_results:
                        return hits
        return hits


def fake_world_factory(files: Mapping[str, str] | None = None) -> Any:
    """Return a ``(record) -> FakeWorld`` factory yielding one shared fake world."""
    world = FakeWorld(files)

    def _factory(_record: Any) -> FakeWorld:
        return world

    return _factory
