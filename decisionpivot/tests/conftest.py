"""Session fixtures: the pinned upstream checkout and one full ``prepare`` (about a minute; no paid calls)."""

from __future__ import annotations

from pathlib import Path

import pytest

from decisionpivot.data import DEFAULT_CACHE, fetch_upstream, prepare


@pytest.fixture(scope="session")
def upstream() -> Path:
    return fetch_upstream(DEFAULT_CACHE)


@pytest.fixture(scope="session")
def prepared(upstream: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("decisionpivot-data")
    prepare(out, upstream)
    return out
