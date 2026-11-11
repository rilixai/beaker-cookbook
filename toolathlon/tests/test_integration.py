from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".beaker"))

from beaker import CaseResult  # noqa: E402
from upload_dataset import rows, split  # noqa: E402

import toolathlon_setup  # noqa: E402
from beaker_integration import Row, Setup, score_case  # noqa: E402


def test_every_finalpool_task_becomes_a_row() -> None:
    task_dirs = [path for path in toolathlon_setup.TASKS.iterdir() if path.is_dir()]
    assert len(task_dirs) == 108
    assert len(rows(no_secrets=False)) == 108


def test_no_secret_rows_need_no_platform_secret() -> None:
    selected = rows(no_secrets=True)
    assert selected
    assert all(not toolathlon_setup.required_env(str(row["id"])) for row in selected)


def test_secret_names_are_declared_in_beaker_yaml() -> None:
    config = (Path(__file__).resolve().parents[1] / ".beaker" / "beaker.yaml").read_text()
    names = {name for names in toolathlon_setup.SERVICE_ENV.values() for name in names}
    assert all(f"- {name}\n" in config for name in names)


def test_split_is_stable() -> None:
    assert split("find-alita-paper") == split("find-alita-paper")
    assert {split(str(row["id"])) for row in rows(no_secrets=False)} == {"train", "test"}


def test_large_upstream_files_are_pinned_and_ignored() -> None:
    manifest = toolathlon_setup.ASSETS
    ignored = (Path(__file__).resolve().parents[1] / ".gitignore").read_text().splitlines()
    assert len(manifest["commit"]) == 40
    for relative, meta in manifest["files"].items():
        assert meta["bytes"] > 1_000_000
        assert f"/upstream/{relative}" in ignored
        path = toolathlon_setup.UPSTREAM / relative
        if path.is_file():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["sha256"]


def test_load_cases_checks_prompt() -> None:
    row = Row.model_validate(rows(no_secrets=True)[0])
    setup = Setup.__new__(Setup)

    async def load() -> list[object]:
        return [case async for case in setup.load_cases(row, runtime=None)]

    assert len(asyncio.run(load())) == 1
    row.input.prompt += " changed"
    with pytest.raises(ValueError, match="Prompt mismatch"):
        asyncio.run(load())


def test_load_cases_requires_task_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    task = next(str(row["id"]) for row in rows(no_secrets=False) if toolathlon_setup.required_env(str(row["id"])))
    for name in toolathlon_setup.required_env(task):
        monkeypatch.delenv(name, raising=False)
    row = Row.model_validate(next(r for r in rows(no_secrets=False) if r["id"] == task))
    setup = Setup.__new__(Setup)

    async def load() -> list[object]:
        return [case async for case in setup.load_cases(row, runtime=None)]

    with pytest.raises(ValueError, match="needs platform secrets"):
        asyncio.run(load())


@pytest.mark.parametrize(("value", "objective"), [(True, 1.0), (False, 0.0), (None, 0.0)])
def test_score_case_maps_toolathlon_pass(value: bool | None, objective: float) -> None:
    result = CaseResult(output={"pass": value, "details": "checked"}, output_kind="record")
    score = asyncio.run(score_case(case=None, result=result, case_files_dir=Path(".")))  # type: ignore[arg-type]
    assert score.objective == objective
    assert score.checks[0].verdict == ("pass" if objective else "fail")


def test_token_session_overrides_are_literal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    for name in ("global_configs_example.py", "token_key_session_example.py"):
        (configs / name).write_text((toolathlon_setup.UPSTREAM / "configs" / name).read_text())
    monkeypatch.setattr(toolathlon_setup, "UPSTREAM", tmp_path)
    monkeypatch.setenv("TOOLATHLON_GITHUB_TOKEN", 'tok"en')
    toolathlon_setup._write_configs()
    written = (configs / "token_key_session.py").read_text()
    assert json.dumps(json.dumps({"github_token": 'tok"en'})) in written
    assert (configs / "token_key_session.py").stat().st_mode & 0o077 == 0
