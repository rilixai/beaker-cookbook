"""The Beaker lever: two extra agent tools that read a filesystem ``skills/`` dir.

The tools read the directory **live on every call** — nothing is cached — so an
optimizer can edit skill files between ``run_one`` calls and the very next
rollout sees the new contents, with no environment rebuild.

The only "consult your skills" nudge lives in the tool descriptions
(docstrings); the benchmark's per-domain system prompts are never touched.
"""

from __future__ import annotations

from pathlib import Path


class _ActiveSkillsDir:
    """Holder for the skills directory the tools read from.

    ``run_one`` points this at its ``skills_dir`` argument before each rollout,
    which is what lets one long-lived environment serve many calls with
    different (or edited) skills folders.
    """

    path: Path | None = None


_ACTIVE = _ActiveSkillsDir()


def set_skills_dir(path: Path | str | None) -> None:
    """Point the skill tools at a directory of ``*.md`` skill files (or None)."""
    _ACTIVE.path = Path(path) if path is not None else None


def get_skills_dir() -> Path | None:
    return _ACTIVE.path


def _skill_files() -> list[Path]:
    if _ACTIVE.path is None or not _ACTIVE.path.is_dir():
        return []
    return sorted(_ACTIVE.path.glob("*.md"))


def list_skills() -> str:
    """List the reusable skill guides available for this workspace. Call this first,
    before planning tool calls: skills contain proven step-by-step procedures for
    common workflows. Returns one `name: summary` line per skill; read a relevant
    skill with read_skill(name) before acting.

    Returns:
        One line per skill: `<name>: <first-line summary>`, or a note that no
        skills are available.
    """
    files = _skill_files()
    if not files:
        return "No skills available."
    lines = []
    for f in files:
        first_line = ""
        try:
            for raw in f.read_text().splitlines():
                stripped = raw.strip().lstrip("#").strip()
                if stripped:
                    first_line = stripped
                    break
        except OSError:
            continue
        lines.append(f"{f.stem}: {first_line}")
    return "\n".join(lines) if lines else "No skills available."


def read_skill(name: str) -> str:
    """Read the full body of one skill guide listed by list_skills. Skills contain
    proven step-by-step procedures — follow the relevant one when executing a
    workflow it covers.

    Args:
        name: The skill name exactly as returned by list_skills (no .md suffix).

    Returns:
        The skill's full markdown body, or an error naming the available skills.
    """
    files = {f.stem: f for f in _skill_files()}
    f = files.get(name) or files.get(name.removesuffix(".md"))
    if f is None:
        available = ", ".join(sorted(files)) or "(none)"
        return f"Unknown skill {name!r}. Available skills: {available}"
    try:
        return f.read_text()
    except OSError as e:
        return f"Could not read skill {name!r}: {e}"


SKILL_TOOLS = [list_skills, read_skill]
