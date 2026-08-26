"""The Beaker lever: two extra agent tools that read a filesystem ``skills/`` dir.

Skills are organized on two axes, one folder per skill, each holding a
``SKILL.md`` (an Anthropic Agent Skill: YAML frontmatter + markdown body)::

    skills/
      domains/<domain>/SKILL.md   # e.g. domains/finance
      apps/<app>/SKILL.md         # e.g. apps/gmail

A skill's ID is its folder path under ``skills/`` (``apps/gmail``,
``domains/finance``), so the two axes never collide.

The tools read the directory **live on every call** — nothing is cached — so an
optimizer can add/edit/split/merge skill folders between ``run_one`` calls and
the very next rollout sees the new contents, with no environment rebuild.

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
    """Point the skill tools at a skills directory (or None)."""
    _ACTIVE.path = Path(path) if path is not None else None


def get_skills_dir() -> Path | None:
    return _ACTIVE.path


def _skill_files() -> dict[str, Path]:
    """Map skill ID (folder path under skills/) -> its SKILL.md, read live."""
    root = _ACTIVE.path
    if root is None or not root.is_dir():
        return {}
    return {f.parent.relative_to(root).as_posix(): f for f in sorted(root.rglob("SKILL.md")) if f.parent != root}


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse minimal ``key: value`` YAML frontmatter; return (meta, body)."""
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    meta: dict[str, str] = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return meta, "\n".join(lines[i + 1 :]).strip("\n")
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip("\"'")
    return {}, text


def list_skills() -> str:
    """List the reusable skill guides available for this workspace. Call this first,
    before planning tool calls: skills contain proven step-by-step procedures for
    common workflows. Returns one `id: description` line per skill; read a relevant
    skill with read_skill(id) before acting.

    Returns:
        One line per skill: `<id>: <description>` (IDs look like `apps/gmail` or
        `domains/finance`), or a note that no skills are available.
    """
    files = _skill_files()
    if not files:
        return "No skills available."
    lines = []
    for skill_id, f in sorted(files.items()):
        try:
            meta, body = _split_frontmatter(f.read_text())
        except OSError:
            continue
        description = meta.get("description", "")
        if not description:
            description = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
        lines.append(f"{skill_id}: {description}")
    return "\n".join(lines) if lines else "No skills available."


def read_skill(skill_id: str) -> str:
    """Read the full body of one skill guide listed by list_skills. Skills contain
    proven step-by-step procedures — follow the relevant one when executing a
    workflow it covers.

    Args:
        skill_id: The skill ID exactly as returned by list_skills, e.g.
            `apps/gmail` or `domains/finance`.

    Returns:
        The skill's markdown body, or an error naming the available skills.
    """
    files = _skill_files()
    f = files.get(skill_id) or files.get(skill_id.strip("/"))
    if f is None:
        available = ", ".join(sorted(files)) or "(none)"
        return f"Unknown skill {skill_id!r}. Available skills: {available}"
    try:
        _, body = _split_frontmatter(f.read_text())
    except OSError as e:
        return f"Could not read skill {skill_id!r}: {e}"
    return body


SKILL_TOOLS = [list_skills, read_skill]
