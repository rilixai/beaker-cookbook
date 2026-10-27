"""The agent's system prompt: ``prompts/system.md`` with skills,
``prompts/system_no_skills.md`` without.

The file is the whole system message; the task (user) message comes from the
dataset row. It is read on every call, like the skills. With no prompts
directory, or a blank file, the row's own system message is used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT_FILE = "system.md"
NO_SKILLS_SYSTEM_PROMPT_FILE = "system_no_skills.md"


def system_prompt_file(*, skills: bool) -> str:
    return SYSTEM_PROMPT_FILE if skills else NO_SKILLS_SYSTEM_PROMPT_FILE


def load_system_prompt(prompts_dir: Path | str | None, *, skills: bool = True) -> str | None:
    if prompts_dir is None:
        return None
    path = Path(prompts_dir) / system_prompt_file(skills=skills)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def task_clock(info: dict[str, Any]) -> str | None:
    """The task's current date/time from ``info["initial_state"]["meta"]["current_time"]``.

    ``initial_state`` may be a dict or its JSON string form; returns ``None``
    when either is missing or carries no ``meta.current_time``.
    """
    initial_state = info.get("initial_state")
    if isinstance(initial_state, str):
        try:
            initial_state = json.loads(initial_state)
        except (ValueError, TypeError):
            return None
    if not isinstance(initial_state, dict):
        return None
    meta = initial_state.get("meta")
    if not isinstance(meta, dict):
        return None
    current_time = meta.get("current_time")
    return current_time if isinstance(current_time, str) and current_time else None


def with_system_prompt(prompt: Any, system_prompt: str | None, *, clock: str | None = None) -> Any:
    """Return ``prompt`` (a chat message list) with ``system_prompt`` as its system message.

    Does not mutate the input. A prompt without a leading system message gets
    one; plain-string prompts are returned unchanged. ``clock``, when given, is
    appended to the system message as ``Current date and time: <clock>`` after
    a blank line. With ``system_prompt`` that message replaces the row's own;
    with ``system_prompt`` ``None`` the row's leading system message is kept
    and the line appended to it (or a clock-only system message is added when
    the prompt has none).
    """
    if not isinstance(prompt, list) or (not system_prompt and not clock):
        return prompt
    line = f"Current date and time: {clock}" if clock else None
    messages = [dict(m) for m in prompt]
    if system_prompt:
        content = f"{system_prompt}\n\n{line}" if line else system_prompt
        system = {"role": "system", "content": content}
        if messages and messages[0].get("role") == "system":
            return [system, *messages[1:]]
        return [system, *messages]
    if messages and messages[0].get("role") == "system":
        messages[0] = {**messages[0], "content": f"{messages[0].get('content', '')}\n\n{line}"}
        return messages
    return [{"role": "system", "content": line}, *messages]
