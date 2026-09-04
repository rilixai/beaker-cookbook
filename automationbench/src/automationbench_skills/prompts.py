"""The agent's system prompt, ``prompts/system.md``.

The file is the whole system message; the task (user) message comes from the
dataset row. It is read on every call, like the skills. With no prompts
directory, or a blank file, the row's own system message is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


SYSTEM_PROMPT_FILE = "system.md"


def load_system_prompt(prompts_dir: Path | str | None) -> str | None:
    if prompts_dir is None:
        return None
    path = Path(prompts_dir) / SYSTEM_PROMPT_FILE
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def with_system_prompt(prompt: Any, system_prompt: str | None) -> Any:
    """Return ``prompt`` (a chat message list) with ``system_prompt`` as its system message.

    Does not mutate the input. A prompt without a leading system message gets
    one; plain-string prompts are returned unchanged.
    """
    if not system_prompt or not isinstance(prompt, list):
        return prompt
    messages = [dict(m) for m in prompt]
    system = {"role": "system", "content": system_prompt}
    if messages and messages[0].get("role") == "system":
        return [system, *messages[1:]]
    return [system, *messages]
