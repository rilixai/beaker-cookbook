"""The second Beaker lever: the skills-mode system prompt, read from ``prompts/system.md``.

The file holds the whole system message. It is seeded with the benchmark's own
``SYSTEM_PROMPT`` (one text, identical across the six domains) followed by a
skill-routing paragraph, and the optimizer may rewrite any of it. When a
prompts directory is given, its ``system.md`` replaces the dataset row's system
message; the user (task) message is untouched. The file is read at rollout time
so an edit (alongside ``skills/``) is observed without rebuilding the
environment. Missing directory or blank file: the prompt is the benchmark's,
unchanged — that is what the ``--no-skills`` baseline runs with.
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
    """Replace the system message of a chat prompt with ``system_prompt``.

    ``prompt`` is the dataset row's message list (``[{"role": "system", ...},
    {"role": "user", ...}]``). Returns a new list; the input is not mutated.
    A prompt without a leading system message gets one. Plain-string prompts
    are returned unchanged.
    """
    if not system_prompt or not isinstance(prompt, list):
        return prompt
    messages = [dict(m) for m in prompt]
    system = {"role": "system", "content": system_prompt}
    if messages and messages[0].get("role") == "system":
        return [system, *messages[1:]]
    return [system, *messages]
