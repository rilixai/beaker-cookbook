"""Launch every Beaker case through the official containerized runner."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from toolathlon_docker import run_containerized_agent

from toolathlon_baseline.prepare import native_path, prepared_source


def run_agent(
    *,
    cache: Path,
    output: Path,
    task: str,
    model: str,
    max_steps: int = 100,
    timeout: int = 1800,
    system_prompt: Path | None = None,
    model_params: Path | None = None,
) -> dict[str, Any]:
    if max_steps < 1 or timeout < 1:
        raise ValueError("max_steps and timeout must be positive")
    if not os.environ.get("TOOLATHLON_OPENAI_BASE_URL"):
        raise ValueError("Set TOOLATHLON_OPENAI_BASE_URL first")
    cache, output = native_path(cache), native_path(output)
    source = prepared_source(cache)
    return run_containerized_agent(
        cache=cache,
        source=source,
        output=output,
        task=task,
        model=model,
        max_steps=max_steps,
        timeout=timeout,
        system_prompt=system_prompt,
        model_params=model_params,
    )
