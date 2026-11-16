# Vendored from AutomationBench by Zapier, Inc.
# Source: automationbench/scripts/eval.py
# Upstream: https://github.com/zapier/AutomationBench
# Commit: 4a8e1061254004d9dac807054eed33fad7d1ff14
# License: MIT (Copyright 2026 Zapier, Inc.) — see ../../../LICENSE
# Modified: the API-resolution, sampling-args, API-key-var, and (non-batch)
# client construction inline in eval.py's run_evaluation() were extracted into
# the standalone functions below so a per-sample runner can reuse them without
# invoking the whole-suite CLI. The batch-API client paths were dropped
# (single-rollout usage has no batching); the logic itself is unchanged.
# _resolve_api / _resolve_adaptive_models / _warn_if_unmatched_preview are
# retained here because PI removed that module.

"""Model routing reused from AutomationBench's own eval entry point.

Anthropic-native for claude-*, Gemini interactions for gemini-*, OpenAI
Chat Completions / Responses otherwise; gateway models via --base-url.
"""

from __future__ import annotations

import json
import os

from anthropic import AsyncAnthropic
from automationbench_skills.vendored.clients import (
    GeminiInteractionsClient,
    OpenAIResponsesClient,
    RetryingOpenAIChatCompletionsClient,
    StreamingAnthropicClient,
)
from automationbench_skills.model_protocol import Client, ClientConfig



_ADAPTIVE_MODELS_PUBLIC = (
    "opus-4-6",
    "opus-4-7",
    "opus-4-8",
    "opus-4.8",  # AI Gateway model ids use dots (anthropic/claude-opus-4.8)
    "opus-5",
    "sonnet-4-6",
    "sonnet-5",
    "haiku-5-5",
    "fable-5",
)

def _resolve_adaptive_models() -> tuple[str, ...]:
    """Adaptive-thinking model substrings: the public list plus AUTO_BENCH_ADAPTIVE_MODELS.

    Preview/EAP codenames stay out of the source tree — they leak via the public sync and repo
    exports. Supply them at launch instead:

        env AUTO_BENCH_ADAPTIVE_MODELS=<name>[,<name>...] uv run auto-bench --model <id> ...

    Preview models are adaptive-ONLY: a manual `budget_tokens` either 400s outright or yields a
    degraded run that still scores, so a missing value must be loud rather than silent.
    """
    extra = tuple(
        part.strip().lower()
        for part in os.environ.get("AUTO_BENCH_ADAPTIVE_MODELS", "").split(",")
        if part.strip()
    )
    return _ADAPTIVE_MODELS_PUBLIC + extra

def _warn_if_unmatched_preview(model: str, adaptive: tuple[str, ...]) -> None:
    """Fail loudly when a model looks like an unreleased preview but matched no adaptive entry.

    Heuristic only: anything routed at the Anthropic API whose id is neither a known public
    family nor an AUTO_BENCH_ADAPTIVE_MODELS entry. That combination almost always means the env
    var is unset, which would 400 every rollout or silently produce budget-thinking results.
    """
    known_public = ("opus", "sonnet", "haiku", "fable", "claude-3", "claude-2")
    if any(m in model for m in adaptive):
        return
    if any(k in model.lower() for k in known_public):
        return
    raise SystemExit(
        f"Refusing to run: model {model!r} matched no adaptive-thinking entry and is not a known "
        "public family. Preview models are adaptive-only; a manual thinking budget either 400s or "
        "produces a silently degraded score. Set AUTO_BENCH_ADAPTIVE_MODELS (comma-separated id "
        "substrings) and re-run."
    )

def _resolve_api(model: str, base_url: str | None, api_override: str = "auto") -> str:
    """Return which API client to use: 'anthropic', 'chat_completions', 'responses',
    or 'gemini_interactions'.

    Explicit `api_override` (anything other than 'auto') wins. Otherwise auto-detects
    the first-party Anthropic/Google APIs by model prefix and base_url; everything
    else (gateway/proxy/alpha endpoints) stays on Chat Completions.
    """
    if api_override != "auto":
        return api_override
    if model.startswith("claude-") and (base_url is None or "anthropic.com" in base_url):
        return "anthropic"
    if model.startswith("gemini-") and (
        base_url is None or "generativelanguage.googleapis.com" in base_url
    ):
        return "gemini_interactions"
    return "chat_completions"


def resolve_api(model: str, base_url: str | None, api_override: str = "auto") -> str:
    """Return which API client to use: 'anthropic', 'chat_completions', 'responses',
    or 'gemini_interactions'."""
    return _resolve_api(model, base_url, api_override)


def build_sampling_args(
    model: str,
    resolved_api: str,
    reasoning_effort: str | None,
    extra_body: str | None = None,
) -> dict | None:
    """Build per-request sampling args exactly as upstream eval.py does."""
    sampling_args = None
    if reasoning_effort:
        if resolved_api == "anthropic":
            # Opus 4.6+ and Sonnet 4.6 support adaptive thinking with output_config effort.
            # Older models (Haiku 4.5, Sonnet 4.5, etc.) require manual budget_tokens.
            _adaptive_models = _resolve_adaptive_models()
            _warn_if_unmatched_preview(model, _adaptive_models)
            if any(m in model for m in _adaptive_models):
                sampling_args = {
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": reasoning_effort},
                    "max_tokens": 64000,
                }
            else:
                # Map effort to thinking budget for older models
                _budget = {"low": 2000, "medium": 8000, "high": 16000, "xhigh": 24000, "max": 32000}
                budget_tokens = _budget.get(reasoning_effort, 8000)
                sampling_args = {
                    "thinking": {"type": "enabled", "budget_tokens": budget_tokens},
                    "max_tokens": 64000,
                }
        else:
            # Top-level reasoning_effort: the gemini_interactions client maps it to
            # generation_config.thinking_level; gateways map it to the provider-native
            # format (for Gemini it becomes thinkingLevel).
            sampling_args = {"reasoning_effort": reasoning_effort}

    if extra_body:
        # Merge raw JSON into every request body (e.g. AI Gateway providerOptions).
        parsed = json.loads(extra_body)
        sampling_args = sampling_args or {}
        sampling_args["extra_body"] = {**sampling_args.get("extra_body", {}), **parsed}

    return sampling_args


def resolve_api_key_var(resolved_api: str, api_key_var: str = "OPENAI_API_KEY") -> str:
    """Which env var holds the key. The Anthropic/Gemini defaults only kick in when
    --api-key-var was left at its default, so an explicit var still wins."""
    if resolved_api == "anthropic" and api_key_var == "OPENAI_API_KEY":
        return "ANTHROPIC_API_KEY"
    if resolved_api == "gemini_interactions" and api_key_var == "OPENAI_API_KEY":
        return "GEMINI_API_KEY"
    return api_key_var


def build_client(
    resolved_api: str,
    api_key_var: str,
    base_url: str | None,
    extra_headers: dict[str, str] | None = None,
) -> Client:
    """Construct the verifiers client for the resolved API (non-batch paths)."""
    if not os.environ.get(api_key_var):
        raise ValueError(f"No API key found. Set the {api_key_var} environment variable.")

    if resolved_api == "anthropic":
        return StreamingAnthropicClient(AsyncAnthropic(api_key=os.environ[api_key_var]))
    if resolved_api == "gemini_interactions":
        return GeminiInteractionsClient(
            ClientConfig(
                api_key_var=api_key_var,
                api_base_url=base_url or GeminiInteractionsClient.DEFAULT_BASE_URL,
                extra_headers=extra_headers or {},
            )
        )
    config = ClientConfig(
        api_key_var=api_key_var,
        api_base_url=base_url or "https://api.openai.com/v1",
        extra_headers=extra_headers or {},
    )
    if resolved_api == "responses":
        return OpenAIResponsesClient(config)
    return RetryingOpenAIChatCompletionsClient(config)
