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
# imported from the upstream module rather than copied.

"""Model routing reused from AutomationBench's own eval entry point.

Anthropic-native for claude-*, Gemini interactions for gemini-*, OpenAI
Chat Completions / Responses otherwise; gateway models via --base-url.
"""

from __future__ import annotations

import json
import os

from anthropic import AsyncAnthropic
from automationbench.clients import (
    GeminiInteractionsClient,
    OpenAIResponsesClient,
    RetryingOpenAIChatCompletionsClient,
    StreamingAnthropicClient,
)
from automationbench.scripts.eval import (
    _resolve_adaptive_models,
    _resolve_api,
    _warn_if_unmatched_preview,
)
from verifiers.legacy.clients.client import Client
from verifiers.types import ClientConfig


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
    if resolved_api == "anthropic":
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
        return StreamingAnthropicClient(AsyncAnthropic())
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
