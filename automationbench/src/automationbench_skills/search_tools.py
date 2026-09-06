"""A compact ``search_tools`` for the ``zapier`` meta-tool toolset.

Upstream's ``search_tools`` returns the full JSON schema of every hit as
indented JSON, and lets the model pick ``top_k`` freely (it escalates to 50-500
when a search misses). Every result then stays in the conversation for the
rest of the rollout, so search output ends up as most of what the model
re-reads on each step.

This variant runs the same BM25 search over the same registry but renders each
hit as a signature line followed by the tool's full docstring, indented::

    xero_find_invoice(organization: str=None, invoice_number: str=None, invoice_id: str=None) — Find an invoice in Xero.

        Args:
            invoice_number: Invoice number to search.
            ...

``*`` marks a required parameter (no default). The docstring is kept whole: it
carries the argument semantics (formats, accepted values, what the tool
returns) that the JSON schema only partially duplicates. ``top_k`` is capped
at ``max_top_k``; the cap is a knob, not a constant.
"""

from __future__ import annotations

import inspect
import re
import textwrap
from collections.abc import Callable


DEFAULT_SEARCH_TOP_K = 10

_OPTIONAL = re.compile(r"^Optional\[(.*)\]$")
_TYPING_ALIASES = (("typing.", ""), ("List[", "list["), ("Dict[", "dict["), ("datetime.datetime", "datetime"))


def _type_name(annotation: object) -> str:
    """Render a parameter annotation the way a Python signature would, minus the ``None`` arm.

    Optionality is already conveyed by the ``=None`` default, so ``Optional[X]``
    and ``X | None`` both become ``X``.
    """
    text = inspect.formatannotation(annotation).strip("'\"")
    for old, new in _TYPING_ALIASES:
        text = text.replace(old, new)
    optional = _OPTIONAL.match(text)
    if optional:
        text = optional.group(1)
    return re.sub(r" \| None$", "", text)


def format_signature(name: str, func: Callable[..., object]) -> str:
    """``name(param*: type, other: type=default)`` — the arguments ``execute_tool`` accepts for ``func``."""
    params = []
    for p in inspect.signature(func).parameters.values():
        if p.name == "world":
            continue
        required = p.default is inspect.Parameter.empty
        marker = "*" if required else ""
        default = "" if required else f"={p.default!r}"
        params.append(f"{p.name}{marker}: {_type_name(p.annotation)}{default}")
    return f"{name}({', '.join(params)})"


def format_hit(name: str, func: Callable[..., object], description: str) -> str:
    """Signature, then the description with its first line inline and the rest indented under it."""
    first, _, rest = description.strip().partition("\n")
    block = f"{format_signature(name, func)} — {first}".rstrip(" —")
    if rest.strip():
        block += "\n" + textwrap.indent(rest.rstrip(), "    ", predicate=lambda line: line.strip() != "")
    return block


def format_search_results(results: list[dict[str, object]], tools: dict[str, Callable[..., object]]) -> str:
    """Render registry search hits (``name``/``description`` dicts) as signature + docstring blocks."""
    blocks = [
        format_hit(str(hit["name"]), tools[str(hit["name"])], str(hit.get("description") or "")) for hit in results
    ]
    return "\n\n".join(blocks) or "No matching tools."


def make_compact_search_tools(max_top_k: int = DEFAULT_SEARCH_TOP_K) -> Callable[[str, int], str]:
    """Build the compact ``search_tools`` over AutomationBench's shared tool registry.

    Same registry (and therefore the same BM25 index and name -> function map)
    that upstream's ``execute_tool`` dispatches through, so every name returned
    here is executable. The docstring below is what the model sees as the
    tool's description; it advertises the cap so the model does not escalate
    ``top_k`` looking for more results.
    """
    from automationbench.tools import ALL_TOOLS
    from automationbench.tools.zapier.meta import _get_registry

    tools: dict[str, Callable[..., object]] = {fn.__name__: fn for fn in ALL_TOOLS}

    def search_tools(query: str, top_k: int = max_top_k) -> str:
        """Find available tools by name or description.

        Tool names follow the pattern {service}_{action} (e.g., salesforce_query,
        gmail_send_email, slack_send_channel_message).

        Uses BM25 keyword-based relevance search. Works with service names,
        action words, or multi-word queries.
        Examples: "salesforce", "send email", "update deal", "slack channel"

        Returns at most MAX tools, one block each:
        `name(param: type=default, ...) — description`. `*` marks a required
        parameter. Pass arguments to execute_tool as a JSON object using exactly
        these parameter names. If the tool you need is not listed, search again
        with different keywords rather than a larger top_k.

        Args:
            query: Search query — service names, keywords, or a description.
            top_k: Maximum number of results to return (default: MAX, max MAX).
        """
        results = _get_registry().bm25(query, top_k=min(top_k, max_top_k))
        return format_search_results(results, tools)

    search_tools.__doc__ = (search_tools.__doc__ or "").replace("MAX", str(max_top_k))
    return search_tools
