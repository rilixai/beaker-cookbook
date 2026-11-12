"""Toolathlon history and context management using public Agents SDK hooks."""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from agents import RunHooks
from agents.run_config import ModelInputData


class Conversation(RunHooks):
    def __init__(self, workspace: Path, instruction: str, context_limit: int) -> None:
        self.history = workspace.parent / "conversation_histories/session_history.jsonl"
        self.history.parent.mkdir(parents=True, exist_ok=True)
        self.meta: dict[str, Any] = {
            "current_turn": 0,
            "total_turns_ever": 1,
            "turns_in_current_sequence": 1,
            "mini_turns_in_current_sequence": 1,
            "boundary_in_current_sequence": [(0, 1)],
            "truncated_turns": 0,
            "truncation_history": [],
            "started_at": datetime.now().isoformat(),
        }
        self.context: dict[str, Any] = {
            "_agent_workspace": str(workspace),
            "_session_id": "session",
            "_history_dir": str(self.history.parent),
            "_context_limit": context_limit,
            "_context_meta": self.meta,
        }
        self.messages: list[dict[str, Any]] = [{"role": "user", "content": instruction}]
        self.stats = {
            "interaction_turns": 1,
            "tool_calls": 0,
            "agent_llm_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
        }
        self.attempts = 0
        self.step = 0
        self.sizes = [1]
        self.retained = [0]
        self.history.write_text(json.dumps({"turn": 0, "type": "user_input", "content": instruction}) + "\n")

    def _record(self, kind: str, raw: Any) -> None:
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump(mode="json")
        record = {
            "in_turn_steps": self.step,
            "turn": self.meta["current_turn"],
            "timestamp": datetime.now().isoformat(),
            "agent": "Assistant",
            "item_type": kind,
            "raw_content": raw,
        }
        with self.history.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.step += 1

    def _extend(self, count: int) -> None:
        self.sizes[-1] += count
        start, end = self.meta["boundary_in_current_sequence"][-1]
        self.meta["boundary_in_current_sequence"][-1] = (start, end + count)
        self.meta["mini_turns_in_current_sequence"] += count

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        self.attempts += 1
        self.step = 0
        self.sizes.append(0)
        self.retained.append(len(self.sizes) - 1)
        for key in ("current_turn", "total_turns_ever", "turns_in_current_sequence"):
            self.meta[key] += 1
        start = self.meta["mini_turns_in_current_sequence"]
        self.meta["boundary_in_current_sequence"].append((start, start))

    async def on_llm_end(self, context, agent, response) -> None:
        usage = response.usage
        self.stats["agent_llm_requests"] += 1
        for key in ("input_tokens", "output_tokens"):
            self.stats[key] += getattr(usage, key)
        self.stats["cached_input_tokens"] += getattr(usage.input_tokens_details, "cached_tokens", 0) or 0
        self.context["_last_response_usage"] = {
            key: getattr(usage, key) for key in ("total_tokens", "input_tokens", "output_tokens", "requests")
        }
        self.context["_cumulative_usage"] = dict(self.stats)
        self._extend(len(response.output))
        text, calls = [], []
        for item in response.output:
            if item.type == "message":
                text.extend(part.text for part in item.content if part.type == "output_text")
                self._record("message_output_item", item)
            elif item.type == "function_call":
                calls.append(
                    {
                        "id": item.call_id,
                        "type": "function",
                        "function": {"name": item.name, "arguments": item.arguments},
                    }
                )
                self._record("tool_call_item", item)
            else:
                self._record(f"{item.type}_item", item)
        self.messages.append({"role": "assistant", "content": "\n".join(text) or None, "tool_calls": calls})

    async def on_tool_end(self, context, agent, tool, result) -> None:
        self.stats["tool_calls"] += 1
        self._extend(1)
        output = result if isinstance(result, str) else json.dumps(result, default=str)
        call_id = context.tool_call_id
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": output})
        self._record("tool_call_output_item", {"type": "function_call_output", "call_id": call_id, "output": output})

    def filter_input(self, data) -> ModelInputData:
        items = data.model_data.input
        # SDK-generated error outputs (e.g. an unknown tool) need not invoke a
        # tool hook. Include them in the preceding turn's actual input size.
        delta = len(items) - sum(self.sizes)
        if delta < 0:
            raise ValueError("Unexpected Agents SDK conversation size")
        if delta:
            self._extend(delta)
        chunks, offset = [], 0
        for size in self.sizes:
            chunks.append(items[offset : offset + size])
            offset += size
        if pending := self.context.pop("_pending_truncate", None):
            protected = self.retained[:1] if pending.get("preserve_first_user_input", True) else []
            eligible = self.retained[len(protected) :]
            method, value = pending["method"], pending.get("value", 0)
            if method == "smart_ranges":
                selected = {index for start, end in pending["ranges"] for index in range(start, end + 1)}
                kept = [turn for index, turn in enumerate(self.retained) if index in selected or turn in protected]
            else:
                count = len(eligible)
                if method == "keep_recent_turns":
                    count = min(int(value), count)
                elif method == "keep_recent_percent":
                    count = max(1, int(count * value / 100)) if count else 0
                elif method == "delete_first_turns":
                    count = max(0, count - int(value))
                elif method == "delete_first_percent":
                    count = max(0, count - int(count * value / 100))
                kept = protected + (eligible[-count:] if count else [])
            removed = len(self.retained) - len(kept)
            if removed:
                self.meta["truncated_turns"] += removed
                self.meta["truncation_history"].append(
                    {
                        "at_turn": self.meta["current_turn"],
                        "method": method,
                        "deleted_turns": removed,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                self.context["_context_truncated"] = True
            self.retained = kept
        filtered, boundaries = [], []
        for index in self.retained:
            start = len(filtered)
            filtered.extend(chunks[index])
            boundaries.append((start, len(filtered)))
        self.meta.update(
            turns_in_current_sequence=len(boundaries),
            mini_turns_in_current_sequence=len(filtered),
            boundary_in_current_sequence=boundaries,
        )
        return ModelInputData(input=filtered, instructions=data.model_data.instructions)

    def reset(self, instruction: str) -> list[dict[str, str]]:
        with self.history.open() as stream:
            summary = "".join(deque(stream, maxlen=10))[-20000:]
        message = f"[Context reset]\nOriginal task:\n{instruction}\n\nRecent history:\n{summary}"
        self.meta["truncation_history"].append(
            {
                "at_turn": self.meta["current_turn"],
                "method": "context_reset",
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.meta["current_turn"] += 1
        self.meta["total_turns_ever"] += 1
        self.sizes, self.retained = [1], [0]
        self.meta.update(
            turns_in_current_sequence=1, mini_turns_in_current_sequence=1, boundary_in_current_sequence=[(0, 1)]
        )
        self.context.pop("_pending_truncate", None)
        self.messages.append({"role": "user", "content": message})
        with self.history.open("a") as stream:
            stream.write(
                json.dumps({"turn": self.meta["current_turn"], "type": "user_input", "content": message}) + "\n"
            )
        return [{"role": "user", "content": message}]
