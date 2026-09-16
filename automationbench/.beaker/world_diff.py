"""Before/after diff of one simulated app between a task's initial and end world state.

``predicted`` on a failed check: what the agent did in the app the assertion
targets, as added / changed / removed records, bounded in size. Pure functions
over ``WorldState`` dumps; no model calls, no randomness.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from typing import Any

from automationbench.schema.world import WorldState
from beaker.sdk.utils import to_json_safe


_SERVICE_FIELDS = sorted((str(f) for f in WorldState.model_fields if f != "meta"), key=len, reverse=True)


def service_for(assertion_type: str) -> str | None:
    """WorldState service an assertion type targets (``gmail_message_sent_to`` -> ``gmail``).

    Falls back to the type's first token when it names an app split over several
    services (``facebook_page_post_exists`` -> ``facebook``).
    """
    for service in _SERVICE_FIELDS:
        if assertion_type == service or assertion_type.startswith(service + "_"):
            return service
    head = assertion_type.split("_", 1)[0]
    return head if any(service.startswith(head + "_") for service in _SERVICE_FIELDS) else None


def clip(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ``predicted`` on a failed check is a before/after diff of the app it targets.
# Records are reduced to their own fields and the diff is re-rendered with
# fewer records / shorter strings until it serializes under the byte cap.
_DIFF_MAX_RECORDS = 5  # per added/changed/removed list
_DIFF_MAX_STR_CHARS = 400
_DIFF_MAX_FIELDS = 40  # per record
_DIFF_MAX_LIST_ITEMS = 20  # per scalar list inside a record
_DIFF_MAX_BYTES = 8_000  # serialized ``predicted`` per check
_DIFF_SHRINK_STEPS = ((_DIFF_MAX_RECORDS, _DIFF_MAX_STR_CHARS), (3, 400), (2, 200), (1, 120), (0, 0))

_Record = dict[str, Any]  # flattened record: scalars and lists of scalars
_Collection = dict[str, _Record]  # record key -> flattened record


class _Authored:
    """What the task author wrote for one record collection, per ``_authored``."""

    def __init__(self) -> None:
        self.ids: set[str] = set()  # author-given ids
        self.key_fields: set[str] | None = None  # field names every authored record has; None when unknown
        self.fields_by_key: dict[str, set[str]] = {}  # record key (see ``_record_keys``) -> fields the author set


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _is_collection(value: Any) -> bool:
    """A non-empty list of mappings: the world's record collections (messages, contacts, rows)."""
    return isinstance(value, list) and bool(value) and all(isinstance(item, Mapping) for item in value)


def _flatten_record(record: Mapping[str, Any]) -> _Record:
    """The record's own fields: scalars and scalar lists, nested mappings flattened to dotted keys.

    Nested collections are left out; they are diffed under their own path.
    """
    flat: _Record = {}

    def walk(mapping: Mapping[str, Any], prefix: str) -> None:
        for key, raw in mapping.items():
            value = to_json_safe(raw)
            name = f"{prefix}{key}"
            if _is_scalar(value):
                flat[name] = value
            elif isinstance(value, list) and all(_is_scalar(item) for item in value):
                flat[name] = value
            elif isinstance(value, Mapping):
                walk(value, name + ".")

    walk(record, "")
    return flat


def _record_key(record: Mapping[str, Any]) -> str | None:
    """Records match across states by ``id``; ``None`` when they have no usable one."""
    record_id = record.get("id")
    return str(record_id) if isinstance(record_id, str | int) and not isinstance(record_id, bool) else None


def _authored(initial: Any, raw: Any, path: str, out: dict[str, _Authored]) -> None:
    """Per collection path, the ids and fields the task author wrote in ``raw``.

    ``initial`` is the same state after ``WorldState`` validation, which fills
    defaults and generates ids and timestamps for what the author left out.
    Only the author's values are stable across the initial and end worlds; the
    run builds its own ``WorldState`` and regenerates everything else.
    """
    if isinstance(initial, Mapping):
        raw_map = raw if isinstance(raw, Mapping) else {}
        for key, value in initial.items():
            _authored(value, raw_map.get(key), f"{path}.{key}" if path else str(key), out)
    elif _is_collection(initial):
        raw_list = raw if isinstance(raw, list) else []
        raw_records: list[Mapping[str, Any]] = [
            raw_list[i] if i < len(raw_list) and isinstance(raw_list[i], Mapping) else {} for i in range(len(initial))
        ]
        authored = out.setdefault(path, _Authored())
        fields_per_record = [set(_flatten_record(r)) for r in raw_records]
        for raw_record, fields in zip(raw_records, fields_per_record, strict=True):
            raw_id = _record_key(raw_record)
            if raw_id is not None:
                authored.ids.add(raw_id)
            authored.key_fields = fields if authored.key_fields is None else authored.key_fields & fields
        keyed = _record_keys(initial, authored)
        for (key, label, _), record, raw_record, fields in zip(
            keyed, initial, raw_records, fields_per_record, strict=True
        ):
            authored.fields_by_key[key] = fields
            _authored(record, raw_record, f"{path}[{label}]", out)


def _view(record: _Record, fields: set[str] | None) -> _Record:
    return record if fields is None else {k: v for k, v in record.items() if k in fields}


def _record_keys(records: list[Mapping[str, Any]], known: _Authored) -> list[tuple[str, str, _Record]]:
    """``(key, path label, flattened record)`` for each record of one collection.

    A record with an author-given ``id`` is keyed by it. Any other (no id, or one
    the schema generated) is keyed by its content on the fields every authored
    record has, plus an occurrence counter so duplicates stay distinct; its label
    is a short digest of that key, so nested paths under it are the same in the
    initial and end world regardless of list position.
    """
    seen: dict[str, int] = {}
    out: list[tuple[str, str, _Record]] = []
    for record in records:
        record_id = _record_key(record)
        flat = _flatten_record(record)
        if record_id is not None and record_id in known.ids:
            out.append((f"id:{record_id}", record_id, flat))
            continue
        content = {k: v for k, v in _view(flat, known.key_fields).items() if k != "id"}
        base = "json:" + json.dumps(content, sort_keys=True)
        seen[base] = seen.get(base, 0) + 1
        key = f"{base}#{seen[base]}"
        out.append((key, "~" + hashlib.sha1(key.encode()).hexdigest()[:8], flat))
    return out


def _collections(node: Any, path: str, authored: Mapping[str, _Authored]) -> Iterator[tuple[str, _Collection]]:
    """Every record collection under ``node`` by path (``gmail.messages``, ``zendesk.tickets[t1].comments``)."""
    if isinstance(node, Mapping):
        for key, value in node.items():
            yield from _collections(value, f"{path}.{key}" if path else str(key), authored)
    elif _is_collection(node):
        known = authored.get(path, _Authored())
        records: _Collection = {}
        for (key, label, flat), record in zip(_record_keys(node, known), node, strict=True):
            records[key] = flat
            yield from _collections(record, f"{path}[{label}]", authored)
        yield path, records


def _shrink(record: _Record, max_str: int) -> dict[str, Any]:
    kept: dict[str, Any] = {}
    for name, value in list(record.items())[:_DIFF_MAX_FIELDS]:
        if isinstance(value, str):
            kept[name] = clip(value, max_str)
        elif isinstance(value, list):
            kept[name] = [clip(v, max_str) if isinstance(v, str) else v for v in value[:_DIFF_MAX_LIST_ITEMS]]
            if len(value) > _DIFF_MAX_LIST_ITEMS:
                kept[name].append(f"… +{len(value) - _DIFF_MAX_LIST_ITEMS} more")
        else:
            kept[name] = value
    if len(record) > _DIFF_MAX_FIELDS:
        kept["truncated_fields"] = len(record) - _DIFF_MAX_FIELDS
    return kept


def _service_diff(
    service: str, fields: list[str], raw_initial: Mapping[str, Any], initial: Mapping[str, Any], end: Mapping[str, Any]
) -> dict[str, Any]:
    """Added / changed / removed records of one app between the initial and end world state.

    ``fields`` are the ``WorldState`` fields the app spans (``["gmail"]``, or
    ``["facebook_conversions", "facebook_lead_ads", "facebook_pages"]`` for
    ``facebook``); record paths start with the concrete field. ``initial`` and
    ``end`` are ``WorldState`` dumps; ``raw_initial`` is the task author's
    initial state, which decides what counts as a real change (see
    ``_authored``): a record present in both is changed when a field the author
    set on it differs.
    """
    authored: dict[str, _Authored] = {}
    before: dict[str, _Collection] = {}
    after: dict[str, _Collection] = {}
    for field in fields:
        _authored(initial.get(field), raw_initial.get(field), field, authored)
        before.update(_collections(initial.get(field), field, authored))
        after.update(_collections(end.get(field), field, authored))
    added: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for path in sorted(before.keys() | after.keys()):
        old, new = before.get(path, {}), after.get(path, {})
        fields_by_key = authored[path].fields_by_key if path in authored else {}
        added.extend({"path": path, "record": new[k]} for k in new if k not in old)
        for key, authored_fields in fields_by_key.items():
            if key in old and key in new and _view(old[key], authored_fields) != _view(new[key], authored_fields):
                changed.append({"path": path, "before": old[key], "after": new[key]})
        removed.extend({"path": path, "record": old[k]} for k in old if k not in new)

    def render(entries: list[dict[str, Any]], max_records: int, max_str: int) -> list[dict[str, Any]]:
        kept = [
            {k: _shrink(v, max_str) if isinstance(v, dict) else v for k, v in entry.items()}
            for entry in entries[:max_records]
        ]
        if len(entries) > max_records:
            kept.append({"truncated": len(entries) - max_records})
        return kept

    lists = {"added": added, "changed": changed, "removed": removed}
    if not any(lists.values()):
        return {"service": service, "added": [], "changed": [], "removed": []}
    for max_records, max_str in _DIFF_SHRINK_STEPS:
        diff: dict[str, Any] = {"service": service}
        diff.update({k: render(v, max_records, max_str) for k, v in lists.items() if v})
        if len(json.dumps(diff, ensure_ascii=False).encode("utf-8")) <= _DIFF_MAX_BYTES:
            break
    return diff


class ServiceDiffs:
    """Per-service diff of the case's world, computed once and shared by every check targeting that app."""

    def __init__(
        self, raw_initial: Mapping[str, Any], initial_world: WorldState | None, end_state: Mapping[str, Any]
    ) -> None:
        self._raw_initial = raw_initial
        self._initial: Mapping[str, Any] = initial_world.model_dump(mode="json") if initial_world is not None else {}
        self._end = end_state
        self._cache: dict[str, dict[str, Any] | None] = {}

    def get(self, service: str | None) -> dict[str, Any] | None:
        if service is None:
            return None
        if service not in self._cache:
            spanned = (
                [service]
                if service in _SERVICE_FIELDS
                else [f for f in _SERVICE_FIELDS if f.startswith(service + "_")]
            )
            fields = sorted(f for f in spanned if f in self._raw_initial or f in self._end)
            self._cache[service] = (
                _service_diff(service, fields, self._raw_initial, self._initial, self._end) if fields else None
            )
        return self._cache[service]
