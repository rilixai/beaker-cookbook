"""Readable Gmail message arguments, encoded before the native API boundary."""

from __future__ import annotations

import base64
import json
from copy import deepcopy
from dataclasses import replace
from typing import Any

from mcp.types import CallToolResult, TextContent


_ENDPOINTS = {
    "gmail.users.messages.send",
    "gmail.users.drafts.create",
    "gmail.users.drafts.update",
}
_FIELDS = {"to", "subject", "text", "body", "cc", "bcc", "threadId"}
_DESCRIPTION = (
    "For a plain-text email, supply to, subject and text as readable strings "
    "(body is also accepted instead of text). The harness encodes UTF-8; do not "
    "base64-encode these fields. Choose verified recipients and complete content "
    "first. The selected endpoint still determines send versus unsent draft. "
    "Use the native raw/payload representation for HTML, attachments or mail without To."
)


def _message_schema() -> dict[str, Any]:
    address = {"anyOf": [
        {"type": "string"},
        {"type": "array", "items": {"type": "string"}},
    ]}
    return {
        "type": "object",
        "properties": {
            "to": deepcopy(address),
            "subject": {"type": "string"},
            "text": {"type": "string", "minLength": 1, "description": "Complete readable message body, with actual newlines."},
            "body": {"type": "string", "minLength": 1, "description": "Alias for text; supply only one of text or body."},
            "cc": deepcopy(address),
            "bcc": deepcopy(address),
            "threadId": {"type": "string", "minLength": 1},
        },
        "required": ["to", "subject"],
        "oneOf": [{"required": ["text"]}, {"required": ["body"]}],
        "additionalProperties": False,
    }


def _body_schema(endpoint: str, native: dict[str, Any]) -> dict[str, Any]:
    plain = _message_schema()
    if endpoint != "gmail.users.messages.send":
        plain = {
            "type": "object", "properties": {"message": plain},
            "required": ["message"], "additionalProperties": False,
        }
    # anyOf preserves the open native draft schema, including valid overlapping
    # requests. Execution recognizes plaintext only by its explicit string field.
    return {"description": _DESCRIPTION, "anyOf": [plain, deepcopy(native)]}


def expose_gmail_plaintext(api: Any) -> None:
    """Extend discovery only; the execution contract still validates native JSON."""
    for endpoint_id in _ENDPOINTS & api.endpoints.keys():
        endpoint = api._search_endpoints.get(endpoint_id)
        if endpoint is not None and endpoint.body_schema is not None:
            api._search_endpoints[endpoint_id] = replace(
                endpoint, body_schema=_body_schema(endpoint_id, endpoint.body_schema)
            )


def gmail_fetch_schema(native: dict[str, Any]) -> dict[str, Any]:
    """Keep the complete endpoint union, adding plaintext to Gmail writes only."""
    schema = deepcopy(native)
    for group in schema.get("allOf", []):
        for branch in group.get("oneOf", []):
            props = branch.get("properties", {})
            endpoint = props.get("endpoint", {}).get("const")
            if endpoint in _ENDPOINTS and isinstance(props.get("body"), dict):
                props["body"] = _body_schema(endpoint, props["body"])
    return schema


def _addresses(value: Any, field: str) -> str:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return ", ".join(value)
    if isinstance(value, str):
        return value
    raise ValueError(f"{field} must be a string or a list of strings")


def prepare_gmail_body(endpoint: str, body: Any) -> Any:
    """Translate explicit plaintext; leave native/other requests untouched."""
    if endpoint not in _ENDPOINTS or not isinstance(body, dict):
        return body
    is_draft = endpoint != "gmail.users.messages.send"
    message = body.get("message") if is_draft else body
    if not isinstance(message, dict) or "raw" in message or "payload" in message:
        return body
    if not ({"text", "body"} & message.keys()):
        return body
    if message.keys() - _FIELDS or (is_draft and body.keys() != {"message"}):
        raise ValueError("Plaintext messages accept only to, subject, text (or body), cc, bcc and threadId")
    if ("text" in message) == ("body" in message):
        raise ValueError("Supply exactly one of text or body")
    text = message.get("text", message.get("body"))
    if not isinstance(text, str) or not isinstance(message.get("subject"), str):
        raise ValueError("Plaintext text/body and subject must be strings")
    if not text:
        raise ValueError("Plaintext text/body must be nonempty")
    to = _addresses(message.get("to"), "to")
    if not to.strip():
        raise ValueError("Plaintext messages require To; use native raw/payload for mail without To")
    headers = [{"name": "To", "value": to}, {"name": "Subject", "value": message["subject"]}]
    for field, name in (("cc", "Cc"), ("bcc", "Bcc")):
        if field in message:
            headers.append({"name": name, "value": _addresses(message[field], field)})
    native_message: dict[str, Any] = {"payload": {
        "mimeType": "text/plain", "headers": headers,
        "body": {"data": base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")},
    }}
    if "threadId" in message:
        if not isinstance(message["threadId"], str) or not message["threadId"]:
            raise ValueError("threadId must be a nonempty string")
        native_message["threadId"] = message["threadId"]
    return {"message": native_message} if is_draft else native_message


def plaintext_error(endpoint: str, error: ValueError) -> CallToolResult:
    payload = {"error": {"type": "invalid_request", "endpoint": endpoint, "message": str(error)}}
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))],
        structured_content=payload, is_error=True,
    )
