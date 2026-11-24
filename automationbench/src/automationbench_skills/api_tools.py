"""Discovery metadata for fields supported by the native API routes."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import replace
from typing import Any


def expose_contact_phone_fields(api: Any) -> None:
    """Distinguish office, mobile and fax numbers in Contact update discovery."""
    endpoint_id = "salesforce.sobjects.contact.update"
    endpoint = api._search_endpoints.get(endpoint_id)
    if endpoint is None or endpoint.body_schema is None:
        return
    body = deepcopy(endpoint.body_schema)
    properties = body.setdefault("properties", {})
    properties.setdefault("Phone", {})["description"] = (
        "Office telephone number. Use MobilePhone for a mobile or cell number and Fax for a fax number."
    )
    properties.setdefault("MobilePhone", {
        "description": "Mobile or cell telephone number; separate from the office Phone field.",
    })
    properties.setdefault("Fax", {
        "description": "Fax number; separate from the voice Phone field.",
    })
    updated = replace(endpoint, body_schema=body)
    api._search_endpoints[endpoint_id] = updated
    if endpoint_id in api.endpoints:
        api.endpoints[endpoint_id] = updated


def explain_salesforce_fields(api: Any) -> None:
    """Describe supported fields whose business meanings are easy to confuse."""
    descriptions = {
        "salesforce.sobjects.opportunity.update": {
            "Type": "Opportunity business type or category, such as new or existing business. Separate from StageName.",
            "StageName": "Sales pipeline stage. For an opportunity type/category update, use Type instead.",
        },
        "salesforce.sobjects.task.create": {
            "WhoId": "Exact existing Contact or Lead ID for the person this task is about. Copy the supplied or retrieved ID unchanged.",
            "WhatId": "Related non-person record ID, such as an Account or Opportunity. A Contact or Lead person belongs in WhoId.",
        },
    }
    for endpoint_id, fields in descriptions.items():
        endpoint = api._search_endpoints.get(endpoint_id)
        if endpoint is None or endpoint.body_schema is None:
            continue
        body = deepcopy(endpoint.body_schema)
        properties = body.setdefault("properties", {})
        for field, description in fields.items():
            properties.setdefault(field, {})["description"] = description
        updated = replace(endpoint, body_schema=body)
        api._search_endpoints[endpoint_id] = updated
        if endpoint_id in api.endpoints:
            api.endpoints[endpoint_id] = updated


def explain_confluence_fields(api: Any) -> None:
    """Keep the destination space distinct from its containing cloud site."""
    endpoint_id = "confluence.pages.create"
    endpoint = api._search_endpoints.get(endpoint_id)
    if endpoint is None or endpoint.body_schema is None:
        return
    body = deepcopy(endpoint.body_schema)
    properties = body.setdefault("properties", {})
    properties.setdefault("spaceId", {})["description"] = (
        "Destination Confluence space ID. Copy the specified space ID here, not the cloud/site ID."
    )
    properties.setdefault("cloudId", {})["description"] = (
        "Confluence cloud/site ID, separate from spaceId. Copy it here when supplied."
    )
    updated = replace(endpoint, body_schema=body)
    api._search_endpoints[endpoint_id] = updated
    if endpoint_id in api.endpoints:
        api.endpoints[endpoint_id] = updated


def prefer_requested_endpoints(api: Any) -> None:
    """Honor a named endpoint or service/resource/operation before text ranking."""
    native_search = api.search
    exact = {}
    operations: dict[tuple[str, ...], list[str]] = {}
    for endpoint_id in api._search_endpoints:
        terms = tuple(re.findall(r"[a-z0-9_]+", endpoint_id.lower()))
        exact[terms] = endpoint_id
        if len(terms) >= 3:
            operations.setdefault((terms[0], terms[-2], terms[-1]), []).append(endpoint_id)
    for matches in operations.values():
        matches.sort(key=lambda endpoint_id: (len(endpoint_id.split(".")), endpoint_id))

    def search(query: str, top_k: int = 5) -> dict[str, Any]:
        payload = native_search(query, top_k)
        terms = tuple(re.findall(r"[a-z0-9_]+", query.lower()))
        if endpoint_id := exact.get(terms):
            preferred = [endpoint_id]
        else:
            preferred = operations.get(terms, [])
        if not preferred or top_k <= 0:
            return payload
        results = []
        for endpoint_id in preferred[:top_k]:
            endpoint = api._search_endpoints[endpoint_id]
            result = {
                "endpoint": endpoint.id,
                "description": endpoint.description,
                "path": endpoint.path_schema,
                "params": endpoint.params_schema,
                "response": endpoint.response,
            }
            if endpoint.body_schema is not None:
                result["body"] = endpoint.body_schema
            results.append(result)
        selected = {result["endpoint"] for result in results}
        results.extend(result for result in payload["results"] if result["endpoint"] not in selected)
        return {**payload, "results": results[:top_k], "count": len(results[:top_k])}

    api.search = search
