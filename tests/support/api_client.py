"""Aides partagées pour piloter ``api.main`` depuis un ``TestClient`` — évite
de dupliquer la clé API et les appels HTTP entre ``tests/api/`` et
``tests/e2e/``.
"""
from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from config.settings import get_settings

API_KEY = get_settings().claimshield_api_key.get_secret_value()


def auth_headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY}


def submit_claim(
    client: TestClient,
    case_id: str,
    source_path: str,
    *,
    required_documents: list[str] | None = None,
) -> Any:
    return client.post(
        "/claims",
        json={
            "case_id": case_id,
            "source_path": source_path,
            "required_documents": required_documents or [],
        },
        headers=auth_headers(),
    )


def submit_human_decision(
    client: TestClient,
    case_id: str,
    *,
    actor: str,
    action: str,
    justification: str,
    target_node: str | None = None,
) -> Any:
    payload: dict[str, Any] = {"actor": actor, "action": action, "justification": justification}
    if target_node is not None:
        payload["target_node"] = target_node
    return client.post(
        f"/claims/{case_id}/human-decision",
        json=payload,
        headers=auth_headers(),
    )
