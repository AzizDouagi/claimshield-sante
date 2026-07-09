"""Client HTTP fin vers l'API ClaimShield Santé (``api/main.py``) — l'UI
Chainlit est un processus séparé, jamais un accès direct au graphe ou aux
agents (même garantie que ``tests/api/test_main.py::TestNoDirectAgentAccess``,
appliquée ici par construction : ce module n'importe jamais ``graph.*`` ni
``agents.*``, uniquement des requêtes HTTP).
"""
from __future__ import annotations

from typing import Any

import httpx

from config.settings import get_settings

# POST /claims et POST .../human-decision déclenchent l'exécution réelle de
# plusieurs agents en séquence, chacun appelant le LLM (Ollama) — sur du
# matériel modeste, la chaîne complète jusqu'à la prochaine interruption
# peut dépasser largement une minute. Timeout volontairement généreux
# (10 min) plutôt qu'un échec prématuré côté UI alors que l'API travaille
# toujours réellement.
_PIPELINE_TIMEOUT_SECONDS = 600.0


def _base_url() -> str:
    return get_settings().claimshield_api_base_url.rstrip("/")


def _auth_headers() -> dict[str, str]:
    return {"X-API-Key": get_settings().claimshield_api_key.get_secret_value()}


async def healthz() -> bool:
    async with httpx.AsyncClient(base_url=_base_url(), timeout=10.0) as client:
        response = await client.get("/healthz")
        return response.status_code == 200


async def submit_claim(
    case_id: str,
    source_path: str,
    *,
    role: str = "ADMINISTRATIVE_MANAGER",
) -> httpx.Response:
    async with httpx.AsyncClient(base_url=_base_url(), timeout=_PIPELINE_TIMEOUT_SECONDS) as client:
        return await client.post(
            "/claims",
            json={"case_id": case_id, "source_path": source_path, "role": role},
            headers=_auth_headers(),
        )


async def get_status(case_id: str) -> httpx.Response:
    async with httpx.AsyncClient(base_url=_base_url(), timeout=30.0) as client:
        return await client.get(f"/claims/{case_id}")


async def submit_human_decision(
    case_id: str,
    *,
    actor: str,
    action: str,
    justification: str,
    target_node: str | None = None,
) -> httpx.Response:
    payload: dict[str, Any] = {"actor": actor, "action": action, "justification": justification}
    if target_node is not None:
        payload["target_node"] = target_node
    async with httpx.AsyncClient(base_url=_base_url(), timeout=_PIPELINE_TIMEOUT_SECONDS) as client:
        return await client.post(
            f"/claims/{case_id}/human-decision",
            json=payload,
            headers=_auth_headers(),
        )
