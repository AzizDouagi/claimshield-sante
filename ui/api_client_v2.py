"""Client HTTP fin vers l'API v2 (``api/v2/*``) — pipeline autonome, plan de
refonte V2 (Phase V2-9). Fichier séparé de ``ui/api_client.py`` (V1, non
modifié — ``ui/api_client.py`` n'est pas un point d'intégration listé au §0
du plan) plutôt qu'une extension : même garantie de coexistence stricte que
le reste de la V2.

Même principe que ``ui/api_client.py`` : requêtes HTTP pures, jamais
d'import de ``graph.*``/``agents.*`` (même garantie que
``tests/api/test_main.py::TestNoDirectAgentAccess`` côté V1, ici assurée
par construction).
"""
from __future__ import annotations

import httpx

from config.settings import get_settings

# Le graphe V2 est strictement séquentiel et ne s'interrompt jamais : une
# soumission traverse toujours les 5 agents jusqu'à une décision terminale
# en un seul appel HTTP — même timeout généreux que ``ui/api_client.py``
# (V1), pour la même raison (LLM local, matériel modeste).
_PIPELINE_TIMEOUT_SECONDS = 600.0

__all__ = ["get_status_v2", "send_chat_message_v2", "submit_claim_v2", "submit_override_v2"]


def _base_url() -> str:
    return get_settings().claimshield_api_base_url.rstrip("/") + "/v2"


def _auth_headers() -> dict[str, str]:
    return {"X-API-Key": get_settings().claimshield_api_key.get_secret_value()}


async def submit_claim_v2(
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


async def get_status_v2(case_id: str) -> httpx.Response:
    async with httpx.AsyncClient(base_url=_base_url(), timeout=30.0) as client:
        return await client.get(f"/claims/{case_id}")


async def send_chat_message_v2(
    message: str,
    *,
    case_id: str | None = None,
    thread_id: str | None = None,
    actor: str | None = None,
) -> httpx.Response:
    """``POST /v2/chat`` — Chat Reasoning Agent (``chat/agent.py``). ``case_id``
    optionnel : le contexte déjà connu de l'appelant (dossier affiché côté
    UI) prime toujours sur un identifiant que le NLU détecterait dans le
    texte (voir ``chat.agent.handle_message``).

    ``thread_id``/``actor`` (Phase 8, mémoire conversationnelle) : optionnels
    — la mémoire n'est activée côté API que si ``actor`` est fourni."""
    async with httpx.AsyncClient(base_url=_base_url(), timeout=_PIPELINE_TIMEOUT_SECONDS) as client:
        return await client.post(
            "/chat",
            json={"case_id": case_id, "message": message, "thread_id": thread_id, "actor": actor},
            headers=_auth_headers(),
        )


async def submit_override_v2(
    case_id: str,
    *,
    actor: str,
    action: str,
    justification: str,
) -> httpx.Response:
    async with httpx.AsyncClient(base_url=_base_url(), timeout=30.0) as client:
        return await client.post(
            f"/claims/{case_id}/override",
            json={"actor": actor, "action": action, "justification": justification},
            headers=_auth_headers(),
        )
