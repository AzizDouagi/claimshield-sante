"""Outils internes du Chat Reasoning Agent — chat/tools.py (plan V2 §6).

4 outils livrés en Phase V2-11a — `get_claim_context`/`run_claim_analysis`/
`explain_claim`/`recommend_corrections` — sont des wrappers HTTP minces vers
`/v2/*`, jamais un accès direct à `graph.*`/`agents.*` (vérifié
statiquement, voir `tests/v2/chat/test_tools.py`), même garantie que
`ui/api_client_v2.py`.

`simulate_changes` (Phase V2-11b) et `get_audit_summary` (Phase V2-11c,
nouveau) font exception à cette règle, documentée dans leurs modules
respectifs (`chat/simulation_engine.py`/`chat/audit_reader.py`, seuls
modules de `chat/` à accéder directement à `graph.*`/`services.*` métier —
un appel HTTP ne peut structurellement pas suffire à une sandbox de
simulation réelle, ni l'audit n'est encore exposé par aucun endpoint `/v2/*`).
`generate_patient_message` (Phase V2-11c, nouveau) reste un wrapper HTTP
mince classique — même source de données que `get_claim_context`, seule la
composition finale (`chat/response_composer.py`) change de destinataire.

`_build_client()` est le point d'injection pour les tests : monkeypatché
vers un client `httpx.AsyncClient(transport=httpx.ASGITransport(app=...))`
pour cibler une application de test en mémoire, jamais un vrai serveur —
même principe d'injection que `llm.factory.get_llm()` ailleurs dans le
projet.
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx

from chat.audit_reader import build_audit_summary
from chat.correction_engine import build_corrections
from chat.explanation_engine import build_explanation_facts
from chat.schemas import (
    AuditSummary,
    CorrectionRecommendation,
    ExplanationFacts,
    SimulationChangeRequest,
    SimulationResult,
)
from chat.simulation_engine import run_simulation
from config.settings import get_settings
from services.audit_service import AuditService

__all__ = [
    "explain_claim",
    "generate_patient_message",
    "get_audit_summary",
    "get_claim_context",
    "recommend_corrections",
    "run_claim_analysis",
    "simulate_changes",
]

_DEFAULT_TIMEOUT_SECONDS = 30.0


def _build_client() -> httpx.AsyncClient:
    settings = get_settings()
    return httpx.AsyncClient(
        base_url=settings.claimshield_api_base_url.rstrip("/") + "/v2",
        headers={"X-API-Key": settings.claimshield_api_key.get_secret_value()},
        timeout=_DEFAULT_TIMEOUT_SECONDS,
    )


async def get_claim_context(case_id: str) -> dict | None:
    """`GET /v2/claims/{case_id}` — état minimisé déjà produit par l'API
    (`api.v2.schemas.ClaimStatusResponseV2`). `None` si le dossier est
    introuvable (404) — jamais une exception propagée à l'appelant pour ce
    cas attendu."""
    async with _build_client() as client:
        response = await client.get(f"/claims/{case_id}")
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


async def run_claim_analysis(case_id: str) -> dict | None:
    """Intention ANALYZE — même source de données que `get_claim_context`
    (l'API v2 n'expose pas de vue d'analyse distincte) ; la mise en forme
    « synthèse » se fait exclusivement dans `chat/response_composer.py`,
    jamais un second calcul ici."""
    return await get_claim_context(case_id)


async def explain_claim(case_id: str) -> ExplanationFacts | None:
    """Intention EXPLAIN — `None` si le dossier est introuvable."""
    context = await get_claim_context(case_id)
    if context is None:
        return None
    return build_explanation_facts(context)


async def recommend_corrections(case_id: str) -> list[CorrectionRecommendation]:
    """Intention CORRECT — liste vide (jamais `None`) si le dossier est
    introuvable ou si aucun motif de correction n'a été identifié."""
    context = await get_claim_context(case_id)
    if context is None:
        return []
    return build_corrections(context)


async def simulate_changes(
    case_id: str,
    changes: SimulationChangeRequest,
    *,
    compiled_graph: Any | None = None,
) -> SimulationResult:
    """Intention SIMULATE — délègue à `chat.simulation_engine.run_simulation`
    (sandbox réelle, jamais une estimation heuristique). Opération coûteuse
    (réinvoque le pipeline complet, jusqu'à 5 appels LLM réels) — exécutée
    dans un thread séparé (`asyncio.to_thread`) pour ne jamais bloquer la
    boucle d'événements pendant sa durée. `compiled_graph` injectable
    (tests uniquement — jamais utilisé en production, où `None` construit
    l'instance depuis les paramètres d'environnement)."""
    return await asyncio.to_thread(
        run_simulation, case_id, changes, compiled_graph=compiled_graph
    )


async def get_audit_summary(
    case_id: str, *, audit_service: AuditService | None = None
) -> AuditSummary:
    """Intention AUDIT — délègue à `chat.audit_reader.build_audit_summary`
    (résumé minimisé, jamais le contenu brut d'un `outcome` d'événement).
    `audit_service` injectable (tests uniquement, voir la limite
    opérationnelle documentée dans `chat/audit_reader.py`)."""
    return await asyncio.to_thread(build_audit_summary, case_id, audit_service=audit_service)


async def generate_patient_message(case_id: str) -> dict | None:
    """Intention DRAFT_MESSAGE — même source de données que
    `get_claim_context` (aucune vue dédiée côté API) ; le changement de
    destinataire (patient plutôt que gestionnaire) est géré exclusivement
    par `chat/response_composer.py` (prompt dédié), jamais un second appel
    ou un calcul supplémentaire ici."""
    return await get_claim_context(case_id)
