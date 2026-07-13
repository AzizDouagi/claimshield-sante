"""Nœud technique terminal du graphe V2 — graph/technical_nodes_v2.py.

Un seul nœud technique en V2 (`finalize`) — contrairement à V1
(`graph/technical_nodes.py`, 7 nœuds dont `node_await_human_review` via
`interrupt()`) : le graphe V2 ne bloque jamais, il n'y a donc ni
quarantaine intermédiaire ni interruption dynamique à modéliser ici (voir
plan V2 §2 — décision AZIZ « override asynchrone optionnel »,
`services/override_store.py`, hors de ce graphe).
"""
from __future__ import annotations

from schemas.domain import ClaimDecisionV2, IntakeSafetyStatus
from state.claim_state_v2 import ClaimStateV2

__all__ = ["node_finalize_v2"]

_INTAKE_STATUS_TO_DECISION: dict[IntakeSafetyStatus, ClaimDecisionV2] = {
    IntakeSafetyStatus.BLOCKED: ClaimDecisionV2.REJECT,
    IntakeSafetyStatus.QUARANTINED: ClaimDecisionV2.QUARANTINE,
    IntakeSafetyStatus.TECHNICAL_FAILURE: ClaimDecisionV2.TECHNICAL_FAILURE,
}


def node_finalize_v2(state: ClaimStateV2) -> dict:
    """Clôture le pipeline — ne recalcule jamais une décision déjà posée par
    `autonomous_decision_agent`. Sur le chemin court-circuité depuis
    `intake_safety` (BLOCKED/QUARANTINED/TECHNICAL_FAILURE, `autonomous_decision`
    jamais atteint), dérive `final_decision` du statut d'admission — seule
    correspondance déterministe possible à ce stade, jamais une valeur
    inventée."""
    updates: dict = {"current_step": "finalize", "completed_steps": ["finalize"]}

    if state.get("final_decision") is not None:
        return updates

    intake_safety_result = state.get("intake_safety_result")
    if intake_safety_result is not None:
        decision = _INTAKE_STATUS_TO_DECISION.get(
            intake_safety_result.status, ClaimDecisionV2.TECHNICAL_FAILURE
        )
        updates["final_decision"] = decision

    return updates
