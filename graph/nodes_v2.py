"""Nœuds du graphe V2 non portés directement par un agent — graph/nodes_v2.py.

Les 5 agents métier (`intake_safety`, `document_understanding`,
`eligibility`, `medical_risk`, `autonomous_decision`) exposent chacun leur
propre `node(state) -> dict` (voir `agents/*/agent.py`) — câblés tels quels
dans `graph/workflow_v2.py`, sans couche d'adaptation supplémentaire
(contrairement à V1, `graph/nodes.py::_build_node`/`_AgentConfig`, devenue
inutile : chaque agent V2 lit et retourne déjà exactement la forme attendue
de `ClaimStateV2`).

Seul `audit_service` (service déterministe, pas un agent — voir
`services/audit_service.py`, Phase V2-0) a besoin d'un nœud dédié ici : il
n'a pas de fonction `node()` propre car ce n'est pas un module agent.
"""
from __future__ import annotations

from collections.abc import Callable

from schemas.audit import AuditEventType
from schemas.domain import ClaimDecisionV2, IntakeSafetyStatus
from services.audit_service import AuditService
from state.claim_state_v2 import ClaimStateV2

__all__ = ["make_audit_service_node"]

_TERMINAL_INTAKE_EVENT_TYPE: dict[IntakeSafetyStatus, AuditEventType] = {
    IntakeSafetyStatus.BLOCKED: AuditEventType.SECURITY_DECISION,
    IntakeSafetyStatus.QUARANTINED: AuditEventType.SECURITY_DECISION,
    IntakeSafetyStatus.TECHNICAL_FAILURE: AuditEventType.FAILURE,
}


def make_audit_service_node(
    audit_service: AuditService | None = None,
) -> Callable[[ClaimStateV2], dict]:
    """Crée le nœud `audit_service` — persiste un événement d'audit
    déterministe (aucun appel LLM, voir `services/audit_service.py`), que le
    dossier atteigne ce nœud directement depuis `intake_safety` (admission
    refusée/mise en quarantaine, court-circuit — voir
    `graph.edges_v2.route_intake_safety`) ou après `autonomous_decision`
    (chemin nominal).

    `audit_service` est injectable (tests) ; `None` construit une instance
    dédiée par défaut (jamais un singleton cache — même convention que
    `graph.nodes.build_orchestrator()`, V1).
    """
    service = audit_service if audit_service is not None else AuditService()

    def _node(state: ClaimStateV2) -> dict:
        case_id = str(state.get("case_id", "UNKNOWN"))
        decision_result = state.get("decision_result")
        intake_safety_result = state.get("intake_safety_result")

        if decision_result is not None:
            event = service.record(
                case_id=case_id,
                event_type=AuditEventType.FINAL_REPORT,
                actor="autonomous_decision_agent",
                outcome=decision_result.decision.value,
                agent_name="autonomous_decision_agent",
                evidence_ids=decision_result.evidence_ids,
                details={
                    "status": decision_result.status.value,
                    "bounded_by_count": str(len(decision_result.bounded_by)),
                },
            )
        elif intake_safety_result is not None:
            event_type = _TERMINAL_INTAKE_EVENT_TYPE.get(
                intake_safety_result.status, AuditEventType.FAILURE
            )
            event = service.record(
                case_id=case_id,
                event_type=event_type,
                actor="intake_safety_agent",
                outcome=intake_safety_result.status.value,
                agent_name="intake_safety_agent",
                details={"status": intake_safety_result.status.value},
            )
        else:
            event = service.record(
                case_id=case_id,
                event_type=AuditEventType.FAILURE,
                actor="audit_service",
                outcome=ClaimDecisionV2.TECHNICAL_FAILURE.value,
                details={"reason": "Aucun résultat exploitable à ce stade du pipeline."},
            )

        return {
            "audit_trail": [event],
            "current_step": "audit_service",
            "completed_steps": ["audit_service"],
        }

    return _node
