"""Audit Agent — interface injectable — ClaimShield Santé.

Cet agent n'est pas encore implémenté.  Ce fichier fournit :
  - ``AuditAgentRunnable`` — Protocol structural.
  - ``_NotImplementedStub`` — synthétise AuditResult depuis audit_trail existant.
  - ``make_node(impl)`` — factory LangGraph injectable.
  - ``node`` — nœud par défaut utilisant le stub.

Le stub est le seul des quatre stubs à produire un résultat partiellement
utile : il compte les événements déjà présents dans audit_trail sans en
inventer de nouveaux.  Aucune donnée métier n'est fabriquée.
"""
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from schemas.domain import VerificationStatus
from schemas.results import AuditEvent, AuditResult
from state.claim_state import ClaimState, validate_state_update

_STEP_NAME = "audit"
_AGENT_NAME = "audit_agent"


# ── Interface ──────────────────────────────────────────────────────────────────


@runtime_checkable
class AuditAgentRunnable(Protocol):
    """Interface minimale requise par le nœud LangGraph."""

    def run(self, state: ClaimState) -> AuditResult: ...


# ── Stub par défaut ────────────────────────────────────────────────────────────


class _NotImplementedStub:
    """Placeholder qui reflète le audit_trail existant sans écriture réelle.

    Compte les AuditEvent déjà présents et retourne NOT_EVALUATED.
    Aucun nouveau journal n'est créé ; aucune persistance n'est effectuée.
    """

    def run(self, state: ClaimState) -> AuditResult:
        case_id = str(state.get("case_id", "UNKNOWN"))
        existing: list = state.get("audit_trail") or []  # type: ignore[assignment]
        valid_events: list[AuditEvent] = [
            e for e in existing if isinstance(e, AuditEvent)
        ]
        return AuditResult(
            case_id=case_id,
            status=VerificationStatus.NOT_EVALUATED,
            events_count=len(valid_events),
            events=[],   # pas de re-injection dans le state — déjà dans audit_trail
            policy_version="1.0.0",
        )


_DEFAULT_IMPL: AuditAgentRunnable = _NotImplementedStub()


# ── Factory et nœud LangGraph ─────────────────────────────────────────────────


def make_node(
    impl: AuditAgentRunnable = _DEFAULT_IMPL,
) -> Callable[[ClaimState], dict]:
    """Crée un nœud LangGraph avec l'implémentation injectable fournie."""
    def _node(state: ClaimState) -> dict:
        result = impl.run(state)
        updates: dict = {
            "audit_result": result,
            "current_step": _STEP_NAME,
            "completed_steps": [_STEP_NAME],
        }
        if result.status is VerificationStatus.FAIL:
            updates["errors"] = [f"[{_AGENT_NAME}] Journalisation échouée."]
        validate_state_update(updates)
        return updates

    _node.__name__ = f"node_{_STEP_NAME}"
    return _node


# Nœud stable — nom utilisé comme clé dans le StateGraph.
node = make_node()
