"""Fraud Detection Agent — interface injectable — ClaimShield Santé.

Cet agent n'est pas encore implémenté.  Ce fichier fournit :
  - ``FraudDetectionRunnable`` — Protocol structural.
  - ``_NotImplementedStub`` — retourne NOT_EVALUATED, risk_score=0.0.
  - ``make_node(impl)`` — factory LangGraph injectable.
  - ``node`` — nœud par défaut utilisant le stub.

Le stub ne pose aucun diagnostic de fraude et ne bloque pas le pipeline.
"""
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from schemas.domain import VerificationStatus
from schemas.results import FraudDetectionResult
from state.claim_state import ClaimState, validate_state_update

_STEP_NAME = "fraud_detection"
_AGENT_NAME = "fraud_detection_agent"


# ── Interface ──────────────────────────────────────────────────────────────────


@runtime_checkable
class FraudDetectionRunnable(Protocol):
    """Interface minimale requise par le nœud LangGraph."""

    def run(self, state: ClaimState) -> FraudDetectionResult: ...


# ── Stub par défaut ────────────────────────────────────────────────────────────


class _NotImplementedStub:
    """Placeholder retournant NOT_EVALUATED — aucun signal de fraude inventé."""

    def run(self, state: ClaimState) -> FraudDetectionResult:
        case_id = str(state.get("case_id", "UNKNOWN"))
        return FraudDetectionResult(
            case_id=case_id,
            status=VerificationStatus.NOT_EVALUATED,
            risk_score=0.0,
            reasons=["[stub] fraud_detection_agent non implémenté — résultat non évalué."],
        )


_DEFAULT_IMPL: FraudDetectionRunnable = _NotImplementedStub()


# ── Factory et nœud LangGraph ─────────────────────────────────────────────────


def make_node(
    impl: FraudDetectionRunnable = _DEFAULT_IMPL,
) -> Callable[[ClaimState], dict]:
    """Crée un nœud LangGraph avec l'implémentation injectable fournie."""
    def _node(state: ClaimState) -> dict:
        result = impl.run(state)
        updates: dict = {
            "fraud_result": result,
            "current_step": _STEP_NAME,
            "completed_steps": [_STEP_NAME],
        }
        if result.status is VerificationStatus.FAIL:
            updates["errors"] = [
                f"[{_AGENT_NAME}] {r}" for r in result.reasons
            ]
        elif result.status in (VerificationStatus.NEEDS_REVIEW, VerificationStatus.NOT_EVALUATED):
            updates["alerts"] = [
                f"Détection fraude : {result.status.value} — score={result.risk_score:.2f}"
            ]
        validate_state_update(updates)
        return updates

    _node.__name__ = f"node_{_STEP_NAME}"
    return _node


# Nœud stable — nom utilisé comme clé dans le StateGraph.
node = make_node()
