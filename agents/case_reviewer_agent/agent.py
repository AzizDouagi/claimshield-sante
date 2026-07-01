"""Case Reviewer Agent — interface injectable — ClaimShield Santé.

Cet agent n'est pas encore implémenté.  Ce fichier fournit :
  - ``CaseReviewerRunnable`` — Protocol structural.
  - ``_NotImplementedStub`` — retourne Recommendation.PENDING, human_review_required=True.
  - ``make_node(impl)`` — factory LangGraph injectable.
  - ``node`` — nœud par défaut utilisant le stub.

Le stub retourne PENDING (jamais APPROVE ou REJECT) afin de ne pas prendre
de décision métier fictive sur le remboursement.
"""
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from schemas.domain import Recommendation
from schemas.results import CaseReviewerResult
from state.claim_state import ClaimState, validate_state_update

_STEP_NAME = "case_reviewer"
_AGENT_NAME = "case_reviewer_agent"


# ── Interface ──────────────────────────────────────────────────────────────────


@runtime_checkable
class CaseReviewerRunnable(Protocol):
    """Interface minimale requise par le nœud LangGraph."""

    def run(self, state: ClaimState) -> CaseReviewerResult: ...


# ── Stub par défaut ────────────────────────────────────────────────────────────


class _NotImplementedStub:
    """Placeholder retournant PENDING — aucune recommandation métier inventée.

    PENDING déclenche ``route_review → needs_review``, ce qui suspend le
    pipeline en attente d'une implémentation réelle ou d'une décision humaine.
    """

    def run(self, state: ClaimState) -> CaseReviewerResult:
        case_id = str(state.get("case_id", "UNKNOWN"))
        return CaseReviewerResult(
            case_id=case_id,
            recommendation=Recommendation.PENDING,
            justification=["[stub] case_reviewer_agent non implémenté — aucune décision possible."],
            human_review_required=True,
            human_review_reasons=["Implémentation de l'agent absente."],
        )


_DEFAULT_IMPL: CaseReviewerRunnable = _NotImplementedStub()


# ── Factory et nœud LangGraph ─────────────────────────────────────────────────


def make_node(
    impl: CaseReviewerRunnable = _DEFAULT_IMPL,
) -> Callable[[ClaimState], dict]:
    """Crée un nœud LangGraph avec l'implémentation injectable fournie."""
    def _node(state: ClaimState) -> dict:
        result = impl.run(state)
        updates: dict = {
            "review_result": result,
            "final_recommendation": result.recommendation,
            "final_justification": list(result.justification),
            "current_step": _STEP_NAME,
            "completed_steps": [_STEP_NAME],
        }
        if result.recommendation is Recommendation.PENDING:
            updates["alerts"] = [
                f"Revue dossier : PENDING — {'; '.join(result.human_review_reasons)}"
            ]
        elif result.recommendation is Recommendation.REJECT:
            updates["errors"] = [
                f"[{_AGENT_NAME}] Dossier rejeté — {'; '.join(result.justification)}"
            ]
        validate_state_update(updates)
        return updates

    _node.__name__ = f"node_{_STEP_NAME}"
    return _node


# Nœud stable — nom utilisé comme clé dans le StateGraph.
node = make_node()
