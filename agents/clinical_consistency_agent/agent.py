"""Clinical Consistency Agent — interface injectable — ClaimShield Santé.

Cet agent n'est pas encore implémenté.  Ce fichier fournit :
  - ``ClinicalConsistencyRunnable`` — Protocol structural (pas d'héritage requis).
  - ``_NotImplementedStub`` — implémentation par défaut retournant NOT_EVALUATED.
  - ``make_node(impl)`` — factory LangGraph injectable pour les tests.
  - ``node`` — nœud par défaut utilisant le stub.

Le stub ne retourne jamais de résultat métier inventé.
Toute implémentation réelle doit satisfaire ``ClinicalConsistencyRunnable``.
"""
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from schemas.domain import VerificationStatus
from schemas.results import ClinicalConsistencyResult
from state.claim_state import ClaimState, validate_state_update

_STEP_NAME = "clinical_consistency"
_AGENT_NAME = "clinical_consistency_agent"


# ── Interface ──────────────────────────────────────────────────────────────────


@runtime_checkable
class ClinicalConsistencyRunnable(Protocol):
    """Interface minimale requise par le nœud LangGraph.

    Lit l'état partagé (résultats OCR, FHIR, codification déjà présents) et
    retourne un ``ClinicalConsistencyResult`` structuré.
    """

    def run(self, state: ClaimState) -> ClinicalConsistencyResult: ...


# ── Stub par défaut ────────────────────────────────────────────────────────────


class _NotImplementedStub:
    """Placeholder retournant NOT_EVALUATED — ne simule aucun résultat métier.

    La valeur NOT_EVALUATED est reconnue par ``route_verification_fan_in``
    comme « non applicable » et ne bloque pas le pipeline.
    """

    def run(self, state: ClaimState) -> ClinicalConsistencyResult:
        case_id = str(state.get("case_id", "UNKNOWN"))
        return ClinicalConsistencyResult(
            case_id=case_id,
            status=VerificationStatus.NOT_EVALUATED,
            reasons=["[stub] clinical_consistency_agent non implémenté — résultat non évalué."],
        )


_DEFAULT_IMPL: ClinicalConsistencyRunnable = _NotImplementedStub()


# ── Factory et nœud LangGraph ─────────────────────────────────────────────────


def make_node(
    impl: ClinicalConsistencyRunnable = _DEFAULT_IMPL,
) -> Callable[[ClaimState], dict]:
    """Crée un nœud LangGraph avec l'implémentation injectable fournie.

    Args:
        impl: Toute classe satisfaisant ``ClinicalConsistencyRunnable``.
              Par défaut : ``_NotImplementedStub`` (retourne NOT_EVALUATED).

    Returns:
        Fonction ``(state) -> dict`` compatible LangGraph.
    """
    def _node(state: ClaimState) -> dict:
        result = impl.run(state)
        updates: dict = {
            "clinical_result": result,
            "current_step": _STEP_NAME,
            "completed_steps": [_STEP_NAME],
        }
        if result.status is VerificationStatus.FAIL:
            updates["errors"] = [
                f"[{_AGENT_NAME}] {r}" for r in result.reasons
            ]
        elif result.status in (VerificationStatus.NEEDS_REVIEW, VerificationStatus.NOT_EVALUATED):
            updates["alerts"] = [
                f"Cohérence clinique : {result.status.value} — {'; '.join(result.reasons)}"
            ]
        validate_state_update(updates)
        return updates

    _node.__name__ = f"node_{_STEP_NAME}"
    return _node


# Nœud stable — nom utilisé comme clé dans le StateGraph.
node = make_node()
