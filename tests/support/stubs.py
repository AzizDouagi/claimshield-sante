"""Faux agents/résultats déterministes partagés entre suites de tests —
évite de dupliquer ces stubs entre ``tests/graph/`` et ``tests/e2e/``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from schemas.domain import Recommendation
from schemas.results import CaseReviewerResult, CaseReviewerResultPayload, LlmMetadata


@dataclass
class StubResult:
    decision: Any = None
    status: Any = None


@dataclass
class StubSubStatus:
    status: Any = None


@dataclass
class StubIdentityCoverageResult:
    identity: StubSubStatus
    coverage: StubSubStatus


class CaseReviewerApproveStub:
    """Faux agent injecté via ``case_reviewer_impl`` — recommandation APPROVE,
    ``auto_decision`` configurable pour exercer les deux chemins (avec/sans
    court-circuit d'auto-approbation P1-4)."""

    def __init__(self, *, auto_decision: str | None) -> None:
        self._auto_decision = auto_decision

    def run(self, state: dict) -> CaseReviewerResult:
        return CaseReviewerResult(
            case_id=str(state.get("case_id", "UNKNOWN")),
            llm_trace=LlmMetadata(model_name="test-llm", prompt_version="test"),
            result_payload=CaseReviewerResultPayload(
                recommendation=Recommendation.APPROVE,
                justification=["Toutes les vérifications ont réussi."],
                human_review_reasons=["Validation humaine obligatoire avant toute décision finale."],
                auto_decision=self._auto_decision,
                auto_decision_criteria=(
                    ["Critères P1-4 réunis (scénario de test)."] if self._auto_decision else []
                ),
            ),
        )
