"""Tests de api/v2/claims.py::_build_status_response — plan de remédiation
« autonomie décisionnelle V2 », Phase 7 (« API + explicabilité chat »).

Vérifie que les champs d'explicabilité déjà calculés par
`agents.autonomous_decision_agent` (Phases 2/4/5) — jamais surfacés par
l'API jusqu'à cette phase — sont désormais correctement exposés dans
`ClaimStatusResponseV2`, sans nouveau calcul (simple exposition)."""
from __future__ import annotations

from api.v2.claims import _build_status_response
from schemas.domain import ClaimDecisionV2, VerificationStatus
from schemas.v2_results import (
    AutonomousDecisionResult,
    DecisionAssumption,
    DecisionCounterfactual,
    DecisionFactor,
    EvidenceCompleteness,
    LlmMetadata,
    MissingInformation,
    MissingInformationDimension,
    MissingInformationImportance,
)


def _llm_trace() -> LlmMetadata:
    return LlmMetadata(model_name="gemma4:latest", prompt_version="1.0.0")


def _decision_result(**overrides) -> AutonomousDecisionResult:
    defaults = dict(
        case_id="CLM-7101",
        status=VerificationStatus.NEEDS_REVIEW,
        decision=ClaimDecisionV2.PARTIAL_APPROVE,
        llm_trace=_llm_trace(),
        missing_information=[
            MissingInformation(
                code="UNRESOLVED_CODING",
                description="Codification non résolue.",
                importance=MissingInformationImportance.IMPORTANT,
                affected_dimension=MissingInformationDimension.CODING,
                source_agent="medical_risk_agent",
                impact_on_decision="Confiance réduite.",
            )
        ],
        assumptions=[
            DecisionAssumption(
                code="DECISION_DESPITE_INCOMPLETE_DATA",
                description="Décision positive malgré des informations incomplètes.",
            )
        ],
        decisive_factors=[
            DecisionFactor(code="IDENTITY_CONFIRMED", description="Identité confirmée.", source_agent="eligibility_agent")
        ],
        counterfactuals=[
            DecisionCounterfactual(
                condition="Code résolu",
                current_value="Non résolu",
                required_value="Résolu",
                resulting_decision=ClaimDecisionV2.APPROVE,
                explanation="Un code résolu permettrait une approbation complète.",
            )
        ],
        recommended_action="Vérifier la codification.",
        evidence_completeness=EvidenceCompleteness.PARTIAL,
    )
    defaults.update(overrides)
    return AutonomousDecisionResult(**defaults)


class TestExplainabilityFieldsExposed:
    def test_all_explainability_fields_carried_through(self):
        result = _decision_result()
        response = _build_status_response(
            "CLM-7101", {"decision_result": result, "final_decision": ClaimDecisionV2.PARTIAL_APPROVE}
        )
        assert response.missing_information == result.missing_information
        assert response.assumptions == result.assumptions
        assert response.decisive_factors == result.decisive_factors
        assert response.counterfactuals == result.counterfactuals
        assert response.recommended_action == "Vérifier la codification."
        assert response.evidence_completeness is EvidenceCompleteness.PARTIAL

    def test_defaults_are_empty_without_decision_result(self):
        response = _build_status_response("CLM-7102", {})
        assert response.missing_information == []
        assert response.assumptions == []
        assert response.decisive_factors == []
        assert response.counterfactuals == []
        assert response.recommended_action == ""
        assert response.evidence_completeness is None

    def test_never_a_new_computation_only_exposure(self):
        """Les valeurs exposées sont strictement identiques (même identité de
        contenu) à celles déjà produites par l'agent — jamais recalculées."""
        result = _decision_result(missing_information=[], assumptions=[], counterfactuals=[])
        response = _build_status_response("CLM-7101", {"decision_result": result})
        assert response.missing_information == []
        assert response.assumptions == []
        assert response.counterfactuals == []
