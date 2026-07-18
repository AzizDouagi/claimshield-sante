"""Tests de graph/recovery_node_v2 — plan de remédiation « autonomie
décisionnelle V2 », Phase 6.

Aucun test n'invoque de vrai LLM ni de vrai bundle FHIR sur disque — les
fonctions internes qui en dépendent (`_load_fhir_bundle_for_case`,
`agents.medical_risk_agent.agent.run`/`node`, `agents.eligibility_agent.agent.run`)
sont monkeypatchées, exactement comme les autres suites V2.
"""
from __future__ import annotations

from unittest.mock import Mock

import pytest

import agents.eligibility_agent.agent as eligibility_agent_module
import agents.medical_risk_agent.agent as medical_risk_agent_module
import graph.recovery_node_v2 as recovery_module
from agents.autonomous_decision_agent.agent import _allowed_decisions
from agents.autonomous_decision_agent.policy import classify_risk_signals, evaluate_acceptance_requirements
from graph.recovery_node_v2 import RecoveryPolicy, _recovery_disabled, make_recovery_node
from schemas.domain import ClaimDecisionV2, IntakeSafetyStatus, SeverityLevel, VerificationStatus
from schemas.results import (
    ClinicalEvidence,
    ClinicalEvidenceSource,
    ClinicalSignal,
    CoverageResult,
    FraudEvidence,
    FraudEvidenceSource,
    FraudSignal,
    IdentityResult,
    LlmMetadata,
    ProcedureCoding,
    StructuredError,
)
from schemas.v2_results import (
    ClassifiedMedicalItem,
    DocumentUnderstandingResult,
    EligibilityResult,
    IntakeSafetyResult,
    MedicalItemType,
    MedicalRiskResult,
    MedicalRiskResultPayload,
)


def _llm_trace() -> LlmMetadata:
    return LlmMetadata(model_name="gemma4:latest", prompt_version="1.0.0")


def _intake_safety(status: IntakeSafetyStatus = IntakeSafetyStatus.ACCEPTED) -> IntakeSafetyResult:
    return IntakeSafetyResult(case_id="CLM-6501", status=status, reasons=["motif"], llm_trace=_llm_trace())


def _eligibility(
    *,
    coverage_status: VerificationStatus = VerificationStatus.PASS,
    coverage_data_available: bool = True,
) -> EligibilityResult:
    return EligibilityResult(
        case_id="CLM-6501",
        status=coverage_status,
        identity=IdentityResult(status=VerificationStatus.PASS),
        coverage=CoverageResult(status=coverage_status),
        coverage_data_available=coverage_data_available,
        llm_trace=_llm_trace(),
    )


def _document_understanding(confidence: float = 0.9) -> DocumentUnderstandingResult:
    return DocumentUnderstandingResult(
        case_id="CLM-6501", status=VerificationStatus.PASS, confidence=confidence, llm_trace=_llm_trace()
    )


def _classified_item(item_type: MedicalItemType, description: str = "acte") -> ClassifiedMedicalItem:
    return ClassifiedMedicalItem(
        description=description,
        item_type=item_type,
        classification_method="unresolved" if item_type is MedicalItemType.UNKNOWN else "referential_match",
        resolution_status=VerificationStatus.NEEDS_REVIEW,
    )


def _medical_risk(
    *,
    codings: list[ProcedureCoding] | None = None,
    fraud_signals: list[FraudSignal] | None = None,
    clinical_signals: list[ClinicalSignal] | None = None,
    classified_items: list[ClassifiedMedicalItem] | None = None,
    procedure_count: int = 1,
    medication_count: int = 0,
    errors: list[StructuredError] | None = None,
) -> MedicalRiskResult:
    default_codings = (
        codings
        if codings is not None
        else [ProcedureCoding(original_description="acte résolu", status=VerificationStatus.PASS, proposed_code="123")]
    )
    return MedicalRiskResult(
        case_id="CLM-6501",
        status=VerificationStatus.PASS,
        llm_trace=_llm_trace(),
        errors=errors or [],
        result_payload=MedicalRiskResultPayload(
            procedure_count=procedure_count,
            medication_count=medication_count,
            codings=default_codings,
            fraud_signals=fraud_signals or [],
            clinical_signals=clinical_signals or [],
            classified_items=classified_items or [],
        ),
    )


def _state(**results) -> dict:
    state = {
        "case_id": "CLM-6501",
        "schema_version": "2.0.0",
        "current_step": "medical_risk",
        "completed_steps": [],
        "intake_safety_result": _intake_safety(),
        "document_understanding_result": _document_understanding(),
        "eligibility_result": _eligibility(),
        "medical_risk_result": _medical_risk(),
    }
    state.update(results)
    return state


def _fraud_signal(signal_type: str) -> FraudSignal:
    return FraudSignal(
        signal_type=signal_type,
        description=f"Signal {signal_type}.",
        risk_contribution=0.4,
        evidence=[FraudEvidence(source=FraudEvidenceSource.IDENTITY_COVERAGE, field="x", value="y")],
    )


def _clinical_critical_signal() -> ClinicalSignal:
    return ClinicalSignal(
        signal_type="MISSING_PRESCRIPTION_REFERENCE",
        description="Médicament facturé sans ordonnance.",
        fields_compared=["medication_count"],
        severity=SeverityLevel.CRITICAL,
        evidence=[ClinicalEvidence(source=ClinicalEvidenceSource.OCR_EXTRACTION, field="medication_count", value="1")],
    )


# ── Désactivation entière ─────────────────────────────────────────────────────


class TestRecoveryDisabled:
    @pytest.mark.parametrize("status", [IntakeSafetyStatus.BLOCKED, IntakeSafetyStatus.QUARANTINED, IntakeSafetyStatus.TECHNICAL_FAILURE])
    def test_disabled_on_terminal_intake_status(self, status):
        node = make_recovery_node()
        state = _state(intake_safety_result=_intake_safety(status))
        updates = node(state)
        assert "recovery_attempts" not in updates
        assert updates == {"current_step": "recovery", "completed_steps": ["recovery"]}

    def test_disabled_on_confirmed_clinical_risk(self):
        node = make_recovery_node()
        state = _state(medical_risk_result=_medical_risk(clinical_signals=[_clinical_critical_signal()]))
        updates = node(state)
        assert "recovery_attempts" not in updates

    def test_disabled_on_confirmed_fraud_risk(self):
        node = make_recovery_node()
        state = _state(medical_risk_result=_medical_risk(fraud_signals=[_fraud_signal("EXACT_DUPLICATE_INVOICE")]))
        updates = node(state)
        assert "recovery_attempts" not in updates

    def test_disabled_on_confirmed_eligibility_failure(self):
        node = make_recovery_node()
        state = _state(medical_risk_result=_medical_risk(fraud_signals=[_fraud_signal("IDENTITY_MISMATCH")]))
        updates = node(state)
        assert "recovery_attempts" not in updates

    def test_never_disabled_on_suspected_fraud_alone(self):
        """`NEAR_DUPLICATE_INVOICE` (SUSPECTED_FRAUD_RISK, non confirmé) ne
        désactive jamais la récupération — même correctif que la matrice de
        décision (Phase 4, point 1 d'AZIZ)."""
        classified = classify_risk_signals(
            intake_safety_result=_intake_safety(),
            medical_risk_result=_medical_risk(fraud_signals=[_fraud_signal("NEAR_DUPLICATE_INVOICE")]),
            document_understanding_result=_document_understanding(),
        )
        assert _recovery_disabled(intake_status="ACCEPTED", classified_signals=classified) is False


# ── Aucun déclenchement sur un dossier propre ─────────────────────────────────


class TestRecoveryEnabledNoTrigger:
    def test_clean_dossier_produces_no_attempt(self):
        node = make_recovery_node()
        state = _state()
        updates = node(state)
        assert "recovery_attempts" not in updates
        assert "medical_risk_result" not in updates
        assert "eligibility_result" not in updates
        assert updates["current_step"] == "recovery"
        assert updates["completed_steps"] == ["recovery"]


# ── RECOMPUTE_ELIGIBILITY ─────────────────────────────────────────────────────


class TestRecomputeEligibility:
    def test_not_triggered_without_bundle(self, monkeypatch):
        monkeypatch.setattr(recovery_module, "_load_fhir_bundle_for_case", lambda state: None)
        node = make_recovery_node()
        state = _state(
            eligibility_result=_eligibility(
                coverage_status=VerificationStatus.NEEDS_REVIEW, coverage_data_available=False
            )
        )
        updates = node(state)
        assert "recovery_attempts" not in updates

    def test_no_improvement_when_bundle_has_no_payer_hint(self, monkeypatch):
        monkeypatch.setattr(recovery_module, "_load_fhir_bundle_for_case", lambda state: {"resourceType": "Bundle"})
        monkeypatch.setattr(recovery_module, "extract_payer_hint_from_coverage", lambda bundle: None)
        node = make_recovery_node()
        state = _state(
            eligibility_result=_eligibility(
                coverage_status=VerificationStatus.NEEDS_REVIEW, coverage_data_available=False
            )
        )
        updates = node(state)
        assert len(updates["recovery_attempts"]) == 1
        attempt = updates["recovery_attempts"][0]
        assert attempt.action.value == "RECOMPUTE_ELIGIBILITY"
        assert attempt.result.value == "NO_IMPROVEMENT"
        assert "eligibility_result" not in updates

    def test_success_updates_eligibility_result(self, monkeypatch):
        monkeypatch.setattr(recovery_module, "_load_fhir_bundle_for_case", lambda state: {"resourceType": "Bundle"})
        monkeypatch.setattr(
            recovery_module,
            "extract_payer_hint_from_coverage",
            lambda bundle: {"payer_name": "Assureur X", "policy_number": "POL-1"},
        )
        monkeypatch.setattr(eligibility_agent_module, "_find_fhir_bundle_path", lambda state: None)
        recomputed = _eligibility(coverage_status=VerificationStatus.PASS, coverage_data_available=True)
        run_spy = Mock(return_value=recomputed)
        monkeypatch.setattr(eligibility_agent_module, "run", run_spy)

        node = make_recovery_node()
        state = _state(
            eligibility_result=_eligibility(
                coverage_status=VerificationStatus.NEEDS_REVIEW, coverage_data_available=False
            )
        )
        updates = node(state)

        run_spy.assert_called_once()
        assert updates["eligibility_result"] is recomputed
        assert len(updates["recovery_attempts"]) == 1
        assert updates["recovery_attempts"][0].result.value == "SUCCESS"


# ── READ_MEDICAL_ITEMS_FROM_FHIR ──────────────────────────────────────────────


class TestReadMedicalItemsFromFhir:
    def test_not_triggered_on_clean_dossier(self, monkeypatch):
        monkeypatch.setattr(recovery_module, "_load_fhir_bundle_for_case", lambda state: {"resourceType": "Bundle"})
        extract_spy = Mock()
        monkeypatch.setattr(recovery_module, "extract_medical_items_from_bundle", extract_spy)
        node = make_recovery_node()
        node(_state())
        extract_spy.assert_not_called()

    def test_triggered_on_unknown_item_and_no_new_items_gives_no_improvement(self, monkeypatch):
        # L'élément reste UNKNOWN après cette action — RESOLVE_MEDICAL_CODE
        # se déclenche donc lui aussi juste après (comportement attendu,
        # couvert séparément par `TestResolveMedicalCode`) ; seule la
        # première tentative (celle testée ici) nous intéresse.
        monkeypatch.setattr(recovery_module, "_load_fhir_bundle_for_case", lambda state: {"resourceType": "Bundle"})
        monkeypatch.setattr(recovery_module, "extract_medical_items_from_bundle", lambda bundle: [])
        node = make_recovery_node()
        state = _state(
            medical_risk_result=_medical_risk(classified_items=[_classified_item(MedicalItemType.UNKNOWN, "mystère")])
        )
        updates = node(state)
        attempt = updates["recovery_attempts"][0]
        assert attempt.action.value == "READ_MEDICAL_ITEMS_FROM_FHIR"
        assert attempt.result.value == "NO_IMPROVEMENT"
        assert "medical_risk_result" not in updates

    def test_success_merges_new_fhir_items_and_reruns_medical_risk(self, monkeypatch):
        fhir_item = _classified_item(MedicalItemType.PROCEDURE, "acte fhir")
        monkeypatch.setattr(recovery_module, "_load_fhir_bundle_for_case", lambda state: {"resourceType": "Bundle"})
        monkeypatch.setattr(recovery_module, "extract_medical_items_from_bundle", lambda bundle: [fhir_item])

        merged_result = _medical_risk(classified_items=[_classified_item(MedicalItemType.UNKNOWN, "mystère"), fhir_item])
        run_spy = Mock(return_value=merged_result)
        monkeypatch.setattr(medical_risk_agent_module, "run", run_spy)

        node = make_recovery_node()
        state = _state(
            medical_risk_result=_medical_risk(classified_items=[_classified_item(MedicalItemType.UNKNOWN, "mystère")])
        )
        updates = node(state)

        run_spy.assert_called_once()
        assert updates["medical_risk_result"] is merged_result
        assert updates["recovery_attempts"][0].result.value == "SUCCESS"


# ── RESOLVE_MEDICAL_CODE ──────────────────────────────────────────────────────


class TestResolveMedicalCode:
    def test_not_triggered_without_unknown_items(self):
        node = make_recovery_node()
        state = _state(medical_risk_result=_medical_risk(classified_items=[_classified_item(MedicalItemType.PROCEDURE)]))
        updates = node(state)
        assert "recovery_attempts" not in updates

    def test_no_improvement_when_reclassification_still_unknown(self, monkeypatch):
        monkeypatch.setattr(
            medical_risk_agent_module,
            "_classify_medical_item",
            lambda description, **kwargs: _classified_item(MedicalItemType.UNKNOWN, description),
        )
        node = make_recovery_node()
        state = _state(medical_risk_result=_medical_risk(classified_items=[_classified_item(MedicalItemType.UNKNOWN, "mystère")]))
        updates = node(state)
        attempt = updates["recovery_attempts"][0]
        assert attempt.action.value == "RESOLVE_MEDICAL_CODE"
        assert attempt.result.value == "NO_IMPROVEMENT"
        assert "medical_risk_result" not in updates

    def test_success_resolves_previously_unknown_item(self, monkeypatch):
        resolved = _classified_item(MedicalItemType.PROCEDURE, "mystère")
        monkeypatch.setattr(
            medical_risk_agent_module, "_classify_medical_item", lambda description, **kwargs: resolved
        )
        rerun_result = _medical_risk(classified_items=[resolved])
        run_spy = Mock(return_value=rerun_result)
        monkeypatch.setattr(medical_risk_agent_module, "run", run_spy)

        node = make_recovery_node()
        state = _state(medical_risk_result=_medical_risk(classified_items=[_classified_item(MedicalItemType.UNKNOWN, "mystère")]))
        updates = node(state)

        run_spy.assert_called_once()
        assert updates["medical_risk_result"] is rerun_result
        assert updates["recovery_attempts"][0].result.value == "SUCCESS"


# ── RETRY_STRUCTURED_LLM_OUTPUT ───────────────────────────────────────────────


class TestRetryStructuredLlmOutput:
    def test_not_triggered_without_llm_failure(self):
        node = make_recovery_node()
        updates = node(_state())
        assert "recovery_attempts" not in updates

    def test_no_improvement_when_retry_still_fails(self, monkeypatch):
        still_failing = _medical_risk(
            errors=[StructuredError(code="LLM_UNAVAILABLE", message="indisponible", field="llm_trace")]
        )
        monkeypatch.setattr(
            medical_risk_agent_module, "node", Mock(return_value={"medical_risk_result": still_failing})
        )
        node = make_recovery_node()
        state = _state(
            medical_risk_result=_medical_risk(
                errors=[StructuredError(code="LLM_UNAVAILABLE", message="indisponible", field="llm_trace")]
            )
        )
        updates = node(state)
        attempt = updates["recovery_attempts"][0]
        assert attempt.action.value == "RETRY_STRUCTURED_LLM_OUTPUT"
        assert attempt.result.value == "NO_IMPROVEMENT"
        assert "medical_risk_result" not in updates

    def test_success_when_retry_recovers(self, monkeypatch):
        recovered = _medical_risk()
        monkeypatch.setattr(medical_risk_agent_module, "node", Mock(return_value={"medical_risk_result": recovered}))
        node = make_recovery_node()
        state = _state(
            medical_risk_result=_medical_risk(
                errors=[StructuredError(code="LLM_UNAVAILABLE", message="indisponible", field="llm_trace")]
            )
        )
        updates = node(state)
        assert updates["medical_risk_result"] is recovered
        assert updates["recovery_attempts"][0].result.value == "SUCCESS"


# ── Bornes configurables (RecoveryPolicy) ─────────────────────────────────────


class TestRecoveryPolicyBounds:
    def test_policy_rejects_invalid_values(self):
        with pytest.raises(ValueError):
            RecoveryPolicy(max_attempts_per_action=0)
        with pytest.raises(ValueError):
            RecoveryPolicy(max_total_attempts=0)
        with pytest.raises(ValueError):
            RecoveryPolicy(version="")

    def test_max_total_attempts_stops_after_first_action(self, monkeypatch):
        """Deux actions sont simultanément déclenchables (RESOLVE_MEDICAL_CODE
        et RETRY_STRUCTURED_LLM_OUTPUT) — `max_total_attempts=1` ne permet
        qu'une seule tentative, jamais les deux."""
        monkeypatch.setattr(
            medical_risk_agent_module,
            "_classify_medical_item",
            lambda description, **kwargs: _classified_item(MedicalItemType.UNKNOWN, description),
        )
        node = make_recovery_node(policy=RecoveryPolicy(max_total_attempts=1))
        state = _state(
            medical_risk_result=_medical_risk(
                classified_items=[_classified_item(MedicalItemType.UNKNOWN, "mystère")],
                errors=[StructuredError(code="LLM_UNAVAILABLE", message="indisponible", field="llm_trace")],
            )
        )
        updates = node(state)
        assert len(updates["recovery_attempts"]) == 1
        assert updates["recovery_attempts"][0].action.value == "RESOLVE_MEDICAL_CODE"

    def test_default_policy_allows_all_four_actions(self):
        assert recovery_module.DEFAULT_RECOVERY_POLICY.max_total_attempts >= 4


# ── Cohérence croisée avec la matrice de décision (plan §11) ──────────────────


class TestCrossConsistencyWithDecisionMatrix:
    """`classify_risk_signals` est l'unique source de vérité partagée entre
    `graph.recovery_node_v2._recovery_disabled` et
    `agents.autonomous_decision_agent.agent._allowed_decisions` — jamais un
    second calcul concurrent, jamais une divergence entre les deux."""

    @pytest.mark.parametrize(
        "medical_risk_result",
        [
            _medical_risk(),
            _medical_risk(clinical_signals=[_clinical_critical_signal()]),
            _medical_risk(fraud_signals=[_fraud_signal("EXACT_DUPLICATE_INVOICE")]),
            _medical_risk(fraud_signals=[_fraud_signal("IDENTITY_MISMATCH")]),
            _medical_risk(fraud_signals=[_fraud_signal("NEAR_DUPLICATE_INVOICE")]),
        ],
    )
    def test_recovery_disabled_implies_decision_matrix_already_locked(self, medical_risk_result):
        intake_safety_result = _intake_safety()
        document_understanding_result = _document_understanding()
        classified = classify_risk_signals(
            intake_safety_result=intake_safety_result,
            medical_risk_result=medical_risk_result,
            document_understanding_result=document_understanding_result,
        )
        disabled = _recovery_disabled(intake_status="ACCEPTED", classified_signals=classified)

        requirements = evaluate_acceptance_requirements(
            identity_status=VerificationStatus.PASS,
            document_status=VerificationStatus.PASS,
            has_confirmed_dangerous_clinical_signal=False,
            has_confirmed_coverage_exclusion=False,
            identity_is_pass=True,
            coverage_is_pass=True,
            has_resolved_medical_item=True,
            ceiling_exceeded=False,
        )
        allowed, _bounded_by = _allowed_decisions(
            upstream_results_present=True,
            intake_status="ACCEPTED",
            classified_signals=classified,
            has_partial_condition=False,
            requirements=requirements,
        )

        if disabled:
            # Un signal confirmé dangereux désactive la récupération ET
            # verrouille la matrice de décision à un singleton dangereux —
            # jamais {APPROVE, REJECT} encore ouvert.
            assert allowed in (frozenset({ClaimDecisionV2.QUARANTINE}), frozenset({ClaimDecisionV2.REJECT}))
        else:
            # Récupération active : la matrice de décision n'a jamais été
            # court-circuitée par un signal confirmé dangereux.
            assert ClaimDecisionV2.QUARANTINE not in allowed
