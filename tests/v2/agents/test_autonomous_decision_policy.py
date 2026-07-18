"""Tests de agents/autonomous_decision_agent/policy.py — plan de remédiation
« autonomie décisionnelle V2 », phase 2.

Fonctions pures testées en isolation totale (aucun graphe LangGraph, aucun
appel LLM, aucun état) — `classify_risk_signals` (§3.3 du plan) et
`evaluate_acceptance_requirements` (§3.4).
"""
from __future__ import annotations

from agents.autonomous_decision_agent.policy import (
    AcceptanceRequirementsVerdict,
    RequirementTier,
    classify_risk_signals,
    evaluate_acceptance_requirements,
    has_confirmed_category,
)
from schemas.domain import IntakeSafetyStatus, SeverityLevel, VerificationStatus
from schemas.results import (
    ClinicalEvidence,
    ClinicalEvidenceSource,
    ClinicalSignal,
    FraudEvidence,
    FraudEvidenceSource,
    FraudSignal,
    LlmMetadata,
)
from schemas.v2_results import (
    DocumentUnderstandingResult,
    IntakeSafetyResult,
    MedicalRiskResult,
    MedicalRiskResultPayload,
    RiskSignalCategory,
)


def _llm_trace() -> LlmMetadata:
    return LlmMetadata(model_name="gemma4:latest", prompt_version="1.0.0", confidence=0.9)


def _fraud_signal(signal_type: str, *, risk_contribution: float = 0.2) -> FraudSignal:
    return FraudSignal(
        signal_type=signal_type,
        description=f"Signal {signal_type}.",
        risk_contribution=risk_contribution,
        evidence=[FraudEvidence(source=FraudEvidenceSource.IDENTITY_COVERAGE, field="x", value="y")],
    )


def _clinical_signal(severity: SeverityLevel) -> ClinicalSignal:
    return ClinicalSignal(
        signal_type="MISSING_PRESCRIPTION_REFERENCE",
        description="Médicament facturé sans ordonnance.",
        fields_compared=["medication_count"],
        severity=severity,
        evidence=[
            ClinicalEvidence(source=ClinicalEvidenceSource.OCR_EXTRACTION, field="medication_count", value="1")
        ],
    )


def _medical_risk_result(
    *, fraud_signals=(), clinical_signals=(), procedure_count=1, medication_count=0, codings=()
) -> MedicalRiskResult:
    payload = MedicalRiskResultPayload(
        procedure_count=procedure_count,
        medication_count=medication_count,
        codings=list(codings),
        fraud_signals=list(fraud_signals),
        clinical_signals=list(clinical_signals),
    )
    return MedicalRiskResult(
        case_id="CLM-0001", status=VerificationStatus.PASS, llm_trace=_llm_trace(), result_payload=payload
    )


def _intake_safety_result(status: IntakeSafetyStatus) -> IntakeSafetyResult:
    return IntakeSafetyResult(case_id="CLM-0001", status=status, reasons=["motif"], llm_trace=_llm_trace())


def _document_understanding_result(confidence: float) -> DocumentUnderstandingResult:
    return DocumentUnderstandingResult(
        case_id="CLM-0001", status=VerificationStatus.PASS, confidence=confidence, llm_trace=_llm_trace()
    )


class TestClassifyRiskSignals:
    def test_near_duplicate_invoice_is_suspected_not_confirmed_fraud(self):
        """Correctif AZIZ — un `NEAR_DUPLICATE_INVOICE` (similarité
        probabiliste) ne doit jamais être classé `CONFIRMED_FRAUD_RISK`,
        même combiné à un `CEILING_EXCEEDED` qui, historiquement, aurait fait
        atteindre `risk_level == HIGH`."""
        result = _medical_risk_result(
            fraud_signals=[_fraud_signal("NEAR_DUPLICATE_INVOICE"), _fraud_signal("CEILING_EXCEEDED")]
        )
        classified = classify_risk_signals(medical_risk_result=result)
        near_duplicate = [c for c in classified if c.signal_type == "NEAR_DUPLICATE_INVOICE"]
        assert len(near_duplicate) == 1
        assert near_duplicate[0].category is RiskSignalCategory.SUSPECTED_FRAUD_RISK
        assert near_duplicate[0].confirmed is False
        assert not has_confirmed_category(classified, RiskSignalCategory.CONFIRMED_FRAUD_RISK)
        # CEILING_EXCEEDED n'appartient à aucune des 6 catégories — géré ailleurs.
        assert not any(c.signal_type == "CEILING_EXCEEDED" for c in classified)

    def test_exact_duplicate_invoice_is_confirmed_fraud(self):
        result = _medical_risk_result(fraud_signals=[_fraud_signal("EXACT_DUPLICATE_INVOICE")])
        classified = classify_risk_signals(medical_risk_result=result)
        assert has_confirmed_category(classified, RiskSignalCategory.CONFIRMED_FRAUD_RISK)

    def test_identity_mismatch_and_coverage_exclusion_are_eligibility_failure(self):
        result = _medical_risk_result(
            fraud_signals=[_fraud_signal("IDENTITY_MISMATCH"), _fraud_signal("COVERAGE_INACTIVE_OR_EXPIRED")]
        )
        classified = classify_risk_signals(medical_risk_result=result)
        eligibility_failures = [c for c in classified if c.category is RiskSignalCategory.ELIGIBILITY_FAILURE]
        assert {c.signal_type for c in eligibility_failures} == {
            "IDENTITY_MISMATCH",
            "COVERAGE_INACTIVE_OR_EXPIRED",
        }
        assert all(c.confirmed for c in eligibility_failures)

    def test_critical_clinical_signal_is_confirmed_clinical_risk(self):
        result = _medical_risk_result(clinical_signals=[_clinical_signal(SeverityLevel.CRITICAL)])
        classified = classify_risk_signals(medical_risk_result=result)
        assert has_confirmed_category(classified, RiskSignalCategory.CONFIRMED_CLINICAL_RISK)

    def test_non_critical_clinical_signal_is_never_confirmed_clinical_risk(self):
        result = _medical_risk_result(clinical_signals=[_clinical_signal(SeverityLevel.MEDIUM)])
        classified = classify_risk_signals(medical_risk_result=result)
        assert not any(c.category is RiskSignalCategory.CONFIRMED_CLINICAL_RISK for c in classified)

    def test_completeness_and_confidence_signals_never_confirmed(self):
        result = _medical_risk_result(
            fraud_signals=[
                _fraud_signal("IDENTITY_AMBIGUOUS"),
                _fraud_signal("PREAUTHORIZATION_MISSING"),
                _fraud_signal("UNRESOLVED_CODING"),
                _fraud_signal("LOW_EXTRACTION_CONFIDENCE"),
            ]
        )
        classified = classify_risk_signals(medical_risk_result=result)
        assert all(not c.confirmed for c in classified)
        categories = {c.category for c in classified}
        assert RiskSignalCategory.COMPLETENESS_GAP in categories
        assert RiskSignalCategory.CONFIDENCE_GAP in categories

    def test_structural_absence_is_completeness_gap(self):
        result = _medical_risk_result(procedure_count=0, medication_count=0, codings=[])
        classified = classify_risk_signals(medical_risk_result=result)
        assert any(c.signal_type == "STRUCTURAL_ABSENCE" for c in classified)

    def test_intake_blocked_or_quarantined_is_confirmed_security_risk(self):
        for status in (IntakeSafetyStatus.BLOCKED, IntakeSafetyStatus.QUARANTINED):
            classified = classify_risk_signals(intake_safety_result=_intake_safety_result(status))
            assert has_confirmed_category(classified, RiskSignalCategory.CONFIRMED_SECURITY_RISK)

    def test_intake_accepted_is_never_a_security_risk_signal(self):
        classified = classify_risk_signals(
            intake_safety_result=_intake_safety_result(IntakeSafetyStatus.ACCEPTED)
        )
        assert classified == []

    def test_low_document_confidence_is_confidence_gap(self):
        classified = classify_risk_signals(document_understanding_result=_document_understanding_result(0.3))
        assert any(c.category is RiskSignalCategory.CONFIDENCE_GAP for c in classified)

    def test_high_document_confidence_produces_no_signal(self):
        classified = classify_risk_signals(document_understanding_result=_document_understanding_result(0.95))
        assert classified == []

    def test_no_inputs_produces_no_signals(self):
        assert classify_risk_signals() == []


class TestEvaluateAcceptanceRequirements:
    def _verdict(self, **overrides) -> AcceptanceRequirementsVerdict:
        defaults = dict(
            identity_status=VerificationStatus.PASS,
            document_status=VerificationStatus.PASS,
            has_confirmed_dangerous_clinical_signal=False,
            has_confirmed_coverage_exclusion=False,
            identity_is_pass=True,
            coverage_is_pass=False,
            has_resolved_medical_item=False,
            ceiling_exceeded=False,
        )
        defaults.update(overrides)
        return evaluate_acceptance_requirements(**defaults)

    def test_identity_confirmed_alone_satisfies_minimum_requirements(self):
        verdict = self._verdict()
        assert verdict.minimum_requirements_satisfied is True

    def test_coverage_unknown_alone_without_any_positive_signal_is_insufficient(self):
        """Corrige explicitement le point 4 d'AZIZ : une couverture
        UNKNOWN/NEEDS_REVIEW seule (aucun autre signal positif confirmé) ne
        doit jamais suffire à `APPROVE`."""
        verdict = self._verdict(identity_is_pass=False, coverage_is_pass=False, has_resolved_medical_item=False)
        assert verdict.minimum_requirements_satisfied is False
        check = next(c for c in verdict.checks if c.code == "HAS_ANY_POSITIVE_CONFIRMED_SIGNAL")
        assert check.satisfied is False
        assert check.tier is RequirementTier.REQUIRED_CONFIRMED

    def test_resolved_medical_item_alone_satisfies_positive_signal_requirement(self):
        verdict = self._verdict(identity_is_pass=False, coverage_is_pass=False, has_resolved_medical_item=True)
        assert verdict.minimum_requirements_satisfied is True

    def test_confirmed_dangerous_clinical_signal_blocks_minimum_requirements(self):
        verdict = self._verdict(has_confirmed_dangerous_clinical_signal=True)
        assert verdict.minimum_requirements_satisfied is False

    def test_confirmed_coverage_exclusion_blocks_minimum_requirements(self):
        verdict = self._verdict(has_confirmed_coverage_exclusion=True)
        assert verdict.minimum_requirements_satisfied is False

    def test_identity_fail_blocks_minimum_requirements(self):
        verdict = self._verdict(identity_status=VerificationStatus.FAIL)
        assert verdict.minimum_requirements_satisfied is False

    def test_document_fail_blocks_minimum_requirements(self):
        verdict = self._verdict(document_status=VerificationStatus.FAIL)
        assert verdict.minimum_requirements_satisfied is False

    def test_ceiling_exceeded_satisfies_partial_requirements_even_without_positive_signal(self):
        verdict = self._verdict(
            identity_is_pass=False, coverage_is_pass=False, has_resolved_medical_item=False, ceiling_exceeded=True
        )
        assert verdict.minimum_requirements_satisfied is False
        assert verdict.partial_requirements_satisfied is True

    def test_checks_are_never_empty(self):
        verdict = self._verdict()
        assert len(verdict.checks) >= 5
