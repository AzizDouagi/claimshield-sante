"""Tests de schemas/v2_results.py (V2) — Phase V2-1.

Chaque schéma est vérifié sur un exemple réaliste inspiré de
datasets/fixtures/valid/CLM-0001/oracle/ground_truth.json (case_id
"CLM-0001", décision APPROVE), `extra='forbid'`, round-trip JSON, et les
garanties anti-fuite/anti-invention héritées du même patron que
schemas/results.py (V1).
"""
from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from schemas.domain import (
    ClaimDecisionV2,
    DocumentType,
    ExtractionStatus,
    IntakeSafetyStatus,
    VerificationStatus,
)
from schemas.results import (
    ClinicalEvidence,
    ClinicalEvidenceSource,
    ClinicalSignal,
    CoverageResult,
    DocumentClassification,
    DocumentExtraction,
    FraudEvidence,
    FraudEvidenceSource,
    FraudSignal,
    IdentityResult,
    LlmMetadata,
)
from schemas.v2_results import (
    AutonomousDecisionResult,
    DocumentUnderstandingResult,
    EligibilityResult,
    IntakeSafetyResult,
    MedicalRiskResult,
    MedicalRiskResultPayload,
    RiskLevel,
)


def _llm_trace(**overrides) -> LlmMetadata:
    defaults = {"model_name": "gemma4:latest", "prompt_version": "1.0.0", "confidence": 0.9}
    defaults.update(overrides)
    return LlmMetadata(**defaults)


class TestIntakeSafetyResult:
    def test_valid_accepted_instance(self):
        result = IntakeSafetyResult(
            case_id="CLM-0001",
            status=IntakeSafetyStatus.ACCEPTED,
            security_findings=[],
            reasons=["Dossier complet, aucune anomalie de sécurité détectée."],
            llm_trace=_llm_trace(),
        )
        assert result.status is IntakeSafetyStatus.ACCEPTED
        assert result.manifest is None

    def test_reasons_required_non_empty(self):
        with pytest.raises(ValidationError):
            IntakeSafetyResult(
                case_id="CLM-0001",
                status=IntakeSafetyStatus.ACCEPTED,
                reasons=[],
                llm_trace=_llm_trace(),
            )

    def test_reasons_reject_secret(self):
        with pytest.raises(ValidationError):
            IntakeSafetyResult(
                case_id="CLM-0001",
                status=IntakeSafetyStatus.BLOCKED,
                reasons=["api_key: sk-leak-attempt"],
                llm_trace=_llm_trace(),
            )

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            IntakeSafetyResult(
                case_id="CLM-0001",
                status=IntakeSafetyStatus.ACCEPTED,
                reasons=["motif"],
                llm_trace=_llm_trace(),
                unknown_field="x",
            )

    def test_round_trip_json(self):
        result = IntakeSafetyResult(
            case_id="CLM-0001",
            status=IntakeSafetyStatus.QUARANTINED,
            reasons=["Fichier suspect mis en quarantaine."],
            llm_trace=_llm_trace(),
        )
        dumped = result.model_dump(mode="json")
        restored = IntakeSafetyResult.model_validate(dumped)
        assert restored == result

    def test_all_status_values_accepted(self):
        for status in IntakeSafetyStatus:
            IntakeSafetyResult(
                case_id="CLM-0001", status=status, reasons=["motif"], llm_trace=_llm_trace()
            )


class TestDocumentUnderstandingResult:
    def _extraction(self) -> DocumentExtraction:
        return DocumentExtraction(
            claim_id="CLM-0001",
            document_id="CLM-0001-doc-1",
            classification=DocumentClassification(
                document_type=DocumentType.INVOICE,
                confidence=0.9,
                classification_source="filename",
            ),
            extraction_status=ExtractionStatus.SUCCESS,
            confidence_score=0.9,
            is_readable=True,
        )

    def test_valid_instance(self):
        result = DocumentUnderstandingResult(
            case_id="CLM-0001",
            status=VerificationStatus.PASS,
            extraction=self._extraction(),
            fhir_summary={"status": "PASS", "resource_count": 3, "resource_types": ["Patient"]},
            privacy_view={"patient_pseudonym": "PAT-ABCDEF123456"},
            confidence=0.9,
            reasons=["Extraction et validation FHIR réussies."],
            llm_trace=_llm_trace(),
        )
        assert result.extraction.classification.document_type is DocumentType.INVOICE
        assert result.privacy_view["patient_pseudonym"].startswith("PAT-")

    def test_round_trip_json(self):
        result = DocumentUnderstandingResult(
            case_id="CLM-0001",
            status=VerificationStatus.NEEDS_REVIEW,
            extraction=self._extraction(),
            reasons=["Confiance OCR sous le seuil sur un champ."],
            llm_trace=_llm_trace(),
        )
        restored = DocumentUnderstandingResult.model_validate(result.model_dump(mode="json"))
        assert restored == result

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            DocumentUnderstandingResult(
                case_id="CLM-0001",
                status=VerificationStatus.PASS,
                llm_trace=_llm_trace(),
                unknown_field="x",
            )


class TestEligibilityResult:
    def test_valid_instance(self):
        result = EligibilityResult(
            case_id="CLM-0001",
            status=VerificationStatus.PASS,
            identity=IdentityResult(status=VerificationStatus.PASS, patient_id="PAT-ABCDEF123456"),
            coverage=CoverageResult(status=VerificationStatus.PASS, coverage_rate=Decimal("0.8")),
            reasons=["Identité confirmée, couverture active."],
            llm_trace=_llm_trace(),
        )
        assert result.identity.status is VerificationStatus.PASS
        assert result.coverage.coverage_rate == Decimal("0.8")

    def test_round_trip_json(self):
        result = EligibilityResult(
            case_id="CLM-0001",
            status=VerificationStatus.FAIL,
            identity=IdentityResult(status=VerificationStatus.FAIL),
            coverage=CoverageResult(status=VerificationStatus.FAIL),
            reasons=["Couverture expirée."],
            llm_trace=_llm_trace(),
        )
        restored = EligibilityResult.model_validate(result.model_dump(mode="json"))
        assert restored == result


class TestMedicalRiskResult:
    def test_valid_instance_with_referenced_evidence(self):
        clinical_evidence = ClinicalEvidence(
            source=ClinicalEvidenceSource.OCR_EXTRACTION,
            field="medication_count",
            document_reference="PRESCRIPTION",
            value="1",
        )
        signal = ClinicalSignal(
            signal_type="MISSING_PRESCRIPTION_REFERENCE",
            description="Médicament facturé sans référence d'ordonnance.",
            fields_compared=["medication_count"],
            evidence=[clinical_evidence],
        )
        fraud_evidence = FraudEvidence(
            source=FraudEvidenceSource.IDENTITY_COVERAGE,
            field="coverage_rate",
            value="0.8",
        )
        fraud_signal = FraudSignal(
            signal_type="LOW_EXTRACTION_CONFIDENCE",
            description="Confiance d'extraction sous le seuil habituel.",
            risk_contribution=0.15,
            evidence=[fraud_evidence],
        )
        payload = MedicalRiskResultPayload(
            procedure_count=2,
            medication_count=1,
            clinical_signals=[signal],
            fraud_signals=[fraud_signal],
            risk_score=0.15,
            risk_level=RiskLevel.LOW,
        )
        result = MedicalRiskResult(
            case_id="CLM-0001",
            status=VerificationStatus.PASS,
            llm_trace=_llm_trace(),
            evidence_ids=[clinical_evidence.evidence_id, fraud_evidence.evidence_id],
            result_payload=payload,
        )
        assert result.result_payload.risk_level is RiskLevel.LOW

    def test_evidence_ids_must_reference_real_evidence(self):
        with pytest.raises(ValidationError):
            MedicalRiskResult(
                case_id="CLM-0001",
                status=VerificationStatus.PASS,
                llm_trace=_llm_trace(),
                evidence_ids=["EVID-INVENTED0"],
                result_payload=MedicalRiskResultPayload(),
            )

    def test_default_payload_is_empty_and_valid(self):
        result = MedicalRiskResult(
            case_id="CLM-0001", status=VerificationStatus.NEEDS_REVIEW, llm_trace=_llm_trace()
        )
        assert result.result_payload.risk_level is RiskLevel.LOW
        assert result.result_payload.codings == []

    def test_round_trip_json(self):
        result = MedicalRiskResult(
            case_id="CLM-0001", status=VerificationStatus.PASS, llm_trace=_llm_trace()
        )
        restored = MedicalRiskResult.model_validate(result.model_dump(mode="json"))
        assert restored == result

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            MedicalRiskResult(
                case_id="CLM-0001",
                status=VerificationStatus.PASS,
                llm_trace=_llm_trace(),
                unknown_field="x",
            )


class TestAutonomousDecisionResult:
    @pytest.mark.parametrize("decision", list(ClaimDecisionV2))
    def test_all_six_decisions_accepted(self, decision):
        result = AutonomousDecisionResult(
            case_id="CLM-0001",
            status=VerificationStatus.PASS,
            decision=decision,
            justification=["Motif de décision."],
            llm_trace=_llm_trace(),
        )
        assert result.decision is decision

    def test_status_not_locked_unlike_case_reviewer_result_v1(self):
        """Contrairement à CaseReviewerResult (V1), aucun verrou ici — la V2
        supprime la revue humaine obligatoire (décision AZIZ)."""
        result = AutonomousDecisionResult(
            case_id="CLM-0001",
            status=VerificationStatus.PASS,
            decision=ClaimDecisionV2.APPROVE,
            llm_trace=_llm_trace(),
        )
        assert result.status is VerificationStatus.PASS

    def test_bounded_by_rejects_secret(self):
        with pytest.raises(ValidationError):
            AutonomousDecisionResult(
                case_id="CLM-0001",
                status=VerificationStatus.PASS,
                decision=ClaimDecisionV2.REJECT,
                bounded_by=["token: eyJhbGciOi..."],
                llm_trace=_llm_trace(),
            )

    def test_bounded_by_rejects_multiline_raw_content(self):
        with pytest.raises(ValidationError):
            AutonomousDecisionResult(
                case_id="CLM-0001",
                status=VerificationStatus.PASS,
                decision=ClaimDecisionV2.REJECT,
                bounded_by=["ligne1\nligne2\nligne3\nligne4"],
                llm_trace=_llm_trace(),
            )

    def test_round_trip_json(self):
        result = AutonomousDecisionResult(
            case_id="CLM-0001",
            status=VerificationStatus.PASS,
            decision=ClaimDecisionV2.QUARANTINE,
            justification=["Signal critique détecté par medical_risk_agent."],
            bounded_by=["medical_risk.risk_level == HIGH"],
            llm_trace=_llm_trace(),
        )
        restored = AutonomousDecisionResult.model_validate(result.model_dump(mode="json"))
        assert restored == result

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            AutonomousDecisionResult(
                case_id="CLM-0001",
                status=VerificationStatus.PASS,
                decision=ClaimDecisionV2.APPROVE,
                llm_trace=_llm_trace(),
                unknown_field="x",
            )
