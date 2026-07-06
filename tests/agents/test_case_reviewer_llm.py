"""Tests LLM du Case Reviewer Agent — pré-recommandation non finale."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agents.case_reviewer_agent import agent
from agents.case_reviewer_agent.schemas import LlmCaseReviewDecision
from schemas.domain import (
    DataClassification,
    DocumentType,
    ExtractionStatus,
    IntakeStatus,
    OcrSource,
    Recommendation,
    SecurityDecision,
    VerificationStatus,
)
from schemas.results import (
    AuditEvent,
    CaseReviewerResult,
    ClaimIntakeResult,
    ClaimManifest,
    ClinicalConsistencyResult,
    CoverageResult,
    DocumentOcrResult,
    FhirValidatorResult,
    FraudDetectionResult,
    FraudResultPayload,
    IdentityCoverageResult,
    IdentityResult,
    LlmMetadata,
    MedicalCodingResult,
    PrivacyResult,
    SecurityGateResult,
)


def _llm_metadata() -> LlmMetadata:
    return LlmMetadata(model_name="test-llm", prompt_version="test")


def _clean_state() -> dict:
    case_id = "CLM-0001"
    return {
        "case_id": case_id,
        "intake_result": ClaimIntakeResult(
            claim_id=case_id,
            status=IntakeStatus.ACCEPTED,
            manifest=ClaimManifest(
                claim_id=case_id,
                file_count=1,
                total_size_bytes=10,
                status=IntakeStatus.ACCEPTED,
            ),
            accepted_count=1,
            quarantined_count=0,
            llm_metadata=_llm_metadata(),
        ),
        "security_result": SecurityGateResult(
            claim_id=case_id,
            decision=SecurityDecision.ALLOW,
            reasons=["Autorisé."],
        ),
        "privacy_result": PrivacyResult(
            case_id=case_id,
            status=VerificationStatus.PASS,
            data_classification=DataClassification.SYNTHETIC_TEST_DATA,
            contains_real_personal_data=False,
        ),
        "identity_coverage_result": IdentityCoverageResult(
            case_id=case_id,
            identity=IdentityResult(status=VerificationStatus.PASS),
            coverage=CoverageResult(status=VerificationStatus.PASS),
        ),
        "fhir_result": FhirValidatorResult(
            case_id=case_id,
            status=VerificationStatus.PASS,
            bundle_expected=True,
            llm_metadata=_llm_metadata(),
        ),
        "ocr_result": DocumentOcrResult(
            claim_id=case_id,
            file_path="incoming/CLM-0001/facture.pdf",
            sha256="a" * 64,
            mime_type="application/pdf",
            extraction_status=ExtractionStatus.SUCCESS,
            status=VerificationStatus.PASS,
            document_type=DocumentType.INVOICE,
            ocr_source=OcrSource.PDF_TEXT,
        ),
        "coding_result": MedicalCodingResult(
            case_id=case_id,
            status=VerificationStatus.PASS,
            llm_metadata=_llm_metadata(),
        ),
        "clinical_result": ClinicalConsistencyResult(
            case_id=case_id,
            status=VerificationStatus.PASS,
            llm_trace=_llm_metadata(),
        ),
        "fraud_result": FraudDetectionResult(
            case_id=case_id,
            status=VerificationStatus.PASS,
            llm_trace=_llm_metadata(),
        ),
        "errors": [],
        "alerts": [],
    }


def _decision(recommendation: Recommendation) -> LlmCaseReviewDecision:
    return LlmCaseReviewDecision(
        recommendation=recommendation,
        summary="Synthèse multi-agent de test.",
        reasons=["Tous les contrôles disponibles ont été synthétisés."],
        human_review_reasons=["Validation humaine requise."],
    )


def test_case_reviewer_appelle_llm_et_synthetise_tous_les_agents():
    calls: list[dict] = []

    def _llm(data: dict):
        calls.append(data)
        return _decision(Recommendation.APPROVE)

    with patch.object(agent, "_invoke_llm_case_review", side_effect=_llm):
        result = agent.run("CLM-0001", _clean_state())

    assert len(calls) == 1
    assert set(calls[0]["agent_results"]) == set(agent._EXPECTED_UPSTREAM_AGENTS)
    assert result.recommendation is Recommendation.APPROVE
    assert result.human_review_required is True
    assert result.llm_metadata.model_name


def test_case_reviewer_llm_indisponible_fail_closed_pending():
    with patch.object(agent, "_invoke_llm_case_review", return_value=None):
        result = agent.run("CLM-0001", _clean_state())

    assert result.recommendation is Recommendation.PENDING
    assert result.human_review_required is True
    assert any("LLM indisponible" in reason for reason in result.justification)


def test_case_reviewer_llm_ne_peut_pas_assouplir_rejet_deterministe():
    state = _clean_state()
    state["fraud_result"] = FraudDetectionResult(
        case_id="CLM-0001",
        status=VerificationStatus.FAIL,
        llm_trace=_llm_metadata(),
        result_payload=FraudResultPayload(risk_score=0.9),
    )

    with patch.object(agent, "_invoke_llm_case_review", return_value=_decision(Recommendation.APPROVE)):
        result = agent.run("CLM-0001", state)

    assert result.recommendation is Recommendation.REJECT
    assert result.human_review_required is True


def test_case_reviewer_llm_ne_peut_pas_approuver_preuves_incompletes():
    with patch.object(agent, "_invoke_llm_case_review", return_value=_decision(Recommendation.APPROVE)):
        result = agent.run("CLM-0001", {"case_id": "CLM-0001"})

    assert result.recommendation is Recommendation.PENDING
    assert any("manquants" in reason for reason in result.justification)


def test_llm_case_review_schema_refuse_controle_revue_humaine():
    with pytest.raises(ValidationError):
        LlmCaseReviewDecision.model_validate({
            "recommendation": "APPROVE",
            "summary": "Synthèse.",
            "reasons": ["Motif."],
            "human_review_required": False,
        })


def test_case_reviewer_node_force_human_review_sur_impl_injectee():
    class _UnsafeImpl:
        def run(self, state):
            return CaseReviewerResult(
                case_id=str(state.get("case_id")),
                recommendation=Recommendation.APPROVE,
                justification=["Pré-approbation injectée."],
                human_review_required=False,
                human_review_reasons=[],
                llm_metadata=_llm_metadata(),
            )

    node_fn = agent.make_node(_UnsafeImpl())
    updates = node_fn({"case_id": "CLM-0001"})

    assert updates["review_result"].human_review_required is True
    assert updates.get("alerts")
    assert isinstance(updates["audit_trail"][0], AuditEvent)
