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
    CaseReviewerResultPayload,
    ClaimIntakeResult,
    ClaimManifest,
    ClinicalConsistencyResult,
    ClinicalEvidence,
    ClinicalEvidenceSource,
    ClinicalResultPayload,
    ClinicalSignal,
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
    assert result.result_payload.recommendation is Recommendation.APPROVE
    assert result.human_review_required is True
    assert result.llm_trace.model_name


def test_case_reviewer_transmet_preuves_risques_et_contradictions_au_llm():
    """Le LLM ne peut citer que des preuves/risques/contradictions déjà
    calculés par la Phase A — transmis explicitement dans le payload."""
    calls: list[dict] = []

    def _llm(data: dict):
        calls.append(data)
        return _decision(Recommendation.APPROVE)

    with patch.object(agent, "_invoke_llm_case_review", side_effect=_llm):
        agent.run("CLM-0001", _clean_state())

    payload = calls[0]
    assert "risks" in payload
    assert "evidence_ids" in payload
    assert "disagreement_ids" in payload
    assert isinstance(payload["risks"], list)
    assert isinstance(payload["evidence_ids"], list)
    assert isinstance(payload["disagreement_ids"], list)


def test_case_reviewer_llm_indisponible_fail_closed_pending():
    with patch.object(agent, "_invoke_llm_case_review", return_value=None):
        result = agent.run("CLM-0001", _clean_state())

    assert result.result_payload.recommendation is Recommendation.PENDING
    assert result.human_review_required is True
    assert any("LLM indisponible" in reason for reason in result.result_payload.justification)
    assert any(err.code == "LLM_UNAVAILABLE" for err in result.errors)


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

    assert result.result_payload.recommendation is Recommendation.REJECT
    assert result.human_review_required is True


def test_case_reviewer_llm_ne_peut_pas_approuver_preuves_incompletes():
    with patch.object(agent, "_invoke_llm_case_review", return_value=_decision(Recommendation.APPROVE)):
        result = agent.run("CLM-0001", {"case_id": "CLM-0001"})

    assert result.result_payload.recommendation is Recommendation.PENDING
    assert any("manquants" in reason for reason in result.result_payload.justification)


def test_llm_case_review_schema_refuse_controle_revue_humaine():
    with pytest.raises(ValidationError):
        LlmCaseReviewDecision.model_validate({
            "recommendation": "APPROVE",
            "summary": "Synthèse.",
            "reasons": ["Motif."],
            "human_review_required": False,
        })


def test_case_reviewer_result_refuse_human_review_required_false():
    """Garantie de schéma : aucune instance valide ne peut désactiver la revue
    humaine, quelle que soit l'implémentation qui tente de la construire."""
    with pytest.raises(ValidationError):
        CaseReviewerResult(
            case_id="CLM-0001",
            llm_trace=_llm_metadata(),
            human_review_required=False,
            result_payload=CaseReviewerResultPayload(
                recommendation=Recommendation.APPROVE,
                justification=["Pré-approbation injectée."],
                human_review_reasons=["Motif."],
            ),
        )


def test_case_reviewer_result_refuse_status_final():
    """Garantie de schéma : ``status`` ne peut jamais représenter une
    décision finale (PASS/FAIL) — toujours NEEDS_REVIEW."""
    with pytest.raises(ValidationError):
        CaseReviewerResult(
            case_id="CLM-0001",
            status=VerificationStatus.PASS,
            llm_trace=_llm_metadata(),
            result_payload=CaseReviewerResultPayload(
                recommendation=Recommendation.APPROVE,
                human_review_reasons=["Motif."],
            ),
        )


def test_case_reviewer_node_force_human_review_sur_impl_injectee():
    """Défense en profondeur côté nœud : même une implémentation injectée
    valide-mais-incomplète (motif standard absent) voit ce motif rétabli."""

    class _IncompleteImpl:
        def run(self, state):
            return CaseReviewerResult(
                case_id=str(state.get("case_id")),
                llm_trace=_llm_metadata(),
                result_payload=CaseReviewerResultPayload(
                    recommendation=Recommendation.APPROVE,
                    justification=["Pré-approbation injectée."],
                    human_review_reasons=["Motif spécifique à l'implémentation injectée."],
                ),
            )

    node_fn = agent.make_node(_IncompleteImpl())
    updates = node_fn({"case_id": "CLM-0001"})

    assert updates["review_result"].human_review_required is True
    assert (
        "Validation humaine obligatoire avant toute décision finale."
        in updates["review_result"].result_payload.human_review_reasons
    )
    assert updates.get("alerts")
    assert isinstance(updates["audit_trail"][0], AuditEvent)


# ── Citations anti-hallucination (preuves/risques/contradictions) ───────────


def test_case_reviewer_ignore_les_references_inventees_par_le_llm():
    """Une preuve/un risque/une contradiction cité par le LLM mais absent des
    valeurs réellement calculées est silencieusement ignoré, jamais accepté."""
    fake_decision = LlmCaseReviewDecision(
        recommendation=Recommendation.APPROVE,
        summary="Synthèse multi-agent de test.",
        reasons=["Tous les contrôles disponibles ont été synthétisés."],
        referenced_evidence_ids=["EVID-invente0000"],
        acknowledged_risks=["Risque totalement inventé par le LLM."],
        acknowledged_disagreements=["agent_inconnu.champ_inconnu"],
        human_review_reasons=["Validation humaine requise."],
    )
    with patch.object(agent, "_invoke_llm_case_review", return_value=fake_decision):
        result = agent.run("CLM-0001", _clean_state())

    assert any(
        "références ignorées" in reason for reason in result.result_payload.justification
    )


def test_case_reviewer_accepte_les_references_reelles_du_llm():
    """Une citation correspondant à un risque réellement calculé est acceptée
    sans déclencher le motif de références ignorées."""
    state = _clean_state()
    state["fraud_result"] = FraudDetectionResult(
        case_id="CLM-0001",
        status=VerificationStatus.NEEDS_REVIEW,
        llm_trace=_llm_metadata(),
        result_payload=FraudResultPayload(risk_score=0.75),
    )
    real_risk = "Score de risque de fraude élevé (0.75)."
    decision = LlmCaseReviewDecision(
        recommendation=Recommendation.PENDING,
        summary="Synthèse multi-agent de test.",
        reasons=["Tous les contrôles disponibles ont été synthétisés."],
        acknowledged_risks=[real_risk],
        human_review_reasons=["Validation humaine requise."],
    )
    with patch.object(agent, "_invoke_llm_case_review", return_value=decision):
        result = agent.run("CLM-0001", state)

    assert not any(
        "références ignorées" in reason for reason in result.result_payload.justification
    )
    assert real_risk in result.result_payload.risks


# ── Interdictions de contenu (paiement, diagnostic, accusation, validation) ──


@pytest.mark.parametrize(
    "forbidden_summary",
    [
        "Remboursement validé, le montant peut être payé.",
        "Le patient souffre de diabète confirmé.",
        "La fraude est confirmée sur ce dossier.",
        "Ceci est la décision finale du dossier.",
    ],
)
def test_llm_case_review_rejette_les_formulations_interdites(forbidden_summary: str):
    with pytest.raises(ValidationError):
        LlmCaseReviewDecision(
            recommendation=Recommendation.APPROVE,
            summary=forbidden_summary,
            reasons=["Motif."],
        )


@pytest.mark.parametrize(
    "allowed_summary",
    [
        "Ceci n'est jamais une décision finale, une revue humaine est requise.",
        "Aucun diagnostic n'est posé par cet agent.",
        "Aucune fraude n'est confirmée à ce stade, seul un risque est signalé.",
    ],
)
def test_llm_case_review_tolere_les_negations_explicites(allowed_summary: str):
    decision = LlmCaseReviewDecision(
        recommendation=Recommendation.PENDING,
        summary=allowed_summary,
        reasons=["Motif."],
    )
    assert decision.summary == allowed_summary


def test_llm_case_review_decision_a_des_champs_de_citation_par_defaut_vides():
    decision = LlmCaseReviewDecision(
        recommendation=Recommendation.PENDING,
        summary="Synthèse.",
        reasons=["Motif."],
    )
    assert decision.referenced_evidence_ids == []
    assert decision.acknowledged_risks == []
    assert decision.acknowledged_disagreements == []


def test_llm_case_review_decision_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        LlmCaseReviewDecision(
            recommendation=Recommendation.PENDING,
            summary="Synthèse.",
            reasons=["Motif."],
            unexpected=True,
        )


# ── Traçabilité d'audit : llm_call_id, prompt_version, preuves, statut ───────


def test_audit_event_carries_llm_traceability_fields_and_status():
    """Audite llm_call_id, model_name, prompt_version, evidence_ids et
    statut — traçabilité obligatoire de l'appel LLM, jamais un contenu brut."""
    updates = agent.make_node()(_clean_state())
    details = updates["audit_trail"][0].details
    assert details["llm_call_id"]
    assert details["model_name"]
    assert details["prompt_version"]
    assert "evidence_ids" in details
    assert details["status"] == "NEEDS_REVIEW"
    assert "errors" in details


def test_llm_call_id_is_unique_per_execution():
    node_fn = agent.make_node()
    first = node_fn(_clean_state())["audit_trail"][0].details["llm_call_id"]
    second = node_fn(_clean_state())["audit_trail"][0].details["llm_call_id"]
    assert first != second


def test_audit_evidence_ids_reflects_upstream_evidence():
    evidence = ClinicalEvidence(
        source=ClinicalEvidenceSource.OCR_EXTRACTION, field="medication_count", value="2"
    )
    state = _clean_state()
    state["clinical_result"] = ClinicalConsistencyResult(
        case_id="CLM-0001",
        status=VerificationStatus.PASS,
        llm_trace=_llm_metadata(),
        evidence_ids=[evidence.evidence_id],
        result_payload=ClinicalResultPayload(
            signals=[
                ClinicalSignal(
                    signal_type="X",
                    description="d",
                    fields_compared=["medication_count"],
                    evidence=[evidence],
                )
            ]
        ),
    )
    updates = agent.make_node()(state)
    details = updates["audit_trail"][0].details
    assert details["evidence_ids"] == evidence.evidence_id


def test_audit_status_is_always_needs_review_never_an_auto_approval():
    """Le statut audité reflète toujours le verrouillage du schéma — jamais
    une valeur qui laisserait croire à une décision finale automatique."""
    with patch.object(agent, "_invoke_llm_case_review", return_value=_decision(Recommendation.APPROVE)):
        updates = agent.make_node()(_clean_state())
    assert updates["audit_trail"][0].details["status"] == "NEEDS_REVIEW"
    assert updates["review_result"].human_review_required is True
