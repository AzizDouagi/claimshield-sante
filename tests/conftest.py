"""Configuration pytest partagée pour tous les tests ClaimShield Santé.

Nettoie le répertoire storage/ partagé (zones incoming/, quarantine/, temporary/)
avant chaque test afin d'éviter les collisions liées aux runs précédents.
Les tests qui nécessitent un stockage isolé injectent leur propre StorageService
via tmp_path — ce fixture ne les affecte pas.
"""
from __future__ import annotations

import shutil
from unittest.mock import MagicMock, patch

import pytest

from config.settings import get_settings
from llm.factory import reset_llm_cache


@pytest.fixture(autouse=True)
def clean_shared_storage() -> None:
    """Supprime les sous-dossiers CLM-* dans les zones de stockage partagées."""
    s = get_settings()
    zones = [
        s.storage_dir / "incoming",
        s.storage_dir / "temporary",
        s.quarantine_dir,
    ]
    for zone in zones:
        if zone.exists():
            for clm_dir in zone.glob("CLM-*"):
                shutil.rmtree(clm_dir, ignore_errors=True)
    yield


@pytest.fixture(autouse=False)
def mock_llm(request):
    """Fixture générique pour mocker ChatOllama."""
    response = getattr(request, "param", None)
    reset_llm_cache()
    with patch("llm.factory.ChatOllama") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        if response is not None:
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = response
            mock_instance.with_structured_output.return_value = mock_structured
        yield mock_instance
    reset_llm_cache()


@pytest.fixture(autouse=True)
def deterministic_agent_llm(monkeypatch) -> None:
    """Donne aux tests hérités une réponse LLM stable et alignée sur la Phase A."""
    from agents.case_reviewer_agent.schemas import LlmCaseReviewDecision
    from agents.claim_intake_agent.schemas import LlmIntakeDecision
    from agents.clinical_consistency_agent.schemas import LlmClinicalDecision
    from agents.document_ocr_agent.schemas import LlmOcrDecision
    from agents.fhir_validator_agent.schemas import LlmFhirDecision
    from agents.fraud_detection_agent.schemas import LlmFraudDecision
    from agents.identity_coverage_agent.schemas import LlmIdentityCoverageDecision
    from agents.medical_coding_agent.schemas import LlmCodingDecision
    from agents.privacy_agent.schemas import LlmPrivacyDecision
    from agents.security_gate_agent.schemas import LlmSecurityDecision
    from schemas.domain import Recommendation

    def intake_decision(**kwargs):
        status = str(kwargs["global_status"]).upper()
        reasons = list(kwargs.get("alerts") or [])
        if not reasons:
            reasons = [f"Décision d'ingestion déterministe conservée : {status}."]
        return LlmIntakeDecision(
            status=status,
            reasons=reasons,
        )

    def security_decision(**kwargs):
        decision = kwargs["deterministic_decision"]
        findings = kwargs.get("findings") or []
        reasons = [
            f.get("description", "Anomalie de sécurité détectée.")
            for f in findings
        ] or ["Aucune menace détectée — dossier autorisé"]
        return LlmSecurityDecision(
            decision=decision,
            reasons=reasons,
            explanation=f"Décision LLM de test alignée sur la Phase A : {decision}.",
        )

    def fhir_decision(**kwargs):
        status = kwargs["deterministic_status"]
        return LlmFhirDecision(
            recommended_status=status,
            clinical_context="Décision FHIR de test alignée sur la validation structurelle.",
            reasons=[f"Statut structurel conservé : {status}."],
        )

    def privacy_decision(data):
        return LlmPrivacyDecision(
            audit_justification="Justification privacy de test alignée sur RBAC.",
            data_classification_reason=(
                f"Classification conservée : {data.get('data_classification', 'UNKNOWN')}."
            ),
        )

    def ocr_decision(data):
        return LlmOcrDecision(
            document_type=data.get("deterministic_document_type", "UNKNOWN"),
            extracted_fields={},
            confidence_assessment="Décision OCR de test alignée sur les outils déterministes.",
            reasons=[],
        )

    def identity_coverage_decision(data):
        return LlmIdentityCoverageDecision(
            recommended_identity_status=data.get("identity_status", "NEEDS_REVIEW"),
            recommended_coverage_status=data.get("coverage_status", "NEEDS_REVIEW"),
            rationale="",
            warnings=[],
        )

    def coding_decision(*_args, **_kwargs):
        return LlmCodingDecision(resolved=[], overall_rationale="")

    def clinical_decision(data):
        return LlmClinicalDecision(
            clinical_context="Contexte clinique de test aligné sur les signaux déterministes.",
            reasons=[],
        )

    def fraud_decision(data):
        return LlmFraudDecision(
            rationale="Justification anti-fraude de test alignée sur les signaux déterministes.",
            reasons=[],
        )

    def case_review_decision(data):
        recommendation = data.get("deterministic_pre_recommendation", "PENDING")
        return LlmCaseReviewDecision(
            recommendation=Recommendation(recommendation),
            summary="Synthèse reviewer de test alignée sur les résultats agents.",
            reasons=["Pré-recommandation LLM de test non finale."],
            human_review_reasons=["Validation humaine obligatoire en test."],
        )

    monkeypatch.setattr("agents.claim_intake_agent.agent._invoke_llm_intake", intake_decision)
    monkeypatch.setattr("agents.security_gate_agent.agent._invoke_llm_security", security_decision)
    monkeypatch.setattr("agents.fhir_validator_agent.agent._invoke_llm_fhir", fhir_decision)
    monkeypatch.setattr(
        "agents.identity_coverage_agent.agent._invoke_llm_identity_coverage",
        identity_coverage_decision,
    )
    monkeypatch.setattr("agents.medical_coding_agent.agent._invoke_llm_react", coding_decision)
    monkeypatch.setattr("agents.privacy_agent.agent._invoke_llm_privacy", privacy_decision)
    monkeypatch.setattr("agents.document_ocr_agent.agent._invoke_llm_ocr", ocr_decision)
    monkeypatch.setattr("agents.clinical_consistency_agent.agent._invoke_llm_clinical", clinical_decision)
    monkeypatch.setattr("agents.fraud_detection_agent.agent._invoke_llm_fraud", fraud_decision)
    monkeypatch.setattr("agents.case_reviewer_agent.agent._invoke_llm_case_review", case_review_decision)
    yield
