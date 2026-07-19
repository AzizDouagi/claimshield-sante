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
    """Supprime les sous-dossiers CLM-* dans les zones de stockage partagées.

    Inclut `manifests/` (fichiers `CLM-*.json`, pas des répertoires) depuis le
    plan de remédiation « rejouabilité des dossiers » (V2, phase 1) :
    `agents/intake_safety_agent/agent.py::_load_previous_active_files` relit
    désormais le manifeste d'un run précédent pour détecter une révision de
    document — un manifeste non nettoyé entre deux tests utilisant le même
    `case_id` littéral (ex. `tests/v2/graph/test_workflow_v2.py`) serait sinon
    interprété à tort comme une soumission déjà connue de ce dossier.
    """
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
    manifests_dir = s.storage_dir / "manifests"
    if manifests_dir.exists():
        for manifest_file in manifests_dir.glob("CLM-*.json"):
            manifest_file.unlink(missing_ok=True)
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
    from agents.clinical_consistency_agent.schemas import LlmClinicalDecision
    from agents.document_ocr_agent.schemas import LlmOcrDecision
    from agents.fhir_validator_agent.schemas import LlmFhirDecision
    from agents.fraud_detection_agent.schemas import LlmFraudDecision
    from agents.identity_coverage_agent.schemas import LlmIdentityCoverageDecision
    from agents.medical_coding_agent.schemas import LlmCodingDecision

    def fhir_decision(**kwargs):
        status = kwargs["deterministic_status"]
        return LlmFhirDecision(
            recommended_status=status,
            clinical_context="Décision FHIR de test alignée sur la validation structurelle.",
            reasons=[f"Statut structurel conservé : {status}."],
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

    monkeypatch.setattr("agents.fhir_validator_agent.agent._invoke_llm_fhir", fhir_decision)
    monkeypatch.setattr(
        "agents.identity_coverage_agent.agent._invoke_llm_identity_coverage",
        identity_coverage_decision,
    )
    monkeypatch.setattr("agents.medical_coding_agent.agent._invoke_llm_react", coding_decision)
    monkeypatch.setattr("agents.document_ocr_agent.agent._invoke_llm_ocr", ocr_decision)
    monkeypatch.setattr("agents.clinical_consistency_agent.agent._invoke_llm_clinical", clinical_decision)
    monkeypatch.setattr("agents.fraud_detection_agent.agent._invoke_llm_fraud", fraud_decision)
    yield
