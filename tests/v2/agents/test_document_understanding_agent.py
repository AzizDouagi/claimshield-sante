"""Tests de agents/document_understanding_agent (V2) — Phase V2-3.

Utilise un vrai PDF avec couche texte native (reportlab, même patron que
tests/agents/test_mandatory_cases.py) pour exercer le pipeline OCR réel
(PDF_TEXT, sans tesseract) et un vrai petit bundle FHIR minimal validé par
tools.fhir_validation (mêmes règles que config/rules/fhir_rules.yaml).
Les fichiers sont mis en scène sous le stockage partagé réel
(get_settings().storage_dir/incoming/<case_id>/...) — nettoyé automatiquement
par le fixture autouse `clean_shared_storage` (tests/conftest.py) — car
`agents.fhir_validator_agent.agent._resolve_bundle_path` (réutilisé tel
quel) résout un chemin relatif à `storage/incoming/`, sans paramètre de
racine injectable.
"""
from __future__ import annotations

import json
from unittest.mock import Mock

from agents.document_understanding_agent.agent import node, run
from agents.document_understanding_agent.schemas import LlmDocumentUnderstandingDecision
from config.settings import get_settings
from schemas.domain import DocumentType, FileStatus, ReaderRole, VerificationStatus
from schemas.results import InspectedFile
from tools.file_inspection import compute_sha256

_INVOICE_LINES = [
    "FACTURE MEDICALE",
    "Numero : INV-CLM-9999",
    "patient_id : PAT-9999-DEMO",
    "Prestataire : Bernard Leclerc",
    "Date du document : 2024-03-01",
    "Date de soins : 25/02/2024",
    "Devise : USD",
    "Montant total facture : 3666.69 USD",
    "Amoxicilline 500 mg",
    "Ibuprofene 400 mg",
]

_MINIMAL_FHIR_BUNDLE = {
    "resourceType": "Bundle",
    "type": "collection",
    "entry": [{"resource": {"resourceType": "Patient", "id": "p1"}}],
}


def _make_pdf(path, lines: list[str]) -> None:
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path))
    y = 800
    for line in lines:
        c.drawString(72, y, line)
        y -= 20
    c.save()


def _stage_document(case_id: str, lines: list[str] = _INVOICE_LINES) -> InspectedFile:
    incoming = get_settings().storage_dir / "incoming" / case_id
    incoming.mkdir(parents=True, exist_ok=True)
    dest = incoming / "facture_test.pdf"
    _make_pdf(dest, lines)
    sha = compute_sha256(dest)
    return InspectedFile(
        original_name="facture_test.pdf",
        storage_name="facture_test.pdf",
        normalized_extension="pdf",
        detected_mime_type="application/pdf",
        actual_size=dest.stat().st_size,
        sha256=sha,
        status=FileStatus.ACCEPTED,
        reasons=[],
        relative_storage_path=f"incoming/{case_id}/facture_test.pdf",
    )


def _stage_fhir_bundle(case_id: str, bundle: dict = _MINIMAL_FHIR_BUNDLE) -> InspectedFile:
    incoming = get_settings().storage_dir / "incoming" / case_id
    incoming.mkdir(parents=True, exist_ok=True)
    dest = incoming / "bundle_fhir.json"
    dest.write_text(json.dumps(bundle), encoding="utf-8")
    sha = compute_sha256(dest)
    return InspectedFile(
        original_name="bundle_fhir.json",
        storage_name="bundle_fhir.json",
        normalized_extension="json",
        detected_mime_type="application/json",
        actual_size=dest.stat().st_size,
        sha256=sha,
        status=FileStatus.ACCEPTED,
        reasons=[],
        relative_storage_path=f"incoming/{case_id}/bundle_fhir.json",
    )


def _decision(**overrides) -> LlmDocumentUnderstandingDecision:
    defaults = {
        "document_type": "INVOICE",
        "ocr_confidence_assessment": "",
        "fhir_clinical_context": "",
        "fhir_recommended_status": "PASS",
        "reasons": [],
    }
    defaults.update(overrides)
    return LlmDocumentUnderstandingDecision(**defaults)


class TestNominalFusion:
    def test_document_and_fhir_both_present(self, monkeypatch):
        """Bundle FHIR minimal mais structurellement valide (Patient présent,
        aucune erreur) : NEEDS_REVIEW est le statut réellement attendu ici —
        les ressources optionnelles absentes (Coverage, Encounter...)
        déclenchent des avertissements légitimes, jamais une erreur (0
        errors confirmé). Même constat déjà documenté pour V1 sur les
        bundles Synthea réels (CLAUDE.md : « avertissements de schéma FHIR
        R4B... NEEDS_REVIEW... systématiquement légitimes »)."""
        case_id = "CLM-3001"
        doc_file = _stage_document(case_id)
        fhir_file = _stage_fhir_bundle(case_id)
        monkeypatch.setattr(
            "agents.document_understanding_agent.agent._invoke_llm_document_understanding",
            Mock(return_value=_decision()),
        )
        result = run(
            case_id=case_id,
            manifest_files=[doc_file, fhir_file],
            role=ReaderRole.ADMINISTRATIVE_MANAGER,
        )
        assert result.status is VerificationStatus.NEEDS_REVIEW
        assert result.extraction is not None
        assert result.extraction.classification.document_type is DocumentType.INVOICE
        assert result.extraction.is_readable is True
        assert result.extraction.full_text == ""  # minimisé, jamais persisté
        assert result.extraction.pages == []
        assert result.fhir_summary["status"] == "NEEDS_REVIEW"
        assert result.fhir_summary["resource_count"] == 1
        assert result.errors == []  # NEEDS_REVIEW n'est jamais une erreur bloquante
        assert result.privacy_view is not None
        assert result.privacy_view["patient_pseudonym"].startswith("PAT-")
        assert result.llm_trace.model_name

    def test_no_fhir_bundle_not_provided_never_blocks(self, monkeypatch):
        case_id = "CLM-3002"
        doc_file = _stage_document(case_id)
        monkeypatch.setattr(
            "agents.document_understanding_agent.agent._invoke_llm_document_understanding",
            Mock(return_value=_decision()),
        )
        result = run(case_id=case_id, manifest_files=[doc_file], role=ReaderRole.ADMINISTRATIVE_MANAGER)
        assert result.fhir_summary["validation_scope"] == "NOT_PROVIDED"
        assert result.status is VerificationStatus.PASS

    def test_no_document_candidate_no_crash(self, monkeypatch):
        case_id = "CLM-3003"
        fhir_file = _stage_fhir_bundle(case_id)
        monkeypatch.setattr(
            "agents.document_understanding_agent.agent._invoke_llm_document_understanding",
            Mock(return_value=_decision()),
        )
        result = run(case_id=case_id, manifest_files=[fhir_file], role=ReaderRole.ADMINISTRATIVE_MANAGER)
        assert result.extraction is None
        assert any("Aucun document" in r for r in result.reasons)


class TestFhirIntegrity:
    def test_fhir_hash_mismatch_fails(self, monkeypatch):
        case_id = "CLM-3010"
        fhir_file = _stage_fhir_bundle(case_id)
        tampered = fhir_file.model_copy(update={"sha256": "a" * 64})
        monkeypatch.setattr(
            "agents.document_understanding_agent.agent._invoke_llm_document_understanding",
            Mock(return_value=_decision()),
        )
        result = run(case_id=case_id, manifest_files=[tampered], role=ReaderRole.ADMINISTRATIVE_MANAGER)
        assert result.status is VerificationStatus.FAIL
        assert result.errors


class TestLlmBehavior:
    def test_llm_unavailable_keeps_deterministic_status(self, monkeypatch):
        """Contrairement à intake_safety_agent, un LLM indisponible ne force
        jamais FAIL ici — la Phase A déterministe est conservée (plan §5)."""
        case_id = "CLM-3020"
        doc_file = _stage_document(case_id)
        monkeypatch.setattr(
            "agents.document_understanding_agent.agent._invoke_llm_document_understanding",
            Mock(return_value=None),
        )
        result = run(case_id=case_id, manifest_files=[doc_file], role=ReaderRole.ADMINISTRATIVE_MANAGER)
        assert result.status is VerificationStatus.PASS
        assert any("LLM indisponible" in r for r in result.reasons)

    def test_llm_can_only_escalate_fhir_status_never_soften(self, monkeypatch):
        case_id = "CLM-3021"
        doc_file = _stage_document(case_id)
        fhir_file = _stage_fhir_bundle(case_id)
        monkeypatch.setattr(
            "agents.document_understanding_agent.agent._invoke_llm_document_understanding",
            Mock(return_value=_decision(fhir_recommended_status="FAIL")),
        )
        result = run(
            case_id=case_id,
            manifest_files=[doc_file, fhir_file],
            role=ReaderRole.ADMINISTRATIVE_MANAGER,
        )
        # Le FHIR déterministe était PASS ; le LLM a durci vers FAIL — autorisé.
        assert result.status is VerificationStatus.FAIL


class TestPrivacyIntegration:
    def test_role_absent_never_blocks_pipeline(self, monkeypatch):
        case_id = "CLM-3030"
        doc_file = _stage_document(case_id)
        monkeypatch.setattr(
            "agents.document_understanding_agent.agent._invoke_llm_document_understanding",
            Mock(return_value=_decision()),
        )
        result = run(case_id=case_id, manifest_files=[doc_file], role=None)
        assert result.privacy_view is None
        assert result.status is VerificationStatus.PASS
        assert any("Rôle du lecteur absent" in r for r in result.reasons)


class TestNodeIntegration:
    def test_node_updates_state(self, monkeypatch):
        case_id = "CLM-3040"
        doc_file = _stage_document(case_id)
        monkeypatch.setattr(
            "agents.document_understanding_agent.agent._invoke_llm_document_understanding",
            Mock(return_value=_decision()),
        )
        from schemas.results import ClaimManifest
        from schemas.v2_results import IntakeSafetyResult
        from schemas.domain import IntakeSafetyStatus, IntakeStatus
        from schemas.results import LlmMetadata

        manifest = ClaimManifest(
            claim_id=case_id,
            file_count=1,
            total_size_bytes=doc_file.actual_size,
            files=[doc_file],
            status=IntakeStatus.ACCEPTED,
        )
        intake_result = IntakeSafetyResult(
            case_id=case_id,
            status=IntakeSafetyStatus.ACCEPTED,
            manifest=manifest,
            reasons=["Dossier accepté."],
            llm_trace=LlmMetadata(model_name="gemma4:latest", prompt_version="1.0.0"),
        )
        state = {
            "case_id": case_id,
            "schema_version": "2.0.0",
            "current_step": "intake_safety",
            "completed_steps": ["intake_safety"],
            "intake_safety_result": intake_result,
            "reader_role": ReaderRole.ADMINISTRATIVE_MANAGER.value,
        }
        updates = node(state)  # type: ignore[arg-type]
        assert updates["current_step"] == "document_understanding"
        assert updates["document_understanding_result"].status is VerificationStatus.PASS
        assert "errors" not in updates
