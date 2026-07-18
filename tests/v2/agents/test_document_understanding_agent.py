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

_CLAIM_REQUEST_LINES = [
    "Demande de remboursement",
    "Dossier : CLM-9999",
    "Assureur : Cigna Health",
    "Contrat POL-TEST1234",
    "Taux de couverture synthetique : 80 %",
]

_INVOICE_LINES_NO_MEDICATION = [
    "FACTURE MEDICALE",
    "Numero : INV-CLM-9999",
    "patient_id : PAT-9999-DEMO",
    "Prestataire : Bernard Leclerc",
    "Date du document : 2024-03-01",
    "Date de soins : 25/02/2024",
    "Devise : USD",
    "Montant total facture : 3666.69 USD",
]

_PRESCRIPTION_LINES = [
    "ORDONNANCE",
    "Dossier : CLM-9999",
    "Amoxicilline 500 mg",
    "Ibuprofene 400 mg",
]

_PRESCRIPTION_LINES_SYNTHEA_FORMAT = [
    "ORDONNANCE SYNTHETIQUE",
    "Dossier : CLM-9999",
    "Medicaments prescrits",
    "sodium fluoride 0.0272 MG/MG Oral Gel",
]
"""Format réel observé sur les fixtures Synthea (`pypdf` sur
datasets/fixtures/valid/CLM-0001/input/ordonnance_CLM-0001.pdf) — dose
décimale, unité en majuscules — que `tools.document_parser._MEDICATION_RE`
(V1) ne capture jamais (pas de `re.IGNORECASE`, entiers uniquement)."""

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


def _stage_claim_request_document(case_id: str) -> InspectedFile:
    incoming = get_settings().storage_dir / "incoming" / case_id
    incoming.mkdir(parents=True, exist_ok=True)
    dest = incoming / "demande_remboursement_test.pdf"
    _make_pdf(dest, _CLAIM_REQUEST_LINES)
    sha = compute_sha256(dest)
    return InspectedFile(
        original_name="demande_remboursement_test.pdf",
        storage_name="demande_remboursement_test.pdf",
        normalized_extension="pdf",
        detected_mime_type="application/pdf",
        actual_size=dest.stat().st_size,
        sha256=sha,
        status=FileStatus.ACCEPTED,
        reasons=[],
        relative_storage_path=f"incoming/{case_id}/demande_remboursement_test.pdf",
    )


def _stage_prescription_document(
    case_id: str, lines: list[str] = _PRESCRIPTION_LINES
) -> InspectedFile:
    incoming = get_settings().storage_dir / "incoming" / case_id
    incoming.mkdir(parents=True, exist_ok=True)
    dest = incoming / "ordonnance_test.pdf"
    _make_pdf(dest, lines)
    sha = compute_sha256(dest)
    return InspectedFile(
        original_name="ordonnance_test.pdf",
        storage_name="ordonnance_test.pdf",
        normalized_extension="pdf",
        detected_mime_type="application/pdf",
        actual_size=dest.stat().st_size,
        sha256=sha,
        status=FileStatus.ACCEPTED,
        reasons=[],
        relative_storage_path=f"incoming/{case_id}/ordonnance_test.pdf",
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


class TestSupplementaryFieldHarvest:
    """Correctif post-mesure V2-10 (AZIZ), Phase 4 : `payer_name`/
    `coverage_rate`/`contract_number` sont récupérés depuis un document
    secondaire (demande de remboursement) quand le document principal
    (facture) ne les contient pas — jamais un deuxième document pleinement
    retraité, jamais un champ déjà présent écrasé."""

    def test_payer_and_coverage_harvested_from_secondary_claim_request(self, monkeypatch):
        case_id = "CLM-3050"
        # La facture (document principal, sélectionné en priorité) ne contient
        # ni assureur ni taux de couverture — reproduit exactement le constat
        # de la mesure V2-10 sur les fixtures réelles.
        invoice_file = _stage_document(case_id)
        claim_request_file = _stage_claim_request_document(case_id)
        monkeypatch.setattr(
            "agents.document_understanding_agent.agent._invoke_llm_document_understanding",
            Mock(return_value=_decision()),
        )
        result = run(
            case_id=case_id,
            manifest_files=[invoice_file, claim_request_file],
            role=ReaderRole.ADMINISTRATIVE_MANAGER,
        )
        assert result.extraction is not None
        # Le document principal reste bien la facture (priorité "facture").
        assert result.extraction.classification.document_type is DocumentType.INVOICE
        fields = result.extraction.fields
        # `payer_name` réutilise le même motif que `tools.document_parser._PAYER_RE`
        # (V1) : détecte la mention d'un assureur (mot-clé), ne capture pas le
        # nom réel — comportement hérité, pas une régression introduite ici.
        assert "payer_name" in fields
        assert fields["payer_name"].value
        assert "coverage_rate" in fields
        assert fields["coverage_rate"].value == "0.80"
        assert "contract_number" in fields
        assert "POL-TEST1234" in fields["contract_number"].value
        assert any("document secondaire" in r for r in result.reasons)

    def test_no_secondary_document_never_crashes(self, monkeypatch):
        case_id = "CLM-3051"
        invoice_file = _stage_document(case_id)
        monkeypatch.setattr(
            "agents.document_understanding_agent.agent._invoke_llm_document_understanding",
            Mock(return_value=_decision()),
        )
        result = run(case_id=case_id, manifest_files=[invoice_file], role=ReaderRole.ADMINISTRATIVE_MANAGER)
        assert result.extraction is not None
        assert "payer_name" not in result.extraction.fields

    def test_existing_primary_field_never_overwritten_by_secondary(self, monkeypatch):
        case_id = "CLM-3052"
        # Facture contenant déjà une mention d'assureur ("Blue Cross", motif
        # distinct de celui du document secondaire "Assureur : Cigna Health")
        # — la valeur du document principal ne doit jamais être écrasée.
        invoice_with_payer = _INVOICE_LINES + ["Blue Cross"]
        invoice_file = _stage_document(case_id, lines=invoice_with_payer)
        claim_request_file = _stage_claim_request_document(case_id)
        monkeypatch.setattr(
            "agents.document_understanding_agent.agent._invoke_llm_document_understanding",
            Mock(return_value=_decision()),
        )
        result = run(
            case_id=case_id,
            manifest_files=[invoice_file, claim_request_file],
            role=ReaderRole.ADMINISTRATIVE_MANAGER,
        )
        assert result.extraction is not None
        assert "blue" in result.extraction.fields["payer_name"].value.lower()
        assert not any("document secondaire" in r and "payer_name" in r for r in result.reasons)


class TestMedicalItemsHarvest:
    """Correctif post-mesure V2-10 (AZIZ), Phase 5 : les actes/médicaments
    sont récupérés depuis un document secondaire (ordonnance) quand le
    document principal (facture) n'en contient aucun — sans appel OCR/
    classification supplémentaire (réutilise le calcul déjà fait pour la
    Phase 4). Toujours jamais une répartition heuristique inventée : les
    éléments du document secondaire sont pris tels quels, uniquement si le
    document principal n'en a lui-même trouvé aucun."""

    def test_medical_items_harvested_from_secondary_prescription(self, monkeypatch):
        case_id = "CLM-3060"
        invoice_file = _stage_document(case_id, lines=_INVOICE_LINES_NO_MEDICATION)
        prescription_file = _stage_prescription_document(case_id)
        monkeypatch.setattr(
            "agents.document_understanding_agent.agent._invoke_llm_document_understanding",
            Mock(return_value=_decision()),
        )
        result = run(
            case_id=case_id,
            manifest_files=[invoice_file, prescription_file],
            role=ReaderRole.ADMINISTRATIVE_MANAGER,
        )
        assert result.extraction is not None
        # Le document principal reste la facture (priorité "facture").
        assert result.extraction.classification.document_type is DocumentType.INVOICE
        medical_items = result.extraction.essential_fields.medical_items
        descriptions = {item.description for item in medical_items}
        assert descriptions == {"Amoxicilline 500 mg", "Ibuprofene 400 mg"}
        assert any("document secondaire" in r and "médicament" in r for r in result.reasons)

    def test_synthea_decimal_uppercase_format_harvested_via_tolerant_fallback(self, monkeypatch):
        """Régression V2-10 (Phase 5) : le format réel Synthea (dose
        décimale, unité en majuscules, ex. `"sodium fluoride 0.0272 MG"`)
        n'est jamais capturé par `tools.document_parser._MEDICATION_RE`
        (V1) — vérifié qu'il l'est bien via le repli tolérant local."""
        case_id = "CLM-3063"
        invoice_file = _stage_document(case_id, lines=_INVOICE_LINES_NO_MEDICATION)
        prescription_file = _stage_prescription_document(
            case_id, lines=_PRESCRIPTION_LINES_SYNTHEA_FORMAT
        )
        monkeypatch.setattr(
            "agents.document_understanding_agent.agent._invoke_llm_document_understanding",
            Mock(return_value=_decision()),
        )
        result = run(
            case_id=case_id,
            manifest_files=[invoice_file, prescription_file],
            role=ReaderRole.ADMINISTRATIVE_MANAGER,
        )
        assert result.extraction is not None
        medical_items = result.extraction.essential_fields.medical_items
        assert len(medical_items) >= 1
        assert any("sodium fluoride" in item.description.lower() for item in medical_items)
        assert any("détection tolérante" in r for r in result.reasons)

    def test_no_secondary_document_leaves_medical_items_empty(self, monkeypatch):
        case_id = "CLM-3061"
        invoice_file = _stage_document(case_id, lines=_INVOICE_LINES_NO_MEDICATION)
        monkeypatch.setattr(
            "agents.document_understanding_agent.agent._invoke_llm_document_understanding",
            Mock(return_value=_decision()),
        )
        result = run(case_id=case_id, manifest_files=[invoice_file], role=ReaderRole.ADMINISTRATIVE_MANAGER)
        assert result.extraction is not None
        assert result.extraction.essential_fields.medical_items == []

    def test_primary_medical_items_never_overwritten_by_secondary(self, monkeypatch):
        case_id = "CLM-3062"
        # La facture (document principal) contient déjà ses propres
        # médicaments — l'ordonnance secondaire ne doit jamais les remplacer.
        invoice_file = _stage_document(case_id)  # _INVOICE_LINES par défaut, avec médicaments
        prescription_file = _stage_prescription_document(case_id)
        monkeypatch.setattr(
            "agents.document_understanding_agent.agent._invoke_llm_document_understanding",
            Mock(return_value=_decision()),
        )
        result = run(
            case_id=case_id,
            manifest_files=[invoice_file, prescription_file],
            role=ReaderRole.ADMINISTRATIVE_MANAGER,
        )
        assert result.extraction is not None
        assert not any(
            "document secondaire" in r and "médicament" in r for r in result.reasons
        )


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


class TestActiveVersionFiltering:
    """Plan de remédiation « rejouabilité des dossiers » (phase 1) — aucun
    agent ne doit jamais traiter une version de document désactivée
    (`is_active=False`, remplacée par une révision plus récente ou une
    substitution suspecte encore en attente de revue humaine). Le filtre vit
    au point unique où le manifeste est parcouru (`_select_document_candidate`/
    `_select_secondary_candidates`/`_select_fhir_bundle_candidate`), jamais
    dupliqué agent par agent."""

    def test_select_document_candidate_ignores_inactive_version(self):
        from agents.document_understanding_agent.agent import _select_document_candidate

        case_id = "CLM-3050"
        active = _stage_document(case_id).model_copy(
            update={"document_id": "doc-active", "document_family_id": "fam-1", "is_active": True}
        )
        inactive = _stage_document(case_id).model_copy(
            update={"document_id": "doc-old", "document_family_id": "fam-1", "is_active": False}
        )
        # L'ancienne version (inactive) apparaît en premier dans le manifeste —
        # si le filtre `is_active` était omis, elle serait sélectionnée à tort.
        candidate = _select_document_candidate([inactive, active])
        assert candidate is not None
        assert candidate.document_id == "doc-active"

    def test_select_document_candidate_returns_none_if_only_inactive_versions(self):
        from agents.document_understanding_agent.agent import _select_document_candidate

        case_id = "CLM-3051"
        inactive = _stage_document(case_id).model_copy(update={"is_active": False})
        assert _select_document_candidate([inactive]) is None

    def test_select_fhir_bundle_candidate_ignores_inactive_version(self):
        from agents.document_understanding_agent.agent import _select_fhir_bundle_candidate

        case_id = "CLM-3052"
        active_bundle = _stage_fhir_bundle(case_id).model_copy(
            update={"document_id": "bundle-active", "is_active": True}
        )
        inactive_bundle = _stage_fhir_bundle(case_id).model_copy(
            update={"document_id": "bundle-old", "is_active": False}
        )
        candidate = _select_fhir_bundle_candidate([inactive_bundle, active_bundle])
        assert candidate is not None
        assert candidate.document_id == "bundle-active"
