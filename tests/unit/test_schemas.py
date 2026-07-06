"""Tests unitaires des schémas Pydantic — Étape 2."""

import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from types import UnionType
from typing import Union, get_args, get_origin, get_type_hints

from pydantic import ValidationError

import pytest

from schemas.domain import (
    ClaimSubmission,
    CoverageInfo,
    DeterministicRules,
    DocumentInfo,
    ExtractedData,
    MedicalProcedure,
    PatientInfo,
    Prescription,
    ProviderInfo,
    EncounterInfo,
)
from schemas.results import (
    AuditResult,
    FraudDetectionResult,
    FraudResultPayload,
    IdentityCoverageResult,
    IdentityResult,
    CoverageResult,
    LlmMetadata,
    SecurityGateResult,
)
from schemas.domain import Recommendation, VerificationStatus, SecurityDecision
from state.claim_state import ClaimState, validate_state_update

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "datasets" / "fixtures" / "valid"


BUSINESS_SCHEMAS = (
    PatientInfo,
    CoverageInfo,
    DocumentInfo,
    ExtractedData,
    ProviderInfo,
    EncounterInfo,
    MedicalProcedure,
    Prescription,
    DeterministicRules,
    ClaimSubmission,
)


def load_ground_truth(case_id: str) -> dict:
    path = FIXTURES_DIR / case_id / "oracle" / "ground_truth.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _is_optional(annotation: object) -> bool:
    return get_origin(annotation) in {Union, UnionType} and type(None) in get_args(annotation)


# ── Dossier valide accepté ────────────────────────────────────────────────────


def test_valid_claim_accepted():
    """Un dossier valide (CLM-0001) est accepté par les schémas."""
    gt = load_ground_truth("CLM-0001")
    ext = gt["expected_extraction"]

    claim = ClaimSubmission(
        case_id=gt["case_id"],
        patient=PatientInfo(
            patient_id=ext["patient_id"],
            patient_name=ext["patient_name"],
        ),
        coverage=CoverageInfo(
            payer_name=ext["payer_name"],
            coverage_rate=gt["deterministic_rules"]["coverage_rate"],
        ),
        extracted=ExtractedData(
            patient_name=ext["patient_name"],
            patient_id=ext["patient_id"],
            payer_name=ext["payer_name"],
            claim_reference=ext["claim_reference"],
            invoice_number=ext["invoice_number"],
            prescription_number=ext["prescription_number"],
            procedure_count=ext["procedure_count"],
            medication_count=ext["medication_count"],
            total_billed=ext["total_billed"],
            amount_requested=ext["amount_requested"],
            patient_share=ext["patient_share"],
            currency=ext["currency"],
        ),
        rules=DeterministicRules(
            coverage_rate=gt["deterministic_rules"]["coverage_rate"],
            authorization_required=gt["deterministic_rules"]["authorization_required"],
            authorization_status=gt["deterministic_rules"]["authorization_status"],
            duplicate_invoice=gt["deterministic_rules"]["duplicate_invoice"],
            prompt_injection_detected=gt["deterministic_rules"]["prompt_injection_detected"],
        ),
    )

    assert claim.case_id == "CLM-0001"
    assert claim.extracted is not None
    assert claim.extracted.total_billed == Decimal("3666.69")
    assert claim.extracted.amount_requested == Decimal("2933.35")
    assert claim.coverage is not None
    assert claim.coverage.coverage_rate == Decimal("0.80")
    assert claim.rules is not None
    assert claim.rules.duplicate_invoice is False
    assert claim.rules.prompt_injection_detected is False


def test_ten_business_schemas_are_strict_and_typed():
    """Les dix schémas métier prévus existent et interdisent les champs inconnus."""

    assert [schema.__name__ for schema in BUSINESS_SCHEMAS] == [
        "PatientInfo",
        "CoverageInfo",
        "DocumentInfo",
        "ExtractedData",
        "ProviderInfo",
        "EncounterInfo",
        "MedicalProcedure",
        "Prescription",
        "DeterministicRules",
        "ClaimSubmission",
    ]
    for schema in BUSINESS_SCHEMAS:
        assert schema.model_config["extra"] == "forbid"

    assert PatientInfo.model_fields["patient_id"].is_required()
    assert PatientInfo.model_fields["patient_name"].is_required()
    assert not _is_optional(PatientInfo.model_fields["patient_id"].annotation)
    assert CoverageInfo.model_fields["coverage_rate"].annotation is Decimal
    assert ExtractedData.model_fields["service_date"].annotation == date | None
    assert MedicalProcedure.model_fields["unit_cost"].is_required()
    assert Prescription.model_fields["unit_cost"].is_required()
    assert ClaimSubmission.model_fields["case_id"].is_required()


def test_mutable_collections_use_default_factory():
    """Les listes et dictionnaires des modèles ne partagent pas de valeur mutable."""

    collection_fields = [
        ClaimSubmission.model_fields["documents"],
        ClaimSubmission.model_fields["procedures"],
        ClaimSubmission.model_fields["prescriptions"],
        ExtractedData.model_fields["provenance"],
        EncounterInfo.model_fields["diagnosis_codes"],
        SecurityGateResult.model_fields["findings"],
        IdentityCoverageResult.model_fields["evidence"],
        FraudResultPayload.model_fields["signals"],
    ]
    for field in collection_fields:
        assert field.default_factory is not None

    first = ClaimSubmission(case_id="CLM-0001")
    second = ClaimSubmission(case_id="CLM-0002")
    first.documents.append(
        DocumentInfo(
            filename="facture.pdf",
            sha256="a" * 64,
            size_bytes=1,
            mime_type="application/pdf",
        )
    )
    assert second.documents == []


@pytest.mark.parametrize(
    ("factory", "expected_loc"),
    [
        (lambda: ClaimSubmission(case_id="DOSSIER-001"), ("case_id",)),
        (lambda: PatientInfo(patient_id="pat-1", patient_name=""), ("patient_name",)),
        (lambda: CoverageInfo(payer_name="Assureur", coverage_rate=Decimal("1.50")), ("coverage_rate",)),
        (
            lambda: DocumentInfo(
                filename="facture.pdf",
                sha256="abc",
                size_bytes=1024,
                mime_type="application/pdf",
            ),
            ("sha256",),
        ),
        (lambda: MedicalProcedure(code="ABC", description="Acte", unit_cost=Decimal("-1")), ("unit_cost",)),
        (lambda: ExtractedData.model_validate({"total_billed": "-1.00"}), ("total_billed",)),
    ],
)
def test_invalid_business_objects_are_rejected_with_localized_errors(factory, expected_loc):
    """Au moins cinq objets invalides sont rejetés avec une localisation de champ."""

    with pytest.raises(ValidationError) as exc_info:
        factory()

    locations = [tuple(error["loc"]) for error in exc_info.value.errors()]
    assert expected_loc in locations


def test_claim_state_rejects_raw_documents_ocr_and_secrets():
    """ClaimState reste minimal : pas de brut documentaire, OCR complet ou secret."""

    forbidden_updates = [
        {"payload": b"%PDF-1.7"},
        {"ocr_result": {"full_text": "texte OCR complet"}},
        {"ocr_result": {"raw_ocr_text": "texte OCR complet"}},
        {"document": {"base64_image": "iVBORw0KGgoAAAANSUhEUgAA"}},
        {"document": {"image_content": "image encodee"}},
        {"llm": {"prompt": "Tu es un agent ClaimShield complet..."}},
        {"llm": {"raw_response": "{\"status\":\"PASS\"}"}},
        {"llm": {"messages": [{"role": "system", "content": "prompt complet"}]}},
        {"metadata": {"api_key": "abc123"}},
        {"metadata": {"note": "Bearer abc.def.ghi"}},
        {"source_path": "/tmp/facture.pdf"},
    ]

    for update in forbidden_updates:
        with pytest.raises(ValueError):
            validate_state_update(update)

    validate_state_update(
        {
            "case_id": "CLM-0001",
            "alerts": ["hash verifie"],
            "metadata": {"artifact_path": "artifacts/ocr/CLM-0001.json"},
        }
    )


def test_claim_state_expose_un_resultat_type_par_agent():
    """Les 11 agents ont chacun un champ résultat dédié, nullable tant qu'indisponible."""

    expected_result_fields = {
        "intake_result",
        "security_result",
        "privacy_result",
        "identity_coverage_result",
        "fhir_result",
        "ocr_result",
        "coding_result",
        "clinical_result",
        "fraud_result",
        "review_result",
        "audit_result",
    }

    annotations = get_type_hints(ClaimState, include_extras=True)
    assert expected_result_fields <= set(annotations)
    for field in expected_result_fields:
        assert _is_optional(annotations[field]), f"{field} doit accepter None"

    assert annotations["audit_result"] == AuditResult | None


def test_claim_state_regroupe_les_champs_generaux_en_tete():
    """Les champs généraux nécessaires au routage restent regroupés en tête."""

    assert list(ClaimState.__annotations__)[:5] == [
        "case_id",
        "schema_version",
        "intake_status",
        "current_step",
        "completed_steps",
    ]


def test_resultats_agents_acceptent_uniquement_metadata_llm_minimales():
    """Chaque résultat d'agent peut porter modèle, version de prompt et confiance."""

    import schemas.results as res

    metadata = LlmMetadata(
        model_name="gemma4:latest",
        prompt_version="1.0.0",
        confidence=0.82,
    )
    result_classes_with_trace_field = [
        (res.ClaimIntakeResult, "llm_metadata"),
        (res.SecurityGateResult, "llm_metadata"),
        (res.PrivacyResult, "llm_metadata"),
        (res.IdentityCoverageResult, "llm_metadata"),
        (res.FhirValidatorResult, "llm_metadata"),
        (res.DocumentOcrResult, "llm_metadata"),
        (res.MedicalCodingResult, "llm_metadata"),
        (res.ClinicalConsistencyResult, "llm_trace"),
        (res.FraudDetectionResult, "llm_trace"),
        (res.CaseReviewerResult, "llm_trace"),
        (res.AuditResult, "llm_metadata"),
    ]
    for result_class, trace_field_name in result_classes_with_trace_field:
        required_trace_classes = {
            res.ClaimIntakeResult,
            res.CaseReviewerResult,
            res.FhirValidatorResult,
            res.MedicalCodingResult,
            res.ClinicalConsistencyResult,
            res.FraudDetectionResult,
        }
        expected_annotation = (
            LlmMetadata if result_class in required_trace_classes else LlmMetadata | None
        )
        assert result_class.model_fields[trace_field_name].annotation == expected_annotation

    gate = res.SecurityGateResult(
        claim_id="CLM-0001",
        decision=SecurityDecision.ALLOW,
        reasons=["Autorise"],
        llm_metadata=metadata,
    )
    assert gate.llm_metadata == metadata

    with pytest.raises(ValidationError):
        LlmMetadata(model_name="api_key=sk-secret", prompt_version="1.0.0")


# ── Champ inconnu rejeté ──────────────────────────────────────────────────────


def test_unknown_field_rejected():
    """Un champ inconnu lève une ValidationError (extra=forbid)."""

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExtractedData.model_validate({"patient_name": "Test", "champ_inconnu": "non"})


# ── Montants négatifs refusés ─────────────────────────────────────────────────


def test_negative_amount_rejected():
    """Un montant négatif lève une ValidationError."""

    with pytest.raises(ValidationError):
        ExtractedData.model_validate({"total_billed": "-1.00"})


# ── case_id hors format CLM-XXXX ─────────────────────────────────────────────


def test_invalid_case_id_rejected():
    """Un case_id ne respectant pas CLM-\\d{{4,}} lève une ValidationError."""

    with pytest.raises(ValidationError):
        ClaimSubmission(case_id="DOSSIER-001")


# ── Taux de couverture hors [0, 1] ────────────────────────────────────────────


def test_coverage_rate_above_one_rejected():
    """Un taux de couverture > 1 lève une ValidationError."""

    with pytest.raises(ValidationError):
        CoverageInfo(payer_name="Test", coverage_rate=Decimal("1.50"))


# ── Nom de patient vide ───────────────────────────────────────────────────────


def test_empty_patient_name_rejected():
    """Un patient_name vide (min_length=1) lève une ValidationError."""

    with pytest.raises(ValidationError):
        PatientInfo(patient_id="uuid-1234", patient_name="")


# ── Hash SHA-256 de longueur incorrecte ──────────────────────────────────────


def test_invalid_sha256_length_rejected():
    """Un hash SHA-256 dont la longueur n'est pas 64 lève une ValidationError."""

    with pytest.raises(ValidationError):
        DocumentInfo(
            filename="facture.pdf",
            sha256="abc123",
            size_bytes=1024,
            mime_type="application/pdf",
        )


# ── procedure_count négatif ───────────────────────────────────────────────────


def test_negative_procedure_count_rejected():
    """Un procedure_count négatif (ge=0) lève une ValidationError."""

    with pytest.raises(ValidationError):
        ExtractedData(
            procedure_count=-1,
            medication_count=None,
            total_billed=None,
            amount_requested=None,
            patient_share=None,
        )


# ── Recommandation correcte ───────────────────────────────────────────────────


def test_recommendation_from_oracle():
    """La recommandation attendue de l'oracle correspond à l'enum."""
    gt = load_ground_truth("CLM-0001")
    rec = Recommendation(gt["expected_recommendation"])
    assert rec == Recommendation.APPROVE


# ── Résultats agents cohérents avec l'oracle ──────────────────────────────────


def test_security_result_from_oracle():
    gt = load_ground_truth("CLM-0001")
    sec = gt["expected_security"]
    result = SecurityGateResult(
        claim_id=gt["case_id"],
        decision=SecurityDecision.ALLOW,
        prompt_injection_detected=sec["prompt_injection_detected"],
        reasons=sec["reasons"],
    )
    assert result.decision == SecurityDecision.ALLOW
    assert result.prompt_injection_detected is False


def test_identity_coverage_result_from_oracle():
    gt = load_ground_truth("CLM-0001")
    eid = gt["expected_identity"]
    ecov = gt["expected_coverage"]

    result = IdentityCoverageResult(
        case_id=gt["case_id"],
        identity=IdentityResult(
            status=VerificationStatus(eid["status"]),
            patient_id=eid["patient_id"],
            patient_name=eid["patient_name"],
            reasons=eid["reasons"],
        ),
        coverage=CoverageResult(
            status=VerificationStatus(ecov["status"]),
            payer_name=ecov["payer_name"],
            coverage_rate=ecov["coverage_rate"],
            amount_requested=ecov["amount_requested"],
            patient_share=ecov["patient_share"],
            reasons=ecov["reasons"],
        ),
    )
    assert result.identity.status == VerificationStatus.PASS
    assert result.coverage.coverage_rate == Decimal("0.80")


# ── Sérialisation JSON de tous les modèles ────────────────────────────────────


def test_all_models_json_serializable():
    """Chaque modèle Pydantic est sérialisable en JSON via model_dump_json()."""
    import schemas.domain as dom
    import schemas.results as res

    # Instances minimales valides pour chaque modèle
    instances = [
        dom.PatientInfo(patient_id="uuid-1", patient_name="Test Patient"),
        dom.CoverageInfo(payer_name="Assureur Test", coverage_rate=Decimal("0.80")),
        dom.DocumentInfo(
            filename="doc.pdf",
            sha256="a" * 64,
            size_bytes=1024,
            mime_type="application/pdf",
        ),
        dom.ExtractedData(
            patient_name="Test",
            procedure_count=None,
            medication_count=None,
            total_billed=None,
            amount_requested=None,
            patient_share=None,
        ),
        dom.ProviderInfo(provider_id="prov-1"),
        dom.EncounterInfo(
            encounter_id="enc-1",
            encounter_class="ambulatory",
            start=datetime(2026, 6, 3, 8, 0),
            patient_id="pat-1",
        ),
        dom.MedicalProcedure(code="185349003", description="Consultation", unit_cost=Decimal("70.00")),
        dom.Prescription(medication_code="M001", medication_name="Paracétamol", unit_cost=Decimal("5.00")),
        dom.DeterministicRules(
            coverage_rate=Decimal("0.80"),
            authorization_required=False,
            authorization_status=dom.AuthorizationStatus.NOT_REQUIRED,
            duplicate_invoice=False,
            prompt_injection_detected=False,
        ),
        dom.ClaimSubmission(case_id="CLM-0001"),
        res.ClaimIntakeResult(
            claim_id="CLM-0001",
            status=dom.IntakeStatus.ACCEPTED,
            manifest=res.ClaimManifest(
                claim_id="CLM-0001",
                file_count=0,
                total_size_bytes=0,
                status=dom.IntakeStatus.ACCEPTED,
            ),
            accepted_count=0,
            quarantined_count=0,
            llm_metadata=res.LlmMetadata(model_name="test-llm", prompt_version="test"),
        ),
        res.SecurityGateResult(
            claim_id="CLM-0001",
            decision=dom.SecurityDecision.ALLOW,
            reasons=["Aucune menace détectée — dossier autorisé"],
        ),
        res.PrivacyResult(
            case_id="CLM-0001",
            status=dom.VerificationStatus.PASS,
            data_classification=dom.DataClassification.SYNTHETIC_TEST_DATA,
            contains_real_personal_data=False,
        ),
        res.IdentityCoverageResult(
            case_id="CLM-0001",
            identity=res.IdentityResult(status=dom.VerificationStatus.PASS),
            coverage=res.CoverageResult(status=dom.VerificationStatus.PASS),
        ),
        res.FhirValidatorResult(
            case_id="CLM-0001",
            status=dom.VerificationStatus.PASS,
            bundle_expected=True,
            llm_metadata=res.LlmMetadata(model_name="test-llm", prompt_version="test"),
        ),
        res.DocumentOcrResult(
            claim_id="CLM-0001",
            file_path="incoming/CLM-0001/facture.pdf",
            sha256="a" * 64,
            mime_type="application/pdf",
            extraction_status=dom.ExtractionStatus.SUCCESS,
            status=dom.VerificationStatus.PASS,
            document_type=dom.DocumentType.INVOICE,
            ocr_source=dom.OcrSource.PDF_TEXT,
        ),
        res.MedicalCodingResult(
            case_id="CLM-0001",
            status=dom.VerificationStatus.PASS,
            llm_metadata=res.LlmMetadata(model_name="test-llm", prompt_version="test"),
        ),
        res.ClinicalConsistencyResult.model_validate(
            {
                "case_id": "CLM-0001",
                "status": "PASS",
                "llm_trace": {"model_name": "test-llm", "prompt_version": "test"},
            }
        ),
        res.FraudDetectionResult(
            case_id="CLM-0001",
            status=dom.VerificationStatus.PASS,
            llm_trace=res.LlmMetadata(model_name="test-llm", prompt_version="test"),
        ),
        res.CaseReviewerResult(
            case_id="CLM-0001",
            llm_trace=res.LlmMetadata(model_name="test-llm", prompt_version="test"),
            result_payload=res.CaseReviewerResultPayload(
                recommendation=dom.Recommendation.APPROVE,
                human_review_reasons=["Validation humaine obligatoire avant toute décision finale."],
            ),
        ),
        res.AuditEvent(case_id="CLM-0001", event_id="evt-1", actor="intake", action="ingest", outcome="ok"),
        res.AuditResult(case_id="CLM-0001", status=dom.VerificationStatus.PASS),
    ]

    for obj in instances:
        json_str = obj.model_dump_json()
        assert isinstance(json_str, str) and len(json_str) > 2, (
            f"{type(obj).__name__} n'est pas sérialisable en JSON"
        )


def test_fraud_result_from_oracle():
    gt = load_ground_truth("CLM-0001")
    ef = gt["expected_fraud"]
    result = FraudDetectionResult(
        case_id=gt["case_id"],
        status=VerificationStatus(ef["status"]),
        llm_trace=LlmMetadata(model_name="test-llm", prompt_version="test"),
        result_payload=FraudResultPayload(
            duplicate_invoice=ef["duplicate_invoice"],
            reasons=ef["reasons"],
        ),
    )
    assert result.result_payload.duplicate_invoice is False
    assert result.status == VerificationStatus.PASS
