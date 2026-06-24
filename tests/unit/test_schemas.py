"""Tests unitaires des schémas Pydantic — Étape 2."""

import json
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

import pytest

from schemas.domain import (
    ClaimSubmission,
    CoverageInfo,
    DeterministicRules,
    DocumentInfo,
    ExtractedData,
    PatientInfo,
)
from schemas.results import (
    FraudDetectionResult,
    IdentityCoverageResult,
    IdentityResult,
    CoverageResult,
    SecurityGateResult,
)
from schemas.domain import Recommendation, VerificationStatus, SecurityDecision

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "datasets" / "fixtures" / "valid"


def load_ground_truth(case_id: str) -> dict:
    path = FIXTURES_DIR / case_id / "oracle" / "ground_truth.json"
    return json.loads(path.read_text(encoding="utf-8"))


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
        case_id=gt["case_id"],
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
    from datetime import datetime

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
        ),
        res.SecurityGateResult(case_id="CLM-0001", decision=dom.SecurityDecision.ALLOW),
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
        res.FhirValidatorResult(case_id="CLM-0001", status=dom.VerificationStatus.PASS, bundle_expected=True),
        res.DocumentOcrResult(case_id="CLM-0001", status=dom.VerificationStatus.PASS),
        res.MedicalCodingResult(case_id="CLM-0001", status=dom.VerificationStatus.PASS),
        res.ClinicalConsistencyResult.model_validate({"case_id": "CLM-0001", "status": "PASS"}),
        res.FraudDetectionResult(case_id="CLM-0001", status=dom.VerificationStatus.PASS),
        res.CaseReviewerResult(case_id="CLM-0001", recommendation=dom.Recommendation.APPROVE),
        res.AuditEvent(case_id="CLM-0001", event_id="evt-1", actor="intake", action="ingest", outcome="ok"),
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
        duplicate_invoice=ef["duplicate_invoice"],
        reasons=ef["reasons"],
    )
    assert result.duplicate_invoice is False
    assert result.status == VerificationStatus.PASS
