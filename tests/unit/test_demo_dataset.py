"""Tests du jeu de données de démonstration — datasets/demo/.

Valide :
- présence des 6 dossiers et de leurs fichiers obligatoires
- conformité des ground_truth.json aux schémas Pydantic
- exactitude des hashes SHA-256 dans manifest.json
- données exclusivement synthétiques (pas de données réelles)
- unicité et cohérence des scénarios métier
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from schemas.domain import (
    ClaimSubmission,
    DataClassification,
    ExtractedData,
    Recommendation,
    SecurityDecision,
    VerificationStatus,
)
from schemas.results import (
    CaseReviewerResult,
    ClinicalConsistencyResult,
    ClinicalResultPayload,
    CoverageResult,
    FhirValidatorResult,
    FraudDetectionResult,
    FraudResultPayload,
    IdentityCoverageResult,
    IdentityResult,
    LlmMetadata,
    PrivacyResult,
    SecurityGateResult,
)

DEMO_DIR = Path(__file__).resolve().parents[2] / "datasets" / "demo"

EXPECTED_CASES = ["CLM-0004", "CLM-0005", "CLM-0015", "CLM-0019", "CLM-0024", "CLM-0032"]
EXPECTED_SCENARIOS = {"SC-01", "SC-02", "SC-03", "SC-04", "SC-05", "SC-06"}
EXPECTED_AGENTS = {
    "CLM-0004": "full_pipeline",
    "CLM-0015": "identity_coverage_agent",
    "CLM-0005": "security_gate_agent",
    "CLM-0019": "claim_intake_agent",
    "CLM-0024": "fraud_detection_agent",
    "CLM-0032": "clinical_consistency_agent",
}
EXPECTED_RECOMMENDATIONS = {
    "CLM-0004": "APPROVE",
    "CLM-0015": "REJECT",
    "CLM-0005": "REJECT",
    "CLM-0019": "REJECT",
    "CLM-0024": "REJECT",
    "CLM-0032": "PENDING",
}


def load_gt(case_id: str) -> dict:
    return json.loads((DEMO_DIR / case_id / "oracle" / "ground_truth.json").read_text())


def load_cd(case_id: str) -> dict:
    return json.loads((DEMO_DIR / case_id / "oracle" / "case_data.json").read_text())


def load_manifest(case_id: str) -> dict:
    return json.loads((DEMO_DIR / case_id / "audit" / "manifest.json").read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(case_dir: Path, fname: str) -> Path | None:
    for d in [case_dir / "input", case_dir / "oracle", case_dir / "audit"]:
        p = d / fname
        if p.exists():
            return p
    return None


def file_signature(path: Path) -> tuple[int, str]:
    return path.stat().st_size, sha256(path)


# ── Structure ─────────────────────────────────────────────────────────────────


def test_six_dossiers_presents():
    """datasets/demo/ contient exactement 6 dossiers CLM-*."""
    cases = sorted(d.name for d in DEMO_DIR.iterdir() if d.is_dir() and d.name.startswith("CLM-"))
    assert cases == EXPECTED_CASES


@pytest.mark.parametrize("case_id", EXPECTED_CASES)
def test_ground_truth_existe(case_id: str):
    assert (DEMO_DIR / case_id / "oracle" / "ground_truth.json").exists()


@pytest.mark.parametrize("case_id", EXPECTED_CASES)
def test_manifest_existe(case_id: str):
    assert (DEMO_DIR / case_id / "audit" / "manifest.json").exists()


@pytest.mark.parametrize("case_id", EXPECTED_CASES)
def test_documents_requis_presents(case_id: str):
    """Chaque fichier de required_documents est présent sauf ceux dans expected_missing_documents."""
    gt = load_gt(case_id)
    input_dir = DEMO_DIR / case_id / "input"
    missing_ok = set(gt.get("expected_missing_documents", []))
    for doc in gt.get("required_documents", []):
        if doc in missing_ok:
            assert not (input_dir / doc).exists(), f"{doc} devrait être absent (SC-04)"
        else:
            assert (input_dir / doc).exists(), f"{doc} manquant dans {case_id}/input/"


def test_documentation_demo_presente():
    """La provenance, la licence et la méthode de génération sont documentées."""
    readme = DEMO_DIR / "README.md"
    provenance = DEMO_DIR / "PROVENANCE.md"
    assert readme.exists()
    assert provenance.exists()

    text = (readme.read_text(encoding="utf-8") + "\n" + provenance.read_text(encoding="utf-8")).lower()
    for expected in [
        "synthea",
        "apache license 2.0",
        "scripts/generate_demo_data.py",
        "synthetic_test_data",
        "sha-256",
    ]:
        assert expected in text


def test_tous_les_json_sont_valides():
    """Tous les fichiers JSON de demo/ sont lisibles comme JSON."""
    json_files = sorted(DEMO_DIR.rglob("*.json"))
    assert json_files
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8"))


# ── Scénarios métier ──────────────────────────────────────────────────────────


def test_scenarios_ids_uniques():
    """Les 6 scenario_id sont distincts et couvrent SC-01 à SC-06."""
    ids = {load_gt(c)["scenario_id"] for c in EXPECTED_CASES}
    assert ids == EXPECTED_SCENARIOS


@pytest.mark.parametrize("case_id,expected_agent", EXPECTED_AGENTS.items())
def test_agent_concerne_present_dans_verite_terrain(case_id: str, expected_agent: str):
    gt = load_gt(case_id)
    assert gt["agent_under_test"] == expected_agent


@pytest.mark.parametrize("case_id,expected_rec", EXPECTED_RECOMMENDATIONS.items())
def test_recommandation_attendue(case_id: str, expected_rec: str):
    gt = load_gt(case_id)
    assert gt["expected_recommendation"] == expected_rec


def test_sc03_securite_bloquee():
    """SC-03 : Security Gate FAIL et prompt_injection_detected=True."""
    gt = load_gt("CLM-0005")
    assert gt["expected_security"]["status"] == "FAIL"
    assert gt["expected_security"]["prompt_injection_detected"] is True


def test_sc04_facture_absente():
    """SC-04 : la facture est physiquement absente de input/."""
    assert not (DEMO_DIR / "CLM-0019" / "input" / "facture_CLM-0019.pdf").exists()
    gt = load_gt("CLM-0019")
    assert "facture_CLM-0019.pdf" in gt["expected_missing_documents"]


def test_sc05_doublon_facture():
    """SC-05 : FraudDetection indique duplicate_invoice=True."""
    gt = load_gt("CLM-0024")
    assert gt["expected_fraud"]["duplicate_invoice"] is True
    assert gt["expected_fraud"]["status"] == "FAIL"


def test_sc06_incohérence_clinique():
    """SC-06 : clinical_consistency NEEDS_REVIEW, human_review_required=True."""
    gt = load_gt("CLM-0032")
    assert gt["expected_clinical_consistency"]["status"] == "NEEDS_REVIEW"
    assert gt["human_review_required"] is True
    assert gt["expected_recommendation"] == "PENDING"


# ── Données synthétiques ──────────────────────────────────────────────────────


@pytest.mark.parametrize("case_id", EXPECTED_CASES)
def test_donnees_synthetiques(case_id: str):
    gt = load_gt(case_id)
    cd = load_cd(case_id)
    assert gt["data_classification"] == "SYNTHETIC_TEST_DATA"
    assert gt["contains_real_personal_data"] is False
    assert cd["data_classification"] == "SYNTHETIC_TEST_DATA"
    assert cd["contains_real_personal_data"] is False
    assert cd["provenance"]["license"] == "Apache-2.0"
    assert cd["source"]["data_type"] == "fully_synthetic"
    assert cd["source"]["contains_real_patient_data"] is False
    assert cd["patient"]["SSN"].startswith("999-")
    assert any(char.isdigit() for char in cd["patient"]["FIRST"])
    assert any(char.isdigit() for char in cd["patient"]["LAST"])


# ── Hashes SHA-256 ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("case_id", EXPECTED_CASES)
def test_hashes_corrects(case_id: str):
    """Chaque hash SHA-256 du manifest correspond au fichier réel."""
    case_dir = DEMO_DIR / case_id
    manifest = load_manifest(case_id)
    for entry in manifest["files"]:
        p = resolve(case_dir, entry["filename"])
        assert p is not None, f"{case_id}: fichier introuvable : {entry['filename']}"
        size_bytes, digest = file_signature(p)
        assert size_bytes == entry["size_bytes"], f"{case_id}: taille incorrecte : {entry['filename']}"
        assert digest == entry["sha256"], f"{case_id}: hash incorrect : {entry['filename']}"


def test_generation_dry_run_reproductible_et_non_destructif():
    """Le générateur peut être rejoué en dry-run sans modifier les signatures."""
    before = {
        path.relative_to(DEMO_DIR).as_posix(): file_signature(path)
        for path in sorted(DEMO_DIR.rglob("*"))
        if path.is_file()
    }

    result = subprocess.run(
        [sys.executable, "scripts/generate_demo_data.py", "--dry-run"],
        cwd=DEMO_DIR.parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    after = {
        path.relative_to(DEMO_DIR).as_posix(): file_signature(path)
        for path in sorted(DEMO_DIR.rglob("*"))
        if path.is_file()
    }
    assert result.returncode == 0
    assert "(dry-run terminé" in result.stdout
    assert before == after


# ── Conformité Pydantic ───────────────────────────────────────────────────────


@pytest.mark.parametrize("case_id", EXPECTED_CASES)
def test_claim_submission_valide(case_id: str):
    gt = load_gt(case_id)
    ext = gt["expected_extraction"]
    ClaimSubmission(
        case_id=gt["case_id"],
        data_classification=gt["data_classification"],
        extracted=ExtractedData(
            patient_name=ext.get("patient_name"),
            patient_id=ext.get("patient_id"),
            payer_name=ext.get("payer_name"),
            claim_reference=ext.get("claim_reference"),
            procedure_count=ext.get("procedure_count"),
            medication_count=ext.get("medication_count"),
            total_billed=ext.get("total_billed"),
            amount_requested=ext.get("amount_requested"),
            patient_share=ext.get("patient_share"),
        ),
    )


@pytest.mark.parametrize("case_id", EXPECTED_CASES)
def test_security_gate_result_valide(case_id: str):
    gt = load_gt(case_id)
    es = gt["expected_security"]
    decision = "BLOCK" if es.get("status") == "FAIL" else "ALLOW"
    reasons = es.get("reasons") or ["Aucune menace détectée — dossier autorisé"]
    SecurityGateResult(
        claim_id=case_id,
        decision=SecurityDecision(decision),
        prompt_injection_detected=es.get("prompt_injection_detected"),
        reasons=reasons,
    )


@pytest.mark.parametrize("case_id", EXPECTED_CASES)
def test_privacy_result_valide(case_id: str):
    gt = load_gt(case_id)
    ep = gt["expected_privacy"]
    PrivacyResult(
        case_id=case_id,
        status=VerificationStatus(ep["status"]),
        data_classification=DataClassification(ep["data_classification"]),
        contains_real_personal_data=ep["contains_real_personal_data"],
        reasons=ep.get("reasons", []),
    )


@pytest.mark.parametrize("case_id", EXPECTED_CASES)
def test_identity_coverage_result_valide(case_id: str):
    gt = load_gt(case_id)
    ei = gt["expected_identity"]
    ec = gt["expected_coverage"]
    IdentityCoverageResult(
        case_id=case_id,
        identity=IdentityResult(
            status=VerificationStatus(ei["status"]),
            patient_id=ei.get("patient_id"),
            patient_name=ei.get("patient_name"),
            source_patient_id=ei.get("source_patient_id"),
            claim_patient_id=ei.get("claim_patient_id"),
            encounter_patient_id=ei.get("encounter_patient_id"),
            reasons=ei.get("reasons", []),
        ),
        coverage=CoverageResult(
            status=VerificationStatus(ec["status"]),
            payer_name=ec.get("payer_name"),
            source_payer_name=ec.get("source_payer_name"),
            coverage_rate=ec.get("coverage_rate"),
            amount_requested=ec.get("amount_requested"),
            patient_share=ec.get("patient_share"),
            policy_active=ec.get("policy_active"),
            reasons=ec.get("reasons", []),
        ),
    )


@pytest.mark.parametrize("case_id", EXPECTED_CASES)
def test_fhir_validator_result_valide(case_id: str):
    gt = load_gt(case_id)
    ef = gt["expected_fhir"]
    FhirValidatorResult(
        case_id=case_id,
        status=VerificationStatus(ef["status"]),
        bundle_expected=ef["bundle_expected"],
        reasons=ef.get("reasons", []),
        llm_metadata=LlmMetadata(model_name="test-llm", prompt_version="test"),
    )


@pytest.mark.parametrize("case_id", EXPECTED_CASES)
def test_fraud_detection_result_valide(case_id: str):
    gt = load_gt(case_id)
    ef = gt["expected_fraud"]
    FraudDetectionResult(
        case_id=case_id,
        status=VerificationStatus(ef["status"]),
        llm_trace=LlmMetadata(model_name="test-llm", prompt_version="test"),
        result_payload=FraudResultPayload(
            duplicate_invoice=ef.get("duplicate_invoice"),
            reasons=ef.get("reasons", []),
        ),
    )


@pytest.mark.parametrize("case_id", EXPECTED_CASES)
def test_clinical_consistency_result_valide(case_id: str):
    gt = load_gt(case_id)
    ecc = gt["expected_clinical_consistency"]
    ClinicalConsistencyResult(
        case_id=case_id,
        status=VerificationStatus(ecc["status"]),
        llm_trace=LlmMetadata(model_name="test-llm", prompt_version="test"),
        result_payload=ClinicalResultPayload(
            procedure_count=ecc.get("procedure_count"),
            medication_count=ecc.get("medication_count"),
            prescription_required=ecc.get("prescription_required"),
            reasons=ecc.get("reasons", []),
        ),
    )


@pytest.mark.parametrize("case_id", EXPECTED_CASES)
def test_case_reviewer_result_valide(case_id: str):
    gt = load_gt(case_id)
    CaseReviewerResult(
        case_id=case_id,
        recommendation=Recommendation(gt["expected_recommendation"]),
        justification=gt.get("recommendation_reasons", []),
        human_review_required=gt.get("human_review_required", False),
        human_review_reasons=gt.get("human_review_reasons", []),
        llm_metadata=LlmMetadata(model_name="test-llm", prompt_version="test"),
    )
