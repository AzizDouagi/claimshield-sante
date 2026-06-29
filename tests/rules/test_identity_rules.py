from agents.identity_coverage_agent.schemas import IdentityCheckStatus
from tools.identity_matching import (
    compare_patient_ids,
    identity_match_result,
    match_identity_contract,
)
from tools.rule_loader import get_rule_version, load_rules
from schemas.domain import VerificationStatus


def test_identity_rules_load_versioned_yaml():
    rules = load_rules("identity_rules.yaml")

    assert get_rule_version("identity_rules.yaml") == "1.0.0"
    assert rules["ruleset"]["name"] == "identity-rules"
    assert rules["rules"][0]["id"] == "IDENTITY_PATIENT_ID_MATCH"
    assert rules["match_mode"] == "exact"
    assert "patient_id" in rules["required_fields"]


def test_compare_patient_ids_detects_match_after_normalization():
    matched, reasons = compare_patient_ids(
        ocr_patient_id=" PAT-001 ",
        fhir_patient_id="pat-001",
        claim_patient_id="PAT-001",
    )

    assert matched is True
    assert reasons


def test_identity_match_needs_review_on_mismatch():
    result = identity_match_result(
        case_id="CLM-0001",
        ocr_patient_id="PAT-001",
        ocr_patient_name="Jane Doe",
        fhir_patient_id="PAT-002",
        claim_patient_id=None,
    )

    assert result["status"] == VerificationStatus.NEEDS_REVIEW
    assert any("Incohérence" in reason for reason in result["reasons"])


def _contract(policy_number="POL-001", patient_pseudonym="PAT-001"):
    return {
        "policy_number": policy_number,
        "patient_pseudonym": patient_pseudonym,
    }


def test_match_identity_contract_patient_correct_and_contract_correct():
    check = match_identity_contract(
        dossier_patient_pseudonym="PAT-001",
        policy_number="POL-001",
        candidate_contracts=[_contract()],
    )

    assert check.status == IdentityCheckStatus.MATCH
    assert check.rule_applied == "IDENTITY_PATIENT_ID_MATCH"
    assert check.compared_fields == ["dossier_patient_pseudonym", "policy_number"]


def test_match_identity_contract_patient_correct_wrong_contract():
    check = match_identity_contract(
        dossier_patient_pseudonym="PAT-001",
        policy_number="POL-999",
        candidate_contracts=[_contract("POL-001", "PAT-001")],
    )

    assert check.status == IdentityCheckStatus.NOT_FOUND
    assert check.rule_applied == "IDENTITY_POLICY_NUMBER_LOOKUP"


def test_match_identity_contract_wrong_patient():
    check = match_identity_contract(
        dossier_patient_pseudonym="PAT-002",
        policy_number="POL-001",
        candidate_contracts=[_contract("POL-001", "PAT-001")],
    )

    assert check.status == IdentityCheckStatus.MISMATCH
    assert check.structured_errors[0].code == "IDENTITY_CONTRACT_MISMATCH"


def test_match_identity_contract_missing_identifier():
    check = match_identity_contract(
        dossier_patient_pseudonym=None,
        policy_number="POL-001",
        candidate_contracts=[_contract()],
    )

    assert check.status == IdentityCheckStatus.NOT_FOUND
    assert check.structured_errors[0].code == "IDENTITY_REQUIRED_FIELD_MISSING"


def test_match_identity_contract_multiple_candidates_is_ambiguous():
    check = match_identity_contract(
        dossier_patient_pseudonym="PAT-001",
        policy_number="POL-001",
        candidate_contracts=[_contract("POL-001", "PAT-001"), _contract("POL-001", "PAT-001")],
    )

    assert check.status == IdentityCheckStatus.AMBIGUOUS
    assert check.structured_errors[0].code == "IDENTITY_MULTIPLE_CONTRACT_CANDIDATES"


def test_match_identity_contract_malformed_pseudonym():
    check = match_identity_contract(
        dossier_patient_pseudonym="Jane Doe",
        policy_number="POL-001",
        candidate_contracts=[_contract()],
    )

    assert check.status == IdentityCheckStatus.NOT_FOUND
    assert check.structured_errors[0].code == "IDENTITY_PSEUDONYM_MALFORMED"


def test_match_identity_contract_does_not_expose_free_name_in_messages():
    check = match_identity_contract(
        dossier_patient_pseudonym="Jane Doe",
        policy_number="POL-001",
        candidate_contracts=[_contract()],
    )

    serialized = check.model_dump_json()

    assert "Jane Doe" in serialized
    assert all("Jane Doe" not in err.message for err in check.structured_errors)
    assert all("Jane Doe" not in warning for warning in check.warnings)
