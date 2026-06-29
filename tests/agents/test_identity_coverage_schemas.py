from decimal import Decimal

import pytest
from pydantic import ValidationError

from agents.identity_coverage_agent.schemas import (
    AuthorizationCheck,
    AuthorizationCheckStatus,
    CoverageCheck,
    CoverageCheckStatus,
    IdentityCheck,
    IdentityCheckStatus,
    IdentityCoverageInput,
    RuleEvidence,
    StructuredRuleError,
)
from schemas.domain import VerificationStatus
from schemas.results import CoverageResult, IdentityCoverageResult, IdentityResult


def test_identity_coverage_input_accepts_minimal_claim_payload_and_serializes_json():
    payload = IdentityCoverageInput(
        claim_id="CLM-0001",
        patient_pseudonym="PAT-001",
        policy_number="POL-001",
        service_date="2026-06-15",
        requested_amount="100.00",
        total_amount="125.00",
        procedure_codes=["11429006"],
        preauthorization_reference="AUTH-001",
        extraction_confidence=0.91,
        provenance={"requested_amount": "invoice.pdf:page_1"},
        rule_version="identity:1.0.0;coverage:1.0.0;authorization:1.0.0",
    )

    data = payload.model_dump(mode="json")

    assert payload.case_id == "CLM-0001"
    assert payload.requested_amount == Decimal("100.00")
    assert data["service_date"] == "2026-06-15"
    assert payload.model_dump_json()


def test_identity_coverage_input_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        IdentityCoverageInput(claim_id="CLM-0001", unknown_field=True)


def test_identity_check_statuses_and_evidence_are_json_serializable():
    check = IdentityCheck(
        status=IdentityCheckStatus.MATCH,
        patient_pseudonym="PAT-001",
        contract_patient_pseudonym="PAT-001",
        dossier_patient_pseudonym="PAT-001",
        evidence=[RuleEvidence(source="contract", field="patient_id", value="PAT-001")],
        warnings=["source OCR non utilisée"],
        structured_errors=[
            StructuredRuleError(code="IDENTITY_SOURCE_AMBIGUOUS", message="Source ambiguë")
        ],
        rule_version="1.0.0",
    )

    dumped = check.model_dump(mode="json")

    assert dumped["status"] == "MATCH"
    assert dumped["evidence"][0]["source"] == "contract"
    assert check.model_dump_json()


def test_coverage_and_authorization_check_statuses_are_supported():
    coverage = CoverageCheck(
        status=CoverageCheckStatus.ACTIVE,
        policy_number="POL-001",
        service_date="2026-06-15",
        coverage_start_date="2026-01-01",
        coverage_end_date="2026-12-31",
        requested_amount="100.00",
        total_amount="125.00",
        ceiling_remaining="500.00",
        evidence=[RuleEvidence(source="contract", field="coverage_end_date", value="2026-12-31")],
    )
    authorization = AuthorizationCheck(
        status=AuthorizationCheckStatus.PRESENT,
        preauthorization_reference="AUTH-001",
        required=True,
        procedure_codes=["11429006"],
        evidence=[RuleEvidence(source="claim", field="preauthorization_reference", value="AUTH-001")],
    )

    assert coverage.model_dump(mode="json")["status"] == "ACTIVE"
    assert authorization.model_dump(mode="json")["status"] == "PRESENT"


def test_identity_coverage_result_carries_rule_version_evidence_errors_and_warnings():
    result = IdentityCoverageResult(
        case_id="CLM-0001",
        identity=IdentityResult(status=VerificationStatus.PASS),
        coverage=CoverageResult(status=VerificationStatus.PASS),
        rule_version="identity:1.0.0;coverage:1.0.0;authorization:1.0.0",
        evidence=[{"source": "contract", "field": "policy_number", "value": "POL-001"}],
        warnings=["préautorisation non requise"],
        structured_errors=[{"code": "NONE", "message": "aucune erreur"}],
    )

    dumped = result.model_dump(mode="json")

    assert dumped["rule_version"].startswith("identity:")
    assert dumped["evidence"][0]["field"] == "policy_number"
    assert result.model_dump_json()
