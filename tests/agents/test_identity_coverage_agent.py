from schemas.domain import VerificationStatus
from schemas.results import IdentityCoverageResult
from agents.identity_coverage_agent.agent import node, run


def test_identity_coverage_run_passes_with_complete_fields():
    result = run(
        case_id="CLM-0001",
        extracted_fields={
            "patient_id": "PAT-001",
            "patient_name": "Jane Doe",
            "payer_name": "Cigna Health",
            "amount_requested": "100.00",
            "coverage_rate": "0.80",
        },
    )

    assert isinstance(result, IdentityCoverageResult)
    assert result.identity.status == VerificationStatus.PASS
    assert result.coverage.status == VerificationStatus.PASS
    assert result.coverage.patient_share is not None


def test_identity_coverage_run_needs_review_without_rate():
    result = run(
        case_id="CLM-0001",
        extracted_fields={
            "patient_id": "PAT-001",
            "patient_name": "Jane Doe",
            "payer_name": "Cigna Health",
            "amount_requested": "100.00",
        },
    )

    assert result.identity.status == VerificationStatus.PASS
    assert result.coverage.status == VerificationStatus.NEEDS_REVIEW


def test_identity_coverage_node_consumes_input_and_adds_audit():
    updates = node(
        {
            "case_id": "CLM-0001",
            "identity_coverage_input": {"case_id": "CLM-0001"},
            "ocr_result": None,
        }
    )

    assert updates["identity_coverage_input"] is None
    assert updates["current_step"] == "identity_coverage"
    assert updates["completed_steps"] == ["identity_coverage"]
    assert updates["audit_trail"]
    assert isinstance(updates["identity_coverage_result"], IdentityCoverageResult)


def test_identity_coverage_compares_dossier_and_contract_identity_without_mutation():
    contract = {
        "patient_id": "PAT-001",
        "payer_name": "Cigna Health",
        "coverage_rate": "0.80",
        "policy_active": True,
        "coverage_start_date": "2026-01-01",
        "coverage_end_date": "2026-12-31",
        "ceiling_remaining": "500.00",
        "preauthorization_required": False,
    }
    before = dict(contract)

    result = run(
        case_id="CLM-0001",
        dossier_patient_id="PAT-001",
        contract=contract,
        extracted_fields={
            "patient_id": "PAT-001",
            "patient_name": "Jane Doe",
            "payer_name": "Cigna Health",
            "amount_requested": "100.00",
            "service_date": "2026-06-15",
        },
    )

    assert contract == before
    assert result.identity.status == VerificationStatus.PASS
    assert result.identity.claim_patient_id == "PAT-001"
    assert result.identity.contract_patient_id == "PAT-001"
    assert result.coverage.status == VerificationStatus.PASS
    assert result.coverage.coverage_start_date.isoformat() == "2026-01-01"


def test_identity_coverage_needs_review_on_dossier_contract_identity_mismatch():
    result = run(
        case_id="CLM-0001",
        dossier_patient_id="PAT-001",
        contract={"patient_id": "PAT-999", "payer_name": "Cigna Health", "coverage_rate": "0.80"},
        extracted_fields={
            "patient_id": "PAT-001",
            "patient_name": "Jane Doe",
            "amount_requested": "100.00",
        },
    )

    assert result.identity.status == VerificationStatus.NEEDS_REVIEW
    assert any("contract" in reason for reason in result.identity.reasons)


def test_identity_coverage_fails_when_service_date_outside_contract_dates():
    result = run(
        case_id="CLM-0001",
        dossier_patient_id="PAT-001",
        contract={
            "patient_id": "PAT-001",
            "payer_name": "Cigna Health",
            "coverage_rate": "0.80",
            "policy_active": True,
            "coverage_start_date": "2026-01-01",
            "coverage_end_date": "2026-03-31",
        },
        extracted_fields={
            "patient_id": "PAT-001",
            "patient_name": "Jane Doe",
            "amount_requested": "100.00",
            "service_date": "2026-06-15",
        },
    )

    assert result.coverage.status == VerificationStatus.FAIL
    assert any("fin de couverture" in reason for reason in result.coverage.reasons)


def test_identity_coverage_fails_when_ceiling_is_exceeded():
    result = run(
        case_id="CLM-0001",
        dossier_patient_id="PAT-001",
        contract={
            "patient_id": "PAT-001",
            "payer_name": "Cigna Health",
            "coverage_rate": "0.80",
            "ceiling_remaining": "50.00",
        },
        extracted_fields={
            "patient_id": "PAT-001",
            "patient_name": "Jane Doe",
            "amount_requested": "100.00",
        },
    )

    assert result.coverage.status == VerificationStatus.FAIL
    assert result.coverage.ceiling_exceeded is True


def test_identity_coverage_needs_review_when_preauthorization_missing():
    result = run(
        case_id="CLM-0001",
        dossier_patient_id="PAT-001",
        contract={
            "patient_id": "PAT-001",
            "payer_name": "Cigna Health",
            "coverage_rate": "0.80",
            "preauthorization_required": True,
            "preauthorization_status": "missing",
        },
        extracted_fields={
            "patient_id": "PAT-001",
            "patient_name": "Jane Doe",
            "amount_requested": "100.00",
        },
    )

    assert result.coverage.status == VerificationStatus.NEEDS_REVIEW
    assert result.coverage.preauthorization_required is True


def test_identity_coverage_structured_pipeline_builds_evidence_and_rule_versions():
    result = run(
        case_id="CLM-0001",
        policy_number="POL-0001",
        patient_pseudonym="PAT-001",
        service_date="2026-06-15",
        requested_amount="100.00",
        total_amount="150.00",
        procedure_codes=["11429006"],
        extraction_confidence=0.91,
        extracted_fields={
            "patient_id": "PAT-001",
            "patient_name": "Jane Doe",
            "payer_name": "Cigna Health",
            "currency": "USD",
        },
    )

    rule_ids = {e["rule_id"] for e in result.evidence}

    assert result.identity.status == VerificationStatus.PASS
    assert result.coverage.status == VerificationStatus.PASS
    assert "identity:1.0.0" in result.rule_version
    assert "coverage:1.0.0" in result.rule_version
    assert "authorization:1.0.0" in result.rule_version
    assert "CONTRACT_EXISTS" in rule_ids
    assert "REQUESTED_AMOUNT_POSITIVE" in rule_ids
    assert "PREAUTHORIZATION_PRESENT_WHEN_REQUIRED" in rule_ids
    assert result.structured_errors == []
    assert not hasattr(result, "final_recommendation")


def test_identity_coverage_structured_pipeline_flags_low_confidence():
    result = run(
        case_id="CLM-0001",
        policy_number="POL-0001",
        patient_pseudonym="PAT-001",
        service_date="2026-06-15",
        requested_amount="100.00",
        total_amount="150.00",
        procedure_codes=["11429006"],
        extraction_confidence=0.20,
        extracted_fields={
            "patient_id": "PAT-001",
            "patient_name": "Jane Doe",
            "payer_name": "Cigna Health",
            "currency": "USD",
        },
    )

    assert any(err["code"] == "LOW_EXTRACTION_CONFIDENCE" for err in result.structured_errors)
    assert any("Confiance" in warning for warning in result.warnings)


def test_identity_coverage_node_produces_minimized_audit_for_structured_input():
    updates = node(
        {
            "case_id": "CLM-0001",
            "identity_coverage_input": {
                "case_id": "CLM-0001",
                "patient_pseudonym": "PAT-001",
                "policy_number": "POL-0001",
                "service_date": "2026-06-15",
                "requested_amount": "100.00",
                "total_amount": "150.00",
                "procedure_codes": ["11429006"],
                "extraction_confidence": 0.91,
            },
            "ocr_result": None,
        }
    )

    audit = updates["audit_trail"][0]

    assert updates["identity_coverage_input"] is None
    assert audit.actor == "identity_coverage_agent"
    assert set(audit.details) == {"identity_status", "coverage_status", "rule_version"}
    assert "contracts" not in str(audit.details)
