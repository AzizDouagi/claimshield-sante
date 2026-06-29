from decimal import Decimal

from agents.identity_coverage_agent.schemas import CoverageCheckStatus
from schemas.domain import VerificationStatus
from tools.coverage_rules import calculate_patient_share, evaluate_coverage_contract, verify_coverage
from tools.rule_loader import get_rule_version, load_rules


def test_coverage_rules_load_versioned_yaml():
    rules = load_rules("coverage_rules.yaml")

    assert get_rule_version("coverage_rules.yaml") == "1.0.0"
    assert rules["ruleset"]["effective_from"] == "2026-01-01"
    assert rules["rules"][0]["id"] == "CONTRACT_EXISTS"
    assert rules["currency"] == "USD"
    assert "payer_name" in rules["required_fields"]


def test_verify_coverage_passes_and_calculates_patient_share():
    result = verify_coverage(
        payer_name="Cigna Health",
        coverage_rate="0.80",
        amount_requested="100.00",
    )

    assert result["status"] == VerificationStatus.PASS
    assert result["patient_share"] == Decimal("20.00")


def test_verify_coverage_fails_without_payer():
    result = verify_coverage(
        payer_name=None,
        coverage_rate="0.80",
        amount_requested="100.00",
    )

    assert result["status"] == VerificationStatus.FAIL


def test_calculate_patient_share_rounds_half_up():
    assert calculate_patient_share(Decimal("10.005"), Decimal("0.00")) == Decimal("10.01")


def test_verify_coverage_checks_dates_ceiling_and_preauthorization():
    result = verify_coverage(
        payer_name="Cigna Health",
        coverage_rate="0.80",
        amount_requested="3100.00",
        service_date="2026-06-15",
        coverage_start_date="2026-01-01",
        coverage_end_date="2026-12-31",
        ceiling_remaining="5000.00",
        preauthorization_status="approved",
    )

    assert result["status"] == VerificationStatus.PASS
    assert result["preauthorization_required"] is True
    assert result["ceiling_exceeded"] is False


def _contract(**overrides):
    data = {
        "policy_number": "POL-001",
        "patient_id": "PAT-001",
        "start_date": "2026-01-01",
        "end_date": "2026-12-31",
        "status": "active",
        "currency": "USD",
        "annual_limit": "5000.00",
        "covered_procedure_codes": ["PROC-CONSULT", "PROC-IRM"],
        "excluded_procedure_codes": ["PROC-COSMETIC"],
        "preauthorization_required_for": ["PROC-IRM"],
    }
    data.update(overrides)
    return data


_DEFAULT_CONTRACT = object()


def _evaluate(contract=_DEFAULT_CONTRACT, **overrides):
    params = {
        "contract": _contract() if contract is _DEFAULT_CONTRACT else contract,
        "service_date": "2026-06-15",
        "requested_amount": "100.00",
        "total_amount": "150.00",
        "currency": "USD",
        "procedure_codes": ["PROC-CONSULT"],
        "preauthorization_reference": None,
    }
    params.update(overrides)
    return evaluate_coverage_contract(**params)


def test_evaluate_coverage_contract_active_with_proof_for_each_rule():
    check = _evaluate()

    rule_ids = {e.rule_id for e in check.evidence}

    assert check.status == CoverageCheckStatus.ACTIVE
    assert check.structured_errors == []
    assert len(check.evidence) == 11
    assert "REQUESTED_AMOUNT_POSITIVE" in rule_ids
    assert all(e.rule_version == "1.0.0" for e in check.evidence)


def test_evaluate_coverage_contract_not_found():
    check = _evaluate(contract=None)

    assert check.status == CoverageCheckStatus.NOT_FOUND
    assert check.structured_errors[0].code == "CONTRACT_EXISTS"


def test_evaluate_coverage_contract_inactive():
    check = _evaluate(contract=_contract(status="inactive"))

    assert check.status == CoverageCheckStatus.INACTIVE
    assert any(err.code == "CONTRACT_ACTIVE" for err in check.structured_errors)


def test_evaluate_coverage_contract_not_started_and_expired():
    not_started = _evaluate(service_date="2025-12-31")
    expired = _evaluate(service_date="2027-01-01")

    assert not_started.status == CoverageCheckStatus.NOT_STARTED
    assert expired.status == CoverageCheckStatus.EXPIRED


def test_evaluate_coverage_contract_currency_mismatch():
    check = _evaluate(currency="EUR")

    assert check.status == CoverageCheckStatus.AMBIGUOUS
    assert any(err.code == "CURRENCY_MATCHES_CONTRACT" for err in check.structured_errors)


def test_evaluate_coverage_contract_requested_amount_positive_and_not_above_total():
    non_positive = _evaluate(requested_amount="0.00")
    above_total = _evaluate(requested_amount="200.00", total_amount="150.00")

    assert any(err.code == "REQUESTED_AMOUNT_POSITIVE" for err in non_positive.structured_errors)
    assert any(err.code == "REQUESTED_NOT_GREATER_THAN_TOTAL" for err in above_total.structured_errors)


def test_evaluate_coverage_contract_requested_amount_not_above_limit():
    check = _evaluate(requested_amount="6000.00", total_amount="7000.00")

    assert check.status == CoverageCheckStatus.AMBIGUOUS
    assert any(
        err.code == "REQUESTED_NOT_GREATER_THAN_AVAILABLE_LIMIT"
        for err in check.structured_errors
    )


def test_evaluate_coverage_contract_procedure_must_be_covered_and_not_excluded():
    not_covered = _evaluate(procedure_codes=["PROC-UNKNOWN"])
    excluded = _evaluate(procedure_codes=["PROC-COSMETIC"])

    assert any(err.code == "PROCEDURE_CODE_COVERED" for err in not_covered.structured_errors)
    assert any(err.code == "PROCEDURE_CODE_NOT_EXCLUDED" for err in excluded.structured_errors)


def test_evaluate_coverage_contract_requires_preauthorization_when_mandatory():
    missing = _evaluate(procedure_codes=["PROC-IRM"])
    present = _evaluate(procedure_codes=["PROC-IRM"], preauthorization_reference="AUTH-001")

    assert missing.status == CoverageCheckStatus.AMBIGUOUS
    assert any(
        err.code == "PREAUTHORIZATION_PRESENT_WHEN_REQUIRED"
        for err in missing.structured_errors
    )
    assert present.status == CoverageCheckStatus.ACTIVE
