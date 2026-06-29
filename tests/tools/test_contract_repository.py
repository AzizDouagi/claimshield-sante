import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from agents.identity_coverage_agent.agent import run
from schemas.domain import VerificationStatus
from tools.contract_repository import (
    ContractRepositoryError,
    SyntheticContract,
    get_contract,
    get_contract_snapshot,
    load_contracts,
)


def _write_contracts(path, contracts):
    path.write_text(json.dumps(contracts), encoding="utf-8")


def _contract(policy_number="POL-T001", **overrides):
    data = {
        "policy_number": policy_number,
        "patient_pseudonym": "PAT-T001",
        "start_date": "2026-01-01",
        "end_date": "2026-12-31",
        "status": "active",
        "currency": "USD",
        "annual_limit": "5000.00",
        "covered_procedure_codes": ["11429006"],
        "excluded_procedure_codes": ["PROC-COSMETIC"],
        "preauthorization_required_for": ["PROC-IRM"],
        "contract_version": "1.0.0",
    }
    data.update(overrides)
    return data


def test_reference_contracts_load_and_use_decimal_limits():
    contracts = load_contracts()

    assert contracts
    assert all(isinstance(contract, SyntheticContract) for contract in contracts)
    assert all(isinstance(contract.annual_limit, Decimal) for contract in contracts)
    assert get_contract("POL-0015") is not None


def test_contract_snapshot_does_not_expose_full_repository():
    snapshot = get_contract_snapshot("POL-0015")

    assert snapshot is not None
    assert snapshot["policy_number"] == "POL-0015"
    assert "contracts" not in snapshot
    assert "all_contracts" not in snapshot
    assert snapshot["ceiling_amount"] == "5000.00"


def test_duplicate_policy_number_is_rejected(tmp_path):
    path = tmp_path / "contracts.json"
    _write_contracts(path, [_contract("POL-DUP"), _contract("POL-DUP")])

    with pytest.raises(ContractRepositoryError, match="dupliqué"):
        load_contracts(path)


def test_end_date_before_start_date_is_rejected(tmp_path):
    path = tmp_path / "contracts.json"
    _write_contracts(path, [_contract(end_date="2025-12-31")])

    with pytest.raises(ContractRepositoryError, match="end_date"):
        load_contracts(path)


def test_unknown_contract_field_is_rejected():
    with pytest.raises(ValidationError):
        SyntheticContract.model_validate({**_contract(), "unexpected": "nope"})


def test_agent_resolves_single_contract_by_policy_number_without_full_repository():
    result = run(
        case_id="CLM-0001",
        policy_number="POL-0001",
        extracted_fields={
            "patient_id": "PAT-001",
            "patient_name": "Jane Doe",
            "payer_name": "Cigna Health",
            "amount_requested": "100.00",
            "service_date": "2026-06-15",
        },
    )

    dumped = result.model_dump(mode="json")

    assert result.identity.status == VerificationStatus.PASS
    assert result.coverage.status == VerificationStatus.NEEDS_REVIEW
    assert "contracts" not in dumped
    assert "POL-0015" not in result.model_dump_json()
