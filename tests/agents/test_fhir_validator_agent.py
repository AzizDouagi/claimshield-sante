import json

from schemas.domain import VerificationStatus
from schemas.results import FhirValidatorResult
from agents.fhir_validator_agent.agent import node, run


def test_fhir_validator_run_passes_when_bundle_not_expected():
    result = run("CLM-0001", None, bundle_expected=False)

    assert isinstance(result, FhirValidatorResult)
    assert result.status == VerificationStatus.PASS
    assert result.bundle_expected is False
    assert result.rule_version == "1.0.0"


def test_fhir_validator_run_fails_on_missing_expected_bundle():
    result = run("CLM-0001", None, bundle_expected=True)

    assert result.status == VerificationStatus.FAIL
    assert result.errors


def test_fhir_validator_run_valid_bundle_with_optional_warnings(tmp_path):
    bundle_path = tmp_path / "patient_bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "resourceType": "Bundle",
                "type": "collection",
                "entry": [
                    {
                        "resource": {
                            "resourceType": "Patient",
                            "id": "PAT-001",
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run("CLM-0001", str(bundle_path), bundle_expected=True)

    assert result.status == VerificationStatus.NEEDS_REVIEW
    assert result.profile_checked == "R4"
    assert result.warnings
    assert result.resource_types == ["Patient"]
    assert result.resource_count == 1
    assert result.references_checked is True
    assert any("structurelle uniquement" in reason for reason in result.reasons)


def test_fhir_validator_run_fails_on_malformed_json(tmp_path):
    bundle_path = tmp_path / "bad.json"
    bundle_path.write_text('{"resourceType": "Bundle"', encoding="utf-8")

    result = run("CLM-0001", str(bundle_path), bundle_expected=True)

    assert result.status == VerificationStatus.FAIL
    assert any("JSON malformé" in error for error in result.errors)


def test_fhir_validator_detects_missing_patient_cardinality(tmp_path):
    bundle_path = tmp_path / "no_patient.json"
    bundle_path.write_text(
        json.dumps({"resourceType": "Bundle", "type": "collection", "entry": []}),
        encoding="utf-8",
    )

    result = run("CLM-0001", str(bundle_path), bundle_expected=True)

    assert result.status == VerificationStatus.FAIL
    assert any("Patient" in error for error in result.errors)


def test_fhir_validator_detects_unresolved_internal_reference(tmp_path):
    bundle_path = tmp_path / "bad_reference.json"
    bundle_path.write_text(
        json.dumps(
            {
                "resourceType": "Bundle",
                "type": "collection",
                "entry": [
                    {"resource": {"resourceType": "Patient", "id": "PAT-001"}},
                    {
                        "resource": {
                            "resourceType": "Coverage",
                            "id": "COV-001",
                            "status": "active",
                            "beneficiary": {"reference": "Patient/PAT-MISSING"},
                        }
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run("CLM-0001", str(bundle_path), bundle_expected=True)

    assert result.status == VerificationStatus.FAIL
    assert any("Référence interne non résolue" in error for error in result.errors)


def test_fhir_validator_warns_on_unsupported_resource_type_and_profile(tmp_path):
    bundle_path = tmp_path / "unsupported.json"
    bundle_path.write_text(
        json.dumps(
            {
                "resourceType": "Bundle",
                "type": "collection",
                "entry": [
                    {
                        "resource": {
                            "resourceType": "Patient",
                            "id": "PAT-001",
                            "meta": {"profile": ["http://example.test/fhir/CustomProfile"]},
                        }
                    },
                    {"resource": {"resourceType": "Observation", "id": "OBS-001"}},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run("CLM-0001", str(bundle_path), bundle_expected=True)

    assert result.status == VerificationStatus.NEEDS_REVIEW
    assert any("resourceType non supporté" in warning for warning in result.warnings)
    assert any("Profil FHIR non supporté" in warning for warning in result.warnings)


def test_fhir_validator_node_consumes_input():
    updates = node(
        {
            "case_id": "CLM-0001",
            "fhir_input": {
                "case_id": "CLM-0001",
                "fhir_bundle_path": None,
                "bundle_expected": False,
            },
        }
    )

    assert updates["fhir_input"] is None
    assert updates["current_step"] == "fhir_validation"
    assert isinstance(updates["fhir_result"], FhirValidatorResult)
