from schemas.domain import VerificationStatus
from schemas.results import MedicalCodingResult
from agents.medical_coding_agent.agent import node, run


def test_medical_coding_run_passes_on_exact_matches():
    result = run(
        case_id="CLM-0001",
        procedures=["Office Visit"],
        medications=["Acetaminophen 325 MG Oral Tablet"],
    )

    assert isinstance(result, MedicalCodingResult)
    assert result.status == VerificationStatus.PASS
    assert len(result.codings) == 2
    assert all(c.proposed_code for c in result.codings)


def test_medical_coding_same_input_and_version_is_deterministic():
    first = run(case_id="CLM-0001", procedures=["Office Visit"])
    second = run(case_id="CLM-0001", procedures=["Office Visit"])

    assert first.table_version == second.table_version == "1.0.0"
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_medical_coding_run_needs_review_on_unknown_description():
    result = run(case_id="CLM-0001", procedures=["Unknown dental procedure"])

    assert result.status == VerificationStatus.NEEDS_REVIEW
    assert result.codings[0].rule_applied == "keyword_match"


def test_medical_coding_run_needs_review_without_items():
    result = run(case_id="CLM-0001")

    assert result.status == VerificationStatus.NEEDS_REVIEW
    assert result.codings == []


def test_medical_coding_node_consumes_input_and_adds_audit():
    updates = node(
        {
            "case_id": "CLM-0001",
            "coding_input": {
                "case_id": "CLM-0001",
                "procedures": ["Office Visit"],
                "medications": [],
            },
        }
    )

    assert updates["coding_input"] is None
    assert updates["current_step"] == "medical_coding"
    assert updates["completed_steps"] == ["medical_coding"]
    assert updates["audit_trail"]
    assert isinstance(updates["coding_result"], MedicalCodingResult)
