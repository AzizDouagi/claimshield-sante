from schemas.domain import VerificationStatus
from tools.medical_coding import code_medications, code_procedures, lookup_code
from tools.rule_loader import get_rule_version, load_rules


def test_medical_codes_load_versioned_table():
    table = load_rules("medical_codes.yaml")

    assert get_rule_version("medical_codes.yaml") == "1.0.0"
    assert table["ruleset"]["status"] == "active"
    assert table["rules"][0]["id"] == "MEDICAL_CODE_EXACT_MATCH"
    assert "Office Visit" in table["procedures"]
    assert "Acetaminophen 325 MG Oral Tablet" in table["medications"]


def test_lookup_code_exact_match_passes():
    coding = lookup_code("Office Visit", "procedures")

    assert coding.status == VerificationStatus.PASS
    assert coding.proposed_code == "11429006"
    assert coding.rule_applied == "exact_match"


def test_lookup_code_keyword_match_requires_review():
    coding = lookup_code("Unknown dental procedure", "procedures")

    assert coding.status == VerificationStatus.NEEDS_REVIEW
    assert coding.proposed_code is None
    assert coding.rule_applied == "keyword_match"


def test_code_helpers_preserve_count():
    codings = [
        *code_procedures(["Office Visit"]),
        *code_medications(["Acetaminophen 325 MG Oral Tablet"]),
    ]

    assert len(codings) == 2
    assert all(c.status == VerificationStatus.PASS for c in codings)
