from decimal import Decimal

from tools.rule_loader import get_rule_version, load_rules


def test_authorization_rules_load_expected_thresholds_and_codes():
    rules = load_rules("authorization_rules.yaml")

    assert get_rule_version("authorization_rules.yaml") == "1.0.0"
    assert rules["ruleset"]["status"] == "active"
    assert rules["rules"][0]["id"] == "PREAUTH_REQUIRED_BY_AMOUNT"
    assert Decimal(rules["preauth_threshold_usd"]) == Decimal("3000.00")
    assert "PREAUTH_REQUIRED" in rules["denial_codes"]
    assert rules["denial_severity"]["INVALID_PATIENT_ID"] == "CRITICAL"
