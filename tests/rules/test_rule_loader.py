from pathlib import Path
from types import MappingProxyType

import pytest
import yaml

import tools.rule_loader as rule_loader


def _write_rules(
    path: Path,
    *,
    rule_id: str | None = "KNOWN_RULE",
    enabled: bool = True,
    duplicate: bool = False,
) -> None:
    rule = {
        "enabled": enabled,
        "severity": "blocking",
        "description": "Règle de test déterministe.",
    }
    if rule_id is not None:
        rule["id"] = rule_id
    payload = {
        "ruleset": {
            "name": "test-rules",
            "version": "1.0.0",
            "effective_from": "2026-01-01",
            "description": "Jeu de règles de test.",
            "parameters": {"mode": "strict"},
            "thresholds": {"limit": "10"},
            "result_codes": ["PASS", "FAIL"],
            "status": "active",
        },
        "rules": [rule, dict(rule)] if duplicate else [rule],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


@pytest.fixture
def isolated_rules_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(rule_loader, "RULES_DIR", tmp_path)
    monkeypatch.setitem(rule_loader._KNOWN_RULE_IDS, "test_rules.yaml", {"KNOWN_RULE"})
    rule_loader.load_rules.cache_clear()
    yield tmp_path
    rule_loader.load_rules.cache_clear()


def test_common_ruleset_format_is_validated_and_flattened(isolated_rules_dir):
    path = isolated_rules_dir / "test_rules.yaml"
    _write_rules(path)

    rules = rule_loader.load_rules("test_rules.yaml")

    assert rules["ruleset"]["name"] == "test-rules"
    assert rules["version"] == "1.0.0"
    assert rules["mode"] == "strict"
    assert rules["limit"] == "10"
    assert rules["result_codes"] == ("PASS", "FAIL")
    assert len(rules["rule_file_hash"]) == 64
    assert isinstance(rules, MappingProxyType)
    with pytest.raises(TypeError):
        rules["version"] = "2.0.0"


def test_ruleset_without_version_is_rejected(isolated_rules_dir):
    path = isolated_rules_dir / "test_rules.yaml"
    _write_rules(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    del payload["ruleset"]["version"]
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(rule_loader.RuleVersionMissingError, match="RULE_VERSION_MISSING") as exc_info:
        rule_loader.load_rules("test_rules.yaml")
    assert exc_info.value.code == "RULE_VERSION_MISSING"


def test_rule_without_identifier_is_rejected(isolated_rules_dir):
    path = isolated_rules_dir / "test_rules.yaml"
    _write_rules(path, rule_id=None)

    with pytest.raises(rule_loader.RuleFileInvalidError, match="RULE_FILE_INVALID") as exc_info:
        rule_loader.load_rules("test_rules.yaml")
    assert exc_info.value.code == "RULE_FILE_INVALID"


def test_unknown_rule_identifier_is_rejected(isolated_rules_dir):
    path = isolated_rules_dir / "test_rules.yaml"
    _write_rules(path, rule_id="UNKNOWN_RULE")

    with pytest.raises(rule_loader.RuleFileInvalidError, match="RULE_FILE_INVALID") as exc_info:
        rule_loader.load_rules("test_rules.yaml")
    assert exc_info.value.code == "RULE_FILE_INVALID"


def test_invalid_yaml_is_rejected(isolated_rules_dir):
    path = isolated_rules_dir / "test_rules.yaml"
    path.write_text("ruleset: [", encoding="utf-8")

    with pytest.raises(rule_loader.RuleFileInvalidError, match="RULE_FILE_INVALID") as exc_info:
        rule_loader.load_rules("test_rules.yaml")
    assert exc_info.value.code == "RULE_FILE_INVALID"


def test_missing_rule_file_returns_controlled_error(isolated_rules_dir):
    with pytest.raises(rule_loader.RuleFileNotFoundError, match="RULE_FILE_NOT_FOUND") as exc_info:
        rule_loader.load_rules("missing.yaml")

    assert exc_info.value.code == "RULE_FILE_NOT_FOUND"


def test_absolute_rule_path_is_rejected(isolated_rules_dir):
    external = isolated_rules_dir.parent / "external.yaml"
    _write_rules(external)

    with pytest.raises(rule_loader.RulePathNotAllowedError, match="RULE_PATH_NOT_ALLOWED") as exc_info:
        rule_loader.load_rules(str(external))

    assert exc_info.value.code == "RULE_PATH_NOT_ALLOWED"


def test_parent_traversal_is_rejected(isolated_rules_dir):
    with pytest.raises(rule_loader.RulePathNotAllowedError, match="RULE_PATH_NOT_ALLOWED") as exc_info:
        rule_loader.load_rules("../test_rules.yaml")

    assert exc_info.value.code == "RULE_PATH_NOT_ALLOWED"


def test_duplicate_rule_identifier_is_rejected(isolated_rules_dir):
    path = isolated_rules_dir / "test_rules.yaml"
    _write_rules(path, duplicate=True)

    with pytest.raises(rule_loader.RuleIdDuplicateError, match="RULE_ID_DUPLICATE") as exc_info:
        rule_loader.load_rules("test_rules.yaml")

    assert exc_info.value.code == "RULE_ID_DUPLICATE"


def test_disabled_rule_is_rejected(isolated_rules_dir):
    path = isolated_rules_dir / "test_rules.yaml"
    _write_rules(path, enabled=False)

    with pytest.raises(rule_loader.RuleDisabledError, match="RULE_DISABLED") as exc_info:
        rule_loader.load_rules("test_rules.yaml")

    assert exc_info.value.code == "RULE_DISABLED"
