import json
from pathlib import Path

from agents.fhir_validator_agent.agent import run
from agents.fhir_validator_agent.schemas import LlmFhirDecision
from schemas.domain import VerificationStatus


def _bundle(tmp_path: Path) -> Path:
    path = tmp_path / "bundle.json"
    path.write_text(
        json.dumps({
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [{"resource": {"resourceType": "Patient", "id": "PAT-001"}}],
        }),
        encoding="utf-8",
    )
    return path


def _bundle_with_warning(tmp_path: Path) -> Path:
    path = tmp_path / "bundle-warning.json"
    path.write_text(
        json.dumps({
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "PAT-001"}},
                {"resource": {"resourceType": "Observation", "id": "OBS-001"}},
            ],
        }),
        encoding="utf-8",
    )
    return path


def test_fhir_validator_llm_nominal(tmp_path, monkeypatch):
    calls: list[dict] = []

    def _llm(**kwargs):
        calls.append(kwargs)
        return LlmFhirDecision(
            recommended_status="NEEDS_REVIEW",
            clinical_context="Contexte clinique à relire.",
            reasons=["Revue LLM demandée."],
        )

    monkeypatch.setattr(
        "agents.fhir_validator_agent.agent._invoke_llm_fhir",
        _llm,
    )

    result = run("CLM-9005", str(_bundle(tmp_path)))

    assert len(calls) == 1
    assert result.status == VerificationStatus.NEEDS_REVIEW
    assert result.llm_metadata.model_name
    assert "Revue LLM demandée." in result.reasons


def test_fhir_validator_llm_indisponible_needs_review(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.fhir_validator_agent.agent._invoke_llm_fhir", lambda **_: None)

    result = run("CLM-9006", str(_bundle(tmp_path)))

    assert result.status == VerificationStatus.NEEDS_REVIEW
    assert result.llm_metadata.model_name
    assert any("LLM indisponible" in reason for reason in result.reasons)


def test_fhir_validator_llm_json_invalide_needs_review(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.fhir_validator_agent.agent._invoke_llm_fhir", lambda **_: None)

    result = run("CLM-9007", str(_bundle(tmp_path)))

    assert result.status == VerificationStatus.NEEDS_REVIEW
    assert result.llm_metadata.model_name


def test_fhir_validator_security_gate_refuse_appelle_llm_fail_closed(monkeypatch):
    calls: list[dict] = []

    def _llm(**kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr("agents.fhir_validator_agent.agent._invoke_llm_fhir", _llm)

    result = run("CLM-9009", security_allowed=False)

    assert len(calls) == 1
    assert calls[0]["deterministic_status"] == "FAIL"
    assert result.status == VerificationStatus.NEEDS_REVIEW
    assert result.llm_metadata.model_name
    assert any("Security Gate non ALLOW" in error for error in result.errors)


def test_fhir_validator_bundle_non_fourni_appelle_llm(monkeypatch):
    calls: list[dict] = []

    def _llm(**kwargs):
        calls.append(kwargs)
        return LlmFhirDecision(
            recommended_status="PASS",
            clinical_context="Bundle non requis confirmé.",
            reasons=["Aucun bundle FHIR attendu."],
        )

    monkeypatch.setattr("agents.fhir_validator_agent.agent._invoke_llm_fhir", _llm)

    result = run("CLM-9010", fhir_bundle_path=None, bundle_expected=False)

    assert len(calls) == 1
    assert calls[0]["validation_scope"] == "STRUCTURAL_ONLY"
    assert result.status == VerificationStatus.PASS
    assert result.llm_metadata.model_name
    assert any("NOT_PROVIDED" in reason for reason in result.reasons)


def test_fhir_validator_hash_incorrect_appelle_llm_sans_assouplir(tmp_path, monkeypatch):
    calls: list[dict] = []

    def _llm(**kwargs):
        calls.append(kwargs)
        return LlmFhirDecision(
            recommended_status="PASS",
            clinical_context="Tentative d'assouplissement.",
            reasons=["Le LLM recommande PASS."],
        )

    monkeypatch.setattr("agents.fhir_validator_agent.agent._invoke_llm_fhir", _llm)

    result = run("CLM-9011", str(_bundle(tmp_path)), expected_sha256="0" * 64)

    assert len(calls) == 1
    assert calls[0]["deterministic_status"] == "FAIL"
    assert result.status == VerificationStatus.FAIL
    assert result.llm_metadata.model_name


def test_fhir_validator_llm_ne_peut_pas_assouplir_needs_review(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agents.fhir_validator_agent.agent._invoke_llm_fhir",
        lambda **_: LlmFhirDecision(
            recommended_status="PASS",
            clinical_context="Tentative d'assouplissement.",
            reasons=["Le LLM recommande PASS."],
        ),
    )
    path = _bundle_with_warning(tmp_path)

    result = run("CLM-9008", str(path), expected_sha256=None)

    assert result.status == VerificationStatus.NEEDS_REVIEW
    assert result.llm_metadata.model_name
    assert any("Observation" in warning for warning in result.warnings)
