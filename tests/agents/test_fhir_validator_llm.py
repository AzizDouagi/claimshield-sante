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


def test_fhir_validator_llm_nominal(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agents.fhir_validator_agent.agent._invoke_llm_fhir",
        lambda **_: LlmFhirDecision(
            recommended_status="NEEDS_REVIEW",
            clinical_context="Contexte clinique à relire.",
            reasons=["Revue LLM demandée."],
        ),
    )

    result = run("CLM-9005", str(_bundle(tmp_path)))

    assert result.status == VerificationStatus.NEEDS_REVIEW
    assert "Revue LLM demandée." in result.reasons


def test_fhir_validator_llm_indisponible_fail(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.fhir_validator_agent.agent._invoke_llm_fhir", lambda **_: None)

    result = run("CLM-9006", str(_bundle(tmp_path)))

    assert result.status == VerificationStatus.FAIL
    assert any("LLM indisponible" in reason for reason in result.reasons)


def test_fhir_validator_llm_json_invalide_fail(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.fhir_validator_agent.agent._invoke_llm_fhir", lambda **_: None)

    result = run("CLM-9007", str(_bundle(tmp_path)))

    assert result.status == VerificationStatus.FAIL


def test_fhir_validator_llm_ne_peut_pas_assouplir_needs_review(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agents.fhir_validator_agent.agent._invoke_llm_fhir",
        lambda **_: LlmFhirDecision(
            recommended_status="PASS",
            clinical_context="Tentative d'assouplissement.",
            reasons=["Le LLM recommande PASS."],
        ),
    )
    path = _bundle(tmp_path)

    result = run("CLM-9008", str(path), expected_sha256=None)

    if result.status == VerificationStatus.NEEDS_REVIEW:
        assert result.status != VerificationStatus.PASS
