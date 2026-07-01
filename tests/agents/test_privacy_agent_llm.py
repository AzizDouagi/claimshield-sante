from agents.privacy_agent.agent import run
from agents.privacy_agent.schemas import LlmPrivacyDecision, PrivacyInput, ReaderRole
from schemas.domain import DataClassification, SecurityDecision, VerificationStatus
from schemas.results import SecurityGateResult


def _input() -> PrivacyInput:
    return PrivacyInput(
        case_id="CLM-9008",
        role=ReaderRole.ADMINISTRATIVE_MANAGER,
        data_classification=DataClassification.SYNTHETIC_TEST_DATA,
    )


def _gate() -> SecurityGateResult:
    return SecurityGateResult(
        claim_id="CLM-9008",
        decision=SecurityDecision.ALLOW,
        reasons=["ok"],
    )


def test_privacy_llm_nominal(monkeypatch):
    monkeypatch.setattr(
        "agents.privacy_agent.agent._invoke_llm_privacy",
        lambda _: LlmPrivacyDecision(
            audit_justification="Justification LLM privacy.",
            data_classification_reason="Classification synthétique confirmée.",
        ),
    )

    result = run(_input(), _gate())

    assert result.status == VerificationStatus.PASS
    assert "Justification LLM privacy." in result.reasons
    assert "Classification synthétique confirmée." in result.reasons


def test_privacy_llm_indisponible_justification_generique(monkeypatch):
    monkeypatch.setattr("agents.privacy_agent.agent._invoke_llm_privacy", lambda _: None)

    result = run(_input(), _gate())

    assert result.status == VerificationStatus.PASS
    assert any("Justification LLM indisponible" in reason for reason in result.reasons)


def test_privacy_llm_json_invalide_justification_generique(monkeypatch):
    monkeypatch.setattr("agents.privacy_agent.agent._invoke_llm_privacy", lambda _: None)

    result = run(_input(), _gate())

    assert any("Justification LLM indisponible" in reason for reason in result.reasons)
