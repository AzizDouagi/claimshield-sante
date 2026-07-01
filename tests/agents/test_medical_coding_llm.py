from agents.medical_coding_agent.agent import run
from agents.medical_coding_agent.schemas import LlmCodingDecision, LlmResolvedCode
from schemas.domain import VerificationStatus


def test_medical_coding_llm_resout_needs_review(monkeypatch):
    monkeypatch.setattr(
        "agents.medical_coding_agent.agent._invoke_llm_react",
        lambda *_: LlmCodingDecision(
            resolved=[
                LlmResolvedCode(
                    description="Unknown dental procedure",
                    proposed_code="34043003",
                    rationale="Correspondance confirmée par outil.",
                )
            ],
            overall_rationale="Résolution LLM appliquée.",
        ),
    )

    result = run("CLM-9009", procedures=["Unknown dental procedure"])

    assert result.status == VerificationStatus.PASS
    assert result.codings[0].proposed_code == "34043003"
    assert "Résolution LLM appliquée." in result.reasons


def test_medical_coding_llm_rejette_code_hallucine(monkeypatch):
    monkeypatch.setattr(
        "agents.medical_coding_agent.agent._invoke_llm_react",
        lambda *_: LlmCodingDecision(
            resolved=[
                LlmResolvedCode(
                    description="Unknown dental procedure",
                    proposed_code="CODE-HALLUCINE",
                    rationale="Code absent du référentiel.",
                )
            ],
            overall_rationale="Tentative rejetée.",
        ),
    )

    result = run("CLM-9012", procedures=["Unknown dental procedure"])

    assert result.status == VerificationStatus.NEEDS_REVIEW
    assert result.codings[0].proposed_code is None
    assert result.codings[0].rule_applied == "llm_rejected_not_in_reference"
    assert any("non déterminé" in reason for reason in result.reasons)


def test_medical_coding_llm_indisponible_conserve_needs_review(monkeypatch):
    monkeypatch.setattr("agents.medical_coding_agent.agent._invoke_llm_react", lambda *_: None)

    result = run("CLM-9010", procedures=["Unknown dental procedure"])

    assert result.status == VerificationStatus.NEEDS_REVIEW
    assert result.codings[0].proposed_code is None


def test_medical_coding_llm_json_invalide_conserve_needs_review(monkeypatch):
    monkeypatch.setattr("agents.medical_coding_agent.agent._invoke_llm_react", lambda *_: None)

    result = run("CLM-9011", procedures=["Unknown dental procedure"])

    assert result.status == VerificationStatus.NEEDS_REVIEW
