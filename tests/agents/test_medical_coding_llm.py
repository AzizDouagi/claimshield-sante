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
    assert result.llm_metadata is not None


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
    assert result.llm_metadata is not None


def test_medical_coding_llm_indisponible_echoue_fail_closed(monkeypatch):
    monkeypatch.setattr("agents.medical_coding_agent.agent._invoke_llm_react", lambda *_: None)

    result = run("CLM-9010", procedures=["Unknown dental procedure"])

    assert result.status == VerificationStatus.FAIL
    assert any("fail-closed" in reason for reason in result.reasons)
    assert result.llm_metadata is not None


def test_medical_coding_llm_json_invalide_echoue_fail_closed(monkeypatch):
    monkeypatch.setattr("agents.medical_coding_agent.agent._invoke_llm_react", lambda *_: None)

    result = run("CLM-9011", procedures=["Unknown dental procedure"])

    assert result.status == VerificationStatus.FAIL
    assert any("réponse invalide" in reason for reason in result.reasons)


def test_medical_coding_llm_ne_peut_pas_remplacer_un_code_exact(monkeypatch):
    monkeypatch.setattr(
        "agents.medical_coding_agent.agent._invoke_llm_react",
        lambda *_: LlmCodingDecision(
            resolved=[
                LlmResolvedCode(
                    description="Office Visit",
                    proposed_code="CODE-HALLUCINE",
                    rationale="Tentative de remplacement interdite.",
                )
            ],
            overall_rationale="Validation LLM exécutée.",
        ),
    )

    result = run("CLM-9013", procedures=["Office Visit"])

    assert result.status == VerificationStatus.PASS
    assert result.codings[0].rule_applied == "exact_match"
    assert result.codings[0].proposed_code != "CODE-HALLUCINE"
    assert "Validation LLM exécutée." in result.reasons
