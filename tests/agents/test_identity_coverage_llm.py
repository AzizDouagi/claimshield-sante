from agents.identity_coverage_agent.agent import run
from agents.identity_coverage_agent.schemas import LlmIdentityCoverageDecision
from schemas.domain import VerificationStatus


def test_identity_coverage_llm_consultatif(monkeypatch):
    monkeypatch.setattr(
        "agents.identity_coverage_agent.agent._invoke_llm_identity_coverage",
        lambda _: LlmIdentityCoverageDecision(
            recommended_identity_status="FAIL",
            recommended_coverage_status="FAIL",
            rationale="Synthèse LLM consultative.",
            warnings=["Revue recommandée."],
        ),
    )

    result = run(
        case_id="CLM-9100",
        extracted_fields={
            "patient_id": "PAT-001",
            "patient_name": "Jane Doe",
            "payer_name": "Cigna Health",
            "amount_requested": "100.00",
            "coverage_rate": "0.80",
        },
    )

    assert result.identity.status == VerificationStatus.PASS
    assert result.coverage.status == VerificationStatus.PASS
    assert "Synthèse LLM consultative." in result.warnings
