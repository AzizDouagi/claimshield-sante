"""Tests de chat/response_composer.py (Phase V2-11a) — synthèse LLM et
garde-fou anti-hallucination sont couverts ici ; le corpus de formulations
libres EXPLAIN/CORRECT/ANALYZE bout en bout vit dans `test_agent.py`."""
from __future__ import annotations

from unittest.mock import Mock

from chat.response_composer import compose
from chat.schemas import ChatIntent, CorrectionRecommendation, ExplanationFacts


class TestComposeNoData:
    def test_empty_tool_results_returns_information_non_disponible(self):
        result = compose(case_id="CLM-1001", intents=[ChatIntent.ANALYZE], tool_results={})
        assert "non disponible" in result.lower()

    def test_all_none_tool_results_returns_information_non_disponible(self):
        result = compose(
            case_id="CLM-1001",
            intents=[ChatIntent.EXPLAIN],
            tool_results={"context": None, "explanation": None},
        )
        assert "non disponible" in result.lower()


class TestComposeWithLlm:
    def test_llm_reply_returned_when_grounded(self, monkeypatch):
        monkeypatch.setattr(
            "chat.response_composer._invoke_llm_compose",
            Mock(return_value="Le dossier CLM-1002 a été rejeté pour motif X."),
        )
        result = compose(
            case_id="CLM-1002",
            intents=[ChatIntent.EXPLAIN],
            tool_results={"context": {"case_id": "CLM-1002", "final_decision": "REJECT"}},
        )
        assert result == "Le dossier CLM-1002 a été rejeté pour motif X."

    def test_llm_unavailable_falls_back_to_deterministic_composition(self, monkeypatch):
        monkeypatch.setattr("chat.response_composer._invoke_llm_compose", Mock(return_value=None))
        result = compose(
            case_id="CLM-1003",
            intents=[ChatIntent.EXPLAIN],
            tool_results={
                "explanation": ExplanationFacts(
                    case_id="CLM-1003", final_decision="REJECT", decision_summary=["motif"]
                )
            },
        )
        assert "CLM-1003" in result
        assert "REJECT" in result

    def test_fallback_never_crashes_on_corrections_only(self, monkeypatch):
        monkeypatch.setattr("chat.response_composer._invoke_llm_compose", Mock(return_value=None))
        result = compose(
            case_id="CLM-1004",
            intents=[ChatIntent.CORRECT],
            tool_results={
                "corrections": [
                    CorrectionRecommendation(trigger="motif brut", action="Faire quelque chose.")
                ]
            },
        )
        assert "Faire quelque chose." in result
