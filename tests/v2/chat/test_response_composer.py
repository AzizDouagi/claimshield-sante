"""Tests de chat/response_composer.py (Phase V2-11a) — synthèse LLM et
garde-fou anti-hallucination sont couverts ici ; le corpus de formulations
libres EXPLAIN/CORRECT/ANALYZE bout en bout vit dans `test_agent.py`."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from chat.response_composer import _invoke_llm_compose, compose
from chat.schemas import ChatIntent, CorrectionRecommendation, ExplanationFacts
from schemas.domain import ClaimDecisionV2
from schemas.v2_results import DecisionAssumption, DecisionCounterfactual, MissingInformation, MissingInformationDimension, MissingInformationImportance


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


class TestAnswerModeWiring:
    """Plan de remédiation « autonomie décisionnelle V2 », Phase 7."""

    def test_llm_payload_carries_answer_modes(self, monkeypatch):
        compose_spy = Mock(return_value="Réponse.")
        monkeypatch.setattr("chat.response_composer._invoke_llm_compose", compose_spy)
        compose(
            case_id="CLM-1006",
            intents=[ChatIntent.EXPLAIN],
            tool_results={
                "explanation": ExplanationFacts(
                    case_id="CLM-1006",
                    missing_information=[
                        MissingInformation(
                            code="UNRESOLVED_CODING",
                            description="Codification non résolue.",
                            importance=MissingInformationImportance.IMPORTANT,
                            affected_dimension=MissingInformationDimension.CODING,
                            source_agent="medical_risk_agent",
                            impact_on_decision="Confiance réduite.",
                        )
                    ],
                )
            },
        )
        payload = compose_spy.call_args[0][0]
        assert "ASSUMPTION" in payload["answer_modes"]
        assert "FACT" in payload["answer_modes"]

    def test_fallback_labels_missing_information_and_assumptions(self, monkeypatch):
        monkeypatch.setattr("chat.response_composer._invoke_llm_compose", Mock(return_value=None))
        result = compose(
            case_id="CLM-1007",
            intents=[ChatIntent.EXPLAIN],
            tool_results={
                "explanation": ExplanationFacts(
                    case_id="CLM-1007",
                    assumptions=[DecisionAssumption(code="X", description="Décision malgré une donnée incomplète.")],
                    counterfactuals=[
                        DecisionCounterfactual(
                            condition="Code résolu",
                            current_value="Non résolu",
                            required_value="Résolu",
                            resulting_decision=ClaimDecisionV2.APPROVE,
                            explanation="Un code résolu changerait la décision.",
                        )
                    ],
                    recommended_action="Vérifier la codification.",
                )
            },
        )
        assert "[HYPOTHÈSE]" in result
        assert "Décision malgré une donnée incomplète." in result
        assert "Code résolu" in result
        assert "Vérifier la codification." in result

    def test_fallback_labels_simulation(self, monkeypatch):
        from chat.schemas import SimulationResult

        monkeypatch.setattr("chat.response_composer._invoke_llm_compose", Mock(return_value=None))
        result = compose(
            case_id="CLM-1008",
            intents=[ChatIntent.SIMULATE],
            tool_results={
                "simulation": SimulationResult(
                    case_id="CLM-1008",
                    applied=True,
                    original_decision="APPROVE",
                    simulated_decision="REJECT",
                    decision_changed=True,
                )
            },
        )
        assert "[SIMULATION]" in result


class TestUsageCapture:
    """`compose(..., usage_sink=...)` — visibilité temps réel des tokens
    demandée par AZIZ (« comme Claude Code »), voir `chat/llm_usage.py`."""

    def test_compose_populates_usage_sink_on_success(self, monkeypatch):
        fake_llm = Mock()
        fake_llm.invoke.return_value = SimpleNamespace(
            content="Le dossier CLM-2001 a été rejeté pour motif X.",
            usage_metadata={"input_tokens": 300, "output_tokens": 60, "total_tokens": 360},
            response_metadata={"model_name": "gemma4:latest"},
        )
        monkeypatch.setattr("chat.response_composer.get_llm", lambda: fake_llm)
        sink: dict = {}
        result = compose(
            case_id="CLM-2001",
            intents=[ChatIntent.EXPLAIN],
            tool_results={"context": {"case_id": "CLM-2001", "final_decision": "REJECT"}},
            usage_sink=sink,
        )
        assert result == "Le dossier CLM-2001 a été rejeté pour motif X."
        assert sink == {"input_tokens": 300, "output_tokens": 60, "model_name": "gemma4:latest"}

    def test_compose_usage_sink_none_by_default_never_raises(self, monkeypatch):
        fake_llm = Mock()
        fake_llm.invoke.return_value = SimpleNamespace(
            content="Le dossier CLM-2002 a été rejeté.",
            usage_metadata={"input_tokens": 1, "output_tokens": 1},
            response_metadata={},
        )
        monkeypatch.setattr("chat.response_composer.get_llm", lambda: fake_llm)
        result = compose(
            case_id="CLM-2002",
            intents=[ChatIntent.EXPLAIN],
            tool_results={"context": {"case_id": "CLM-2002", "final_decision": "REJECT"}},
        )
        assert result == "Le dossier CLM-2002 a été rejeté."

    def test_usage_sink_untouched_on_llm_failure(self, monkeypatch):
        fake_llm = Mock()
        fake_llm.invoke.side_effect = RuntimeError("Ollama indisponible")
        monkeypatch.setattr("chat.response_composer.get_llm", lambda: fake_llm)
        sink: dict = {}
        result = _invoke_llm_compose({"case_id": "CLM-2003"}, sink)
        assert result is None
        assert sink == {}
