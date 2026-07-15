"""Tests de chat/nlu.py (Phase V2-11a/V2-11b)."""
from __future__ import annotations

from unittest.mock import Mock

from chat.nlu import extract_intent
from chat.schemas import ChatIntent, LlmIntentDecision, SimulationChangeRequest


class TestExtractIntent:
    def test_explain_intent_detected(self, monkeypatch):
        monkeypatch.setattr(
            "chat.nlu._invoke_llm_intent",
            Mock(return_value=LlmIntentDecision(intents=[ChatIntent.EXPLAIN], case_id="CLM-1001")),
        )
        result = extract_intent("Pourquoi ce dossier est rejeté ?")
        assert result is not None
        assert ChatIntent.EXPLAIN in result.intents
        assert result.case_id == "CLM-1001"

    def test_caller_case_id_always_wins_over_llm_detected_case_id(self, monkeypatch):
        """Le `case_id` déjà connu de l'appelant (contexte UI) prime
        toujours sur celui que le LLM prétend avoir détecté dans le texte."""
        monkeypatch.setattr(
            "chat.nlu._invoke_llm_intent",
            Mock(return_value=LlmIntentDecision(intents=[ChatIntent.EXPLAIN], case_id="CLM-9999")),
        )
        result = extract_intent("Explique-moi ce dossier", case_id="CLM-1001")
        assert result is not None
        assert result.case_id == "CLM-1001"

    def test_llm_detected_case_id_used_when_caller_provides_none(self, monkeypatch):
        monkeypatch.setattr(
            "chat.nlu._invoke_llm_intent",
            Mock(return_value=LlmIntentDecision(intents=[ChatIntent.EXPLAIN], case_id="CLM-2002")),
        )
        result = extract_intent("Explique-moi le dossier CLM-2002")
        assert result is not None
        assert result.case_id == "CLM-2002"

    def test_llm_unavailable_returns_none(self, monkeypatch):
        monkeypatch.setattr("chat.nlu._invoke_llm_intent", Mock(return_value=None))
        result = extract_intent("N'importe quoi")
        assert result is None

    def test_multiple_intents_preserved(self, monkeypatch):
        monkeypatch.setattr(
            "chat.nlu._invoke_llm_intent",
            Mock(
                return_value=LlmIntentDecision(
                    intents=[ChatIntent.ANALYZE, ChatIntent.CORRECT], case_id="CLM-3003"
                )
            ),
        )
        result = extract_intent("Analyse ce dossier et dis-moi quoi corriger", case_id="CLM-3003")
        assert result is not None
        assert set(result.intents) == {ChatIntent.ANALYZE, ChatIntent.CORRECT}

    def test_ambiguous_message_yields_clarify_needed(self, monkeypatch):
        monkeypatch.setattr(
            "chat.nlu._invoke_llm_intent",
            Mock(return_value=LlmIntentDecision(intents=[ChatIntent.CLARIFY_NEEDED])),
        )
        result = extract_intent("bonjour")
        assert result is not None
        assert result.intents == [ChatIntent.CLARIFY_NEEDED]

    def test_no_case_id_from_either_source_stays_none(self, monkeypatch):
        monkeypatch.setattr(
            "chat.nlu._invoke_llm_intent",
            Mock(return_value=LlmIntentDecision(intents=[ChatIntent.ANALYZE], case_id=None)),
        )
        result = extract_intent("Analyse ce dossier")
        assert result is not None
        assert result.case_id is None

    def test_simulation_changes_propagated_to_nlu_result(self, monkeypatch):
        """Régression V2-11b : `LlmIntentDecision.simulation_changes` doit
        être reporté tel quel dans `NluResult` — un oubli de propagation
        rendait `SIMULATE` structurellement impossible à exécuter (le
        planner recevait toujours `simulation_changes=None`)."""
        monkeypatch.setattr(
            "chat.nlu._invoke_llm_intent",
            Mock(
                return_value=LlmIntentDecision(
                    intents=[ChatIntent.SIMULATE],
                    case_id="CLM-4001",
                    simulation_changes=SimulationChangeRequest(remove_document="ordonnance"),
                )
            ),
        )
        result = extract_intent("Et si on retirait l'ordonnance ?", case_id="CLM-4001")
        assert result is not None
        assert result.simulation_changes == SimulationChangeRequest(remove_document="ordonnance")

    def test_no_simulation_changes_stays_none(self, monkeypatch):
        monkeypatch.setattr(
            "chat.nlu._invoke_llm_intent",
            Mock(return_value=LlmIntentDecision(intents=[ChatIntent.EXPLAIN], case_id="CLM-4002")),
        )
        result = extract_intent("Pourquoi ?", case_id="CLM-4002")
        assert result is not None
        assert result.simulation_changes is None
