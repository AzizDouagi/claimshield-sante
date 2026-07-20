"""Tests de chat/nlu.py (Phase V2-11a/V2-11b/Phase 8 mémoire)."""
from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

from chat.memory_schemas import ConversationSemanticState
from chat.nlu import _invoke_llm_intent, extract_intent
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


class TestScenarioReferenceResolution:
    """Phase 8 (plan de remédiation « autonomie décisionnelle V2 », §6.3) —
    résolution **déterministe, jamais confiée au LLM** d'une référence à un
    scénario déjà discuté."""

    def _semantic_state(self, resolved_references: dict[str, str]) -> ConversationSemanticState:
        return ConversationSemanticState(
            conversation_summary="résumé", resolved_references=resolved_references, updated_at=datetime.now(UTC)
        )

    def test_known_phrase_resolves_to_scenario_id(self, monkeypatch):
        monkeypatch.setattr(
            "chat.nlu._invoke_llm_intent",
            Mock(return_value=LlmIntentDecision(intents=[ChatIntent.EXPLAIN], case_id="CLM-5001")),
        )
        semantic_state = self._semantic_state({"le premier scénario": "SCENARIO-1"})
        result = extract_intent(
            "Compare avec le premier scénario", case_id="CLM-5001", semantic_state=semantic_state
        )
        assert result is not None
        assert result.resolved_scenario_id == "SCENARIO-1"

    def test_phrase_matching_is_case_insensitive(self, monkeypatch):
        monkeypatch.setattr(
            "chat.nlu._invoke_llm_intent",
            Mock(return_value=LlmIntentDecision(intents=[ChatIntent.EXPLAIN], case_id="CLM-5002")),
        )
        semantic_state = self._semantic_state({"cette hypothèse": "SCENARIO-2"})
        result = extract_intent(
            "Et pour CETTE HYPOTHÈSE, ça change quoi ?", case_id="CLM-5002", semantic_state=semantic_state
        )
        assert result is not None
        assert result.resolved_scenario_id == "SCENARIO-2"

    def test_no_known_phrase_in_message_stays_none(self, monkeypatch):
        monkeypatch.setattr(
            "chat.nlu._invoke_llm_intent",
            Mock(return_value=LlmIntentDecision(intents=[ChatIntent.EXPLAIN], case_id="CLM-5003")),
        )
        semantic_state = self._semantic_state({"le premier scénario": "SCENARIO-1"})
        result = extract_intent(
            "Pourquoi ce dossier est-il refusé ?", case_id="CLM-5003", semantic_state=semantic_state
        )
        assert result is not None
        assert result.resolved_scenario_id is None

    def test_no_semantic_state_never_resolves_anything(self, monkeypatch):
        monkeypatch.setattr(
            "chat.nlu._invoke_llm_intent",
            Mock(return_value=LlmIntentDecision(intents=[ChatIntent.EXPLAIN], case_id="CLM-5004")),
        )
        result = extract_intent("Compare avec le premier scénario", case_id="CLM-5004")
        assert result is not None
        assert result.resolved_scenario_id is None

    def test_never_invents_a_scenario_id_absent_from_resolved_references(self, monkeypatch):
        """Même si le LLM (mocké ici de façon adversariale) tentait de
        suggérer un identifiant, seule la résolution déterministe de ce
        module compte — jamais une valeur hors de `resolved_references`."""
        monkeypatch.setattr(
            "chat.nlu._invoke_llm_intent",
            Mock(return_value=LlmIntentDecision(intents=[ChatIntent.EXPLAIN], case_id="CLM-5005")),
        )
        semantic_state = self._semantic_state({})
        result = extract_intent(
            "Compare avec le scénario SCENARIO-99", case_id="CLM-5005", semantic_state=semantic_state
        )
        assert result is not None
        assert result.resolved_scenario_id is None


class _FakeStructured:
    def __init__(self, raw_result: dict):
        self._raw_result = raw_result

    def invoke(self, messages):
        return self._raw_result


class _FakeLlm:
    """Simule `ChatOllama.with_structured_output(..., include_raw=True)` —
    voir `chat/llm_usage.py` (visibilité temps réel des tokens, AZIZ)."""

    def __init__(self, raw_result: dict):
        self._raw_result = raw_result
        self.last_kwargs: dict = {}

    def with_structured_output(self, model, **kwargs):
        self.last_kwargs = kwargs
        return _FakeStructured(self._raw_result)


class TestUsageCapture:
    """`_invoke_llm_intent`/`extract_intent` capturent les tokens via
    `include_raw=True` — voir `chat/llm_usage.py`."""

    def test_include_raw_is_requested(self, monkeypatch):
        fake_llm = _FakeLlm(
            {
                "raw": SimpleNamespace(usage_metadata=None, response_metadata={}),
                "parsed": LlmIntentDecision(intents=[ChatIntent.EXPLAIN]),
                "parsing_error": None,
            }
        )
        monkeypatch.setattr("chat.nlu.get_llm", lambda: fake_llm)
        _invoke_llm_intent("Pourquoi ce dossier est rejeté ?", None)
        assert fake_llm.last_kwargs.get("include_raw") is True

    def test_usage_sink_populated_on_success(self, monkeypatch):
        fake_llm = _FakeLlm(
            {
                "raw": SimpleNamespace(
                    usage_metadata={"input_tokens": 36, "output_tokens": 25, "total_tokens": 61},
                    response_metadata={"model_name": "gemma4:latest"},
                ),
                "parsed": LlmIntentDecision(intents=[ChatIntent.EXPLAIN], case_id="CLM-6001"),
                "parsing_error": None,
            }
        )
        monkeypatch.setattr("chat.nlu.get_llm", lambda: fake_llm)
        sink: dict = {}
        result = _invoke_llm_intent("Explique-moi le dossier CLM-6001", None, usage_sink=sink)
        assert result is not None
        assert result.case_id == "CLM-6001"
        assert sink == {"input_tokens": 36, "output_tokens": 25, "model_name": "gemma4:latest"}

    def test_usage_sink_stays_empty_on_parsing_error(self, monkeypatch):
        fake_llm = _FakeLlm(
            {
                "raw": SimpleNamespace(
                    usage_metadata={"input_tokens": 10, "output_tokens": 3}, response_metadata={}
                ),
                "parsed": None,
                "parsing_error": ValueError("sortie non conforme"),
            }
        )
        monkeypatch.setattr("chat.nlu.get_llm", lambda: fake_llm)
        sink: dict = {}
        result = _invoke_llm_intent("bonjour", None, usage_sink=sink)
        assert result is None
        assert sink == {}

    def test_usage_sink_none_by_default_never_raises(self, monkeypatch):
        fake_llm = _FakeLlm(
            {
                "raw": SimpleNamespace(usage_metadata={"input_tokens": 1, "output_tokens": 1}, response_metadata={}),
                "parsed": LlmIntentDecision(intents=[ChatIntent.EXPLAIN]),
                "parsing_error": None,
            }
        )
        monkeypatch.setattr("chat.nlu.get_llm", lambda: fake_llm)
        result = _invoke_llm_intent("Pourquoi ?", None)
        assert result is not None
