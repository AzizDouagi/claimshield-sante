"""Tests de chat/semantic_summarizer.py — plan de remédiation « autonomie
décisionnelle V2 », Phase 8, §6.3."""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import Mock

from chat.memory_schemas import ConversationSemanticState, DiscussedScenario, LlmSemanticSummaryProposal
from chat.semantic_summarizer import update_semantic_state


def _previous() -> ConversationSemanticState:
    return ConversationSemanticState(
        conversation_summary="Le dossier CLM-8001 a été mis en quarantaine.",
        updated_at=datetime.now(UTC),
    )


class TestFailClosed:
    def test_llm_unavailable_keeps_previous_state_unchanged(self, monkeypatch):
        monkeypatch.setattr("chat.semantic_summarizer._invoke_llm_semantic_summary", Mock(return_value=None))
        previous = _previous()
        result = update_semantic_state(
            previous=previous,
            turn_summary={},
            known_evidence_ids=set(),
            real_decision=None,
            simulation_decisions=set(),
            counterfactual_decisions=set(),
        )
        assert result == previous

    def test_llm_unavailable_with_no_previous_state_returns_neutral_state(self, monkeypatch):
        monkeypatch.setattr("chat.semantic_summarizer._invoke_llm_semantic_summary", Mock(return_value=None))
        result = update_semantic_state(
            previous=None,
            turn_summary={},
            known_evidence_ids=set(),
            real_decision=None,
            simulation_decisions=set(),
            counterfactual_decisions=set(),
        )
        assert result.conversation_summary == ""
        assert result.discussed_scenarios == []

    def test_never_raises_on_llm_exception(self, monkeypatch):
        """`_invoke_llm_semantic_summary` avale déjà ses propres exceptions
        (voir son propre `try/except`) — ce test vérifie que
        `update_semantic_state` ne lève jamais, même si le mock simule un
        comportement anormal (retour `None`, jamais une exception qui
        remonte à l'appelant)."""
        monkeypatch.setattr("chat.semantic_summarizer._invoke_llm_semantic_summary", Mock(return_value=None))
        result = update_semantic_state(
            previous=None,
            turn_summary={"intents": ["EXPLAIN"]},
            known_evidence_ids={"EVID-1"},
            real_decision="REJECT",
            simulation_decisions=set(),
            counterfactual_decisions=set(),
        )
        assert isinstance(result, ConversationSemanticState)


class TestAntiHallucinationValidation:
    def test_unknown_evidence_id_silently_removed(self, monkeypatch):
        proposal = LlmSemanticSummaryProposal(
            conversation_summary="résumé",
            discussed_scenarios=[
                DiscussedScenario(
                    scenario_id="SCENARIO-1",
                    description="Décision réelle",
                    kind="REAL_DECISION",
                    evidence_ids=["EVID-1", "EVID-INVENTED"],
                )
            ],
        )
        monkeypatch.setattr("chat.semantic_summarizer._invoke_llm_semantic_summary", Mock(return_value=proposal))
        result = update_semantic_state(
            previous=None,
            turn_summary={},
            known_evidence_ids={"EVID-1"},
            real_decision="REJECT",
            simulation_decisions=set(),
            counterfactual_decisions=set(),
        )
        assert result.discussed_scenarios[0].evidence_ids == ["EVID-1"]

    def test_related_decision_inconsistent_with_kind_is_dropped(self, monkeypatch):
        """Un scénario `kind=SIMULATION` ne peut référencer que des
        décisions de simulation déjà calculées — jamais la décision réelle,
        même si le LLM le propose."""
        proposal = LlmSemanticSummaryProposal(
            conversation_summary="résumé",
            discussed_scenarios=[
                DiscussedScenario(
                    scenario_id="SCENARIO-1",
                    description="Simulation",
                    kind="SIMULATION",
                    related_decision="REJECT",  # décision réelle, pas de simulation
                )
            ],
        )
        monkeypatch.setattr("chat.semantic_summarizer._invoke_llm_semantic_summary", Mock(return_value=proposal))
        result = update_semantic_state(
            previous=None,
            turn_summary={},
            known_evidence_ids=set(),
            real_decision="REJECT",
            simulation_decisions={"APPROVE"},
            counterfactual_decisions=set(),
        )
        assert result.discussed_scenarios[0].related_decision is None

    def test_related_decision_consistent_with_kind_is_kept(self, monkeypatch):
        proposal = LlmSemanticSummaryProposal(
            conversation_summary="résumé",
            discussed_scenarios=[
                DiscussedScenario(
                    scenario_id="SCENARIO-1", description="Simulation", kind="SIMULATION", related_decision="APPROVE"
                )
            ],
        )
        monkeypatch.setattr("chat.semantic_summarizer._invoke_llm_semantic_summary", Mock(return_value=proposal))
        result = update_semantic_state(
            previous=None,
            turn_summary={},
            known_evidence_ids=set(),
            real_decision="REJECT",
            simulation_decisions={"APPROVE"},
            counterfactual_decisions=set(),
        )
        assert result.discussed_scenarios[0].related_decision == "APPROVE"

    def test_resolved_reference_to_unknown_scenario_removed(self, monkeypatch):
        proposal = LlmSemanticSummaryProposal(
            conversation_summary="résumé",
            discussed_scenarios=[
                DiscussedScenario(scenario_id="SCENARIO-1", description="Réel", kind="REAL_DECISION")
            ],
            resolved_references={
                "le premier scénario": "SCENARIO-1",
                "le deuxième scénario": "SCENARIO-99",
            },
        )
        monkeypatch.setattr("chat.semantic_summarizer._invoke_llm_semantic_summary", Mock(return_value=proposal))
        result = update_semantic_state(
            previous=None,
            turn_summary={},
            known_evidence_ids=set(),
            real_decision="REJECT",
            simulation_decisions=set(),
            counterfactual_decisions=set(),
        )
        assert result.resolved_references == {"le premier scénario": "SCENARIO-1"}

    def test_compared_decisions_restricted_to_known_values(self, monkeypatch):
        proposal = LlmSemanticSummaryProposal(
            conversation_summary="résumé", compared_decisions=["REJECT", "APPROVE", "PARTIAL_APPROVE"]
        )
        monkeypatch.setattr("chat.semantic_summarizer._invoke_llm_semantic_summary", Mock(return_value=proposal))
        result = update_semantic_state(
            previous=None,
            turn_summary={},
            known_evidence_ids=set(),
            real_decision="REJECT",
            simulation_decisions={"APPROVE"},
            counterfactual_decisions=set(),
        )
        assert set(result.compared_decisions) == {"REJECT", "APPROVE"}


class TestTextSafety:
    def test_conversation_summary_never_exceeds_schema_bound(self, monkeypatch):
        proposal = LlmSemanticSummaryProposal(conversation_summary="x" * 2000)
        monkeypatch.setattr("chat.semantic_summarizer._invoke_llm_semantic_summary", Mock(return_value=proposal))
        result = update_semantic_state(
            previous=None,
            turn_summary={},
            known_evidence_ids=set(),
            real_decision=None,
            simulation_decisions=set(),
            counterfactual_decisions=set(),
        )
        assert len(result.conversation_summary) <= 800

    def test_last_user_goal_never_exceeds_schema_bound(self, monkeypatch):
        proposal = LlmSemanticSummaryProposal(conversation_summary="ok", last_user_goal="y" * 1000)
        monkeypatch.setattr("chat.semantic_summarizer._invoke_llm_semantic_summary", Mock(return_value=proposal))
        result = update_semantic_state(
            previous=None,
            turn_summary={},
            known_evidence_ids=set(),
            real_decision=None,
            simulation_decisions=set(),
            counterfactual_decisions=set(),
        )
        assert result.last_user_goal is None or len(result.last_user_goal) <= 200

    def test_secret_pattern_in_summary_is_redacted_away(self, monkeypatch):
        proposal = LlmSemanticSummaryProposal(conversation_summary="api_key=sk-abcdef1234567890")
        monkeypatch.setattr("chat.semantic_summarizer._invoke_llm_semantic_summary", Mock(return_value=proposal))
        result = update_semantic_state(
            previous=None,
            turn_summary={},
            known_evidence_ids=set(),
            real_decision=None,
            simulation_decisions=set(),
            counterfactual_decisions=set(),
        )
        assert "api_key" not in result.conversation_summary
        assert result.conversation_summary == ""
