"""Tests de chat/memory_schemas.py — plan de remédiation « autonomie
décisionnelle V2 », Phase 8."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from chat.answer_mode import AnswerMode
from chat.memory_schemas import (
    ConversationContext,
    ConversationSemanticState,
    ConversationTurn,
    DiscussedScenario,
    LlmSemanticSummaryProposal,
    SimulationContext,
)
from chat.schemas import ChatIntent


def _now():
    return datetime.now(UTC)


class TestConversationTurn:
    def test_never_carries_raw_message_text(self):
        """Aucun champ de `ConversationTurn` ne peut porter le texte
        intégral d'un message — uniquement des empreintes."""
        turn = ConversationTurn(
            turn_id="t1",
            message_digest="a" * 64,
            reply_digest="b" * 64,
            intents=[ChatIntent.EXPLAIN],
            case_id="CLM-8001",
            answer_modes=[AnswerMode.FACT],
            created_at=_now(),
        )
        assert "message" not in turn.model_dump()
        assert turn.message_digest == "a" * 64

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            ConversationTurn(
                turn_id="t1",
                message_digest="a" * 64,
                reply_digest="b" * 64,
                created_at=_now(),
                raw_message="jamais ceci",
            )

    def test_invalid_case_id_pattern_rejected(self):
        with pytest.raises(ValidationError):
            ConversationTurn(
                turn_id="t1", message_digest="a" * 64, reply_digest="b" * 64, case_id="not-a-case-id", created_at=_now()
            )


class TestDiscussedScenario:
    def test_valid_kinds(self):
        for kind in ("REAL_DECISION", "SIMULATION", "COUNTERFACTUAL"):
            scenario = DiscussedScenario(scenario_id="SCENARIO-1", description="x", kind=kind)
            assert scenario.kind == kind

    def test_invalid_kind_rejected(self):
        with pytest.raises(ValidationError):
            DiscussedScenario(scenario_id="SCENARIO-1", description="x", kind="INVENTED_KIND")

    def test_scenario_id_pattern_enforced(self):
        with pytest.raises(ValidationError):
            DiscussedScenario(scenario_id="not-a-scenario", description="x", kind="REAL_DECISION")


class TestConversationSemanticState:
    def test_bounded_lists(self):
        with pytest.raises(ValidationError):
            ConversationSemanticState(
                conversation_summary="résumé",
                discussed_scenarios=[
                    DiscussedScenario(scenario_id=f"SCENARIO-{i}", description="x", kind="REAL_DECISION")
                    for i in range(11)
                ],
                updated_at=_now(),
            )

    def test_defaults_are_empty(self):
        state = ConversationSemanticState(conversation_summary="résumé", updated_at=_now())
        assert state.discussed_scenarios == []
        assert state.open_questions == []
        assert state.resolved_references == {}
        assert state.compared_decisions == []
        assert state.last_user_goal is None

    def test_round_trip_json(self):
        state = ConversationSemanticState(
            conversation_summary="Le dossier CLM-8001 a été refusé.",
            discussed_scenarios=[
                DiscussedScenario(scenario_id="SCENARIO-1", description="Décision réelle", kind="REAL_DECISION")
            ],
            resolved_references={"le premier scénario": "SCENARIO-1"},
            updated_at=_now(),
        )
        restored = ConversationSemanticState.model_validate(state.model_dump(mode="json"))
        assert restored == state


class TestConversationContext:
    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            ConversationContext(thread_id="t", user_id="u", updated_at=_now(), unknown_field="x")

    def test_defaults_empty_without_semantic_state(self):
        context = ConversationContext(thread_id="t", user_id="u", updated_at=_now())
        assert context.turns == []
        assert context.simulations == []
        assert context.semantic_state is None


class TestSimulationContext:
    def test_scenario_id_pattern_enforced(self):
        with pytest.raises(ValidationError):
            SimulationContext(scenario_id="bad-id")


class TestLlmSemanticSummaryProposal:
    def test_defaults_are_empty(self):
        proposal = LlmSemanticSummaryProposal()
        assert proposal.conversation_summary == ""
        assert proposal.discussed_scenarios == []
        assert proposal.resolved_references == {}
