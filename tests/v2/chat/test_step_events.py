"""Tests de `chat/schemas.py::ChatStepEvent`/`ChatStepStatus`/`ChatTurnSummary`
— visibilité temps réel des étapes/tokens du chat (demandée par AZIZ, « comme
Claude Code »)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from chat.schemas import ChatStepEvent, ChatStepStatus, ChatTurnSummary


class TestChatStepEvent:
    def test_minimal_valid_event(self):
        event = ChatStepEvent(step_name="comprehension", label="Compréhension", status=ChatStepStatus.STARTED)
        assert event.model_name is None
        assert event.input_tokens is None
        assert event.output_tokens is None
        assert event.duration_ms is None
        assert event.detail == ""

    def test_completed_event_with_tokens(self):
        event = ChatStepEvent(
            step_name="composition",
            label="Composition de la réponse",
            status=ChatStepStatus.COMPLETED,
            model_name="gemma4:latest",
            input_tokens=120,
            output_tokens=45,
            duration_ms=980,
        )
        assert event.status is ChatStepStatus.COMPLETED
        assert event.input_tokens == 120
        assert event.output_tokens == 45

    def test_failed_event_carries_detail(self):
        event = ChatStepEvent(
            step_name="comprehension",
            label="Compréhension",
            status=ChatStepStatus.FAILED,
            detail="LLM indisponible ou réponse invalide",
        )
        assert event.status is ChatStepStatus.FAILED
        assert "LLM" in event.detail

    def test_negative_tokens_rejected(self):
        with pytest.raises(ValidationError):
            ChatStepEvent(step_name="x", label="x", status=ChatStepStatus.COMPLETED, input_tokens=-1)

    def test_negative_duration_rejected(self):
        with pytest.raises(ValidationError):
            ChatStepEvent(step_name="x", label="x", status=ChatStepStatus.COMPLETED, duration_ms=-1)

    def test_unknown_status_rejected(self):
        with pytest.raises(ValidationError):
            ChatStepEvent(step_name="x", label="x", status="RUNNING")

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            ChatStepEvent(step_name="x", label="x", status=ChatStepStatus.STARTED, unknown_field="oops")

    def test_empty_step_name_rejected(self):
        with pytest.raises(ValidationError):
            ChatStepEvent(step_name="", label="x", status=ChatStepStatus.STARTED)

    def test_empty_label_rejected(self):
        with pytest.raises(ValidationError):
            ChatStepEvent(step_name="x", label="", status=ChatStepStatus.STARTED)

    def test_round_trip_json(self):
        event = ChatStepEvent(
            step_name="outil_simulate",
            label="Simulation ciblée : autonomous_decision",
            status=ChatStepStatus.COMPLETED,
            duration_ms=500,
        )
        restored = ChatStepEvent.model_validate_json(event.model_dump_json())
        assert restored == event


class TestChatTurnSummary:
    def test_defaults(self):
        summary = ChatTurnSummary(reply="ok")
        assert summary.thread_id is None
        assert summary.total_input_tokens == 0
        assert summary.total_output_tokens == 0
        assert summary.total_duration_ms == 0
        assert summary.step_count == 0

    def test_full_summary(self):
        summary = ChatTurnSummary(
            reply="Voici la réponse.",
            thread_id="thread-1",
            total_input_tokens=800,
            total_output_tokens=120,
            total_duration_ms=13000,
            step_count=3,
        )
        assert summary.total_input_tokens == 800
        assert summary.step_count == 3

    def test_negative_totals_rejected(self):
        with pytest.raises(ValidationError):
            ChatTurnSummary(reply="ok", total_input_tokens=-1)

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            ChatTurnSummary(reply="ok", unknown_field=1)
