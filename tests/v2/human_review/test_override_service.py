"""Tests de human_review/override_models.py + override_service.py (V2) — Phase V2-8."""
from __future__ import annotations

import pytest

from human_review.override_service import (
    OverrideValidationError,
    validate_and_record_override,
    validate_override_request,
)
from services.override_store import OverrideAction, OverrideStore


def _raw(**overrides) -> dict:
    defaults = {
        "case_id": "CLM-8001",
        "actor": "gestionnaire.dupont",
        "action": "OVERRIDE_APPROVE",
        "justification": "Document complémentaire reçu hors pipeline, dossier réexaminé manuellement.",
    }
    defaults.update(overrides)
    return defaults


class TestValidateOverrideRequest:
    def test_valid_request_accepted(self):
        request = validate_override_request(_raw())
        assert request.case_id == "CLM-8001"
        assert request.action is OverrideAction.OVERRIDE_APPROVE

    def test_non_mapping_rejected(self):
        with pytest.raises(OverrideValidationError) as exc_info:
            validate_override_request("not a mapping")
        assert exc_info.value.errors[0].code == "OVERRIDE_REQUEST_UNSTRUCTURED"

    def test_missing_justification_rejected(self):
        with pytest.raises(OverrideValidationError) as exc_info:
            validate_override_request(_raw(justification=""))
        assert exc_info.value.errors[0].code == "OVERRIDE_REQUEST_INVALID"

    def test_unknown_action_rejected(self):
        with pytest.raises(OverrideValidationError):
            validate_override_request(_raw(action="AUTO_APPROVE_EVERYTHING"))

    def test_invalid_case_id_rejected(self):
        with pytest.raises(OverrideValidationError):
            validate_override_request(_raw(case_id="not-a-case-id"))

    def test_extra_field_rejected(self):
        with pytest.raises(OverrideValidationError):
            validate_override_request(_raw(unknown_field="x"))

    def test_validation_error_never_leaks_raw_justification_text(self):
        secret_text = "api_key: sk-should-never-leak-in-error-message"
        try:
            validate_override_request(_raw(action="BOGUS", justification=secret_text))
        except OverrideValidationError as exc:
            for error in exc.errors:
                assert secret_text not in error.message
                assert secret_text not in str(exc)

    @pytest.mark.parametrize("action", list(OverrideAction))
    def test_all_four_actions_accepted(self, action):
        request = validate_override_request(_raw(action=action.value))
        assert request.action is action


class TestValidateAndRecordOverride:
    def test_record_persisted_in_store(self):
        store = OverrideStore()
        record = validate_and_record_override(_raw(), store=store, original_decision="REJECT")
        assert record.case_id == "CLM-8001"
        assert record.original_decision == "REJECT"
        stored = store.read_by_case_id("CLM-8001")
        assert stored == (record,)

    def test_invalid_request_never_persisted(self):
        store = OverrideStore()
        with pytest.raises(OverrideValidationError):
            validate_and_record_override(_raw(justification=""), store=store)
        assert store.read_by_case_id("CLM-8001") == ()

    def test_original_decision_never_overwrites_decision_result(self):
        """L'override reste purement informatif — il ne mute jamais
        decision_result, auquel ce service n'a même pas accès."""
        store = OverrideStore()
        record = validate_and_record_override(_raw(), store=store, original_decision="APPROVE")
        assert not hasattr(record, "decision_result")
        assert record.original_decision == "APPROVE"

    def test_reopen_action_only_records_intent_never_resumes_graph(self):
        """`REOPEN` ne fait ici que journaliser l'intention — ce service ne
        connaît ni le graphe ni LangGraph, aucune reprise n'est possible
        depuis ce point d'entrée (garde-fou structurel, pas seulement
        documentaire)."""
        import human_review.override_service as override_service_module

        store = OverrideStore()
        record = validate_and_record_override(
            _raw(action="REOPEN"), store=store, original_decision="QUARANTINE"
        )
        assert record.action is OverrideAction.REOPEN
        # Aucune référence à LangGraph/graph_v2 dans le module — vérifié
        # statiquement, pas seulement par convention.
        with open(override_service_module.__file__, encoding="utf-8") as f:
            content = f.read()
        assert "langgraph" not in content.lower()
        assert "compile_workflow" not in content

    def test_multiple_overrides_on_same_case_all_recorded(self):
        store = OverrideStore()
        validate_and_record_override(_raw(action="CONFIRM"), store=store)
        validate_and_record_override(
            _raw(action="OVERRIDE_REJECT", justification="Nouvelle information disqualifiante."),
            store=store,
        )
        stored = store.read_by_case_id("CLM-8001")
        assert len(stored) == 2
        assert [r.action for r in stored] == [OverrideAction.CONFIRM, OverrideAction.OVERRIDE_REJECT]
