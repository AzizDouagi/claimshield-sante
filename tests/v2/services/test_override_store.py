"""Tests de services/override_store.py (V2) — Phase V2-0."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from services.override_store import OverrideAction, OverrideRecord, OverrideStore


def _record(**overrides) -> OverrideRecord:
    defaults = {
        "case_id": "CLM-0001",
        "actor": "gestionnaire.dupont",
        "action": OverrideAction.OVERRIDE_APPROVE,
        "justification": (
            "Document complémentaire reçu hors pipeline, dossier réexaminé manuellement."
        ),
        "original_decision": "REJECT",
    }
    defaults.update(overrides)
    return OverrideRecord(**defaults)


class TestOverrideRecordSchema:
    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            OverrideRecord(
                case_id="CLM-0001",
                actor="a",
                action=OverrideAction.CONFIRM,
                justification="motif",
                unknown_field="x",
            )

    def test_justification_required(self):
        with pytest.raises(ValidationError):
            OverrideRecord(
                case_id="CLM-0001", actor="a", action=OverrideAction.CONFIRM, justification=""
            )

    def test_case_id_pattern_enforced(self):
        with pytest.raises(ValidationError):
            _record(case_id="not-a-case-id")

    def test_override_id_and_recorded_at_auto_generated(self):
        record = _record()
        assert record.override_id
        assert record.recorded_at is not None


class TestOverrideStoreAppendOnly:
    def test_record_and_read_back(self):
        store = OverrideStore()
        record = _record()
        store.record_override(record)
        stored = store.read_by_case_id("CLM-0001")
        assert stored == (record,)

    def test_unknown_case_returns_empty_tuple(self):
        store = OverrideStore()
        assert store.read_by_case_id("CLM-9999") == ()

    def test_multiple_overrides_preserve_order(self):
        store = OverrideStore()
        first = _record(
            action=OverrideAction.CONFIRM, justification="Confirmation initiale du dossier."
        )
        second = _record(
            action=OverrideAction.REOPEN, justification="Nouvelle pièce jointe à évaluer."
        )
        store.record_override(first)
        store.record_override(second)
        stored = store.read_by_case_id("CLM-0001")
        assert stored == (first, second)

    def test_len_counts_all_cases(self):
        store = OverrideStore()
        store.record_override(_record(case_id="CLM-0001"))
        store.record_override(_record(case_id="CLM-0002"))
        assert len(store) == 2

    def test_read_returns_defensive_copies(self):
        store = OverrideStore()
        store.record_override(_record())
        stored = store.read_by_case_id("CLM-0001")
        stored_again = store.read_by_case_id("CLM-0001")
        assert stored == stored_again
        assert stored[0] is not stored_again[0]

    def test_never_overwrites_original_decision(self):
        """L'override ne mute jamais decision_result — il n'y a même pas accès,
        seulement un champ informatif `original_decision`."""
        store = OverrideStore()
        record = _record(original_decision="REJECT")
        store.record_override(record)
        stored = store.read_by_case_id("CLM-0001")[0]
        assert stored.original_decision == "REJECT"
        assert not hasattr(stored, "decision_result")
