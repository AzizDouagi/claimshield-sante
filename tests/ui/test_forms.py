"""Tests unitaires purs sur ``ui/forms.py`` — aucun import ``chainlit``,
aucune dépendance au runtime UI (module vérifié séparément, manuellement,
via ``chainlit run ui/app.py``)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from human_review.models import ReviewAction
from ui.forms import form_from_pending_review, required_fields_for_action


class TestFormFromPendingReview:
    def test_builds_form_from_minimal_payload(self):
        form = form_from_pending_review(
            {"case_id": "CLM-0001", "summary": ["motif 1"], "evidence": {"a": "b"}, "options": ["APPROVE", "REJECT"]}
        )

        assert form.case_id == "CLM-0001"
        assert form.summary == ["motif 1"]
        assert form.evidence == {"a": "b"}
        assert form.actions == (ReviewAction.APPROVE, ReviewAction.REJECT)

    def test_missing_optional_fields_default_empty(self):
        form = form_from_pending_review({"case_id": "CLM-0002", "options": []})

        assert form.summary == []
        assert form.evidence == {}
        assert form.actions == ()

    def test_missing_case_id_raises(self):
        with pytest.raises(KeyError):
            form_from_pending_review({"options": []})

    def test_unknown_action_value_raises(self):
        with pytest.raises(ValueError):
            form_from_pending_review({"case_id": "CLM-0003", "options": ["NOT_AN_ACTION"]})


class TestRequiredFieldsForAction:
    def test_justification_always_required(self):
        form = form_from_pending_review({"case_id": "CLM-0004", "options": ["APPROVE"]})

        fields = required_fields_for_action(form, ReviewAction.APPROVE)

        names = [f.name for f in fields]
        assert "justification" in names
        assert "target_node" not in names

    def test_target_node_only_for_retry(self):
        form = form_from_pending_review({"case_id": "CLM-0005", "options": ["RETRY"]})

        fields = required_fields_for_action(form, ReviewAction.RETRY)

        names = [f.name for f in fields]
        assert "justification" in names
        assert "target_node" in names

    def test_justification_required_lock_still_enforced(self):
        """Non-régression : ``HumanReviewFormView`` verrouille
        ``justification_required`` à True — ``ui/forms.py`` ne doit jamais
        pouvoir contourner ce verrou."""
        with pytest.raises(ValidationError):
            from human_review.views import HumanReviewFormView

            HumanReviewFormView(case_id="CLM-0006", justification_required=False)
