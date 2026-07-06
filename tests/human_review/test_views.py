"""Tests de la vue formulaire HITL — ``human_review/views.py``.

Couvre :
  - ``build_human_review_form`` : recommandation, preuves, alertes, risques,
    contradictions et champs modifiables, jamais de contenu brut.
  - Les quatre actions du formulaire (valider/modifier/refuser/relancer).
  - Le verrouillage structurel de ``justification_required`` (toujours True).
  - ``submit_human_review_decision`` : aucune soumission sans justification.
  - Les adaptateurs de présentation FastAPI/Chainlit (structures pures,
    aucune dépendance à FastAPI ni Chainlit).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from human_review.models import HumanDecision, ReviewAction
from human_review.service import HumanDecisionValidationError
from human_review.views import (
    FORM_FIELDS,
    JUSTIFICATION_FIELD,
    TARGET_NODE_FIELD,
    HumanReviewFormView,
    build_human_review_form,
    render_for_chainlit_actions,
    render_for_fastapi,
    submit_human_review_decision,
)
from schemas.domain import Recommendation
from schemas.results import (
    CaseReviewerResult,
    CaseReviewerResultPayload,
    DisagreementPoint,
    LlmMetadata,
)


def _review_result(**payload_overrides) -> CaseReviewerResult:
    fields = {
        "recommendation": Recommendation.PENDING,
        "human_review_reasons": ["Validation humaine obligatoire avant toute décision finale."],
    }
    fields.update(payload_overrides)
    return CaseReviewerResult(
        case_id="CLM-0001",
        llm_trace=LlmMetadata(model_name="test-llm", prompt_version="test"),
        result_payload=CaseReviewerResultPayload(**fields),
    )


# ── build_human_review_form ──────────────────────────────────────────────────


def test_form_on_empty_state_has_no_recommendation():
    form = build_human_review_form({"case_id": "CLM-0001"})
    assert isinstance(form, HumanReviewFormView)
    assert form.recommendation is None
    assert form.risks == []
    assert form.disagreements == []


def test_form_reads_recommendation_from_review_result():
    state = {"case_id": "CLM-0001", "review_result": _review_result(recommendation=Recommendation.REJECT)}
    form = build_human_review_form(state)
    assert form.recommendation is Recommendation.REJECT


def test_form_reads_risks_and_disagreements_from_review_result():
    disagreement = DisagreementPoint(agent="fhir_validator", field="status", expected="PASS", observed="FAIL")
    state = {
        "case_id": "CLM-0001",
        "review_result": _review_result(
            risks=["Score de risque de fraude élevé (0.80)."],
            disagreements=[disagreement],
        ),
    }
    form = build_human_review_form(state)
    assert form.risks == ["Score de risque de fraude élevé (0.80)."]
    assert form.disagreements == [disagreement]


def test_form_reads_alerts_directly_from_state():
    state = {"case_id": "CLM-0001", "alerts": ["Revue dossier requise"]}
    form = build_human_review_form(state)
    assert form.alerts == ["Revue dossier requise"]


def test_form_evidence_comes_from_minimized_service_payload():
    from types import SimpleNamespace

    state = {"case_id": "CLM-0001", "ocr_result": SimpleNamespace(status="NEEDS_REVIEW")}
    form = build_human_review_form(state)
    assert form.evidence == {"ocr_result": "NEEDS_REVIEW"}


def test_form_never_leaks_raw_document_or_ocr_text():
    from schemas.domain import DocumentType, ExtractionStatus, OcrSource, VerificationStatus
    from schemas.results import DocumentOcrResult

    secret_text = "password=hunter2 — texte OCR complet non destiné à l'humain brut"
    state = {
        "case_id": "CLM-0001",
        "ocr_result": DocumentOcrResult(
            claim_id="CLM-0001",
            file_path="incoming/CLM-0001/facture.pdf",
            sha256="a" * 64,
            mime_type="application/pdf",
            extraction_status=ExtractionStatus.SUCCESS,
            status=VerificationStatus.NEEDS_REVIEW,
            document_type=DocumentType.INVOICE,
            ocr_source=OcrSource.PDF_TEXT,
            full_text=secret_text,
        ),
    }
    form = build_human_review_form(state)
    dumped = form.model_dump_json()
    assert secret_text not in dumped
    assert "hunter2" not in dumped


def test_form_exposes_all_four_actions_with_french_labels():
    form = build_human_review_form({"case_id": "CLM-0001"})
    assert set(form.actions) == set(ReviewAction)


def test_form_exposes_editable_fields():
    form = build_human_review_form({"case_id": "CLM-0001"})
    assert form.fields == FORM_FIELDS
    names = {f.name for f in form.fields}
    assert names == {"justification", "target_node"}


def test_justification_field_applies_to_all_actions_and_is_required():
    assert JUSTIFICATION_FIELD.required is True
    assert set(JUSTIFICATION_FIELD.applies_to) == set(ReviewAction)


def test_target_node_field_applies_only_to_retry():
    assert TARGET_NODE_FIELD.applies_to == (ReviewAction.RETRY,)


def test_form_is_json_serializable():
    form = build_human_review_form({"case_id": "CLM-0001", "alerts": ["A"]})
    assert form.model_dump_json()


def test_form_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        HumanReviewFormView(case_id="CLM-0001", unexpected=True)


# ── Verrouillage justification_required ──────────────────────────────────────


def test_justification_required_defaults_to_true():
    form = build_human_review_form({"case_id": "CLM-0001"})
    assert form.justification_required is True


def test_justification_required_cannot_be_disabled():
    with pytest.raises(ValidationError):
        HumanReviewFormView(case_id="CLM-0001", justification_required=False)


# ── submit_human_review_decision ─────────────────────────────────────────────


def test_submit_rejects_missing_justification():
    with pytest.raises(HumanDecisionValidationError) as exc_info:
        submit_human_review_decision({"case_id": "CLM-0001", "actor": "reviewer1", "action": "APPROVE"})
    assert exc_info.value.errors[0].code == "HUMAN_DECISION_INVALID"


def test_submit_accepts_valid_decision_with_justification():
    decision = submit_human_review_decision(
        {"case_id": "CLM-0001", "actor": "reviewer1", "action": "MODIFY", "justification": "Motif détaillé."}
    )
    assert isinstance(decision, HumanDecision)
    assert decision.action is ReviewAction.MODIFY


def test_submit_retry_requires_target_node():
    with pytest.raises(HumanDecisionValidationError):
        submit_human_review_decision(
            {"case_id": "CLM-0001", "actor": "reviewer1", "action": "RETRY", "justification": "Motif."}
        )
    decision = submit_human_review_decision(
        {
            "case_id": "CLM-0001",
            "actor": "reviewer1",
            "action": "RETRY",
            "justification": "Motif.",
            "target_node": "document_ocr",
        }
    )
    assert decision.target_node == "document_ocr"


# ── Adaptateurs FastAPI / Chainlit ───────────────────────────────────────────


def test_render_for_fastapi_returns_plain_json_dict():
    form = build_human_review_form({"case_id": "CLM-0001"})
    rendered = render_for_fastapi(form)
    assert isinstance(rendered, dict)
    assert rendered["case_id"] == "CLM-0001"
    assert rendered["actions"] == [a.value for a in ReviewAction]


def test_render_for_chainlit_actions_matches_chainlit_action_shape():
    form = build_human_review_form({"case_id": "CLM-0001"})
    actions = render_for_chainlit_actions(form)
    assert len(actions) == 4
    for action in actions:
        assert set(action.keys()) == {"name", "value", "label"}
    labels = {a["label"] for a in actions}
    assert labels == {"Valider", "Modifier", "Refuser", "Relancer"}


def test_no_chainlit_or_fastapi_import_at_module_level():
    """Garantie de compatibilité sans dépendance dure : ce module ne doit
    jamais importer chainlit (absent de requirements.txt) ni fastapi."""
    import human_review.views as views_module

    assert "chainlit" not in views_module.__dict__
    assert "fastapi" not in views_module.__dict__
