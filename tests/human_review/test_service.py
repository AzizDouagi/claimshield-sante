"""Tests du service HITL — payload humain minimisé et validation de décision.

Couvre ``human_review/service.py`` :
  - ``build_human_review_payload`` : résumé, preuves, options — jamais de
    document brut, de texte OCR complet, de secret ou de prompt complet.
  - ``validate_human_decision`` : validation stricte contre ``HumanDecision``,
    erreurs structurées (jamais l'exception Pydantic brute ni la valeur
    fautive) en cas de décision invalide.
  - ``build_human_decision_audit_event``/``validate_and_audit_human_decision`` :
    événement d'audit minimal (action, justification, auteur, horodatage,
    evidence_ids) pour chaque décision humaine (APPROVE/MODIFY/REJECT/RETRY),
    jamais de document brut/prompt complet/OCR complet.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from human_review.models import HumanDecision, ReviewAction
from human_review.service import (
    HumanDecisionValidationError,
    HumanReviewPayload,
    build_human_decision_audit_event,
    build_human_review_payload,
    validate_and_audit_human_decision,
    validate_human_decision,
)
from schemas.domain import DocumentType, ExtractionStatus, OcrSource, VerificationStatus
from schemas.results import AuditEvent, DocumentOcrResult


# ── build_human_review_payload ───────────────────────────────────────────────


def test_payload_on_empty_state_has_default_summary_and_no_evidence():
    payload = build_human_review_payload({"case_id": "CLM-0001"})
    assert isinstance(payload, HumanReviewPayload)
    assert payload.case_id == "CLM-0001"
    assert payload.evidence == {}
    assert payload.summary == ["Revue humaine requise — aucun motif spécifique enregistré."]


def test_payload_case_id_defaults_when_absent():
    payload = build_human_review_payload({})
    assert payload.case_id == "INCONNU"


def test_payload_summary_combines_alerts_then_errors_without_duplicates():
    state = {
        "case_id": "CLM-0001",
        "alerts": ["Motif A", "Motif B"],
        "errors": ["Motif B", "Motif C"],
    }
    payload = build_human_review_payload(state)
    assert payload.summary == ["Motif A", "Motif B", "Motif C"]


def test_payload_evidence_extracts_only_status_or_decision_markers():
    state = {
        "case_id": "CLM-0001",
        "security_result": SimpleNamespace(decision="ALLOW"),
        "ocr_result": SimpleNamespace(status="NEEDS_REVIEW"),
    }
    payload = build_human_review_payload(state)
    assert payload.evidence == {"security_result": "ALLOW", "ocr_result": "NEEDS_REVIEW"}


def test_payload_evidence_ignores_result_without_status_or_decision():
    state = {"case_id": "CLM-0001", "coding_result": SimpleNamespace(other_field="x")}
    payload = build_human_review_payload(state)
    assert payload.evidence == {}


def test_payload_options_always_lists_all_four_review_actions():
    payload = build_human_review_payload({"case_id": "CLM-0001"})
    assert set(payload.options) == set(ReviewAction)
    assert len(payload.options) == 4


def test_payload_never_leaks_full_ocr_text_or_raw_document():
    """Garantie centrale : un texte OCR complet ou une valeur brute présente
    sur un résultat d'agent ne doit jamais apparaître dans le payload —
    uniquement le statut, jamais ``full_text``/``extracted_fields``."""
    secret_text = "Numéro de sécurité sociale : 1 85 12 78 456 789 12 — secret=hunter2"
    ocr_result = DocumentOcrResult(
        claim_id="CLM-0001",
        file_path="incoming/CLM-0001/facture.pdf",
        sha256="a" * 64,
        mime_type="application/pdf",
        extraction_status=ExtractionStatus.SUCCESS,
        status=VerificationStatus.NEEDS_REVIEW,
        document_type=DocumentType.INVOICE,
        ocr_source=OcrSource.PDF_TEXT,
        full_text=secret_text,
    )
    state = {"case_id": "CLM-0001", "ocr_result": ocr_result}
    payload = build_human_review_payload(state)

    assert payload.evidence == {"ocr_result": "NEEDS_REVIEW"}
    dumped = payload.model_dump_json()
    assert secret_text not in dumped
    assert "hunter2" not in dumped


def test_payload_is_json_serializable():
    payload = build_human_review_payload({"case_id": "CLM-0001", "alerts": ["A"]})
    assert payload.model_dump_json()


def test_payload_forbids_unknown_fields():
    with pytest.raises(Exception):
        HumanReviewPayload(case_id="CLM-0001", unexpected=True)


# ── validate_human_decision ──────────────────────────────────────────────────


@pytest.mark.parametrize("action", ["APPROVE", "MODIFY", "REJECT"])
def test_validate_human_decision_accepts_valid_non_retry_actions(action):
    decision = validate_human_decision(
        {"case_id": "CLM-0001", "actor": "reviewer1", "action": action, "justification": "Motif."}
    )
    assert decision.action.value == action
    assert decision.target_node is None


def test_validate_human_decision_accepts_valid_retry_with_target_node():
    decision = validate_human_decision(
        {
            "case_id": "CLM-0001",
            "actor": "reviewer1",
            "action": "RETRY",
            "justification": "Pièce manquante.",
            "target_node": "document_ocr",
        }
    )
    assert decision.action is ReviewAction.RETRY
    assert decision.target_node == "document_ocr"


def test_validate_human_decision_rejects_retry_without_target_node():
    with pytest.raises(HumanDecisionValidationError) as exc_info:
        validate_human_decision(
            {"case_id": "CLM-0001", "actor": "reviewer1", "action": "RETRY", "justification": "Motif."}
        )
    assert exc_info.value.errors[0].code == "HUMAN_DECISION_INVALID"


def test_validate_human_decision_rejects_target_node_outside_retry():
    with pytest.raises(HumanDecisionValidationError):
        validate_human_decision(
            {
                "case_id": "CLM-0001",
                "actor": "reviewer1",
                "action": "APPROVE",
                "justification": "Motif.",
                "target_node": "document_ocr",
            }
        )


def test_validate_human_decision_rejects_missing_justification():
    with pytest.raises(HumanDecisionValidationError):
        validate_human_decision({"case_id": "CLM-0001", "actor": "reviewer1", "action": "APPROVE"})


def test_validate_human_decision_rejects_empty_justification():
    with pytest.raises(HumanDecisionValidationError):
        validate_human_decision(
            {"case_id": "CLM-0001", "actor": "reviewer1", "action": "APPROVE", "justification": ""}
        )


def test_validate_human_decision_rejects_unknown_action():
    with pytest.raises(HumanDecisionValidationError):
        validate_human_decision(
            {
                "case_id": "CLM-0001",
                "actor": "reviewer1",
                "action": "NEEDS_MORE_INFO",
                "justification": "Motif.",
            }
        )


def test_validate_human_decision_rejects_invalid_case_id_pattern():
    with pytest.raises(HumanDecisionValidationError):
        validate_human_decision(
            {"case_id": "ABC", "actor": "reviewer1", "action": "APPROVE", "justification": "Motif."}
        )


def test_validate_human_decision_rejects_unknown_field():
    with pytest.raises(HumanDecisionValidationError):
        validate_human_decision(
            {
                "case_id": "CLM-0001",
                "actor": "reviewer1",
                "action": "APPROVE",
                "justification": "Motif.",
                "unexpected": True,
            }
        )


@pytest.mark.parametrize("raw", ["texte libre", ["une", "liste"], 42, None])
def test_validate_human_decision_rejects_non_mapping(raw):
    with pytest.raises(HumanDecisionValidationError) as exc_info:
        validate_human_decision(raw)
    assert exc_info.value.errors[0].code == "HUMAN_DECISION_UNSTRUCTURED"


def test_validate_human_decision_error_never_leaks_raw_sensitive_value():
    """Une erreur de validation ne doit jamais reproduire la valeur brute
    fautive (potentiellement sensible) — uniquement les chemins de champs."""
    secret_comment = "password=hunter2 api_key=sk-abcdef1234567890"
    with pytest.raises(HumanDecisionValidationError) as exc_info:
        validate_human_decision(
            {
                "case_id": "CLM-0001",
                "actor": "reviewer1",
                "action": "INVALID_ACTION",
                "justification": "Motif.",
                "comment": secret_comment,
            }
        )
    message = str(exc_info.value)
    assert secret_comment not in message
    for error in exc_info.value.errors:
        assert secret_comment not in error.message


def test_validate_human_decision_error_is_structured_error_list():
    with pytest.raises(HumanDecisionValidationError) as exc_info:
        validate_human_decision({"case_id": "CLM-0001", "actor": "", "action": "APPROVE", "justification": "M."})
    for error in exc_info.value.errors:
        assert error.code
        assert error.message


# ── build_human_decision_audit_event / validate_and_audit_human_decision ────


def _decision(action: str, *, target_node: str | None = None) -> HumanDecision:
    fields = {
        "case_id": "CLM-0001",
        "actor": "reviewer@example.com",
        "action": action,
        "justification": "Motif de décision de test.",
    }
    if target_node is not None:
        fields["target_node"] = target_node
    return HumanDecision(**fields)


@pytest.mark.parametrize(
    ("action", "target_node"),
    [("APPROVE", None), ("MODIFY", None), ("REJECT", None), ("RETRY", "document_ocr")],
)
def test_audit_event_present_after_each_action(action, target_node):
    """Un événement d'audit minimal existe après APPROVE, MODIFY, REJECT et
    RETRY — jamais absent, quelle que soit l'action choisie."""
    decision = _decision(action, target_node=target_node)
    event = build_human_decision_audit_event(decision, evidence_ids=["EVID-abc0000001"])

    assert isinstance(event, AuditEvent)
    assert event.outcome == action
    assert event.details["justification"] == "Motif de décision de test."
    assert event.actor == "reviewer@example.com"
    assert event.timestamp == decision.decided_at
    assert event.details["evidence_ids"] == "EVID-abc0000001"


def test_audit_event_case_id_matches_decision():
    decision = _decision("APPROVE")
    event = build_human_decision_audit_event(decision)
    assert event.case_id == "CLM-0001"


def test_audit_event_target_node_present_only_for_retry():
    retry_event = build_human_decision_audit_event(_decision("RETRY", target_node="fhir_validator"))
    assert retry_event.details["target_node"] == "fhir_validator"

    approve_event = build_human_decision_audit_event(_decision("APPROVE"))
    assert approve_event.details["target_node"] == ""


def test_audit_event_evidence_ids_defaults_to_empty():
    event = build_human_decision_audit_event(_decision("APPROVE"))
    assert event.details["evidence_ids"] == ""


def test_audit_event_aggregates_multiple_evidence_ids():
    event = build_human_decision_audit_event(
        _decision("MODIFY"), evidence_ids=["EVID-aaa0000001", "EVID-bbb0000002"]
    )
    assert event.details["evidence_ids"] == "EVID-aaa0000001,EVID-bbb0000002"


def test_audit_event_never_stores_more_than_the_truncation_bound():
    """Garantie centrale : jamais de document brut/texte OCR complet/prompt
    complet stocké dans l'audit — une justification anormalement longue
    (simulant un contenu collé par erreur) est tronquée, jamais conservée
    intégralement."""
    huge_text = "x" * 999  # borne HumanDecision.justification.max_length
    decision = _decision("APPROVE")
    decision = decision.model_copy(update={"justification": huge_text})
    event = build_human_decision_audit_event(decision)
    assert len(event.details["justification"]) == 500
    assert event.details["justification"] == huge_text[:500]


def test_human_decision_justification_has_a_length_bound():
    """La borne existe déjà au niveau du schéma — pas seulement à la
    construction de l'audit — pour empêcher qu'un document entier soit
    jamais accepté comme justification."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        HumanDecision(
            case_id="CLM-0001",
            actor="r1",
            action="APPROVE",
            justification="x" * 1001,
        )


def test_validate_and_audit_human_decision_returns_both():
    decision, event = validate_and_audit_human_decision(
        {
            "case_id": "CLM-0001",
            "actor": "reviewer@example.com",
            "action": "REJECT",
            "justification": "Preuves insuffisantes.",
        },
        evidence_ids=["EVID-abc0000001"],
    )
    assert decision.action is decision.action.REJECT
    assert event.outcome == "REJECT"
    assert event.details["evidence_ids"] == "EVID-abc0000001"


def test_validate_and_audit_human_decision_raises_before_building_audit_on_invalid_input():
    """Une décision invalide ne produit jamais d'événement d'audit —
    l'audit ne peut porter que sur une décision réellement acceptée."""
    with pytest.raises(HumanDecisionValidationError):
        validate_and_audit_human_decision(
            {"case_id": "CLM-0001", "actor": "r1", "action": "RETRY", "justification": "Motif."}
        )


def test_audit_event_never_leaks_full_ocr_text_or_secret():
    """Même si un humain colle par erreur un contenu sensible dans la
    justification, l'audit ne le conserve que tronqué à 500 caractères —
    jamais un secret ou un document complet."""
    secret_text = "password=hunter2 " + "texte confidentiel " * 40
    decision = _decision("APPROVE").model_copy(update={"justification": secret_text[:1000]})
    event = build_human_decision_audit_event(decision)
    assert len(event.details["justification"]) <= 500
    assert event.details["justification"] != secret_text
