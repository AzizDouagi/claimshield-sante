"""Tests de schemas/audit.py — AuditEventType, RedactionStatus, AuditEvent."""
from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from schemas.audit import (
    AUDIT_EVENT_TYPE_DESCRIPTIONS,
    AuditEvent,
    AuditEventType,
    RedactionStatus,
)

EXPECTED_EVENT_TYPES = {
    "claim_started",
    "agent_called",
    "tool_called",
    "error",
    "human_decision",
    "security_decision",
    "retry",
    "failure",
    "final_report",
    "anomaly",
}

_VALID_HASH = hashlib.sha256(b"event").hexdigest()


def _build_event(**overrides: object) -> AuditEvent:
    payload: dict = {
        "case_id": "CLM-0001",
        "event_type": AuditEventType.CLAIM_STARTED,
        "actor": "audit_agent",
        "outcome": "success",
        "event_hash": _VALID_HASH,
        "redaction_status": RedactionStatus.NOT_REDACTED,
    }
    payload.update(overrides)
    return AuditEvent(**payload)


# ── Complétude et documentation des event_type ───────────────────────────────


def test_event_type_couvre_les_10_categories_demandees():
    values = {event_type.value for event_type in AuditEventType}
    assert values == EXPECTED_EVENT_TYPES


def test_chaque_event_type_a_une_description_non_vide():
    for event_type in AuditEventType:
        assert AUDIT_EVENT_TYPE_DESCRIPTIONS[event_type]


def test_descriptions_ne_couvrent_aucun_type_orphelin():
    # Le dict de descriptions ne doit ni omettre ni dépasser l'énumération.
    assert set(AUDIT_EVENT_TYPE_DESCRIPTIONS.keys()) == set(AuditEventType)


@pytest.mark.parametrize("event_type", list(AuditEventType))
def test_chaque_event_type_valide_est_accepte(event_type: AuditEventType):
    event = _build_event(event_type=event_type)
    assert event.event_type is event_type


# ── Rejet des event_type inconnus ────────────────────────────────────────────


@pytest.mark.parametrize(
    "unknown_value",
    [
        "not_a_real_event_type",
        "AGENT_CALLED",  # nom du membre, pas sa valeur — toujours refusé
        "claim_finished",
        "",
        "authorization",  # ancien vocabulaire spéculatif, jamais retenu ici
    ],
)
def test_event_type_inconnu_est_refuse(unknown_value: str):
    with pytest.raises(ValidationError):
        _build_event(event_type=unknown_value)


def test_event_type_inconnu_ne_leve_pas_autre_chose_qu_une_validation_error():
    # Un type inconnu doit être une erreur de validation Pydantic normale,
    # jamais une exception non gérée (KeyError, AttributeError...).
    try:
        _build_event(event_type="bogus")
    except ValidationError as exc:
        assert any(err["loc"] == ("event_type",) for err in exc.errors())
    else:
        pytest.fail("un event_type inconnu aurait dû être refusé")


# ── Sérialisation ─────────────────────────────────────────────────────────────


def test_event_type_serialise_en_simple_chaine():
    event = _build_event(event_type=AuditEventType.FINAL_REPORT)
    dumped = event.model_dump(mode="json")
    assert dumped["event_type"] == "final_report"
