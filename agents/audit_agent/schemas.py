"""Schémas internes de l'Audit Agent.

Le modèle LLM ne produit pas un événement persistant complet : il normalise
seulement les champs métier autorisés. Le chaînage, les hashes et l'écriture
append-only restent exclusivement délégués à ``services.audit_store``.
"""
from __future__ import annotations

import re

from pydantic import Field, field_validator

from schemas.audit import AuditEventType, RedactionStatus
from schemas.domain import DataClassification, StrictModel

_ABSOLUTE_PATH_RE = re.compile(r"^(?:/|[A-Za-z]:[/\\]|\\\\)")
_SECRET_HINT_RE = re.compile(
    r"(?:api[_-]?key|secret\s*[:=]|password\s*[:=]|token\s*[:=]|bearer\s+[a-z0-9._~+/=-]+)",
    re.IGNORECASE,
)


def _reject_sensitive_text(value: str | None, field_name: str) -> str | None:
    if value is None:
        return value
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} : valeur vide interdite")
    if _ABSOLUTE_PATH_RE.match(value) or ".." in value:
        raise ValueError(f"{field_name} : chemin absolu ou traversée interdits")
    if _SECRET_HINT_RE.search(value):
        raise ValueError(f"{field_name} : secret potentiel interdit")
    return cleaned


class LlmAuditNormalizedEvent(StrictModel):
    """Sortie structurée du LLM pour un événement d'audit normalisé."""

    event_type: AuditEventType
    actor: str = Field(..., min_length=1, max_length=120)
    outcome: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="Résultat normalisé, court, sans donnée brute ni secret.",
    )
    summary: str = Field(
        ...,
        min_length=1,
        max_length=600,
        description="Résumé métier minimisé de l'événement structuré.",
    )
    redaction_status: RedactionStatus
    classification: DataClassification = Field(
        ...,
        description="Classification de donnees normalisee pour l'evenement audite.",
    )
    anomalies: list[str] = Field(default_factory=list, max_length=10)
    redactions: list[str] = Field(default_factory=list, max_length=10)
    agent_name: str | None = Field(default=None, max_length=120)
    tool_calls: list[str] = Field(default_factory=list, max_length=20)
    evidence_ids: list[str] = Field(default_factory=list, max_length=50)
    reasons: list[str] = Field(default_factory=list, max_length=10)
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("actor", "outcome", "summary", "agent_name")
    @classmethod
    def no_sensitive_text(cls, v: str | None, info) -> str | None:
        return _reject_sensitive_text(v, info.field_name)

    @field_validator("tool_calls", "evidence_ids", "reasons", "anomalies", "redactions")
    @classmethod
    def no_sensitive_items(cls, values: list[str], info) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            cleaned_value = _reject_sensitive_text(value, info.field_name)
            if not cleaned_value:
                raise ValueError(f"{info.field_name} : élément vide interdit")
            cleaned.append(cleaned_value)
        return cleaned


__all__ = ["LlmAuditNormalizedEvent"]
