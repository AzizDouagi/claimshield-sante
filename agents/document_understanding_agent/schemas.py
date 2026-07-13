"""Schéma de décision LLM de document_understanding_agent (V2, plan Phase V2-3).

Fusionne `agents/document_ocr_agent/schemas.py::LlmOcrDecision` +
`agents/fhir_validator_agent/schemas.py::LlmFhirDecision` en un seul appel
LLM. Le résultat structuré final est `schemas.v2_results.DocumentUnderstandingResult`
(réutilisé, jamais dupliqué) — ce module ne définit que le schéma de
décision LLM intermédiaire (jamais persisté dans `ClaimStateV2`).
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, field_validator

from schemas.domain import StrictModel

__all__ = ["LlmDocumentUnderstandingDecision"]

_ABSOLUTE_PATH_RE = re.compile(r"^(?:/|[A-Za-z]:[/\\]|\\\\)")
_SECRET_HINT_RE = re.compile(
    r"(?:api[_-]?key|secret\s*[:=]|password\s*[:=]|token\s*[:=]|bearer\s+[a-z0-9._~+/=-]+)",
    re.IGNORECASE,
)


def _reject_leak(v: str, field_name: str) -> str:
    if _ABSOLUTE_PATH_RE.match(v):
        raise ValueError(f"Chemin absolu interdit dans {field_name}")
    if _SECRET_HINT_RE.search(v):
        raise ValueError(f"Secret potentiel interdit dans {field_name}")
    return v


class LlmDocumentUnderstandingDecision(StrictModel):
    """Décision LLM combinée : aide à la classification OCR + interprétation
    FHIR. Ne porte aucune autorité sur l'extraction de champs elle-même
    (toujours déterministe, `tools.document_parser`) — uniquement sur le
    type de document proposé et le statut FHIR recommandé, tous deux
    contraints à n'escalader que vers plus de restriction (voir
    `agent.py::_merge_status`, jamais un adoucissement)."""

    document_type: str = Field(default="UNKNOWN", max_length=50)
    ocr_confidence_assessment: str = Field(default="", max_length=500)
    fhir_clinical_context: str = Field(default="", max_length=500)
    fhir_recommended_status: Literal["PASS", "NEEDS_REVIEW", "FAIL"] = "NEEDS_REVIEW"
    reasons: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("document_type", "ocr_confidence_assessment", "fhir_clinical_context")
    @classmethod
    def _no_sensitive_string(cls, v: str, info) -> str:
        return _reject_leak(v, info.field_name)

    @field_validator("reasons")
    @classmethod
    def _no_sensitive_reasons(cls, v: list[str]) -> list[str]:
        return [_reject_leak(item, "reasons") for item in v]
