"""Schémas d'entrée du Medical Coding Agent — ClaimShield Santé."""
from __future__ import annotations

import re

from pydantic import Field, field_validator

from schemas.domain import StrictModel
from schemas.results import MedicalCodingResult, ProcedureCoding

_ABSOLUTE_PATH_RE = re.compile(r"^(?:/|[A-Za-z]:[/\\]|\\\\)")
_SECRET_HINT_RE = re.compile(
    r"(?:api[_-]?key|secret\s*[:=]|password\s*[:=]|token\s*[:=]|bearer\s+\S+)",
    re.IGNORECASE,
)


def _reject_llm_leak(value: str, field_name: str) -> str:
    if _ABSOLUTE_PATH_RE.match(value):
        raise ValueError(f"Chemin absolu interdit dans {field_name}")
    if _SECRET_HINT_RE.search(value):
        raise ValueError(f"Secret potentiel interdit dans {field_name}")
    return value


class MedicalCodingInput(StrictModel):
    """Données d'entrée pour la codification médicale."""

    case_id: str
    procedures: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)


# ── Schémas de sortie LLM (intermédiaires — jamais dans ClaimState) ───────────

class LlmResolvedCode(StrictModel):
    """Code résolu par le LLM pour une description NEEDS_REVIEW."""

    description: str = Field(..., max_length=300)
    proposed_code: str | None = Field(default=None, max_length=100)
    rationale: str = Field(default="", max_length=500)

    @field_validator("description", "proposed_code", "rationale")
    @classmethod
    def no_sensitive_value(cls, v: str | None, info) -> str | None:
        return _reject_llm_leak(v, info.field_name) if v is not None else v


class LlmCodingDecision(StrictModel):
    """Décision LLM pour la codification médicale."""

    resolved: list[LlmResolvedCode] = Field(default_factory=list)
    overall_rationale: str = Field(default="", max_length=500)

    @field_validator("overall_rationale")
    @classmethod
    def no_sensitive_value(cls, v: str) -> str:
        return _reject_llm_leak(v, "overall_rationale")


__all__ = [
    "LlmCodingDecision",
    "LlmResolvedCode",
    "MedicalCodingInput",
    "MedicalCodingResult",
    "ProcedureCoding",
]
