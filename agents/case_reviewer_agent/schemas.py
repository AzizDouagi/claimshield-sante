"""Schémas du Case Reviewer Agent — ClaimShield Santé.

Le Case Reviewer produit une pré-recommandation révisable par un humain à
partir d'une synthèse minimisée des résultats d'agents déjà validés. Le LLM
ne reçoit jamais de document brut ni de données patient, et il ne peut pas
désactiver la revue humaine obligatoire.
"""
from __future__ import annotations

import re

from pydantic import Field, field_validator

from schemas.domain import Recommendation, StrictModel
from schemas.results import CaseReviewerResult, DisagreementPoint

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


class LlmCaseReviewDecision(StrictModel):
    """Synthèse LLM structurée — pré-recommandation, jamais décision finale."""

    recommendation: Recommendation
    summary: str = Field(..., min_length=1, max_length=700)
    reasons: list[str] = Field(default_factory=list, min_length=1, max_length=10)
    human_review_reasons: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("summary")
    @classmethod
    def no_sensitive_summary(cls, v: str) -> str:
        return _reject_llm_leak(v, "summary")

    @field_validator("reasons", "human_review_reasons")
    @classmethod
    def no_sensitive_reasons(cls, v: list[str]) -> list[str]:
        return [_reject_llm_leak(str(item), "reasons") for item in v]


__all__ = [
    "CaseReviewerResult",
    "DisagreementPoint",
    "LlmCaseReviewDecision",
]
