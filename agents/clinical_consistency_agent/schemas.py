"""Schémas du Clinical Consistency Agent — ClaimShield Santé.

Tous les modèles :
  - héritent de StrictModel (extra='forbid') — tout champ inconnu lève ValidationError.
  - sont JSON-sérialisables via model_dump(mode="json").
  - n'exposent jamais de donnée personnelle brute ni de secret.

``ClinicalConsistencyResult``, ``ClinicalSignal``, ``ClinicalInconsistency``,
``ClinicalEvidence`` et ``ClinicalEvidenceSource`` sont définis dans
``schemas/results.py`` (source unique de vérité — consommée aussi par
``orchestrator.orchestrator.AGENT_RESULT_MODELS``) et simplement re-exportés
ici, jamais dupliqués (voir le module ``fhir_validator_agent.schemas`` pour
le même patron). ``LlmClinicalDecision`` n'a jamais le pouvoir de changer le
statut déterministe (PASS/NEEDS_REVIEW/FAIL), la confiance ou le besoin de
revue finaux (tous calculés en Phase A/C, voir ``agents/clinical_consistency_agent/agent.py``)
— il ne fournit qu'un contexte explicatif en français, des références à des
preuves et incohérences déjà calculées, une confiance perçue et un signal de
revue purement indicatifs. Aucune sortie libre non structurée n'est acceptée
ailleurs dans ce module : chaque signal et chaque incohérence doit référencer
des champs, des documents et des preuves structurées (voir
``schemas/results.py``) ; toute preuve ou incohérence citée par le LLM mais
absente des signaux réellement calculés est ignorée par l'agent (jamais une
affirmation non prouvée acceptée telle quelle — voir
``agent.py::_merge_llm_decision``).
"""
from __future__ import annotations

import re

from pydantic import Field, field_validator
from schemas.domain import StrictModel
from schemas.results import (  # re-export public
    ClinicalConsistencyResult,
    ClinicalEvidence,
    ClinicalEvidenceSource,
    ClinicalInconsistency,
    ClinicalResultPayload,
    ClinicalSignal,
)

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


class LlmClinicalDecision(StrictModel):
    """Contexte explicatif LLM — jamais d'autorité sur le statut, la
    confiance ou le besoin de revue finaux.

    ``referenced_evidence_ids``/``acknowledged_inconsistencies`` ne
    permettent au LLM que de *citer* des preuves/incohérences déjà calculées
    par la Phase A (jamais d'en inventer une nouvelle — aucun champ libre de
    description de document n'existe ici) : l'agent revérifie chaque
    identifiant contre les preuves réellement présentes et ignore silencieusement
    toute référence inconnue. ``llm_confidence``/``suggests_human_review``
    sont purement indicatifs : ils n'écrasent jamais ``confidence`` ni
    ``human_review_required`` du résultat final, tous deux dérivés
    exclusivement de la Phase A.
    """

    clinical_context: str = Field(default="", max_length=500)
    referenced_evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    acknowledged_inconsistencies: list[str] = Field(default_factory=list, max_length=20)
    llm_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    suggests_human_review: bool = False
    reasons: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("clinical_context")
    @classmethod
    def no_sensitive_context(cls, v: str) -> str:
        return _reject_llm_leak(v, "clinical_context")

    @field_validator("reasons", "acknowledged_inconsistencies")
    @classmethod
    def no_sensitive_reasons(cls, v: list[str]) -> list[str]:
        return [_reject_llm_leak(str(item), "reasons") for item in v]


__all__ = [
    "ClinicalConsistencyResult",
    "ClinicalEvidence",
    "ClinicalEvidenceSource",
    "ClinicalInconsistency",
    "ClinicalResultPayload",
    "ClinicalSignal",
    "LlmClinicalDecision",
]
