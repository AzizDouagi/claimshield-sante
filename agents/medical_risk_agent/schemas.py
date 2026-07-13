"""Schéma de décision LLM de medical_risk_agent (V2, plan Phase V2-5).

Fusionne les trois schémas de décision LLM de V1 — `LlmCodingDecision`
(`medical_coding_agent`), `LlmClinicalDecision` (`clinical_consistency_agent`,
via `severity_assessments`) et `LlmFraudDecision` (`fraud_detection_agent`,
via `signal_assessments`) — en un seul schéma pour un seul appel ReAct.

Réutilise par import les sous-schémas déjà validés de V1 (`LlmResolvedCode`,
`ClinicalSignalAssessment`, `SignalAssessment`) — jamais dupliqués. Mêmes
garanties d'autonomie bornée que chaque agent V1 d'origine : le LLM ne fixe
jamais lui-même un statut, un score ou une valeur — uniquement des
ajustements bornés sur des signaux déjà calculés par la Phase A (voir
`agent.py` pour l'application effective de chaque borne, portée telle
quelle depuis les fonctions V1 correspondantes).
"""
from __future__ import annotations

import re

from pydantic import Field, field_validator

from agents.clinical_consistency_agent.schemas import ClinicalSignalAssessment
from agents.fraud_detection_agent.schemas import SignalAssessment
from agents.medical_coding_agent.schemas import LlmResolvedCode
from schemas.domain import StrictModel

__all__ = ["LlmMedicalRiskDecision"]

# Dupliqué volontairement (jamais importé depuis schemas.v2_results, module
# privé non destiné à l'export cross-fichier) — même convention que
# schemas.v2_results._reject_unstructured_content lui-même.
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


class LlmMedicalRiskDecision(StrictModel):
    """Décision LLM combinée — codification + cohérence clinique + fraude,
    un seul appel ReAct (plan V2 Phase V2-5, garde-fou §5 : fallback
    officiel V2-5-bis si cette fusion dégrade la qualité mesurée)."""

    coding_resolved: list[LlmResolvedCode] = Field(default_factory=list)
    coding_rationale: str = Field(default="", max_length=500)

    clinical_context: str = Field(default="", max_length=500)
    clinical_severity_assessments: list[ClinicalSignalAssessment] = Field(default_factory=list)
    clinical_referenced_evidence_ids: list[str] = Field(default_factory=list)
    clinical_acknowledged_inconsistencies: list[str] = Field(default_factory=list)

    fraud_rationale: str = Field(default="", max_length=500)
    fraud_signal_assessments: list[SignalAssessment] = Field(default_factory=list)
    fraud_referenced_signal_types: list[str] = Field(default_factory=list)

    reasons: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("coding_rationale", "clinical_context", "fraud_rationale")
    @classmethod
    def _no_raw_content_scalar(cls, v: str, info) -> str:
        return _reject_leak(v, info.field_name)

    @field_validator("reasons")
    @classmethod
    def _no_raw_content_list(cls, v: list[str]) -> list[str]:
        return [_reject_leak(item, "reasons") for item in v]
