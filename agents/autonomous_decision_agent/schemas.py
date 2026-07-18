"""Schéma de décision LLM de autonomous_decision_agent (V2).

Plan de remédiation « autonomie décisionnelle V2 », phase 5 — restructure
`LlmAutonomousDecision` d'une simple sélection binaire (`decision`) vers un
**rôle d'analyse structurée** : le LLM analyse les relations entre les
preuves déjà calculées, identifie les conflits non résolus, propose une
recommandation ET une alternative bornée avec ses conditions, mais ne fixe
jamais lui-même une valeur absolue de confiance (seulement un ajustement
borné `[-0.3, 0.3]` appliqué à une confiance de base calculée en Python).

Autorité réelle mais bornée, inchangée dans son principe (V2-6) :
`recommended_decision` n'a d'effet que si elle figure dans l'ensemble
autorisé calculé en Python par `agent.py::_allowed_decisions` — les bornes
elles-mêmes ne sont jamais laissées au LLM ; une valeur hors bornes est
toujours ignorée au profit d'un repli déterministe fondé sur les preuves
disponibles (voir `agent.py::_merge_llm_analysis`/
`choose_accept_or_reject_from_available_evidence`). `supporting_factor_ids`/
`adverse_factor_ids` ne sont acceptés que s'ils référencent des
`DecisionFactor.code` réellement calculés par la Phase A — toute référence
inconnue est silencieusement ignorée et signalée, jamais acceptée comme
preuve. `alternative_decision`/`alternative_conditions` alimentent
`AutonomousDecisionResult.counterfactuals`, jamais un contournement de la
décision finale elle-même.

Conserve, de V1/V2-6, l'interdiction de diagnostic médical et d'accusation
de fraude avérée (garde-fous universels, toujours pertinents). Abandonne
volontairement les interdictions V1 de « décision de paiement » et de
« validation finale » — contradictoires avec le rôle même de cet agent en
V2 (décider est précisément sa fonction, plus une synthèse révisable).
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, field_validator, model_validator

from schemas.domain import StrictModel

__all__ = ["LlmAutonomousDecision"]

_ABSOLUTE_PATH_RE = re.compile(r"^(?:/|[A-Za-z]:[/\\]|\\\\)")
_SECRET_HINT_RE = re.compile(
    r"(?:api[_-]?key|secret\s*[:=]|password\s*[:=]|token\s*[:=]|bearer\s+[a-z0-9._~+/=-]+)",
    re.IGNORECASE,
)

_ACCUSATORY_RE = re.compile(
    r"fraude(?:\s+\w+){0,3}?\s+(?:confirm\w*|av[ée]r[ée]e?\w*|certaine?s?|prouv[ée]e?s?|[ée]tabli\w*)"
    r"|(?:confirm\w*|prouv\w*|[ée]tabli\w*)(?:\s+\w+){0,3}?\s+fraude"
    r"|confirmed\s+fraud|fraud\s+confirmed|proven\s+fraud"
    r"|\bcoupable\b|\bescroc\w*\b|\bfraudeur\w*\b",
    re.IGNORECASE,
)
_DIAGNOSIS_RE = re.compile(
    r"\bdiagnosti\w*\b|\batteint\w*\s+de\b|\bsouffre\w*\s+de\b|\bporteur\w*\s+de\b",
    re.IGNORECASE,
)
_NEGATION_WORDS: frozenset[str] = frozenset(
    {"non", "pas", "jamais", "aucune", "aucun", "ni", "not", "never", "no"}
)
_PROHIBITED_ASSERTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (_ACCUSATORY_RE, "Accusation de fraude"),
    (_DIAGNOSIS_RE, "Diagnostic médical"),
)


def _reject_leak(value: str, field_name: str) -> str:
    if _ABSOLUTE_PATH_RE.match(value):
        raise ValueError(f"Chemin absolu interdit dans {field_name}")
    if _SECRET_HINT_RE.search(value):
        raise ValueError(f"Secret potentiel interdit dans {field_name}")
    return value


def _reject_prohibited_assertions(value: str, field_name: str) -> str:
    """Interdit diagnostic médical et accusation de fraude avérée — même
    tolérance de négation que V1 (`case_reviewer_agent`)."""
    for pattern, label in _PROHIBITED_ASSERTIONS:
        for match in pattern.finditer(value):
            preceding_words = re.findall(r"\w+", value[: match.start()].casefold())
            span_words = re.findall(r"\w+", match.group(0).casefold())
            nearby_words = preceding_words[-3:] + span_words
            if any(word in _NEGATION_WORDS for word in nearby_words):
                continue
            raise ValueError(
                f"{label} interdit(e) dans {field_name} : {match.group(0)!r}."
            )
    return value


class LlmAutonomousDecision(StrictModel):
    """Analyse structurée du LLM — `recommended_decision` n'a d'effet réel
    que si elle figure dans l'ensemble autorisé calculé par
    `agent.py::_allowed_decisions` pour ce dossier ; sinon
    `agent.py::_merge_llm_analysis` l'ignore et retombe sur un repli fondé
    sur les preuves disponibles, jamais sur la valeur hors bornes proposée.
    """

    recommended_decision: Literal[
        "APPROVE", "PARTIAL_APPROVE", "REJECT", "REQUEST_MORE_INFO", "QUARANTINE", "TECHNICAL_FAILURE"
    ]
    reasoning_summary: str = Field(..., min_length=1, max_length=1000)
    supporting_factor_ids: list[str] = Field(default_factory=list)
    adverse_factor_ids: list[str] = Field(default_factory=list)
    unresolved_conflicts: list[str] = Field(default_factory=list, max_length=10)
    assumptions: list[str] = Field(default_factory=list, max_length=10)
    alternative_decision: Literal["APPROVE", "PARTIAL_APPROVE", "REJECT", "QUARANTINE"] | None = None
    alternative_conditions: list[str] = Field(default_factory=list, max_length=5)
    confidence_adjustment: float = Field(default=0.0, ge=-0.3, le=0.3)
    escalation_required: bool = False
    escalation_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _escalation_reasons_required_if_escalation(self) -> "LlmAutonomousDecision":
        if self.escalation_required and not self.escalation_reasons:
            raise ValueError("escalation_reasons obligatoire dès que escalation_required=True")
        return self

    @field_validator("reasoning_summary")
    @classmethod
    def _reasoning_summary_checked(cls, v: str) -> str:
        _reject_leak(v, "reasoning_summary")
        return _reject_prohibited_assertions(v, "reasoning_summary")

    @field_validator("unresolved_conflicts", "assumptions", "alternative_conditions", "escalation_reasons")
    @classmethod
    def _list_checked(cls, v: list[str], info) -> list[str]:
        checked = [_reject_leak(item, info.field_name) for item in v]
        return [_reject_prohibited_assertions(item, info.field_name) for item in checked]

    @field_validator("supporting_factor_ids", "adverse_factor_ids")
    @classmethod
    def _ids_checked(cls, v: list[str], info) -> list[str]:
        return [_reject_leak(item, info.field_name) for item in v]
