"""Schémas du Fraud Detection Agent — ClaimShield Santé.

Tous les modèles :
  - héritent de StrictModel (extra='forbid') — tout champ inconnu lève ValidationError.
  - sont JSON-sérialisables via model_dump(mode="json").
  - n'exposent jamais de donnée personnelle brute ni de secret.

``FraudDetectionResult``, ``FraudResultPayload``, ``FraudSignal``,
``FraudEvidence`` et ``FraudEvidenceSource`` sont définis dans
``schemas/results.py`` (source unique de vérité — consommée aussi par
``orchestrator.orchestrator.AGENT_RESULT_MODELS``) et simplement re-exportés
ici, jamais dupliqués (même patron que ``clinical_consistency_agent.schemas``).
``LlmFraudDecision`` n'a jamais le pouvoir de changer le score de risque, le
statut ou le besoin de revue déterministes calculés en Phase A : il ne
fournit qu'une justification en français, des références à des signaux déjà
calculés, une perception de risque et un signal de revue purement
indicatifs — jamais un verdict de fraude, jamais une accusation, jamais un
blocage définitif et jamais une décision sans revue humaine (voir
``agent.py::_merge_llm_decision`` et ``FraudDetectionResult.human_review_required``,
toujours dérivé du seul statut). ``_reject_accusatory_language`` interdit en
plus toute formulation de fraude avérée/confirmée/prouvée ou toute
qualification de la personne (« fraude confirmée », « coupable »,
« escroc », « fraudeur »...) dans ``rationale``/``reasons`` — un risque de
fraude reste toujours signalable, une fraude ne peut jamais être déclarée
établie par ce champ.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, field_validator, model_validator
from schemas.domain import StrictModel
from schemas.results import (  # re-export public
    FraudDetectionResult,
    FraudEvidence,
    FraudEvidenceSource,
    FraudResultPayload,
    FraudSignal,
)

_ABSOLUTE_PATH_RE = re.compile(r"^(?:/|[A-Za-z]:[/\\]|\\\\)")
_SECRET_HINT_RE = re.compile(
    r"(?:api[_-]?key|secret\s*[:=]|password\s*[:=]|token\s*[:=]|bearer\s+\S+)",
    re.IGNORECASE,
)

_ACCUSATORY_RE = re.compile(
    r"fraude(?:\s+\w+){0,3}?\s+(?:confirm\w*|av[ée]r[ée]e?\w*|certaine?s?|prouv[ée]e?s?|[ée]tabli\w*)"
    r"|(?:confirm\w*|prouv\w*|[ée]tabli\w*)(?:\s+\w+){0,3}?\s+fraude"
    r"|confirmed\s+fraud|fraud\s+confirmed|proven\s+fraud"
    r"|\bcoupable\b|\bescroc\w*\b|\bfraudeur\w*\b",
    re.IGNORECASE,
)
"""Formulations accusatoires interdites — ce module signale un risque,
jamais une conclusion de fraude avérée ni une qualification de la personne.
Tolère jusqu'à 3 mots entre « fraude » et le qualificatif de confirmation
(« la fraude est clairement confirmée ») pour couvrir les tournures
naturelles, sans bloquer une simple mention de risque (« risque de
fraude », « signal de fraude potentielle » restent autorisés — aucun mot de
confirmation n'y apparaît)."""

_NEGATION_WORDS: frozenset[str] = frozenset(
    {"non", "pas", "jamais", "aucune", "aucun", "ni", "not", "never", "no"}
)
"""Une négation explicite à l'intérieur de la formulation détectée (« fraude
non confirmée », « pas de fraude avérée ») exprime une incertitude légitime,
jamais une accusation — volontairement exclue du rejet."""


def _reject_llm_leak(value: str, field_name: str) -> str:
    if _ABSOLUTE_PATH_RE.match(value):
        raise ValueError(f"Chemin absolu interdit dans {field_name}")
    if _SECRET_HINT_RE.search(value):
        raise ValueError(f"Secret potentiel interdit dans {field_name}")
    return value


def _reject_accusatory_language(value: str, field_name: str) -> str:
    """Interdit toute formulation accusatoire (« fraude confirmée », etc.) —
    ce champ ne doit jamais dépasser un signalement de risque. Une négation
    explicite précédant ou incluse dans la formulation détectée (« fraude
    non confirmée », « aucune fraude avérée ») est tolérée : elle exprime
    une incertitude, jamais une accusation.
    """
    for match in _ACCUSATORY_RE.finditer(value):
        preceding_words = re.findall(r"\w+", value[: match.start()].casefold())
        span_words = re.findall(r"\w+", match.group(0).casefold())
        nearby_words = preceding_words[-2:] + span_words
        if any(word in _NEGATION_WORDS for word in nearby_words):
            continue
        raise ValueError(
            f"Formulation accusatoire interdite dans {field_name} : {match.group(0)!r} — "
            "signale un risque, n'accuse jamais."
        )
    return value


class SignalAssessment(StrictModel):
    """Ajustement borné de pondération LLM sur un signal déjà calculé.

    ``signal_type`` doit référencer un ``FraudSignal.signal_type`` réellement
    produit par la Phase A — jamais un signal inventé (aucun mécanisme ne
    permet de créer un nouveau signal ici, seulement de citer un type
    connu ; toute référence à un type inconnu est ignorée silencieusement
    par ``agent.py::_apply_signal_assessments``, même garantie
    anti-hallucination que ``referenced_signal_types``). ``rationale`` est
    obligatoire dès que ``severity_adjustment`` s'écarte de ``NEUTRAL`` — un
    ajustement de pondération sans justification n'est jamais accepté.
    """

    signal_type: str = Field(..., min_length=1, max_length=100)
    severity_adjustment: Literal["DOWNGRADE", "NEUTRAL", "UPGRADE"] = "NEUTRAL"
    rationale: str = Field(default="", max_length=300)

    @field_validator("signal_type")
    @classmethod
    def no_sensitive_signal_type(cls, v: str) -> str:
        return _reject_llm_leak(v, "signal_type")

    @field_validator("rationale")
    @classmethod
    def no_sensitive_rationale(cls, v: str) -> str:
        v = _reject_llm_leak(v, "rationale")
        return _reject_accusatory_language(v, "rationale")

    @model_validator(mode="after")
    def rationale_required_unless_neutral(self) -> "SignalAssessment":
        if self.severity_adjustment != "NEUTRAL" and not self.rationale.strip():
            raise ValueError(
                "rationale obligatoire dès que severity_adjustment n'est pas NEUTRAL "
                "— un ajustement de pondération sans justification n'est jamais accepté."
            )
        return self


class LlmFraudDecision(StrictModel):
    """Justification explicative LLM — jamais d'autorité directe sur le
    score de risque, le statut ou le besoin de revue.

    ``referenced_signal_types`` ne permet au LLM que de *citer* des signaux
    déjà calculés par la Phase A (jamais d'en inventer un nouveau — aucun
    champ libre de nature de signal n'existe ici) : l'agent revérifie chaque
    type contre les signaux réellement présents et ignore silencieusement
    toute référence inconnue. ``llm_risk_perception``/``suggests_human_review``
    sont purement indicatifs : ils n'écrasent jamais ``risk_score`` ni
    ``human_review_required``, tous deux dérivés exclusivement de la Phase A.

    ``signal_assessments`` (P1-1 — autonomie bornée) est la seule influence
    indirecte du LLM sur le score final : pour chaque signal déjà calculé et
    déjà attribué à une preuve, il peut proposer de le pondérer moins
    (DOWNGRADE), normalement (NEUTRAL) ou plus (UPGRADE) qu'un cas typique.
    Le calcul lui-même (multiplicateur fixe, seuils de statut) reste
    entièrement déterministe (``agent.py::_apply_signal_assessments`` /
    ``_determine_status``) — le LLM ne fixe jamais lui-même une valeur
    numérique de score ni un statut.
    """

    rationale: str = Field(default="", max_length=500)
    referenced_signal_types: list[str] = Field(default_factory=list, max_length=20)
    signal_assessments: list[SignalAssessment] = Field(default_factory=list, max_length=10)
    llm_risk_perception: float | None = Field(default=None, ge=0.0, le=1.0)
    suggests_human_review: bool = False
    reasons: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("rationale")
    @classmethod
    def no_sensitive_rationale(cls, v: str) -> str:
        v = _reject_llm_leak(v, "rationale")
        return _reject_accusatory_language(v, "rationale")

    @field_validator("reasons")
    @classmethod
    def no_sensitive_reasons(cls, v: list[str]) -> list[str]:
        checked = [_reject_llm_leak(str(item), "reasons") for item in v]
        return [_reject_accusatory_language(item, "reasons") for item in checked]

    @field_validator("referenced_signal_types")
    @classmethod
    def no_sensitive_signal_types(cls, v: list[str]) -> list[str]:
        return [_reject_llm_leak(str(item), "referenced_signal_types") for item in v]


__all__ = [
    "FraudDetectionResult",
    "FraudEvidence",
    "FraudEvidenceSource",
    "FraudResultPayload",
    "FraudSignal",
    "LlmFraudDecision",
    "SignalAssessment",
]
