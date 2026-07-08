"""Schémas du Case Reviewer Agent — ClaimShield Santé.

Le Case Reviewer produit une pré-recommandation révisable par un humain à
partir d'une synthèse minimisée des résultats d'agents déjà validés. Le LLM
ne reçoit jamais de document brut ni de données patient, et il ne peut pas
désactiver la revue humaine obligatoire.

``LlmCaseReviewDecision`` ne porte jamais d'autorité sur ``status`` ou
``human_review_required`` (verrouillés au niveau de ``CaseReviewerResult``,
voir ``schemas/results.py``) : ``referenced_evidence_ids``/
``acknowledged_risks``/``acknowledged_disagreements`` ne permettent que de
*citer* des preuves, risques et contradictions déjà calculés par la Phase A
déterministe — jamais d'en inventer de nouveaux (revérifiés contre les
valeurs réelles par ``agent.py::_merge_llm_decision``, référence inconnue
silencieusement ignorée). ``_reject_prohibited_assertions`` interdit en plus,
dans tout champ texte libre, toute formulation de paiement/remboursement
autorisé, de diagnostic médical, d'accusation de fraude avérée ou de
validation finale du dossier — ce schéma ne documente qu'une synthèse
révisable, jamais une décision.
"""
from __future__ import annotations

import re

from pydantic import Field, field_validator, model_validator

from schemas.domain import Recommendation, StrictModel
from schemas.results import CaseReviewerResult, DisagreementPoint

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
"""Même patron que ``fraud_detection_agent`` — le Case Reviewer synthétise
aussi des résultats de fraude et ne doit jamais les transformer en
accusation avérée, même en les résumant."""

_DIAGNOSIS_RE = re.compile(
    r"\bdiagnosti\w*\b|\batteint\w*\s+de\b|\bsouffre\w*\s+de\b|\bporteur\w*\s+de\b",
    re.IGNORECASE,
)
"""Aucun diagnostic médical — ce n'est jamais le rôle du Case Reviewer
(voir ``clinical_consistency_agent``)."""

_PAYMENT_DECISION_RE = re.compile(
    r"(?:rembours\w*|paiement\w*|payer)(?:\s+\w+){0,3}?\s+(?:valid\w*|autoris\w*|approuv\w*|accord\w*)"
    r"|(?:valid\w*|autoris\w*|approuv\w*|accord\w*)(?:\s+\w+){0,3}?\s+(?:rembours\w*|paiement\w*)",
    re.IGNORECASE,
)
"""Aucune autorisation de paiement/remboursement — décision financière hors
du rôle de cet agent, toujours humaine."""

_FINAL_DECISION_RE = re.compile(
    r"d[ée]cision\s+finale|validation\s+finale"
    r"|valid\w*\s+d[ée]finitiv\w*|d[ée]finitiv\w*\s+valid\w*"
    r"|approuv\w*\s+d[ée]finitiv\w*|rejet[ée]\w*\s+d[ée]finitiv\w*"
    r"|clôtur\w*\s+d[ée]finitiv\w*|sans\s+revue\s+humaine",
    re.IGNORECASE,
)
"""Aucune validation finale — la pré-recommandation reste toujours révisable,
la revue humaine est toujours obligatoire."""

_NEGATION_WORDS: frozenset[str] = frozenset(
    {"non", "pas", "jamais", "aucune", "aucun", "ni", "not", "never", "no"}
)
"""Une négation explicite (« pas de diagnostic », « jamais une décision
finale ») exprime exactement la garde-fou attendu — jamais l'interdiction
elle-même — et reste donc toujours autorisée."""

_PROHIBITED_ASSERTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (_ACCUSATORY_RE, "Accusation de fraude"),
    (_DIAGNOSIS_RE, "Diagnostic médical"),
    (_PAYMENT_DECISION_RE, "Décision de paiement/remboursement"),
    (_FINAL_DECISION_RE, "Validation finale"),
)


def _reject_llm_leak(value: str, field_name: str) -> str:
    if _ABSOLUTE_PATH_RE.match(value):
        raise ValueError(f"Chemin absolu interdit dans {field_name}")
    if _SECRET_HINT_RE.search(value):
        raise ValueError(f"Secret potentiel interdit dans {field_name}")
    return value


def _reject_prohibited_assertions(value: str, field_name: str) -> str:
    """Interdit paiement/remboursement autorisé, diagnostic, accusation de
    fraude avérée et validation finale — ce champ ne doit jamais dépasser
    une synthèse révisable. Une négation explicite proche de la formulation
    détectée reste tolérée : elle exprime l'absence de la chose interdite,
    jamais l'interdiction elle-même.
    """
    for pattern, label in _PROHIBITED_ASSERTIONS:
        for match in pattern.finditer(value):
            preceding_words = re.findall(r"\w+", value[: match.start()].casefold())
            span_words = re.findall(r"\w+", match.group(0).casefold())
            nearby_words = preceding_words[-3:] + span_words
            if any(word in _NEGATION_WORDS for word in nearby_words):
                continue
            raise ValueError(
                f"{label} interdit(e) dans {field_name} : {match.group(0)!r} — "
                "case_reviewer_agent ne synthétise jamais qu'une pré-recommandation "
                "révisable, jamais cette affirmation."
            )
    return value


class LlmCaseReviewDecision(StrictModel):
    """Synthèse LLM structurée — pré-recommandation, jamais décision finale.

    ``referenced_evidence_ids``/``acknowledged_risks``/``acknowledged_disagreements``
    permettent au LLM de citer les preuves, risques et contradictions déjà
    calculés par la Phase A — jamais d'en inventer un nouveau (voir
    ``agent.py::_merge_llm_decision``). Aucun champ ici n'a d'autorité sur
    ``status``/``human_review_required``, verrouillés dans
    ``CaseReviewerResult``.

    ``confidence``/``escalation_required``/``escalation_reasons`` (P1-4)
    alimentent — en complément de critères Phase A tout aussi nécessaires —
    l'éligibilité à l'auto-approbation bornée
    (``CaseReviewerResultPayload.auto_decision``, calculée par
    ``agent.py::run()``, jamais par ce schéma lui-même) : ``confidence``
    doit atteindre ``Settings.claimshield_auto_approve_confidence_threshold``,
    et ``escalation_required`` doit être ``False``. Ces trois champs
    n'écrasent jamais ``status``/``human_review_required`` — l'auto-décision
    reste un signal additionnel non verrouillé, jamais une réouverture du
    verrou de schéma.
    """

    recommendation: Recommendation
    summary: str = Field(..., min_length=1, max_length=700)
    reasons: list[str] = Field(default_factory=list, min_length=1, max_length=10)
    referenced_evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    acknowledged_risks: list[str] = Field(default_factory=list, max_length=20)
    acknowledged_disagreements: list[str] = Field(default_factory=list, max_length=20)
    human_review_reasons: list[str] = Field(default_factory=list, max_length=10)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    escalation_required: bool = False
    escalation_reasons: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("summary")
    @classmethod
    def no_sensitive_summary(cls, v: str) -> str:
        v = _reject_llm_leak(v, "summary")
        return _reject_prohibited_assertions(v, "summary")

    @field_validator("reasons", "human_review_reasons", "acknowledged_risks", "escalation_reasons")
    @classmethod
    def no_sensitive_reasons(cls, v: list[str], info) -> list[str]:
        checked = [_reject_llm_leak(str(item), info.field_name) for item in v]
        return [_reject_prohibited_assertions(item, info.field_name) for item in checked]

    @field_validator("referenced_evidence_ids", "acknowledged_disagreements")
    @classmethod
    def no_sensitive_ids(cls, v: list[str], info) -> list[str]:
        return [_reject_llm_leak(str(item), info.field_name) for item in v]

    @model_validator(mode="after")
    def escalation_reasons_required_if_escalating(self) -> "LlmCaseReviewDecision":
        if self.escalation_required and not self.escalation_reasons:
            raise ValueError(
                "escalation_reasons obligatoire dès que escalation_required est True "
                "— une escalade sans motif n'est jamais acceptée."
            )
        return self


__all__ = [
    "CaseReviewerResult",
    "DisagreementPoint",
    "LlmCaseReviewDecision",
]
