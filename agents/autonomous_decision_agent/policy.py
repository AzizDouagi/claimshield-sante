"""Politiques déterministes centralisées de `autonomous_decision_agent` (V2).

Module unique regroupant toutes les règles pures et testables isolément
utilisées par la matrice de décision — plan de remédiation « autonomie
décisionnelle V2 » :

  - `classify_risk_signals` : reclassifie les signaux déjà calculés par les
    agents amont (jamais un nouveau calcul métier) par nature et solidité
    (`schemas.v2_results.RiskSignalCategory`) — remplace le plafonnement de
    la matrice de décision sur la seule valeur agrégée `risk_level`
    (`agents/medical_risk_agent/agent.py`), qui pouvait atteindre `HIGH` par
    une combinaison de signaux non confirmés (ex. `CEILING_EXCEEDED` +
    `NEAR_DUPLICATE_INVOICE`).
  - `evaluate_acceptance_requirements` : politique d'acceptation minimale —
    distingue conditions obligatoires confirmées, hypothèses tolérées et
    informations non essentielles. Une couverture `UNKNOWN`/`NEEDS_REVIEW`
    seule (sans aucun autre signal positif confirmé) ne suffit jamais à
    autoriser `APPROVE`.

Aucune fonction de ce module n'appelle de LLM, ne mute d'état, ni ne décide
seule d'une décision finale — elles produisent des verdicts structurés
consommés par `agents/autonomous_decision_agent/agent.py`.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, field_validator

from schemas.domain import SeverityLevel, StrictModel, VerificationStatus
from schemas.v2_results import ClassifiedRiskSignal, RiskSignalCategory, _reject_unstructured_content

__all__ = [
    "AcceptanceRequirementCheck",
    "AcceptanceRequirementsVerdict",
    "RequirementTier",
    "classify_risk_signals",
    "evaluate_acceptance_requirements",
]

# ── Classification des signaux de risque ──────────────────────────────────────

_FRAUD_SIGNAL_CATEGORY: dict[str, RiskSignalCategory] = {
    "EXACT_DUPLICATE_INVOICE": RiskSignalCategory.CONFIRMED_FRAUD_RISK,
    "NEAR_DUPLICATE_INVOICE": RiskSignalCategory.SUSPECTED_FRAUD_RISK,
    "IDENTITY_MISMATCH": RiskSignalCategory.ELIGIBILITY_FAILURE,
    "COVERAGE_INACTIVE_OR_EXPIRED": RiskSignalCategory.ELIGIBILITY_FAILURE,
    "IDENTITY_AMBIGUOUS": RiskSignalCategory.COMPLETENESS_GAP,
    "PREAUTHORIZATION_MISSING": RiskSignalCategory.COMPLETENESS_GAP,
    "UNRESOLVED_CODING": RiskSignalCategory.COMPLETENESS_GAP,
    "LOW_EXTRACTION_CONFIDENCE": RiskSignalCategory.CONFIDENCE_GAP,
}
"""Correspondance `FraudSignal.signal_type` -> catégorie — dérivée directement
des poids/conditions déjà calculés par `agents/fraud_detection_agent/agent.py`
(réutilisés tels quels par `medical_risk_agent`), jamais un recalcul. Exclut
volontairement `CEILING_EXCEEDED` : une couverture partielle confirmée n'est
jamais une « défaillance » — elle reste gérée par le mécanisme distinct
`confirmed_partial_coverage` de `agent.py`."""

_CONFIRMED_CATEGORIES: frozenset[RiskSignalCategory] = frozenset(
    {
        RiskSignalCategory.CONFIRMED_SECURITY_RISK,
        RiskSignalCategory.CONFIRMED_FRAUD_RISK,
        RiskSignalCategory.CONFIRMED_CLINICAL_RISK,
        RiskSignalCategory.ELIGIBILITY_FAILURE,
    }
)

_LOW_DOCUMENT_CONFIDENCE_THRESHOLD = 0.5


def _clinical_evidence_ids(signal: Any) -> list[str]:
    return [e.evidence_id for e in getattr(signal, "evidence", [])]


def classify_risk_signals(
    *,
    intake_safety_result: Any | None = None,
    medical_risk_result: Any | None = None,
    document_understanding_result: Any | None = None,
) -> list[ClassifiedRiskSignal]:
    """Reclassifie les signaux déjà calculés par les agents amont — jamais un
    nouveau calcul métier, jamais une décision. Voir le tableau du plan de
    remédiation pour la correspondance signal -> catégorie."""
    classified: list[ClassifiedRiskSignal] = []

    if intake_safety_result is not None:
        status = getattr(intake_safety_result.status, "value", intake_safety_result.status)
        if status in ("BLOCKED", "QUARANTINED"):
            classified.append(
                ClassifiedRiskSignal(
                    category=RiskSignalCategory.CONFIRMED_SECURITY_RISK,
                    signal_type=f"INTAKE_SAFETY_{status}",
                    source_agent="intake_safety_agent",
                    confirmed=True,
                    evidence_ids=[],
                    description=f"Statut d'admission confirmé : {status}.",
                )
            )

    payload = getattr(medical_risk_result, "result_payload", None) if medical_risk_result else None

    if payload is not None:
        for fraud_signal in payload.fraud_signals:
            category = _FRAUD_SIGNAL_CATEGORY.get(fraud_signal.signal_type)
            if category is None:
                continue
            classified.append(
                ClassifiedRiskSignal(
                    category=category,
                    signal_type=fraud_signal.signal_type,
                    source_agent="medical_risk_agent",
                    confirmed=category in _CONFIRMED_CATEGORIES,
                    evidence_ids=[e.evidence_id for e in fraud_signal.evidence],
                    description=fraud_signal.description,
                )
            )

        for clinical_signal in payload.clinical_signals:
            if clinical_signal.severity is SeverityLevel.CRITICAL:
                classified.append(
                    ClassifiedRiskSignal(
                        category=RiskSignalCategory.CONFIRMED_CLINICAL_RISK,
                        signal_type=clinical_signal.signal_type,
                        source_agent="medical_risk_agent",
                        confirmed=True,
                        evidence_ids=_clinical_evidence_ids(clinical_signal),
                        description=clinical_signal.description,
                    )
                )

        structural_absence = (
            not payload.codings
            and not (payload.procedure_count or 0)
            and not (payload.medication_count or 0)
        )
        if structural_absence:
            classified.append(
                ClassifiedRiskSignal(
                    category=RiskSignalCategory.COMPLETENESS_GAP,
                    signal_type="STRUCTURAL_ABSENCE",
                    source_agent="medical_risk_agent",
                    confirmed=False,
                    evidence_ids=[],
                    description="Aucun acte ou médicament soumis à la codification.",
                )
            )

    if document_understanding_result is not None:
        confidence = getattr(document_understanding_result, "confidence", None)
        if confidence is not None and confidence < _LOW_DOCUMENT_CONFIDENCE_THRESHOLD:
            classified.append(
                ClassifiedRiskSignal(
                    category=RiskSignalCategory.CONFIDENCE_GAP,
                    signal_type="LOW_DOCUMENT_UNDERSTANDING_CONFIDENCE",
                    source_agent="document_understanding_agent",
                    confirmed=False,
                    evidence_ids=[],
                    description=f"Confiance d'extraction documentaire faible ({confidence:.2f}).",
                )
            )

    return classified


def has_confirmed_category(
    classified_signals: list[ClassifiedRiskSignal], category: RiskSignalCategory
) -> bool:
    """`True` ssi un signal `confirmed=True` de cette catégorie est présent —
    jamais un signal suspecté/non confirmé, quelle que soit sa catégorie."""
    return any(s.category is category and s.confirmed for s in classified_signals)


# ── Politique d'acceptation minimale ──────────────────────────────────────────


class RequirementTier(str, Enum):
    """Distingue, pour la politique d'acceptation minimale, ce qui est
    strictement obligatoire de ce qui est une hypothèse tolérée ou une
    information non essentielle — jamais mélangés dans un même calcul."""

    REQUIRED_CONFIRMED = "REQUIRED_CONFIRMED"
    PERMITTED_ASSUMPTION = "PERMITTED_ASSUMPTION"
    NON_ESSENTIAL = "NON_ESSENTIAL"


class AcceptanceRequirementCheck(StrictModel):
    code: str = Field(..., min_length=1)
    tier: RequirementTier
    satisfied: bool
    description: str = Field(..., min_length=1)

    @field_validator("description")
    @classmethod
    def _description_no_raw_content(cls, v: str) -> str:
        return _reject_unstructured_content(v, "description")


class AcceptanceRequirementsVerdict(StrictModel):
    """Verdict de la politique d'acceptation minimale — jamais une décision
    finale, uniquement une entrée du tie-break
    (`choose_accept_or_reject_from_available_evidence`)."""

    minimum_requirements_satisfied: bool
    partial_requirements_satisfied: bool
    checks: list[AcceptanceRequirementCheck] = Field(default_factory=list)


def evaluate_acceptance_requirements(
    *,
    identity_status: VerificationStatus | None,
    document_status: VerificationStatus | None,
    has_confirmed_dangerous_clinical_signal: bool,
    has_confirmed_coverage_exclusion: bool,
    identity_is_pass: bool,
    coverage_is_pass: bool,
    has_resolved_medical_item: bool,
    ceiling_exceeded: bool,
) -> AcceptanceRequirementsVerdict:
    """Politique d'acceptation minimale centralisée et testable — traduit en
    code les « informations essentielles » (identité non incompatible,
    aucun danger clinique/d'exclusion confirmé, document lisible) et
    l'exigence explicite d'AZIZ : au moins un signal réellement confirmé
    favorable doit exister (`identity_is_pass` OU `coverage_is_pass` OU un
    élément médical résolu) — une couverture `UNKNOWN`/`NEEDS_REVIEW` seule,
    sans rien d'autre de confirmé, ne suffit jamais à `APPROVE`.
    """
    checks: list[AcceptanceRequirementCheck] = []

    identity_ok = identity_status is not VerificationStatus.FAIL
    checks.append(
        AcceptanceRequirementCheck(
            code="IDENTITY_NOT_CONFIRMED_MISMATCH",
            tier=RequirementTier.REQUIRED_CONFIRMED,
            satisfied=identity_ok,
            description="Identité non confirmée incompatible avec le bénéficiaire.",
        )
    )

    no_dangerous_clinical = not has_confirmed_dangerous_clinical_signal
    checks.append(
        AcceptanceRequirementCheck(
            code="NO_CONFIRMED_DANGEROUS_CLINICAL_SIGNAL",
            tier=RequirementTier.REQUIRED_CONFIRMED,
            satisfied=no_dangerous_clinical,
            description="Aucune incohérence clinique dangereuse confirmée.",
        )
    )

    no_coverage_exclusion = not has_confirmed_coverage_exclusion
    checks.append(
        AcceptanceRequirementCheck(
            code="NO_CONFIRMED_COVERAGE_EXCLUSION",
            tier=RequirementTier.REQUIRED_CONFIRMED,
            satisfied=no_coverage_exclusion,
            description="Aucune exclusion contractuelle confirmée.",
        )
    )

    document_ok = document_status is not VerificationStatus.FAIL
    checks.append(
        AcceptanceRequirementCheck(
            code="DOCUMENT_READABLE",
            tier=RequirementTier.REQUIRED_CONFIRMED,
            satisfied=document_ok,
            description="Document compréhensible (statut non FAIL).",
        )
    )

    has_positive_signal = identity_is_pass or coverage_is_pass or has_resolved_medical_item
    checks.append(
        AcceptanceRequirementCheck(
            code="HAS_ANY_POSITIVE_CONFIRMED_SIGNAL",
            tier=RequirementTier.REQUIRED_CONFIRMED,
            satisfied=has_positive_signal,
            description=(
                "Au moins un signal réellement confirmé favorable "
                "(identité confirmée, couverture confirmée, ou élément médical résolu)."
            ),
        )
    )

    minimum_requirements_satisfied = (
        identity_ok and no_dangerous_clinical and no_coverage_exclusion and document_ok and has_positive_signal
    )

    checks.append(
        AcceptanceRequirementCheck(
            code="CONFIRMED_PARTIAL_COVERAGE",
            tier=RequirementTier.PERMITTED_ASSUMPTION,
            satisfied=ceiling_exceeded,
            description="Couverture partielle confirmée (plafond dépassé).",
        )
    )
    partial_requirements_satisfied = (
        identity_ok and no_dangerous_clinical and no_coverage_exclusion and document_ok and ceiling_exceeded
    )

    return AcceptanceRequirementsVerdict(
        minimum_requirements_satisfied=minimum_requirements_satisfied,
        partial_requirements_satisfied=partial_requirements_satisfied,
        checks=checks,
    )
