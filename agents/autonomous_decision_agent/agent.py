"""Autonomous Decision Agent (V2) — remplace `case_reviewer_agent` (V1).

Décision finale bornée, 6 issues possibles (`ClaimDecisionV2`), **sans**
verrou « toujours NEEDS_REVIEW/human_review_required=True » (décision AZIZ :
« override asynchrone optionnel » — voir `services/override_store.py`,
Phase V2-8, pour la correction humaine post-décision, hors de ce graphe).

Pipeline :
  Phase A — pré-décision déterministe bornée : `tools.consistency.detect_result_disagreements`
            (réutilisé tel quel) + `agents.autonomous_decision_agent.policy.classify_risk_signals`
            (classification des signaux déjà calculés par nature/solidité —
            plan de remédiation « autonomie décisionnelle V2 », phase 4,
            remplace le plafonnement sur la seule valeur agrégée
            `medical_risk.risk_level`) + `policy.evaluate_acceptance_requirements`
            (politique d'acceptation minimale centralisée). Calcule
            l'ensemble des décisions *autorisées* pour ce dossier
            (`_allowed_decisions`) — jamais la décision elle-même.
  Phase B — un LLM d'**analyse structurée** (plan de remédiation, phase 5 —
            remplace la simple sélection binaire de la phase 4) :
            `LlmAutonomousDecision.recommended_decision` n'a d'effet que si
            elle appartient à l'ensemble autorisé calculé en Phase A ; sinon
            `_merge_llm_analysis` retombe sur
            `choose_accept_or_reject_from_available_evidence` (repli
            déterministe fondé sur les preuves disponibles) — jamais la
            valeur hors bornes proposée. Le LLM cite des identifiants de
            facteurs déjà calculés (`supporting_factor_ids`/
            `adverse_factor_ids`, revalidés — toute référence inconnue
            silencieusement ignorée et signalée), identifie des conflits
            non résolus, propose une alternative bornée avec conditions
            (`alternative_decision`/`alternative_conditions` →
            `AutonomousDecisionResult.counterfactuals`), et n'ajuste la
            confiance que par un delta borné `[-0.3, 0.3]`
            (`confidence_adjustment`) appliqué à une confiance de base
            calculée en Python (`evidence_completeness`) — jamais une
            valeur absolue fixée par le LLM. **Panne LLM isolée ≠
            TECHNICAL_FAILURE** : si le LLM est indisponible/invalide et que
            les 4 résultats amont existent, la décision reste fondée sur les
            preuves disponibles (`choose_accept_or_reject_from_available_evidence`),
            confiance plafonnée, `errors=[LLM_UNAVAILABLE]` — `TECHNICAL_FAILURE`
            reste réservé exclusivement au gate structurel de la Phase A.
  Phase C — construction de `AutonomousDecisionResult`, explicabilité
            comprise (`missing_information`/`decisive_factors`/
            `supporting_factors`/`adverse_factors`/`counterfactuals`/
            `recommended_action`/`evidence_completeness`/
            `risk_signal_classification`, tous calculés/validés en Python).

Bornes non contournables (voir `_allowed_decisions`) :
  - Un résultat d'agent amont structurellement absent → TECHNICAL_FAILURE
    forcé, LLM jamais consulté (panne réelle empêchant toute analyse).
  - `intake_safety.status == TECHNICAL_FAILURE` → TECHNICAL_FAILURE forcé.
  - `intake_safety.status == BLOCKED` → REJECT forcé, LLM jamais consulté.
  - `intake_safety.status == QUARANTINED` → QUARANTINE forcé, LLM jamais consulté.
  - Signal clinique CRITIQUE confirmé (`RiskSignalCategory.CONFIRMED_CLINICAL_RISK`)
    → QUARANTINE forcé.
  - Signal de fraude confirmé (`CONFIRMED_FRAUD_RISK` — correspondance
    d'octets, jamais une similarité) → QUARANTINE forcé.
  - Défaillance d'éligibilité confirmée (`ELIGIBILITY_FAILURE` — identité
    incompatible ou couverture exclue, donnée disponible) → REJECT forcé.
  - Un signal `SUSPECTED_FRAUD_RISK`/`COMPLETENESS_GAP`/`CONFIDENCE_GAP` (non
    confirmé) ne force **jamais** seul QUARANTINE/REJECT — il n'influence
    que le score doux du repli déterministe (voir
    `choose_accept_or_reject_from_available_evidence`).
  - `PARTIAL_APPROVE` n'est proposable que si `medical_risk_result.codings`
    contient un mélange réel de PASS et de non-PASS, ou que la couverture
    partielle est confirmée (plafond dépassé) — jamais choisi par le LLM.
  - `TECHNICAL_FAILURE` n'est jamais une valeur choisissable par le LLM —
    réservé exclusivement à l'indisponibilité/invalidité de la Phase B ou à
    une panne structurelle amont.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.autonomous_decision_agent.policy import (
    AcceptanceRequirementsVerdict,
    classify_risk_signals,
    evaluate_acceptance_requirements,
    has_confirmed_category,
)
from agents.autonomous_decision_agent.prompt import load_autonomous_decision_prompt
from agents.autonomous_decision_agent.schemas import LlmAutonomousDecision
from llm.factory import get_llm
from llm.metadata import build_llm_metadata
from schemas.domain import ClaimDecisionV2, VerificationStatus
from schemas.results import DisagreementPoint, StructuredError
from schemas.v2_results import (
    AutonomousDecisionResult,
    ClassifiedRiskSignal,
    DecisionAssumption,
    DecisionCounterfactual,
    DecisionFactor,
    EvidenceCompleteness,
    MissingInformation,
    MissingInformationDimension,
    MissingInformationImportance,
    RiskSignalCategory,
)
from state.claim_state_v2 import ClaimStateV2, validate_state_update_v2
from tools.consistency import detect_result_disagreements

_AGENT_NAME = "autonomous_decision_agent"

_GENERIC_STATUS_FIELDS: tuple[str, ...] = (
    "document_understanding_result",
    "eligibility_result",
    "medical_risk_result",
)
"""Champs `ClaimStateV2` dont le schéma expose un `status: VerificationStatus`
de premier niveau — `intake_safety_result.status` est un `IntakeSafetyStatus`
distinct, exclu (même logique d'exclusion que V1 pour `intake_result`/
`security_result`, voir `tools.consistency.GENERIC_STATUS_FIELDS`)."""

_UPSTREAM_RESULT_FIELDS: tuple[str, ...] = (
    "intake_safety_result",
    "document_understanding_result",
    "eligibility_result",
    "medical_risk_result",
)

_SOFT_REJECT_WEIGHTS: dict[RiskSignalCategory, float] = {
    RiskSignalCategory.COMPLETENESS_GAP: 0.5,
    RiskSignalCategory.CONFIDENCE_GAP: 0.5,
    RiskSignalCategory.SUSPECTED_FRAUD_RISK: 0.75,
}
"""Poids doux contribuant à `reject_score` (jamais un poids dur, jamais un
forçage de branche) — plan de remédiation, point 1 d'AZIZ :
`SUSPECTED_FRAUD_RISK` (`NEAR_DUPLICATE_INVOICE`, similarité probabiliste)
pèse un peu plus qu'une simple donnée manquante, sans jamais devenir une
certitude."""

_DIMENSION_BY_SIGNAL_TYPE: dict[str, MissingInformationDimension] = {
    "UNRESOLVED_CODING": MissingInformationDimension.CODING,
    "STRUCTURAL_ABSENCE": MissingInformationDimension.MEDICAL,
    "IDENTITY_AMBIGUOUS": MissingInformationDimension.IDENTITY,
    "PREAUTHORIZATION_MISSING": MissingInformationDimension.COVERAGE,
    "LOW_EXTRACTION_CONFIDENCE": MissingInformationDimension.DOCUMENT,
    "LOW_DOCUMENT_UNDERSTANDING_CONFIDENCE": MissingInformationDimension.DOCUMENT,
    "NEAR_DUPLICATE_INVOICE": MissingInformationDimension.FRAUD,
}
"""Dimension affectée par chaque type de signal non confirmé — dérivée
directement du `signal_type` déjà produit par
`agents.autonomous_decision_agent.policy.classify_risk_signals`, jamais un
nouveau calcul métier. `MissingInformationDimension.MEDICAL` sert de repli
pour un `signal_type` non répertorié ici (ne devrait jamais arriver en
pratique)."""

_BASE_CONFIDENCE_BY_COMPLETENESS: dict[EvidenceCompleteness, float] = {
    EvidenceCompleteness.COMPLETE: 1.0,
    EvidenceCompleteness.PARTIAL: 0.7,
    EvidenceCompleteness.INSUFFICIENT: 0.4,
}
"""Confiance de base calculée en Python à partir de `evidence_completeness`
(même patron que `agents/medical_risk_agent/agent.py`) — le LLM ne fixe
jamais une confiance absolue, seulement un `confidence_adjustment` borné
`[-0.3, 0.3]` (`LlmAutonomousDecision`) appliqué en clamp par-dessus cette
base (plan de remédiation, phase 5)."""


def _value(value: object | None) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return str(raw)


def _upper(value: object | None) -> str | None:
    raw = _value(value)
    return raw.upper() if raw is not None else None


def _count_items(value: object | None) -> int:
    return len(value) if isinstance(value, (list, tuple, dict, set)) else 0


def _build_snapshot(state: ClaimStateV2) -> dict[str, dict[str, object]]:
    """Résumé sûr des 4 résultats agents V2 — jamais de contenu métier brut."""
    intake_safety = state.get("intake_safety_result")
    document_understanding = state.get("document_understanding_result")
    eligibility = state.get("eligibility_result")
    medical_risk = state.get("medical_risk_result")

    medical_risk_payload = getattr(medical_risk, "result_payload", None) if medical_risk else None

    return {
        "intake_safety": {
            "present": intake_safety is not None,
            "status": _upper(getattr(intake_safety, "status", None)),
        },
        "document_understanding": {
            "present": document_understanding is not None,
            "status": _upper(getattr(document_understanding, "status", None)),
            "confidence": getattr(document_understanding, "confidence", None),
        },
        "eligibility": {
            "present": eligibility is not None,
            "status": _upper(getattr(eligibility, "status", None)),
            "identity_status": _upper(
                getattr(getattr(eligibility, "identity", None), "status", None)
            ),
            "coverage_status": _upper(
                getattr(getattr(eligibility, "coverage", None), "status", None)
            ),
            "coverage_data_available": bool(getattr(eligibility, "coverage_data_available", True))
            if eligibility is not None
            else None,
            "ceiling_exceeded": bool(
                getattr(getattr(eligibility, "coverage", None), "ceiling_exceeded", False)
            ),
            "preauthorization_required": bool(
                getattr(getattr(eligibility, "coverage", None), "preauthorization_required", False)
            ),
        },
        "medical_risk": {
            "present": medical_risk is not None,
            "status": _upper(getattr(medical_risk, "status", None)),
            "risk_level": _upper(getattr(medical_risk_payload, "risk_level", None)),
            "risk_score": getattr(medical_risk_payload, "risk_score", None),
            "evidence_completeness": _upper(
                getattr(medical_risk_payload, "evidence_completeness", None)
            ),
            "duplicate_invoice": getattr(medical_risk_payload, "duplicate_invoice", None),
            "signal_count": _count_items(getattr(medical_risk_payload, "clinical_signals", None))
            + _count_items(getattr(medical_risk_payload, "fraud_signals", None)),
        },
    }


def _collect_risks(snapshot: dict[str, dict[str, object]]) -> list[str]:
    risks: list[str] = []
    eligibility = snapshot["eligibility"]
    if eligibility.get("ceiling_exceeded"):
        risks.append("Plafond de couverture dépassé.")
    if eligibility.get("preauthorization_required"):
        risks.append("Pré-autorisation requise — à confirmer.")

    medical_risk = snapshot["medical_risk"]
    risk_score = medical_risk.get("risk_score")
    if isinstance(risk_score, (int, float)) and risk_score >= 0.7:
        risks.append(f"Score de risque élevé ({risk_score:.2f}).")
    if medical_risk.get("duplicate_invoice") is True:
        risks.append("Facture potentiellement en doublon.")

    return risks


def _collect_evidence_ids(state: ClaimStateV2) -> list[str]:
    medical_risk = state.get("medical_risk_result")
    return list(getattr(medical_risk, "evidence_ids", None) or [])


def _disagreement_id(point: DisagreementPoint) -> str:
    return f"{point.agent}.{point.field}"


def _has_partial_approve_condition(state: ClaimStateV2) -> bool:
    """`PARTIAL_APPROVE` n'est proposable que si un mélange réel de codes
    PASS/non-PASS existe, ou qu'une couverture partielle est confirmée
    (plafond dépassé) — jamais un choix arbitraire du LLM."""
    medical_risk = state.get("medical_risk_result")
    payload = getattr(medical_risk, "result_payload", None) if medical_risk else None
    codings = getattr(payload, "codings", None) if payload is not None else None
    mixed_codings = False
    if codings:
        statuses = {c.status for c in codings}
        mixed_codings = VerificationStatus.PASS in statuses and len(statuses - {VerificationStatus.PASS}) > 0

    eligibility = state.get("eligibility_result")
    coverage = getattr(eligibility, "coverage", None) if eligibility is not None else None
    ceiling_exceeded = bool(getattr(coverage, "ceiling_exceeded", False))

    return mixed_codings or ceiling_exceeded


def _allowed_decisions(
    *,
    upstream_results_present: bool,
    intake_status: str | None,
    classified_signals: list[ClassifiedRiskSignal],
    has_partial_condition: bool,
    requirements: AcceptanceRequirementsVerdict,
) -> tuple[frozenset[ClaimDecisionV2], list[str]]:
    """Calcule, en Python pur, l'ensemble des décisions autorisées pour ce
    dossier — jamais laissé à l'appréciation du LLM, et jamais borné sur la
    seule valeur agrégée `risk_level` (plan de remédiation « autonomie
    décisionnelle V2 », phase 4 — corrige le point 1 d'AZIZ).

    Ordre de priorité strict, chaque branche court-circuite les suivantes :
      1. Résultat d'agent amont structurellement absent → TECHNICAL_FAILURE
         forcé (panne réelle empêchant toute analyse, Cas 11).
      2. `intake_safety.status == TECHNICAL_FAILURE` → TECHNICAL_FAILURE forcé.
      3. `intake_safety.status == BLOCKED` → REJECT forcé (défensif : le
         graphe court-circuite déjà avant d'atteindre cet agent sur ce cas).
      4. `intake_safety.status == QUARANTINED` → QUARANTINE forcé (idem).
      5. Signal clinique CRITIQUE confirmé (`CONFIRMED_CLINICAL_RISK`) →
         QUARANTINE forcé.
      6. Signal de fraude confirmé (`CONFIRMED_FRAUD_RISK` — correspondance
         d'octets, jamais une similarité) → QUARANTINE forcé.
      7. Défaillance d'éligibilité confirmée (`ELIGIBILITY_FAILURE` —
         identité incompatible ou couverture exclue, donnée disponible) →
         REJECT forcé.
      8. Par défaut : {REJECT} toujours ; + APPROVE seulement si
         `requirements.minimum_requirements_satisfied` (au moins un signal
         réellement confirmé favorable — identité/couverture/élément médical
         résolu, jamais une couverture `UNKNOWN`/`NEEDS_REVIEW` seule) ; +
         PARTIAL_APPROVE si une couverture partielle confirmée ou un mélange
         de codings l'autorise — `SUSPECTED_FRAUD_RISK`/`COMPLETENESS_GAP`/
         `CONFIDENCE_GAP` n'y forcent jamais rien, ils n'influencent que le
         score doux du repli déterministe
         (`choose_accept_or_reject_from_available_evidence`).
    """
    bounded_by: list[str] = []

    if not upstream_results_present:
        bounded_by.append(
            "Un résultat d'agent amont est structurellement absent — analyse impossible, "
            "TECHNICAL_FAILURE forcé, LLM non consulté."
        )
        return frozenset({ClaimDecisionV2.TECHNICAL_FAILURE}), bounded_by

    if intake_status == "TECHNICAL_FAILURE":
        bounded_by.append(
            "intake_safety.status == TECHNICAL_FAILURE → TECHNICAL_FAILURE forcé, LLM non consulté."
        )
        return frozenset({ClaimDecisionV2.TECHNICAL_FAILURE}), bounded_by
    if intake_status == "BLOCKED":
        bounded_by.append("intake_safety.status == BLOCKED → REJECT forcé, LLM non consulté.")
        return frozenset({ClaimDecisionV2.REJECT}), bounded_by
    if intake_status == "QUARANTINED":
        bounded_by.append("intake_safety.status == QUARANTINED → QUARANTINE forcé, LLM non consulté.")
        return frozenset({ClaimDecisionV2.QUARANTINE}), bounded_by

    if has_confirmed_category(classified_signals, RiskSignalCategory.CONFIRMED_CLINICAL_RISK):
        bounded_by.append(
            "Signal clinique CRITIQUE confirmé (incohérence dangereuse) → QUARANTINE forcé, "
            "LLM non consulté."
        )
        return frozenset({ClaimDecisionV2.QUARANTINE}), bounded_by

    if has_confirmed_category(classified_signals, RiskSignalCategory.CONFIRMED_FRAUD_RISK):
        bounded_by.append(
            "Signal de fraude confirmé (correspondance d'octets, jamais une similarité) → "
            "QUARANTINE forcé, LLM non consulté."
        )
        return frozenset({ClaimDecisionV2.QUARANTINE}), bounded_by

    if has_confirmed_category(classified_signals, RiskSignalCategory.ELIGIBILITY_FAILURE):
        bounded_by.append(
            "Défaillance d'éligibilité confirmée (identité incompatible ou couverture "
            "confirmée exclue) → REJECT forcé, LLM non consulté."
        )
        return frozenset({ClaimDecisionV2.REJECT}), bounded_by

    # `APPROVE` n'est jamais offert — même au LLM — sans au moins un signal
    # réellement confirmé favorable (`requirements.minimum_requirements_satisfied`,
    # voir `policy.evaluate_acceptance_requirements`) : corrige explicitement
    # le point 4 d'AZIZ (« une couverture UNKNOWN/NEEDS_REVIEW seule ne doit
    # jamais suffire à autoriser APPROVE ») — appliqué ici, à la construction
    # de `allowed`, jamais seulement dans le repli du tie-break, pour que la
    # contrainte tienne même si le LLM propose directement APPROVE.
    allowed = {ClaimDecisionV2.REJECT}
    if requirements.minimum_requirements_satisfied:
        allowed.add(ClaimDecisionV2.APPROVE)
    else:
        bounded_by.append(
            "Aucun signal réellement confirmé favorable (identité/couverture/élément "
            "médical résolu) — APPROVE non proposable, même au LLM."
        )
    if has_partial_condition:
        allowed.add(ClaimDecisionV2.PARTIAL_APPROVE)

    return frozenset(allowed), bounded_by


def _compute_soft_scores(
    *,
    classified_signals: list[ClassifiedRiskSignal],
    identity_is_pass: bool,
    coverage_is_pass: bool,
    has_resolved_medical_item: bool,
) -> tuple[float, float]:
    """Score doux (jamais un forçage de branche) — `accept_score` compte les
    signaux réellement confirmés favorables, `reject_score` pondère chaque
    signal non confirmé (`_SOFT_REJECT_WEIGHTS`)."""
    accept_score = (
        (1.0 if identity_is_pass else 0.0)
        + (1.0 if coverage_is_pass else 0.0)
        + (1.0 if has_resolved_medical_item else 0.0)
    )
    reject_score = sum(
        _SOFT_REJECT_WEIGHTS.get(signal.category, 0.0) for signal in classified_signals if not signal.confirmed
    )
    return accept_score, reject_score


def choose_accept_or_reject_from_available_evidence(
    *,
    allowed: frozenset[ClaimDecisionV2],
    classified_signals: list[ClassifiedRiskSignal],
    requirements: AcceptanceRequirementsVerdict,
    identity_is_pass: bool,
    coverage_is_pass: bool,
    has_resolved_medical_item: bool,
    confirmed_partial_coverage: bool,
) -> tuple[ClaimDecisionV2, str]:
    """Règle de départage déterministe — remplace l'ancien ordre de repli
    statique (`_FALLBACK_PRIORITY`). N'approuve jamais uniquement parce
    qu'aucun signal négatif n'a été trouvé, ne refuse jamais uniquement
    parce qu'il manque une information non essentielle. Restreinte à
    `allowed` (déjà filtré par `_allowed_decisions`) — en pratique toujours
    `{APPROVE, REJECT}` (+`PARTIAL_APPROVE`) puisque les branches
    dangereuses confirmées ont déjà court-circuité avant d'atteindre cette
    fonction ; `hard_reject_condition`/`has_confirmed_coverage_exclusion`
    (dans `requirements`) restent donc redondants ici, en défense en
    profondeur uniquement.

    Paliers, dans l'ordre :
      1. Condition de rejet dur déjà confirmée (redondant, voir ci-dessus).
      2. Couverture partielle confirmée → PARTIAL_APPROVE.
      3. Score majoritaire (`accept_score`/`reject_score`).
      4. Politique d'acceptation minimale satisfaite → APPROVE.
      5. Politique d'acceptation partielle satisfaite → PARTIAL_APPROVE.
      6. Dernier repli documenté (jamais le premier réflexe) → REJECT.
    """
    if has_confirmed_category(classified_signals, RiskSignalCategory.ELIGIBILITY_FAILURE):
        if ClaimDecisionV2.REJECT in allowed:
            return ClaimDecisionV2.REJECT, "hard_reject_condition"

    if confirmed_partial_coverage and ClaimDecisionV2.PARTIAL_APPROVE in allowed:
        return ClaimDecisionV2.PARTIAL_APPROVE, "confirmed_partial_coverage"

    accept_score, reject_score = _compute_soft_scores(
        classified_signals=classified_signals,
        identity_is_pass=identity_is_pass,
        coverage_is_pass=coverage_is_pass,
        has_resolved_medical_item=has_resolved_medical_item,
    )
    if accept_score > reject_score and ClaimDecisionV2.APPROVE in allowed:
        return ClaimDecisionV2.APPROVE, "accept_score_majority"
    if reject_score > accept_score and ClaimDecisionV2.REJECT in allowed:
        return ClaimDecisionV2.REJECT, "reject_score_majority"

    if requirements.minimum_requirements_satisfied and ClaimDecisionV2.APPROVE in allowed:
        return ClaimDecisionV2.APPROVE, "minimum_acceptance_requirements_satisfied"
    if requirements.partial_requirements_satisfied and ClaimDecisionV2.PARTIAL_APPROVE in allowed:
        return ClaimDecisionV2.PARTIAL_APPROVE, "partial_acceptance_requirements_satisfied"

    if ClaimDecisionV2.REJECT in allowed:
        return ClaimDecisionV2.REJECT, "no_requirement_satisfied"
    return next(iter(allowed)), "no_requirement_satisfied_restricted_set"


def _status_for_decision(decision: ClaimDecisionV2) -> VerificationStatus:
    if decision is ClaimDecisionV2.TECHNICAL_FAILURE:
        return VerificationStatus.FAIL
    if decision in (ClaimDecisionV2.APPROVE, ClaimDecisionV2.REJECT):
        return VerificationStatus.PASS
    return VerificationStatus.NEEDS_REVIEW


# ── Phase B : LLM ─────────────────────────────────────────────────────────────


def _invoke_llm_autonomous_decision(data: dict[str, Any]) -> LlmAutonomousDecision | None:
    try:
        prompt = load_autonomous_decision_prompt()
        llm = get_llm()
        structured = llm.with_structured_output(LlmAutonomousDecision, method="json_schema")
        result = structured.invoke(
            [
                SystemMessage(content=prompt.system_prompt),
                HumanMessage(content=json.dumps(data, ensure_ascii=False, default=str)),
            ]
        )
        if isinstance(result, LlmAutonomousDecision):
            return result
        if isinstance(result, dict):
            return LlmAutonomousDecision(**result)
        return None
    except Exception:
        return None


def _merge_llm_analysis(
    allowed: frozenset[ClaimDecisionV2],
    llm_decision: LlmAutonomousDecision | None,
    *,
    classified_signals: list[ClassifiedRiskSignal],
    requirements: AcceptanceRequirementsVerdict,
    identity_is_pass: bool,
    coverage_is_pass: bool,
    has_resolved_medical_item: bool,
    confirmed_partial_coverage: bool,
) -> tuple[ClaimDecisionV2, list[str]]:
    """N'accepte la recommandation LLM que si elle appartient à `allowed` —
    sinon repli déterministe fondé sur les preuves disponibles
    (`choose_accept_or_reject_from_available_evidence`), jamais la valeur
    hors bornes proposée. **Panne LLM isolée ≠ TECHNICAL_FAILURE** (plan de
    remédiation, phase 5) : si `llm_decision is None`, le même repli fondé
    sur les preuves est utilisé — `TECHNICAL_FAILURE` reste réservé
    exclusivement au gate structurel de `_allowed_decisions` (résultat amont
    absent). Conflit LLM vs règle dure impossible par construction :
    `allowed` est déjà filtré avant tout appel LLM."""
    if llm_decision is None:
        fallback, reason_code = choose_accept_or_reject_from_available_evidence(
            allowed=allowed,
            classified_signals=classified_signals,
            requirements=requirements,
            identity_is_pass=identity_is_pass,
            coverage_is_pass=coverage_is_pass,
            has_resolved_medical_item=has_resolved_medical_item,
            confirmed_partial_coverage=confirmed_partial_coverage,
        )
        return fallback, [
            f"LLM indisponible ou réponse invalide — décision fondée sur les preuves "
            f"disponibles, repli sur {fallback.value} ({reason_code})."
        ]

    try:
        proposed = ClaimDecisionV2(llm_decision.recommended_decision)
    except ValueError:
        proposed = None

    if proposed is not None and proposed in allowed:
        return proposed, []

    fallback, reason_code = choose_accept_or_reject_from_available_evidence(
        allowed=allowed,
        classified_signals=classified_signals,
        requirements=requirements,
        identity_is_pass=identity_is_pass,
        coverage_is_pass=coverage_is_pass,
        has_resolved_medical_item=has_resolved_medical_item,
        confirmed_partial_coverage=confirmed_partial_coverage,
    )
    return fallback, [
        f"Décision LLM {llm_decision.recommended_decision!r} hors des bornes autorisées "
        f"({sorted(d.value for d in allowed)}) pour ce dossier — repli sur {fallback.value} "
        f"({reason_code})."
    ]


def _validate_llm_factor_references(
    llm_decision: LlmAutonomousDecision, *, known_codes: set[str]
) -> list[str]:
    """Toute référence de `supporting_factor_ids`/`adverse_factor_ids` à un
    identifiant non réellement calculé par la Phase A est silencieusement
    ignorée — mais jamais sans trace : retourne une note explicite pour
    `justification`, même patron anti-hallucination que
    `clinical_consistency_agent`/`case_reviewer_agent`."""
    unknown = sorted(
        {i for i in llm_decision.supporting_factor_ids if i not in known_codes}
        | {i for i in llm_decision.adverse_factor_ids if i not in known_codes}
    )
    if not unknown:
        return []
    return [f"Références de facteurs ignorées (identifiants inconnus) : {unknown}."]


def _build_counterfactuals_from_llm(
    llm_decision: LlmAutonomousDecision, *, final_decision: ClaimDecisionV2
) -> list[DecisionCounterfactual]:
    """Construit des `DecisionCounterfactual` à partir de
    `alternative_decision`/`alternative_conditions` — jamais un nouveau
    calcul métier, une simple mise en forme validée de l'analyse du LLM.
    `alternative_decision` reste borné par le schéma (`Literal` à 4 valeurs,
    jamais `TECHNICAL_FAILURE`/`REQUEST_MORE_INFO`) — non restreint à
    `allowed` : un contrefactuel décrit par nature une situation *différente*
    de l'état actuel, potentiellement hors de l'ensemble borné présent."""
    if llm_decision.alternative_decision is None or not llm_decision.alternative_conditions:
        return []
    try:
        alternative = ClaimDecisionV2(llm_decision.alternative_decision)
    except ValueError:
        return []
    if alternative is final_decision:
        return []
    return [
        DecisionCounterfactual(
            condition=condition,
            current_value="Condition non remplie en l'état actuel du dossier.",
            required_value=condition,
            resulting_decision=alternative,
            explanation=llm_decision.reasoning_summary,
        )
        for condition in llm_decision.alternative_conditions
    ]


def _collect_missing_information_and_factors(
    *,
    classified_signals: list[ClassifiedRiskSignal],
    identity_is_pass: bool,
    coverage_is_pass: bool,
    has_resolved_medical_item: bool,
) -> tuple[list[MissingInformation], list[DecisionFactor], list[DecisionFactor]]:
    """Première moitié de l'explicabilité — calculée *avant* l'appel LLM
    (Phase B) afin que `supporting_factors`/`adverse_factors` puissent être
    transmis comme contexte citable (`supporting_factor_ids`/
    `adverse_factor_ids`, voir `LlmAutonomousDecision`). Tous les champs
    calculés en Python à partir de faits déjà établis, jamais par le LLM.
    Retourne (missing_information, supporting_factors, adverse_factors)."""
    missing_information: list[MissingInformation] = []
    for signal in classified_signals:
        if signal.confirmed:
            continue
        dimension = _DIMENSION_BY_SIGNAL_TYPE.get(signal.signal_type, MissingInformationDimension.MEDICAL)
        missing_information.append(
            MissingInformation(
                code=signal.signal_type,
                description=signal.description,
                importance=MissingInformationImportance.IMPORTANT,
                affected_dimension=dimension,
                source_agent=signal.source_agent,
                evidence_ids=signal.evidence_ids,
                impact_on_decision=(
                    "Confiance réduite — n'a jamais pu forcer seul un rejet ou une mise en quarantaine."
                ),
                impact_on_confidence=_SOFT_REJECT_WEIGHTS.get(signal.category, 0.3),
            )
        )

    supporting_factors: list[DecisionFactor] = []
    if identity_is_pass:
        supporting_factors.append(
            DecisionFactor(
                code="IDENTITY_CONFIRMED",
                description="Identité confirmée concordante.",
                source_agent="eligibility_agent",
            )
        )
    if coverage_is_pass:
        supporting_factors.append(
            DecisionFactor(
                code="COVERAGE_CONFIRMED",
                description="Couverture confirmée active.",
                source_agent="eligibility_agent",
            )
        )
    if has_resolved_medical_item:
        supporting_factors.append(
            DecisionFactor(
                code="MEDICAL_ITEM_RESOLVED",
                description="Au moins un acte ou médicament résolu par le référentiel.",
                source_agent="medical_risk_agent",
            )
        )

    adverse_factors: list[DecisionFactor] = [
        DecisionFactor(
            code=signal.signal_type,
            description=signal.description,
            source_agent=signal.source_agent,
            evidence_ids=signal.evidence_ids,
        )
        for signal in classified_signals
        if signal.confirmed
    ]

    return missing_information, supporting_factors, adverse_factors


def _finalize_explainability(
    *,
    missing_information: list[MissingInformation],
    supporting_factors: list[DecisionFactor],
    adverse_factors: list[DecisionFactor],
    final_decision: ClaimDecisionV2,
) -> tuple[list[DecisionAssumption], list[DecisionFactor], str]:
    """Seconde moitié de l'explicabilité — dépend de `final_decision`, donc
    calculée après la fusion Phase B. Retourne (assumptions, decisive_factors,
    recommended_action). Les hypothèses citées par le LLM
    (`LlmAutonomousDecision.assumptions`) sont ajoutées séparément par
    l'appelant (`run()`), jamais ici."""
    if adverse_factors:
        decisive_factors = adverse_factors
    elif final_decision in (ClaimDecisionV2.APPROVE, ClaimDecisionV2.PARTIAL_APPROVE):
        decisive_factors = supporting_factors
    else:
        decisive_factors = []

    assumptions: list[DecisionAssumption] = []
    if missing_information and final_decision in (ClaimDecisionV2.APPROVE, ClaimDecisionV2.PARTIAL_APPROVE):
        assumptions.append(
            DecisionAssumption(
                code="DECISION_DESPITE_INCOMPLETE_DATA",
                description=(
                    "Décision positive malgré des informations incomplètes ou ambiguës, "
                    "traitées comme neutres plutôt que défavorables."
                ),
                confidence_impact=min(0.5, sum(m.impact_on_confidence for m in missing_information)),
            )
        )

    recommended_action = f"Vérifier : {missing_information[0].description}" if missing_information else ""

    return assumptions, decisive_factors, recommended_action


# ── Fonction principale (testable sans LangGraph) ─────────────────────────────


def run(case_id: str, state: ClaimStateV2 | None = None) -> AutonomousDecisionResult:
    """Synthétise les 4 résultats agents V2 et retourne une décision finale bornée."""
    decision_state: ClaimStateV2 = dict(state or {})  # type: ignore[assignment]
    decision_state["case_id"] = case_id

    snapshot = _build_snapshot(decision_state)
    disagreements = list(detect_result_disagreements(decision_state, fields=_GENERIC_STATUS_FIELDS))
    risks = _collect_risks(snapshot)
    evidence_ids = _collect_evidence_ids(decision_state)
    disagreement_ids = [_disagreement_id(point) for point in disagreements]

    intake_safety_result = decision_state.get("intake_safety_result")
    document_understanding_result = decision_state.get("document_understanding_result")
    eligibility_result = decision_state.get("eligibility_result")
    medical_risk_result = decision_state.get("medical_risk_result")

    upstream_results_present = all(decision_state.get(field) is not None for field in _UPSTREAM_RESULT_FIELDS)

    classified_signals = classify_risk_signals(
        intake_safety_result=intake_safety_result,
        medical_risk_result=medical_risk_result,
        document_understanding_result=document_understanding_result,
    )

    identity_status = getattr(getattr(eligibility_result, "identity", None), "status", None)
    coverage_status = getattr(getattr(eligibility_result, "coverage", None), "status", None)
    document_status = getattr(document_understanding_result, "status", None)

    identity_is_pass = identity_status is VerificationStatus.PASS
    coverage_is_pass = coverage_status is VerificationStatus.PASS

    medical_risk_payload = getattr(medical_risk_result, "result_payload", None) if medical_risk_result else None
    codings = getattr(medical_risk_payload, "codings", None) or []
    has_resolved_medical_item = any(c.status is VerificationStatus.PASS for c in codings)

    coverage = getattr(eligibility_result, "coverage", None) if eligibility_result is not None else None
    ceiling_exceeded = bool(getattr(coverage, "ceiling_exceeded", False))

    has_confirmed_dangerous_clinical_signal = has_confirmed_category(
        classified_signals, RiskSignalCategory.CONFIRMED_CLINICAL_RISK
    )
    has_confirmed_coverage_exclusion = has_confirmed_category(
        classified_signals, RiskSignalCategory.ELIGIBILITY_FAILURE
    )

    requirements = evaluate_acceptance_requirements(
        identity_status=identity_status,
        document_status=document_status,
        has_confirmed_dangerous_clinical_signal=has_confirmed_dangerous_clinical_signal,
        has_confirmed_coverage_exclusion=has_confirmed_coverage_exclusion,
        identity_is_pass=identity_is_pass,
        coverage_is_pass=coverage_is_pass,
        has_resolved_medical_item=has_resolved_medical_item,
        ceiling_exceeded=ceiling_exceeded,
    )

    intake_status = snapshot["intake_safety"].get("status")
    has_partial_condition = _has_partial_approve_condition(decision_state)
    allowed, bounded_by = _allowed_decisions(
        upstream_results_present=upstream_results_present,
        intake_status=str(intake_status) if intake_status else None,
        classified_signals=classified_signals,
        has_partial_condition=has_partial_condition,
        requirements=requirements,
    )

    evidence_completeness = getattr(medical_risk_payload, "evidence_completeness", EvidenceCompleteness.COMPLETE)
    base_confidence = _BASE_CONFIDENCE_BY_COMPLETENESS.get(evidence_completeness, 0.5)

    missing_information, supporting_factors, adverse_factors = _collect_missing_information_and_factors(
        classified_signals=classified_signals,
        identity_is_pass=identity_is_pass,
        coverage_is_pass=coverage_is_pass,
        has_resolved_medical_item=has_resolved_medical_item,
    )
    known_factor_codes = {f.code for f in supporting_factors} | {f.code for f in adverse_factors}

    # Jamais consulté si `allowed` est un singleton forcé (panne structurelle/
    # BLOCKED/QUARANTINED/signal confirmé) ; toujours consulté sinon.
    llm_consulted = len(allowed) > 1
    llm_decision: LlmAutonomousDecision | None = None
    if llm_consulted:
        llm_decision = _invoke_llm_autonomous_decision(
            {
                "case_id": case_id,
                "agent_results": snapshot,
                "disagreements": [d.model_dump(mode="json") for d in disagreements],
                "disagreement_ids": disagreement_ids,
                "risks": risks,
                "evidence_ids": evidence_ids,
                "risk_signal_classification": [s.model_dump(mode="json") for s in classified_signals],
                "supporting_factors": [f.code for f in supporting_factors],
                "adverse_factors": [f.code for f in adverse_factors],
                "missing_information": [m.code for m in missing_information],
                "allowed_decisions": sorted(d.value for d in allowed),
                "instruction": (
                    "Choisis une recommandation UNIQUEMENT parmi allowed_decisions. Cite "
                    "uniquement des identifiants de facteurs/preuves déjà fournis dans "
                    "supporting_factors/adverse_factors/evidence_ids — jamais une "
                    "affirmation inventée. `confidence_adjustment` reste un ajustement "
                    "borné, jamais une confiance absolue."
                ),
            }
        )

    counterfactuals: list[DecisionCounterfactual] = []
    llm_assumption_notes: list[DecisionAssumption] = []

    if not llm_consulted:
        final_decision = next(iter(allowed))
        merge_notes: list[str] = []
        justification = list(bounded_by)
        errors: list[StructuredError] = []
        confidence = 1.0
    else:
        final_decision, merge_notes = _merge_llm_analysis(
            allowed,
            llm_decision,
            classified_signals=classified_signals,
            requirements=requirements,
            identity_is_pass=identity_is_pass,
            coverage_is_pass=coverage_is_pass,
            has_resolved_medical_item=has_resolved_medical_item,
            confirmed_partial_coverage=ceiling_exceeded,
        )
        bounded_by.extend(merge_notes)
        justification = []
        errors = []
        if llm_decision is not None:
            justification.append(llm_decision.reasoning_summary)
            justification.extend(llm_decision.unresolved_conflicts)
            justification.extend(merge_notes)
            justification.extend(_validate_llm_factor_references(llm_decision, known_codes=known_factor_codes))
            llm_assumption_notes = [
                DecisionAssumption(code="LLM_ASSUMPTION", description=text, confidence_impact=0.0)
                for text in llm_decision.assumptions
            ]
            counterfactuals = _build_counterfactuals_from_llm(llm_decision, final_decision=final_decision)
            confidence = max(0.0, min(1.0, base_confidence + llm_decision.confidence_adjustment))
        else:
            justification.extend(merge_notes)
            errors = [
                StructuredError(
                    code="LLM_UNAVAILABLE",
                    message=(
                        "LLM indisponible ou réponse invalide — décision fondée sur les "
                        "preuves disponibles, jamais un TECHNICAL_FAILURE automatique."
                    ),
                    field="llm_decision",
                )
            ]
            confidence = min(base_confidence, 0.4)

    if not justification:
        justification.append("Synthèse multi-agent sans motif exploitable.")

    status = _status_for_decision(final_decision)

    assumptions, decisive_factors, recommended_action = _finalize_explainability(
        missing_information=missing_information,
        supporting_factors=supporting_factors,
        adverse_factors=adverse_factors,
        final_decision=final_decision,
    )
    assumptions = assumptions + llm_assumption_notes

    return AutonomousDecisionResult(
        case_id=case_id,
        status=status,
        decision=final_decision,
        justification=justification,
        disagreements=disagreements,
        risks=risks,
        bounded_by=bounded_by,
        confidence=confidence,
        errors=errors,
        evidence_ids=evidence_ids,
        llm_trace=build_llm_metadata(_AGENT_NAME, confidence=confidence),
        missing_information=missing_information,
        assumptions=assumptions,
        decisive_factors=decisive_factors,
        supporting_factors=supporting_factors,
        adverse_factors=adverse_factors,
        counterfactuals=counterfactuals,
        recommended_action=recommended_action,
        evidence_completeness=evidence_completeness,
        risk_signal_classification=classified_signals,
    )


# ── Nœud du graphe V2 ──────────────────────────────────────────────────────────


def node(state: ClaimStateV2) -> dict:
    """Nœud du graphe V2 — délègue à `run()` et met à jour `ClaimStateV2`."""
    case_id = str(state.get("case_id", "UNKNOWN"))
    result = run(case_id, state)

    updates: dict = {
        "decision_result": result,
        "final_decision": result.decision,
        "current_step": "autonomous_decision",
        "completed_steps": ["autonomous_decision"],
    }
    if result.decision is ClaimDecisionV2.REJECT:
        updates["errors"] = [f"[{_AGENT_NAME}] {r}" for r in result.justification]
    elif result.decision in (
        ClaimDecisionV2.QUARANTINE,
        ClaimDecisionV2.REQUEST_MORE_INFO,
        ClaimDecisionV2.PARTIAL_APPROVE,
    ):
        updates["alerts"] = [
            f"Décision : {result.decision.value} — {'; '.join(result.justification[:5])}"
        ]

    validate_state_update_v2(updates)
    return updates
