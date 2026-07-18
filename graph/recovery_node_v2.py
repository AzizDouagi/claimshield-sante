"""Boucle de récupération autonome bornée du graphe V2 — graph/recovery_node_v2.py.

Plan de remédiation « autonomie décisionnelle V2 », Phase 6. Un seul nœud
(``recovery``), inséré entre ``medical_risk`` et ``autonomous_decision`` dans
``graph/workflow_v2.py`` — **aucune boucle LangGraph** (aucune arête ne
revient en arrière) : les tentatives bornées décrites ci-dessous sont
exécutées à l'intérieur de ce nœud, par appel direct aux fonctions pures des
agents (``run()``/``node()``, callables à volonté hors LangGraph — §0 du
plan), jamais par un second passage dans le graphe.

Désactivation entière (``_recovery_disabled``) alignée sur
``agents.autonomous_decision_agent.policy.classify_risk_signals`` — jamais un
second calcul concurrent de la nature/solidité des signaux (source unique de
vérité, partagée avec la matrice de décision, Phase 4 du plan) : un dossier
déjà refusé/mis en quarantaine à l'admission, en panne technique, ou porteur
d'un signal confirmé (clinique/fraude/éligibilité) n'a rien à gagner d'une
récupération automatique — aucune tentative n'est produite, aucun
``RecoveryAttempt`` n'est jamais créé dans ce cas.

Quatre actions bornées (``RecoveryPolicy`` — configurable, jamais verrouillé
dans le schéma, voir ``schemas.v2_results.RecoveryAttempt``), tentées dans
un ordre fixe, chacune au plus ``max_attempts_per_action`` fois et le total
au plus ``max_total_attempts`` fois par exécution du nœud :

  1. ``RECOMPUTE_ELIGIBILITY`` — si la couverture est restée NEEDS_REVIEW
     faute de donnée payeur/montant (``coverage_data_available=False``) et
     qu'un indice payeur/police existe dans le bundle FHIR déjà chargé
     (``tools.fhir_medical_items.extract_payer_hint_from_coverage``),
     réévalue l'éligibilité avec cet indice.
  2. ``READ_MEDICAL_ITEMS_FROM_FHIR`` — si les données médicales sont
     absentes, incomplètes, ambiguës (``UNKNOWN``) ou peu fiables (signal
     clinique présent, confiance documentaire faible), complète les
     actes/médicaments en lisant les ressources FHIR standards
     (``tools.fhir_medical_items.extract_medical_items_from_bundle``).
  3. ``RESOLVE_MEDICAL_CODE`` — retente une correspondance floue élargie
     (seuil abaissé) sur les éléments restés ``UNKNOWN``.
  4. ``RETRY_STRUCTURED_LLM_OUTPUT`` — si la dernière évaluation médicale
     porte une erreur ``LLM_UNAVAILABLE`` (panne isolée), rejoue une seule
     fois l'appel LLM correspondant.

Chaque tentative produit un ``RecoveryAttempt`` audité (jamais silencieux),
qu'elle réussisse, échoue ou n'améliore rien — voir
``schemas.v2_results.RecoveryOutcome``.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from agents.autonomous_decision_agent.policy import classify_risk_signals, has_confirmed_category
from schemas.domain import VerificationStatus
from schemas.v2_results import (
    ClassifiedMedicalItem,
    ClassifiedRiskSignal,
    MedicalItemType,
    RecoveryAction,
    RecoveryAttempt,
    RecoveryOutcome,
    RiskSignalCategory,
)
from state.claim_state_v2 import ClaimStateV2, validate_state_update_v2
from tools.fhir_medical_items import extract_medical_items_from_bundle, extract_payer_hint_from_coverage

__all__ = ["DEFAULT_RECOVERY_POLICY", "RecoveryPolicy", "make_recovery_node"]

_SOURCE_AGENT = "recovery_node_v2"
_LOW_DOCUMENT_CONFIDENCE_THRESHOLD = 0.5
"""Même seuil que `agents.autonomous_decision_agent.policy._LOW_DOCUMENT_CONFIDENCE_THRESHOLD`
— dupliqué volontairement (module indépendant, jamais un import croisé entre
deux modules de politique distincts)."""
_RECOVERY_FUZZY_SCORE_CUTOFF = 0.60
"""Seuil abaissé par rapport à `agents.medical_risk_agent.agent._CLASSIFICATION_SCORE_CUTOFF`
(0.80) — une seconde tentative volontairement plus permissive, jamais
appliquée à la classification initiale."""

_DISABLING_INTAKE_STATUSES = frozenset({"BLOCKED", "QUARANTINED", "TECHNICAL_FAILURE"})


@dataclass(frozen=True)
class RecoveryPolicy:
    """Bornes configurables de la boucle de récupération — jamais verrouillées
    dans `schemas.v2_results.RecoveryAttempt` (`attempt_number` n'est borné
    qu'à `ge=1`, voir sa docstring). Versionnée comme les autres politiques du
    projet (`RiskThresholds`, `DuplicateDetectionPolicy`)."""

    version: str = "1.0.0"
    max_attempts_per_action: int = 2
    max_total_attempts: int = 4

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("RecoveryPolicy.version ne peut pas être vide")
        if self.max_attempts_per_action < 1:
            raise ValueError("RecoveryPolicy.max_attempts_per_action doit être >= 1")
        if self.max_total_attempts < 1:
            raise ValueError("RecoveryPolicy.max_total_attempts doit être >= 1")


DEFAULT_RECOVERY_POLICY = RecoveryPolicy()


def _status_value(value: object | None) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _recovery_disabled(*, intake_status: str | None, classified_signals: list[ClassifiedRiskSignal]) -> bool:
    """`True` ssi une récupération automatique n'apporterait aucune valeur —
    même classification que la matrice de décision (`classify_risk_signals`),
    jamais un second calcul concurrent."""
    if intake_status in _DISABLING_INTAKE_STATUSES:
        return True
    return (
        has_confirmed_category(classified_signals, RiskSignalCategory.CONFIRMED_CLINICAL_RISK)
        or has_confirmed_category(classified_signals, RiskSignalCategory.CONFIRMED_FRAUD_RISK)
        or has_confirmed_category(classified_signals, RiskSignalCategory.ELIGIBILITY_FAILURE)
    )


# ── Chargement du bundle FHIR déjà identifié (réutilisation, aucun nouveau ────
# parseur) ─────────────────────────────────────────────────────────────────


def _load_fhir_bundle_for_case(state: dict) -> dict | None:
    """Charge le bundle FHIR déjà identifié par `intake_safety_agent` — `None`
    si aucun bundle n'est attendu ou que le chargement échoue (jamais une
    exception propagée : l'absence de bundle est un cas normal, pas une
    panne). Réutilise les mêmes helpers que `eligibility_agent`/
    `document_understanding_agent` — jamais un second parseur."""
    from agents.eligibility_agent.agent import _find_fhir_bundle_path
    from agents.fhir_validator_agent.agent import _resolve_bundle_path
    from tools.fhir_validation import load_fhir_bundle

    relative_path = _find_fhir_bundle_path(state)  # type: ignore[arg-type]
    if not relative_path:
        return None
    resolved = _resolve_bundle_path(relative_path)
    bundle, errors = load_fhir_bundle(resolved)
    if errors or bundle is None:
        return None
    return bundle


def _ocr_result_like(document_understanding_result: Any) -> object | None:
    """Même construction que `agents.medical_risk_agent.agent.node()` —
    duck-typing volontaire, jamais un nouveau schéma dupliqué."""
    if document_understanding_result is None or document_understanding_result.extraction is None:
        return None
    extraction = document_understanding_result.extraction
    return SimpleNamespace(
        extracted_fields=extraction.fields,
        confidence_score=extraction.confidence_score,
        document_type=None,
        sha256=None,
    )


def _identity_coverage_like(eligibility_result: Any) -> object | None:
    if eligibility_result is None:
        return None
    return SimpleNamespace(identity=eligibility_result.identity, coverage=eligibility_result.coverage)


def _run_medical_risk(
    working_state: dict,
    *,
    classified_items: list[ClassifiedMedicalItem],
) -> Any:
    from agents.medical_risk_agent.agent import run as medical_risk_run

    procedures = [i.description for i in classified_items if i.item_type is MedicalItemType.PROCEDURE]
    medications = [i.description for i in classified_items if i.item_type is MedicalItemType.MEDICATION]

    return medical_risk_run(
        case_id=str(working_state.get("case_id", "")),
        procedures=procedures,
        medications=medications,
        ocr_result=_ocr_result_like(working_state.get("document_understanding_result")),
        identity_coverage_result=_identity_coverage_like(working_state.get("eligibility_result")),
        classified_items=classified_items,
    )


# ── Actions de récupération — chacune retourne (déclenchée, updates, attempt) ─


def _try_recompute_eligibility(
    working_state: dict, *, bundle: dict | None, attempt_number: int
) -> tuple[bool, dict, RecoveryAttempt | None]:
    eligibility_result = working_state.get("eligibility_result")
    if eligibility_result is None or bundle is None:
        return False, {}, None
    if eligibility_result.coverage.status is not VerificationStatus.NEEDS_REVIEW:
        return False, {}, None
    if eligibility_result.coverage_data_available:
        return False, {}, None

    hint = extract_payer_hint_from_coverage(bundle)
    if not hint or not (hint.get("payer_name") or hint.get("policy_number")):
        return True, {}, RecoveryAttempt(
            action=RecoveryAction.RECOMPUTE_ELIGIBILITY,
            reason=(
                "Couverture non évaluable par donnée manquante mais aucun indice payeur/police "
                "exploitable dans le bundle FHIR."
            ),
            source_agent=_SOURCE_AGENT,
            attempt_number=attempt_number,
            result=RecoveryOutcome.NO_IMPROVEMENT,
        )

    from agents.eligibility_agent.agent import _find_fhir_bundle_path
    from agents.eligibility_agent.agent import run as eligibility_run

    document_understanding_result = working_state.get("document_understanding_result")
    extracted_fields: dict = {}
    if document_understanding_result is not None and document_understanding_result.extraction is not None:
        extracted_fields = dict(document_understanding_result.extraction.fields)
    if hint.get("payer_name"):
        extracted_fields["payer_name"] = hint["payer_name"]

    new_result = eligibility_run(
        case_id=str(working_state.get("case_id", "")),
        extracted_fields=extracted_fields,
        fhir_bundle_path=_find_fhir_bundle_path(working_state),  # type: ignore[arg-type]
        policy_number=hint.get("policy_number"),
    )

    outcome = RecoveryOutcome.SUCCESS if new_result.coverage_data_available else RecoveryOutcome.NO_IMPROVEMENT
    attempt = RecoveryAttempt(
        action=RecoveryAction.RECOMPUTE_ELIGIBILITY,
        reason="Indice payeur/police trouvé dans le bundle FHIR — nouvelle évaluation de couverture tentée.",
        source_agent=_SOURCE_AGENT,
        attempt_number=attempt_number,
        result=outcome,
    )
    updates = {"eligibility_result": new_result} if outcome is RecoveryOutcome.SUCCESS else {}
    return True, updates, attempt


def _try_read_medical_items_from_fhir(
    working_state: dict, *, bundle: dict | None, attempt_number: int
) -> tuple[bool, dict, RecoveryAttempt | None]:
    medical_risk_result = working_state.get("medical_risk_result")
    if medical_risk_result is None or bundle is None:
        return False, {}, None

    payload = medical_risk_result.result_payload
    structural_absence = not payload.codings and not payload.procedure_count and not payload.medication_count
    has_unknown_item = any(i.item_type is MedicalItemType.UNKNOWN for i in payload.classified_items)
    has_clinical_signal = bool(payload.clinical_signals)
    document_understanding_result = working_state.get("document_understanding_result")
    confidence = getattr(document_understanding_result, "confidence", None)
    low_confidence = confidence is not None and confidence < _LOW_DOCUMENT_CONFIDENCE_THRESHOLD

    if not (structural_absence or has_unknown_item or has_clinical_signal or low_confidence):
        return False, {}, None

    fhir_items = extract_medical_items_from_bundle(bundle)
    known_descriptions = {i.description.strip().lower() for i in payload.classified_items}
    new_items = [i for i in fhir_items if i.description.strip().lower() not in known_descriptions]

    if not new_items:
        return True, {}, RecoveryAttempt(
            action=RecoveryAction.READ_MEDICAL_ITEMS_FROM_FHIR,
            reason=(
                "Données médicales incomplètes/incertaines mais aucun acte/médicament nouveau "
                "exploitable dans le bundle FHIR."
            ),
            source_agent=_SOURCE_AGENT,
            attempt_number=attempt_number,
            result=RecoveryOutcome.NO_IMPROVEMENT,
        )

    merged_classified_items = list(payload.classified_items) + new_items
    new_result = _run_medical_risk(working_state, classified_items=merged_classified_items)

    outcome = (
        RecoveryOutcome.SUCCESS
        if len(new_result.result_payload.classified_items) > len(payload.classified_items)
        else RecoveryOutcome.NO_IMPROVEMENT
    )
    attempt = RecoveryAttempt(
        action=RecoveryAction.READ_MEDICAL_ITEMS_FROM_FHIR,
        reason=f"{len(new_items)} acte(s)/médicament(s) lu(s) depuis les ressources FHIR standards.",
        source_agent=_SOURCE_AGENT,
        attempt_number=attempt_number,
        result=outcome,
        evidence_ids=[i.source_document_id for i in new_items if i.source_document_id],
    )
    updates = {"medical_risk_result": new_result} if outcome is RecoveryOutcome.SUCCESS else {}
    return True, updates, attempt


def _try_resolve_medical_code(
    working_state: dict, *, attempt_number: int
) -> tuple[bool, dict, RecoveryAttempt | None]:
    medical_risk_result = working_state.get("medical_risk_result")
    if medical_risk_result is None:
        return False, {}, None

    payload = medical_risk_result.result_payload
    unknown_items = [i for i in payload.classified_items if i.item_type is MedicalItemType.UNKNOWN]
    if not unknown_items:
        return False, {}, None

    from agents.medical_risk_agent.agent import _classify_medical_item

    resolved_items: list[ClassifiedMedicalItem] = []
    for item in unknown_items:
        reclassified = _classify_medical_item(
            item.description,
            source_document_id=item.source_document_id,
            score_cutoff=_RECOVERY_FUZZY_SCORE_CUTOFF,
        )
        if reclassified.item_type is not MedicalItemType.UNKNOWN:
            resolved_items.append(reclassified)

    if not resolved_items:
        return True, {}, RecoveryAttempt(
            action=RecoveryAction.RESOLVE_MEDICAL_CODE,
            reason="Élargissement du seuil de correspondance floue sans résultat exploitable.",
            source_agent=_SOURCE_AGENT,
            attempt_number=attempt_number,
            result=RecoveryOutcome.NO_IMPROVEMENT,
        )

    resolved_descriptions = {i.description.strip().lower() for i in resolved_items}
    merged_classified_items = [
        i for i in payload.classified_items if i.description.strip().lower() not in resolved_descriptions
    ] + resolved_items
    new_result = _run_medical_risk(working_state, classified_items=merged_classified_items)

    new_unknown_count = sum(
        1 for i in new_result.result_payload.classified_items if i.item_type is MedicalItemType.UNKNOWN
    )
    outcome = RecoveryOutcome.SUCCESS if new_unknown_count < len(unknown_items) else RecoveryOutcome.NO_IMPROVEMENT
    attempt = RecoveryAttempt(
        action=RecoveryAction.RESOLVE_MEDICAL_CODE,
        reason=f"{len(resolved_items)} élément(s) précédemment non résolu(s) reclassé(s) par seuil élargi.",
        source_agent=_SOURCE_AGENT,
        attempt_number=attempt_number,
        result=outcome,
    )
    updates = {"medical_risk_result": new_result} if outcome is RecoveryOutcome.SUCCESS else {}
    return True, updates, attempt


def _try_retry_structured_llm_output(
    working_state: dict, *, attempt_number: int
) -> tuple[bool, dict, RecoveryAttempt | None]:
    medical_risk_result = working_state.get("medical_risk_result")
    if medical_risk_result is None:
        return False, {}, None
    had_llm_failure = any(e.code == "LLM_UNAVAILABLE" for e in medical_risk_result.errors)
    if not had_llm_failure:
        return False, {}, None

    from agents.medical_risk_agent.agent import node as medical_risk_node

    node_updates = medical_risk_node(working_state)  # type: ignore[arg-type]
    new_result = node_updates.get("medical_risk_result")
    still_failed = new_result is None or any(e.code == "LLM_UNAVAILABLE" for e in new_result.errors)
    outcome = RecoveryOutcome.NO_IMPROVEMENT if still_failed else RecoveryOutcome.SUCCESS
    attempt = RecoveryAttempt(
        action=RecoveryAction.RETRY_STRUCTURED_LLM_OUTPUT,
        reason="Nouvel appel LLM tenté après une panne isolée précédente (résultat déterministe conservé).",
        source_agent=_SOURCE_AGENT,
        attempt_number=attempt_number,
        result=outcome,
    )
    updates = {"medical_risk_result": new_result} if outcome is RecoveryOutcome.SUCCESS else {}
    return True, updates, attempt


_ActionRunner = Callable[[dict, int], tuple[bool, dict, RecoveryAttempt | None]]


def make_recovery_node(*, policy: RecoveryPolicy | None = None) -> Callable[[ClaimStateV2], dict]:
    """Crée le nœud `recovery` — tente, dans cet ordre fixe, les 4 actions
    bornées par `policy` (défaut `DEFAULT_RECOVERY_POLICY`, injectable pour
    les tests). Jamais de boucle : chaque action s'exécute au plus une fois
    par appel de nœud (bornée par `max_attempts_per_action`), le nombre total
    d'actions tentées est plafonné par `max_total_attempts`."""
    active_policy = policy or DEFAULT_RECOVERY_POLICY

    def _node(state: ClaimStateV2) -> dict:
        intake_safety_result = state.get("intake_safety_result")
        medical_risk_result = state.get("medical_risk_result")
        document_understanding_result = state.get("document_understanding_result")

        intake_status = _status_value(getattr(intake_safety_result, "status", None))
        classified_signals = classify_risk_signals(
            intake_safety_result=intake_safety_result,
            medical_risk_result=medical_risk_result,
            document_understanding_result=document_understanding_result,
        )

        base_updates: dict = {"current_step": "recovery", "completed_steps": ["recovery"]}

        if _recovery_disabled(intake_status=intake_status, classified_signals=classified_signals):
            validate_state_update_v2(base_updates)
            return base_updates

        working_state: dict = dict(state)
        bundle_cache: dict[str, Any] = {"loaded": False, "bundle": None}

        def _bundle() -> dict | None:
            if not bundle_cache["loaded"]:
                bundle_cache["bundle"] = _load_fhir_bundle_for_case(working_state)
                bundle_cache["loaded"] = True
            return bundle_cache["bundle"]

        action_runners: list[tuple[RecoveryAction, _ActionRunner]] = [
            (
                RecoveryAction.RECOMPUTE_ELIGIBILITY,
                lambda s, n: _try_recompute_eligibility(s, bundle=_bundle(), attempt_number=n),
            ),
            (
                RecoveryAction.READ_MEDICAL_ITEMS_FROM_FHIR,
                lambda s, n: _try_read_medical_items_from_fhir(s, bundle=_bundle(), attempt_number=n),
            ),
            (
                RecoveryAction.RESOLVE_MEDICAL_CODE,
                lambda s, n: _try_resolve_medical_code(s, attempt_number=n),
            ),
            (
                RecoveryAction.RETRY_STRUCTURED_LLM_OUTPUT,
                lambda s, n: _try_retry_structured_llm_output(s, attempt_number=n),
            ),
        ]

        attempts: list[RecoveryAttempt] = []
        attempts_per_action: dict[RecoveryAction, int] = {}
        total_attempts = 0

        for action, runner in action_runners:
            if total_attempts >= active_policy.max_total_attempts:
                break
            used = attempts_per_action.get(action, 0)
            if used >= active_policy.max_attempts_per_action:
                continue

            triggered, updates, attempt = runner(working_state, used + 1)
            if not triggered:
                continue

            total_attempts += 1
            attempts_per_action[action] = used + 1
            if attempt is not None:
                attempts.append(attempt)
            if updates:
                working_state.update(updates)

        result_updates: dict = dict(base_updates)
        if attempts:
            result_updates["recovery_attempts"] = attempts
        for key in ("document_understanding_result", "eligibility_result", "medical_risk_result"):
            if working_state.get(key) is not state.get(key):
                result_updates[key] = working_state[key]

        validate_state_update_v2(result_updates)
        return result_updates

    return _node
