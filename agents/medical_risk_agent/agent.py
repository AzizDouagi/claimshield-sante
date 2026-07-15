"""Medical & Risk Agent (V2) — fusion de `medical_coding_agent` +
`clinical_consistency_agent` + `fraud_detection_agent` (V1).

Un seul agent, trois Phases A déterministes réunies dans le même `run()`,
un seul appel ReAct LLM avec les trois outils réunis (plan de refonte V2,
Phase V2-5 — fallback officiel V2-5-bis documenté dans le plan si cette
fusion dégrade la qualité mesurée sur les 37 fixtures).

Réutilise par import les fonctions déterministes ET les fonctions
d'ajustement borné LLM déjà pures de V1 — jamais dupliquées, jamais
modifiées (§0 du plan) :
  - `tools.medical_coding` : `code_procedures`, `code_medications`,
    `compute_global_status`.
  - `agents.medical_coding_agent.agent._merge_with_llm` (fusion codification
    bornée — n'accepte un code LLM que s'il figure dans les candidats déjà
    proposés ET existe dans le référentiel).
  - `agents.clinical_consistency_agent.agent` : `_collect_signals`,
    `_status_from_signals`, `_apply_signal_assessments` (ajustement de
    sévérité borné à un cran), `_fhir_summary`, `_medical_view_summary`.
  - `agents.fraud_detection_agent.agent` : `_collect_signals`,
    `_determine_status`, `_apply_signal_assessments` (ajustement de
    pondération borné DOWNGRADE/NEUTRAL/UPGRADE), `_check_duplicate`.
  - Les 3 outils `@tool` déjà autorisés en V1 : `rechercher_code`,
    `verifier_chronologie`, `verifier_doublon`.

`_collect_signals` (clinique et fraude, V1) attendent un objet
`coding_result` avec attributs `.codings`/`.status` — ce module leur passe
un `types.SimpleNamespace` construit depuis `initial_codings` (duck-typing
volontaire, jamais un nouveau schéma dupliqué) : aucune modification des
fonctions V1 réutilisées n'est nécessaire.

Correctif post-mesure V2-10 (AZIZ) : `risk_score`/`risk_level` ne sont plus
calculés sur la totalité des signaux fraude V1 — voir `_RISK_SIGNAL_TYPES`/
`_COMPLETENESS_SIGNAL_TYPES`/`RiskThresholds` plus bas, qui séparent
désormais le danger réel (alimente `risk_level`, jusqu'à `CRITICAL`) de la
complétude des preuves (`evidence_completeness`, nouveau). Les poids
`risk_contribution` de V1 (`_collect_signals`) restent inchangés — seule
leur agrégation par cet agent change.

Limite MVP assumée (héritée de V1, voir CLAUDE.md « câblage minimal ») :
`procedures`/`medications` restent des listes vides tant qu'aucun
discriminant acte/médicament n'est disponible depuis un seul document —
jamais une répartition heuristique inventée (même décision que
`graph/input_builders.py::build_coding_input`, V1).

Simplification volontaire par rapport à V1 : `medical_view`/`fraud_view`
(vues privacy pseudonymisées spécifiques à un rôle) ne sont pas dérivées
automatiquement par `node()` — `document_understanding_agent` (V2) ne
construit qu'une seule vue, pour le rôle unique posé à la soumission
(`ClaimStateV2.reader_role`), jamais une vue par rôle métier comme en V1.
`run()` accepte ces paramètres si un appelant les fournit explicitement
(tests), mais `node()` ne les peuple pas — la Phase A n'en dépend jamais
pour ses signaux, seul le contexte transmis au LLM est appauvri d'autant.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

from langchain_core.messages import HumanMessage, SystemMessage

try:
    from langgraph.prebuilt import create_react_agent
except ModuleNotFoundError:  # pragma: no cover - dépendance optionnelle en tests locaux
    def create_react_agent(*_args, **_kwargs):
        raise RuntimeError("langgraph indisponible")

from agents.clinical_consistency_agent.agent import (
    _apply_signal_assessments as _apply_clinical_assessments,
)
from agents.clinical_consistency_agent.agent import (
    _collect_signals as _collect_clinical_signals,
)
from agents.clinical_consistency_agent.agent import (
    _fhir_summary,
    _medical_view_summary,
    _status_from_signals,
)
from agents.clinical_consistency_agent.tools import verifier_chronologie
from agents.fraud_detection_agent.agent import (
    _apply_signal_assessments as _apply_fraud_assessments,
)
from agents.fraud_detection_agent.agent import (
    _check_duplicate,
)
from agents.fraud_detection_agent.agent import (
    _collect_signals as _collect_fraud_signals,
)
from agents.fraud_detection_agent.agent import (
    _determine_status as _determine_fraud_status,
)
from agents.fraud_detection_agent.tools import _DEFAULT_DUPLICATE_INDEX, verifier_doublon
from agents.medical_coding_agent.agent import _merge_with_llm
from agents.medical_coding_agent.tools import rechercher_code
from agents.medical_risk_agent.prompt import load_medical_risk_prompt
from agents.medical_risk_agent.schemas import LlmMedicalRiskDecision
from agents.privacy_agent.schemas import FraudView
from llm.factory import get_llm
from llm.metadata import build_llm_metadata
from schemas.domain import VerificationStatus
from schemas.results import FraudSignal, ProcedureCoding, StructuredError
from schemas.v2_results import (
    EvidenceCompleteness,
    MedicalRiskResult,
    MedicalRiskResultPayload,
    RiskLevel,
)
from services.duplicate_index import DuplicateIndex
from state.claim_state_v2 import ClaimStateV2, validate_state_update_v2
from tools.medical_coding import code_medications, code_procedures, compute_global_status

_AGENT_NAME = "medical_risk_agent"

_STATUS_RANK: dict[VerificationStatus, int] = {
    VerificationStatus.PASS: 0,
    VerificationStatus.NEEDS_REVIEW: 1,
    VerificationStatus.FAIL: 2,
}


def _worst(a: VerificationStatus, b: VerificationStatus) -> VerificationStatus:
    return a if _STATUS_RANK[a] >= _STATUS_RANK[b] else b


# ── Séparation risque réel / complétude des preuves (correctif post-mesure ────
# V2-10, AZIZ) ─────────────────────────────────────────────────────────────
#
# La mesure réelle sur les 37 dossiers de démo a montré que `risk_level`
# atteignait HIGH sur 36/37 dossiers alors qu'aucun n'était réellement
# suspect — parce que `_collect_fraud_signals` (V1, réutilisée telle quelle)
# mélange dans la même somme pondérée des signaux de danger réel
# (identité confirmée non concordante, doublon de facture, plafond dépassé)
# et des signaux de données manquantes (codification jamais tentée,
# identité ambiguë faute de preuve, préautorisation non renseignée, faible
# confiance d'extraction). Ces derniers sont désormais exclus du calcul de
# `risk_score`/`risk_level` — ils alimentent uniquement `evidence_completeness`
# (nouveau, voir `schemas.v2_results.EvidenceCompleteness`), jamais un
# plafonnement vers QUARANTINE. Les poids `risk_contribution` eux-mêmes ne
# sont jamais modifiés (fonction V1 `_collect_signals` non touchée) — seule
# la façon dont `medical_risk_agent` les agrège change.

_RISK_SIGNAL_TYPES: frozenset[str] = frozenset(
    {
        "IDENTITY_MISMATCH",
        "COVERAGE_INACTIVE_OR_EXPIRED",
        "CEILING_EXCEEDED",
        "EXACT_DUPLICATE_INVOICE",
        "NEAR_DUPLICATE_INVOICE",
    }
)
"""Signaux de danger réel — alimentent `risk_score`/`risk_level`."""

_COMPLETENESS_SIGNAL_TYPES: frozenset[str] = frozenset(
    {
        "IDENTITY_AMBIGUOUS",
        "PREAUTHORIZATION_MISSING",
        "UNRESOLVED_CODING",
        "LOW_EXTRACTION_CONFIDENCE",
    }
)
"""Signaux de données manquantes/ambiguës — alimentent `evidence_completeness`,
jamais `risk_score`."""

_CRITICAL_SIGNAL_TYPES: frozenset[str] = frozenset({"EXACT_DUPLICATE_INVOICE"})
"""Signaux qui forcent `RiskLevel.CRITICAL` indépendamment du score — preuve
de danger déjà quasi certaine (octets identiques à une facture déjà connue)."""


@dataclass(frozen=True)
class RiskThresholds:
    """Seuils de `risk_score` (calculé sur les signaux de danger réel
    uniquement) → `RiskLevel` — versionnés et configurables, jamais codés en
    dur ailleurs. Valeurs proposées à valider empiriquement sur un
    échantillon réel (voir plan de remédiation post-mesure V2-10, étape 2)
    avant le rejeu complet des 37 dossiers — pas encore confirmées par une
    mesure LLM réelle."""

    version: str = "1.0.0"
    critical_score: float = 0.70
    high_score: float = 0.40
    medium_score: float = 0.15

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("RiskThresholds.version ne peut pas être vide")
        if not (0.0 <= self.medium_score <= self.high_score <= self.critical_score <= 1.0):
            raise ValueError(
                "RiskThresholds doit vérifier 0 <= medium_score <= high_score <= "
                "critical_score <= 1"
            )


DEFAULT_RISK_THRESHOLDS = RiskThresholds()


def _split_fraud_signals(
    signals: list[FraudSignal],
) -> tuple[list[FraudSignal], list[FraudSignal]]:
    """Sépare les signaux fraude déjà calculés (jamais recalculés ici) en
    (signaux de risque réel, signaux de complétude) selon leur `signal_type`.
    Un signal de type inconnu des deux ensembles (ne devrait jamais arriver
    en pratique — les `signal_type` sont tous produits par
    `agents.fraud_detection_agent.agent._collect_signals`/`_check_duplicate`,
    eux-mêmes non modifiés) est classé prudemment côté risque réel plutôt que
    silencieusement ignoré."""
    risk_signals: list[FraudSignal] = []
    completeness_signals: list[FraudSignal] = []
    for signal in signals:
        if signal.signal_type in _COMPLETENESS_SIGNAL_TYPES:
            completeness_signals.append(signal)
        else:
            risk_signals.append(signal)
    return risk_signals, completeness_signals


def _risk_level_from_score(
    score: float,
    *,
    risk_signals: list[FraudSignal],
    thresholds: RiskThresholds = DEFAULT_RISK_THRESHOLDS,
) -> RiskLevel:
    """Dérive `RiskLevel` du score de risque réel (signaux de danger
    uniquement, voir `_split_fraud_signals`) — un signal de
    `_CRITICAL_SIGNAL_TYPES` (ex. doublon exact de facture) force CRITICAL
    indépendamment du score, jamais l'inverse (un score élevé sans preuve
    de ce type ne peut pas dépasser HIGH)."""
    if any(s.signal_type in _CRITICAL_SIGNAL_TYPES for s in risk_signals):
        return RiskLevel.CRITICAL
    if score >= thresholds.critical_score:
        return RiskLevel.CRITICAL
    if score >= thresholds.high_score:
        return RiskLevel.HIGH
    if score >= thresholds.medium_score:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _evidence_completeness(
    *,
    completeness_signals: list[FraudSignal],
    structural_absence: bool,
) -> EvidenceCompleteness:
    """`structural_absence` : aucune procédure/médicament ni preuve n'a même
    été soumise à la codification (limite MVP du câblage, voir
    `node()`/CLAUDE.md « câblage minimal ») — distinct d'une résolution
    tentée et restée ambiguë (1 à 2 signaux de complétude).

    Seuil `>= 3` (décision AZIZ post-mesure V2-10, sur mesure réelle des 37
    fixtures) : le cas réel le plus fréquent — nom patient absent de l'OCR
    (`IDENTITY_AMBIGUOUS`) + code médicament resté en correspondance
    approximative (`UNRESOLVED_CODING`) — reste `PARTIAL` (LLM consulté,
    `APPROVE` reste atteignable si le reste du dossier est propre), pas
    `INSUFFICIENT` (`REQUEST_MORE_INFO` forcé). Le seuil précédent (`>= 2`)
    bloquait systématiquement ce cas pourtant courant vers `REQUEST_MORE_INFO`
    sans jamais consulter le LLM."""
    if structural_absence or len(completeness_signals) >= 3:
        return EvidenceCompleteness.INSUFFICIENT
    if len(completeness_signals) >= 1:
        return EvidenceCompleteness.PARTIAL
    return EvidenceCompleteness.COMPLETE


# ── Phase B : un seul appel ReAct LLM combiné ─────────────────────────────────


def _invoke_llm_medical_risk(data: dict) -> LlmMedicalRiskDecision | None:
    """Lance l'agent ReAct LLM (appel obligatoire à chaque exécution) avec
    les trois outils réunis — seuls outils physiquement joignables :
    `rechercher_code`, `verifier_chronologie`, `verifier_doublon`."""
    try:
        prompt = load_medical_risk_prompt()
        llm = get_llm()
        agent = create_react_agent(
            model=llm,
            tools=[rechercher_code, verifier_chronologie, verifier_doublon],
            response_format=LlmMedicalRiskDecision,
        )
        result = agent.invoke(
            {
                "messages": [
                    SystemMessage(content=prompt.system_prompt),
                    HumanMessage(content=json.dumps(data, ensure_ascii=False, default=str)),
                ]
            }
        )
        structured = result.get("structured_response")
        if isinstance(structured, LlmMedicalRiskDecision):
            return structured
        if isinstance(structured, dict):
            return LlmMedicalRiskDecision(**structured)
        return None
    except Exception:
        return None


# ── Fonction principale (testable sans LangGraph) ─────────────────────────────


def run(
    case_id: str,
    procedures: list[str] | None = None,
    medications: list[str] | None = None,
    *,
    ocr_result: object | None = None,
    fhir_result: object | None = None,
    identity_coverage_result: object | None = None,
    medical_view: object | None = None,
    fraud_view: object | None = None,
    duplicate_index: DuplicateIndex | None = None,
) -> MedicalRiskResult:
    """Évalue codification + cohérence clinique + risque de fraude en une
    seule passe.

    Args:
        case_id: identifiant du dossier.
        procedures/medications: descriptions à coder (limite MVP : listes
            vides en pratique, voir docstring du module).
        ocr_result: objet exposant `.extracted_fields`/`.confidence_score`/
            `.document_type`/`.sha256` (duck-typing, `DocumentOcrResult`
            V1 ou équivalent).
        fhir_result: objet exposant `.status`/`.resource_count`/
            `.resource_types` (résumé FHIR, jamais le bundle brut).
        identity_coverage_result: objet exposant `.identity`/`.coverage`.
        medical_view/fraud_view: vues privacy déjà minimisées (optionnelles).
        duplicate_index: index injecté (tests) ; `None` retombe sur l'index
            partagé par défaut de `fraud_detection_agent`.

    Returns:
        MedicalRiskResult — statut PASS/NEEDS_REVIEW/FAIL.
    """
    procedures = procedures or []
    medications = medications or []
    section_by_description = {
        **{d: "procedures" for d in procedures},
        **{d: "medications" for d in medications},
    }

    # ── Phase A.1 : codification déterministe ────────────────────────────────
    initial_codings: list[ProcedureCoding] = [
        *code_procedures(procedures),
        *code_medications(medications),
    ]
    coding_status = compute_global_status(initial_codings) if initial_codings else VerificationStatus.NEEDS_REVIEW
    coding_snapshot = SimpleNamespace(codings=initial_codings, status=coding_status)

    # ── Phase A.2 : cohérence clinique déterministe (réutilise V1 telle quelle) ─
    (
        clinical_signals,
        inconsistencies,
        procedure_count,
        medication_count,
        _prescription_required,
        clinical_status,
        clinical_reasons,
    ) = _collect_clinical_signals(ocr_result, coding_snapshot)

    # ── Phase A.3 : fraude déterministe (réutilise V1 telle quelle) ────────────
    insufficient_fraud_evidence = identity_coverage_result is None and ocr_result is None
    fraud_signals, _duplicate_placeholder = _collect_fraud_signals(
        identity_coverage_result, coding_snapshot, ocr_result
    )
    index = duplicate_index if duplicate_index is not None else _DEFAULT_DUPLICATE_INDEX
    duplicate_signal, duplicate_invoice, duplicate_reason = _check_duplicate(
        case_id, fraud_view if isinstance(fraud_view, FraudView) else None, ocr_result, index
    )
    if duplicate_signal is not None:
        fraud_signals.append(duplicate_signal)
    # `fraud_status`/`risk_score` (Phase A) ne sont plus calculés ici — la
    # séparation risque réel/complétude (voir `_split_fraud_signals`) n'est
    # appliquée qu'une seule fois, en Phase C, sur `fraud_signals_final`
    # (identique à `fraud_signals` quand l'ajustement LLM ne change rien).

    # ── Phase B : un seul appel ReAct LLM combiné ────────────────────────────
    needs_review_codings = [c for c in initial_codings if c.status == VerificationStatus.NEEDS_REVIEW]
    already_coded = [c for c in initial_codings if c.status == VerificationStatus.PASS]

    clinical_evidence_ids = [e.evidence_id for s in clinical_signals for e in s.evidence] + [
        e.evidence_id for i in inconsistencies for e in i.evidence
    ]
    clinical_inconsistency_types = [i.inconsistency_type for i in inconsistencies]
    fraud_evidence_ids = [e.evidence_id for s in fraud_signals for e in s.evidence]
    fraud_signal_types = sorted({s.signal_type for s in fraud_signals})

    llm_decision = _invoke_llm_medical_risk(
        {
            "case_id": case_id,
            "coding": {
                "needs_review": [
                    {
                        "description": c.original_description,
                        "rule_applied": c.rule_applied,
                        "candidates": c.alternatives,
                        "evidence": c.evidence,
                    }
                    for c in needs_review_codings
                ],
                "already_coded": [
                    {"description": c.original_description, "code": c.proposed_code}
                    for c in already_coded
                ],
            },
            "clinical": {
                "signals": [
                    {"signal_type": s.signal_type, "severity": s.severity.value}
                    for s in clinical_signals
                ],
                "evidence_ids": clinical_evidence_ids,
                "inconsistency_types": clinical_inconsistency_types,
                "fhir_minimise": _fhir_summary(fhir_result),
                "vue_medicale_minimisee": _medical_view_summary(medical_view),
            },
            "fraud": {
                "signal_types": fraud_signal_types,
                "signaux_detailles": [
                    {"signal_type": s.signal_type, "risk_contribution": round(s.risk_contribution, 2)}
                    for s in fraud_signals
                ],
                "doublons": {
                    "duplicate_invoice": duplicate_invoice,
                    "has_exact_duplicate": duplicate_signal is not None
                    and duplicate_signal.signal_type == "EXACT_DUPLICATE_INVOICE",
                    "has_near_duplicate": duplicate_signal is not None
                    and duplicate_signal.signal_type == "NEAR_DUPLICATE_INVOICE",
                },
                "evidence_ids": fraud_evidence_ids,
                "montant": {
                    "amount_requested": str(fraud_view.amount_requested)
                    if isinstance(fraud_view, FraudView) and fraud_view.amount_requested
                    else None,
                },
            },
            "instruction": (
                "Codification : ne choisis un code que parmi les candidats déjà fournis. "
                "Cohérence clinique : ajustement de sévérité borné à un cran, jamais de "
                "statut. Fraude : ajustement de pondération borné, jamais de score ni "
                "d'accusation. Ne cite que des identifiants déjà présents ci-dessus."
            ),
        }
    )

    # ── Phase C : fusion bornée (réutilise les fonctions V1 telles quelles) ──
    if llm_decision is not None:
        final_codings = _merge_with_llm(
            initial_codings, SimpleNamespace(resolved=llm_decision.coding_resolved), section_by_description
        )
    else:
        final_codings = initial_codings
    coding_status_final = (
        compute_global_status(final_codings) if final_codings else VerificationStatus.NEEDS_REVIEW
    )

    clinical_signals_final, clinical_notes, clinical_changed = _apply_clinical_assessments(
        clinical_signals, llm_decision.clinical_severity_assessments if llm_decision is not None else []
    )
    if clinical_changed:
        clinical_status_final, clinical_status_reasons = _status_from_signals(clinical_signals_final)
    else:
        clinical_status_final, clinical_status_reasons = clinical_status, []

    fraud_signals_final, fraud_notes = _apply_fraud_assessments(
        fraud_signals, llm_decision.fraud_signal_assessments if llm_decision is not None else []
    )

    # ── Séparation risque réel / complétude (post-mesure V2-10, AZIZ) ────────
    # `risk_score`/`risk_level` ne sont plus jamais dérivés des signaux de
    # données manquantes (voir `_RISK_SIGNAL_TYPES`/`_COMPLETENESS_SIGNAL_TYPES`
    # en tête de module) — ces derniers alimentent exclusivement
    # `evidence_completeness`, jamais un plafonnement vers QUARANTINE.
    risk_signals_final, completeness_signals_final = _split_fraud_signals(fraud_signals_final)
    fraud_status_final, risk_score_final = _determine_fraud_status(
        risk_signals_final, insufficient_evidence=insufficient_fraud_evidence
    )
    risk_level = _risk_level_from_score(risk_score_final, risk_signals=risk_signals_final)

    structural_absence = not final_codings and not procedures and not medications
    evidence_completeness = _evidence_completeness(
        completeness_signals=completeness_signals_final, structural_absence=structural_absence
    )
    # Une complétude insuffisante dégrade au plus à NEEDS_REVIEW — jamais
    # FAIL (réservé aux vrais échecs cliniques/de risque) : c'est une donnée
    # manquante, pas une preuve de danger.
    completeness_status = (
        VerificationStatus.PASS
        if evidence_completeness is EvidenceCompleteness.COMPLETE
        else VerificationStatus.NEEDS_REVIEW
    )

    overall_status = _worst(
        coding_status_final, _worst(clinical_status_final, _worst(fraud_status_final, completeness_status))
    )

    reasons: list[str] = list(clinical_reasons)
    reasons.append(duplicate_reason)
    if not final_codings:
        reasons.append("Aucun acte ou médicament fourni pour codification.")
    if completeness_signals_final:
        reasons.append(
            f"{len(completeness_signals_final)} signal(aux) de données manquantes/ambiguës "
            f"({', '.join(sorted({s.signal_type for s in completeness_signals_final}))}) — "
            "comptés en complétude, jamais en risque."
        )
    reasons.extend(clinical_status_reasons)
    reasons.extend(clinical_notes)
    reasons.extend(fraud_notes)

    if llm_decision is None:
        reasons.append(
            "LLM indisponible : codification/cohérence clinique/fraude conservées telles "
            "que calculées par la Phase A, sans résolution ni justification complémentaire."
        )
    else:
        if llm_decision.coding_rationale:
            reasons.append(llm_decision.coding_rationale)
        if llm_decision.clinical_context:
            reasons.append(llm_decision.clinical_context)
        if llm_decision.fraud_rationale:
            reasons.append(llm_decision.fraud_rationale)
        reasons.extend(llm_decision.reasons)

        unknown_evidence = [
            e for e in llm_decision.clinical_referenced_evidence_ids if e not in clinical_evidence_ids
        ]
        unknown_inconsistencies = [
            i
            for i in llm_decision.clinical_acknowledged_inconsistencies
            if i not in clinical_inconsistency_types
        ]
        if unknown_evidence or unknown_inconsistencies:
            reasons.append(
                "LLM a référencé des preuves ou incohérences cliniques inexistantes — ignorées."
            )

        unknown_signals = [
            s for s in llm_decision.fraud_referenced_signal_types if s not in fraud_signal_types
        ]
        if unknown_signals:
            reasons.append("LLM a référencé des signaux de fraude inexistants — ignorés.")

    evidence_ids = (
        [e.evidence_id for s in clinical_signals_final for e in s.evidence]
        + [e.evidence_id for i in inconsistencies for e in i.evidence]
        + [e.evidence_id for s in fraud_signals_final for e in s.evidence]
    )

    errors: list[StructuredError] = []
    if llm_decision is None:
        errors.append(
            StructuredError(
                code="LLM_UNAVAILABLE",
                message="LLM indisponible ou réponse invalide : résultats déterministes seuls conservés.",
                field="llm_trace",
            )
        )

    confidence = max(0.4, 1.0 - 0.1 * (len(clinical_signals_final) + len(fraud_signals_final)))
    # Plafonnée par la complétude des preuves — une confiance élevée n'a pas
    # de sens sur un dossier dont l'évaluation manque de données, même si le
    # peu de signaux présents ne suggère aucun danger particulier.
    if evidence_completeness is EvidenceCompleteness.INSUFFICIENT:
        confidence = min(confidence, 0.4)
    elif evidence_completeness is EvidenceCompleteness.PARTIAL:
        confidence = min(confidence, 0.7)

    payload = MedicalRiskResultPayload(
        procedure_count=procedure_count,
        medication_count=medication_count,
        codings=final_codings,
        clinical_signals=clinical_signals_final,
        clinical_inconsistencies=inconsistencies,
        fraud_signals=fraud_signals_final,
        duplicate_invoice=duplicate_invoice,
        risk_score=risk_score_final,
        risk_level=risk_level,
        evidence_completeness=evidence_completeness,
        reasons=[r for r in reasons if r],
    )

    return MedicalRiskResult(
        case_id=case_id,
        status=overall_status,
        llm_trace=build_llm_metadata(_AGENT_NAME, confidence=confidence),
        confidence=confidence,
        errors=errors,
        evidence_ids=evidence_ids,
        human_review_required=overall_status is not VerificationStatus.PASS,
        result_payload=payload,
    )


# ── Nœud LangGraph V2 ──────────────────────────────────────────────────────────


def node(state: ClaimStateV2) -> dict:
    """Nœud du graphe V2 — délègue à `run()` et met à jour `ClaimStateV2`.

    Attend dans le state :
        case_id                        : identifiant du dossier
        document_understanding_result  : DocumentUnderstandingResult
        eligibility_result             : EligibilityResult
    """
    case_id: str = state.get("case_id", "")  # type: ignore[assignment]
    document_understanding_result = state.get("document_understanding_result")
    eligibility_result = state.get("eligibility_result")

    ocr_result_like: object | None = None
    medications: list[str] = []
    if document_understanding_result is not None:
        extraction = (
            document_understanding_result.extraction
            if not isinstance(document_understanding_result, dict)
            else document_understanding_result.get("extraction")
        )
        if extraction is not None:
            fields = extraction.fields if hasattr(extraction, "fields") else extraction.get("fields", {})
            confidence_score = (
                extraction.confidence_score
                if hasattr(extraction, "confidence_score")
                else extraction.get("confidence_score")
            )
            ocr_result_like = SimpleNamespace(
                extracted_fields=fields,
                confidence_score=confidence_score,
                document_type=None,
                sha256=None,
            )
            # Câblage best-effort (correctif post-mesure V2-10, AZIZ) :
            # `essential_fields.medical_items` n'est peuplé que par un regex
            # médicament-only (`tools.document_parser._MEDICATION_RE`) — tout
            # élément qui y figure est donc déjà, par construction, classé
            # médicament, jamais un acte. `procedures` reste volontairement
            # vide : aucun discriminant fiable acte/médicament n'existe
            # encore côté extraction (même limite que V1,
            # `graph/input_builders.py::build_coding_input` — jamais une
            # répartition heuristique inventée ici).
            essential_fields = (
                extraction.essential_fields
                if hasattr(extraction, "essential_fields")
                else extraction.get("essential_fields")
            )
            medical_items = (
                getattr(essential_fields, "medical_items", None)
                if essential_fields is not None and not isinstance(essential_fields, dict)
                else (essential_fields or {}).get("medical_items")
            )
            if medical_items:
                for item in medical_items:
                    description = (
                        item.description if hasattr(item, "description") else item.get("description")
                    )
                    if description:
                        medications.append(description)

    identity_like: object | None = None
    if eligibility_result is not None:
        identity = (
            eligibility_result.identity
            if not isinstance(eligibility_result, dict)
            else eligibility_result.get("identity")
        )
        coverage = (
            eligibility_result.coverage
            if not isinstance(eligibility_result, dict)
            else eligibility_result.get("coverage")
        )
        identity_like = SimpleNamespace(identity=identity, coverage=coverage)

    result = run(
        case_id=case_id,
        procedures=[],
        medications=medications,
        ocr_result=ocr_result_like,
        identity_coverage_result=identity_like,
    )

    updates: dict = {
        "medical_risk_result": result,
        "current_step": "medical_risk",
        "completed_steps": ["medical_risk"],
    }
    if result.status is VerificationStatus.FAIL:
        updates["errors"] = [f"[medical_risk] {r}" for r in result.result_payload.reasons]
    elif result.status is VerificationStatus.NEEDS_REVIEW:
        updates["alerts"] = [f"[medical_risk] {r}" for r in result.result_payload.reasons[:5]]

    validate_state_update_v2(updates)
    return updates
