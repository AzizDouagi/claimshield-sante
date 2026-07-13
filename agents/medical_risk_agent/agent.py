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
    pondération borné DOWNGRADE/NEUTRAL/UPGRADE), `_check_duplicate`,
    seuils `_NEEDS_REVIEW_THRESHOLD`/`_FAIL_THRESHOLD`.
  - Les 3 outils `@tool` déjà autorisés en V1 : `rechercher_code`,
    `verifier_chronologie`, `verifier_doublon`.

`_collect_signals` (clinique et fraude, V1) attendent un objet
`coding_result` avec attributs `.codings`/`.status` — ce module leur passe
un `types.SimpleNamespace` construit depuis `initial_codings` (duck-typing
volontaire, jamais un nouveau schéma dupliqué) : aucune modification des
fonctions V1 réutilisées n'est nécessaire.

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
from agents.fraud_detection_agent.agent import (
    _FAIL_THRESHOLD,
    _NEEDS_REVIEW_THRESHOLD,
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
from schemas.results import ProcedureCoding, StructuredError
from schemas.v2_results import MedicalRiskResult, MedicalRiskResultPayload, RiskLevel
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


def _risk_level_from_score(score: float) -> RiskLevel:
    """Mêmes seuils que `agents.fraud_detection_agent.agent` (V1) — jamais
    recalculés indépendamment."""
    if score >= _FAIL_THRESHOLD:
        return RiskLevel.HIGH
    if score >= _NEEDS_REVIEW_THRESHOLD:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


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
    fraud_status, risk_score = _determine_fraud_status(
        fraud_signals, insufficient_evidence=insufficient_fraud_evidence
    )

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
    if fraud_notes:
        fraud_status_final, risk_score_final = _determine_fraud_status(
            fraud_signals_final, insufficient_evidence=insufficient_fraud_evidence
        )
    else:
        fraud_status_final, risk_score_final = fraud_status, risk_score

    overall_status = _worst(coding_status_final, _worst(clinical_status_final, fraud_status_final))
    risk_level = _risk_level_from_score(risk_score_final)

    reasons: list[str] = list(clinical_reasons)
    reasons.append(duplicate_reason)
    if not final_codings:
        reasons.append("Aucun acte ou médicament fourni pour codification.")
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
        medications=[],
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
