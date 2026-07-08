"""Case Reviewer Agent — synthèse multi-agent non finale.

Produit une pré-recommandation révisable par un humain à partir des résultats
déjà présents dans le ``ClaimState``. L'agent appelle toujours le LLM pour
obtenir une synthèse structurée ; si le LLM est indisponible ou invalide, la
sortie échoue en fail-closed vers ``PENDING`` avec revue humaine obligatoire.

Interdictions strictes :
  - Aucune décision finale : ``human_review_required`` est toujours forcé à True.
  - Aucun arbitrage inventé : le LLM ne voit qu'un résumé minimisé des résultats.
  - Aucun document brut, texte OCR complet, secret ou chemin de fichier.

P1-4 — auto-approbation bornée (mécanisme additif, verrou intact)
-------------------------------------------------------------------
``CaseReviewerResult.status``/``human_review_required`` restent verrouillés
à ``NEEDS_REVIEW``/``True`` dans TOUS les cas — ce module n'y touche jamais.
Un signal additionnel non verrouillé,
``CaseReviewerResultPayload.auto_decision``, peut valoir
``"AUTO_APPROVED_LOW_RISK"`` quand une conjonction stricte de critères est
réunie (``_auto_decision_eligibility``) : pré-recommandation Phase A APPROVE
(donc hors-périmètre et désaccord inter-agents déjà exclus, seule condition
de ``_deterministic_pre_recommendation`` pour retourner APPROVE), LLM
disponible et recommandant lui aussi APPROVE, LLM ne signalant
aucune escalade (``escalation_required=False``), confiance LLM au moins
égale à ``Settings.claimshield_auto_approve_confidence_threshold``, aucun
risque ni désaccord détecté. Seul ``graph/edges.py::route_review``, informé
de ce signal, peut router directement vers ``end`` sans passer par
``needs_review`` — et ce chemin traverse quand même ``audit`` avant
``finalize`` (voir le ``path_map`` de ``graph/workflow.py``), jamais un
contournement de l'audit. Un consommateur qui ignore ce champ reste en
sécurité maximale (comportement inchangé, toujours ``needs_review``).
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Protocol, runtime_checkable

from langchain_core.messages import HumanMessage, SystemMessage

from agents.case_reviewer_agent.prompt import load_case_reviewer_prompt
from agents.case_reviewer_agent.schemas import LlmCaseReviewDecision
from config.logging import get_logger
from config.settings import get_settings
from llm.factory import get_llm
from llm.metadata import build_llm_metadata
from schemas.domain import Recommendation, VerificationStatus
from schemas.results import (
    AuditEvent,
    CaseReviewerResult,
    CaseReviewerResultPayload,
    DisagreementPoint,
    StructuredError,
)
from state.claim_state import ClaimState, validate_state_update
from tools.consistency import detect_result_disagreements

_STEP_NAME = "case_reviewer"
_AGENT_NAME = "case_reviewer_agent"

logger = get_logger(__name__)

_EXPECTED_UPSTREAM_AGENTS: tuple[str, ...] = (
    "claim_intake",
    "security_gate",
    "privacy",
    "identity_coverage",
    "fhir_validator",
    "document_ocr",
    "medical_coding",
    "clinical_consistency",
    "fraud_detection",
)
_REVIEW_STATUSES = {"NEEDS_REVIEW", "PENDING", "NOT_EVALUATED"}
_BLOCKING_INTAKE_STATUSES = {"blocked", "quarantined", "error"}
_BLOCKING_SECURITY_DECISIONS = {"BLOCK", "QUARANTINE"}


# ── Interface ──────────────────────────────────────────────────────────────────


@runtime_checkable
class CaseReviewerRunnable(Protocol):
    """Interface minimale requise par le nœud LangGraph."""

    def run(self, state: ClaimState) -> CaseReviewerResult: ...


# ── Synthèse déterministe minimisée ───────────────────────────────────────────


def _value(value: object | None) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return str(raw)


def _upper(value: object | None) -> str | None:
    raw = _value(value)
    return raw.upper() if raw is not None else None


def _result_status(result: object | None) -> str | None:
    return _upper(getattr(result, "status", None)) if result is not None else None


def _count_items(value: object | None) -> int:
    return len(value) if isinstance(value, (list, tuple, dict, set)) else 0


def _build_agent_snapshot(state: ClaimState) -> dict[str, dict[str, object]]:
    """Construit un résumé sûr des résultats agents, sans contenu métier brut."""
    intake = state.get("intake_result")
    security = state.get("security_result")
    privacy = state.get("privacy_result")
    identity_coverage = state.get("identity_coverage_result")
    fhir = state.get("fhir_result")
    ocr = state.get("ocr_result")
    coding = state.get("coding_result")
    clinical = state.get("clinical_result")
    fraud = state.get("fraud_result")

    identity = getattr(identity_coverage, "identity", None) if identity_coverage else None
    coverage = getattr(identity_coverage, "coverage", None) if identity_coverage else None

    return {
        "claim_intake": {
            "present": intake is not None,
            "status": _value(getattr(intake, "status", None)),
            "accepted_count": getattr(intake, "accepted_count", None),
            "quarantined_count": getattr(intake, "quarantined_count", None),
            "error_count": getattr(intake, "error_count", None),
        },
        "security_gate": {
            "present": security is not None,
            "decision": _upper(getattr(security, "decision", None)),
            "finding_count": _count_items(getattr(security, "findings", None)),
        },
        "privacy": {
            "present": privacy is not None,
            "status": _result_status(privacy),
            "reason_code_count": _count_items(getattr(privacy, "reason_codes", None)),
        },
        "identity_coverage": {
            "present": identity_coverage is not None,
            "identity_status": _result_status(identity),
            "coverage_status": _result_status(coverage),
            "ceiling_exceeded": bool(getattr(coverage, "ceiling_exceeded", False)),
            "preauthorization_required": bool(
                getattr(coverage, "preauthorization_required", False)
            ),
        },
        "fhir_validator": {
            "present": fhir is not None,
            "status": _result_status(fhir),
            "resource_count": getattr(fhir, "resource_count", None),
            "error_count": _count_items(getattr(fhir, "errors", None)),
            "warning_count": _count_items(getattr(fhir, "warnings", None)),
        },
        "document_ocr": {
            "present": ocr is not None,
            "status": _result_status(ocr),
            "extraction_status": _upper(getattr(ocr, "extraction_status", None)),
            "document_type": _value(getattr(ocr, "document_type", None)),
            "confidence_score": getattr(ocr, "confidence_score", None),
            "human_review_required": bool(
                getattr(ocr, "human_review_required", False)
            ),
        },
        "medical_coding": {
            "present": coding is not None,
            "status": _result_status(coding),
            "coding_count": _count_items(getattr(coding, "codings", None)),
        },
        "clinical_consistency": {
            "present": clinical is not None,
            "status": _result_status(clinical),
            "signal_count": _count_items(
                getattr(getattr(clinical, "result_payload", None), "signals", None)
            ),
            "inconsistency_count": _count_items(
                getattr(getattr(clinical, "result_payload", None), "inconsistencies", None)
            ),
        },
        "fraud_detection": {
            "present": fraud is not None,
            "status": _result_status(fraud),
            "risk_score": getattr(getattr(fraud, "result_payload", None), "risk_score", None),
            "signal_count": _count_items(
                getattr(getattr(fraud, "result_payload", None), "signals", None)
            ),
            "duplicate_invoice": getattr(
                getattr(fraud, "result_payload", None), "duplicate_invoice", None
            ),
        },
    }


def _collect_disagreements(state: ClaimState) -> list[DisagreementPoint]:
    return list(detect_result_disagreements(state))


def _collect_risks(snapshot: dict[str, dict[str, object]]) -> list[str]:
    """Dérive des signaux de risque à porter à l'attention de l'humain.

    Toujours calculés à partir de données déjà produites par les agents amont
    (jamais une affirmation inventée ici) — voir ``_build_agent_snapshot``.
    """
    risks: list[str] = []

    identity_coverage = snapshot["identity_coverage"]
    if identity_coverage.get("ceiling_exceeded"):
        risks.append("Plafond de couverture dépassé.")
    if identity_coverage.get("preauthorization_required"):
        risks.append("Pré-autorisation requise — à confirmer par l'humain.")

    fraud = snapshot["fraud_detection"]
    risk_score = fraud.get("risk_score")
    if isinstance(risk_score, (int, float)) and risk_score >= 0.7:
        risks.append(f"Score de risque de fraude élevé ({risk_score:.2f}).")
    if fraud.get("duplicate_invoice") is True:
        risks.append("Facture potentiellement en doublon.")

    clinical = snapshot["clinical_consistency"]
    inconsistency_count = clinical.get("inconsistency_count") or 0
    if isinstance(inconsistency_count, int) and inconsistency_count > 0:
        risks.append(f"{inconsistency_count} incohérence(s) clinique(s) détectée(s).")

    return risks


def _collect_evidence_ids(state: ClaimState) -> list[str]:
    """Agrège les identifiants de preuves déjà validées par les agents amont.

    Ne construit jamais de nouvel identifiant — reprend tel quel
    ``evidence_ids`` de ``ClinicalConsistencyResult``/``FraudDetectionResult``,
    seuls résultats amont à porter des objets de preuve structurés.
    """
    ids: list[str] = []
    for key in ("clinical_result", "fraud_result"):
        result = state.get(key)
        if result is not None:
            ids.extend(getattr(result, "evidence_ids", None) or [])
    return ids


def _deterministic_pre_recommendation(
    snapshot: dict[str, dict[str, object]],
    disagreements: list[DisagreementPoint],
    *,
    error_count: int,
    alert_count: int,
) -> tuple[Recommendation, list[str]]:
    """Borne la pré-recommandation avant LLM, sans remplacer le LLM."""
    reasons: list[str] = []

    intake_status = str(snapshot["claim_intake"].get("status") or "")
    if intake_status in _BLOCKING_INTAKE_STATUSES:
        reasons.append(f"Ingestion bloquante : statut {intake_status}.")

    security_decision = str(snapshot["security_gate"].get("decision") or "")
    if security_decision in _BLOCKING_SECURITY_DECISIONS:
        reasons.append(f"Security Gate bloquant : décision {security_decision}.")

    for agent_name, data in snapshot.items():
        for field_name in ("status", "identity_status", "coverage_status"):
            if data.get(field_name) == "FAIL":
                reasons.append(f"{agent_name}.{field_name}=FAIL.")

    if error_count:
        reasons.append(f"{error_count} erreur(s) bloquante(s) déjà présentes dans le state.")

    if reasons:
        return Recommendation.REJECT, reasons

    if disagreements:
        reasons.append(f"{len(disagreements)} désaccord(s) inter-agents à arbitrer.")

    missing_agents = [
        agent_name
        for agent_name in _EXPECTED_UPSTREAM_AGENTS
        if not snapshot[agent_name].get("present")
    ]
    if missing_agents:
        reasons.append("Résultats agents manquants : " + ", ".join(missing_agents) + ".")

    for agent_name, data in snapshot.items():
        for field_name in ("status", "identity_status", "coverage_status"):
            if data.get(field_name) in _REVIEW_STATUSES:
                reasons.append(f"{agent_name}.{field_name}={data[field_name]}.")
        if data.get("extraction_status") in _REVIEW_STATUSES:
            reasons.append(
                f"{agent_name}.extraction_status={data['extraction_status']}."
            )
        if data.get("human_review_required") is True:
            reasons.append(f"{agent_name} demande une revue humaine.")

    if alert_count:
        reasons.append(f"{alert_count} alerte(s) non bloquante(s) à relire.")

    if reasons:
        return Recommendation.PENDING, reasons

    return Recommendation.APPROVE, [
        "Tous les résultats agents disponibles sont compatibles avec une pré-approbation."
    ]


def _merge_recommendation(
    deterministic: Recommendation,
    llm_recommendation: Recommendation | None,
) -> Recommendation:
    """Le LLM peut durcir une pré-recommandation, jamais assouplir un rejet."""
    if llm_recommendation is None:
        return Recommendation.PENDING
    if deterministic is Recommendation.REJECT:
        return Recommendation.REJECT
    if deterministic is Recommendation.PENDING:
        return Recommendation.REJECT if llm_recommendation is Recommendation.REJECT else Recommendation.PENDING
    return llm_recommendation


_AUTO_APPROVE_LABEL = "AUTO_APPROVED_LOW_RISK"


def _auto_decision_eligibility(
    deterministic_recommendation: Recommendation,
    llm_decision: LlmCaseReviewDecision | None,
    risks: list[str],
    disagreements: list[DisagreementPoint],
) -> tuple[str | None, list[str]]:
    """P1-4 — critères d'éligibilité à l'auto-approbation bornée.

    Conjonction stricte, calculée entièrement en Python — jamais laissée à
    la seule appréciation du LLM. Chaque critère répond à l'un des quatre
    motifs d'escalade attendus (incertitude LLM, confiance basse, désaccord
    inter-agents, hors périmètre) :

      1. Phase A déterministe conclut déjà APPROVE
         (``_deterministic_pre_recommendation`` ne retourne APPROVE que si
         aucun motif bloquant/de revue n'a été collecté — son unique
         « raison » dans ce cas est un message récapitulatif, jamais un
         motif d'escalade ; un désaccord détecté par
         ``detect_result_disagreements`` empêche déjà APPROVE en amont).
      2. Le LLM est disponible (pas d'incertitude d'indisponibilité).
      3. Le LLM recommande lui aussi APPROVE.
      4. Le LLM ne signale explicitement aucune escalade
         (``escalation_required is False`` — pas d'incertitude explicite).
      5. La confiance déclarée par le LLM atteint le seuil configuré (pas de
         confiance basse).
      6. Aucun risque ni désaccord détecté par la Phase A (redondant avec
         (1) en pratique, revérifié explicitement par prudence).

    Ne modifie jamais ``status``/``human_review_required`` — l'appelant
    (``run()``) ne fait que poser ce signal sur
    ``CaseReviewerResultPayload.auto_decision``, un champ non verrouillé.
    """
    criteria: list[str] = []

    if deterministic_recommendation is not Recommendation.APPROVE:
        return None, []
    criteria.append("Pré-recommandation déterministe APPROVE.")

    if llm_decision is None:
        return None, []
    criteria.append("Synthèse LLM disponible.")

    if llm_decision.recommendation is not Recommendation.APPROVE:
        return None, []
    criteria.append("Recommandation LLM APPROVE.")

    if llm_decision.escalation_required:
        return None, []
    criteria.append("LLM ne signale aucune escalade requise.")

    threshold = get_settings().claimshield_auto_approve_confidence_threshold
    if llm_decision.confidence < threshold:
        return None, []
    criteria.append(f"Confiance LLM {llm_decision.confidence:.2f} ≥ seuil {threshold:.2f}.")

    if risks:
        return None, []
    criteria.append("Aucun risque détecté par la Phase A.")

    if disagreements:
        return None, []
    criteria.append("Aucun désaccord inter-agents détecté.")

    return _AUTO_APPROVE_LABEL, criteria


def _confidence(snapshot: dict[str, dict[str, object]], recommendation: Recommendation) -> float:
    present_count = sum(1 for data in snapshot.values() if data.get("present"))
    ratio = present_count / len(_EXPECTED_UPSTREAM_AGENTS)
    base = max(0.3, min(0.95, ratio))
    if recommendation is Recommendation.PENDING:
        return min(base, 0.65)
    if recommendation is Recommendation.REJECT:
        return min(base, 0.8)
    return base


def _disagreement_id(point: DisagreementPoint) -> str:
    return f"{point.agent}.{point.field}"


# ── Phase B : LLM ─────────────────────────────────────────────────────────────


def _invoke_llm_case_review(data: dict[str, Any]) -> LlmCaseReviewDecision | None:
    """Demande au LLM une synthèse structurée, jamais une décision finale."""
    try:
        prompt = load_case_reviewer_prompt()
        llm = get_llm()
        structured = llm.with_structured_output(
            LlmCaseReviewDecision,
            method="json_schema",
        )
        result = structured.invoke(
            [
                SystemMessage(content=prompt.system_prompt),
                HumanMessage(content=json.dumps(data, ensure_ascii=False, default=str)),
            ]
        )
        if isinstance(result, LlmCaseReviewDecision):
            return result
        if isinstance(result, dict):
            return LlmCaseReviewDecision(**result)
        return None
    except Exception:
        return None


def _merge_llm_decision(
    llm_decision: LlmCaseReviewDecision | None,
    *,
    known_evidence_ids: set[str],
    known_risks: set[str],
    known_disagreement_ids: set[str],
) -> tuple[Recommendation | None, list[str], list[str]]:
    """Fusionne la décision LLM — jamais dans ``status``/``human_review_required``,
    verrouillés au niveau du schéma (voir ``CaseReviewerResult``).

    Toute preuve (``referenced_evidence_ids``), risque
    (``acknowledged_risks``) ou contradiction (``acknowledged_disagreements``)
    citée par le LLM mais absente des valeurs réellement calculées par la
    Phase A est silencieusement ignorée — jamais une affirmation non prouvée
    acceptée telle quelle.
    """
    justification: list[str] = []
    human_review_reasons: list[str] = [
        "Validation humaine obligatoire avant toute décision finale."
    ]

    if llm_decision is None:
        justification.append(
            "LLM indisponible ou réponse invalide : pré-recommandation mise en attente."
        )
        human_review_reasons.append("Synthèse LLM absente ou invalide.")
        return None, justification, human_review_reasons

    justification.append(llm_decision.summary)
    justification.extend(llm_decision.reasons)
    human_review_reasons.extend(llm_decision.human_review_reasons)

    unknown_evidence = [
        e for e in llm_decision.referenced_evidence_ids if e not in known_evidence_ids
    ]
    unknown_risks = [r for r in llm_decision.acknowledged_risks if r not in known_risks]
    unknown_disagreements = [
        d for d in llm_decision.acknowledged_disagreements if d not in known_disagreement_ids
    ]
    if unknown_evidence or unknown_risks or unknown_disagreements:
        justification.append(
            "LLM a référencé des preuves, risques ou contradictions inexistants — "
            "références ignorées (aucune affirmation non prouvée acceptée)."
        )

    return llm_decision.recommendation, justification, human_review_reasons


# ── Fonction principale (testable sans LangGraph) ─────────────────────────────


def run(case_id: str, state: ClaimState | None = None) -> CaseReviewerResult:
    """Synthétise les résultats agents et retourne une pré-recommandation."""
    review_state: ClaimState = dict(state or {})
    review_state["case_id"] = case_id

    snapshot = _build_agent_snapshot(review_state)
    disagreements = _collect_disagreements(review_state)
    risks = _collect_risks(snapshot)
    evidence_ids = _collect_evidence_ids(review_state)
    disagreement_ids = [_disagreement_id(point) for point in disagreements]
    deterministic_recommendation, deterministic_reasons = _deterministic_pre_recommendation(
        snapshot,
        disagreements,
        error_count=len(review_state.get("errors", []) or []),
        alert_count=len(review_state.get("alerts", []) or []),
    )

    llm_decision = _invoke_llm_case_review(
        {
            "case_id": case_id,
            "agent_results": snapshot,
            "disagreements": [d.model_dump(mode="json") for d in disagreements],
            "disagreement_ids": disagreement_ids,
            "risks": risks,
            "evidence_ids": evidence_ids,
            "deterministic_pre_recommendation": deterministic_recommendation.value,
            "deterministic_reasons": deterministic_reasons[:10],
            "instruction": (
                "Synthèse explicable uniquement : cite les preuves, risques et "
                "contradictions fournis, pose les questions nécessaires à l'humain, "
                "recommandation non finale, revue humaine obligatoire, aucun "
                "résultat inventé."
            ),
        }
    )

    llm_recommendation, justification, human_review_reasons = _merge_llm_decision(
        llm_decision,
        known_evidence_ids=set(evidence_ids),
        known_risks=set(risks),
        known_disagreement_ids=set(disagreement_ids),
    )
    errors: list[StructuredError] = (
        [
            StructuredError(
                code="LLM_UNAVAILABLE",
                message=(
                    "Synthèse LLM indisponible ou réponse invalide — "
                    "pré-recommandation mise en attente."
                ),
                field="llm_decision",
            )
        ]
        if llm_decision is None
        else []
    )

    final_recommendation = _merge_recommendation(
        deterministic_recommendation,
        llm_recommendation,
    )
    justification.extend(deterministic_reasons)
    if not justification:
        justification.append("Synthèse multi-agent sans motif exploitable : revue humaine requise.")

    if final_recommendation is Recommendation.PENDING and not any(
        "attente" in reason.casefold() or "manquant" in reason.casefold()
        for reason in human_review_reasons
    ):
        human_review_reasons.append("Pré-recommandation en attente d'arbitrage humain.")

    confidence = _confidence(snapshot, final_recommendation)
    if llm_decision is None:
        confidence = 0.2

    auto_decision, auto_decision_criteria = _auto_decision_eligibility(
        deterministic_recommendation,
        llm_decision,
        risks,
        disagreements,
    )
    if auto_decision == _AUTO_APPROVE_LABEL:
        # P1-4/P3-2 : point de décision autonome — journalisé pour
        # traçabilité opérationnelle (échantillonnage d'audit), en plus de
        # l'alerte déjà ajoutée à state["alerts"] par make_node().
        logger.info(
            "case_reviewer_auto_approved",
            case_id=case_id,
            criteria_count=len(auto_decision_criteria),
        )

    payload = CaseReviewerResultPayload(
        recommendation=final_recommendation,
        justification=justification,
        disagreements=disagreements,
        risks=risks,
        human_review_reasons=human_review_reasons,
        auto_decision=auto_decision,
        auto_decision_criteria=auto_decision_criteria,
    )

    return CaseReviewerResult(
        case_id=case_id,
        status=VerificationStatus.NEEDS_REVIEW,
        llm_trace=build_llm_metadata(_AGENT_NAME, confidence=confidence),
        confidence=confidence,
        errors=errors,
        evidence_ids=evidence_ids,
        human_review_required=True,
        result_payload=payload,
    )


# ── Implémentation par défaut (réelle) ────────────────────────────────────────


class _RealImplementation:
    """Adapte ``run()`` à l'interface ``CaseReviewerRunnable``."""

    def run(self, state: ClaimState) -> CaseReviewerResult:
        case_id = str(state.get("case_id", "UNKNOWN"))
        return run(case_id, state)


_DEFAULT_IMPL: CaseReviewerRunnable = _RealImplementation()


# ── Factory et nœud LangGraph ─────────────────────────────────────────────────


def _force_human_review(result: CaseReviewerResult) -> CaseReviewerResult:
    """Défense en profondeur côté nœud.

    Le schéma ``CaseReviewerResult`` verrouille déjà ``status=NEEDS_REVIEW`` et
    ``human_review_required=True`` (aucune instance valide ne peut porter
    d'autre valeur) — cette fonction reste néanmoins nécessaire pour garantir
    la présence du motif standard dans ``human_review_reasons``, même si une
    implémentation injectée omet de l'y placer elle-même.
    """
    payload = result.result_payload
    reasons = list(payload.human_review_reasons)
    if "Validation humaine obligatoire avant toute décision finale." not in reasons:
        reasons.insert(0, "Validation humaine obligatoire avant toute décision finale.")
    if reasons == payload.human_review_reasons:
        return result
    new_payload = payload.model_copy(update={"human_review_reasons": reasons})
    return result.model_copy(update={"result_payload": new_payload})


def make_node(
    impl: CaseReviewerRunnable = _DEFAULT_IMPL,
) -> Callable[[ClaimState], dict]:
    """Crée un nœud LangGraph avec l'implémentation injectable fournie."""

    def _node(state: ClaimState) -> dict:
        result = _force_human_review(impl.run(state))
        payload = result.result_payload
        case_id = str(state.get("case_id", result.case_id))
        llm_call_id = str(uuid.uuid4())
        audit = AuditEvent(
            event_id=str(uuid.uuid4()),
            case_id=case_id,
            actor=_AGENT_NAME,
            action="case_review",
            outcome=payload.recommendation.value,
            details={
                "status": result.status.value,
                "human_review_required": str(result.human_review_required),
                "disagreement_count": str(len(payload.disagreements)),
                "justification_count": str(len(payload.justification)),
                "risk_count": str(len(payload.risks)),
                "evidence_ids": ",".join(result.evidence_ids),
                "llm_call_id": llm_call_id,
                "model_name": result.llm_trace.model_name,
                "prompt_version": result.llm_trace.prompt_version,
                "errors": ",".join(e.code for e in result.errors),
            },
        )
        alerts = [
            f"Revue dossier : {payload.recommendation.value} non finale — "
            f"{'; '.join(payload.human_review_reasons)}"
        ]
        if payload.auto_decision == _AUTO_APPROVE_LABEL:
            # P1-4 — traçabilité renforcée : rend les dossiers auto-approuvés
            # trivialement requêtables a posteriori (échantillonnage d'audit),
            # sans changer le statut/human_review_required (toujours verrouillés).
            alerts.append(
                f"{_AUTO_APPROVE_LABEL} : dossier approuvé sans revue humaine — "
                f"critères : {'; '.join(payload.auto_decision_criteria)}"
            )
        updates: dict = {
            "review_result": result,
            "final_recommendation": payload.recommendation,
            "final_justification": list(payload.justification),
            "current_step": _STEP_NAME,
            "completed_steps": [_STEP_NAME],
            "audit_trail": [audit],
            "alerts": alerts,
        }
        if payload.recommendation is Recommendation.REJECT:
            updates["errors"] = [
                f"[{_AGENT_NAME}] Pré-recommandation de rejet — "
                f"{'; '.join(payload.justification)}"
            ]
        validate_state_update(updates)
        return updates

    _node.__name__ = f"node_{_STEP_NAME}"
    return _node


# Nœud stable — nom utilisé comme clé dans le StateGraph.
node = make_node()
