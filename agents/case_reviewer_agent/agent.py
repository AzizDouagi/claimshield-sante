"""Case Reviewer Agent — synthèse multi-agent non finale.

Produit une pré-recommandation révisable par un humain à partir des résultats
déjà présents dans le ``ClaimState``. L'agent appelle toujours le LLM pour
obtenir une synthèse structurée ; si le LLM est indisponible ou invalide, la
sortie échoue en fail-closed vers ``PENDING`` avec revue humaine obligatoire.

Interdictions strictes :
  - Aucune décision finale : ``human_review_required`` est toujours forcé à True.
  - Aucun arbitrage inventé : le LLM ne voit qu'un résumé minimisé des résultats.
  - Aucun document brut, texte OCR complet, secret ou chemin de fichier.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Protocol, runtime_checkable

from langchain_core.messages import HumanMessage, SystemMessage

from agents.case_reviewer_agent.prompt import load_case_reviewer_prompt
from agents.case_reviewer_agent.schemas import LlmCaseReviewDecision
from llm.factory import get_llm
from llm.metadata import build_llm_metadata
from schemas.domain import Recommendation
from schemas.results import AuditEvent, CaseReviewerResult, DisagreementPoint
from state.claim_state import ClaimState, validate_state_update
from tools.consistency import detect_result_disagreements

_STEP_NAME = "case_reviewer"
_AGENT_NAME = "case_reviewer_agent"

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


def _confidence(snapshot: dict[str, dict[str, object]], recommendation: Recommendation) -> float:
    present_count = sum(1 for data in snapshot.values() if data.get("present"))
    ratio = present_count / len(_EXPECTED_UPSTREAM_AGENTS)
    base = max(0.3, min(0.95, ratio))
    if recommendation is Recommendation.PENDING:
        return min(base, 0.65)
    if recommendation is Recommendation.REJECT:
        return min(base, 0.8)
    return base


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


# ── Fonction principale (testable sans LangGraph) ─────────────────────────────


def run(case_id: str, state: ClaimState | None = None) -> CaseReviewerResult:
    """Synthétise les résultats agents et retourne une pré-recommandation."""
    review_state: ClaimState = dict(state or {})
    review_state["case_id"] = case_id

    snapshot = _build_agent_snapshot(review_state)
    disagreements = _collect_disagreements(review_state)
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
            "deterministic_pre_recommendation": deterministic_recommendation.value,
            "deterministic_reasons": deterministic_reasons[:10],
            "instruction": (
                "Synthèse explicable uniquement : recommandation non finale, "
                "revue humaine obligatoire et aucun résultat inventé."
            ),
        }
    )

    llm_recommendation: Recommendation | None = None
    justification: list[str] = []
    human_review_reasons: list[str] = [
        "Validation humaine obligatoire avant toute décision finale."
    ]

    if llm_decision is None:
        justification.append(
            "LLM indisponible ou réponse invalide : pré-recommandation mise en attente."
        )
        human_review_reasons.append("Synthèse LLM absente ou invalide.")
    else:
        llm_recommendation = llm_decision.recommendation
        justification.append(llm_decision.summary)
        justification.extend(llm_decision.reasons)
        human_review_reasons.extend(llm_decision.human_review_reasons)

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

    return CaseReviewerResult(
        case_id=case_id,
        recommendation=final_recommendation,
        justification=justification,
        disagreements=disagreements,
        human_review_required=True,
        human_review_reasons=human_review_reasons,
        llm_metadata=build_llm_metadata(_AGENT_NAME, confidence=confidence),
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
    reasons = list(result.human_review_reasons)
    if "Validation humaine obligatoire avant toute décision finale." not in reasons:
        reasons.insert(0, "Validation humaine obligatoire avant toute décision finale.")
    if result.human_review_required and reasons == result.human_review_reasons:
        return result
    return result.model_copy(
        update={
            "human_review_required": True,
            "human_review_reasons": reasons,
        }
    )


def make_node(
    impl: CaseReviewerRunnable = _DEFAULT_IMPL,
) -> Callable[[ClaimState], dict]:
    """Crée un nœud LangGraph avec l'implémentation injectable fournie."""

    def _node(state: ClaimState) -> dict:
        result = _force_human_review(impl.run(state))
        case_id = str(state.get("case_id", result.case_id))
        audit = AuditEvent(
            event_id=str(uuid.uuid4()),
            case_id=case_id,
            actor=_AGENT_NAME,
            action="case_review",
            outcome=result.recommendation.value,
            details={
                "human_review_required": str(result.human_review_required),
                "disagreement_count": str(len(result.disagreements)),
                "justification_count": str(len(result.justification)),
            },
        )
        updates: dict = {
            "review_result": result,
            "final_recommendation": result.recommendation,
            "final_justification": list(result.justification),
            "current_step": _STEP_NAME,
            "completed_steps": [_STEP_NAME],
            "audit_trail": [audit],
            "alerts": [
                f"Revue dossier : {result.recommendation.value} non finale — "
                f"{'; '.join(result.human_review_reasons)}"
            ],
        }
        if result.recommendation is Recommendation.REJECT:
            updates["errors"] = [
                f"[{_AGENT_NAME}] Pré-recommandation de rejet — "
                f"{'; '.join(result.justification)}"
            ]
        validate_state_update(updates)
        return updates

    _node.__name__ = f"node_{_STEP_NAME}"
    return _node


# Nœud stable — nom utilisé comme clé dans le StateGraph.
node = make_node()
