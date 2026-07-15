"""Autonomous Decision Agent (V2) — remplace `case_reviewer_agent` (V1).

Décision finale bornée, 6 issues possibles (`ClaimDecisionV2`), **sans**
verrou « toujours NEEDS_REVIEW/human_review_required=True » (décision AZIZ :
« override asynchrone optionnel » — voir `services/override_store.py`,
Phase V2-8, pour la correction humaine post-décision, hors de ce graphe).

Pipeline :
  Phase A — pré-décision déterministe bornée : reprend `_deterministic_pre_recommendation`
            (V1, `case_reviewer_agent`) adaptée aux 4 résultats V2, et
            `tools.consistency.detect_result_disagreements` (réutilisé tel
            quel, paramétré sur les champs V2 comparables). Calcule
            l'ensemble des décisions *autorisées* pour ce dossier
            (`_allowed_decisions`) — jamais la décision elle-même.
  Phase B — un appel LLM structuré, avec une autorité réelle mais bornée :
            `LlmAutonomousDecision.decision` n'a d'effet que si elle
            appartient à l'ensemble autorisé calculé en Phase A ; sinon
            `_merge_decision` l'ignore et retombe sur un repli déterministe
            conservateur — jamais la valeur hors bornes proposée.
  Phase C — construction de `AutonomousDecisionResult`.

Bornes non contournables (voir `_allowed_decisions`) :
  - `intake_safety.status == BLOCKED` → REJECT forcé, LLM jamais consulté.
  - `intake_safety.status == QUARANTINED` → QUARANTINE forcé, LLM jamais consulté.
  - `medical_risk.risk_level == HIGH` → décision plafonnée à
    {REJECT, QUARANTINE, REQUEST_MORE_INFO} — jamais APPROVE/PARTIAL_APPROVE.
  - `PARTIAL_APPROVE` n'est proposable que si `medical_risk_result.codings`
    contient un mélange réel de PASS et de non-PASS (répartition déjà
    calculée par la Phase A, jamais choisie par le LLM).
  - `TECHNICAL_FAILURE` n'est jamais une valeur choisissable par le LLM —
    réservé exclusivement à l'indisponibilité/invalidité de la Phase B.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.autonomous_decision_agent.prompt import load_autonomous_decision_prompt
from agents.autonomous_decision_agent.schemas import LlmAutonomousDecision
from llm.factory import get_llm
from llm.metadata import build_llm_metadata
from schemas.domain import ClaimDecisionV2, VerificationStatus
from schemas.results import DisagreementPoint, StructuredError
from schemas.v2_results import AutonomousDecisionResult
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

_ALWAYS_ALLOWED_WHEN_UNBLOCKED: frozenset[ClaimDecisionV2] = frozenset(
    {
        ClaimDecisionV2.APPROVE,
        ClaimDecisionV2.REJECT,
        ClaimDecisionV2.REQUEST_MORE_INFO,
    }
)
"""Correctif post-mesure V2-10 (AZIZ) : QUARANTINE retiré de l'ensemble par
défaut — il n'est plus jamais proposable « par erreur » au LLM, réservé
exclusivement aux branches forcées (`intake_safety.status == QUARANTINED`,
`medical_risk.risk_level == CRITICAL`). Un dossier sans danger réel confirmé
ne doit jamais pouvoir atterrir sur QUARANTINE, y compris via le repli de
`_merge_decision`."""

_HIGH_RISK_ALLOWED: frozenset[ClaimDecisionV2] = frozenset(
    {ClaimDecisionV2.REJECT, ClaimDecisionV2.QUARANTINE}
)
"""`risk_level == HIGH` : danger réel mais pas certain — REQUEST_MORE_INFO
volontairement absent (plus d'information ne résout pas un danger déjà
confirmé par au moins un signal réel, voir `medical_risk_agent`)."""

_CRITICAL_RISK_ALLOWED: frozenset[ClaimDecisionV2] = frozenset({ClaimDecisionV2.QUARANTINE})
"""`risk_level == CRITICAL` : danger confirmé (doublon exact ou score de
risque réel au-delà du seuil) — décision forcée, LLM non consulté, même
patron que BLOCKED/QUARANTINED."""

_INSUFFICIENT_EVIDENCE_ALLOWED: frozenset[ClaimDecisionV2] = frozenset(
    {ClaimDecisionV2.REQUEST_MORE_INFO}
)
"""`evidence_completeness == INSUFFICIENT` (et risque réel non HIGH/CRITICAL) :
données manquantes, jamais un danger — décision forcée vers REQUEST_MORE_INFO,
jamais QUARANTINE. Voir `schemas.v2_results.EvidenceCompleteness`."""


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
    PASS/non-PASS existe — jamais un choix arbitraire du LLM."""
    medical_risk = state.get("medical_risk_result")
    payload = getattr(medical_risk, "result_payload", None) if medical_risk else None
    codings = getattr(payload, "codings", None) if payload is not None else None
    if not codings:
        return False
    statuses = {c.status for c in codings}
    return VerificationStatus.PASS in statuses and len(statuses - {VerificationStatus.PASS}) > 0


def _allowed_decisions(
    *,
    intake_status: str | None,
    risk_level: str | None,
    evidence_completeness: str | None,
    has_partial_condition: bool,
) -> tuple[frozenset[ClaimDecisionV2], list[str]]:
    """Calcule, en Python pur, l'ensemble des décisions autorisées pour ce
    dossier — jamais laissé à l'appréciation du LLM.

    Matrice révisée post-mesure V2-10 (AZIZ) — ordre de priorité strict,
    chaque branche court-circuite les suivantes :
      1. intake BLOCKED/QUARANTINED (inchangé, décision technique/sécurité
         déjà prise en amont, LLM jamais consulté) ;
      2. `risk_level == CRITICAL` (danger confirmé) → QUARANTINE forcé ;
      3. `risk_level == HIGH` (danger réel non certain) → {REJECT, QUARANTINE} ;
      4. `evidence_completeness == INSUFFICIENT` (données manquantes, jamais
         un danger réel confirmé à ce stade) → REQUEST_MORE_INFO forcé,
         jamais QUARANTINE ;
      5. par défaut : {APPROVE, REJECT, REQUEST_MORE_INFO} (+ PARTIAL_APPROVE
         si un mélange réel de codings l'autorise) — QUARANTINE n'apparaît
         plus jamais dans cet ensemble par défaut (voir
         `_ALWAYS_ALLOWED_WHEN_UNBLOCKED`).
    """
    bounded_by: list[str] = []

    if intake_status == "BLOCKED":
        bounded_by.append("intake_safety.status == BLOCKED → REJECT forcé, LLM non consulté.")
        return frozenset({ClaimDecisionV2.REJECT}), bounded_by
    if intake_status == "QUARANTINED":
        bounded_by.append("intake_safety.status == QUARANTINED → QUARANTINE forcé, LLM non consulté.")
        return frozenset({ClaimDecisionV2.QUARANTINE}), bounded_by

    if risk_level == "CRITICAL":
        bounded_by.append(
            "medical_risk.risk_level == CRITICAL → QUARANTINE forcé (danger réel confirmé), "
            "LLM non consulté."
        )
        return frozenset(_CRITICAL_RISK_ALLOWED), bounded_by

    if risk_level == "HIGH":
        bounded_by.append(
            "medical_risk.risk_level == HIGH → décision plafonnée à REJECT/QUARANTINE "
            "(danger réel, information supplémentaire non pertinente)."
        )
        return frozenset(_HIGH_RISK_ALLOWED), bounded_by

    if evidence_completeness == "INSUFFICIENT":
        bounded_by.append(
            "medical_risk.evidence_completeness == INSUFFICIENT → REQUEST_MORE_INFO forcé "
            "(données manquantes, jamais un danger réel confirmé), LLM non consulté."
        )
        return frozenset(_INSUFFICIENT_EVIDENCE_ALLOWED), bounded_by

    allowed = set(_ALWAYS_ALLOWED_WHEN_UNBLOCKED)
    if has_partial_condition:
        allowed.add(ClaimDecisionV2.PARTIAL_APPROVE)

    return frozenset(allowed), bounded_by


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


_FALLBACK_PRIORITY: tuple[ClaimDecisionV2, ...] = (
    ClaimDecisionV2.REQUEST_MORE_INFO,
    ClaimDecisionV2.REJECT,
    ClaimDecisionV2.QUARANTINE,
    ClaimDecisionV2.PARTIAL_APPROVE,
)
"""Ordre de repli quand la décision LLM sort de `allowed` — correctif
post-mesure V2-10 (AZIZ) : l'ancien repli systématique vers QUARANTINE
(`fallback = QUARANTINE if QUARANTINE in allowed else next(iter(allowed))`)
biaisait la décision indépendamment même du plafond `risk_level == HIGH`
lui-même. Le nouvel ordre préfère la voie la moins irréversible
(demander plus d'information) puis la plus prudente parmi celles restant
disponibles — jamais APPROVE (qui n'apparaît volontairement pas dans cette
liste : un repli ne peut jamais approuver automatiquement un dossier)."""


def _merge_decision(
    allowed: frozenset[ClaimDecisionV2],
    llm_decision: LlmAutonomousDecision | None,
) -> tuple[ClaimDecisionV2, list[str]]:
    """N'accepte la décision LLM que si elle appartient à `allowed` — sinon
    repli déterministe conservateur (voir `_FALLBACK_PRIORITY`), jamais la
    valeur hors bornes proposée, jamais un repli automatique vers APPROVE."""
    if llm_decision is None:
        return ClaimDecisionV2.TECHNICAL_FAILURE, [
            "LLM indisponible ou réponse invalide — décision impossible sans synthèse."
        ]
    try:
        proposed = ClaimDecisionV2(llm_decision.decision)
    except ValueError:
        proposed = None

    if proposed is not None and proposed in allowed:
        return proposed, []

    fallback = next((d for d in _FALLBACK_PRIORITY if d in allowed), None) or next(iter(allowed))
    return fallback, [
        f"Décision LLM {llm_decision.decision!r} hors des bornes autorisées "
        f"({sorted(d.value for d in allowed)}) pour ce dossier — repli sur {fallback.value}."
    ]


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

    intake_status = snapshot["intake_safety"].get("status")
    risk_level = snapshot["medical_risk"].get("risk_level")
    evidence_completeness = snapshot["medical_risk"].get("evidence_completeness")
    has_partial_condition = _has_partial_approve_condition(decision_state)
    allowed, bounded_by = _allowed_decisions(
        intake_status=str(intake_status) if intake_status else None,
        risk_level=str(risk_level) if risk_level else None,
        evidence_completeness=str(evidence_completeness) if evidence_completeness else None,
        has_partial_condition=has_partial_condition,
    )

    # Jamais consulté si `allowed` est un singleton forcé (BLOCKED/QUARANTINED/
    # CRITICAL/INSUFFICIENT) ; toujours consulté sinon, y compris pour HIGH
    # (deux options restantes : REJECT vs QUARANTINE).
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
                "allowed_decisions": sorted(d.value for d in allowed),
                "instruction": (
                    "Choisis une décision UNIQUEMENT parmi allowed_decisions. Cite "
                    "uniquement des preuves/risques/désaccords déjà fournis — jamais "
                    "une affirmation inventée."
                ),
            }
        )

    if not llm_consulted:
        final_decision = next(iter(allowed))
        merge_notes: list[str] = []
        justification = list(bounded_by)
        errors: list[StructuredError] = []
        confidence = 1.0
    else:
        final_decision, merge_notes = _merge_decision(allowed, llm_decision)
        bounded_by.extend(merge_notes)
        justification: list[str] = []
        if llm_decision is not None:
            justification.append(llm_decision.summary)
            justification.extend(llm_decision.reasons)
        justification.extend(merge_notes)
        errors = (
            [
                StructuredError(
                    code="LLM_UNAVAILABLE",
                    message="LLM indisponible ou réponse invalide — décision TECHNICAL_FAILURE.",
                    field="llm_decision",
                )
            ]
            if llm_decision is None
            else []
        )
        confidence = llm_decision.confidence if llm_decision is not None else 0.0

    if not justification:
        justification.append("Synthèse multi-agent sans motif exploitable.")

    status = _status_for_decision(final_decision)

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
