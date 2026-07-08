"""Medical Coding Agent — ClaimShield Santé.

Agent LLM (gemma4:latest via ChatOllama) avec tools SNOMED-CT/RxNorm.

Pipeline :
  Phase A — correspondance déterministe (lookup exact + synonymes)
  Phase B — agent ReAct LLM à chaque exécution pour valider/justifier
  Phase C — fusion et construction de MedicalCodingResult
"""
from __future__ import annotations

import json
import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

try:
    from langgraph.prebuilt import create_react_agent
except ModuleNotFoundError:  # pragma: no cover - dépendance optionnelle en tests locaux
    def create_react_agent(*_args, **_kwargs):
        raise RuntimeError("langgraph indisponible")

from agents.medical_coding_agent.schemas import LlmCodingDecision, LlmResolvedCode, MedicalCodingInput
from agents.medical_coding_agent.tools import rechercher_code
from schemas.domain import VerificationStatus
from schemas.results import AuditEvent, MedicalCodingResult, ProcedureCoding
from state.claim_state import ClaimState, validate_state_update
from tools.medical_coding import (
    code_exists_in_reference,
    code_medications,
    code_procedures,
    compute_global_status,
)
from tools.rule_loader import get_rule_version
from llm.factory import get_llm
from llm.metadata import build_llm_metadata
from llm.prompts import load_prompt

_RULE_VERSION_FILE = "medical_codes.yaml"
_AGENT_NAME = "medical_coding_agent"


def _get_table_version() -> str:
    try:
        return get_rule_version(_RULE_VERSION_FILE)
    except FileNotFoundError:
        return "1.1.0"


def _invoke_llm_react(
    needs_review: list[ProcedureCoding],
    already_coded: list[ProcedureCoding],
) -> LlmCodingDecision | None:
    """Lance l'agent ReAct LLM pour valider/justifier toute codification.

    Les descriptions déjà codées sont transmises au LLM pour justification,
    mais leurs codes restent ceux du référentiel local. Les descriptions en
    NEEDS_REVIEW peuvent recevoir une proposition LLM, acceptée uniquement si
    elle existe dans le référentiel local actif.
    """
    try:
        llm = get_llm()
        agent = create_react_agent(
            model=llm,
            tools=[rechercher_code],
            response_format=LlmCodingDecision,
        )
        data = {
            "needs_review": [c.original_description for c in needs_review],
            "already_coded": [
                {"description": c.original_description, "code": c.proposed_code}
                for c in already_coded
            ],
        }
        result = agent.invoke({
            "messages": [
                SystemMessage(content=load_prompt(_AGENT_NAME)),
                HumanMessage(content=json.dumps(data, ensure_ascii=False)),
            ]
        })
        structured = result.get("structured_response")
        if isinstance(structured, LlmCodingDecision):
            return structured
        if isinstance(structured, dict):
            return LlmCodingDecision(**structured)
        return None
    except Exception:
        return None


def _merge_with_llm(
    initial_codings: list[ProcedureCoding],
    llm_decision: LlmCodingDecision | None,
    section_by_description: dict[str, str],
) -> list[ProcedureCoding]:
    """Fusionne les codifications déterministes avec les résolutions LLM vérifiées."""
    if llm_decision is None:
        return initial_codings

    llm_map: dict[str, LlmResolvedCode] = {
        r.description: r for r in llm_decision.resolved
    }
    merged: list[ProcedureCoding] = []
    for coding in initial_codings:
        if coding.status == VerificationStatus.NEEDS_REVIEW and coding.original_description in llm_map:
            resolved = llm_map[coding.original_description]
            section = section_by_description.get(coding.original_description, "procedures")
            if resolved.proposed_code and code_exists_in_reference(resolved.proposed_code, section):
                merged.append(ProcedureCoding(
                    original_description=coding.original_description,
                    proposed_code=resolved.proposed_code,
                    rule_applied="llm_tool_verified",
                    status=VerificationStatus.PASS,
                    alternatives=coding.alternatives,
                    evidence=[
                        *coding.evidence,
                        "Code LLM accepté car présent et actif dans le référentiel local.",
                        resolved.rationale,
                    ],
                ))
                continue
            if resolved.proposed_code:
                merged.append(coding.model_copy(update={
                    "rule_applied": "llm_rejected_not_in_reference",
                    "evidence": [
                        *coding.evidence,
                        f"Code proposé par LLM rejeté : {resolved.proposed_code}",
                    ],
                }))
                continue
        merged.append(coding)
    return merged


def run(
    case_id: str,
    procedures: list[str] | None = None,
    medications: list[str] | None = None,
) -> MedicalCodingResult:
    """Code les procédures et médicaments du dossier via LLM + table locale.

    Phase A : correspondance déterministe (exact + synonymes).
    Phase B : agent ReAct LLM obligatoire pour validation/justification.
    Phase C : fusion et construction du résultat.
    """
    procedures = procedures or []
    medications = medications or []
    section_by_description = {
        **{description: "procedures" for description in procedures},
        **{description: "medications" for description in medications},
    }

    # ── Phase A : codification déterministe ──────────────────────────────────
    initial_codings: list[ProcedureCoding] = [
        *code_procedures(procedures),
        *code_medications(medications),
    ]

    needs_review = [c for c in initial_codings if c.status == VerificationStatus.NEEDS_REVIEW]
    already_coded = [c for c in initial_codings if c.status == VerificationStatus.PASS]

    # ── Phase B : agent ReAct LLM obligatoire ────────────────────────────────
    llm_metadata = build_llm_metadata(_AGENT_NAME)
    llm_decision = _invoke_llm_react(needs_review, already_coded)
    if llm_decision is None:
        # P1-3 : sans LLM, le résultat est exactement celui de la Phase A
        # seule — jamais un FAIL forcé qui dégraderait activement un
        # dossier dont la Phase A avait déjà PASS/NEEDS_REVIEW. Absence de
        # résolution complémentaire des NEEDS_REVIEW, mais aucune
        # dégradation artificielle des items déjà PASS.
        return MedicalCodingResult(
            case_id=case_id,
            status=compute_global_status(initial_codings),
            codings=initial_codings,
            table_version=_get_table_version(),
            reasons=[
                "LLM_UNAVAILABLE_NO_ADDITIONAL_RESOLUTION : LLM indisponible ou "
                "réponse invalide — résolution complémentaire des items "
                "NEEDS_REVIEW impossible, résultat de la Phase A déterministe "
                "seule conservé tel quel (jamais dégradé artificiellement)."
            ],
            llm_metadata=llm_metadata,
        )

    # ── Phase C : fusion + résultat ───────────────────────────────────────────
    final_codings = _merge_with_llm(initial_codings, llm_decision, section_by_description)

    if not final_codings:
        status = VerificationStatus.NEEDS_REVIEW
        reasons = ["Aucun acte ou médicament fourni pour codification."]
    else:
        status = compute_global_status(final_codings)
        if status == VerificationStatus.PASS:
            rationale = llm_decision.overall_rationale
            reasons = ["Toutes les descriptions ont une correspondance référentielle validée par LLM."]
            if rationale:
                reasons.append(rationale)
        elif status == VerificationStatus.NEEDS_REVIEW:
            unresolved = [c.original_description for c in final_codings if c.status != VerificationStatus.PASS]
            reasons = [
                "Codification incomplète — code non déterminé, revue humaine requise pour : "
                + ", ".join(unresolved[:10])
            ]
            if llm_decision.overall_rationale:
                reasons.append(llm_decision.overall_rationale)
        else:
            reasons = ["Codification échouée."]
            if llm_decision.overall_rationale:
                reasons.append(llm_decision.overall_rationale)

    return MedicalCodingResult(
        case_id=case_id,
        status=status,
        codings=final_codings,
        table_version=_get_table_version(),
        reasons=reasons,
        llm_metadata=llm_metadata,
    )


def node(state: ClaimState) -> dict:
    """Nœud LangGraph du Medical Coding Agent."""
    raw: dict | None = state.get("coding_input")

    if raw is None:
        case_id = str(state.get("case_id", "UNKNOWN"))
        result = MedicalCodingResult(
            case_id=case_id,
            status=VerificationStatus.FAIL,
            codings=[],
            table_version=_get_table_version(),
            reasons=["coding_input absent du state."],
            llm_metadata=build_llm_metadata(_AGENT_NAME),
        )
        updates = _build_updates(case_id, result, fail=True)
        validate_state_update(updates)
        return updates

    try:
        coding_input = MedicalCodingInput(**raw)
    except (ValidationError, TypeError) as exc:
        case_id = str(raw.get("case_id", state.get("case_id", "UNKNOWN")))
        result = MedicalCodingResult(
            case_id=case_id,
            status=VerificationStatus.FAIL,
            codings=[],
            table_version=_get_table_version(),
            reasons=[f"coding_input invalide : {exc}"],
            llm_metadata=build_llm_metadata(_AGENT_NAME),
        )
        updates = _build_updates(case_id, result, fail=True)
        validate_state_update(updates)
        return updates

    result = run(
        case_id=coding_input.case_id,
        procedures=coding_input.procedures,
        medications=coding_input.medications,
    )
    updates = _build_updates(coding_input.case_id, result)
    validate_state_update(updates)
    return updates


def _build_updates(case_id: str, result: MedicalCodingResult, *, fail: bool = False) -> dict:
    audit = AuditEvent(
        event_id=str(uuid.uuid4()),
        case_id=case_id,
        actor=_AGENT_NAME,
        action="medical_coding",
        outcome=result.status.value,
        details={
            "status": result.status.value,
            "coding_count": str(len(result.codings)),
            "table_version": result.table_version,
        },
    )
    updates: dict = {
        "coding_input": None,
        "coding_result": result,
        "current_step": "medical_coding",
        "completed_steps": ["medical_coding"],
        "audit_trail": [audit],
    }
    if result.status == VerificationStatus.FAIL or fail:
        updates["errors"] = [f"[medical_coding] {r}" for r in result.reasons]
    elif result.status == VerificationStatus.NEEDS_REVIEW:
        updates["alerts"] = [f"Codification médicale : NEEDS_REVIEW — {'; '.join(result.reasons)}"]
    return updates
