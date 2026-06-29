"""Medical Coding Agent — ClaimShield Santé.

Code les actes médicaux et médicaments extraits à partir d'une table locale
versionnée. Agent déterministe : aucun appel LLM, aucun effet de bord.
"""
from __future__ import annotations

import uuid

from pydantic import ValidationError

from agents.medical_coding_agent.schemas import MedicalCodingInput
from schemas.domain import VerificationStatus
from schemas.results import AuditEvent, MedicalCodingResult
from state.claim_state import ClaimState, validate_state_update
from tools.medical_coding import code_medications, code_procedures, compute_global_status
from tools.rule_loader import get_rule_version

_RULE_VERSION_FILE = "medical_codes.yaml"


def _get_table_version() -> str:
    """Retourne la version de la table de codes."""
    try:
        return get_rule_version(_RULE_VERSION_FILE)
    except FileNotFoundError:
        return "1.0.0"


def run(
    case_id: str,
    procedures: list[str] | None = None,
    medications: list[str] | None = None,
) -> MedicalCodingResult:
    """Code les procédures et médicaments du dossier.

    Args:
        case_id: Identifiant du dossier.
        procedures: Descriptions d'actes à coder.
        medications: Descriptions de médicaments à coder.

    Returns:
        MedicalCodingResult avec un statut global PASS/NEEDS_REVIEW/FAIL.
    """
    procedures = procedures or []
    medications = medications or []

    codings = [
        *code_procedures(procedures),
        *code_medications(medications),
    ]
    status = compute_global_status(codings)

    reasons: list[str] = []
    if not codings:
        reasons.append("Aucun acte ou médicament fourni pour codification")
    elif status == VerificationStatus.PASS:
        reasons.append("Toutes les descriptions ont une correspondance exacte")
    elif status == VerificationStatus.NEEDS_REVIEW:
        unresolved = [c.original_description for c in codings if c.status != VerificationStatus.PASS]
        reasons.append(
            "Codification incomplète — revue humaine requise pour : "
            + ", ".join(unresolved[:10])
        )
    else:
        reasons.append("Codification échouée")

    return MedicalCodingResult(
        case_id=case_id,
        status=status,
        codings=codings,
        table_version=_get_table_version(),
        reasons=reasons,
    )


def node(state: ClaimState) -> dict:
    """Nœud LangGraph du Medical Coding Agent.

    Lit coding_input, écrit coding_result, puis consomme coding_input.
    """
    raw: dict | None = state.get("coding_input")

    if raw is None:
        case_id = str(state.get("case_id", "UNKNOWN"))
        result = MedicalCodingResult(
            case_id=case_id,
            status=VerificationStatus.FAIL,
            codings=[],
            table_version=_get_table_version(),
            reasons=["coding_input absent du state"],
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
    """Construit le dict de mise à jour ClaimState."""
    audit_event = AuditEvent(
        event_id=str(uuid.uuid4()),
        case_id=case_id,
        actor="medical_coding_agent",
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
        "audit_trail": [audit_event],
    }
    if result.status == VerificationStatus.FAIL or fail:
        updates["errors"] = [f"[medical_coding] {r}" for r in result.reasons]
    elif result.status == VerificationStatus.NEEDS_REVIEW:
        updates["alerts"] = [f"Codification médicale : NEEDS_REVIEW — {'; '.join(result.reasons)}"]
    return updates
