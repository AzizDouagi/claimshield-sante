"""État partagé du workflow LangGraph V2 — state/claim_state_v2.py.

`ClaimStateV2` est le seul objet qui traverse les nœuds de `graph/workflow_v2.py`
(plan de refonte V2, Phase V2-1). Distinct de `state.claim_state.ClaimState`
(V1, non modifié) — coexistence stricte, voir §0 du plan.

Différences volontaires avec `ClaimState` (V1) :
  - 5 champs `*_result` (un par agent fusionné) au lieu de 11 ;
  - un seul champ d'entrée consommé, `intake_input` (point d'entrée du
    pipeline) — les agents suivants lisent directement le `*_result` de
    l'agent précédent, jamais un `*_input` intermédiaire reconstruit ;
  - aucun `human_decision` : le graphe V2 ne bloque jamais (pas
    d'`interrupt()`), les corrections humaines vivent dans
    `services.override_store.OverrideStore`, hors de ce state ;
  - `audit_trail` porte `schemas.audit.AuditEvent` (version enrichie et
    chaînée SHA-256, voir `services/audit_service.py`) — pas la version
    légère de `schemas.results.AuditEvent` utilisée par `ClaimState` (V1) ;
  - `final_decision` est un `schemas.domain.ClaimDecisionV2` (6 issues),
    pas un `schemas.domain.Recommendation` (V1, 3 issues).

Mêmes garanties de contenu que V1 (voir `validate_state_update`) : pas de
texte OCR brut, pas de contenu binaire, pas de chemin absolu, pas de secret.
"""
from __future__ import annotations

import io
import operator
import re
from typing import Annotated, Any, Mapping, TypedDict

from schemas.audit import AuditEvent
from schemas.domain import ClaimDecisionV2
from schemas.v2_results import (
    AutonomousDecisionResult,
    DocumentUnderstandingResult,
    EligibilityResult,
    IntakeSafetyResult,
    MedicalRiskResult,
    RecoveryAttempt,
)

__all__ = ["ClaimStateV2", "validate_claim_state_v2", "validate_state_update_v2"]

_ABSOLUTE_PATH_RE = re.compile(r"^(?:/|[A-Za-z]:[/\\]|\\\\)")
_SECRET_HINT_RE = re.compile(
    r"(?:api[_-]?key|secret\s*[:=]|password\s*[:=]|token\s*[:=]|bearer\s+[a-z0-9._~+/=-]+)",
    re.IGNORECASE,
)
_SECRET_KEY_RE = re.compile(r"(?:api[_-]?key|secret|password|token|credential)", re.IGNORECASE)
_RAW_DOCUMENT_KEYS: frozenset[str] = frozenset({
    "full_text",
    "text_ocr",
    "raw_text",
    "ocr_text",
    "raw_ocr_text",
    "document_bytes",
    "document_content",
    "image_content",
    "pdf_content",
    "base64_image",
    "base64_pdf",
})
_FORBIDDEN_LLM_PAYLOAD_KEYS: frozenset[str] = frozenset({
    "prompt",
    "system_prompt",
    "developer_prompt",
    "user_prompt",
    "messages",
    "raw_response",
    "raw_model_response",
    "model_response",
    "completion",
})


class ClaimStateV2(TypedDict, total=False):
    """État partagé passé à travers les 7 nœuds de `graph/workflow_v2.py`
    (5 agents + `audit_service` + `finalize` — voir plan V2 §2/§3).

    Un nœud LangGraph retourne uniquement les clés qu'il modifie ; LangGraph
    fusionne le dict retourné avec le state existant (mêmes reducers
    `operator.add` que V1 pour les listes append-only).
    """

    # ── Routage ────────────────────────────────────────────────────────────
    case_id: str
    schema_version: str  # "2.0.0"
    current_step: str
    completed_steps: Annotated[list[str], operator.add]

    # ── Entrée — consommée et vidée par intake_safety_agent uniquement ──────
    intake_input: dict | None

    # ── Rôle du lecteur — posé une fois à la soumission (voir api/v2, Phase
    # V2-9), jamais consommé/vidé : lu par document_understanding_agent pour
    # construire la vue minimisée (services.privacy_service.PrivacyService).
    # Contrairement à V1, ne transite jamais par un *_input intermédiaire.
    reader_role: str | None

    # ── Résultats des 5 agents (un par agent, écrasable) ─────────────────────
    intake_safety_result: IntakeSafetyResult | None
    document_understanding_result: DocumentUnderstandingResult | None
    eligibility_result: EligibilityResult | None
    medical_risk_result: MedicalRiskResult | None
    decision_result: AutonomousDecisionResult | None

    # ── Erreurs bloquantes / alertes non bloquantes (append-only) ───────────
    errors: Annotated[list[str], operator.add]
    alerts: Annotated[list[str], operator.add]

    # ── Audit chaîné (append-only) — schemas.audit.AuditEvent, pas la
    # version légère de schemas.results.AuditEvent ────────────────────────
    audit_trail: Annotated[list[AuditEvent], operator.add]

    # ── Récupération autonome bornée (append-only) — un nœud technique
    # unique (`graph.recovery_node_v2`, Phase 6 du plan de remédiation
    # « autonomie décisionnelle V2 »), jamais une boucle LangGraph. Remplace
    # l'ancien `correction_attempts: int` (compteur de resoumissions
    # REQUEST_MORE_INFO, jamais câblé — REQUEST_MORE_INFO est désormais
    # structurellement improductible depuis la Phase 4 du plan) ─────────────
    recovery_attempts: Annotated[list[RecoveryAttempt], operator.add]

    # ── Décision finale ───────────────────────────────────────────────────
    final_decision: ClaimDecisionV2 | None


# ── Garde-fou : validation du contenu du state ───────────────────────────────


def _scan_for_forbidden(value: object, breadcrumb: str) -> list[str]:
    """Retourne la liste des violations trouvées dans value (récursif) —
    même logique que `state.claim_state._scan_for_forbidden` (V1), portée
    ici pour ne dépendre d'aucun fichier V1 (§0)."""
    violations: list[str] = []

    if isinstance(value, (bytes, bytearray)):
        violations.append(
            f"{breadcrumb} : contenu binaire interdit (bytes/{type(value).__name__})"
        )
    elif isinstance(value, io.IOBase):
        violations.append(
            f"{breadcrumb} : objet fichier ouvert interdit ({type(value).__name__})"
        )
    elif isinstance(value, str):
        if _ABSOLUTE_PATH_RE.match(value):
            violations.append(f"{breadcrumb} : chemin absolu interdit — {value!r}")
        if _SECRET_HINT_RE.search(value):
            violations.append(f"{breadcrumb} : secret potentiel interdit dans le ClaimStateV2")
    elif isinstance(value, dict):
        for k, v in value.items():
            key = str(k)
            child_breadcrumb = f"{breadcrumb}.{key}"
            if _SECRET_KEY_RE.search(key):
                violations.append(f"{child_breadcrumb} : clé secrète interdite dans le ClaimStateV2")
            if key in _RAW_DOCUMENT_KEYS and v not in (None, "", [], {}):
                violations.append(f"{child_breadcrumb} : document brut ou texte OCR complet interdit")
            if key.lower() in _FORBIDDEN_LLM_PAYLOAD_KEYS and v not in (None, "", [], {}):
                violations.append(f"{child_breadcrumb} : prompt, messages ou réponse brute LLM interdits")
            violations.extend(_scan_for_forbidden(v, child_breadcrumb))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            violations.extend(_scan_for_forbidden(item, f"{breadcrumb}[{i}]"))
    elif hasattr(value, "model_dump"):
        violations.extend(_scan_for_forbidden(value.model_dump(), breadcrumb))
    # datetime, int, float, bool, None, Enum → autorisés, non inspectés

    return violations


_CONSUMED_INPUT_KEYS: frozenset[str] = frozenset({"intake_input"})

_REQUIRED_STATE_KEYS: frozenset[str] = frozenset({
    "case_id",
    "schema_version",
    "current_step",
    "completed_steps",
})

_RESULT_MODELS: dict[str, type] = {
    "intake_safety_result": IntakeSafetyResult,
    "document_understanding_result": DocumentUnderstandingResult,
    "eligibility_result": EligibilityResult,
    "medical_risk_result": MedicalRiskResult,
    "decision_result": AutonomousDecisionResult,
}


def _is_list_of(value: object, item_type: type, *, allow_dict_items: bool = False) -> bool:
    if not isinstance(value, list):
        return False
    return all(
        isinstance(item, item_type) or (allow_dict_items and isinstance(item, dict))
        for item in value
    )


def _validate_model_value(key: str, value: object, errors: list[str]) -> None:
    if value is None:
        return
    model = _RESULT_MODELS[key]
    if isinstance(value, model):
        return
    if isinstance(value, dict):
        try:
            model.model_validate(value)
        except Exception as exc:  # noqa: BLE001 - convertit en message de contrat stable.
            errors.append(f"{key} : résultat d'agent invalide ({exc})")
        return
    errors.append(f"{key} : type invalide, attendu {model.__name__} | dict | None")


def validate_state_update_v2(updates: dict) -> None:
    """Vérifie qu'une mise à jour du ClaimStateV2 ne contient pas de contenu interdit.

    Mêmes règles que `state.claim_state.validate_state_update` (V1) : pas de
    contenu binaire, pas d'objet fichier ouvert, pas de chemin absolu, pas
    de secret. Les clés d'entrée consommées (`intake_input`) vidées à `None`
    sont ignorées sans inspection. Lève `ValueError` avec la liste des
    violations détectées — à appeler par chaque nœud avant de retourner.
    """
    violations: list[str] = []
    for key, value in updates.items():
        if key in _CONSUMED_INPUT_KEYS and value is None:
            continue
        violations.extend(_scan_for_forbidden(value, key))

    if violations:
        raise ValueError(
            "Mise à jour du ClaimStateV2 refusée — contenu interdit détecté :\n"
            + "\n".join(f"  • {v}" for v in violations)
        )


def validate_claim_state_v2(state: Mapping[str, Any]) -> None:
    """Valide le contrat complet du ClaimStateV2 avant checkpoint ou reprise."""
    errors: list[str] = []
    allowed_keys = set(ClaimStateV2.__annotations__)
    unknown = sorted(set(state) - allowed_keys)
    if unknown:
        errors.append(f"champs inconnus interdits : {', '.join(unknown)}")

    missing = sorted(key for key in _REQUIRED_STATE_KEYS if key not in state)
    if missing:
        errors.append(f"champs obligatoires manquants : {', '.join(missing)}")

    if "case_id" in state and not isinstance(state["case_id"], str):
        errors.append("case_id : type invalide, attendu str")
    if "schema_version" in state and not isinstance(state["schema_version"], str):
        errors.append("schema_version : type invalide, attendu str")
    if "current_step" in state and not isinstance(state["current_step"], str):
        errors.append("current_step : type invalide, attendu str")
    if "recovery_attempts" in state and not _is_list_of(
        state["recovery_attempts"], RecoveryAttempt, allow_dict_items=True
    ):
        errors.append("recovery_attempts : type invalide, attendu list[RecoveryAttempt]")
    if "completed_steps" in state and not _is_list_of(state["completed_steps"], str):
        errors.append("completed_steps : type invalide, attendu list[str]")
    if "errors" in state and not _is_list_of(state["errors"], str):
        errors.append("errors : type invalide, attendu list[str]")
    if "alerts" in state and not _is_list_of(state["alerts"], str):
        errors.append("alerts : type invalide, attendu list[str]")
    if "audit_trail" in state and not _is_list_of(state["audit_trail"], AuditEvent, allow_dict_items=True):
        errors.append("audit_trail : type invalide, attendu list[AuditEvent]")

    if "final_decision" in state and state["final_decision"] is not None:
        try:
            ClaimDecisionV2(state["final_decision"])
        except ValueError:
            errors.append("final_decision : valeur invalide")

    for key in _RESULT_MODELS:
        if key in state:
            _validate_model_value(key, state[key], errors)

    try:
        validate_state_update_v2(dict(state))
    except ValueError as exc:
        errors.append(str(exc))

    if errors:
        raise ValueError("ClaimStateV2 invalide :\n" + "\n".join(f"  • {error}" for error in errors))
