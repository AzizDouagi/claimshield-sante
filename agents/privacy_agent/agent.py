"""Privacy Agent — vues minimisées par rôle de lecteur (politique DENY-by-default).

Agent purement déterministe — aucun appel LLM.

Pipeline (14 étapes) :
  1.  Vérification que le Security Gate a produit ALLOW
  2.  Validation du rôle (obligatoire, tout rôle inconnu → blocage)
  3.  Validation Pydantic de PrivacyInput
  4.  Vérification de la disponibilité de la clé de pseudonymisation
  5.  Calcul DENY-by-default : champs refusés = ALL_KNOWN_FIELDS − allowlist du rôle
  6.  Pseudonymisation des identifiants personnels fournis (optionnel)
  7.  Masquage partiel des champs partiellement autorisés (via les builders de vues)
  8.  Suppression des champs interdits (DENY-by-default, inclus dans redacted_fields)
  9.  Construction de la vue minimisée selon le rôle (si claim_data fourni)
 10.  Validation Pydantic de la vue produite (interne aux builders)
 11.  Vérification post-vue : aucun identifiant brut, aucun champ secret
 12.  Détermination du statut PASS / NEEDS_REVIEW / FAIL
 13.  Construction de la trace d'audit minimisée (PrivacyAuditEntry)
 14.  Retour de PrivacyResult — objet source jamais modifié

Interdictions strictes :
  - Aucune analyse médicale ou clinique.
  - Aucune décision de remboursement.
  - Aucun accès au contenu brut des fichiers.
  - Aucun secret, token ou chemin absolu dans le résultat.
  - Aucune modification de l'objet source ni des documents originaux.

Règles de la politique DENY-by-default :
  - Un rôle est obligatoire pour demander une vue.
  - Un rôle inconnu provoque un blocage (FAIL).
  - Tout champ absent de l'allowlist du rôle est automatiquement refusé.
  - Aucun rôle n'a accès à l'intégralité des champs connus.
"""
from __future__ import annotations

import uuid

from pydantic import ValidationError

from agents.privacy_agent.schemas import PrivacyInput, ReaderRole
from agents.privacy_agent.views import build_view
from schemas.domain import DataClassification, PrivacyCode, SecurityDecision, VerificationStatus
from schemas.results import (
    AuditEvent,
    PrivacyAuditEntry,
    PrivacyResult,
    SecurityGateResult,
)
from security.access_policies import POLICY_VERSION, compute_masked_fields, verify_view_privacy
from state.claim_state import ClaimState, validate_state_update
from tools.pseudonymize import (
    pseudonymization_key_is_available,
    pseudonymize_fields,
)


# ── Helpers de construction d'audit ──────────────────────────────────────────


def _make_audit_entry(
    case_id: str,
    role_str: str,
    decision: VerificationStatus,
    *,
    redacted_count: int = 0,
    view_built: bool = False,
    pseudonymization_applied: bool = False,
    reason_codes: list[PrivacyCode] | None = None,
) -> PrivacyAuditEntry:
    """Construit la trace d'audit minimisée — sans données personnelles ni secrets."""
    return PrivacyAuditEntry(
        claim_id=case_id,
        role=role_str,
        outcome=decision.value,
        decision=decision,
        policy_version=POLICY_VERSION,
        redacted_count=redacted_count,
        view_built=view_built,
        pseudonymization_applied=pseudonymization_applied,
        reason_codes=reason_codes or [],
    )


def _violation_to_codes(violations: list) -> list[PrivacyCode]:
    """Convertit les PolicyViolation en codes PrivacyCode stables (sans doublon)."""
    codes: list[PrivacyCode] = []
    for v in violations:
        if v.reason_code in ("SECRET_FIELD_IN_VIEW", "RAW_IDENTITY_IN_VIEW"):
            if PrivacyCode.FORBIDDEN_FIELD_EXPOSED not in codes:
                codes.append(PrivacyCode.FORBIDDEN_FIELD_EXPOSED)
        elif v.reason_code == "INVALID_PSEUDONYM_FORMAT":
            if PrivacyCode.UNMASKED_IDENTIFIER not in codes:
                codes.append(PrivacyCode.UNMASKED_IDENTIFIER)
    return codes


# ── Helpers de résultats FAIL ─────────────────────────────────────────────────


def _determine_status(
    privacy_input: PrivacyInput,
    redacted_fields: list[str],
) -> tuple[VerificationStatus, list[str]]:
    """Retourne (statut, raisons) selon la classification et la présence de données réelles."""
    reasons: list[str] = []

    if privacy_input.contains_real_personal_data:
        status = VerificationStatus.NEEDS_REVIEW
        reasons.append(
            "Données personnelles réelles détectées — revue humaine requise avant diffusion"
        )
    elif privacy_input.data_classification == DataClassification.CONFIDENTIAL:
        status = VerificationStatus.NEEDS_REVIEW
        reasons.append(
            "Données classifiées CONFIDENTIAL — politique DENY-by-default appliquée, revue conseillée"
        )
    else:
        status = VerificationStatus.PASS
        reasons.append(
            f"Vue DENY-by-default appliquée pour le rôle {privacy_input.role.value}"
        )

    if redacted_fields:
        count = len(redacted_fields)
        sample = ", ".join(redacted_fields[:5])
        suffix = "…" if count > 5 else ""
        reasons.append(f"{count} champ(s) refusé(s) : {sample}{suffix}")
    else:
        reasons.append(
            f"Aucun champ refusé pour le rôle {privacy_input.role.value} "
            f"sur l'univers évalué"
        )

    return status, reasons


def _security_gate_blocked_result(
    case_id: str,
    decision: str,
    role_str: str = "UNKNOWN",
) -> PrivacyResult:
    """FAIL produit quand le Security Gate n'a pas renvoyé ALLOW."""
    msg = (
        f"Security Gate requis avant le Privacy Agent — décision reçue : {decision}. "
        "Traitement privacy annulé (DENY-by-default)."
    )
    return PrivacyResult(
        case_id=case_id,
        status=VerificationStatus.FAIL,
        data_classification=DataClassification.CONFIDENTIAL,
        contains_real_personal_data=False,
        redacted_fields=[],
        reasons=[msg],
        errors=[msg],
        policy_version=POLICY_VERSION,
        reason_codes=[],
        audit_entry=_make_audit_entry(
            case_id, role_str, VerificationStatus.FAIL, reason_codes=[],
        ),
    )


def _pseudonymization_key_blocked_result(case_id: str, role_str: str) -> PrivacyResult:
    """FAIL produit quand la clé HMAC de pseudonymisation est inaccessible."""
    msg = (
        "Clé de pseudonymisation inaccessible ou vide — "
        "traitement privacy bloqué (DENY-by-default)"
    )
    codes = [PrivacyCode.MISSING_PSEUDONYMIZATION_KEY]
    return PrivacyResult(
        case_id=case_id,
        status=VerificationStatus.FAIL,
        data_classification=DataClassification.CONFIDENTIAL,
        contains_real_personal_data=False,
        redacted_fields=[],
        reasons=[msg],
        errors=[msg],
        policy_version=POLICY_VERSION,
        reason_codes=codes,
        audit_entry=_make_audit_entry(
            case_id, role_str, VerificationStatus.FAIL, reason_codes=codes,
        ),
    )


def _role_blocked_result(
    case_id: str,
    reason: str,
    role_str: str = "UNKNOWN",
    *,
    reason_codes: list[PrivacyCode] | None = None,
) -> PrivacyResult:
    """FAIL produit quand le rôle est absent, inconnu ou l'entrée invalide."""
    codes = reason_codes or []
    msg = f"Rôle invalide ou absent — blocage DENY-by-default : {reason}"
    return PrivacyResult(
        case_id=case_id,
        status=VerificationStatus.FAIL,
        data_classification=DataClassification.CONFIDENTIAL,
        contains_real_personal_data=False,
        redacted_fields=[],
        reasons=[msg],
        errors=[msg],
        policy_version=POLICY_VERSION,
        reason_codes=codes,
        audit_entry=_make_audit_entry(
            case_id, role_str, VerificationStatus.FAIL, reason_codes=codes,
        ),
    )


def _validation_error_result(
    case_id: str,
    error: ValidationError,
    role_str: str = "UNKNOWN",
    *,
    reason_codes: list[PrivacyCode] | None = None,
) -> PrivacyResult:
    """FAIL produit quand PrivacyInput ne passe pas la validation Pydantic."""
    codes = reason_codes or [PrivacyCode.INVALID_PRIVACY_INPUT]
    first_msg = (
        error.errors()[0].get("msg", "Erreur de validation")
        if error.errors()
        else "Erreur"
    )
    msg = f"Entrée PrivacyInput invalide — blocage DENY-by-default : {first_msg}"
    return PrivacyResult(
        case_id=case_id,
        status=VerificationStatus.FAIL,
        data_classification=DataClassification.CONFIDENTIAL,
        contains_real_personal_data=False,
        redacted_fields=[],
        reasons=[msg],
        errors=[msg],
        policy_version=POLICY_VERSION,
        reason_codes=codes,
        audit_entry=_make_audit_entry(
            case_id, role_str, VerificationStatus.FAIL, reason_codes=codes,
        ),
    )


def _view_validation_error_result(
    case_id: str,
    role_str: str,
    error: ValidationError,
) -> PrivacyResult:
    """FAIL produit quand le builder de vue lève une ValidationError Pydantic."""
    codes = [PrivacyCode.INVALID_PRIVACY_OUTPUT]
    first_msg = (
        error.errors()[0].get("msg", "Erreur de validation")
        if error.errors()
        else "Erreur"
    )
    msg = f"Vue minimisée invalide — identifiant potentiellement brut bloqué : {first_msg}"
    return PrivacyResult(
        case_id=case_id,
        status=VerificationStatus.FAIL,
        data_classification=DataClassification.CONFIDENTIAL,
        contains_real_personal_data=True,
        redacted_fields=[],
        reasons=[msg],
        errors=[msg],
        policy_version=POLICY_VERSION,
        reason_codes=codes,
        audit_entry=_make_audit_entry(
            case_id, role_str, VerificationStatus.FAIL, reason_codes=codes,
        ),
    )


def _view_violation_result(
    case_id: str,
    role_str: str,
    violations: list,
) -> PrivacyResult:
    """FAIL produit quand verify_view_privacy détecte un identifiant brut dans la vue."""
    codes = _violation_to_codes(violations)
    msgs = [f"[VUE BLOQUÉE] {v.reason_code} — {v.message}" for v in violations]
    return PrivacyResult(
        case_id=case_id,
        status=VerificationStatus.FAIL,
        data_classification=DataClassification.CONFIDENTIAL,
        contains_real_personal_data=True,
        redacted_fields=[],
        reasons=msgs,
        errors=msgs,
        policy_version=POLICY_VERSION,
        reason_codes=codes,
        audit_entry=_make_audit_entry(
            case_id, role_str, VerificationStatus.FAIL,
            view_built=False, reason_codes=codes,
        ),
    )


def _policy_not_found_result(case_id: str, role_str: str) -> PrivacyResult:
    """FAIL produit quand aucune politique n'est trouvée pour le rôle (incohérence interne)."""
    codes = [PrivacyCode.UNKNOWN_POLICY]
    msg = (
        f"Politique d'accès introuvable pour le rôle '{role_str}' — "
        "incohérence interne, traitement bloqué (DENY-by-default)"
    )
    return PrivacyResult(
        case_id=case_id,
        status=VerificationStatus.FAIL,
        data_classification=DataClassification.CONFIDENTIAL,
        contains_real_personal_data=False,
        redacted_fields=[],
        reasons=[msg],
        errors=[msg],
        policy_version=POLICY_VERSION,
        reason_codes=codes,
        audit_entry=_make_audit_entry(
            case_id, role_str, VerificationStatus.FAIL, reason_codes=codes,
        ),
    )


def _pseudonymization_error_result(
    case_id: str,
    role_str: str,
) -> PrivacyResult:
    """FAIL produit quand la pseudonymisation lève une erreur inattendue."""
    codes = [PrivacyCode.PSEUDONYMIZATION_ERROR]
    msg = (
        "Erreur technique lors de la pseudonymisation — "
        "traitement privacy bloqué (DENY-by-default)"
    )
    return PrivacyResult(
        case_id=case_id,
        status=VerificationStatus.FAIL,
        data_classification=DataClassification.CONFIDENTIAL,
        contains_real_personal_data=False,
        redacted_fields=[],
        reasons=[msg],
        errors=[msg],
        policy_version=POLICY_VERSION,
        reason_codes=codes,
        audit_entry=_make_audit_entry(
            case_id, role_str, VerificationStatus.FAIL, reason_codes=codes,
        ),
    )


# ── Fonction principale (testable sans LangGraph) ─────────────────────────────


def run(
    privacy_input: PrivacyInput,
    security_result: SecurityGateResult | None = None,
) -> PrivacyResult:
    """Exécute le pipeline de confidentialité DENY-by-default pour un dossier.

    Args:
        privacy_input   : données structurées validées par Pydantic — rôle obligatoire.
        security_result : résultat du Security Gate — doit être ALLOW pour continuer.
                          Si None, le traitement est bloqué (gate non exécuté).

    Returns:
        PrivacyResult avec statut PASS, NEEDS_REVIEW ou FAIL, champs refusés
        calculés par DENY-by-default, trace d'audit minimisée et motifs lisibles.

    Garanties :
        - L'objet privacy_input n'est jamais modifié.
        - claim_data est copié défensivement avant d'être passé aux builders.
        - Aucun document original n'est accédé ni modifié.
    """
    # role_str est toujours valide — PrivacyInput garantit un rôle connu (ReaderRole)
    role_str: str = privacy_input.role.value

    # ── Étape 1 : vérification du Security Gate ───────────────────────────────
    if security_result is None:
        return _security_gate_blocked_result(privacy_input.case_id, "ABSENT", role_str)

    if security_result.decision != SecurityDecision.ALLOW:
        return _security_gate_blocked_result(
            privacy_input.case_id,
            security_result.decision.value,
            role_str,
        )

    # ── Étape 4 : vérification de la clé de pseudonymisation ─────────────────
    if not pseudonymization_key_is_available():
        return _pseudonymization_key_blocked_result(privacy_input.case_id, role_str)

    # ── Étape 5 : calcul DENY-by-default ─────────────────────────────────────
    try:
        redacted_fields = compute_masked_fields(
            role=privacy_input.role,
            fields_to_evaluate=privacy_input.fields_to_evaluate or None,
        )
    except KeyError:
        return _policy_not_found_result(privacy_input.case_id, role_str)

    # ── Étape 6 : pseudonymisation des identifiants personnels ────────────────
    personal_values: dict[str, str | None] = {
        "patient_name": privacy_input.patient_name,
        "patient_id": privacy_input.patient_id,
        "payer_name": privacy_input.payer_name,
        "invoice_number": privacy_input.invoice_number,
        "prescription_number": privacy_input.prescription_number,
    }
    provided: dict[str, str | None] = {k: v for k, v in personal_values.items() if v is not None}
    pseudonymization_applied = False
    if provided:
        try:
            pseudonymize_fields(provided, redacted_fields)
        except Exception:
            return _pseudonymization_error_result(privacy_input.case_id, role_str)
        pseudonymization_applied = True

    # ── Étapes 7-8 : vue minimisée et vérification post-vue ──────────────────
    # Étape 7 (masquage partiel) et 8 (suppression des interdits) sont effectués
    # par les builders de vues via mask_field_value et DENY-by-default.
    # Étape 9-11 : construction, validation Pydantic, et vérification post-vue.
    # claim_data est copié défensivement — l'objet source n'est jamais modifié.
    view: dict | None = None
    view_built = False
    if privacy_input.claim_data is not None:
        claim_data = dict(privacy_input.claim_data)
        try:
            view = build_view(privacy_input.role, privacy_input.case_id, claim_data)
        except ValidationError as exc:
            return _view_validation_error_result(privacy_input.case_id, role_str, exc)

        violations = verify_view_privacy(view)
        if violations:
            return _view_violation_result(privacy_input.case_id, role_str, violations)
        view_built = True

    # ── Étape 12 : statut et motifs ───────────────────────────────────────────
    status, reasons = _determine_status(privacy_input, redacted_fields)

    # ── Étape 13 : trace d'audit minimisée ────────────────────────────────────
    audit_entry = _make_audit_entry(
        case_id=privacy_input.case_id,
        role_str=role_str,
        decision=status,
        redacted_count=len(redacted_fields),
        view_built=view_built,
        pseudonymization_applied=pseudonymization_applied,
    )

    return PrivacyResult(
        case_id=privacy_input.case_id,
        status=status,
        data_classification=privacy_input.data_classification,
        contains_real_personal_data=privacy_input.contains_real_personal_data,
        redacted_fields=redacted_fields,
        reasons=reasons,
        errors=[],
        policy_version=POLICY_VERSION,
        reason_codes=[],
        view=view,
        view_role=privacy_input.role.value if view is not None else None,
        audit_entry=audit_entry,
    )


# ── Nœud LangGraph ────────────────────────────────────────────────────────────


def node(state: ClaimState) -> dict:
    """Nœud LangGraph — construit PrivacyInput depuis le state et délègue à run().

    Attend dans le state :
        case_id          : identifiant du dossier
        security_result  : SecurityGateResult — doit avoir decision=ALLOW
        privacy_input    : dict contenant au minimum {"role": "<valeur>"}

    Règles DENY-by-default appliquées dans le nœud :
        - role absent dans privacy_input → FAIL immédiat
        - role inconnu (hors ReaderRole)   → FAIL immédiat
        - security_result absent/non ALLOW → FAIL immédiat
        - clé de pseudonymisation inaccessible → FAIL immédiat

    Clés reconnues dans privacy_input :
        role (OBLIGATOIRE), data_classification, contains_real_personal_data,
        fields_to_evaluate, patient_name, patient_id, payer_name,
        invoice_number, prescription_number, claim_data (optionnel — produit la vue)
    """
    case_id: str = state.get("case_id", "CLM-0000")  # type: ignore[assignment]
    raw: dict = state.get("privacy_input", {}) or {}  # type: ignore[assignment]
    security_result: SecurityGateResult | None = state.get("security_result")  # type: ignore[assignment]

    if not case_id or not case_id.startswith("CLM-"):
        result = _role_blocked_result(
            "CLM-0000",
            "case_id invalide ou absent",
            reason_codes=[PrivacyCode.INVALID_PRIVACY_INPUT],
        )
    else:
        raw_role: str | None = raw.get("role")

        if raw_role is None:
            result = _role_blocked_result(
                case_id,
                "Le champ 'role' est obligatoire (politique DENY-by-default)",
                reason_codes=[PrivacyCode.MISSING_ROLE],
            )
        else:
            try:
                role = ReaderRole(raw_role)
            except ValueError as exc:
                result = _role_blocked_result(
                    case_id, str(exc), raw_role,
                    reason_codes=[PrivacyCode.UNKNOWN_ROLE],
                )
            else:
                try:
                    privacy_input = PrivacyInput(
                        case_id=case_id,
                        role=role,
                        data_classification=DataClassification(
                            raw.get(
                                "data_classification",
                                DataClassification.SYNTHETIC_TEST_DATA.value,
                            )
                        ),
                        contains_real_personal_data=bool(
                            raw.get("contains_real_personal_data", False)
                        ),
                        fields_to_evaluate=list(raw.get("fields_to_evaluate", [])),
                        patient_name=raw.get("patient_name"),
                        patient_id=raw.get("patient_id"),
                        payer_name=raw.get("payer_name"),
                        invoice_number=raw.get("invoice_number"),
                        prescription_number=raw.get("prescription_number"),
                        claim_data=raw.get("claim_data"),
                    )
                except ValidationError as exc:
                    result = _validation_error_result(
                        case_id, exc, raw_role,
                        reason_codes=[PrivacyCode.INVALID_PRIVACY_INPUT],
                    )
                else:
                    result = run(privacy_input, security_result)

    # ── Trace d'audit dans l'état LangGraph ───────────────────────────────────
    audit_role = result.view_role or (raw.get("role") if raw else None) or "UNKNOWN"
    audit_event = AuditEvent(
        event_id=str(uuid.uuid4()),
        case_id=case_id,
        actor="privacy_agent",
        action="view_request",
        outcome=result.status.value,
        details={
            "role": audit_role,
            "policy_version": result.policy_version,
        },
    )

    updates: dict = {
        "privacy_result": result,
        "privacy_input": None,
        "current_step": "privacy",
        "completed_steps": ["privacy"],
        "audit_trail": [audit_event],
    }

    if result.status == VerificationStatus.FAIL:
        updates["errors"] = [f"[privacy] {r}" for r in result.errors or result.reasons]
    elif result.status == VerificationStatus.NEEDS_REVIEW:
        updates["alerts"] = [
            f"Privacy : NEEDS_REVIEW — {len(result.redacted_fields)} champ(s) refusé(s) "
            f"(DENY-by-default)"
        ]

    validate_state_update(updates)
    return updates
