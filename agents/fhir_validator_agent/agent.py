"""FHIR Validator Agent — validation structurelle du bundle FHIR R4.

Agent LLM (gemma4:latest via ChatOllama) + outils déterministes.

Pipeline :
  Phase A — validation déterministe : SHA-256, structure FHIR, ressources, références.
  Phase B — LLM (with_structured_output) : contexte clinique, statut final recommandé, motifs.
  Phase C — construction FhirValidatorResult + audit minimisé.

Interdictions strictes :
  - Aucune décision médicale ou de remboursement.
  - Aucun contenu brut des ressources FHIR dans le résultat.
  - Aucun secret, token ou chemin absolu dans le résultat.
"""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from agents.fhir_validator_agent.schemas import FhirValidatorInput, LlmFhirDecision
from agents.fhir_validator_agent.tools import extraire_types_ressources, valider_bundle_fhir
from langchain_core.messages import HumanMessage, SystemMessage
from llm.factory import get_llm
from llm.metadata import build_llm_metadata
from llm.prompts import load_prompt
from pydantic import ValidationError
try:
    from langgraph.prebuilt import create_react_agent
except ModuleNotFoundError:  # pragma: no cover - dépendance optionnelle en tests locaux
    def create_react_agent(*_args, **_kwargs):
        raise RuntimeError("langgraph indisponible")
from schemas.domain import SecurityDecision, VerificationStatus
from schemas.results import AuditEvent, FhirValidatorResult, SecurityGateResult
from state.claim_state import ClaimState, validate_state_update
from tools.fhir_validation import extract_resource_types, load_fhir_bundle, validate_fhir_bundle
from tools.file_inspection import compute_sha256
from tools.rule_loader import get_rule_version, load_rules

_AGENT_NAME = "fhir_validator_agent"


# ── Résolution du chemin du bundle ───────────────────────────────────────────


def _resolve_bundle_path(relative_path: str) -> str:
    """Résout un chemin relatif de bundle en chemin utilisable (lecture seule).

    Stratégie :
      1. Chemin direct (fixtures de test)
      2. Chemin sous storage/incoming/ (production)
      3. Chemin direct si non trouvé — load_fhir_bundle renverra l'erreur.
    """
    direct = Path(relative_path)
    if direct.exists():
        return str(direct)

    under_storage = Path("storage") / "incoming" / relative_path
    if under_storage.exists():
        return str(under_storage)

    return str(direct)


def _find_bundle_sha256(
    intake_result: object,
    bundle_path: str | None,
) -> str | None:
    """Cherche le SHA-256 attendu du bundle dans le manifest d'ingestion.

    Compare le nom de fichier du bundle (basename) avec original_name dans
    chaque InspectedFile du manifest. Retourne le premier hash trouvé.
    Silencieux en cas d'absence du manifest ou de toute erreur d'attribut.
    """
    if bundle_path is None or intake_result is None:
        return None
    try:
        name = Path(bundle_path).name
        for f in intake_result.manifest.files:  # type: ignore[union-attr]
            if getattr(f, "original_name", None) == name and getattr(f, "sha256", None):
                return f.sha256
    except (AttributeError, TypeError):
        pass
    return None


def _make_not_evaluated_result(
    case_id: str,
    *,
    rule_version: str,
    reason: str,
) -> FhirValidatorResult:
    return FhirValidatorResult(
        case_id=case_id,
        status=VerificationStatus.FAIL,
        bundle_expected=True,
        profile_checked=None,
        rule_version=rule_version,
        validation_scope="NOT_EVALUATED",
        errors=[reason],
        warnings=[],
        reasons=[reason, "Validation FHIR non exécutée."],
    )


def _make_not_provided_result(
    case_id: str,
    *,
    rule_version: str,
    fhir_version: str,
    validation_scope: str,
) -> FhirValidatorResult:
    return FhirValidatorResult(
        case_id=case_id,
        status=VerificationStatus.PASS,
        bundle_expected=False,
        profile_checked=fhir_version,
        rule_version=rule_version,
        validation_scope=validation_scope,
        errors=[],
        warnings=[],
        reasons=[
            "Bundle FHIR non fourni et non attendu pour ce dossier (NOT_PROVIDED).",
            "Validation structurelle ignorée.",
        ],
    )


# ── Audit minimisé ────────────────────────────────────────────────────────────


def _build_audit_event(
    case_id: str,
    result: FhirValidatorResult,
    *,
    sha256_verified: bool,
    security_gate_checked: bool,
) -> AuditEvent:
    """Construit un AuditEvent minimal — aucune donnée FHIR brute incluse."""
    return AuditEvent(
        event_id=str(uuid.uuid4()),
        case_id=case_id,
        actor="fhir_validator_agent",
        action="fhir_validation",
        outcome=result.status.value,
        agent_version=result.rule_version,
        timestamp=datetime.now(UTC),
        details={
            "profile_checked": result.profile_checked or "",
            "resource_count": str(result.resource_count),
            "references_checked": str(result.references_checked),
            "validation_scope": result.validation_scope,
            "sha256_verified": str(sha256_verified),
            "security_gate_checked": str(security_gate_checked),
            "error_count": str(len(result.errors)),
            "warning_count": str(len(result.warnings)),
        },
    )


# ── Phase B : décision LLM ────────────────────────────────────────────────────


def _invoke_llm_fhir(
    *,
    deterministic_status: str,
    errors: list[str],
    warnings: list[str],
    resource_types: list[str],
    sha256_verified: bool,
    validation_scope: str,
) -> LlmFhirDecision | None:
    """Envoie les résultats déterministes au LLM pour contexte clinique et statut final."""
    try:
        llm = get_llm()
        agent = create_react_agent(
            model=llm,
            tools=[valider_bundle_fhir, extraire_types_ressources],
            response_format=LlmFhirDecision,
        )
        data = {
            "deterministic_status": deterministic_status,
            "errors": errors[:10],
            "warnings": warnings[:10],
            "resource_types": resource_types,
            "sha256_verified": sha256_verified,
            "validation_scope": validation_scope,
            "instruction": (
                "Conformité technique uniquement : ne conclus jamais à la validité clinique "
                "et ne crée aucune ressource FHIR absente."
            ),
        }
        result = agent.invoke({
            "messages": [
                SystemMessage(content=load_prompt(_AGENT_NAME)),
                HumanMessage(content=json.dumps(data, ensure_ascii=False)),
            ]
        })
        structured = result.get("structured_response")
        if isinstance(structured, LlmFhirDecision):
            return structured
        if isinstance(structured, dict):
            return LlmFhirDecision(**structured)
        return None
    except Exception:
        return None


# ── Fonction principale ───────────────────────────────────────────────────────


def run(
    case_id: str,
    fhir_bundle_path: str | None = None,
    *,
    bundle_expected: bool = True,
    fhir_version: str = "R4",
    validation_scope: str = "STRUCTURAL_ONLY",
    security_allowed: bool = True,
    expected_sha256: str | None = None,
) -> FhirValidatorResult:
    """Valide le bundle FHIR R4 associé au dossier.

    Testable sans LangGraph — ne lit pas ClaimState.

    Args:
        case_id           : identifiant du dossier.
        fhir_bundle_path  : chemin relatif vers le bundle (None si absent).
        bundle_expected   : True si le bundle est obligatoire pour ce dossier.
        fhir_version      : version FHIR attendue ("R4" ou "R4B").
        validation_scope  : "STRUCTURAL_ONLY" ou "FULL".
        security_allowed  : False si le Security Gate n'a pas accordé ALLOW.
        expected_sha256   : SHA-256 attendu du bundle (depuis le manifest).

    Returns:
        FhirValidatorResult avec statut PASS / NEEDS_REVIEW / FAIL.
    """
    rule_version = get_rule_version("fhir_rules.yaml")

    # Étape 2 — vérification de l'autorisation du Security Gate
    if not security_allowed:
        return _make_not_evaluated_result(
            case_id,
            rule_version=rule_version,
            reason="Security Gate non ALLOW — validation FHIR non autorisée.",
        )

    # Étape 3 — bundle optionnel absent (NOT_PROVIDED)
    if fhir_bundle_path is None and not bundle_expected:
        return _make_not_provided_result(
            case_id,
            rule_version=rule_version,
            fhir_version=fhir_version,
            validation_scope=validation_scope,
        )

    # Étape 4a — résolution du chemin
    resolved_path: str | None = None
    if fhir_bundle_path is not None:
        resolved_path = _resolve_bundle_path(fhir_bundle_path)

    # Étape 4b — vérification SHA-256
    sha256_verified = False
    sha256_errors: list[str] = []
    if resolved_path is not None and expected_sha256 is not None:
        try:
            actual = compute_sha256(Path(resolved_path))
            if actual == expected_sha256:
                sha256_verified = True
            else:
                sha256_errors.append(
                    f"bundle.sha256: intégrité échouée — "
                    f"attendu {expected_sha256[:16]}…, calculé {actual[:16]}…"
                )
        except OSError as exc:
            sha256_errors.append(
                f"bundle.sha256: impossible de vérifier l'intégrité — {exc}"
            )

    # Étape 5 — chargement des règles
    rules = load_rules("fhir_rules.yaml")

    # Arrêt précoce sur échec SHA-256
    if sha256_errors:
        return FhirValidatorResult(
            case_id=case_id,
            status=VerificationStatus.FAIL,
            bundle_expected=bundle_expected,
            profile_checked=None,
            rule_version=rule_version,
            validation_scope=validation_scope,
            errors=sha256_errors,
            warnings=[],
            reasons=["Validation FHIR interrompue : intégrité du bundle non confirmée."],
        )

    # Étape 6 — validation structurelle et des ressources
    status, errors, warnings, profile_checked = validate_fhir_bundle(
        resolved_path,
        bundle_expected=bundle_expected,
        rules=rules,
    )

    # Extraction des types de ressources (lecture seule, bundle non retourné)
    resource_types: list[str] = []
    if resolved_path is not None and status != VerificationStatus.FAIL:
        bundle, load_errors = load_fhir_bundle(resolved_path)
        if bundle is not None and not load_errors:
            resource_types = extract_resource_types(bundle)

    # ── Phase B : LLM — contexte clinique et statut final ────────────────────
    llm_decision = _invoke_llm_fhir(
        deterministic_status=status.value,
        errors=list(errors),
        warnings=list(warnings),
        resource_types=resource_types,
        sha256_verified=sha256_verified,
        validation_scope=validation_scope,
    )

    # Le statut final est celui du LLM si disponible ; on ne peut pas assouplir un FAIL
    final_status = status
    if llm_decision is None:
        final_status = VerificationStatus.FAIL
    else:
        llm_status_map = {
            "PASS": VerificationStatus.PASS,
            "NEEDS_REVIEW": VerificationStatus.NEEDS_REVIEW,
            "FAIL": VerificationStatus.FAIL,
        }
        llm_vs = llm_status_map.get(llm_decision.recommended_status, status)
        # Le LLM peut durcir une décision technique, jamais l'assouplir.
        if status == VerificationStatus.PASS:
            final_status = llm_vs
        elif status == VerificationStatus.NEEDS_REVIEW:
            final_status = VerificationStatus.FAIL if llm_vs == VerificationStatus.FAIL else status

    # ── Phase C : construction des raisons ────────────────────────────────────
    reasons: list[str] = []

    if llm_decision is None:
        reasons.append("LLM indisponible : validation FHIR interrompue en mode FAIL.")
    elif llm_decision.clinical_context:
        reasons.append(llm_decision.clinical_context)

    if llm_decision and llm_decision.reasons:
        reasons.extend(llm_decision.reasons)
    elif llm_decision is not None:
        # Fallback de motifs déterministes si le LLM n'a pas fourni de raisons.
        if final_status == VerificationStatus.PASS:
            reasons.append(
                f"Bundle FHIR {profile_checked or fhir_version} valide — "
                "structure et ressources conformes."
            )
        elif final_status == VerificationStatus.NEEDS_REVIEW:
            reasons.append("Bundle FHIR chargé avec avertissements — revue recommandée.")
            reasons.extend(list(warnings)[:5])
        else:
            reasons.append("Validation FHIR échouée — voir la liste des erreurs.")
            reasons.extend(list(errors)[:5])

    # Statut de l'intégrité SHA-256
    if final_status != VerificationStatus.FAIL or not sha256_errors:
        if sha256_verified:
            reasons.append("Intégrité SHA-256 confirmée.")
        elif expected_sha256 is None:
            reasons.append("Intégrité SHA-256 non vérifiable — hash attendu absent du manifest.")

    reasons.append(
        "Validation structurelle uniquement : la conformité FHIR ne prouve pas "
        "la vérité médicale ni la cohérence clinique du contenu."
    )

    return FhirValidatorResult(
        case_id=case_id,
        status=final_status,
        bundle_expected=bundle_expected,
        profile_checked=profile_checked,
        rule_version=rule_version,
        resource_types=resource_types,
        resource_count=len(resource_types),
        references_checked=resolved_path is not None and final_status != VerificationStatus.FAIL,
        validation_scope=validation_scope,
        errors=errors,
        warnings=warnings,
        reasons=reasons,
        llm_metadata=build_llm_metadata(_AGENT_NAME),
    )


# ── Nœud LangGraph ───────────────────────────────────────────────────────────


def _security_is_allowed(state: ClaimState) -> bool:
    """Retourne True uniquement si security_result.decision == ALLOW."""
    security_result: SecurityGateResult | None = state.get("security_result")
    if security_result is None:
        return False
    try:
        return security_result.decision == SecurityDecision.ALLOW
    except AttributeError:
        return False


def node(state: ClaimState) -> dict:
    """Nœud LangGraph du FHIR Validator Agent.

    Lit fhir_input depuis le state, exécute la validation, écrit fhir_result.
    Vide fhir_input à None après traitement (consommation du champ d'entrée).
    Produit un AuditEvent minimisé dans audit_trail.

    Args:
        state: État partagé du workflow LangGraph.

    Returns:
        Dictionnaire de mise à jour du state.
    """
    rule_version = get_rule_version("fhir_rules.yaml")
    fhir_input_raw: dict | None = state.get("fhir_input")

    # ── Cas fhir_input absent ────────────────────────────────────────────────
    if fhir_input_raw is None:
        case_id = str(state.get("case_id", "UNKNOWN"))
        result = _make_not_evaluated_result(
            case_id,
            rule_version=rule_version,
            reason="fhir_input absent du state — impossible d'exécuter la validation FHIR.",
        )
        audit = _build_audit_event(
            case_id, result, sha256_verified=False, security_gate_checked=False
        )
        updates: dict = {
            "fhir_input": None,
            "fhir_result": result,
            "completed_steps": ["fhir_validation"],
            "current_step": "fhir_validation",
            "audit_trail": [audit],
        }
        validate_state_update(updates)
        return updates

    # ── Validation Pydantic de l'entrée ──────────────────────────────────────
    try:
        fhir_input = FhirValidatorInput(**fhir_input_raw)
    except (ValidationError, TypeError) as exc:
        case_id = str(fhir_input_raw.get("case_id", state.get("case_id", "UNKNOWN")))
        result = FhirValidatorResult(
            case_id=case_id,
            status=VerificationStatus.FAIL,
            bundle_expected=bool(fhir_input_raw.get("bundle_expected", True)),
            profile_checked=None,
            rule_version=rule_version,
            validation_scope="STRUCTURAL_ONLY",
            errors=[f"fhir_input invalide : {exc}"],
            warnings=[],
            reasons=["Entrée FHIR invalide — validation Pydantic échouée."],
        )
        audit = _build_audit_event(
            case_id, result, sha256_verified=False, security_gate_checked=False
        )
        updates = {
            "fhir_input": None,
            "fhir_result": result,
            "completed_steps": ["fhir_validation"],
            "current_step": "fhir_validation",
            "audit_trail": [audit],
        }
        validate_state_update(updates)
        return updates

    # ── Vérification Security Gate ────────────────────────────────────────────
    security_allowed = _security_is_allowed(state)
    security_gate_checked = state.get("security_result") is not None

    # ── Extraction SHA-256 depuis le manifest d'ingestion ─────────────────────
    expected_sha256 = _find_bundle_sha256(
        state.get("intake_result"),
        fhir_input.fhir_bundle_path,
    )

    # ── Exécution de la validation ────────────────────────────────────────────
    result = run(
        case_id=fhir_input.case_id,
        fhir_bundle_path=fhir_input.fhir_bundle_path,
        bundle_expected=fhir_input.bundle_expected,
        fhir_version=fhir_input.fhir_version,
        validation_scope=fhir_input.validation_scope,
        security_allowed=security_allowed,
        expected_sha256=expected_sha256,
    )

    # ── Audit minimisé ────────────────────────────────────────────────────────
    sha256_verified = (
        expected_sha256 is not None
        and result.status != VerificationStatus.FAIL
        and "Intégrité SHA-256 confirmée." in result.reasons
    )
    audit = _build_audit_event(
        fhir_input.case_id,
        result,
        sha256_verified=sha256_verified,
        security_gate_checked=security_gate_checked,
    )

    updates = {
        "fhir_input": None,
        "fhir_result": result,
        "completed_steps": ["fhir_validation"],
        "current_step": "fhir_validation",
        "audit_trail": [audit],
    }

    validate_state_update(updates)
    return updates
