"""Identity and Coverage Agent — ClaimShield Santé.

Vérifie la concordance des identifiants patient et la validité de la couverture
assurance à partir des données extraites par l'OCR et du bundle FHIR.

Décisions déterministes ; le LLM produit uniquement une synthèse consultative
à partir de preuves minimisées.

Pipeline (9 étapes) :
  1. Lecture de identity_coverage_input depuis le state (dict → IdentityCoverageInput)
  2. Lecture de ocr_result depuis le state (DocumentOcrResult | None)
  3. Extraction des champs patient/couverture depuis ocr_result.extracted_fields
  4. Lecture optionnelle du bundle FHIR pour extraire le patient_id FHIR
  5. Appel de identity_match_result() pour la vérification d'identité
  6. Appel de verify_coverage() pour la vérification de couverture
  7. Construction de IdentityCoverageResult
  8. Appel de validate_state_update()
  9. Retour des mises à jour du state

Points d'entrée :
  run(case_id, ocr_result, fhir_bundle_path=None)  → IdentityCoverageResult
  node(state)                                        → dict
"""
from __future__ import annotations

import json
import uuid
from copy import deepcopy
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

try:
    from langgraph.prebuilt import create_react_agent
except ModuleNotFoundError:  # pragma: no cover - dépendance optionnelle en tests locaux
    def create_react_agent(*_args, **_kwargs):
        raise RuntimeError("langgraph indisponible")

from agents.identity_coverage_agent.schemas import (
    AuthorizationCheck,
    AuthorizationCheckStatus,
    CoverageCheck,
    CoverageCheckStatus,
    IdentityCheck,
    IdentityCheckStatus,
    IdentityCoverageInput,
    LlmIdentityCoverageDecision,
    StructuredRuleError,
)
from agents.identity_coverage_agent.tools import (
    charger_contrat,
    verifier_couverture_contrat,
    verifier_identite_contrat,
)
from llm.factory import get_llm
from llm.metadata import build_llm_metadata
from llm.prompts import load_prompt
from schemas.domain import VerificationStatus
from schemas.results import (
    AuditEvent,
    CoverageResult,
    IdentityCoverageResult,
    IdentityResult,
)
from state.claim_state import ClaimState, validate_state_update
from tools.coverage_rules import evaluate_coverage_contract, verify_coverage
from tools.contract_repository import get_contract, load_contracts
from tools.identity_matching import identity_match_result, match_identity_contract
from tools.rule_loader import get_rule_version, load_rules

# Version des règles utilisées (lue depuis le YAML)
_RULE_VERSION_FILE = "identity_rules.yaml"
_AGENT_NAME = "identity_coverage_agent"

# Zone de stockage incoming — chemin résolu depuis la racine du projet
_INCOMING_DIR = Path("storage") / "incoming"


def _extract_field_value(
    extracted_fields: dict,
    field_name: str,
) -> str | None:
    """Extrait la valeur d'un champ depuis le dict extracted_fields d'un DocumentOcrResult.

    Accepte à la fois un dict de ExtractedField Pydantic (avec attribut .value)
    et un dict brut (avec clé "value") pour faciliter les tests.

    Retourne None si le champ est absent ou si la valeur est vide.
    """
    field = extracted_fields.get(field_name)
    if field is None:
        return None
    # Objet Pydantic ExtractedField
    if hasattr(field, "value"):
        val = field.value
    # Dict brut (tests ou désérialisation JSON)
    elif isinstance(field, dict):
        val = field.get("value")
    elif isinstance(field, str):
        val = field
    else:
        return None
    return val if isinstance(val, str) and val.strip() else None


def _read_fhir_patient_id(fhir_bundle_path: str) -> str | None:
    """Lit le patient_id depuis un bundle FHIR JSON.

    Cherche la ressource Patient dans bundle["entry"] et retourne resource["id"].
    Retourne None si le fichier est absent, illisible ou ne contient pas de Patient.

    Le chemin est relatif à la zone storage/incoming/.
    """
    try:
        resolved = _INCOMING_DIR / fhir_bundle_path
        if not resolved.exists():
            return None
        with resolved.open(encoding="utf-8") as f:
            bundle = json.load(f)
        entries = bundle.get("entry", [])
        for entry in entries:
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "Patient":
                return resource.get("id")
    except Exception:  # noqa: BLE001 — tout échec de lecture FHIR est non bloquant
        return None
    return None


def _fail_identity_result(reason: str) -> IdentityResult:
    """Construit un IdentityResult FAIL propre."""
    return IdentityResult(
        status=VerificationStatus.FAIL,
        reasons=[reason],
    )


def _fail_coverage_result(reason: str) -> CoverageResult:
    """Construit un CoverageResult FAIL propre."""
    return CoverageResult(
        status=VerificationStatus.FAIL,
        reasons=[reason],
    )


def _get_rule_version() -> str:
    """Retourne les versions des règles utilisées par l'agent."""
    try:
        identity = get_rule_version(_RULE_VERSION_FILE)
        coverage = get_rule_version("coverage_rules.yaml")
        authorization = get_rule_version("authorization_rules.yaml")
        return f"identity:{identity};coverage:{coverage};authorization:{authorization}"
    except FileNotFoundError:
        return "1.0.0"


def _contract_value(contract: dict | None, *keys: str) -> object | None:
    """Lit une valeur de contrat sans jamais modifier le snapshot fourni."""
    if not contract:
        return None
    for key in keys:
        if key in contract and contract[key] not in ("", None):
            return contract[key]
    return None


def _to_int(value: object | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: object | None) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes", "y", "active"}:
        return True
    if text in {"false", "0", "no", "n", "inactive"}:
        return False
    return None


def validate_input(
    *,
    case_id: str,
    fhir_bundle_path: str | None = None,
    dossier_patient_id: str | None = None,
    contract: dict | None = None,
    policy_number: str | None = None,
    patient_pseudonym: str | None = None,
    service_date: object | None = None,
    requested_amount: object | None = None,
    total_amount: object | None = None,
    procedure_codes: list[str] | None = None,
    preauthorization_reference: str | None = None,
    extraction_confidence: float | None = None,
    provenance: dict[str, str] | None = None,
) -> IdentityCoverageInput:
    """Valide l'entrée avec Pydantic."""
    return IdentityCoverageInput(
        case_id=case_id,
        fhir_bundle_path=fhir_bundle_path,
        dossier_patient_id=dossier_patient_id,
        contract=deepcopy(contract) if contract is not None else None,
        policy_number=policy_number,
        patient_pseudonym=patient_pseudonym or dossier_patient_id,
        service_date=service_date,
        requested_amount=requested_amount,
        total_amount=total_amount,
        procedure_codes=procedure_codes or [],
        preauthorization_reference=preauthorization_reference,
        extraction_confidence=extraction_confidence,
        provenance=provenance or {},
        rule_version=_get_rule_version(),
    )


def check_input_confidence(icv_input: IdentityCoverageInput) -> tuple[list[str], list[StructuredRuleError]]:
    """Vérifie la fiabilité des champs extraits."""
    warnings: list[str] = []
    errors: list[StructuredRuleError] = []
    if icv_input.extraction_confidence is None:
        warnings.append("Confiance d'extraction absente — revue recommandée")
    elif icv_input.extraction_confidence < 0.65:
        warnings.append("Confiance d'extraction faible — revue humaine requise")
        errors.append(
            StructuredRuleError(
                code="LOW_EXTRACTION_CONFIDENCE",
                message="Confiance d'extraction inférieure au seuil",
                field="extraction_confidence",
                severity="WARNING",
            )
        )
    return warnings, errors


def load_contract(icv_input: IdentityCoverageInput) -> tuple[dict | None, list[object]]:
    """Charge le référentiel et recherche le contrat cible."""
    if icv_input.contract is not None:
        contract = deepcopy(icv_input.contract)
        return contract, [contract]
    if not icv_input.policy_number:
        return None, []
    contracts = load_contracts()
    contract = get_contract(icv_input.policy_number)
    snapshot = contract.to_agent_snapshot() if contract is not None else None
    return snapshot, list(contracts)


def _load_rule_versions() -> str:
    """Charge explicitement les trois jeux de règles et retourne leurs versions."""
    load_rules("identity_rules.yaml")
    load_rules("coverage_rules.yaml")
    load_rules("authorization_rules.yaml")
    return _get_rule_version()


def check_identity(
    *,
    icv_input: IdentityCoverageInput,
    fields_source: dict,
    contract: dict | None,
    candidates: list[object],
    fhir_patient_id: str | None,
) -> tuple[IdentityResult, IdentityCheck | None]:
    """Exécute le contrôle d'identité."""
    ocr_patient_id = _extract_field_value(fields_source, "patient_id")
    ocr_patient_name = _extract_field_value(fields_source, "patient_name")
    dossier_patient_id = icv_input.patient_pseudonym or icv_input.dossier_patient_id
    contract_patient_id = _contract_value(
        contract,
        "patient_id",
        "patient_pseudonym",
        "pseudonymized_patient_id",
        "PATIENT",
        "member_patient_id",
    )

    identity_check: IdentityCheck | None = None
    policy_for_check = icv_input.policy_number or str(_contract_value(contract, "policy_number") or "")
    if policy_for_check:
        identity_check = match_identity_contract(
            dossier_patient_pseudonym=dossier_patient_id or ocr_patient_id,
            policy_number=policy_for_check,
            candidate_contracts=candidates,
            rule_version=get_rule_version("identity_rules.yaml"),
        )

    identity_dict = identity_match_result(
        case_id=icv_input.case_id or "CLM-0000",
        ocr_patient_id=ocr_patient_id,
        ocr_patient_name=ocr_patient_name,
        fhir_patient_id=fhir_patient_id,
        claim_patient_id=dossier_patient_id,
        contract_patient_id=str(contract_patient_id) if contract_patient_id else None,
    )
    if identity_check is not None:
        if identity_check.status == IdentityCheckStatus.MATCH:
            identity_dict["status"] = VerificationStatus.PASS
        elif identity_check.status == IdentityCheckStatus.MISMATCH:
            identity_dict["status"] = VerificationStatus.NEEDS_REVIEW
        elif identity_check.status in {IdentityCheckStatus.AMBIGUOUS, IdentityCheckStatus.NOT_FOUND}:
            identity_dict["status"] = VerificationStatus.NEEDS_REVIEW
        identity_dict["reasons"] = [
            *identity_dict.get("reasons", []),
            f"Contrôle identité contrat : {identity_check.status.value}",
        ]
    return IdentityResult(**identity_dict), identity_check


def check_coverage_period(
    *,
    icv_input: IdentityCoverageInput,
    fields_source: dict,
    contract: dict | None,
) -> CoverageCheck | None:
    """Exécute les contrôles couverture, montants et préautorisation structurés."""
    if contract is None and not icv_input.policy_number:
        return None
    if not (icv_input.policy_number or icv_input.procedure_codes):
        return None
    requested_amount = (
        icv_input.requested_amount
        or _extract_field_value(fields_source, "amount_requested")
        or _extract_field_value(fields_source, "total_billed")
    )
    total_amount = (
        icv_input.total_amount
        or _extract_field_value(fields_source, "total_amount")
        or _extract_field_value(fields_source, "total_billed")
        or requested_amount
    )
    service_date = icv_input.service_date or _extract_field_value(fields_source, "service_date")
    currency = _extract_field_value(fields_source, "currency") or str(_contract_value(contract, "currency") or "")
    procedure_codes = icv_input.procedure_codes or []
    return evaluate_coverage_contract(
        contract=contract,
        service_date=service_date,
        requested_amount=requested_amount,
        total_amount=total_amount,
        currency=currency,
        procedure_codes=procedure_codes,
        preauthorization_reference=icv_input.preauthorization_reference,
    )


def check_amounts(coverage_check: CoverageCheck | None) -> CoverageCheck | None:
    """Étape explicite du pipeline : les montants sont contrôlés dans CoverageCheck."""
    return coverage_check


def check_authorization(
    *,
    icv_input: IdentityCoverageInput,
    coverage_check: CoverageCheck | None,
) -> AuthorizationCheck:
    """Construit le contrôle de préautorisation depuis les preuves de couverture."""
    if coverage_check is None:
        return AuthorizationCheck(
            status=AuthorizationCheckStatus.NOT_EVALUATED,
            rule_version=get_rule_version("authorization_rules.yaml"),
        )
    errors = [
        err for err in coverage_check.structured_errors
        if err.code == "PREAUTHORIZATION_PRESENT_WHEN_REQUIRED"
    ]
    required = any(
        ev.rule_id == "PREAUTHORIZATION_PRESENT_WHEN_REQUIRED" and ev.value is True
        for ev in coverage_check.evidence
    )
    if not required:
        status = AuthorizationCheckStatus.NOT_REQUIRED
    elif errors:
        status = AuthorizationCheckStatus.MISSING
    elif icv_input.preauthorization_reference:
        status = AuthorizationCheckStatus.PRESENT
    else:
        status = AuthorizationCheckStatus.NOT_REQUIRED
    return AuthorizationCheck(
        status=status,
        preauthorization_reference=icv_input.preauthorization_reference,
        required=required,
        procedure_codes=icv_input.procedure_codes,
        evidence=[
            ev for ev in coverage_check.evidence
            if ev.rule_id == "PREAUTHORIZATION_PRESENT_WHEN_REQUIRED"
        ],
        structured_errors=errors,
        rule_version=get_rule_version("authorization_rules.yaml"),
    )


def _legacy_coverage_result(
    *,
    fields_source: dict,
    contract: dict | None,
) -> CoverageResult:
    payer_name = _extract_field_value(fields_source, "payer_name")
    amount_requested = (
        _extract_field_value(fields_source, "amount_requested")
        or _extract_field_value(fields_source, "total_billed")
    )
    patient_share = _extract_field_value(fields_source, "patient_share")
    coverage_rate = _extract_field_value(fields_source, "coverage_rate")
    service_date = (
        _extract_field_value(fields_source, "service_date")
        or _contract_value(contract, "service_date", "SERVICEDATE")
    )
    procedure_count = _to_int(_extract_field_value(fields_source, "procedure_count"))
    medication_count = _to_int(_extract_field_value(fields_source, "medication_count"))
    contract_payer = _contract_value(contract, "payer_name", "payer", "PAYER_NAME")
    coverage_dict = verify_coverage(
        payer_name=payer_name or (str(contract_payer) if contract_payer else None),
        coverage_rate=coverage_rate or _contract_value(contract, "coverage_rate"),
        amount_requested=amount_requested,
        patient_share=patient_share,
        policy_active=_to_bool(_contract_value(contract, "policy_active", "active")),
        service_date=service_date,
        coverage_start_date=_contract_value(contract, "coverage_start_date", "start_date", "START_DATE"),
        coverage_end_date=_contract_value(contract, "coverage_end_date", "end_date", "END_DATE"),
        ceiling_amount=_contract_value(contract, "ceiling_amount", "annual_ceiling", "benefit_limit"),
        ceiling_remaining=_contract_value(contract, "ceiling_remaining", "remaining_ceiling", "remaining_benefit"),
        preauthorization_required=_to_bool(
            _contract_value(contract, "preauthorization_required", "authorization_required", "preauth_required")
        ),
        preauthorization_status=str(
            _contract_value(contract, "preauthorization_status", "authorization_status", "preauth_status") or ""
        )
        or None,
        procedure_count=procedure_count,
        medication_count=medication_count,
    )
    return CoverageResult(**coverage_dict)


def _coverage_result_from_check(coverage_check: CoverageCheck) -> CoverageResult:
    if coverage_check.status == CoverageCheckStatus.ACTIVE:
        status = VerificationStatus.PASS
    elif coverage_check.status == CoverageCheckStatus.NOT_FOUND:
        status = VerificationStatus.NEEDS_REVIEW
    else:
        status = VerificationStatus.FAIL if coverage_check.status in {
            CoverageCheckStatus.INACTIVE,
            CoverageCheckStatus.EXPIRED,
            CoverageCheckStatus.NOT_STARTED,
        } else VerificationStatus.NEEDS_REVIEW
    return CoverageResult(
        status=status,
        payer_name=None,
        service_date=coverage_check.service_date,
        coverage_start_date=coverage_check.coverage_start_date,
        coverage_end_date=coverage_check.coverage_end_date,
        amount_requested=coverage_check.requested_amount,
        ceiling_amount=coverage_check.ceiling_amount,
        ceiling_remaining=coverage_check.ceiling_remaining,
        ceiling_exceeded=any(
            err.code == "REQUESTED_NOT_GREATER_THAN_AVAILABLE_LIMIT"
            for err in coverage_check.structured_errors
        ),
        preauthorization_required=any(
            ev.rule_id == "PREAUTHORIZATION_PRESENT_WHEN_REQUIRED" and ev.value is True
            for ev in coverage_check.evidence
        ),
        reasons=[
            f"Contrôle couverture : {coverage_check.status.value}",
            *[err.message for err in coverage_check.structured_errors],
        ],
    )


def build_evidence(
    *,
    identity_check: IdentityCheck | None,
    coverage_check: CoverageCheck | None,
    authorization_check: AuthorizationCheck,
) -> tuple[list[dict[str, str]], list[str], list[dict[str, str]]]:
    """Construit les preuves, avertissements et erreurs sérialisables."""
    checks = [check for check in (identity_check, coverage_check, authorization_check) if check is not None]
    evidence: list[dict[str, str]] = []
    warnings: list[str] = []
    structured_errors: list[dict[str, str]] = []
    for check in checks:
        for ev in check.evidence:
            evidence.append({
                "source": ev.source,
                "field": ev.field,
                "value": "" if ev.value is None else str(ev.value),
                "rule_id": ev.rule_id or "",
                "rule_version": ev.rule_version or check.rule_version,
            })
        warnings.extend(check.warnings)
        for err in check.structured_errors:
            structured_errors.append({
                "code": err.code,
                "message": err.message,
                "field": err.field or "",
                "severity": err.severity,
            })
    return evidence, warnings, structured_errors


def validate_output(result: IdentityCoverageResult) -> IdentityCoverageResult:
    """Valide le résultat final avec Pydantic."""
    return IdentityCoverageResult.model_validate(result.model_dump())


def _invoke_llm_identity_coverage(data: dict) -> LlmIdentityCoverageDecision | None:
    """Appelle le LLM avec outils explicitement autorisés pour une synthèse d'audit."""
    try:
        llm = get_llm()
        agent = create_react_agent(
            model=llm,
            tools=[charger_contrat, verifier_identite_contrat, verifier_couverture_contrat],
            response_format=LlmIdentityCoverageDecision,
        )
        result = agent.invoke({
            "messages": [
                SystemMessage(content=load_prompt(_AGENT_NAME)),
                HumanMessage(content=json.dumps(data, ensure_ascii=False, default=str)),
            ]
        })
        structured = result.get("structured_response")
        if isinstance(structured, LlmIdentityCoverageDecision):
            return structured
        if isinstance(structured, dict):
            return LlmIdentityCoverageDecision(**structured)
        return None
    except Exception:
        return None


# ── Fonction principale (testable sans LangGraph) ─────────────────────────────


def run(
    case_id: str,
    ocr_result=None,  # DocumentOcrResult | None
    fhir_bundle_path: str | None = None,
    *,
    extracted_fields: dict | None = None,
    dossier_patient_id: str | None = None,
    contract: dict | None = None,
    policy_number: str | None = None,
    patient_pseudonym: str | None = None,
    service_date: object | None = None,
    requested_amount: object | None = None,
    total_amount: object | None = None,
    procedure_codes: list[str] | None = None,
    preauthorization_reference: str | None = None,
    extraction_confidence: float | None = None,
    provenance: dict[str, str] | None = None,
) -> IdentityCoverageResult:
    """Exécute le pipeline de vérification identité + couverture pour un dossier.

    Args:
        case_id          : identifiant du dossier (CLM-XXXX).
        ocr_result       : résultat de l'OCR (DocumentOcrResult | None).
                           Utilisé pour extraire les champs patient et couverture.
        fhir_bundle_path : chemin relatif du bundle FHIR sous incoming/ (optionnel).
        extracted_fields : dict {field_name: str} passé directement pour les tests.
                           Prioritaire sur ocr_result si fourni.
        dossier_patient_id: identité pseudonymisée issue du dossier.
        contract        : snapshot de contrat synthétique, lu en entrée seulement.
        policy_number   : numéro de contrat pour résolution dans le référentiel local.

    Returns:
        IdentityCoverageResult avec identité et couverture vérifiées.
    """
    rule_version = _load_rule_versions()
    extracted_fields = extracted_fields or {}
    icv_input = validate_input(
        case_id=case_id,
        fhir_bundle_path=fhir_bundle_path,
        dossier_patient_id=dossier_patient_id,
        contract=contract,
        policy_number=policy_number,
        patient_pseudonym=(
            patient_pseudonym
            or dossier_patient_id
            or _extract_field_value(extracted_fields, "patient_id")
        ),
        service_date=service_date or _extract_field_value(extracted_fields, "service_date"),
        requested_amount=(
            requested_amount
            or _extract_field_value(extracted_fields, "amount_requested")
            or _extract_field_value(extracted_fields, "total_billed")
        ),
        total_amount=(
            total_amount
            or _extract_field_value(extracted_fields, "total_amount")
            or _extract_field_value(extracted_fields, "total_billed")
        ),
        procedure_codes=procedure_codes or [
            code.strip()
            for code in (_extract_field_value(extracted_fields, "procedure_codes") or "").split(",")
            if code.strip()
        ],
        preauthorization_reference=(
            preauthorization_reference
            or _extract_field_value(extracted_fields, "preauthorization_reference")
        ),
        extraction_confidence=extraction_confidence,
        provenance=provenance or {},
    )
    confidence_warnings, confidence_errors = check_input_confidence(icv_input)
    resolved_contract, candidate_contracts = load_contract(icv_input)

    # ── Étape 3 : extraction des champs depuis ocr_result ou extracted_fields ─
    if extracted_fields:
        fields_source: dict = extracted_fields
    elif ocr_result is not None:
        fields_source = getattr(ocr_result, "extracted_fields", {}) or {}
    else:
        fields_source = {}

    # ── Étape 4 : lecture optionnelle du bundle FHIR ──────────────────────────
    fhir_patient_id: str | None = None
    if fhir_bundle_path:
        fhir_patient_id = _read_fhir_patient_id(fhir_bundle_path)

    # ── Étape 5 : vérification d'identité ────────────────────────────────────
    identity, identity_check = check_identity(
        icv_input=icv_input,
        fields_source=fields_source,
        contract=resolved_contract,
        candidates=candidate_contracts,
        fhir_patient_id=fhir_patient_id,
    )

    # ── Étape 6 : vérification de couverture ─────────────────────────────────
    coverage_check = check_coverage_period(
        icv_input=icv_input,
        fields_source=fields_source,
        contract=resolved_contract,
    )
    coverage_check = check_amounts(coverage_check)
    authorization_check = check_authorization(icv_input=icv_input, coverage_check=coverage_check)
    if coverage_check is not None:
        coverage = _coverage_result_from_check(coverage_check)
    else:
        coverage = _legacy_coverage_result(fields_source=fields_source, contract=resolved_contract)

    evidence, warnings, structured_errors = build_evidence(
        identity_check=identity_check,
        coverage_check=coverage_check,
        authorization_check=authorization_check,
    )
    warnings.extend(confidence_warnings)
    structured_errors.extend(
        {
            "code": err.code,
            "message": err.message,
            "field": err.field or "",
            "severity": err.severity,
        }
        for err in confidence_errors
    )

    llm_decision = _invoke_llm_identity_coverage({
        "case_id": case_id,
        "identity_status": identity.status.value,
        "coverage_status": coverage.status.value,
        "rule_version": rule_version,
        "policy_number_present": bool(icv_input.policy_number),
        "service_date_present": icv_input.service_date is not None,
        "contract_loaded": resolved_contract is not None,
        "evidence": evidence[:20],
        "structured_errors": structured_errors[:20],
        "warnings": warnings[:20],
        "instruction": (
            "Synthèse uniquement : ne crée ni contrat, ni patient, ni ressource, "
            "et ne remplace jamais les statuts déterministes."
        ),
    })
    if llm_decision is not None:
        if llm_decision.rationale:
            warnings.append(llm_decision.rationale)
        warnings.extend(llm_decision.warnings)

    # ── Étape 7 : construction du résultat ───────────────────────────────────
    return validate_output(IdentityCoverageResult(
        case_id=case_id,
        identity=identity,
        coverage=coverage,
        rule_version=rule_version,
        evidence=evidence,
        warnings=warnings,
        structured_errors=structured_errors,
        llm_metadata=build_llm_metadata(_AGENT_NAME),
    ))


# ── Nœud LangGraph ────────────────────────────────────────────────────────────


def node(state: ClaimState) -> dict:
    """Nœud LangGraph — lit identity_coverage_input + ocr_result et délègue à run().

    Attend dans le state :
        case_id                  : identifiant du dossier.
        identity_coverage_input  : dict avec case_id + fhir_bundle_path (optionnel).
        ocr_result               : DocumentOcrResult (optionnel — NEEDS_REVIEW si absent).

    Retourne :
        identity_coverage_input  → None (consommé)
        identity_coverage_result → IdentityCoverageResult
        completed_steps          → ["identity_coverage"]
        current_step             → "identity_coverage"
        + erreurs ou alertes selon le statut
    """
    case_id: str = state.get("case_id", "CLM-0000")  # type: ignore[assignment]
    raw: dict = state.get("identity_coverage_input", {}) or {}  # type: ignore[assignment]
    ocr_result = state.get("ocr_result")

    # ── Validation de l'entrée ────────────────────────────────────────────────
    if not case_id or not case_id.startswith("CLM-"):
        # Cas d'erreur : case_id invalide
        fail_result = IdentityCoverageResult(
            case_id=case_id or "CLM-0000",
            identity=_fail_identity_result("case_id invalide ou absent"),
            coverage=_fail_coverage_result("case_id invalide ou absent"),
            rule_version=_get_rule_version(),
        )
        updates = _build_updates(case_id or "CLM-0000", fail_result, fail=True)
        validate_state_update(updates)
        return updates

    try:
        icv_input = IdentityCoverageInput(
            case_id=raw.get("case_id", case_id),
            claim_id=raw.get("claim_id"),
            patient_pseudonym=raw.get("patient_pseudonym"),
            policy_number=raw.get("policy_number"),
            service_date=raw.get("service_date"),
            requested_amount=raw.get("requested_amount"),
            total_amount=raw.get("total_amount"),
            procedure_codes=raw.get("procedure_codes", []),
            preauthorization_reference=raw.get("preauthorization_reference"),
            extraction_confidence=raw.get("extraction_confidence"),
            provenance=raw.get("provenance", {}),
            fhir_bundle_path=raw.get("fhir_bundle_path"),
            dossier_patient_id=raw.get("dossier_patient_id"),
            contract=raw.get("contract"),
        )
    except ValidationError as exc:
        first_msg = (
            exc.errors()[0].get("msg", "Entrée invalide")
            if exc.errors()
            else "Erreur de validation"
        )
        fail_result = IdentityCoverageResult(
            case_id=case_id,
            identity=_fail_identity_result(f"Entrée invalide : {first_msg}"),
            coverage=_fail_coverage_result(f"Entrée invalide : {first_msg}"),
            rule_version=_get_rule_version(),
        )
        updates = _build_updates(case_id, fail_result, fail=True)
        validate_state_update(updates)
        return updates

    # ── Exécution du pipeline principal ──────────────────────────────────────
    result = run(
        case_id=icv_input.case_id,
        ocr_result=ocr_result,
        fhir_bundle_path=icv_input.fhir_bundle_path,
        dossier_patient_id=icv_input.dossier_patient_id,
        contract=icv_input.contract,
        policy_number=icv_input.policy_number,
        patient_pseudonym=icv_input.patient_pseudonym,
        service_date=icv_input.service_date,
        requested_amount=icv_input.requested_amount,
        total_amount=icv_input.total_amount,
        procedure_codes=icv_input.procedure_codes,
        preauthorization_reference=icv_input.preauthorization_reference,
        extraction_confidence=icv_input.extraction_confidence,
        provenance=icv_input.provenance,
    )

    # ── Construction des mises à jour du state ────────────────────────────────
    updates = _build_updates(case_id, result)
    validate_state_update(updates)
    return updates


def _build_updates(
    case_id: str,
    result: IdentityCoverageResult,
    *,
    fail: bool = False,
) -> dict:
    """Construit le dict de mises à jour du ClaimState."""
    audit_event = AuditEvent(
        event_id=str(uuid.uuid4()),
        case_id=case_id,
        actor="identity_coverage_agent",
        action="identity_coverage_verification",
        outcome=result.identity.status.value,
        details={
            "identity_status": result.identity.status.value,
            "coverage_status": result.coverage.status.value,
            "rule_version": result.rule_version,
        },
    )

    updates: dict = {
        "identity_coverage_input": None,
        "identity_coverage_result": result,
        "current_step": "identity_coverage",
        "completed_steps": ["identity_coverage"],
        "audit_trail": [audit_event],
    }

    # Alertes et erreurs selon les statuts
    identity_status = result.identity.status
    coverage_status = result.coverage.status

    errors: list[str] = []
    alerts: list[str] = []

    if identity_status == VerificationStatus.FAIL or fail:
        errors.extend(
            f"[identity_coverage] {r}" for r in result.identity.reasons
        )
    elif identity_status == VerificationStatus.NEEDS_REVIEW:
        alerts.append(
            f"Identité patient : NEEDS_REVIEW — {'; '.join(result.identity.reasons)}"
        )

    if coverage_status == VerificationStatus.FAIL:
        errors.extend(
            f"[identity_coverage] {r}" for r in result.coverage.reasons
        )
    elif coverage_status == VerificationStatus.NEEDS_REVIEW:
        alerts.append(
            f"Couverture : NEEDS_REVIEW — {'; '.join(result.coverage.reasons)}"
        )

    if errors:
        updates["errors"] = errors
    if alerts:
        updates["alerts"] = alerts

    return updates
