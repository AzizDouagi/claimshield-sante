"""Fonctions de vérification de couverture assurance — ClaimShield Santé.

Fonctions pures déterministes — aucun appel LLM, aucun effet de bord.
Utilisées par l'Identity and Coverage Agent pour vérifier la couverture
assurance maladie d'un dossier de remboursement.
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from agents.identity_coverage_agent.schemas import (
    CoverageCheck,
    CoverageCheckStatus,
    RuleEvidence,
    StructuredRuleError,
)
from tools.rule_loader import get_rule_version, load_rules


def _to_decimal(value: str | int | float | Decimal | None) -> Decimal | None:
    """Convertit une valeur en Decimal. Retourne None si absent ou invalide."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _to_date(value: str | date | datetime | None) -> date | None:
    """Convertit une valeur ISO date/datetime en date."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None


def calculate_patient_share(total: Decimal, coverage_rate: Decimal) -> Decimal:
    """Calcule la part patient à partir du montant total et du taux de couverture.

    Formule : patient_share = total * (1 - coverage_rate)
    Arrondi à 2 décimales selon ROUND_HALF_UP.

    Args:
        total         : montant total demandé en Decimal.
        coverage_rate : taux de couverture en [0.0, 1.0] en Decimal.

    Returns:
        Montant à charge du patient en Decimal, arrondi à 2 décimales.
    """
    from decimal import ROUND_HALF_UP
    patient_share = total * (Decimal("1") - coverage_rate)
    return patient_share.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _contract_value(contract: object | None, *keys: str) -> object | None:
    if contract is None:
        return None
    for key in keys:
        if isinstance(contract, dict) and key in contract:
            return contract[key]
        if hasattr(contract, key):
            return getattr(contract, key)
    return None


def _code_set(value: object | None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value.strip()} if value.strip() else set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return set()


def _evidence(rule_id: str, field: str, value: object, rule_version: str) -> RuleEvidence:
    if isinstance(value, Decimal | date | bool) or value is None:
        clean_value = value
    else:
        clean_value = str(value)
    return RuleEvidence(
        source="coverage_rules",
        field=field,
        value=clean_value,
        rule_id=rule_id,
        rule_version=rule_version,
    )


def _error(rule_id: str, message: str, field: str, severity: str = "ERROR") -> StructuredRuleError:
    return StructuredRuleError(
        code=rule_id,
        message=message,
        field=field,
        severity=severity,
    )


def evaluate_coverage_contract(
    *,
    contract: object | None,
    service_date: str | date | datetime | None,
    requested_amount: str | Decimal | None,
    total_amount: str | Decimal | None,
    currency: str | None,
    procedure_codes: list[str] | tuple[str, ...] | None = None,
    preauthorization_reference: str | None = None,
    available_limit: str | Decimal | None = None,
) -> CoverageCheck:
    """Évalue les règles de couverture contrat + dossier.

    Retourne une preuve pour chaque contrôle et enregistre la version du jeu de
    règles appliqué. Ne retourne jamais le référentiel complet des contrats.
    """
    rule_version = get_rule_version("coverage_rules.yaml")
    evidence: list[RuleEvidence] = []
    errors: list[StructuredRuleError] = []
    warnings: list[str] = []

    def add(rule_id: str, field: str, value: object, ok: bool, message: str) -> None:
        evidence.append(_evidence(rule_id, field, ok, rule_version))
        if not ok:
            errors.append(_error(rule_id, message, field))

    contract_exists = contract is not None
    add("CONTRACT_EXISTS", "contract", contract_exists, contract_exists, "Contrat introuvable")
    if not contract_exists:
        return CoverageCheck(
            status=CoverageCheckStatus.NOT_FOUND,
            service_date=_to_date(service_date),
            requested_amount=_to_decimal(requested_amount),
            total_amount=_to_decimal(total_amount),
            evidence=evidence,
            structured_errors=errors,
            warnings=warnings,
            rule_version=rule_version,
        )

    policy_number = _contract_value(contract, "policy_number")
    status = str(_contract_value(contract, "status") or "").casefold()
    contract_active = status == "active" or _contract_value(contract, "policy_active") is True
    add("CONTRACT_ACTIVE", "contract.status", contract_active, contract_active, "Contrat inactif")

    service_dt = _to_date(service_date)
    start_dt = _to_date(_contract_value(contract, "start_date", "coverage_start_date"))
    end_dt = _to_date(_contract_value(contract, "end_date", "coverage_end_date"))
    requested = _to_decimal(requested_amount)
    total = _to_decimal(total_amount)
    limit = (
        _to_decimal(available_limit)
        or _to_decimal(_contract_value(contract, "ceiling_remaining", "available_limit"))
        or _to_decimal(_contract_value(contract, "annual_limit", "ceiling_amount"))
    )

    on_or_after_start = service_dt is not None and start_dt is not None and service_dt >= start_dt
    add(
        "SERVICE_DATE_ON_OR_AFTER_START",
        "service_date",
        service_dt,
        on_or_after_start,
        "Date de soin antérieure au début de couverture",
    )
    on_or_before_end = service_dt is not None and end_dt is not None and service_dt <= end_dt
    add(
        "SERVICE_DATE_ON_OR_BEFORE_END",
        "service_date",
        service_dt,
        on_or_before_end,
        "Date de soin postérieure à la fin de couverture",
    )

    contract_currency = str(_contract_value(contract, "currency") or "").upper()
    claim_currency = str(currency or "").upper()
    currency_matches = bool(claim_currency) and claim_currency == contract_currency
    add(
        "CURRENCY_MATCHES_CONTRACT",
        "currency",
        claim_currency,
        currency_matches,
        "Devise dossier différente de la devise contrat",
    )

    amount_positive = requested is not None and requested > Decimal("0")
    add(
        "REQUESTED_AMOUNT_POSITIVE",
        "requested_amount",
        requested,
        amount_positive,
        "Montant demandé absent, invalide ou non positif",
    )

    not_greater_than_total = (
        requested is not None and total is not None and requested <= total
    )
    add(
        "REQUESTED_NOT_GREATER_THAN_TOTAL",
        "requested_amount",
        requested,
        not_greater_than_total,
        "Montant demandé supérieur au montant total",
    )

    within_limit = requested is not None and limit is not None and requested <= limit
    add(
        "REQUESTED_NOT_GREATER_THAN_AVAILABLE_LIMIT",
        "requested_amount",
        requested,
        within_limit,
        "Montant demandé supérieur au plafond disponible",
    )

    requested_codes = _code_set(procedure_codes)
    covered_codes = _code_set(_contract_value(contract, "covered_procedure_codes"))
    excluded_codes = _code_set(_contract_value(contract, "excluded_procedure_codes"))
    required_preauth_codes = _code_set(_contract_value(contract, "preauthorization_required_for"))

    covered = bool(requested_codes) and requested_codes.issubset(covered_codes)
    add(
        "PROCEDURE_CODE_COVERED",
        "procedure_codes",
        ",".join(sorted(requested_codes)),
        covered,
        "Au moins un acte demandé n'est pas couvert",
    )

    not_excluded = bool(requested_codes) and requested_codes.isdisjoint(excluded_codes)
    add(
        "PROCEDURE_CODE_NOT_EXCLUDED",
        "procedure_codes",
        ",".join(sorted(requested_codes)),
        not_excluded,
        "Au moins un acte demandé est exclu",
    )

    preauth_required = bool(requested_codes.intersection(required_preauth_codes))
    preauth_ok = (not preauth_required) or bool(preauthorization_reference)
    evidence.append(
        _evidence(
            "PREAUTHORIZATION_PRESENT_WHEN_REQUIRED",
            "preauthorization_reference",
            preauth_required,
            rule_version,
        )
    )
    if not preauth_ok:
        errors.append(
            _error(
                "PREAUTHORIZATION_PRESENT_WHEN_REQUIRED",
                "Préautorisation obligatoire absente",
                "preauthorization_reference",
                severity="WARNING",
            )
        )

    if not contract_active:
        final_status = CoverageCheckStatus.INACTIVE
    elif service_dt is not None and start_dt is not None and service_dt < start_dt:
        final_status = CoverageCheckStatus.NOT_STARTED
    elif service_dt is not None and end_dt is not None and service_dt > end_dt:
        final_status = CoverageCheckStatus.EXPIRED
    elif errors:
        final_status = CoverageCheckStatus.AMBIGUOUS
    else:
        final_status = CoverageCheckStatus.ACTIVE

    return CoverageCheck(
        status=final_status,
        policy_number=str(policy_number) if policy_number else None,
        service_date=service_dt,
        coverage_start_date=start_dt,
        coverage_end_date=end_dt,
        requested_amount=requested,
        total_amount=total,
        ceiling_amount=_to_decimal(_contract_value(contract, "annual_limit", "ceiling_amount")),
        ceiling_remaining=limit,
        evidence=evidence,
        structured_errors=errors,
        warnings=warnings,
        rule_version=rule_version,
    )


def verify_coverage(
    *,
    payer_name: str | None,
    coverage_rate: str | Decimal | None,
    amount_requested: str | Decimal | None,
    patient_share: str | Decimal | None = None,
    policy_active: bool | None = None,
    service_date: str | date | datetime | None = None,
    coverage_start_date: str | date | datetime | None = None,
    coverage_end_date: str | date | datetime | None = None,
    ceiling_amount: str | Decimal | None = None,
    ceiling_remaining: str | Decimal | None = None,
    preauthorization_required: bool | None = None,
    preauthorization_status: str | None = None,
    procedure_count: int | None = None,
    medication_count: int | None = None,
) -> dict:
    """Vérifie les données de couverture assurance et calcule les montants manquants.

    Logique :
      - payer_name obligatoire → FAIL si absent.
      - amount_requested obligatoire → FAIL si absent.
      - coverage_rate optionnel → NEEDS_REVIEW si absent mais amount_requested présent.
      - patient_share calculée si coverage_rate et amount_requested présents et patient_share absent.
      - PASS si payer_name + amount_requested + coverage_rate présents et règles contrat OK.
      - FAIL si date hors couverture ou plafond dépassé.
      - NEEDS_REVIEW si préautorisation requise absente/non approuvée.
      - Tous les montants sont retournés en Decimal.

    Args:
        payer_name      : nom de l'assureur (obligatoire).
        coverage_rate   : taux de couverture en [0.0, 1.0] (optionnel).
        amount_requested: montant total demandé (obligatoire).
        patient_share   : part patient déjà calculée (optionnel — calculée si absent).
        policy_active   : statut de la police (optionnel).

    Returns:
        dict compatible avec CoverageResult.
    """
    from schemas.domain import VerificationStatus

    coverage_rules = load_rules("coverage_rules.yaml")
    authorization_rules = load_rules("authorization_rules.yaml")
    reasons: list[str] = []

    # ── Validation payer_name ─────────────────────────────────────────────────
    clean_payer = payer_name.strip() if payer_name else None
    if not clean_payer:
        return {
            "status": VerificationStatus.FAIL,
            "payer_name": None,
            "source_payer_name": None,
            "coverage_rate": None,
            "amount_requested": None,
            "patient_share": None,
            "policy_active": policy_active,
            "service_date": _to_date(service_date),
            "coverage_start_date": _to_date(coverage_start_date),
            "coverage_end_date": _to_date(coverage_end_date),
            "ceiling_amount": _to_decimal(ceiling_amount),
            "ceiling_remaining": _to_decimal(ceiling_remaining),
            "ceiling_exceeded": None,
            "preauthorization_required": preauthorization_required,
            "preauthorization_status": preauthorization_status,
            "reasons": ["Assureur (payer_name) absent ou vide — couverture non vérifiable"],
        }

    # ── Validation amount_requested ───────────────────────────────────────────
    dec_amount = _to_decimal(amount_requested)
    if dec_amount is None:
        return {
            "status": VerificationStatus.FAIL,
            "payer_name": clean_payer,
            "source_payer_name": clean_payer,
            "coverage_rate": None,
            "amount_requested": None,
            "patient_share": None,
            "policy_active": policy_active,
            "service_date": _to_date(service_date),
            "coverage_start_date": _to_date(coverage_start_date),
            "coverage_end_date": _to_date(coverage_end_date),
            "ceiling_amount": _to_decimal(ceiling_amount),
            "ceiling_remaining": _to_decimal(ceiling_remaining),
            "ceiling_exceeded": None,
            "preauthorization_required": preauthorization_required,
            "preauthorization_status": preauthorization_status,
            "reasons": ["Montant demandé (amount_requested) absent ou invalide — couverture non vérifiable"],
        }

    reasons.append(f"Assureur : {clean_payer!r}")
    reasons.append(f"Montant demandé : {dec_amount}")

    # ── coverage_rate (optionnel) ─────────────────────────────────────────────
    dec_rate = _to_decimal(coverage_rate)

    # ── patient_share : calculée ou fournie ───────────────────────────────────
    dec_share = _to_decimal(patient_share)
    if dec_share is None and dec_rate is not None:
        dec_share = calculate_patient_share(dec_amount, dec_rate)
        reasons.append(f"Part patient calculée : {dec_share} (taux {dec_rate})")
    elif dec_share is not None:
        reasons.append(f"Part patient fournie : {dec_share}")

    # ── Statut ────────────────────────────────────────────────────────────────
    status = VerificationStatus.PASS

    if dec_rate is None:
        status = VerificationStatus.NEEDS_REVIEW
        reasons.append("Taux de couverture absent — revue humaine requise")
    else:
        reasons.append(f"Taux de couverture : {dec_rate}")

    bounds = coverage_rules.get("coverage_rate_bounds", {})
    min_rate = _to_decimal(bounds.get("min"))
    max_rate = _to_decimal(bounds.get("max"))
    if dec_rate is not None and (
        (min_rate is not None and dec_rate < min_rate)
        or (max_rate is not None and dec_rate > max_rate)
    ):
        status = VerificationStatus.FAIL
        reasons.append("Taux de couverture hors bornes configurées")

    service_dt = _to_date(service_date)
    start_dt = _to_date(coverage_start_date)
    end_dt = _to_date(coverage_end_date)
    if service_dt is None:
        if start_dt is not None or end_dt is not None:
            status = VerificationStatus.NEEDS_REVIEW if status != VerificationStatus.FAIL else status
            reasons.append("Date de soin absente ou invalide — période de couverture non vérifiable")
    else:
        reasons.append(f"Date de soin : {service_dt.isoformat()}")
        if start_dt is not None and service_dt < start_dt:
            status = VerificationStatus.FAIL
            reasons.append("Date de soin antérieure au début de couverture")
        if end_dt is not None and service_dt > end_dt:
            status = VerificationStatus.FAIL
            reasons.append("Date de soin postérieure à la fin de couverture")

    if policy_active is False:
        status = VerificationStatus.FAIL
        reasons.append("Contrat synthétique inactif à la date d'évaluation")
    elif policy_active is True:
        reasons.append("Contrat synthétique marqué actif")

    dec_ceiling = _to_decimal(ceiling_amount)
    dec_remaining = _to_decimal(ceiling_remaining)
    ceiling_exceeded: bool | None = None
    ceiling_reference = dec_remaining if dec_remaining is not None else dec_ceiling
    if ceiling_reference is not None:
        ceiling_exceeded = dec_amount > ceiling_reference
        if ceiling_exceeded:
            status = VerificationStatus.FAIL
            reasons.append("Montant demandé supérieur au plafond disponible")
        else:
            reasons.append("Montant demandé compatible avec le plafond disponible")

    amount_threshold = _to_decimal(authorization_rules.get("preauth_threshold_usd"))
    procedure_threshold = authorization_rules.get("preauth_procedure_count")
    medication_threshold = authorization_rules.get("preauth_medication_count")
    derived_preauth_required = bool(preauthorization_required)
    if amount_threshold is not None and dec_amount >= amount_threshold:
        derived_preauth_required = True
    if procedure_count is not None and procedure_threshold is not None:
        derived_preauth_required = derived_preauth_required or procedure_count >= int(procedure_threshold)
    if medication_count is not None and medication_threshold is not None:
        derived_preauth_required = derived_preauth_required or medication_count >= int(medication_threshold)

    normalized_auth_status = preauthorization_status.strip().casefold() if preauthorization_status else None
    if derived_preauth_required:
        if normalized_auth_status not in {"approved", "valid", "authorized"}:
            status = VerificationStatus.NEEDS_REVIEW if status != VerificationStatus.FAIL else status
            reasons.append("Préautorisation requise mais non approuvée dans le contrat synthétique")
        else:
            reasons.append("Préautorisation requise et approuvée")
    else:
        reasons.append("Préautorisation non requise selon les règles versionnées")

    return {
        "status": status,
        "payer_name": clean_payer,
        "source_payer_name": clean_payer,
        "coverage_rate": dec_rate,
        "amount_requested": dec_amount,
        "patient_share": dec_share,
        "policy_active": policy_active,
        "service_date": service_dt,
        "coverage_start_date": start_dt,
        "coverage_end_date": end_dt,
        "ceiling_amount": dec_ceiling,
        "ceiling_remaining": dec_remaining,
        "ceiling_exceeded": ceiling_exceeded,
        "preauthorization_required": derived_preauth_required,
        "preauthorization_status": preauthorization_status,
        "reasons": reasons,
    }
