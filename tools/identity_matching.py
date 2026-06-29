"""Fonctions de comparaison d'identité patient — ClaimShield Santé.

Fonctions pures déterministes — aucun appel LLM, aucun effet de bord.
Utilisées par l'Identity and Coverage Agent pour vérifier la cohérence
des identifiants patient issus de sources multiples (OCR, FHIR, dossier).
"""
from __future__ import annotations

import re
from typing import Iterable

from agents.identity_coverage_agent.schemas import (
    IdentityCheck,
    IdentityCheckStatus,
    RuleEvidence,
    StructuredRuleError,
)

_PSEUDONYM_RE = re.compile(r"^PAT-[A-Z0-9][A-Z0-9-]{2,64}$", re.IGNORECASE)


def _normalize_id(value: str | None) -> str | None:
    """Normalise un identifiant : strip + casefold. Retourne None si vide."""
    if value is None:
        return None
    normalized = value.strip().casefold()
    return normalized if normalized else None


def _normalize_name(value: str | None) -> str | None:
    """Normalise un nom : strip + casefold. Retourne None si vide."""
    if value is None:
        return None
    normalized = value.strip().casefold()
    return normalized if normalized else None


def _normalize_policy(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().upper()
    return normalized if normalized else None


def _is_well_formed_pseudonym(value: str | None) -> bool:
    return bool(value and _PSEUDONYM_RE.fullmatch(value.strip()))


def _contract_field(contract: object, field: str) -> object | None:
    if isinstance(contract, dict):
        return contract.get(field)
    return getattr(contract, field, None)


def match_identity_contract(
    *,
    dossier_patient_pseudonym: str | None,
    policy_number: str | None,
    candidate_contracts: Iterable[object],
    rule_version: str = "1.0.0",
) -> IdentityCheck:
    """Compare l'identité pseudonymisée du dossier avec les contrats candidats.

    Cette fonction ne compare jamais sur un nom libre. Elle exige un pseudonyme
    patient et un numéro de contrat, puis conserve la règle appliquée et les
    champs comparés sans exposer d'identité brute dans les messages.
    """
    compared_fields = ["dossier_patient_pseudonym", "policy_number"]
    evidence = [
        RuleEvidence(source="claim", field="patient_pseudonym", value="present" if dossier_patient_pseudonym else None),
        RuleEvidence(source="claim", field="policy_number", value=policy_number),
    ]

    if not dossier_patient_pseudonym or not policy_number:
        return IdentityCheck(
            status=IdentityCheckStatus.NOT_FOUND,
            rule_applied="IDENTITY_REQUIRED_FIELDS_PRESENT",
            compared_fields=compared_fields,
            dossier_patient_pseudonym=dossier_patient_pseudonym,
            evidence=evidence,
            structured_errors=[
                StructuredRuleError(
                    code="IDENTITY_REQUIRED_FIELD_MISSING",
                    message="Pseudonyme patient ou numéro de contrat absent",
                    field="patient_pseudonym|policy_number",
                )
            ],
            rule_version=rule_version,
        )

    if not _is_well_formed_pseudonym(dossier_patient_pseudonym):
        return IdentityCheck(
            status=IdentityCheckStatus.NOT_FOUND,
            rule_applied="IDENTITY_PSEUDONYM_FORMAT",
            compared_fields=compared_fields,
            dossier_patient_pseudonym=dossier_patient_pseudonym,
            evidence=evidence,
            structured_errors=[
                StructuredRuleError(
                    code="IDENTITY_PSEUDONYM_MALFORMED",
                    message="Pseudonyme patient mal formé",
                    field="patient_pseudonym",
                )
            ],
            rule_version=rule_version,
        )

    normalized_policy = _normalize_policy(policy_number)
    candidates = [
        contract
        for contract in candidate_contracts
        if _normalize_policy(str(_contract_field(contract, "policy_number") or "")) == normalized_policy
    ]

    if not candidates:
        return IdentityCheck(
            status=IdentityCheckStatus.NOT_FOUND,
            rule_applied="IDENTITY_POLICY_NUMBER_LOOKUP",
            compared_fields=compared_fields,
            dossier_patient_pseudonym=dossier_patient_pseudonym,
            evidence=evidence,
            warnings=["Aucun contrat candidat trouvé pour le numéro fourni"],
            rule_version=rule_version,
        )

    if len(candidates) > 1:
        return IdentityCheck(
            status=IdentityCheckStatus.AMBIGUOUS,
            rule_applied="IDENTITY_POLICY_NUMBER_LOOKUP",
            compared_fields=compared_fields,
            dossier_patient_pseudonym=dossier_patient_pseudonym,
            evidence=evidence,
            warnings=["Plusieurs contrats candidats correspondent au numéro fourni"],
            structured_errors=[
                StructuredRuleError(
                    code="IDENTITY_MULTIPLE_CONTRACT_CANDIDATES",
                    message="Plusieurs contrats candidats correspondent",
                    field="policy_number",
                    severity="WARNING",
                )
            ],
            rule_version=rule_version,
        )

    contract = candidates[0]
    contract_patient = str(_contract_field(contract, "patient_pseudonym") or "")
    evidence.append(RuleEvidence(source="contract", field="patient_pseudonym", value="present"))
    evidence.append(RuleEvidence(source="contract", field="policy_number", value=policy_number))

    matched = _normalize_id(dossier_patient_pseudonym) == _normalize_id(contract_patient)
    return IdentityCheck(
        status=IdentityCheckStatus.MATCH if matched else IdentityCheckStatus.MISMATCH,
        rule_applied="IDENTITY_PATIENT_ID_MATCH",
        compared_fields=compared_fields,
        patient_pseudonym=dossier_patient_pseudonym,
        contract_patient_pseudonym=contract_patient,
        dossier_patient_pseudonym=dossier_patient_pseudonym,
        evidence=evidence,
        warnings=[
            f"Champs comparés : {', '.join(compared_fields)}",
        ],
        structured_errors=[]
        if matched
        else [
            StructuredRuleError(
                code="IDENTITY_CONTRACT_MISMATCH",
                message="Le pseudonyme dossier ne correspond pas au contrat",
                field="patient_pseudonym",
            )
        ],
        rule_version=rule_version,
    )


def compare_patient_ids(
    *,
    ocr_patient_id: str | None,
    fhir_patient_id: str | None,
    claim_patient_id: str | None,
    contract_patient_id: str | None = None,
) -> tuple[bool, list[str]]:
    """Compare les identifiants patient issus de sources multiples.

    Règles :
      - Normalisation : strip + casefold sur chaque ID disponible.
      - Un ID absent (None ou vide) ne compte pas comme source.
      - Si tous les IDs présents concordent → True (concordance).
      - Si deux sources ou plus sont présentes et discordent → False (incohérence).
      - Si aucun ID fourni → False avec motif.
      - min_sources_agree : au moins 2 sources concordantes → PASS.

    Args:
        ocr_patient_id  : ID extrait par OCR.
        fhir_patient_id : ID extrait du bundle FHIR.
        claim_patient_id: ID pseudonymisé du dossier.
        contract_patient_id: ID associé au contrat synthétique.

    Returns:
        (concordance: bool, motifs: list[str])
    """
    sources: dict[str, str] = {}
    if (n := _normalize_id(ocr_patient_id)) is not None:
        sources["ocr"] = n
    if (n := _normalize_id(fhir_patient_id)) is not None:
        sources["fhir"] = n
    if (n := _normalize_id(claim_patient_id)) is not None:
        sources["claim"] = n
    if (n := _normalize_id(contract_patient_id)) is not None:
        sources["contract"] = n

    if not sources:
        return False, ["Aucun identifiant patient disponible (OCR, FHIR, dossier)"]

    unique_values = set(sources.values())
    if len(unique_values) == 1:
        source_labels = ", ".join(sources.keys())
        return True, [f"Identifiants concordants ({source_labels})"]

    # Des incohérences existent
    motifs: list[str] = []
    source_list = list(sources.items())
    for i, (src_a, val_a) in enumerate(source_list):
        for src_b, val_b in source_list[i + 1 :]:
            if val_a != val_b:
                motifs.append(
                    f"Incohérence identifiant : {src_a}={val_a!r} ≠ {src_b}={val_b!r}"
                )

    return False, motifs


def check_patient_name(name: str | None) -> tuple[bool, str]:
    """Vérifie qu'un nom patient est présent et non vide.

    Args:
        name: Nom patient extrait (peut être None).

    Returns:
        (valide: bool, message: str)
    """
    if name is None or not name.strip():
        return False, "Nom patient absent ou vide"
    return True, f"Nom patient présent : {name.strip()!r}"


def identity_match_result(
    *,
    case_id: str,
    ocr_patient_id: str | None,
    ocr_patient_name: str | None,
    fhir_patient_id: str | None,
    claim_patient_id: str | None,
    contract_patient_id: str | None = None,
) -> dict:
    """Calcule le résultat de vérification d'identité patient.

    Logique :
      - Si aucun ID disponible → FAIL.
      - Si incohérence entre sources → NEEDS_REVIEW avec motifs.
      - Si nom absent → NEEDS_REVIEW.
      - Si tout concordant → PASS.

    ID canonique : préférence fhir > ocr > claim.

    Args:
        case_id         : identifiant du dossier (pour traçabilité).
        ocr_patient_id  : ID extrait par OCR (peut être None).
        ocr_patient_name: Nom extrait par OCR (peut être None).
        fhir_patient_id : ID extrait du bundle FHIR (peut être None).
        claim_patient_id: ID pseudonymisé du dossier (peut être None).
        contract_patient_id: ID associé au contrat synthétique (peut être None).

    Returns:
        dict compatible avec IdentityResult.
    """
    from schemas.domain import VerificationStatus

    reasons: list[str] = []

    # ── Vérification des IDs ──────────────────────────────────────────────────
    ids_ok, id_reasons = compare_patient_ids(
        ocr_patient_id=ocr_patient_id,
        fhir_patient_id=fhir_patient_id,
        claim_patient_id=claim_patient_id,
        contract_patient_id=contract_patient_id,
    )
    reasons.extend(id_reasons)

    # ── Vérification du nom ───────────────────────────────────────────────────
    name_ok, name_reason = check_patient_name(ocr_patient_name)
    reasons.append(name_reason)

    # ── ID canonique (priorité : fhir > ocr > claim) ─────────────────────────
    canonical_id = (
        _normalize_id(fhir_patient_id)
        or _normalize_id(ocr_patient_id)
        or _normalize_id(claim_patient_id)
        or _normalize_id(contract_patient_id)
    )

    # ── Calcul du statut ──────────────────────────────────────────────────────
    if not ids_ok and canonical_id is None:
        # Aucun ID disponible
        status = VerificationStatus.FAIL
    elif not ids_ok or not name_ok:
        # Incohérence ou nom absent
        status = VerificationStatus.NEEDS_REVIEW
    else:
        status = VerificationStatus.PASS

    # Conserver les IDs bruts (non normalisés) pour la traçabilité
    return {
        "status": status,
        "patient_id": canonical_id,
        "patient_name": ocr_patient_name.strip() if ocr_patient_name else None,
        "source_patient_id": fhir_patient_id,
        "claim_patient_id": claim_patient_id,
        "contract_patient_id": contract_patient_id,
        "encounter_patient_id": ocr_patient_id,
        "reasons": reasons,
    }
