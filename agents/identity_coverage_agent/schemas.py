"""Schémas d'entrée du Identity and Coverage Agent — ClaimShield Santé."""
from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from schemas.domain import StrictModel
from schemas.results import CoverageResult, IdentityCoverageResult, IdentityResult

_ABSOLUTE_PATH_RE = re.compile(r"^(?:/|[A-Za-z]:[/\\]|\\\\)")


class IdentityCheckStatus(str, Enum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_FOUND = "NOT_FOUND"
    NOT_EVALUATED = "NOT_EVALUATED"


class CoverageCheckStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    EXPIRED = "EXPIRED"
    NOT_STARTED = "NOT_STARTED"
    NOT_FOUND = "NOT_FOUND"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_EVALUATED = "NOT_EVALUATED"


class AuthorizationCheckStatus(str, Enum):
    NOT_REQUIRED = "NOT_REQUIRED"
    PRESENT = "PRESENT"
    MISSING = "MISSING"
    INVALID = "INVALID"
    NOT_EVALUATED = "NOT_EVALUATED"


class RuleEvidence(StrictModel):
    """Preuve minimale utilisée par une vérification déterministe."""

    source: str = Field(..., min_length=1)
    field: str = Field(..., min_length=1)
    value: str | int | Decimal | date | bool | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rule_id: str | None = None
    rule_version: str | None = None


class StructuredRuleError(StrictModel):
    """Erreur structurée stable pour l'Identity and Coverage Agent."""

    code: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    field: str | None = None
    severity: str = Field(default="ERROR", min_length=1)


class IdentityCheck(StrictModel):
    """Résultat dédié de comparaison d'identité."""

    status: IdentityCheckStatus = IdentityCheckStatus.NOT_EVALUATED
    rule_applied: str | None = None
    compared_fields: list[str] = Field(default_factory=list)
    patient_pseudonym: str | None = None
    contract_patient_pseudonym: str | None = None
    dossier_patient_pseudonym: str | None = None
    evidence: list[RuleEvidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    structured_errors: list[StructuredRuleError] = Field(default_factory=list)
    rule_version: str = "1.0.0"


class CoverageCheck(StrictModel):
    """Résultat dédié de vérification de couverture."""

    status: CoverageCheckStatus = CoverageCheckStatus.NOT_EVALUATED
    policy_number: str | None = None
    service_date: date | None = None
    coverage_start_date: date | None = None
    coverage_end_date: date | None = None
    requested_amount: Decimal | None = None
    total_amount: Decimal | None = None
    ceiling_amount: Decimal | None = None
    ceiling_remaining: Decimal | None = None
    evidence: list[RuleEvidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    structured_errors: list[StructuredRuleError] = Field(default_factory=list)
    rule_version: str = "1.0.0"


class AuthorizationCheck(StrictModel):
    """Résultat dédié de vérification de préautorisation."""

    status: AuthorizationCheckStatus = AuthorizationCheckStatus.NOT_EVALUATED
    preauthorization_reference: str | None = None
    required: bool | None = None
    procedure_codes: list[str] = Field(default_factory=list)
    evidence: list[RuleEvidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    structured_errors: list[StructuredRuleError] = Field(default_factory=list)
    rule_version: str = "1.0.0"


class IdentityCoverageInput(StrictModel):
    """Entrée du nœud Identity and Coverage Agent.

    Contient l'identifiant du dossier et, optionnellement, le chemin relatif
    du bundle FHIR sous la zone incoming/. Les données patient et couverture
    extraites proviennent de ocr_result dans le ClaimState.
    """

    case_id: str | None = Field(
        default=None,
        min_length=1,
        description="Identifiant technique du dossier (CLM-XXXX)",
    )
    claim_id: str | None = Field(
        default=None,
        description="Alias métier de case_id pour les entrées minimales",
    )
    patient_pseudonym: str | None = Field(default=None, description="Identité pseudonymisée")
    policy_number: str | None = Field(default=None, description="Numéro de contrat synthétique")
    service_date: date | None = None
    requested_amount: Decimal | None = Field(default=None, ge=Decimal("0"))
    total_amount: Decimal | None = Field(default=None, ge=Decimal("0"))
    procedure_codes: list[str] = Field(default_factory=list)
    preauthorization_reference: str | None = None
    extraction_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    provenance: dict[str, str] = Field(default_factory=dict)
    fhir_bundle_path: str | None = Field(
        default=None,
        description="Chemin relatif du bundle FHIR sous incoming/ (optionnel)",
    )
    dossier_patient_id: str | None = Field(
        default=None,
        description="Identité pseudonymisée issue du dossier",
    )
    contract: dict | None = Field(
        default=None,
        description="Snapshot de contrat synthétique en lecture seule",
    )
    rule_version: str = "1.0.0"
    evidence: list[RuleEvidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    structured_errors: list[StructuredRuleError] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sync_case_and_claim_id(self) -> "IdentityCoverageInput":
        if self.case_id is None and self.claim_id is None:
            raise ValueError("claim_id ou case_id requis")
        if self.case_id is None:
            self.case_id = self.claim_id
        if self.claim_id is None:
            self.claim_id = self.case_id
        return self

    @field_validator("case_id", "claim_id")
    @classmethod
    def _id_non_empty(cls, v: str | None) -> str | None:
        return v.strip() if v is not None else None

    @field_validator("fhir_bundle_path")
    @classmethod
    def _no_absolute_path(cls, v: str | None) -> str | None:
        """Refuse les chemins absolus POSIX, Windows et UNC."""
        if v is None:
            return v
        if _ABSOLUTE_PATH_RE.match(v):
            raise ValueError(
                f"Chemin absolu interdit dans fhir_bundle_path : {v!r} — "
                "utiliser un chemin relatif sous incoming/"
            )
        if ".." in v:
            raise ValueError(
                f"Traversée de répertoire interdite dans fhir_bundle_path : {v!r}"
            )
        return v


__all__ = [
    "AuthorizationCheck",
    "AuthorizationCheckStatus",
    "CoverageCheck",
    "CoverageCheckStatus",
    "IdentityCoverageInput",
    "IdentityCoverageResult",
    "IdentityCheck",
    "IdentityCheckStatus",
    "IdentityResult",
    "CoverageResult",
    "RuleEvidence",
    "StructuredRuleError",
]
