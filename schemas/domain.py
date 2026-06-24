"""Modèles Pydantic du domaine ClaimShield Santé.

Chaque modèle interdit les champs inconnus (extra='forbid') pour détecter
immédiatement toute divergence entre agents.
"""

from __future__ import annotations

from datetime import date, datetime  # noqa: TCH003
from decimal import Decimal
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Enums ────────────────────────────────────────────────────────────────────


class Recommendation(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    PENDING = "PENDING"


class VerificationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    PENDING = "PENDING"
    NOT_EVALUATED = "NOT_EVALUATED"


class SecurityDecision(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    QUARANTINE = "QUARANTINE"


class DataClassification(str, Enum):
    SYNTHETIC_TEST_DATA = "SYNTHETIC_TEST_DATA"
    ANONYMIZED = "ANONYMIZED"
    CONFIDENTIAL = "CONFIDENTIAL"


class AuthorizationStatus(str, Enum):
    APPROVED = "approved"
    PENDING = "pending"
    NOT_REQUIRED = "not_required"
    REJECTED = "rejected"


class IntakeStatus(str, Enum):
    """Statut global du dossier d'ingestion."""

    ACCEPTED = "accepted"
    QUARANTINED = "quarantined"
    BLOCKED = "blocked"
    ERROR = "error"


class FileStatus(str, Enum):
    """Statut d'un fichier individuel après inspection.

    DUPLICATE et ERROR n'existent qu'au niveau fichier ;
    ils remontent respectivement en QUARANTINED et ERROR au niveau dossier.
    """

    ACCEPTED = "accepted"
    QUARANTINED = "quarantined"
    BLOCKED = "blocked"
    DUPLICATE = "duplicate"
    ERROR = "error"


class IntakeReasonCode(str, Enum):
    """Codes stables identifiant la cause d'un rejet ou d'une alerte d'ingestion.

    Chaque valeur correspond à une entrée dans REASON_DESCRIPTIONS.
    Les codes ne changent pas entre versions — seul le message peut évoluer.
    """

    EMPTY_CLAIM = "EMPTY_CLAIM"
    EMPTY_FILE = "EMPTY_FILE"
    UNSUPPORTED_EXTENSION = "UNSUPPORTED_EXTENSION"
    UNSUPPORTED_MIME_TYPE = "UNSUPPORTED_MIME_TYPE"
    MIME_EXTENSION_MISMATCH = "MIME_EXTENSION_MISMATCH"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    CLAIM_TOO_LARGE = "CLAIM_TOO_LARGE"
    PATH_TRAVERSAL_ATTEMPT = "PATH_TRAVERSAL_ATTEMPT"
    DUPLICATE_FILE = "DUPLICATE_FILE"
    STORAGE_ERROR = "STORAGE_ERROR"
    TOO_MANY_FILES = "TOO_MANY_FILES"
    FOLDER_QUOTA_EXCEEDED = "FOLDER_QUOTA_EXCEEDED"
    INVALID_FILENAME = "INVALID_FILENAME"


REASON_DESCRIPTIONS: dict[str, str] = {
    IntakeReasonCode.EMPTY_CLAIM: (
        "Dossier vide — aucun fichier soumis"
    ),
    IntakeReasonCode.EMPTY_FILE: (
        "Fichier vide (0 octet) — le contenu est absent"
    ),
    IntakeReasonCode.UNSUPPORTED_EXTENSION: (
        "Extension non autorisée — seuls PDF, PNG, JPEG et JSON sont acceptés"
    ),
    IntakeReasonCode.UNSUPPORTED_MIME_TYPE: (
        "Type MIME non autorisé — le contenu réel du fichier n'est pas reconnu"
    ),
    IntakeReasonCode.MIME_EXTENSION_MISMATCH: (
        "Incohérence MIME/extension — le contenu détecté ne correspond pas à l'extension déclarée"
    ),
    IntakeReasonCode.FILE_TOO_LARGE: (
        "Fichier trop volumineux — dépasse la limite de taille individuelle configurée"
    ),
    IntakeReasonCode.CLAIM_TOO_LARGE: (
        "Dossier trop volumineux — le quota cumulé est dépassé"
    ),
    IntakeReasonCode.PATH_TRAVERSAL_ATTEMPT: (
        "Tentative de traversée de répertoire — nom de fichier dangereux refusé"
    ),
    IntakeReasonCode.DUPLICATE_FILE: (
        "Fichier en double — SHA-256 identique à un fichier déjà reçu dans ce dossier"
    ),
    IntakeReasonCode.STORAGE_ERROR: (
        "Échec technique de stockage — écriture ou déplacement impossible"
    ),
    IntakeReasonCode.TOO_MANY_FILES: (
        "Trop de fichiers — le nombre maximum de fichiers par dossier est atteint"
    ),
    IntakeReasonCode.FOLDER_QUOTA_EXCEEDED: (
        "Quota dépassé — la taille cumulée du dossier dépasse la limite configurée"
    ),
    IntakeReasonCode.INVALID_FILENAME: (
        "Nom de fichier invalide — caractères ou structure non autorisés"
    ),
}


# ── Montants ─────────────────────────────────────────────────────────────────

PositiveDecimal = Annotated[Decimal, Field(gt=Decimal("0"))]
NonNegativeDecimal = Annotated[Decimal, Field(ge=Decimal("0"))]


# ── Modèles de base ──────────────────────────────────────────────────────────


class StrictModel(BaseModel):
    """Classe de base : champs inconnus interdits, assignation validée."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ── Patient ───────────────────────────────────────────────────────────────────


class PatientInfo(StrictModel):
    patient_id: str = Field(..., description="UUID Synthea du patient")
    patient_name: str = Field(..., min_length=1)
    birth_date: date | None = None
    gender: str | None = None


# ── Couverture assurance ──────────────────────────────────────────────────────


class CoverageInfo(StrictModel):
    payer_name: str = Field(..., min_length=1)
    coverage_rate: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    policy_active: bool = True
    policy_start: date | None = None
    policy_end: date | None = None

    @field_validator("coverage_rate", mode="before")
    @classmethod
    def parse_coverage_rate(cls, v: object) -> Decimal:
        return Decimal(str(v))


# ── Documents ─────────────────────────────────────────────────────────────────


class DocumentInfo(StrictModel):
    filename: str
    sha256: str = Field(..., min_length=64, max_length=64)
    size_bytes: int = Field(..., gt=0)
    mime_type: str


# ── Données extraites par l'agent OCR ────────────────────────────────────────


class ExtractedData(StrictModel):
    patient_name: str | None = None
    patient_id: str | None = None
    payer_name: str | None = None
    service_date: date | None = None
    claim_reference: str | None = None
    invoice_number: str | None = None
    prescription_number: str | None = None
    procedure_count: int | None = Field(default=None, ge=0)
    medication_count: int | None = Field(default=None, ge=0)
    total_billed: Decimal | None = Field(default=None, ge=Decimal("0"))
    amount_requested: Decimal | None = Field(default=None, ge=Decimal("0"))
    patient_share: Decimal | None = Field(default=None, ge=Decimal("0"))
    currency: str = "USD"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: dict[str, str] = Field(
        default_factory=dict,
        description="Champ → nom de fichier source + numéro de page",
    )

    @field_validator("total_billed", "amount_requested", "patient_share", mode="before")
    @classmethod
    def parse_decimal(cls, v: object) -> Decimal | None:
        if v is None:
            return None
        return Decimal(str(v))


# ── Prestataire de soins ──────────────────────────────────────────────────────


class ProviderInfo(StrictModel):
    provider_id: str
    organization_id: str | None = None
    name: str | None = None
    specialty: str | None = None


# ── Consultation médicale ─────────────────────────────────────────────────────


class EncounterInfo(StrictModel):
    encounter_id: str
    encounter_class: str = Field(..., description="ambulatory, inpatient, emergency…")
    start: datetime
    stop: datetime | None = None
    patient_id: str
    provider: ProviderInfo | None = None
    diagnosis_codes: list[str] = Field(default_factory=list)

    @field_validator("start", "stop", mode="before")
    @classmethod
    def parse_dt(cls, v: object) -> datetime | None:
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))


# ── Acte médical ──────────────────────────────────────────────────────────────


class MedicalProcedure(StrictModel):
    code: str = Field(..., description="Code SNOMED CT ou CIM-10")
    description: str
    unit_cost: NonNegativeDecimal
    quantity: int = Field(default=1, ge=1)
    performed_date: date | None = None

    @field_validator("unit_cost", mode="before")
    @classmethod
    def parse_cost(cls, v: object) -> Decimal:
        return Decimal(str(v))


# ── Prescription ──────────────────────────────────────────────────────────────


class Prescription(StrictModel):
    medication_code: str
    medication_name: str
    dispenses: int = Field(default=1, ge=1)
    unit_cost: NonNegativeDecimal

    @field_validator("unit_cost", mode="before")
    @classmethod
    def parse_cost(cls, v: object) -> Decimal:
        return Decimal(str(v))


# ── Règles déterministes ──────────────────────────────────────────────────────


class DeterministicRules(StrictModel):
    coverage_rate: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    authorization_required: bool
    authorization_status: AuthorizationStatus
    duplicate_invoice: bool
    prompt_injection_detected: bool

    @field_validator("coverage_rate", mode="before")
    @classmethod
    def parse_rate(cls, v: object) -> Decimal:
        return Decimal(str(v))


# ── Dossier de remboursement ──────────────────────────────────────────────────


class ClaimSubmission(StrictModel):
    case_id: str = Field(..., pattern=r"^CLM-\d{4,}$")
    schema_version: str = "1.0.0"
    data_classification: DataClassification = DataClassification.SYNTHETIC_TEST_DATA
    contains_real_personal_data: bool = False
    submitted_at: datetime | None = None
    patient: PatientInfo | None = None
    coverage: CoverageInfo | None = None
    encounter: EncounterInfo | None = None
    documents: list[DocumentInfo] = Field(default_factory=list)
    procedures: list[MedicalProcedure] = Field(default_factory=list)
    prescriptions: list[Prescription] = Field(default_factory=list)
    extracted: ExtractedData | None = None
    rules: DeterministicRules | None = None
