"""Schémas de sortie des 11 agents ClaimShield Santé.

Chaque résultat :
- hérite de StrictModel (extra='forbid')
- expose un champ `status` typé VerificationStatus ou SecurityDecision
- expose une liste `reasons` pour la traçabilité
- est JSON-sérialisable (pour les checkpoints LangGraph)
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated

from pydantic import Field

from schemas.domain import (
    DataClassification,
    ExtractedData,
    IntakeStatus,
    Recommendation,
    SecurityDecision,
    StrictModel,
    VerificationStatus,
)


# ── 1. Claim Intake Agent ─────────────────────────────────────────────────────


class UploadedFileInfo(StrictModel):
    """Fichier reçu avant toute validation (données annoncées par le déposant)."""

    original_name: str
    announced_size: int = Field(..., ge=0)
    announced_mime_type: str
    temp_id: str | None = None


class InspectedFile(StrictModel):
    """Résultat réel de l'inspection d'un fichier après lecture physique."""

    original_name: str
    storage_name: str
    normalized_extension: str
    detected_mime_type: str
    actual_size: int = Field(..., ge=0)
    sha256: Annotated[str, Field(min_length=64, max_length=64)] | None = None
    status: IntakeStatus
    block_reasons: list[str] = Field(default_factory=list)
    relative_storage_path: str | None = None


class StructuredError(StrictModel):
    """Erreur structurée rattachée à un fichier ou à une règle de validation."""

    code: str
    message: str
    field: str | None = None


class ClaimManifest(StrictModel):
    """Manifeste complet du dossier ingéré."""

    claim_id: str
    received_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    depositor_id: str | None = None
    file_count: int = Field(..., ge=0)
    total_size_bytes: int = Field(..., ge=0)
    files: list[InspectedFile] = Field(default_factory=list)
    status: IntakeStatus
    alerts: list[str] = Field(default_factory=list)
    schema_version: str = "1.0.0"


class ClaimIntakeResult(StrictModel):
    """Sortie finale de l'agent d'ingestion documentaire."""

    claim_id: str
    status: IntakeStatus
    manifest: ClaimManifest
    accepted_count: int = Field(..., ge=0)
    quarantined_count: int = Field(..., ge=0)
    reasons: list[str] = Field(default_factory=list)
    errors: list[StructuredError] = Field(default_factory=list)


# ── 2. Security Gate Agent ────────────────────────────────────────────────────


class SecurityGateResult(StrictModel):
    """ALLOW / BLOCK / QUARANTINE avant chaque étape sensible."""

    case_id: str
    decision: SecurityDecision
    prompt_injection_detected: bool | None = None
    blocked_fields: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


# ── 3. Privacy Agent ──────────────────────────────────────────────────────────


class PrivacyResult(StrictModel):
    """Vues minimisées selon le rôle du lecteur."""

    case_id: str
    status: VerificationStatus
    data_classification: DataClassification
    contains_real_personal_data: bool
    masked_fields: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


# ── 4. Identity & Coverage Agent ─────────────────────────────────────────────


class IdentityResult(StrictModel):
    status: VerificationStatus
    patient_id: str | None = None
    patient_name: str | None = None
    source_patient_id: str | None = None
    claim_patient_id: str | None = None
    encounter_patient_id: str | None = None
    reasons: list[str] = Field(default_factory=list)


class CoverageResult(StrictModel):
    status: VerificationStatus
    payer_name: str | None = None
    source_payer_name: str | None = None
    coverage_rate: Decimal | None = None
    amount_requested: Decimal | None = None
    patient_share: Decimal | None = None
    policy_active: bool | None = None
    reasons: list[str] = Field(default_factory=list)


class IdentityCoverageResult(StrictModel):
    """Vérification identité patient + couverture assurance."""

    case_id: str
    identity: IdentityResult
    coverage: CoverageResult
    rule_version: str = "1.0.0"


# ── 5. FHIR Validator Agent ───────────────────────────────────────────────────


class FhirValidatorResult(StrictModel):
    """Validation structure du bundle FHIR R4."""

    case_id: str
    status: VerificationStatus
    bundle_expected: bool
    profile_checked: str | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


# ── 6. Document & OCR Agent ───────────────────────────────────────────────────


class DocumentOcrResult(StrictModel):
    """Classification des pièces et extraction des champs avec provenance."""

    case_id: str
    status: VerificationStatus
    extracted: ExtractedData | None = None
    unreadable_documents: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


# ── 7. Medical Coding Agent ───────────────────────────────────────────────────


class ProcedureCoding(StrictModel):
    original_description: str
    proposed_code: str | None = None
    rule_applied: str | None = None
    status: VerificationStatus


class MedicalCodingResult(StrictModel):
    """Correspondance actes → table de codes locale versionnée."""

    case_id: str
    status: VerificationStatus
    codings: list[ProcedureCoding] = Field(default_factory=list)
    table_version: str = "1.0.0"
    reasons: list[str] = Field(default_factory=list)


# ── 8. Clinical Consistency Agent ─────────────────────────────────────────────


class ClinicalSignal(StrictModel):
    signal_type: str
    description: str
    fields_compared: list[str] = Field(default_factory=list)
    severity: str = "WARNING"


class ClinicalConsistencyResult(StrictModel):
    """Cohérence clinique entre ordonnance, consultation et chronologie."""

    case_id: str
    status: VerificationStatus
    procedure_count: int | None = None
    medication_count: int | None = None
    prescription_required: bool | None = None
    signals: list[ClinicalSignal] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


# ── 9. Fraud Detection Agent ──────────────────────────────────────────────────


class FraudSignal(StrictModel):
    signal_type: str
    description: str
    risk_contribution: float = Field(ge=0.0, le=1.0)


class FraudDetectionResult(StrictModel):
    """Détection de doublons et anomalies sur données synthétiques."""

    case_id: str
    status: VerificationStatus
    duplicate_invoice: bool | None = None
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    signals: list[FraudSignal] = Field(default_factory=list)
    threshold_version: str = "1.0.0"
    reasons: list[str] = Field(default_factory=list)


# ── 10. Case Reviewer Agent ───────────────────────────────────────────────────


class DisagreementPoint(StrictModel):
    agent: str
    field: str
    expected: str
    observed: str


class CaseReviewerResult(StrictModel):
    """Synthèse et recommandation révisable par un humain."""

    case_id: str
    recommendation: Recommendation
    justification: list[str] = Field(default_factory=list)
    disagreements: list[DisagreementPoint] = Field(default_factory=list)
    human_review_required: bool = True
    human_review_reasons: list[str] = Field(default_factory=list)


# ── 11. Audit Agent ───────────────────────────────────────────────────────────


class AuditEvent(StrictModel):
    """Événement append-only corrélé au claim_id."""

    event_id: str
    case_id: str
    actor: str = Field(..., description="Agent ou identifiant utilisateur")
    action: str
    outcome: str
    agent_version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    details: dict[str, str] = Field(default_factory=dict)
