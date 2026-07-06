"""Schémas de sortie des 11 agents ClaimShield Santé.

Chaque résultat :
- hérite de StrictModel (extra='forbid')
- expose un champ `status` typé VerificationStatus ou SecurityDecision
- expose une liste `reasons` pour la traçabilité
- est JSON-sérialisable (pour les checkpoints LangGraph)
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated

from pydantic import Field, computed_field, field_validator, model_validator

from schemas.domain import (
    DataClassification,
    DocumentType,
    ExtractionStatus,
    FileStatus,
    FindingCode,
    InputType,
    IntakeStatus,
    OCR_ERROR_CODE_DESCRIPTIONS,
    OCR_ERROR_CODE_RETRYABLE,
    OCR_ERROR_CODE_SEVERITIES,
    OcrCode,
    OcrSource,
    PrivacyCode,
    PrivacyDecision,
    Recommendation,
    SecurityDecision,
    SeverityLevel,
    StrictModel,
    VerificationStatus,
)

_ABSOLUTE_PATH_RE = re.compile(r"^(?:/|[A-Za-z]:[/\\]|\\\\)")
_SECRET_HINT_RE = re.compile(
    r"(?:api[_-]?key|secret\s*[:=]|password\s*[:=]|token\s*[:=]|bearer\s+[a-z0-9._~+/=-]+)",
    re.IGNORECASE,
)


def _reject_security_leak(value: str | None, field_name: str) -> str | None:
    """Refuse les valeurs qui ne doivent jamais être persistées dans le gate."""
    if value is None:
        return value
    if _ABSOLUTE_PATH_RE.match(value):
        raise ValueError(f"Chemin absolu interdit dans {field_name}")
    if _SECRET_HINT_RE.search(value):
        raise ValueError(f"Secret potentiel interdit dans {field_name}")
    return value


_MAX_NEWLINES_IN_STRUCTURED_FIELD = 2


def _reject_unstructured_content(value: str, field_name: str) -> str:
    """Interdit, en plus des chemins absolus et secrets, tout contenu
    multi-lignes assimilable à un document brut, un texte OCR complet ou un
    prompt complet dans un champ de preuve ou de motif structuré."""
    _reject_security_leak(value, field_name)
    if value.count("\n") > _MAX_NEWLINES_IN_STRUCTURED_FIELD:
        raise ValueError(
            f"Contenu multi-lignes interdit dans {field_name} — jamais un "
            "document brut, un texte OCR complet ou un prompt complet."
        )
    return value


class LlmMetadata(StrictModel):
    """Métadonnées LLM minimales autorisées dans un résultat d'agent.

    Ne contient jamais le prompt complet, les messages, la réponse brute du
    modèle, une clé API ou un secret.
    """

    model_name: str = Field(..., min_length=1, max_length=120)
    prompt_version: str = Field(..., min_length=1, max_length=50)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("model_name", "prompt_version")
    @classmethod
    def no_sensitive_value(cls, v: str | None, info) -> str | None:
        return _reject_security_leak(v, info.field_name)


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
    status: FileStatus
    reasons: list[StructuredError] = Field(default_factory=list)
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
    duplicate_count: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)
    reasons: list[str] = Field(default_factory=list)
    errors: list[StructuredError] = Field(default_factory=list)
    llm_metadata: LlmMetadata


# ── 2. Security Gate Agent ────────────────────────────────────────────────────


class SecurityFinding(StrictModel):
    """Anomalie de sécurité détectée lors de l'évaluation.

    Le champ `evidence` est volontairement limité et ne contient jamais
    de document brut, de secret ou de donnée personnelle complète.
    """

    code: FindingCode = Field(..., description="Code stable de l'anomalie")
    severity: SeverityLevel = Field(..., description="Niveau de sévérité")
    description: str = Field(..., min_length=1, description="Description compréhensible")
    detection_source: str = Field(
        ...,
        description="Source de la détection (ex. 'regex_scanner')",
    )
    affected_element: str = Field(..., description="Champ ou élément concerné")
    evidence: str | None = Field(
        default=None,
        max_length=200,
        description="Preuve minimisée — tronquée, sans secret ni document brut",
    )

    @field_validator(
        "description",
        "detection_source",
        "affected_element",
        "evidence",
    )
    @classmethod
    def no_sensitive_value(cls, v: str | None, info) -> str | None:
        return _reject_security_leak(v, info.field_name)


class SecurityAuditEntry(StrictModel):
    """Événement d'audit minimal embarqué dans SecurityGateResult.

    Complément léger de l'AuditEvent global — ne contient pas de secret.
    """

    claim_id: str = Field(default="", description="Identifiant du dossier contrôlé")
    actor: str = Field(default="security_gate_agent")
    action: str = Field(default="security_evaluation")
    input_type: InputType | None = Field(
        default=None,
        description="Type d'entrée contrôlée",
    )
    outcome: str = Field(
        ...,
        description="Valeur de SecurityDecision ayant conclu l'évaluation",
    )
    decision: SecurityDecision | None = Field(
        default=None,
        description="Décision structurée du Security Gate",
    )
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    policy_applied: str = Field(
        default="default",
        description="Nom de la SecurityPolicy utilisée",
    )
    policy_version: str = Field(
        default="1.0.0",
        description="Version de la SecurityPolicy utilisée",
    )
    reason_codes: list[FindingCode] = Field(
        default_factory=list,
        description="Codes stables des motifs de décision",
    )
    file_sha256: Annotated[str, Field(min_length=64, max_length=64)] | None = Field(
        default=None,
        description="SHA-256 du fichier concerné, si disponible",
    )

    @field_validator(
        "claim_id",
        "actor",
        "action",
        "outcome",
        "policy_applied",
        "policy_version",
        "file_sha256",
    )
    @classmethod
    def no_sensitive_value(cls, v: str | None, info) -> str | None:
        return _reject_security_leak(v, info.field_name)


class SecurityGateResult(StrictModel):
    """Décision de sécurité ALLOW / BLOCK / QUARANTINE.

    Invariants :
      - `reasons` contient au moins un motif (min_length=1).
      - `policy_version` identifie la politique ayant produit la décision.
      - `findings` est la liste structurée des anomalies détectées.
      - `evaluated_at` et `audit_entry` sont renseignés par l'agent.
      - `confidence_score` est borné entre 0.0 et 1.0.
      - Le schéma est JSON-sérialisable (StrictModel Pydantic).
      - Aucun document brut, secret ou chemin absolu.
    """

    claim_id: str
    decision: SecurityDecision
    findings: list[SecurityFinding] = Field(
        default_factory=list,
        description="Liste structurée des anomalies détectées",
    )
    reason_codes: list[FindingCode] = Field(
        default_factory=list,
        description="Codes stables des motifs de la décision",
    )
    applied_policy: str = Field(
        default="default",
        description="Nom de la SecurityPolicy appliquée",
    )
    policy_version: str = Field(default="1.0.0", description="Version de la SecurityPolicy")
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Niveau de confiance de la décision finale, borné entre 0.0 et 1.0",
    )
    evidence_summary: str | None = Field(
        default=None,
        max_length=300,
        description="Preuve minimisée résumant le signal principal, sans contenu brut",
    )
    next_allowed_action: str = Field(
        default="",
        description="Action suivante autorisée après cette décision",
    )
    audit_entry: SecurityAuditEntry | None = Field(
        default=None,
        description="Événement d'audit minimal — renseigné par l'agent",
    )
    prompt_injection_detected: bool | None = None
    blocked_fields: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(
        min_length=1,
        description="Au moins un motif humainement lisible obligatoire",
    )
    llm_metadata: LlmMetadata | None = None

    @field_validator(
        "claim_id",
        "applied_policy",
        "policy_version",
        "evidence_summary",
        "next_allowed_action",
    )
    @classmethod
    def no_sensitive_scalar(cls, v: str | None, info) -> str | None:
        return _reject_security_leak(v, info.field_name)

    @field_validator("blocked_fields", "reasons")
    @classmethod
    def no_sensitive_list(cls, v: list[str], info) -> list[str]:
        for item in v:
            _reject_security_leak(item, info.field_name)
        return v


# ── 3. Privacy Agent ──────────────────────────────────────────────────────────


class PrivacyAuditEntry(StrictModel):
    """Trace d'audit minimisée embarquée dans PrivacyResult.

    Contient uniquement des métadonnées de traitement — jamais de données
    personnelles, de secret, de chemin absolu ni de texte OCR brut.
    Conforme aux mêmes règles que SecurityAuditEntry.
    """

    claim_id: str = Field(default="", description="Identifiant du dossier traité")
    actor: str = Field(default="privacy_agent", description="Agent ayant produit la vue")
    action: str = Field(default="view_request", description="Action effectuée")
    role: str = Field(
        ...,
        description="Rôle du demandeur (valeur de ReaderRole) ou 'UNKNOWN' si absent",
    )
    outcome: str = Field(
        ...,
        description="Valeur de VerificationStatus ayant conclu le traitement",
    )
    decision: VerificationStatus | None = Field(
        default=None,
        description="Décision structurée du Privacy Agent",
    )
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    policy_version: str = Field(
        default="1.0.0",
        description="Version de la politique d'accès appliquée",
    )
    redacted_count: int = Field(
        default=0,
        ge=0,
        description="Nombre de champs refusés par DENY-by-default",
    )
    view_built: bool = Field(
        default=False,
        description="Vue minimisée construite avec succès (claim_data fourni et valide)",
    )
    pseudonymization_applied: bool = Field(
        default=False,
        description="Pseudonymisation des identifiants personnels appliquée",
    )
    reason_codes: list[PrivacyCode] = Field(
        default_factory=list,
        description="Codes stables identifiant la cause de la décision ou d'une erreur",
    )

    @field_validator("claim_id", "actor", "action", "role", "outcome", "policy_version")
    @classmethod
    def no_sensitive_value(cls, v: str | None, info) -> str | None:
        return _reject_security_leak(v, info.field_name)


class PrivacyResult(StrictModel):
    """Résultat du Privacy Agent — vues minimisées et décision PASS/NEEDS_REVIEW/FAIL.

    Le champ `decision` est calculé automatiquement depuis `status` :
      PASS / NEEDS_REVIEW → PrivacyDecision.ALLOW
      FAIL               → PrivacyDecision.BLOCK
    """

    case_id: str
    status: VerificationStatus
    data_classification: DataClassification
    contains_real_personal_data: bool
    redacted_fields: list[str] = Field(
        default_factory=list,
        description="Champs refusés par la politique DENY-by-default pour ce rôle",
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Motifs lisibles — informations et alertes (PASS, NEEDS_REVIEW, FAIL)",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Motifs de blocage — non vide uniquement quand status == FAIL",
    )
    policy_version: str = Field(
        default="1.0.0",
        description="Version de la politique d'accès appliquée",
    )
    reason_codes: list[PrivacyCode] = Field(
        default_factory=list,
        description="Codes stables identifiant la cause de la décision ou d'une erreur",
    )
    view: dict | None = Field(
        default=None,
        description="Vue minimisée construite selon le rôle (JSON-sérialisable)",
    )
    view_role: str | None = Field(
        default=None,
        description="Rôle ayant produit la vue (valeur de PrivacyRole)",
    )
    audit_entry: PrivacyAuditEntry | None = Field(
        default=None,
        description="Trace d'audit minimisée du traitement privacy",
    )
    llm_metadata: LlmMetadata | None = None

    @computed_field
    @property
    def decision(self) -> PrivacyDecision:
        """Décision binaire dérivée du statut : ALLOW si PASS/NEEDS_REVIEW, BLOCK si FAIL."""
        return (
            PrivacyDecision.BLOCK
            if self.status == VerificationStatus.FAIL
            else PrivacyDecision.ALLOW
        )


# ── 4. Identity & Coverage Agent ─────────────────────────────────────────────


class IdentityResult(StrictModel):
    status: VerificationStatus
    patient_id: str | None = None
    patient_name: str | None = None
    source_patient_id: str | None = None
    claim_patient_id: str | None = None
    contract_patient_id: str | None = None
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
    service_date: date | None = None
    coverage_start_date: date | None = None
    coverage_end_date: date | None = None
    ceiling_amount: Decimal | None = None
    ceiling_remaining: Decimal | None = None
    ceiling_exceeded: bool | None = None
    preauthorization_required: bool | None = None
    preauthorization_status: str | None = None
    reasons: list[str] = Field(default_factory=list)


class IdentityCoverageResult(StrictModel):
    """Vérification identité patient + couverture assurance."""

    case_id: str
    identity: IdentityResult
    coverage: CoverageResult
    rule_version: str = "1.0.0"
    evidence: list[dict[str, str]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    structured_errors: list[dict[str, str]] = Field(default_factory=list)
    llm_metadata: LlmMetadata | None = None


# ── 5. FHIR Validator Agent ───────────────────────────────────────────────────


class FhirValidatorResult(StrictModel):
    """Validation structure du bundle FHIR R4."""

    case_id: str
    status: VerificationStatus
    bundle_expected: bool
    profile_checked: str | None = None
    rule_version: str = "1.0.0"
    resource_types: list[str] = Field(default_factory=list)
    resource_count: int = 0
    references_checked: bool = False
    validation_scope: str = "STRUCTURAL_ONLY"
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    llm_metadata: LlmMetadata


# ── Champs essentiels — Document/OCR Agent (Étape 7) ─────────────────────────


class MonetaryAmount(StrictModel):
    """Montant monétaire avec devise — jamais de float."""

    amount: Decimal = Field(..., ge=Decimal("0"), description="Montant en Decimal")
    currency: str = Field(default="USD", min_length=1, max_length=3)

    @field_validator("amount", mode="before")
    @classmethod
    def _parse_amount(cls, v: object) -> Decimal:
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))


class MedicalItem(StrictModel):
    """Acte médical ou médicament extrait d'un document."""

    description: str = Field(..., min_length=1)
    code: str | None = None
    quantity: int = Field(default=1, ge=1)
    unit_amount: MonetaryAmount | None = None


ESSENTIAL_FIELD_NAMES: frozenset[str] = frozenset({
    "patient_identifier",
    "document_reference",
    "document_date",
    "service_date",
    "provider_identifier_or_name",
    "total_amount",
    "requested_amount",
    "medical_items",
})


class EssentialFields(StrictModel):
    """Les huit champs essentiels à extraire de tout document médical.

    Règles :
      - None si le champ n'est pas détecté (ne jamais inventer une valeur absente).
      - Montants en Decimal via MonetaryAmount — jamais de float.
      - Dates en date Python — jamais de string brute.
      - Valeur brute conservée dans DocumentExtraction.fields[...].value (ExtractedField).
      - Ne pas corriger silencieusement une valeur ambiguë.
    """

    patient_identifier: str | None = None
    document_reference: str | None = None
    document_date: date | None = None
    service_date: date | None = None
    provider_identifier_or_name: str | None = None
    total_amount: MonetaryAmount | None = None
    requested_amount: MonetaryAmount | None = None
    medical_items: list[MedicalItem] = Field(default_factory=list)


# ── 6. Document & OCR Agent ───────────────────────────────────────────────────


class DocumentPageContent(StrictModel):
    """Contenu extrait d'une page — jamais exécuté, jamais interprété comme instruction."""

    page_number: int = Field(..., ge=1)
    text: str
    char_count: int = Field(..., ge=0)
    ocr_source: OcrSource
    confidence: float = Field(..., ge=0.0, le=1.0)


class PageText(StrictModel):
    """Page OCR enrichie — utilisée dans DocumentExtraction."""

    page_number: int = Field(..., ge=1)
    text: str
    char_count: int = Field(..., ge=0)
    method: OcrSource
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_text_based: bool = True


class FieldProvenance(StrictModel):
    """Traçabilité complète d'un champ extrait — assure l'auditabilité document par document.

    Jamais de donnée personnelle dans les champs d'infrastructure (filename, sha256, method).
    source_text est tronqué à 200 caractères.
    """

    filename: str = Field(..., min_length=1, max_length=255)
    sha256: str = Field(default="", description="SHA-256 du fichier source (64 hex ou vide)")
    page_number: int | None = Field(None, ge=1)
    method: OcrSource
    source_text: str = Field(default="", description="Extrait utile du texte source (≤ 200 car.)")
    position: dict[str, int] | None = Field(
        default=None,
        description="Position dans le texte source si disponible : {'start': int, 'end': int}",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    parser_version: str = Field(..., min_length=1)
    extracted_at: datetime

    @field_validator("filename")
    @classmethod
    def _no_absolute_filename(cls, v: str) -> str:
        if re.match(r"^(?:/|[A-Za-z]:[/\\]|\\\\)", v):
            raise ValueError(f"Chemin absolu interdit dans filename : {v!r}")
        return v

    @field_validator("sha256")
    @classmethod
    def _valid_sha256_or_empty(cls, v: str) -> str:
        if v and not re.fullmatch(r"[0-9a-f]{64}", v.lower()):
            raise ValueError(f"SHA-256 invalide (64 hex ou vide) : {v!r}")
        return v.lower()

    @field_validator("source_text")
    @classmethod
    def _truncate_source_text(cls, v: str) -> str:
        return v[:200]

    @field_validator("position")
    @classmethod
    def _valid_position(cls, v: dict[str, int] | None) -> dict[str, int] | None:
        if v is None:
            return v
        if set(v) != {"start", "end"}:
            raise ValueError("position doit contenir exactement start et end")
        if v["start"] < 0 or v["end"] < v["start"]:
            raise ValueError("position invalide")
        return v


class ExtractedField(StrictModel):
    """Champ extrait avec valeur brute, valeur normalisée, provenance et indicateurs de confiance."""

    field_name: str = Field(..., min_length=1)
    value: str
    normalized_value: str = ""
    confidence: float = Field(..., ge=0.0, le=1.0)
    provenance: FieldProvenance | None = None
    warnings: list[str] = Field(default_factory=list)
    requires_review: bool = False


class DocumentClassification(StrictModel):
    """Résultat Pydantic de la classification du type de document médical."""

    document_type: DocumentType
    confidence: float = Field(..., ge=0.0, le=1.0)
    classification_source: str = Field(
        ..., description="Signal utilisé : filename | mime | keywords | combined | unknown"
    )
    is_ambiguous: bool = False
    scores: dict[str, float] = Field(default_factory=dict)
    rules_version: str = "document-classifier-rules-v1"


class DocumentExtraction(StrictModel):
    """Vue riche et structurée de l'extraction pour un document unique.

    Coexiste avec DocumentOcrResult (vue plate LangGraph).
    Contient la provenance complète de chaque champ extrait.

    Garanties :
      - `errors` : anomalies bloquantes ayant empêché ou altéré l'extraction.
      - `warnings` : anomalies non bloquantes — fallbacks, champs optionnels absents,
                     confiance légèrement sous-optimale. N'empêchent pas l'extraction.
      - `tool_versions` : versions des outils utilisés pour ce document.
      - Tous les champs sont JSON-sérialisables via model_dump(mode="json").
    """

    claim_id: str = Field(..., min_length=1, max_length=50)
    document_id: str = Field(..., min_length=1, max_length=100)
    classification: DocumentClassification
    pages: list[PageText] = Field(default_factory=list)
    full_text: str = ""
    fields: dict[str, ExtractedField] = Field(default_factory=dict)
    extraction_status: ExtractionStatus
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_readable: bool = False
    human_review_required: bool = False
    human_review_reasons: list[str] = Field(default_factory=list)
    errors: list[str] = Field(
        default_factory=list,
        description="Anomalies bloquantes ayant empêché ou altéré l'extraction",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Anomalies non bloquantes — fallbacks, champs optionnels absents, "
                    "confiance légèrement sous-optimale",
    )
    tool_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Versions des outils utilisés : pdf_reader, ocr_engine, classifier, "
                    "confidence, parser, ocr_thresholds",
    )
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    essential_fields: EssentialFields | None = Field(
        default=None,
        description="Les huit champs essentiels extraits et typés (Étape 7)",
    )
    artifact_id: str | None = None
    artifact_path: str | None = None
    security_findings: list[SecurityFinding] = Field(default_factory=list)
    structured_errors: list[OcrError] = Field(
        default_factory=list,
        description="Erreurs structurées Étape 18 — code stable, sévérité, document, retryable",
    )


class OcrError(StrictModel):
    """Erreur structurée du Document/OCR Agent — Étape 18.

    Invariants :
      - `code` est un code stable de OcrCode — ne change jamais entre versions.
      - `message` est contrôlé — jamais de donnée personnelle ni de secret.
      - `severity` et `retryable` sont dérivés des tables centralisées si non fournis.
      - `document` désigne le fichier concerné par son nom de stockage (sans chemin absolu).
      - `page_number` est None quand l'erreur est globale au document.
    """

    code: OcrCode = Field(..., description="Code stable identifiant la cause de l'erreur")
    message: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Message contrôlé — aucune donnée personnelle ni secret",
    )
    severity: SeverityLevel = Field(..., description="Niveau de sévérité de l'erreur")
    document: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Nom de stockage du document concerné (sans chemin absolu)",
    )
    page_number: int | None = Field(
        default=None,
        ge=1,
        description="Numéro de page concernée — None si l'erreur est globale au document",
    )
    retryable: bool = Field(
        ...,
        description="Vrai si l'opération peut être retentée sans modification du document",
    )

    @field_validator("message", "document")
    @classmethod
    def _no_sensitive_value(cls, v: str, info) -> str:
        return _reject_security_leak(v, info.field_name)

    @field_validator("document")
    @classmethod
    def _no_absolute_path(cls, v: str) -> str:
        if re.match(r"^(?:/|[A-Za-z]:[/\\]|\\\\)", v):
            raise ValueError(f"Chemin absolu interdit dans document : {v!r}")
        if ".." in v:
            raise ValueError(f"Traversée de répertoire interdite dans document : {v!r}")
        return v

    @classmethod
    def from_code(
        cls,
        code: OcrCode,
        document: str,
        *,
        page_number: int | None = None,
        message: str | None = None,
    ) -> "OcrError":
        """Construit une OcrError à partir d'un code stable.

        Le message, la sévérité et le flag retryable sont déduits des tables
        centralisées si non fournis — garantissant la cohérence globale.
        """
        resolved_message = message or OCR_ERROR_CODE_DESCRIPTIONS.get(
            code, f"Erreur OCR : {code.value}"
        )
        return cls(
            code=code,
            message=resolved_message,
            severity=OCR_ERROR_CODE_SEVERITIES.get(code, SeverityLevel.HIGH),
            document=document,
            page_number=page_number,
            retryable=OCR_ERROR_CODE_RETRYABLE.get(code, False),
        )


class DocumentOcrAuditEntry(StrictModel):
    """Trace d'audit minimisée — jamais de donnée personnelle ni de secret."""

    claim_id: str
    file_path: str
    sha256_verified: bool
    document_type: DocumentType
    ocr_source: OcrSource
    page_count: int = Field(..., ge=0)
    total_chars: int = Field(..., ge=0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    is_readable: bool
    human_review_required: bool
    reason_codes: list[OcrCode] = Field(default_factory=list)
    evaluated_at: datetime
    extraction_status: ExtractionStatus
    status: VerificationStatus


class DocumentOcrResult(StrictModel):
    """Résultat de classification et d'extraction avec provenance complète.

    Schéma Étape 19 — champs complets et garanties documentées :

    Statut & classification :
      - `extraction_status`  : ExtractionStatus fin-grain (SUCCESS/NEEDS_REVIEW/FAILED/SKIPPED/BLOCKED)
      - `status`             : VerificationStatus LangGraph (PASS/NEEDS_REVIEW/FAIL)
      - `classification`     : DocumentClassification complète (type, confiance, source, scores, version)
      - `document_type`      : DocumentType déduit de la classification (raccourci)

    Données extraites :
      - `extracted_fields`   : dict[str, ExtractedField] — champs avec provenance fine
      - `extraction`         : DocumentExtraction — vue riche complète (pages + champs + essential_fields)

    Erreurs & avertissements :
      - `errors`             : anomalies bloquantes (liste de messages contrôlés)
      - `warnings`           : anomalies non bloquantes — fallbacks, champs optionnels absents
      - `structured_errors`  : list[OcrError] — erreurs avec code stable, sévérité, document, retryable
      - `reason_codes`       : list[OcrCode] — codes stables pour la machine
      - `reasons`            : list[str] — messages lisibles pour l'humain

    Scores :
      - `confidence_score`   : score global en [0.0, 1.0]

    Versions des outils :
      - `tool_versions`      : dict[str, str] — pdf_reader, ocr_engine, classifier, confidence, parser

    Artefacts :
      - `artifact_id`        : UUID de l'artefact OCR complet écrit hors ClaimState
      - `artifact_path`      : chemin relatif sous storage/ (jamais absolu)

    Sérialisation :
      - JSON-sérialisable via model_dump(mode="json") — garanti par StrictModel Pydantic v2.
      - Tous les types non-JSON natifs (datetime, Decimal, Enum) sont convertis automatiquement.
    """

    claim_id: str
    file_path: str
    sha256: str
    mime_type: str
    extraction_status: ExtractionStatus
    status: VerificationStatus
    classification: DocumentClassification | None = Field(
        default=None,
        description="Classification complète du document (type, confiance, source, scores, version)",
    )
    document_type: DocumentType
    ocr_source: OcrSource
    pages: list[DocumentPageContent] = Field(default_factory=list)
    full_text: str = ""
    extracted_fields: dict[str, ExtractedField] = Field(default_factory=dict)
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Score global de confiance de l'extraction en [0.0, 1.0]",
    )
    is_readable: bool = False
    human_review_required: bool = False
    human_review_reasons: list[str] = Field(default_factory=list)
    reason_codes: list[OcrCode] = Field(default_factory=list)
    unreadable_documents: list[str] = Field(default_factory=list)
    errors: list[str] = Field(
        default_factory=list,
        description="Anomalies bloquantes ayant empêché ou altéré l'extraction",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Anomalies non bloquantes — fallbacks, champs optionnels absents, "
                    "confiance légèrement sous-optimale",
    )
    reasons: list[str] = Field(default_factory=list)
    tool_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Versions des outils utilisés : pdf_reader, ocr_engine, classifier, "
                    "confidence, parser, ocr_thresholds",
    )
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    audit_entry: DocumentOcrAuditEntry | None = None
    extraction: DocumentExtraction | None = None
    artifact_id: str | None = None
    artifact_path: str | None = None
    security_findings: list[SecurityFinding] = Field(default_factory=list)
    structured_errors: list[OcrError] = Field(
        default_factory=list,
        description="Erreurs structurées Étape 18 — code stable, sévérité, document, retryable",
    )
    llm_metadata: LlmMetadata | None = None


# ── 7. Medical Coding Agent ───────────────────────────────────────────────────


class ProcedureCoding(StrictModel):
    original_description: str
    proposed_code: str | None = None
    rule_applied: str | None = None
    status: VerificationStatus
    alternatives: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class MedicalCodingResult(StrictModel):
    """Correspondance actes → table de codes locale versionnée."""

    case_id: str
    status: VerificationStatus
    codings: list[ProcedureCoding] = Field(default_factory=list)
    table_version: str = "1.0.0"
    reasons: list[str] = Field(default_factory=list)
    llm_metadata: LlmMetadata


# ── 8. Clinical Consistency Agent ─────────────────────────────────────────────


class ClinicalEvidenceSource(str, Enum):
    """Origine d'une preuve clinique — jamais une valeur brute non attribuée."""

    OCR_EXTRACTION = "ocr_extraction"
    MEDICAL_CODING = "medical_coding"


class ClinicalEvidence(StrictModel):
    """Preuve structurée minimale appuyant un signal ou une incohérence clinique.

    Toujours attribuée à une source et à un champ précis — jamais une valeur
    flottante sans origine. ``value`` est toujours une chaîne courte (jamais un
    objet complexe, un document brut ou un texte OCR complet) : la preuve
    référence une donnée déjà extraite/validée, elle ne transporte jamais de
    contenu de document. ``evidence_id`` est l'identifiant opaque référencé
    par ``ClinicalConsistencyResult.evidence_ids`` — jamais dérivé d'un
    contenu sensible, jamais réutilisable pour retrouver la valeur brute.
    """

    evidence_id: str = Field(default_factory=lambda: f"EVID-{uuid.uuid4().hex[:10]}")
    source: ClinicalEvidenceSource
    field: str = Field(..., min_length=1, max_length=100)
    document_reference: str | None = Field(
        default=None,
        max_length=100,
        description=(
            "Type de document (ex. 'INVOICE', 'PRESCRIPTION') si la preuve provient "
            "de l'OCR, ou identifiant de l'évaluation source (ex. 'coding_result') "
            "sinon — jamais un chemin de fichier ni un contenu de document."
        ),
    )
    value: str = Field(..., min_length=1, max_length=500)

    @field_validator("value")
    @classmethod
    def _value_no_raw_document(cls, v: str) -> str:
        return _reject_unstructured_content(v, "value")


class ClinicalSignal(StrictModel):
    """Anomalie de cohérence clinique détectée — toujours référencée.

    ``fields_compared``/``documents_compared`` : au moins un des deux est
    obligatoire (validé ci-dessous) — un signal ne peut jamais reposer sur une
    simple description libre sans référence structurée à ce qui a été comparé.
    """

    signal_type: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=1000)
    fields_compared: list[str] = Field(default_factory=list)
    documents_compared: list[str] = Field(default_factory=list)
    evidence: list[ClinicalEvidence] = Field(default_factory=list)
    severity: SeverityLevel = SeverityLevel.MEDIUM

    @model_validator(mode="after")
    def _must_reference_fields_or_documents(self) -> "ClinicalSignal":
        if not self.fields_compared and not self.documents_compared:
            raise ValueError(
                "ClinicalSignal doit référencer au moins un champ "
                "(fields_compared) ou un document (documents_compared) comparé "
                "— jamais une sortie libre non structurée."
            )
        return self


class ClinicalInconsistency(StrictModel):
    """Incohérence concrète entre deux valeurs déjà validées (ex. compteur OCR
    vs codes résolus) — toujours appuyée par au moins une preuve structurée.

    Distinct de ``DisagreementPoint`` (désaccord générique de statut entre
    agents, consommé par ``CaseReviewerResult``/``tools.consistency`` — sans
    ``evidence`` ni ``severity``) : une ``ClinicalInconsistency`` est un
    désaccord de valeur interne à l'analyse clinique, jamais un nouveau
    désaccord de statut entre agents.
    """

    inconsistency_type: str = Field(..., min_length=1, max_length=100)
    expected: str = Field(..., min_length=1, max_length=500)
    observed: str = Field(..., min_length=1, max_length=500)
    severity: SeverityLevel = SeverityLevel.MEDIUM
    evidence: list[ClinicalEvidence] = Field(..., min_length=1)

    @field_validator("expected", "observed")
    @classmethod
    def _no_raw_document(cls, v: str) -> str:
        return _reject_unstructured_content(v, "expected/observed")


class ClinicalResultPayload(StrictModel):
    """Détail métier de l'évaluation clinique — jamais de contenu brut.

    Distinct de l'enveloppe ``ClinicalConsistencyResult`` (statut, trace LLM,
    confiance, erreurs, identifiants de preuve, besoin de revue) : ce
    sous-modèle porte uniquement les données propres à l'analyse clinique
    elle-même, réutilisable tel quel si un futur agent devait produire une
    enveloppe générique similaire.
    """

    procedure_count: int | None = Field(default=None, ge=0)
    medication_count: int | None = Field(default=None, ge=0)
    prescription_required: bool | None = None
    signals: list[ClinicalSignal] = Field(default_factory=list)
    inconsistencies: list[ClinicalInconsistency] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    @field_validator("reasons")
    @classmethod
    def _reasons_no_raw_document(cls, v: list[str]) -> list[str]:
        return [_reject_unstructured_content(item, "reasons") for item in v]


class ClinicalConsistencyResult(StrictModel):
    """Cohérence clinique entre ordonnance, consultation et chronologie.

    Enveloppe générique : ``status``/``llm_trace``/``confidence``/``errors``/
    ``evidence_ids``/``human_review_required`` sont le contrat commun exposé à
    l'orchestrateur et à la revue humaine ; ``result_payload`` porte le détail
    métier (voir ``ClinicalResultPayload``). ``llm_trace`` est obligatoire et
    ne peut jamais être ``None`` : un résultat sans trace LLM ne représente
    jamais une exécution valide (règle projet fail-closed).
    """

    case_id: str
    status: VerificationStatus
    llm_trace: LlmMetadata
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    errors: list[StructuredError] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    human_review_required: bool = False
    result_payload: ClinicalResultPayload = Field(default_factory=ClinicalResultPayload)

    @model_validator(mode="after")
    def _evidence_ids_must_reference_real_evidence(self) -> "ClinicalConsistencyResult":
        real_ids = {
            evidence.evidence_id
            for signal in self.result_payload.signals
            for evidence in signal.evidence
        } | {
            evidence.evidence_id
            for inconsistency in self.result_payload.inconsistencies
            for evidence in inconsistency.evidence
        }
        unknown = [i for i in self.evidence_ids if i not in real_ids]
        if unknown:
            raise ValueError(
                f"evidence_ids référence des preuves inexistantes : {unknown} — "
                "jamais un identifiant inventé."
            )
        return self


# ── 9. Fraud Detection Agent ──────────────────────────────────────────────────


class FraudEvidenceSource(str, Enum):
    """Origine d'une preuve anti-fraude — jamais une valeur brute non attribuée."""

    OCR_EXTRACTION = "ocr_extraction"
    MEDICAL_CODING = "medical_coding"
    IDENTITY_COVERAGE = "identity_coverage"
    DUPLICATE_INDEX = "duplicate_index"


class FraudEvidence(StrictModel):
    """Preuve structurée minimale appuyant un signal anti-fraude.

    Même patron que ``ClinicalEvidence`` : toujours attribuée, jamais de
    contenu de document brut, jamais de texte OCR complet.
    """

    evidence_id: str = Field(default_factory=lambda: f"EVID-{uuid.uuid4().hex[:10]}")
    source: FraudEvidenceSource
    field: str = Field(..., min_length=1, max_length=100)
    document_reference: str | None = Field(
        default=None,
        max_length=100,
        description=(
            "Identifiant de l'évaluation source (ex. 'identity_coverage_result', "
            "'coding_result') — jamais un chemin de fichier ni un contenu de document."
        ),
    )
    value: str = Field(..., min_length=1, max_length=500)

    @field_validator("value")
    @classmethod
    def _value_no_raw_document(cls, v: str) -> str:
        return _reject_unstructured_content(v, "value")


class FraudSignal(StrictModel):
    """Signal de risque pondéré — toujours référencé par au moins une preuve.

    ``evidence`` ne peut jamais être vide : un signal de fraude combine
    uniquement des preuves déjà validées par d'autres agents (voir
    ``agents/fraud_detection_agent/agent.py``), jamais une affirmation non
    appuyée.
    """

    signal_type: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=1000)
    risk_contribution: float = Field(ge=0.0, le=1.0)
    severity: SeverityLevel = SeverityLevel.MEDIUM
    evidence: list[FraudEvidence] = Field(..., min_length=1)


class FraudResultPayload(StrictModel):
    """Détail métier de la détection de fraude — jamais de contenu brut.

    Distinct de l'enveloppe ``FraudDetectionResult`` (voir
    ``ClinicalResultPayload`` pour le même patron côté cohérence clinique).
    """

    duplicate_invoice: bool | None = None
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    signals: list[FraudSignal] = Field(default_factory=list)
    threshold_version: str = "1.0.0"
    reasons: list[str] = Field(default_factory=list)

    @field_validator("reasons")
    @classmethod
    def _reasons_no_raw_document(cls, v: list[str]) -> list[str]:
        return [_reject_unstructured_content(item, "reasons") for item in v]


class FraudDetectionResult(StrictModel):
    """Détection de doublons et anomalies sur données synthétiques.

    Même enveloppe générique que ``ClinicalConsistencyResult`` — voir sa
    docstring pour le rôle de chaque champ commun.
    """

    case_id: str
    status: VerificationStatus
    llm_trace: LlmMetadata
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    errors: list[StructuredError] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    human_review_required: bool = False
    result_payload: FraudResultPayload = Field(default_factory=FraudResultPayload)

    @model_validator(mode="after")
    def _evidence_ids_must_reference_real_evidence(self) -> "FraudDetectionResult":
        real_ids = {
            evidence.evidence_id
            for signal in self.result_payload.signals
            for evidence in signal.evidence
        }
        unknown = [i for i in self.evidence_ids if i not in real_ids]
        if unknown:
            raise ValueError(
                f"evidence_ids référence des preuves inexistantes : {unknown} — "
                "jamais un identifiant inventé."
            )
        return self


# ── 10. Case Reviewer Agent ───────────────────────────────────────────────────


class DisagreementPoint(StrictModel):
    agent: str
    field: str
    expected: str
    observed: str


class CaseReviewerResultPayload(StrictModel):
    """Détail métier de la synthèse multi-agent — jamais de contenu brut.

    Distinct de l'enveloppe ``CaseReviewerResult`` (statut verrouillé, trace
    LLM, confiance, erreurs, identifiants de preuve, revue humaine toujours
    requise) : ce sous-modèle porte la pré-recommandation elle-même et son
    argumentaire, même patron que ``ClinicalResultPayload``/``FraudResultPayload``.

    ``risks`` : signaux de risque à porter à l'attention de l'humain (plafond
    dépassé, score de fraude élevé, incohérence clinique…), toujours dérivés de
    données déjà calculées par les agents amont — jamais une affirmation
    inventée par ce module. ``human_review_reasons`` (« questions humain ») ne
    peut jamais être vide : une revue humaine toujours obligatoire doit
    toujours porter au moins un motif explicite à examiner.
    """

    recommendation: Recommendation
    justification: list[str] = Field(default_factory=list)
    disagreements: list[DisagreementPoint] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    human_review_reasons: list[str] = Field(..., min_length=1)

    @field_validator("justification", "risks", "human_review_reasons")
    @classmethod
    def _no_raw_content(cls, v: list[str], info) -> list[str]:
        return [_reject_unstructured_content(item, info.field_name) for item in v]


class CaseReviewerResult(StrictModel):
    """Synthèse multi-agent — jamais de décision finale, revue humaine toujours obligatoire.

    Même enveloppe générique que ``ClinicalConsistencyResult``/``FraudDetectionResult``
    (voir leur docstring pour le rôle des champs communs) : ``status``/``llm_trace``/
    ``confidence``/``errors``/``evidence_ids``/``human_review_required`` sont le
    contrat commun exposé à l'orchestrateur et à la revue humaine ; ``result_payload``
    porte la pré-recommandation elle-même (voir ``CaseReviewerResultPayload``).

    Contrairement aux autres agents de l'enveloppe générique, ``status`` et
    ``human_review_required`` sont **verrouillés** : ``status`` ne peut jamais être
    autre chose que ``NEEDS_REVIEW`` et ``human_review_required`` ne peut jamais être
    ``False`` — aucune instance valide de ``CaseReviewerResult`` ne peut donc
    représenter une décision finale automatique, quelle que soit l'implémentation qui
    la construit (garantie de schéma, pas seulement de nœud LangGraph — voir aussi
    ``agents/case_reviewer_agent/agent.py::_force_human_review`` pour la défense en
    profondeur côté nœud). La pré-recommandation réelle (APPROVE/REJECT/PENDING,
    toujours révisable) reste disponible dans ``result_payload.recommendation``.
    """

    case_id: str
    status: VerificationStatus = VerificationStatus.NEEDS_REVIEW
    llm_trace: LlmMetadata
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    errors: list[StructuredError] = Field(default_factory=list)
    evidence_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Identifiants de preuves déjà validées par les agents amont (ex. "
            "ClinicalConsistencyResult.evidence_ids, FraudDetectionResult."
            "evidence_ids) — CaseReviewerResult ne porte aucun objet de preuve "
            "propre, contrairement à ClinicalConsistencyResult/FraudDetectionResult."
        ),
    )
    human_review_required: bool = True
    result_payload: CaseReviewerResultPayload

    @field_validator("status")
    @classmethod
    def _status_locked_to_needs_review(cls, v: VerificationStatus) -> VerificationStatus:
        if v is not VerificationStatus.NEEDS_REVIEW:
            raise ValueError(
                "status doit toujours être NEEDS_REVIEW : case_reviewer_agent ne "
                "produit jamais de statut final (PASS/FAIL) — la décision reste "
                "toujours révisable par un humain."
            )
        return v

    @field_validator("human_review_required")
    @classmethod
    def _human_review_locked_to_true(cls, v: bool) -> bool:
        if not v:
            raise ValueError(
                "human_review_required doit toujours être True : aucune décision "
                "finale automatique n'est autorisée pour case_reviewer_agent."
            )
        return v

    @field_validator("evidence_ids")
    @classmethod
    def _evidence_ids_no_raw_content(cls, v: list[str]) -> list[str]:
        return [_reject_unstructured_content(item, "evidence_ids") for item in v]


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


class AuditResult(StrictModel):
    """Synthèse typée produite par l'Audit Agent.

    Le journal détaillé reste dans `audit_trail` côté ClaimState. Ce résultat
    conserve seulement une synthèse sérialisable et, si nécessaire, les
    événements minimisés déjà autorisés par AuditEvent.
    """

    case_id: str
    status: VerificationStatus
    events_count: int = Field(default=0, ge=0)
    events: list[AuditEvent] = Field(default_factory=list)
    policy_version: str = "1.0.0"
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reasons: list[str] = Field(default_factory=list)
    llm_metadata: LlmMetadata | None = None
