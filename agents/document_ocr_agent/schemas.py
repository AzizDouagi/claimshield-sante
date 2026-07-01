"""Schéma d'entrée du Document/OCR Agent.

Réexporte également les types de sortie depuis schemas.results et schemas.domain
pour un import unifié depuis l'extérieur du package.
"""

from __future__ import annotations

import re

from pydantic import Field, field_validator

from schemas.domain import (
    OCR_ERROR_CODE_DESCRIPTIONS,
    OCR_ERROR_CODE_RETRYABLE,
    OCR_ERROR_CODE_SEVERITIES,
    ExtractionStatus,
    OcrCode,
    OcrSource,
    SecurityDecision,
    StrictModel,
)
from schemas.results import (
    DocumentClassification,
    DocumentExtraction,
    DocumentOcrAuditEntry,
    DocumentOcrResult,
    DocumentPageContent,
    EssentialFields,
    ExtractedField,
    FieldProvenance,
    MedicalItem,
    MonetaryAmount,
    OcrError,
    PageText,
)
from schemas.results import ESSENTIAL_FIELD_NAMES

__all__ = [
    "DocumentClassification",
    "DocumentExtraction",
    "DocumentOcrAuditEntry",
    "DocumentOcrInput",
    "DocumentOcrResult",
    "DocumentPageContent",
    "ESSENTIAL_FIELD_NAMES",
    "EssentialFields",
    "ExtractedField",
    "ExtractionStatus",
    "FieldProvenance",
    "MedicalItem",
    "MonetaryAmount",
    "OCR_ERROR_CODE_DESCRIPTIONS",
    "OCR_ERROR_CODE_RETRYABLE",
    "OCR_ERROR_CODE_SEVERITIES",
    "OcrCode",
    "OcrError",
    "OcrSource",
    "PageText",
    "LlmOcrDecision",
]

_ABSOLUTE_PATH_RE = re.compile(r"^(?:/|[A-Za-z]:[/\\]|\\\\)")
_SECRET_HINT_RE = re.compile(
    r"(?:api[_-]?key|secret\s*[:=]|password\s*[:=]|token\s*[:=]|bearer\s+\S+)",
    re.IGNORECASE,
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED_MIMES = frozenset(
    {
        "application/pdf",
        "image/png",
        "image/jpeg",
        "image/jpg",
        "application/json",  # FHIR Bundle JSON — traité comme SKIPPED par l'agent
    }
)

_SCHEMA_VERSION = "1.0.0"


class DocumentOcrInput(StrictModel):
    """Entrée du Document/OCR Agent.

    Tous les chemins sont relatifs à la racine du projet (jamais absolus).
    Le fichier doit se trouver dans la zone incoming/ assainie.
    La décision du Security Gate est embarquée pour une entrée auto-suffisante.
    """

    claim_id: str = Field(..., min_length=1, max_length=50)
    document_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Identifiant unique du document dans le dossier (ex: CLM-0001-doc-0)",
    )
    filename: str = Field(..., min_length=1, max_length=255, description="Nom original du fichier")
    mime_type: str = Field(..., description="Type MIME validé par le Security Gate")
    sha256: str = Field(..., description="Empreinte SHA-256 hex du fichier (64 caractères)")
    sanitized_path: str = Field(
        ...,
        description="Chemin relatif assaini sous incoming/ (ex: incoming/CLM-0001/facture.pdf)",
    )
    security_decision: SecurityDecision = Field(
        ..., description="Décision du Security Gate (ALLOW requis pour l'extraction)"
    )
    schema_version: str = Field(
        default=_SCHEMA_VERSION,
        min_length=1,
        description="Version du schéma pour compatibilité des checkpoints LangGraph",
    )
    file_index: int = Field(default=0, ge=0, description="Index du fichier dans le dossier (0-indexé)")

    @field_validator("sanitized_path", "filename")
    @classmethod
    def _no_absolute_path(cls, v: str) -> str:
        if _ABSOLUTE_PATH_RE.match(v):
            raise ValueError(f"Chemin absolu interdit : {v!r}")
        if ".." in v:
            raise ValueError(f"Traversée de répertoire interdite : {v!r}")
        return v

    @field_validator("sha256")
    @classmethod
    def _valid_sha256(cls, v: str) -> str:
        if not _SHA256_RE.match(v.lower()):
            raise ValueError(f"SHA-256 invalide (doit être 64 hex) : {v!r}")
        return v.lower()

    @field_validator("mime_type")
    @classmethod
    def _supported_mime(cls, v: str) -> str:
        if v.lower() not in _SUPPORTED_MIMES:
            raise ValueError(
                f"Type MIME non supporté par l'agent OCR : {v!r}. "
                f"Types acceptés : {sorted(_SUPPORTED_MIMES)}"
            )
        return v.lower()


# ── Schéma de décision LLM (intermédiaire — jamais dans ClaimState) ───────────


def _reject_llm_leak(v: str, field_name: str) -> str:
    if _ABSOLUTE_PATH_RE.match(v):
        raise ValueError(f"Chemin absolu interdit dans {field_name}")
    if _SECRET_HINT_RE.search(v):
        raise ValueError(f"Secret potentiel interdit dans {field_name}")
    return v


class LlmOcrDecision(StrictModel):
    """Décision LLM pour classification et extraction OCR."""

    document_type: str = Field(default="UNKNOWN", max_length=50)
    extracted_fields: dict[str, str] = Field(default_factory=dict)
    confidence_assessment: str = Field(default="", max_length=500)
    reasons: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("document_type", "confidence_assessment")
    @classmethod
    def _no_sensitive_string(cls, v: str, info) -> str:
        return _reject_llm_leak(v, info.field_name)

    @field_validator("extracted_fields")
    @classmethod
    def _no_sensitive_fields(cls, v: dict[str, str]) -> dict[str, str]:
        return {
            _reject_llm_leak(str(key), "extracted_fields.key"): _reject_llm_leak(
                str(value), f"extracted_fields[{key!r}]"
            )
            for key, value in v.items()
        }

    @field_validator("reasons")
    @classmethod
    def _no_sensitive_reasons(cls, v: list[str]) -> list[str]:
        return [_reject_llm_leak(str(reason), "reasons") for reason in v]
