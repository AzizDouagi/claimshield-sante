"""Schémas d'entrée et types intermédiaires du FHIR Validator Agent — ClaimShield Santé.

Tous les modèles :
  - héritent de StrictModel (extra='forbid') — tout champ inconnu lève ValidationError.
  - sont JSON-sérialisables via model_dump(mode="json").
  - n'exposent jamais de donnée personnelle brute ni de secret.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import Enum

from pydantic import Field, field_validator
from schemas.domain import SeverityLevel, StrictModel
from schemas.results import FhirValidatorResult  # re-export public

_ABSOLUTE_PATH_RE = re.compile(r"^(?:/|[A-Za-z]:[/\\]|\\\\)")
_SECRET_HINT_RE = re.compile(
    r"(?:api[_-]?key|secret\s*[:=]|password\s*[:=]|token\s*[:=]|bearer\s+\S+)",
    re.IGNORECASE,
)


def _reject_llm_leak(value: str, field_name: str) -> str:
    if _ABSOLUTE_PATH_RE.match(value):
        raise ValueError(f"Chemin absolu interdit dans {field_name}")
    if _SECRET_HINT_RE.search(value):
        raise ValueError(f"Secret potentiel interdit dans {field_name}")
    return value


# ── Statut de validation FHIR ─────────────────────────────────────────────────


class FhirValidationStatus(str, Enum):
    """Statut de validation d'un bundle ou d'une ressource FHIR.

    Plus fin que VerificationStatus pour le domaine FHIR :
      VALID              — bundle valide sans anomalie.
      INVALID            — erreurs bloquantes présentes.
      VALID_WITH_WARNINGS — structurellement valide mais avec avertissements.
      NOT_PROVIDED       — fhir_bundle_path absent ou None.
      UNSUPPORTED        — version FHIR ou profil non supporté.
      NOT_EVALUATED      — évaluation non déclenchée (ex. Security Gate BLOCK en amont).
    """

    VALID = "VALID"
    INVALID = "INVALID"
    VALID_WITH_WARNINGS = "VALID_WITH_WARNINGS"
    NOT_PROVIDED = "NOT_PROVIDED"
    UNSUPPORTED = "UNSUPPORTED"
    NOT_EVALUATED = "NOT_EVALUATED"


# ── FhirIssue ─────────────────────────────────────────────────────────────────


class FhirIssue(StrictModel):
    """Anomalie détectée lors de la validation FHIR.

    Conforme à HL7 FHIR OperationOutcome.issue :
      - severity   : niveau aligné sur SeverityLevel (LOW → CRITICAL).
      - code       : code stable HL7 (ex. "required", "invalid", "not-found").
      - location   : chemin FHIRPath ou XPath identifiant l'élément concerné.
      - details    : message lisible — jamais de donnée personnelle brute.
      - expression : expression FHIRPath si disponible.
    """

    severity: SeverityLevel = Field(..., description="Niveau de sévérité de l'anomalie")
    code: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Code stable HL7 FHIR (ex. 'required', 'invalid', 'not-found')",
    )
    location: str | None = Field(
        default=None,
        max_length=500,
        description="Chemin FHIRPath ou XPath vers l'élément concerné",
    )
    details: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Message lisible décrivant l'anomalie — jamais de donnée personnelle",
    )
    expression: str | None = Field(
        default=None,
        max_length=500,
        description="Expression FHIRPath si disponible",
    )


# ── FhirResourceSummary ───────────────────────────────────────────────────────


class FhirResourceSummary(StrictModel):
    """Résumé de validation pour un type de ressource FHIR.

    Permet d'identifier quels types de ressources posent problème
    sans exposer le contenu des ressources elles-mêmes.
    """

    resource_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Type FHIR (ex. 'Patient', 'Claim', 'Coverage', 'Encounter')",
    )
    count: int = Field(..., ge=0, description="Nombre d'instances dans le bundle")
    valid_count: int = Field(default=0, ge=0, description="Instances sans erreur bloquante")
    invalid_count: int = Field(default=0, ge=0, description="Instances avec au moins une erreur bloquante")
    warning_count: int = Field(default=0, ge=0, description="Instances avec uniquement des avertissements")
    status: FhirValidationStatus = Field(
        ...,
        description="Statut agrégé pour ce type de ressource",
    )
    profile_url: str | None = Field(
        default=None,
        max_length=500,
        description="URL du profil HL7 appliqué, si disponible",
    )
    issues: list[FhirIssue] = Field(
        default_factory=list,
        description="Anomalies détectées pour ce type de ressource",
    )


# ── Référence non résolue ─────────────────────────────────────────────────────


class UnresolvedReference(StrictModel):
    """Référence FHIR présente dans le bundle mais non résolue.

    Une référence est non résolue si la ressource cible est absente du bundle
    et n'est pas une référence externe déclarée admissible.
    """

    source_resource_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Type de la ressource source (ex. 'Claim')",
    )
    source_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Identifiant logique de la ressource source",
    )
    reference_path: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Chemin FHIRPath pointant vers la référence (ex. 'Claim.patient.reference')",
    )
    reference_value: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Valeur de la référence non résolue (ex. 'Patient/12345')",
    )
    is_external: bool = Field(
        default=False,
        description="Vrai si la référence pointe vers une ressource externe au bundle",
    )


# ── FhirValidatorInput ────────────────────────────────────────────────────────

_SUPPORTED_FHIR_VERSIONS: frozenset[str] = frozenset({"R4", "R4B"})


class FhirValidatorInput(StrictModel):
    """Entrée du nœud FHIR Validator Agent dans le workflow LangGraph.

    Contient les informations nécessaires pour localiser et valider
    le bundle FHIR R4 associé à un dossier de remboursement.

    Contraintes de sécurité :
      - fhir_bundle_path doit être un chemin relatif (jamais absolu).
      - Aucun secret ni donnée personnelle brute dans ce schéma.
      - fhir_version doit appartenir aux versions supportées (R4, R4B).
    """

    case_id: str = Field(..., min_length=1, description="Identifiant du dossier")
    fhir_bundle_path: str | None = Field(
        default=None,
        description="Chemin relatif vers le bundle FHIR sous storage/incoming/",
    )
    bundle_expected: bool = Field(
        default=True,
        description="Indique si un bundle FHIR est attendu pour ce dossier",
    )
    fhir_version: str = Field(
        default="R4",
        description="Version FHIR attendue du bundle (R4 ou R4B)",
    )
    profile_url: str | None = Field(
        default=None,
        max_length=500,
        description="URL du profil HL7 à appliquer (None = validation structurelle uniquement)",
    )
    validator_version: str = Field(
        default="1.0.0",
        description="Version du validateur FHIR utilisé",
    )
    rules_version: str = Field(
        default="1.0.0",
        description="Version des règles métier de validation ClaimShield",
    )
    validation_scope: str = Field(
        default="STRUCTURAL_ONLY",
        description="Périmètre : 'STRUCTURAL_ONLY' ou 'FULL' (validation de profil incluse)",
    )

    @field_validator("fhir_bundle_path")
    @classmethod
    def no_absolute_path(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if _ABSOLUTE_PATH_RE.match(v):
            raise ValueError(
                f"Chemin absolu interdit dans fhir_bundle_path : {v!r}. "
                "Utilisez un chemin relatif sous storage/incoming/."
            )
        return v

    @field_validator("fhir_version")
    @classmethod
    def supported_fhir_version(cls, v: str) -> str:
        normalised = v.upper()
        if normalised not in _SUPPORTED_FHIR_VERSIONS:
            raise ValueError(
                f"Version FHIR non supportée : {v!r}. "
                f"Versions supportées : {sorted(_SUPPORTED_FHIR_VERSIONS)}"
            )
        return normalised


# ── FhirValidationDetail ──────────────────────────────────────────────────────


class FhirValidationDetail(StrictModel):
    """Résultat détaillé de la validation FHIR — produit par l'agent, consommé par FhirValidatorResult.

    Regroupe l'ensemble des artefacts de validation :
      - statut fin-grain (FhirValidationStatus)
      - résumés par type de ressource (FhirResourceSummary)
      - références non résolues (UnresolvedReference)
      - versions du validateur et des règles
      - erreurs bloquantes et avertissements non bloquants (FhirIssue)
    """

    fhir_validation_status: FhirValidationStatus = Field(
        ...,
        description="Statut fin-grain (VALID, INVALID, VALID_WITH_WARNINGS, NOT_PROVIDED, …)",
    )
    fhir_version: str = Field(
        default="R4",
        description="Version FHIR du bundle validé",
    )
    validator_version: str = Field(
        default="1.0.0",
        description="Version du validateur FHIR",
    )
    rules_version: str = Field(
        default="1.0.0",
        description="Version des règles métier de validation ClaimShield",
    )
    validation_scope: str = Field(
        default="STRUCTURAL_ONLY",
        description="Périmètre de validation : 'STRUCTURAL_ONLY' ou 'FULL'",
    )
    resource_summaries: list[FhirResourceSummary] = Field(
        default_factory=list,
        description="Résumé de validation par type de ressource FHIR",
    )
    unresolved_references: list[UnresolvedReference] = Field(
        default_factory=list,
        description="Références FHIR présentes dans le bundle mais non résolues",
    )
    errors: list[FhirIssue] = Field(
        default_factory=list,
        description="Erreurs bloquantes (sévérité HIGH ou CRITICAL)",
    )
    warnings: list[FhirIssue] = Field(
        default_factory=list,
        description="Avertissements non bloquants (sévérité LOW ou MEDIUM)",
    )
    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Horodatage UTC de la validation",
    )


# ── Schéma de décision LLM (intermédiaire — jamais dans ClaimState) ───────────


class LlmFhirDecision(StrictModel):
    """Décision LLM pour la validation FHIR."""

    recommended_status: str = Field(default="NEEDS_REVIEW", max_length=50)
    clinical_context: str = Field(default="", max_length=500)
    reasons: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("recommended_status")
    @classmethod
    def supported_status(cls, v: str) -> str:
        value = _reject_llm_leak(v, "recommended_status").upper()
        if value not in {"PASS", "NEEDS_REVIEW", "FAIL"}:
            raise ValueError(f"Statut LLM non supporté : {v!r}")
        return value

    @field_validator("clinical_context")
    @classmethod
    def no_sensitive_context(cls, v: str) -> str:
        return _reject_llm_leak(v, "clinical_context")

    @field_validator("reasons")
    @classmethod
    def no_sensitive_reasons(cls, v: list[str]) -> list[str]:
        return [_reject_llm_leak(str(item), "reasons") for item in v]


__all__ = [
    "FhirValidationDetail",
    "FhirValidationStatus",
    "FhirIssue",
    "FhirResourceSummary",
    "FhirValidatorInput",
    "FhirValidatorResult",
    "LlmFhirDecision",
    "UnresolvedReference",
]
