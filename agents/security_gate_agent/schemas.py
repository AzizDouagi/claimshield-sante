"""Schémas d'entrée et de décision du security_gate_agent.

Décisions possibles (SecurityDecision) :
┌─────────────┬──────────────────────────────────────────────────────────┐
│ Décision    │ Signification                                            │
├─────────────┼──────────────────────────────────────────────────────────┤
│ ALLOW       │ L'entrée peut continuer vers l'étape suivante            │
│ BLOCK       │ L'action est interdite et le traitement s'arrête         │
│ QUARANTINE  │ Le fichier est isolé pour contrôle ou revue humaine      │
└─────────────┴──────────────────────────────────────────────────────────┘

Contraintes communes à chaque décision (portées par SecurityGateResult) :
  - au moins un motif dans `reasons` (min_length=1)
  - version de la politique appliquée dans `policy_version`
  - liste structurée des anomalies dans `findings`
  - horodatage dans `evaluated_at`
  - sérialisable JSON (StrictModel Pydantic)
  - aucun document brut, secret ou chemin absolu
"""
from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import Field, field_validator

from schemas.domain import (
    FindingCode,
    InputType,
    SecurityDecision,
    SeverityLevel,
    StrictModel,
)

__all__ = [
    "LlmSecurityDecision",
    "SecurityDecision",
    "InputType",
    "SeverityLevel",
    "FindingCode",
    "SecurityGateInput",
]

_ABSOLUTE_PATH_RE = re.compile(r"^(?:/|[A-Za-z]:[/\\]|\\\\)")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SECRET_HINT_RE = re.compile(
    r"(?:api[_-]?key|secret\s*[:=]|password\s*[:=]|token\s*[:=]|bearer\s+[a-z0-9._~+/=-]+)",
    re.IGNORECASE,
)


class SecurityGateInput(StrictModel):
    """Données nécessaires à l'évaluation de sécurité d'une entrée.

    Représente UN élément à évaluer : un fichier, un texte, une URL,
    un appel d'outil ou la sortie d'un agent.

    Contraintes :
      - `relative_path` ne peut pas être un chemin absolu.
      - `text_excerpt` est limité à 2 000 caractères — jamais le document brut.
      - Aucun secret (mot de passe, clé API, token) ne doit figurer ici.
      - `sha256`, s'il est fourni, doit faire exactement 64 caractères hex.
    """

    # ── Identifiants ─────────────────────────────────────────────────────────

    claim_id: str = Field(
        ...,
        description="Identifiant du dossier (ex. CLM-0001)",
    )
    entry_id: str = Field(
        ...,
        description="Identifiant unique de ce fichier ou de cette action dans le dossier",
    )
    input_type: InputType = Field(
        ...,
        description="Type d'entrée : fichier, texte, URL, outil ou sortie d'agent",
    )

    # ── Métadonnées fichier ───────────────────────────────────────────────────

    filename: str | None = Field(
        default=None,
        description="Nom original du fichier",
    )
    extension: str | None = Field(
        default=None,
        description="Extension normalisée (ex. .pdf)",
    )
    detected_mime: str | None = Field(
        default=None,
        description="Type MIME détecté à la lecture physique du fichier",
    )
    actual_size: int | None = Field(
        default=None,
        ge=0,
        description="Taille réelle en octets",
    )
    sha256: Annotated[str, Field(min_length=64, max_length=64)] | None = Field(
        default=None,
        description="Hash SHA-256 du fichier (64 caractères hex)",
    )
    relative_path: str | None = Field(
        default=None,
        description="Chemin relatif dans la zone de stockage — jamais absolu",
    )

    # ── Données complémentaires ───────────────────────────────────────────────

    url: str | None = Field(
        default=None,
        description="URL éventuelle à analyser",
    )
    text_excerpt: str | None = Field(
        default=None,
        max_length=2_000,
        description=(
            "Extrait de texte à analyser — limité à 2 000 caractères, "
            "jamais le document brut"
        ),
    )
    text_source: Literal[
        "text_excerpt",
        "pdf_text",
        "ocr_preview",
        "agent_output",
        "tool_arguments",
        "metadata",
        "url",
    ] | None = Field(
        default=None,
        description="Origine contrôlée de l'extrait texte minimisé",
    )
    requesting_agent: str | None = Field(
        default=None,
        description="Nom de l'agent ou de l'outil demandeur",
    )

    # ── Oracle (tests uniquement) ─────────────────────────────────────────────

    deterministic_injection_flag: bool | None = Field(
        default=None,
        description="Flag déterministe issu de ground_truth — usage test uniquement",
    )

    # ── Validateurs ───────────────────────────────────────────────────────────

    @field_validator("relative_path", mode="before")
    @classmethod
    def no_absolute_path(cls, v: object) -> object:
        if isinstance(v, str) and _ABSOLUTE_PATH_RE.match(v):
            raise ValueError(
                f"Chemin absolu interdit dans relative_path : {v!r}. "
                "Utiliser un chemin relatif à la racine du stockage."
            )
        if isinstance(v, str) and any(part == ".." for part in re.split(r"[/\\]+", v)):
            raise ValueError(
                f"Traversée de répertoire interdite dans relative_path : {v!r}."
            )
        return v

    @field_validator("sha256")
    @classmethod
    def sha256_must_be_hex(cls, v: str | None) -> str | None:
        if v is not None and not _SHA256_RE.fullmatch(v):
            raise ValueError("sha256 doit contenir exactement 64 caractères hexadécimaux")
        return v

    @field_validator(
        "filename",
        "extension",
        "detected_mime",
        "relative_path",
        "url",
        "text_excerpt",
        "text_source",
        "requesting_agent",
    )
    @classmethod
    def no_secret_hint(cls, v: str | None) -> str | None:
        if v is not None and _SECRET_HINT_RE.search(v):
            raise ValueError("Secret potentiel interdit dans SecurityGateInput")
        return v


# ── Schéma de décision LLM (intermédiaire — jamais dans ClaimState) ───────────


class LlmSecurityDecision(StrictModel):
    """Décision structurée du LLM de sécurité.

    Le schéma est volontairement strict : aucun statut ambigu, preuve
    minimisée et score de confiance borné.
    """

    decision: Literal["ALLOW", "BLOCK", "QUARANTINE"] = "BLOCK"
    reasons: list[str] = Field(default_factory=list)
    explanation: str = Field(default="", max_length=500)
    evidence: str = Field(
        default="",
        max_length=200,
        description="Preuve minimisée fournie par le LLM, sans contenu brut",
    )
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("reasons")
    @classmethod
    def no_secret_in_reasons(cls, v: list[str]) -> list[str]:
        for reason in v:
            if _SECRET_HINT_RE.search(reason):
                raise ValueError("Secret potentiel interdit dans LlmSecurityDecision")
        return v

    @field_validator("explanation", "evidence")
    @classmethod
    def no_secret_in_text(cls, v: str) -> str:
        if _SECRET_HINT_RE.search(v):
            raise ValueError("Secret potentiel interdit dans LlmSecurityDecision")
        return v
