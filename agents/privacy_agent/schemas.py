"""Schémas du domaine Privacy Agent.

Hub de tous les types Pydantic utilisés par le Privacy Agent :

  PrivacyRole      — rôles de lecture (alias de ReaderRole, 4 rôles stables)
  PrivacyDecision  — décision du Privacy Agent (alias de VerificationStatus)
  AdministrativeView, MedicalView, FraudView, AuditView — vues minimisées
  PrivacyRequest   — entrée structurée du Privacy Agent (alias : PrivacyInput)
  PrivacyResult    — résultat du Privacy Agent (re-export depuis schemas.results)

Invariants communs à toutes les vues :
  - extra='forbid' via StrictModel — aucun champ inconnu accepté.
  - Pas de nom complet, adresse, téléphone ni e-mail.
  - Pas de chemin absolu, secret, token ou texte OCR brut.
  - Pseudonymes PAT-… obligatoires sur identifiants patients.
  - Pseudonymes PRV-… obligatoires sur identifiants prestataires.
  - document_hashes : valeurs SHA-256 64 hex chars validées.
"""
from __future__ import annotations

import re

from pydantic import Field, field_validator

from schemas.domain import (
    DataClassification,
    PrivacyCode,
    PrivacyDecision,
    ReaderRole,
    StrictModel,
)
from schemas.results import PrivacyResult  # re-export

# ── Aliases sémantiques du domaine ───────────────────────────────────────────

#: Rôle du lecteur dans le contexte du Privacy Agent.
PrivacyRole = ReaderRole

# ── Validators partagés entre les vues ───────────────────────────────────────

_ABSOLUTE_PATH_RE = re.compile(r"^(?:/|[A-Za-z]:[/\\]|\\\\)")
_SECRET_HINT_RE = re.compile(
    r"(?:api[_-]?key|secret\s*[:=]|password\s*[:=]|token\s*[:=]|bearer\s+\S+)",
    re.IGNORECASE,
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _reject_view_leak(v: str | None, field_name: str) -> str | None:
    """Refuse chemins absolus et marqueurs évidents de secrets dans les vues."""
    if v is None:
        return v
    if _ABSOLUTE_PATH_RE.match(v):
        raise ValueError(f"Chemin absolu interdit dans {field_name}")
    if _SECRET_HINT_RE.search(v):
        raise ValueError(f"Secret potentiel interdit dans {field_name}")
    return v


# ── AdministrativeView (ADMINISTRATIVE_MANAGER) ───────────────────────────────


class AdministrativeView(StrictModel):
    """Vue minimisée pour le gestionnaire administratif.

    Expose :  claim_id, statut dossier, documents présents/manquants, dates
              administratives, montants, référence facture masquée, pseudonyme
              patient.
    Masque :  nom complet, adresse, téléphone, e-mail (pas de champ prévu).
    Exclure : diagnostics et détails médicaux (pas de champ prévu).
    """

    claim_id: str
    dossier_status: str = Field(..., description="Statut courant du dossier")
    present_documents: list[str] = Field(
        default_factory=list,
        description="Documents reçus et validés",
    )
    missing_documents: list[str] = Field(
        default_factory=list,
        description="Documents attendus mais absents",
    )
    submitted_at: str | None = Field(default=None, description="Date de soumission (ISO 8601)")
    service_date: str | None = Field(default=None, description="Date de soins (ISO 8601)")
    total_billed: str | None = Field(default=None, description="Montant total facturé")
    amount_requested: str | None = Field(default=None, description="Montant demandé")
    patient_share: str | None = Field(default=None, description="Part patient")
    coverage_rate: str | None = Field(default=None, description="Taux de couverture")
    payer_name: str | None = Field(default=None, description="Nom de l'assureur")
    invoice_reference: str | None = Field(
        default=None,
        description="Référence facture masquée (***XXXX) — jamais le numéro brut",
    )
    patient_pseudonym: str = Field(
        ...,
        description="Identifiant patient pseudonymisé (PAT-…) — jamais le nom réel",
    )

    @field_validator("patient_pseudonym")
    @classmethod
    def pseudonym_must_have_pat_prefix(cls, v: str) -> str:
        if not v.startswith("PAT-"):
            raise ValueError("patient_pseudonym doit commencer par 'PAT-'")
        return v

    @field_validator(
        "claim_id", "dossier_status", "submitted_at", "service_date",
        "total_billed", "amount_requested", "patient_share", "coverage_rate",
        "payer_name", "invoice_reference",
    )
    @classmethod
    def no_sensitive_value(cls, v: str | None, info) -> str | None:
        return _reject_view_leak(v, info.field_name)


# ── MedicalView (MEDICAL_REVIEWER) ───────────────────────────────────────────


class MedicalView(StrictModel):
    """Vue minimisée pour le médecin conseil ou réviseur médical.

    Expose :  pseudonyme patient, dates de soins, actes, prescriptions, codes
              diagnostics, classe de consultation, pseudonyme prestataire.
    Masque :  coordonnées personnelles (pas de champ prévu).
    Exclure : informations bancaires, chemins de stockage, secrets (pas de champ prévu).
    """

    patient_pseudonym: str = Field(
        ...,
        description="Pseudonyme patient (PAT-…) — jamais le nom ni l'identifiant réel",
    )
    service_date: str | None = Field(default=None, description="Date de soins (ISO 8601)")
    procedures: list[str] = Field(
        default_factory=list,
        description="Descriptions des actes médicaux réalisés",
    )
    prescription_names: list[str] = Field(
        default_factory=list,
        description="Noms des médicaments prescrits",
    )
    diagnosis_codes: list[str] = Field(
        default_factory=list,
        description="Codes diagnostics (CIM-10, SNOMED…)",
    )
    encounter_class: str | None = Field(
        default=None,
        description="Type de consultation (ambulatory, inpatient…)",
    )
    provider_pseudonym: str | None = Field(
        default=None,
        description="Identifiant prestataire pseudonymisé (PRV-…)",
    )

    @field_validator("patient_pseudonym")
    @classmethod
    def patient_pseudonym_format(cls, v: str) -> str:
        if not v.startswith("PAT-"):
            raise ValueError("patient_pseudonym doit commencer par 'PAT-'")
        return v

    @field_validator("provider_pseudonym")
    @classmethod
    def provider_pseudonym_format(cls, v: str | None) -> str | None:
        if v is not None and not v.startswith("PRV-"):
            raise ValueError("provider_pseudonym doit commencer par 'PRV-'")
        return v

    @field_validator("service_date", "encounter_class")
    @classmethod
    def no_sensitive_value(cls, v: str | None, info) -> str | None:
        return _reject_view_leak(v, info.field_name)


# ── FraudView (FRAUD_ANALYST) ────────────────────────────────────────────────


class FraudView(StrictModel):
    """Vue minimisée pour l'analyste antifraude.

    Expose :  pseudonyme stable patient (comparaison historique), hashes
              documents, montants, dates, référence facture masquée, référence
              prestataire pseudonymisée.
    Masque :  noms, adresses, coordonnées personnelles (pas de champ prévu).
    Exclure : détails médicaux non nécessaires à la détection (pas de champ prévu).
    """

    patient_pseudonym: str = Field(
        ...,
        description="Pseudonyme stable (PAT-…) — identique entre appels pour le même patient",
    )
    document_hashes: dict[str, str] = Field(
        default_factory=dict,
        description="SHA-256 par type de document — permet la comparaison historique",
    )
    total_billed: str | None = Field(default=None, description="Montant total facturé")
    amount_requested: str | None = Field(default=None, description="Montant demandé")
    patient_share: str | None = Field(default=None, description="Part patient")
    service_date: str | None = Field(default=None, description="Date de soins (ISO 8601)")
    submitted_at: str | None = Field(default=None, description="Date de soumission (ISO 8601)")
    invoice_reference: str | None = Field(
        default=None,
        description="Référence facture masquée (***XXXX)",
    )
    provider_reference: str | None = Field(
        default=None,
        description="Référence prestataire pseudonymisée (PRV-…)",
    )

    @field_validator("patient_pseudonym")
    @classmethod
    def pseudonym_format(cls, v: str) -> str:
        if not v.startswith("PAT-"):
            raise ValueError("patient_pseudonym doit commencer par 'PAT-'")
        return v

    @field_validator("provider_reference")
    @classmethod
    def provider_format(cls, v: str | None) -> str | None:
        if v is not None and not v.startswith("PRV-"):
            raise ValueError("provider_reference doit commencer par 'PRV-'")
        return v

    @field_validator("document_hashes")
    @classmethod
    def validate_hashes(cls, v: dict[str, str]) -> dict[str, str]:
        for key, value in v.items():
            if value and not _SHA256_RE.fullmatch(value):
                raise ValueError(
                    f"Hash invalide pour '{key}' — 64 caractères hexadécimaux attendus, "
                    f"reçu : '{value[:16]}…'"
                )
        return v

    @field_validator(
        "total_billed", "amount_requested", "patient_share",
        "service_date", "submitted_at", "invoice_reference",
    )
    @classmethod
    def no_sensitive_value(cls, v: str | None, info) -> str | None:
        return _reject_view_leak(v, info.field_name)


#: Alias de FraudView pour la compatibilité avec le code antérieur.
AntiFraudView = FraudView


# ── AuditView (AUDITOR) ───────────────────────────────────────────────────────


class AuditView(StrictModel):
    """Vue minimisée pour l'auditeur interne.

    Expose :  claim_id, acteur + rôle, action, horodatage, version politique,
              résultat, codes structurés.
    Exclure : texte OCR brut, noms, diagnostics, ordonnances, coordonnées,
              prompts, tokens, clés et secrets (pas de champ prévu, validators actifs).
    """

    claim_id: str
    actor: str = Field(
        ...,
        description="Identifiant de l'acteur (agent ou pseudonyme utilisateur)",
    )
    actor_role: str = Field(..., description="Rôle de l'acteur (valeur de PrivacyRole)")
    action: str = Field(..., description="Action réalisée (ex. 'security_evaluation')")
    timestamp: str = Field(..., description="Horodatage ISO 8601")
    policy_version: str = Field(..., description="Version de la politique appliquée")
    outcome: str = Field(..., description="Résultat de l'action (ALLOW, BLOCK, PASS…)")
    reason_codes: list[str] = Field(
        default_factory=list,
        description="Codes d'erreur ou motifs structurés",
    )

    @field_validator(
        "claim_id", "actor", "actor_role", "action",
        "timestamp", "policy_version", "outcome",
    )
    @classmethod
    def no_sensitive_value(cls, v: str | None, info) -> str | None:
        return _reject_view_leak(v, info.field_name)

    @field_validator("reason_codes")
    @classmethod
    def no_sensitive_codes(cls, v: list[str]) -> list[str]:
        for code in v:
            _reject_view_leak(code, "reason_codes")
        return v


# ── PrivacyRequest (entrée du Privacy Agent) ─────────────────────────────────


class PrivacyRequest(StrictModel):
    """Données nécessaires à l'évaluation de la confidentialité d'un dossier.

    Le rôle est OBLIGATOIRE — aucune valeur par défaut n'est fournie.
    Un rôle absent ou inconnu est rejeté (politique DENY-by-default).
    Le rôle détermine les champs visibles via l'allowlist de RoleAccessPolicy.
    claim_data est optionnel — s'il est fourni, la vue minimisée est construite.
    """

    case_id: str = Field(
        ...,
        pattern=r"^CLM-\d{4,}$",
        description="Identifiant du dossier (ex. CLM-0001)",
    )
    role: PrivacyRole = Field(
        ...,
        description=(
            "Rôle du lecteur — OBLIGATOIRE. "
            "Tout rôle absent ou inconnu déclenche un blocage (DENY-by-default)."
        ),
    )
    data_classification: DataClassification = Field(
        default=DataClassification.SYNTHETIC_TEST_DATA,
        description="Classification de confidentialité du dossier",
    )
    contains_real_personal_data: bool = Field(
        default=False,
        description="True si le dossier contient des données personnelles réelles",
    )
    fields_to_evaluate: list[str] = Field(
        default_factory=list,
        description=(
            "Champs à évaluer — si vide, l'intégralité de ALL_KNOWN_FIELDS est évaluée. "
            "Tout champ absent de l'allowlist du rôle est refusé, qu'il soit connu ou non."
        ),
    )

    # Valeurs optionnelles des champs personnels — pour pseudonymisation
    patient_name: str | None = Field(
        default=None, max_length=200,
        description="Nom du patient à pseudonymiser (optionnel)",
    )
    patient_id: str | None = Field(
        default=None, max_length=100,
        description="Identifiant patient à pseudonymiser (optionnel)",
    )
    payer_name: str | None = Field(
        default=None, max_length=200,
        description="Nom de l'assureur (optionnel)",
    )
    invoice_number: str | None = Field(
        default=None, max_length=100,
        description="Numéro de facture (optionnel)",
    )
    prescription_number: str | None = Field(
        default=None, max_length=100,
        description="Numéro d'ordonnance (optionnel)",
    )
    claim_data: dict | None = Field(
        default=None,
        description=(
            "Données du dossier pour construire la vue minimisée. "
            "Ne doit pas contenir de chemin absolu, secret, token ou texte OCR brut. "
            "Clés reconnues : patient_id, dossier_status, present_documents, "
            "missing_documents, submitted_at, service_date, total_billed, "
            "amount_requested, patient_share, coverage_rate, payer_name, "
            "invoice_number, procedures, prescription_names, diagnosis_codes, "
            "encounter_class, provider_id, document_hashes, actor, actor_role, "
            "action, timestamp, policy_version, outcome, reason_codes."
        ),
    )

    @field_validator(
        "patient_name", "patient_id", "payer_name",
        "invoice_number", "prescription_number",
    )
    @classmethod
    def no_sensitive_value(cls, v: str | None, info) -> str | None:
        if v is None:
            return v
        if _ABSOLUTE_PATH_RE.match(v):
            raise ValueError(f"Chemin absolu interdit dans {info.field_name}")
        if _SECRET_HINT_RE.search(v):
            raise ValueError(f"Secret potentiel interdit dans {info.field_name}")
        return v

    @field_validator("claim_data")
    @classmethod
    def validate_claim_data(cls, v: dict | None) -> dict | None:
        if v is None:
            return v
        for key, value in v.items():
            if isinstance(value, str):
                if _ABSOLUTE_PATH_RE.match(value):
                    raise ValueError(f"Chemin absolu interdit dans claim_data['{key}']")
                if _SECRET_HINT_RE.search(value):
                    raise ValueError(f"Secret potentiel interdit dans claim_data['{key}']")
        return v


#: Alias de PrivacyRequest pour la compatibilité avec le code antérieur.
PrivacyInput = PrivacyRequest


# ── Schéma de décision LLM (intermédiaire — jamais dans ClaimState) ───────────


class LlmPrivacyDecision(StrictModel):
    """Enrichissement LLM pour l'audit privacy."""

    audit_justification: str = Field(default="", max_length=500)
    data_classification_reason: str = Field(default="", max_length=500)

    @field_validator("audit_justification", "data_classification_reason")
    @classmethod
    def no_sensitive_value(cls, v: str, info) -> str:
        return _reject_view_leak(v, info.field_name) or ""


__all__ = [
    # Aliases sémantiques
    "PrivacyRole",
    "PrivacyDecision",
    # Codes stables
    "PrivacyCode",
    # Vues minimisées
    "AdministrativeView",
    "MedicalView",
    "FraudView",
    "AntiFraudView",  # backward compat — alias de FraudView
    "AuditView",
    # Entrée
    "PrivacyRequest",
    "PrivacyInput",  # backward compat — alias de PrivacyRequest
    "LlmPrivacyDecision",
    # Résultat (re-export depuis schemas.results)
    "PrivacyResult",
    # Re-exports pratiques
    "ReaderRole",
]
