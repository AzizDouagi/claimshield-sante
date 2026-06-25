"""Builders des quatre vues minimisées du Privacy Agent.

Ce module contient uniquement les fonctions pures qui transforment un dict de
données brutes en vue structurée. Les schémas Pydantic sont définis dans
`agents/privacy_agent/schemas.py`.

Fonctions publiques :
  build_administrative_view(case_id, data) → AdministrativeView
  build_medical_view(case_id, data)        → MedicalView
  build_anti_fraud_view(case_id, data)     → FraudView
  build_audit_view(case_id, data)          → AuditView
  build_view(role, case_id, data)          → dict (JSON-sérialisable)
"""
from __future__ import annotations

from agents.privacy_agent.schemas import (
    AdministrativeView,
    AuditView,
    FraudView,
    MedicalView,
)
from schemas.domain import ReaderRole
from tools.pseudonymize import mask_field_value, pseudonymize_patient_id, pseudonymize_provider_id

# Re-exports pour la compatibilité avec le code existant
AntiFraudView = FraudView

__all__ = [
    # Schémas (re-exports depuis schemas.py)
    "AdministrativeView",
    "MedicalView",
    "FraudView",
    "AntiFraudView",  # backward compat
    "AuditView",
    # Builders
    "build_administrative_view",
    "build_medical_view",
    "build_anti_fraud_view",
    "build_audit_view",
    "build_view",
]


# ── Helpers de construction ───────────────────────────────────────────────────


def _str_or_none(value: object) -> str | None:
    """Convertit en str ou retourne None si vide/absent."""
    if value is None:
        return None
    s = str(value)
    return s or None


def _patient_pseudonym_from(data: dict) -> str:
    """Retourne le pseudonyme stable du patient (PAT-…) ou 'PAT-INCONNU' si absent."""
    pid = data.get("patient_id")
    return pseudonymize_patient_id(str(pid)) if pid else "PAT-INCONNU"


# ── Builders ──────────────────────────────────────────────────────────────────


def build_administrative_view(case_id: str, data: dict) -> AdministrativeView:
    """Construit la vue administrative : pseudonymise le patient, masque la facture."""
    patient_pseudonym = _patient_pseudonym_from(data)
    invoice_raw = data.get("invoice_number")
    invoice_ref = mask_field_value("invoice_number", str(invoice_raw)) if invoice_raw else None
    return AdministrativeView(
        claim_id=case_id,
        dossier_status=str(data.get("dossier_status", "UNKNOWN")),
        present_documents=[str(d) for d in data.get("present_documents", [])],
        missing_documents=[str(d) for d in data.get("missing_documents", [])],
        submitted_at=_str_or_none(data.get("submitted_at")),
        service_date=_str_or_none(data.get("service_date")),
        total_billed=_str_or_none(data.get("total_billed")),
        amount_requested=_str_or_none(data.get("amount_requested")),
        patient_share=_str_or_none(data.get("patient_share")),
        coverage_rate=_str_or_none(data.get("coverage_rate")),
        payer_name=_str_or_none(data.get("payer_name")),
        invoice_reference=invoice_ref,
        patient_pseudonym=patient_pseudonym,
    )


def build_medical_view(case_id: str, data: dict) -> MedicalView:  # noqa: ARG001
    """Construit la vue médicale : pseudonymise patient et prestataire."""
    patient_pseudonym = _patient_pseudonym_from(data)
    provider_id = data.get("provider_id")
    provider_pseudonym = pseudonymize_provider_id(str(provider_id)) if provider_id else None
    return MedicalView(
        patient_pseudonym=patient_pseudonym,
        service_date=_str_or_none(data.get("service_date")),
        procedures=[str(p) for p in data.get("procedures", [])],
        prescription_names=[str(p) for p in data.get("prescription_names", [])],
        diagnosis_codes=[str(c) for c in data.get("diagnosis_codes", [])],
        encounter_class=_str_or_none(data.get("encounter_class")),
        provider_pseudonym=provider_pseudonym,
    )


def build_anti_fraud_view(case_id: str, data: dict) -> FraudView:  # noqa: ARG001
    """Construit la vue antifraude : pseudonymise patient et prestataire, masque la facture."""
    patient_pseudonym = _patient_pseudonym_from(data)
    invoice_raw = data.get("invoice_number")
    invoice_ref = mask_field_value("invoice_number", str(invoice_raw)) if invoice_raw else None
    provider_id = data.get("provider_id")
    provider_ref = pseudonymize_provider_id(str(provider_id)) if provider_id else None
    return FraudView(
        patient_pseudonym=patient_pseudonym,
        document_hashes=dict(data.get("document_hashes", {})),
        total_billed=_str_or_none(data.get("total_billed")),
        amount_requested=_str_or_none(data.get("amount_requested")),
        patient_share=_str_or_none(data.get("patient_share")),
        service_date=_str_or_none(data.get("service_date")),
        submitted_at=_str_or_none(data.get("submitted_at")),
        invoice_reference=invoice_ref,
        provider_reference=provider_ref,
    )


def build_audit_view(case_id: str, data: dict) -> AuditView:
    """Construit la vue audit : aucune donnée personnelle, aucun secret."""
    return AuditView(
        claim_id=case_id,
        actor=str(data.get("actor", "unknown")),
        actor_role=str(data.get("actor_role", "")),
        action=str(data.get("action", "")),
        timestamp=str(data.get("timestamp", "")),
        policy_version=str(data.get("policy_version", "1.0.0")),
        outcome=str(data.get("outcome", "")),
        reason_codes=[str(c) for c in data.get("reason_codes", [])],
    )


def build_view(role: ReaderRole, case_id: str, data: dict) -> dict:
    """Construit la vue minimisée adaptée au rôle et retourne un dict JSON-sérialisable.

    Args:
        role    : détermine le builder appelé.
        case_id : identifiant du dossier.
        data    : données brutes du dossier (sans secrets ni chemins absolus).

    Returns:
        Vue minimisée sous forme de dict JSON-sérialisable.
    """
    builders = {
        ReaderRole.ADMINISTRATIVE_MANAGER: build_administrative_view,
        ReaderRole.MEDICAL_REVIEWER: build_medical_view,
        ReaderRole.FRAUD_ANALYST: build_anti_fraud_view,
        ReaderRole.AUDITOR: build_audit_view,
    }
    return builders[role](case_id, data).model_dump()
