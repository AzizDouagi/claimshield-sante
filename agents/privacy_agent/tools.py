"""Outils @tool du privacy_agent — wrappers read-only sur security/access_policies.py."""
from __future__ import annotations

from langchain_core.tools import tool

from schemas.domain import ReaderRole
from security.access_policies import compute_masked_fields, get_role_policy, verify_view_privacy


@tool
def calculer_champs_masques(role: str) -> dict:
    """Retourne les champs autorisés et refusés pour un rôle donné."""
    try:
        reader_role = ReaderRole(role)
        policy = get_role_policy(reader_role)
        refused = compute_masked_fields(reader_role)
        return {
            "role": reader_role.value,
            "autorises": sorted(policy.allowed_fields),
            "refuses": refused,
        }
    except Exception as exc:
        return {"role": role, "autorises": [], "refuses": [], "erreur": str(exc)}


@tool
def verifier_vie_privee_vue(vue: dict) -> list[dict]:
    """Vérifie qu'aucun identifiant brut n'est exposé dans la vue."""
    return [
        {"field": v.field, "reason_code": v.reason_code, "message": v.message}
        for v in verify_view_privacy(vue)
    ]
