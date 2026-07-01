"""Outils @tool du medical_coding_agent — wrappers read-only sur tools/medical_coding.py."""
from __future__ import annotations

from langchain_core.tools import tool

from tools.medical_coding import lookup_code


@tool
def rechercher_code(description: str, section: str) -> dict:
    """Recherche le code SNOMED-CT ou RxNorm pour une description médicale.

    Args:
        description: Description brute de l'acte ou du médicament.
        section: "procedures" pour les actes médicaux, "medications" pour les médicaments.

    Returns:
        Dict avec les clés :
          - original_description (str)
          - proposed_code (str | None) — None si non trouvé
          - rule_applied (str) — "exact_match", "keyword_match" ou "not_found"
          - status (str) — "PASS" ou "NEEDS_REVIEW"
    """
    if section not in ("procedures", "medications"):
        return {
            "original_description": description,
            "proposed_code": None,
            "rule_applied": "invalid_section",
            "status": "NEEDS_REVIEW",
        }
    result = lookup_code(description, section)
    return result.model_dump()
