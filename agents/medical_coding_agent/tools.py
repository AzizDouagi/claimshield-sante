"""Outils @tool du medical_coding_agent — wrappers read-only sur tools/medical_coding.py."""
from __future__ import annotations

from langchain_core.tools import tool

from tools.medical_coding import find_fuzzy_candidates, lookup_code


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
          - rule_applied (str) — "exact_match", "fuzzy_candidates_found",
            "keyword_match" ou "not_determined"
          - status (str) — "PASS" ou "NEEDS_REVIEW"
          - fuzzy_candidates (list[dict]) — présent uniquement si
            rule_applied == "fuzzy_candidates_found" (P4-1) : candidats
            bornés au référentiel local (code/label/similarity_score),
            jamais un code inventé. Le LLM ne peut choisir que parmi ces
            candidats ou proposer aucun code — toute autre valeur est
            rejetée côté agent (voir agents/medical_coding_agent/agent.py
            ::_merge_with_llm).
    """
    if section not in ("procedures", "medications"):
        return {
            "original_description": description,
            "proposed_code": None,
            "rule_applied": "invalid_section",
            "status": "NEEDS_REVIEW",
        }
    result = lookup_code(description, section)
    payload = result.model_dump()
    if result.rule_applied == "fuzzy_candidates_found":
        payload["fuzzy_candidates"] = [
            candidate.model_dump()
            for candidate in find_fuzzy_candidates(description, section)
        ]
    return payload
