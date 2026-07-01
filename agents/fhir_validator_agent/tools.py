"""Outils @tool du fhir_validator_agent — wrappers read-only sur tools/fhir_validation.py."""
from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool
from tools.fhir_validation import extract_resource_types, load_fhir_bundle, validate_fhir_bundle
from tools.file_inspection import compute_sha256
from tools.rule_loader import load_rules


@tool
def valider_bundle_fhir(chemin_relatif: str, bundle_attendu: bool = True) -> dict:
    """Valide la structure d'un bundle FHIR R4.

    Args:
        chemin_relatif: chemin relatif vers le fichier JSON du bundle.
        bundle_attendu: True si le bundle est obligatoire pour ce dossier.

    Returns:
        Dict avec status, errors (list), warnings (list), profile_checked.
    """
    rules = load_rules("fhir_rules.yaml")
    status, errors, warnings, profile = validate_fhir_bundle(
        chemin_relatif,
        bundle_expected=bundle_attendu,
        rules=rules,
    )
    return {
        "status": status.value,
        "errors": list(errors),
        "warnings": list(warnings),
        "profile_checked": profile or "",
    }


@tool
def extraire_types_ressources(chemin_relatif: str) -> list[str]:
    """Retourne les types de ressources FHIR présents dans le bundle.

    Args:
        chemin_relatif: chemin relatif vers le fichier JSON du bundle.

    Returns:
        Liste de chaînes de types (ex. ["Patient", "Coverage", "Claim"]).
    """
    bundle, errors = load_fhir_bundle(chemin_relatif)
    if bundle is None or errors:
        return []
    return extract_resource_types(bundle)


@tool
def calculer_sha256_fhir(chemin_relatif: str) -> str:
    """Calcule le SHA-256 du bundle FHIR.

    Args:
        chemin_relatif: chemin relatif vers le fichier JSON.

    Returns:
        Empreinte SHA-256 hexadécimale (64 caractères) ou chaîne vide en cas d'erreur.
    """
    try:
        return compute_sha256(Path(chemin_relatif))
    except OSError:
        return ""
