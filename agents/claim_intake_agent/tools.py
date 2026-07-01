"""Outils @tool du claim_intake_agent — wrappers read-only pour le LLM."""
from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool
from tools.file_inspection import compute_sha256


@tool
def verifier_documents_requis(fichiers_presents: list[str], requis: list[str]) -> dict:
    """Vérifie si tous les documents obligatoires sont présents dans le dossier.

    Args:
        fichiers_presents: Noms de fichiers présents dans le dossier.
        requis: Noms de fichiers obligatoires attendus.

    Returns:
        Dict avec manquants (list[str]) et complet (bool).
    """
    manquants = [r for r in requis if r not in fichiers_presents]
    return {"manquants": manquants, "complet": len(manquants) == 0}


@tool
def obtenir_extensions_autorisees() -> list[str]:
    """Retourne la liste des extensions de fichiers autorisées par la politique.

    Returns:
        Liste d'extensions (ex. [".pdf", ".json"]).
    """
    from config.settings import get_settings
    return list(get_settings().allowed_extensions)


@tool
def verifier_sha256(chemin_relatif: str, sha256_attendu: str) -> dict:
    """Vérifie l'intégrité d'un fichier par comparaison de son SHA-256.

    Args:
        chemin_relatif: Chemin relatif vers le fichier.
        sha256_attendu: Hash SHA-256 attendu (64 caractères hex).

    Returns:
        Dict avec ok (bool) et sha256_calcule (str).
    """
    try:
        actual = compute_sha256(Path(chemin_relatif))
        return {"ok": actual == sha256_attendu, "sha256_calcule": actual}
    except OSError as exc:
        return {"ok": False, "sha256_calcule": "", "erreur": str(exc)}
      