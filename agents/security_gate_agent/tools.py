"""Outils @tool du security_gate_agent — wrappers read-only sur security/."""
from __future__ import annotations

from langchain_core.tools import tool

from security.policies import DEFAULT_POLICY, validate_file_policy, validate_storage_path, validate_url_policy
from security.scanners import scan_text_security


@tool
def scanner_texte(texte: str) -> dict:
    """Scanne un extrait de texte pour détecter injections de prompt et menaces.

    Args:
        texte: Texte à analyser (max 2000 caractères).

    Returns:
        Dict avec injection_detectee (bool), niveau (str), patterns_trouves (list[str]).
    """
    result = scan_text_security(texte[:2000], DEFAULT_POLICY)
    return {
        "injection_detectee": result.detected,
        "niveau": result.severity,
        "patterns_trouves": result.triggers,
        "categories": [finding.category for finding in result.findings],
    }


@tool
def valider_politique_fichier(extension: str, mime: str, taille_octets: int) -> dict:
    """Valide extension, MIME et taille d'un fichier selon la politique de sécurité.

    Args:
        extension: Extension normalisée (ex. ".pdf").
        mime: Type MIME détecté (ex. "application/pdf").
        taille_octets: Taille réelle en octets.

    Returns:
        Dict avec autorise (bool), raisons (list[str]).
    """
    filename = f"document.{extension.strip().lstrip('.')}" if extension else "document"
    ok, reasons = validate_file_policy(
        filename=filename,
        detected_mime=mime,
        size_bytes=taille_octets,
        policy=DEFAULT_POLICY.file,
    )
    return {"autorise": ok, "raisons": reasons}


@tool
def valider_url(url: str) -> dict:
    """Vérifie si une URL est autorisée selon la politique réseau.

    Args:
        url: URL à vérifier.

    Returns:
        Dict avec autorise (bool), raisons (list[str]).
    """
    ok, reasons = validate_url_policy(url, DEFAULT_POLICY.url)
    return {"autorise": ok, "raisons": reasons}


@tool
def valider_chemin_stockage(chemin_relatif: str) -> dict:
    """Vérifie qu'un chemin relatif est sous storage/ et sans traversée.

    Args:
        chemin_relatif: Chemin relatif à vérifier.

    Returns:
        Dict avec autorise (bool), raisons (list[str]).
    """
    ok, reasons = validate_storage_path(chemin_relatif, DEFAULT_POLICY.path)
    return {"autorise": ok, "raisons": reasons}
