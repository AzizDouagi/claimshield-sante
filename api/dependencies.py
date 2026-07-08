"""Dépendances FastAPI — authentification minimale.

Contrôle d'accès volontairement simple (clé statique en en-tête) : suffisant
pour un usage démo/interne, pas un système d'authentification complet
(OAuth, sessions, rôles) — hors périmètre d'une exposition minimale du
graphe (voir ``api/main.py``). À renforcer avant tout déploiement au-delà
d'un usage local/démo (voir ``Settings.claimshield_api_key``).
"""
from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from config.settings import get_settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Vérifie l'en-tête ``X-API-Key`` contre ``Settings.claimshield_api_key``.

    Appliquée uniquement aux endpoints qui déclenchent une action métier
    (soumission de dossier, décision humaine) — jamais à ``GET /healthz``.
    Comparaison en temps constant (``secrets.compare_digest``) pour ne pas
    faciliter une attaque temporelle sur la clé.
    """
    expected = get_settings().claimshield_api_key.get_secret_value()
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API manquante ou invalide (en-tête X-API-Key requis).",
        )


__all__ = ["require_api_key"]
