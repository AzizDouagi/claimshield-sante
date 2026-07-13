"""Dépendances FastAPI v2 — authentification.

Aucune réimplémentation : `api.dependencies.require_api_key` reste l'unique
source de vérité pour la vérification de l'en-tête ``X-API-Key`` — ce
module se contente de le réexposer sous l'espace de noms `api.v2` (§0 du
plan de refonte V2 : `api/dependencies.py`, V1, n'est pas touché).
"""
from __future__ import annotations

from api.dependencies import require_api_key

__all__ = ["require_api_key"]
