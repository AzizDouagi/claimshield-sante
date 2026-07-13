"""Routeur API v2 — pipeline autonome ClaimShield Santé (plan de refonte V2, Phase V2-9).

Point d'intégration unique avec V1 (§0 du plan) : `api/main.py` importe
`v2_router` et l'inclut via une seule ligne
``app.include_router(v2_router, prefix="/v2")`` — aucune autre modification
de `api/main.py`. Compile le graphe V2 une seule fois, à l'import de ce
module (même convention que ``api/main.py::app = create_app()`` pour V1).
"""
from __future__ import annotations

from fastapi import APIRouter

from api.v2.chat import build_chat_router
from api.v2.claims import build_v2_router

v2_router = APIRouter()
v2_router.include_router(build_v2_router())
v2_router.include_router(build_chat_router())

__all__ = ["v2_router"]
