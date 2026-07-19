"""API HTTP ClaimShield Santé — point d'entrée FastAPI, V2 uniquement.

Ce module ne fait plus que deux choses : exposer ``GET /healthz`` (liveness,
sans authentification) et monter le routeur V2 complet (``api/v2``, soumission
de dossier, statut, override, chat) sous le préfixe ``/v2``. Le pipeline V1
(LangGraph orchestré, agents ``claim_intake``/``security_gate``/
``case_reviewer``/``audit``, HITL classique) a été retiré du dépôt — voir
``CLAUDE.md`` pour l'historique du nettoyage V1→V2.

Lancer en développement ::

    uvicorn api.main:app --reload --host $API_HOST --port $API_PORT
"""
from __future__ import annotations

from fastapi import FastAPI

from api.v2 import v2_router
from config.logging import configure_logging
from schemas.domain import StrictModel


class HealthResponse(StrictModel):
    status: str = "ok"


def create_app() -> FastAPI:
    """Construit l'application FastAPI : routeur V2 + liveness uniquement."""
    configure_logging()

    app = FastAPI(
        title="ClaimShield Santé API",
        description="API exposant le pipeline autonome V2 de traitement des réclamations.",
        version="2.0.0",
    )

    app.include_router(v2_router, prefix="/v2")

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse()

    return app


app = create_app()

__all__ = ["app", "create_app"]
