"""Endpoint ``/v2/chat`` — stub, plan de refonte V2 §4/§6.

Retourne systématiquement ``501 Not Implemented`` tant que `chat/` n'existe
pas : le Chat Reasoning Agent n'est jamais entamé avant le jalon V2-10 (E2E
stable), puis livré en 3 sous-phases progressives (V2-11a expliquer/corriger,
V2-11b simulation, V2-11c rédaction/audit — voir le plan). Le contrat de
requête est déjà figé ici pour ne jamais casser un client une fois le chat
réel branché (V2-11a remplacera uniquement le corps de `chat_v2`, jamais
`ChatMessageRequestV2`).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import Field

from api.v2.dependencies import require_api_key
from schemas.domain import StrictModel

__all__ = ["ChatMessageRequestV2", "build_chat_router"]


class ChatMessageRequestV2(StrictModel):
    """Corps de ``POST /v2/chat`` — texte libre + dossier optionnel.

    ``case_id`` optionnel : certaines intentions du chat (ex. question
    générale sur le fonctionnement) n'ont pas besoin d'un dossier précis —
    voir `chat/planner.py` (non encore livré, Phase V2-11a).
    """

    case_id: str | None = Field(default=None, pattern=r"^CLM-\d{4,}$")
    message: str = Field(..., min_length=1, max_length=4000)


def build_chat_router() -> APIRouter:
    router = APIRouter(tags=["v2-chat"])

    @router.post("/chat", dependencies=[Depends(require_api_key)])
    def chat_v2(payload: ChatMessageRequestV2) -> None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                "Chat Reasoning Agent non encore livré — voir Phase V2-11a "
                "du plan de refonte V2 (jalon V2-10 requis au préalable)."
            ),
        )

    return router
