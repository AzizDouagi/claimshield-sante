"""Endpoint ``/v2/chat`` — Chat Reasoning Agent, plan de refonte V2 §4/§6.

Phase V2-11a (livraison 1/3) : intentions `ANALYZE`/`EXPLAIN`/`CORRECT`
exécutables — `SIMULATE`/`AUDIT`/`DRAFT_MESSAGE` détectées par le NLU mais
répondent toujours « bientôt disponible » (voir `chat/planner.py`), jamais
une tentative silencieuse. Aucun accès direct à ``graph.*``/``agents.*`` —
délègue entièrement à `chat.agent.handle_message`, qui lui-même ne parle
qu'à `/v2/claims/*` en HTTP (`chat/tools.py`).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import Field

from api.v2.dependencies import require_api_key
from chat.agent import handle_message
from schemas.domain import StrictModel

__all__ = ["ChatMessageRequestV2", "ChatResponseV2", "build_chat_router"]


class ChatMessageRequestV2(StrictModel):
    """Corps de ``POST /v2/chat`` — texte libre + dossier optionnel.

    ``case_id`` optionnel : certaines intentions du chat (ex. question
    générale sur le fonctionnement) n'ont pas besoin d'un dossier précis.
    """

    case_id: str | None = Field(default=None, pattern=r"^CLM-\d{4,}$")
    message: str = Field(..., min_length=1, max_length=4000)


class ChatResponseV2(StrictModel):
    """Réponse minimisée — texte déjà passé par le garde-fou
    anti-hallucination de `chat/response_composer.py`, jamais de contenu
    brut de document ni de prompt système."""

    reply: str


def build_chat_router() -> APIRouter:
    router = APIRouter(tags=["v2-chat"])

    @router.post("/chat", response_model=ChatResponseV2, dependencies=[Depends(require_api_key)])
    async def chat_v2(payload: ChatMessageRequestV2) -> ChatResponseV2:
        reply = await handle_message(payload.message, payload.case_id)
        return ChatResponseV2(reply=reply)

    return router
