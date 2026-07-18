"""Endpoint ``/v2/chat`` — Chat Reasoning Agent, plan de refonte V2 §4/§6.

Phase V2-11a (livraison 1/3) : intentions `ANALYZE`/`EXPLAIN`/`CORRECT`
exécutables — `SIMULATE`/`AUDIT`/`DRAFT_MESSAGE` détectées par le NLU mais
répondent toujours « bientôt disponible » (voir `chat/planner.py`), jamais
une tentative silencieuse. Aucun accès direct à ``graph.*``/``agents.*`` —
délègue entièrement à `chat.agent.handle_message`, qui lui-même ne parle
qu'à `/v2/claims/*` en HTTP (`chat/tools.py`).

Mémoire conversationnelle (Phase 8, plan de remédiation « autonomie
décisionnelle V2 ») — **entièrement opt-in** : `thread_id`/`actor` restent
optionnels, la mémoire n'est activée que si `actor` est fourni (le
`thread_id`, s'il est absent, est alors généré côté serveur et renvoyé dans
la réponse pour être réutilisé au message suivant). Sans `actor`, cet
endpoint se comporte à l'identique d'avant cette phase.
"""
from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import Field

from api.v2.dependencies import require_api_key
from chat.agent import handle_message
from chat.conversation_store import ConversationAccessError, ConversationStore
from schemas.domain import StrictModel

__all__ = ["ChatMessageRequestV2", "ChatResponseV2", "build_chat_router"]


class ChatMessageRequestV2(StrictModel):
    """Corps de ``POST /v2/chat`` — texte libre + dossier optionnel.

    ``case_id`` optionnel : certaines intentions du chat (ex. question
    générale sur le fonctionnement) n'ont pas besoin d'un dossier précis.

    ``thread_id``/``actor`` (Phase 8) : activent la mémoire conversationnelle
    uniquement si ``actor`` est fourni — jamais un état partiel silencieux
    (voir `chat.agent.handle_message`, qui exige les trois paramètres
    mémoire ensemble).
    """

    case_id: str | None = Field(default=None, pattern=r"^CLM-\d{4,}$")
    message: str = Field(..., min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, min_length=1, max_length=100)
    actor: str | None = Field(default=None, min_length=1, max_length=255)


class ChatResponseV2(StrictModel):
    """Réponse minimisée — texte déjà passé par le garde-fou
    anti-hallucination de `chat/response_composer.py`, jamais de contenu
    brut de document ni de prompt système.

    ``thread_id`` (Phase 8) : présent uniquement si la mémoire était activée
    pour cette requête (``actor`` fourni) — à réutiliser tel quel par
    l'appelant pour les messages suivants du même fil de conversation."""

    reply: str
    thread_id: str | None = None


def build_chat_router(*, conversation_store: ConversationStore | None = None) -> APIRouter:
    """``conversation_store`` injectable (tests) ; ``None`` construit une
    instance neuve par défaut (jamais un singleton caché, même convention
    que `graph.workflow.compile_workflow`/`api.v2.claims.build_v2_router`)."""
    resolved_store = conversation_store if conversation_store is not None else ConversationStore()
    router = APIRouter(tags=["v2-chat"])

    @router.post("/chat", response_model=ChatResponseV2, dependencies=[Depends(require_api_key)])
    async def chat_v2(payload: ChatMessageRequestV2) -> ChatResponseV2:
        memory_enabled = payload.actor is not None
        thread_id = (payload.thread_id or str(uuid4())) if memory_enabled else None
        try:
            reply = await handle_message(
                payload.message,
                payload.case_id,
                thread_id=thread_id,
                user_id=payload.actor,
                conversation_store=resolved_store if memory_enabled else None,
            )
        except ConversationAccessError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        return ChatResponseV2(reply=reply, thread_id=thread_id)

    return router
