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

``POST /chat/stream`` (visibilité temps réel des étapes/tokens, demandée par
AZIZ — « comme Claude Code ») : endpoint **additif**, ``POST /chat``
ci-dessus reste strictement inchangé. Flux Server-Sent-Events natif FastAPI
(``StreamingResponse``, ``text/event-stream``) — aucune dépendance
supplémentaire (``sse-starlette`` n'est installé que transitivement via
``mcp``, jamais déclaré, volontairement pas utilisé ici). Chaque ligne
``data: {...}`` porte une enveloppe ``{"type": "step"|"final"|"error", ...}``."""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import Field

from api.v2.dependencies import require_api_key
from chat.agent import handle_message, handle_message_streaming
from chat.conversation_store import ConversationAccessError, ConversationStore
from chat.schemas import ChatStepEvent, ChatTurnSummary
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

    @router.post("/chat/stream", dependencies=[Depends(require_api_key)])
    async def chat_v2_stream(payload: ChatMessageRequestV2) -> StreamingResponse:
        memory_enabled = payload.actor is not None
        thread_id = (payload.thread_id or str(uuid4())) if memory_enabled else None

        async def event_stream() -> AsyncIterator[str]:
            queue: asyncio.Queue[ChatStepEvent] = asyncio.Queue()

            async def on_step(event: ChatStepEvent) -> None:
                await queue.put(event)

            task = asyncio.create_task(
                handle_message_streaming(
                    payload.message,
                    payload.case_id,
                    thread_id=thread_id,
                    user_id=payload.actor,
                    conversation_store=resolved_store if memory_enabled else None,
                    on_step=on_step,
                )
            )

            total_input = 0
            total_output = 0
            total_duration = 0
            step_count = 0
            while not task.done() or not queue.empty():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                step_count += 1
                total_input += event.input_tokens or 0
                total_output += event.output_tokens or 0
                total_duration += event.duration_ms or 0
                yield f"data: {json.dumps({'type': 'step', **event.model_dump(mode='json')}, ensure_ascii=False)}\n\n"

            try:
                reply = await task
            except ConversationAccessError as exc:
                yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)}, ensure_ascii=False)}\n\n"
                return

            summary = ChatTurnSummary(
                reply=reply,
                thread_id=thread_id,
                total_input_tokens=total_input,
                total_output_tokens=total_output,
                total_duration_ms=total_duration,
                step_count=step_count,
            )
            yield f"data: {json.dumps({'type': 'final', **summary.model_dump(mode='json')}, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return router
