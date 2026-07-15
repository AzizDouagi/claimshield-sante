"""Dynamic Planner du Chat Reasoning Agent — chat/planner.py (plan V2 §6).

Fonction pure — aucune E/S, aucun appel LLM, aucun état mutable. Traduit un
`NluResult` (ou son absence, LLM indisponible) en `ChatPlan` — jamais
laissé à l'appréciation du LLM lui-même : un intent non livré ou un
`case_id` manquant sont toujours détectés ici, en Python, avant toute
tentative d'exécution (voir `chat/agent.py`).
"""
from __future__ import annotations

from chat.schemas import ChatIntent, ChatPlan, ChatPlanAction, NluResult

__all__ = ["plan"]

_SUPPORTED_INTENTS: frozenset[ChatIntent] = frozenset(
    {
        ChatIntent.ANALYZE,
        ChatIntent.EXPLAIN,
        ChatIntent.CORRECT,
        ChatIntent.SIMULATE,
        ChatIntent.AUDIT,
        ChatIntent.DRAFT_MESSAGE,
    }
)
"""Intentions exécutables — ANALYZE/EXPLAIN/CORRECT (V2-11a) + SIMULATE
(V2-11b) + AUDIT/DRAFT_MESSAGE (V2-11c, nouveau — dernier lot du plan §4,
les 7 intentions du NLU sont maintenant toutes couvertes sauf
CLARIFY_NEEDED, qui n'est jamais « exécutée » par construction)."""

_REQUIRES_CASE_ID: frozenset[ChatIntent] = frozenset(
    {
        ChatIntent.ANALYZE,
        ChatIntent.EXPLAIN,
        ChatIntent.CORRECT,
        ChatIntent.SIMULATE,
        ChatIntent.AUDIT,
        ChatIntent.DRAFT_MESSAGE,
    }
)
"""Les 6 intentions livrées portent toutes sur un dossier précis — aucune
ne peut s'exécuter sans `case_id` résolu."""


def plan(nlu_result: NluResult | None) -> ChatPlan:
    """`nlu_result is None` (NLU indisponible/invalide) → `CLARIFY_NEEDED`,
    jamais une hallucination de dossier ni une intention devinée."""
    if nlu_result is None:
        return ChatPlan(action=ChatPlanAction.CLARIFY_NEEDED, intents=[])

    intents = nlu_result.intents
    if ChatIntent.CLARIFY_NEEDED in intents:
        return ChatPlan(action=ChatPlanAction.CLARIFY_NEEDED, intents=list(intents))

    supported = [i for i in intents if i in _SUPPORTED_INTENTS]
    unsupported = [i for i in intents if i not in _SUPPORTED_INTENTS]

    if not supported:
        return ChatPlan(
            action=ChatPlanAction.NOT_YET_AVAILABLE,
            intents=list(intents),
            unsupported_intents=unsupported,
        )

    needs_case_id = any(intent in _REQUIRES_CASE_ID for intent in supported)
    if needs_case_id and not nlu_result.case_id:
        return ChatPlan(action=ChatPlanAction.CLARIFY_NEEDED, intents=supported)

    return ChatPlan(
        action=ChatPlanAction.EXECUTE,
        intents=supported,
        case_id=nlu_result.case_id,
        simulation_changes=nlu_result.simulation_changes,
        unsupported_intents=unsupported,
    )
