"""Chat Reasoning Agent — point d'entrée unique, chat/agent.py (plan V2 §6).

`handle_message` délègue systématiquement : compréhension (`chat/nlu.py`)
→ planification (`chat/planner.py`) → outils (`chat/tools.py`) →
composition (`chat/response_composer.py`) — ne répond jamais directement,
ne contient aucune logique métier propre.
"""
from __future__ import annotations

from chat.nlu import extract_intent
from chat.planner import plan
from chat.response_composer import compose
from chat.schemas import ChatIntent, ChatPlanAction
from chat.tools import (
    explain_claim,
    generate_patient_message,
    get_audit_summary,
    get_claim_context,
    recommend_corrections,
    simulate_changes,
)

__all__ = ["handle_message"]

_CLARIFY_NEEDED_MESSAGE = (
    "Pour vous répondre, j'ai besoin de l'identifiant du dossier concerné "
    "(ex. CLM-0001) et d'une question plus précise."
)
_NOT_YET_AVAILABLE_MESSAGE = (
    "Cette fonctionnalité n'est pas encore disponible dans cette version du chat."
)
_CASE_NOT_FOUND_MESSAGE = "Dossier introuvable — jamais soumis ou thread expiré."
_SIMULATE_MISSING_CHANGES_MESSAGE = (
    "Merci de préciser le changement à simuler : retirer un document déjà "
    "présent (ex. « et si on retirait l'ordonnance ? ») ou changer le rôle "
    "de lecture (ex. « simule avec le rôle FRAUD_ANALYST »). Un nouveau "
    "document ne peut pas être ajouté via le chat."
)


async def handle_message(message: str, case_id: str | None = None) -> str:
    """Traite un message libre — `case_id` optionnel (contexte déjà connu
    de l'appelant, ex. dossier affiché côté UI) prime toujours sur un
    identifiant que le NLU prétendrait avoir détecté dans le texte."""
    nlu_result = extract_intent(message, case_id)
    chat_plan = plan(nlu_result)

    if chat_plan.action is ChatPlanAction.CLARIFY_NEEDED:
        return _CLARIFY_NEEDED_MESSAGE
    if chat_plan.action is ChatPlanAction.NOT_YET_AVAILABLE:
        return _NOT_YET_AVAILABLE_MESSAGE

    resolved_case_id = chat_plan.case_id
    if resolved_case_id is None:
        # Défensif — `chat.planner.plan` garantit déjà un case_id non None
        # pour ACTION.EXECUTE, jamais atteint en pratique.
        return _CLARIFY_NEEDED_MESSAGE

    if ChatIntent.SIMULATE in chat_plan.intents and chat_plan.simulation_changes is None:
        return _SIMULATE_MISSING_CHANGES_MESSAGE

    tool_results: dict = {}
    if ChatIntent.ANALYZE in chat_plan.intents:
        tool_results["context"] = await get_claim_context(resolved_case_id)
    if ChatIntent.EXPLAIN in chat_plan.intents:
        tool_results["explanation"] = await explain_claim(resolved_case_id)
    if ChatIntent.CORRECT in chat_plan.intents:
        tool_results["corrections"] = await recommend_corrections(resolved_case_id)
    if ChatIntent.SIMULATE in chat_plan.intents and chat_plan.simulation_changes is not None:
        tool_results["simulation"] = await simulate_changes(
            resolved_case_id, chat_plan.simulation_changes
        )
    if ChatIntent.AUDIT in chat_plan.intents:
        tool_results["audit_summary"] = await get_audit_summary(resolved_case_id)
    if ChatIntent.DRAFT_MESSAGE in chat_plan.intents:
        tool_results["patient_message_context"] = await generate_patient_message(resolved_case_id)

    if not tool_results or all(value in (None, [], {}) for value in tool_results.values()):
        return _CASE_NOT_FOUND_MESSAGE

    return compose(case_id=resolved_case_id, intents=chat_plan.intents, tool_results=tool_results)
