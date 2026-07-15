"""Semantic Understanding du Chat Reasoning Agent — chat/nlu.py (plan V2 §6).

Un seul appel LLM structuré par message — classification uniquement, jamais
une réponse ni une action. Fail-closed : LLM indisponible ou réponse
invalide → `None`, jamais une intention devinée silencieusement (voir
`chat/planner.py`, qui traite `None` comme `CLARIFY_NEEDED`).
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from chat.prompt import load_chat_intent_extraction_prompt
from chat.schemas import LlmIntentDecision, NluResult
from llm.factory import get_llm

__all__ = ["extract_intent"]


def _invoke_llm_intent(message: str, case_id: str | None) -> LlmIntentDecision | None:
    try:
        prompt = load_chat_intent_extraction_prompt()
        llm = get_llm()
        structured = llm.with_structured_output(LlmIntentDecision, method="json_schema")
        data = {
            "message_utilisateur": message,
            "case_id_deja_connu": case_id,
        }
        result = structured.invoke(
            [
                SystemMessage(content=prompt.system_prompt),
                HumanMessage(content=json.dumps(data, ensure_ascii=False)),
            ]
        )
        if isinstance(result, dict):
            result = LlmIntentDecision(**result)
        if isinstance(result, LlmIntentDecision):
            return result
        return None
    except Exception:
        return None


def extract_intent(message: str, case_id: str | None = None) -> NluResult | None:
    """Classe `message` en une ou plusieurs `ChatIntent` — `case_id` fourni
    par l'appelant (contexte de conversation déjà connu, ex. dossier
    actuellement affiché côté UI) est toujours préféré à un identifiant
    que le LLM prétendrait avoir détecté dans le texte : le contexte
    explicite de l'appelant est une source plus fiable qu'une extraction
    de texte libre."""
    result = _invoke_llm_intent(message, case_id)
    if result is None:
        return None
    resolved_case_id = case_id or result.case_id
    return NluResult(
        intents=result.intents,
        case_id=resolved_case_id,
        simulation_changes=result.simulation_changes,
    )
