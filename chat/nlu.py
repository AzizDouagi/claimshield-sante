"""Semantic Understanding du Chat Reasoning Agent — chat/nlu.py (plan V2 §6).

Un seul appel LLM structuré par message — classification uniquement, jamais
une réponse ni une action. Fail-closed : LLM indisponible ou réponse
invalide → `None`, jamais une intention devinée silencieusement (voir
`chat/planner.py`, qui traite `None` comme `CLARIFY_NEEDED`).

`recent_turns`/`semantic_state` (Phase 8, mémoire conversationnelle, §6.3
du plan) : la résolution d'une référence à un scénario déjà discuté ("le
premier scénario", "cette hypothèse") est **déterministe, jamais confiée au
LLM** — une simple recherche de sous-chaîne (insensible à la casse) dans
`semantic_state.resolved_references`, calculée en Python avant même l'appel
LLM. `recent_turns` sert uniquement de contexte informatif transmis au LLM
d'extraction (continuité de la conversation), jamais une source de vérité
pour `case_id`/`resolved_scenario_id` (toujours calculés déterministiquement
par l'appelant/ce module, jamais par le LLM)."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from chat.llm_usage import record_usage
from chat.prompt import load_chat_intent_extraction_prompt
from chat.schemas import LlmIntentDecision, NluResult
from llm.factory import get_llm

if TYPE_CHECKING:
    from chat.memory_schemas import ConversationSemanticState, ConversationTurn

__all__ = ["extract_intent"]


def _invoke_llm_intent(
    message: str,
    case_id: str | None,
    *,
    recent_turns: list[ConversationTurn] | None = None,
    usage_sink: dict | None = None,
) -> LlmIntentDecision | None:
    """`usage_sink` (optionnel, `None` par défaut — aucun changement de
    comportement pour les appelants existants) : dict mutable rempli avec
    `model_name`/`input_tokens`/`output_tokens` (via `include_raw=True`,
    seul moyen d'obtenir l'`AIMessage` brut — donc `usage_metadata` — tout
    en gardant la sortie structurée déjà parsée) si l'appel réussit — voir
    `chat/agent.py` (visibilité temps réel des tokens, AZIZ)."""
    try:
        prompt = load_chat_intent_extraction_prompt()
        llm = get_llm()
        structured = llm.with_structured_output(LlmIntentDecision, method="json_schema", include_raw=True)
        data = {
            "message_utilisateur": message,
            "case_id_deja_connu": case_id,
            "tours_recents": (
                [
                    {"intents": [i.value for i in turn.intents], "case_id": turn.case_id}
                    for turn in recent_turns
                ]
                if recent_turns
                else []
            ),
        }
        raw_result = structured.invoke(
            [
                SystemMessage(content=prompt.system_prompt),
                HumanMessage(content=json.dumps(data, ensure_ascii=False)),
            ]
        )
        if raw_result.get("parsing_error") is not None:
            return None
        record_usage(raw_result.get("raw"), usage_sink)
        result = raw_result.get("parsed")
        if isinstance(result, dict):
            result = LlmIntentDecision(**result)
        if isinstance(result, LlmIntentDecision):
            return result
        return None
    except Exception:
        return None


def _resolve_scenario_reference(message: str, semantic_state: ConversationSemanticState | None) -> str | None:
    """Résolution déterministe (jamais par le LLM) d'une phrase de
    référence connue (`semantic_state.resolved_references`) présente dans le
    message — recherche de sous-chaîne insensible à la casse, premier match
    dans l'ordre d'insertion. `None` si aucun état sémantique n'est
    disponible ou si aucune phrase connue n'apparaît dans le message —
    jamais un scénario deviné."""
    if semantic_state is None:
        return None
    lowered_message = message.lower()
    for phrase, scenario_id in semantic_state.resolved_references.items():
        if phrase.lower() in lowered_message:
            return scenario_id
    return None


def extract_intent(
    message: str,
    case_id: str | None = None,
    *,
    recent_turns: list[ConversationTurn] | None = None,
    semantic_state: ConversationSemanticState | None = None,
    usage_sink: dict | None = None,
) -> NluResult | None:
    """Classe `message` en une ou plusieurs `ChatIntent` — `case_id` fourni
    par l'appelant (contexte de conversation déjà connu, ex. dossier
    actuellement affiché côté UI) est toujours préféré à un identifiant
    que le LLM prétendrait avoir détecté dans le texte : le contexte
    explicite de l'appelant est une source plus fiable qu'une extraction
    de texte libre.

    `recent_turns`/`semantic_state` sont optionnels (mémoire conversationnelle,
    Phase 8) — `None` reproduit exactement le comportement d'avant cette
    phase (aucune mémoire, aucune résolution de référence). `usage_sink`
    optionnel (`None` par défaut, aucun changement de comportement) — voir
    `chat/llm_usage.py`."""
    result = _invoke_llm_intent(message, case_id, recent_turns=recent_turns, usage_sink=usage_sink)
    if result is None:
        return None
    resolved_case_id = case_id or result.case_id
    resolved_scenario_id = _resolve_scenario_reference(message, semantic_state)
    return NluResult(
        intents=result.intents,
        case_id=resolved_case_id,
        simulation_changes=result.simulation_changes,
        resolved_scenario_id=resolved_scenario_id,
    )
