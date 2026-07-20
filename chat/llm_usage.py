"""Capture de l'usage token d'un appel LLM — helper partagé unique.

Utilisé par les 4 sites d'appel LLM propres à `chat/` (`chat/nlu.py`,
`chat/response_composer.py` ×2, `chat/semantic_summarizer.py`) pour
alimenter la visibilité temps réel demandée par AZIZ (comme Claude Code) —
voir `chat/agent.py::ChatStepEvent`. Jamais dupliqué : un seul point de
lecture de `usage_metadata`/`response_metadata` sur un message LangChain.
"""
from __future__ import annotations

__all__ = ["record_usage"]


def record_usage(result: object, usage_sink: dict | None) -> None:
    """Lit `usage_metadata`/le nom du modèle sur un `AIMessage` déjà
    obtenu (champs natifs de `langchain_core`, jamais lus ailleurs dans le
    projet avant cette fonctionnalité). `usage_sink` reste `None` par
    défaut pour tout appelant qui ne s'y intéresse pas — aucun changement
    de comportement dans ce cas ; `result` peut être `None` (ex. échec
    `raw_result.get("raw")`) sans jamais lever."""
    if usage_sink is None or result is None:
        return
    usage = getattr(result, "usage_metadata", None)
    if usage:
        usage_sink["input_tokens"] = usage.get("input_tokens")
        usage_sink["output_tokens"] = usage.get("output_tokens")
    metadata = getattr(result, "response_metadata", None) or {}
    model_name = metadata.get("model_name") or metadata.get("model")
    if model_name:
        usage_sink["model_name"] = model_name
