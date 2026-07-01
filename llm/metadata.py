"""Construction de métadonnées LLM minimales pour les résultats d'agents."""
from __future__ import annotations

from config.settings import get_settings
from llm.prompts import load_prompt_version
from schemas.results import LlmMetadata


def build_llm_metadata(agent_name: str, confidence: float | None = None) -> LlmMetadata:
    """Retourne modèle, version de prompt et confiance, sans contenu de prompt."""
    settings = get_settings()
    return LlmMetadata(
        model_name=settings.claimshield_llm_model,
        prompt_version=load_prompt_version(agent_name),
        confidence=confidence,
    )
