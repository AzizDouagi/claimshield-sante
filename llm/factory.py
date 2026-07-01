"""Factory ChatOllama partagée — une seule instance mise en cache par processus."""
from __future__ import annotations

from functools import lru_cache

try:  # pragma: no cover - exercised indirectly when the optional package is absent.
    from langchain_ollama import ChatOllama
except ModuleNotFoundError:  # pragma: no cover
    class ChatOllama:  # type: ignore[no-redef]
        """Import différé pour permettre aux tests mockés de charger la factory."""

        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "Le paquet 'langchain_ollama' est requis pour appeler le LLM. "
                "Installez les dépendances du projet avant un appel réel."
            )

from config.settings import get_settings


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    """Retourne l'instance ChatOllama mise en cache.

    temperature=0 → reproductibilité maximale pour l'audit.
    """
    s = get_settings()
    return ChatOllama(
        model=s.claimshield_llm_model,
        base_url=str(s.ollama_base_url),
        temperature=0,
        num_predict=2048,
    )


def reset_llm_cache() -> None:
    """Vide le cache LRU — à appeler dans les tests pour injecter un mock."""
    get_llm.cache_clear()
