"""Chargement des prompts système depuis prompts/{agent_name}.yaml."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


@lru_cache(maxsize=20)
def load_prompt(agent_name: str) -> str:
    """Charge et met en cache le prompt système d'un agent.

    Args:
        agent_name: nom de l'agent (ex. "medical_coding_agent").

    Returns:
        Contenu de la clé system_prompt du fichier YAML.

    Raises:
        FileNotFoundError: si le fichier prompts/{agent_name}.yaml est absent.
        KeyError: si la clé system_prompt est manquante dans le YAML.
    """
    path = _PROMPTS_DIR / f"{agent_name}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["system_prompt"]


@lru_cache(maxsize=20)
def load_prompt_version(agent_name: str) -> str:
    """Charge uniquement la version du prompt, jamais son contenu."""
    path = _PROMPTS_DIR / f"{agent_name}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return str(data["version"])


def reset_prompt_cache() -> None:
    """Vide le cache des prompts — utile dans les tests."""
    load_prompt.cache_clear()
    load_prompt_version.cache_clear()
