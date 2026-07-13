"""Prompt système versionné de intake_safety_agent (V2)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

PROMPT_VERSION = "1.0.0"
_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "intake_safety_agent.yaml"


@dataclass(frozen=True)
class IntakeSafetyPrompt:
    """Prompt système chargé depuis prompts/ avec sa version déclarée."""

    version: str
    system_prompt: str


@lru_cache(maxsize=1)
def load_intake_safety_prompt() -> IntakeSafetyPrompt:
    """Charge le prompt système versionné de l'agent.

    Le fichier YAML est la source de vérité pour le texte envoyé au LLM.
    La constante PROMPT_VERSION sert de garde-fou local : un écart de
    version rend le démarrage explicite plutôt que silencieux.
    """
    data = yaml.safe_load(_PROMPT_PATH.read_text(encoding="utf-8"))
    version = str(data["version"])
    if version != PROMPT_VERSION:
        raise ValueError(
            f"Version de prompt intake_safety_agent inattendue : {version!r} "
            f"(attendu {PROMPT_VERSION!r})"
        )
    return IntakeSafetyPrompt(version=version, system_prompt=str(data["system_prompt"]))
