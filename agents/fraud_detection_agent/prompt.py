"""Prompt système versionné du Fraud Detection Agent."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

PROMPT_VERSION = "1.2.0"
_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "fraud_detection_agent.yaml"


@dataclass(frozen=True)
class FraudDetectionPrompt:
    """Prompt système chargé depuis prompts/ avec sa version déclarée."""

    version: str
    system_prompt: str


@lru_cache(maxsize=1)
def load_fraud_detection_prompt() -> FraudDetectionPrompt:
    """Charge le prompt système versionné de l'agent.

    Le fichier YAML est la source de vérité pour le texte envoyé au LLM.
    La constante PROMPT_VERSION sert de garde-fou local : un écart de version
    rend le démarrage explicite plutôt que silencieux.
    """
    data = yaml.safe_load(_PROMPT_PATH.read_text(encoding="utf-8"))
    version = str(data["version"])
    if version != PROMPT_VERSION:
        raise ValueError(
            f"Version de prompt fraud_detection_agent inattendue : {version!r} "
            f"(attendu {PROMPT_VERSION!r})"
        )
    return FraudDetectionPrompt(
        version=version,
        system_prompt=str(data["system_prompt"]),
    )
