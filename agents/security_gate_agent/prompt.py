"""Prompt système versionné du Security Gate Agent."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

PROMPT_VERSION = "1.0.0"
_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "security_gate_agent.yaml"


@dataclass(frozen=True)
class SecurityGatePrompt:
    """Prompt système chargé depuis prompts/ avec sa version déclarée."""

    version: str
    system_prompt: str


@lru_cache(maxsize=1)
def load_security_gate_prompt() -> SecurityGatePrompt:
    """Charge le prompt système versionné envoyé au LLM de sécurité."""
    data = yaml.safe_load(_PROMPT_PATH.read_text(encoding="utf-8"))
    version = str(data["version"])
    if version != PROMPT_VERSION:
        raise ValueError(
            f"Version de prompt security_gate_agent inattendue : {version!r} "
            f"(attendu {PROMPT_VERSION!r})"
        )
    return SecurityGatePrompt(
        version=version,
        system_prompt=str(data["system_prompt"]),
    )
