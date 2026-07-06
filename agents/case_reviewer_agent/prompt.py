"""Prompt système versionné du Case Reviewer Agent."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

PROMPT_VERSION = "1.1.0"
_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "case_reviewer_agent.yaml"


@dataclass(frozen=True)
class CaseReviewerPrompt:
    """Prompt système chargé depuis prompts/ avec sa version déclarée."""

    version: str
    system_prompt: str


@lru_cache(maxsize=1)
def load_case_reviewer_prompt() -> CaseReviewerPrompt:
    """Charge le prompt système versionné de l'agent."""
    data = yaml.safe_load(_PROMPT_PATH.read_text(encoding="utf-8"))
    version = str(data["version"])
    if version != PROMPT_VERSION:
        raise ValueError(
            f"Version de prompt case_reviewer_agent inattendue : {version!r} "
            f"(attendu {PROMPT_VERSION!r})"
        )
    return CaseReviewerPrompt(
        version=version,
        system_prompt=str(data["system_prompt"]),
    )
