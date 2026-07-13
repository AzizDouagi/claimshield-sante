"""Prompt système versionné de document_understanding_agent (V2)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

PROMPT_VERSION = "1.0.0"
_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "document_understanding_agent.yaml"


@dataclass(frozen=True)
class DocumentUnderstandingPrompt:
    version: str
    system_prompt: str


@lru_cache(maxsize=1)
def load_document_understanding_prompt() -> DocumentUnderstandingPrompt:
    data = yaml.safe_load(_PROMPT_PATH.read_text(encoding="utf-8"))
    version = str(data["version"])
    if version != PROMPT_VERSION:
        raise ValueError(
            f"Version de prompt document_understanding_agent inattendue : {version!r} "
            f"(attendu {PROMPT_VERSION!r})"
        )
    return DocumentUnderstandingPrompt(version=version, system_prompt=str(data["system_prompt"]))
