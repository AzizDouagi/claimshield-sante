"""Prompts système versionnés du Chat Reasoning Agent (V2) — chat/prompt.py.

Même patron que `agents/*/prompt.py` (V1/V2) : version figée, garde-fou
contre un YAML désynchronisé, cache mémoire (`lru_cache`)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

__all__ = [
    "ChatIntentExtractionPrompt",
    "ChatPatientMessagePrompt",
    "ChatReasoningPrompt",
    "ChatSemanticSummaryPrompt",
    "load_chat_intent_extraction_prompt",
    "load_chat_patient_message_prompt",
    "load_chat_reasoning_prompt",
    "load_chat_semantic_summary_prompt",
]

INTENT_EXTRACTION_PROMPT_VERSION = "1.1.0"
REASONING_PROMPT_VERSION = "1.2.0"
PATIENT_MESSAGE_PROMPT_VERSION = "1.0.0"
SEMANTIC_SUMMARY_PROMPT_VERSION = "1.1.0"
_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
_INTENT_EXTRACTION_PATH = _PROMPTS_DIR / "chat_intent_extraction.yaml"
_REASONING_PATH = _PROMPTS_DIR / "chat_reasoning_agent.yaml"
_PATIENT_MESSAGE_PATH = _PROMPTS_DIR / "chat_patient_message.yaml"
_SEMANTIC_SUMMARY_PATH = _PROMPTS_DIR / "chat_semantic_summary.yaml"


@dataclass(frozen=True)
class ChatIntentExtractionPrompt:
    version: str
    system_prompt: str


@dataclass(frozen=True)
class ChatReasoningPrompt:
    version: str
    system_prompt: str


@dataclass(frozen=True)
class ChatPatientMessagePrompt:
    version: str
    system_prompt: str


@dataclass(frozen=True)
class ChatSemanticSummaryPrompt:
    version: str
    system_prompt: str


@lru_cache(maxsize=1)
def load_chat_intent_extraction_prompt() -> ChatIntentExtractionPrompt:
    data = yaml.safe_load(_INTENT_EXTRACTION_PATH.read_text(encoding="utf-8"))
    version = str(data["version"])
    if version != INTENT_EXTRACTION_PROMPT_VERSION:
        raise ValueError(
            f"Version de prompt chat_intent_extraction inattendue : {version!r} "
            f"(attendu {INTENT_EXTRACTION_PROMPT_VERSION!r})"
        )
    return ChatIntentExtractionPrompt(version=version, system_prompt=str(data["system_prompt"]))


@lru_cache(maxsize=1)
def load_chat_reasoning_prompt() -> ChatReasoningPrompt:
    data = yaml.safe_load(_REASONING_PATH.read_text(encoding="utf-8"))
    version = str(data["version"])
    if version != REASONING_PROMPT_VERSION:
        raise ValueError(
            f"Version de prompt chat_reasoning_agent inattendue : {version!r} "
            f"(attendu {REASONING_PROMPT_VERSION!r})"
        )
    return ChatReasoningPrompt(version=version, system_prompt=str(data["system_prompt"]))


@lru_cache(maxsize=1)
def load_chat_patient_message_prompt() -> ChatPatientMessagePrompt:
    data = yaml.safe_load(_PATIENT_MESSAGE_PATH.read_text(encoding="utf-8"))
    version = str(data["version"])
    if version != PATIENT_MESSAGE_PROMPT_VERSION:
        raise ValueError(
            f"Version de prompt chat_patient_message inattendue : {version!r} "
            f"(attendu {PATIENT_MESSAGE_PROMPT_VERSION!r})"
        )
    return ChatPatientMessagePrompt(version=version, system_prompt=str(data["system_prompt"]))


@lru_cache(maxsize=1)
def load_chat_semantic_summary_prompt() -> ChatSemanticSummaryPrompt:
    data = yaml.safe_load(_SEMANTIC_SUMMARY_PATH.read_text(encoding="utf-8"))
    version = str(data["version"])
    if version != SEMANTIC_SUMMARY_PROMPT_VERSION:
        raise ValueError(
            f"Version de prompt chat_semantic_summary inattendue : {version!r} "
            f"(attendu {SEMANTIC_SUMMARY_PROMPT_VERSION!r})"
        )
    return ChatSemanticSummaryPrompt(version=version, system_prompt=str(data["system_prompt"]))
