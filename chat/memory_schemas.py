"""Schémas de mémoire conversationnelle — chat/memory_schemas.py.

Plan de remédiation « autonomie décisionnelle V2 », Phase 8 (« mémoire
conversationnelle »). Isolation stricte par clé composite `(user_id,
thread_id)` — voir `chat/conversation_store.py`. Ne conserve **jamais** le
texte intégral d'un message (`ConversationTurn.message_digest`/
`reply_digest` — empreintes non réversibles, voir
`chat.agent._digest`) — uniquement des métadonnées structurées déjà
validées ailleurs dans le projet.

Résumé sémantique structuré et validé (`ConversationSemanticState`, §6.3 du
plan) permet de résoudre une référence à un scénario déjà discuté ("le
premier scénario", "cette hypothèse") sans jamais réinventer une preuve, une
décision ou un scénario — voir `chat/semantic_summarizer.py` pour la
validation obligatoire avant acceptation d'une proposition LLM.

`AnswerMode` n'est **pas** redéfini ici — importé de `chat/answer_mode.py`
(Phase 7), qui reste la source unique de vérité (utilisable indépendamment
de toute mémoire conversationnelle).
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from chat.answer_mode import AnswerMode
from chat.schemas import ChatIntent, SimulationPatch
from schemas.domain import StrictModel

__all__ = [
    "ConversationContext",
    "ConversationSemanticState",
    "ConversationTurn",
    "DiscussedScenario",
    "LlmSemanticSummaryProposal",
    "SimulationContext",
]


class ConversationTurn(StrictModel):
    """Un tour de conversation conservé — jamais le texte intégral du
    message utilisateur ni de la réponse (`message_digest`/`reply_digest`,
    empreintes SHA-256 non réversibles), uniquement des métadonnées déjà
    structurées et validées ailleurs (intentions, dossier concerné, modes de
    réponse engagés, identifiants de preuve cités)."""

    turn_id: str = Field(..., min_length=1)
    message_digest: str = Field(..., min_length=1)
    reply_digest: str = Field(..., min_length=1)
    intents: list[ChatIntent] = Field(default_factory=list)
    case_id: str | None = Field(default=None, pattern=r"^CLM-\d{4,}$")
    answer_modes: list[AnswerMode] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    created_at: datetime


class SimulationContext(StrictModel):
    """Une simulation déjà exécutée pendant la conversation — jamais le
    dossier synthétique interne (voir `chat/simulation_engine.py`),
    uniquement le résumé déjà minimisé (`chat.schemas.SimulationResult`)."""

    scenario_id: str = Field(..., pattern=r"^SCENARIO-\d+$")
    original_decision: str | None = None
    simulated_decision: str | None = None
    decision_changed: bool = False


class DiscussedScenario(StrictModel):
    """Un scénario (décision réelle, simulation ou contrefactuel) déjà
    évoqué dans la conversation — identifiant stable et ordinal,
    référençable explicitement par l'utilisateur ("le premier scénario",
    "cette hypothèse"). `related_decision` doit toujours être cohérent avec
    `kind` — vérifié par `chat/semantic_summarizer.py`, jamais laissé au
    seul LLM."""

    scenario_id: str = Field(..., pattern=r"^SCENARIO-\d+$")
    description: str = Field(..., max_length=300)
    kind: Literal["REAL_DECISION", "SIMULATION", "COUNTERFACTUAL"]
    related_decision: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class ConversationSemanticState(StrictModel):
    """Résumé sémantique structuré — jamais le texte intégral des tours,
    jamais une preuve/décision/scénario inventé (validation obligatoire dans
    `chat/semantic_summarizer.py` avant construction). Expire avec le reste
    de `ConversationContext` (même TTL, aucun cycle de vie séparé)."""

    conversation_summary: str = Field(..., max_length=800)
    last_user_goal: str | None = Field(default=None, max_length=200)
    discussed_scenarios: list[DiscussedScenario] = Field(default_factory=list, max_length=10)
    open_questions: list[str] = Field(default_factory=list, max_length=10)
    resolved_references: dict[str, str] = Field(default_factory=dict)
    compared_decisions: list[str] = Field(default_factory=list)
    updated_at: datetime


class LlmSemanticSummaryProposal(StrictModel):
    """Sortie brute du LLM de résumé sémantique (`chat/semantic_summarizer.py`)
    — **jamais** acceptée telle quelle : chaque référence citée (preuve,
    scénario, décision) est revalidée contre les données réellement connues
    de la conversation avant construction du `ConversationSemanticState`
    final, même patron anti-hallucination que partout ailleurs dans le
    projet (ex. `agents.autonomous_decision_agent.agent._validate_llm_factor_references`)."""

    conversation_summary: str = Field(default="", max_length=2000)
    last_user_goal: str | None = Field(default=None, max_length=2000)
    discussed_scenarios: list[DiscussedScenario] = Field(default_factory=list, max_length=10)
    open_questions: list[str] = Field(default_factory=list, max_length=10)
    resolved_references: dict[str, str] = Field(default_factory=dict)
    compared_decisions: list[str] = Field(default_factory=list)


class ConversationContext(StrictModel):
    """État conversationnel complet d'un thread — fenêtre bornée de tours
    récents (`ConversationStore.append_turn` tronque, jamais ce schéma),
    simulations déjà exécutées, résumé sémantique optionnel. Expire
    entièrement ensemble (`ConversationStore.expire_older_than`) — jamais un
    tour isolé conservé au-delà du TTL.

    `active_simulation_patches` (Phase 9, plan de remédiation « autonomie
    décisionnelle V2 », §7) : accumulation des `SimulationPatch` d'une
    simulation ciblée en cours de discussion — permet un « et si on
    changeait aussi... » qui complète la simulation précédente plutôt que de
    repartir du dossier réel à chaque message (voir
    `chat.conversation_store.ConversationStore.update_active_simulation`)."""

    thread_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    turns: list[ConversationTurn] = Field(default_factory=list)
    simulations: list[SimulationContext] = Field(default_factory=list)
    semantic_state: ConversationSemanticState | None = None
    active_simulation_patches: list[SimulationPatch] = Field(default_factory=list)
    updated_at: datetime
