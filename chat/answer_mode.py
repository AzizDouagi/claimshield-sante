"""Détecteur déterministe du mode de réponse — chat/answer_mode.py.

Plan de remédiation « autonomie décisionnelle V2 », Phase 7 (« API +
explicabilité chat, sans mémoire »). Distingue, pour une réponse composée
par `chat/response_composer.py`, si elle relève d'un **FAIT** déjà établi
(`FACT`), d'une **HYPOTHÈSE** retenue malgré une information incomplète
(`ASSUMPTION`), ou d'un résultat de **SIMULATION** hypothétique
(`SIMULATION`) — jamais mélangés sans étiquette explicite (point
« explique précisément ce qui ferait changer sa décision » de la définition
de l'autonomie cible, §2 du plan).

Fonction pure : aucun appel LLM, aucune mutation, aucun état conversationnel
— dérive uniquement des données déjà structurées produites par
`chat/tools.py` (`ExplanationFacts`/`SimulationResult`). `AnswerMode` est
défini ici (et non dans `chat/memory_schemas.py`, Phase 8) pour rester
utilisable indépendamment de toute mémoire conversationnelle — la Phase 8
réutilisera cet enum par import, jamais une redéfinition dupliquée.
"""
from __future__ import annotations

from enum import Enum

from chat.schemas import ChatIntent, ExplanationFacts, SimulationResult

__all__ = ["AnswerMode", "detect_answer_modes"]


class AnswerMode(str, Enum):
    """Mode de réponse — jamais un mode inventé, toujours dérivé du contenu
    déjà structuré réellement présent dans les résultats d'outils."""

    FACT = "FACT"
    ASSUMPTION = "ASSUMPTION"
    SIMULATION = "SIMULATION"


def detect_answer_modes(*, intents: list[ChatIntent], tool_results: dict) -> list[AnswerMode]:
    """Retourne l'ensemble ordonné (sans doublon) des modes de réponse
    effectivement engagés par cette réponse.

    - `SIMULATION` dès qu'une simulation a réellement été exécutée
      (`SimulationResult.applied=True`) — un résultat de simulation n'est
      jamais présenté comme un fait établi, quel que soit `decision_changed`.
    - `ASSUMPTION` dès qu'une hypothèse (`ExplanationFacts.assumptions`) ou
      une information manquante (`ExplanationFacts.missing_information`) a
      été retenue par `autonomous_decision_agent` — signale que tout ou
      partie de la décision expliquée repose sur autre chose qu'une preuve
      entièrement confirmée.
    - `FACT` dès qu'au moins une donnée déjà établie (contexte, décision,
      corrections, audit, message patient) est communiquée — mode de repli
      neutre si aucun des deux autres n'est engagé (jamais un mode inventé
      de toutes pièces : une réponse groundée reste toujours au moins
      factuelle par défaut).
    """
    modes: list[AnswerMode] = []

    simulation = tool_results.get("simulation")
    if isinstance(simulation, SimulationResult) and simulation.applied:
        modes.append(AnswerMode.SIMULATION)

    explanation = tool_results.get("explanation")
    if isinstance(explanation, ExplanationFacts) and (
        explanation.assumptions or explanation.missing_information
    ):
        modes.append(AnswerMode.ASSUMPTION)

    has_established_fact = any(
        tool_results.get(key) not in (None, [], {})
        for key in (
            "context",
            "explanation",
            "corrections",
            "audit_summary",
            "patient_message_context",
            "resolved_scenario",
        )
    )
    if has_established_fact or not modes:
        modes.append(AnswerMode.FACT)

    return modes
