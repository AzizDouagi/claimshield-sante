"""Moteur d'explication — chat/explanation_engine.py (plan V2 §6, Phase V2-11a).

Fonction pure : réorganise des champs déjà présents dans une réponse API v2
minimisée (`api.v2.schemas.ClaimStatusResponseV2`, reçue ici comme `dict`
JSON déjà sûr) en `ExplanationFacts` — jamais une nouvelle inférence, jamais
un calcul métier. Le Response Composer (`chat/response_composer.py`) est
seul responsable de la mise en forme en langage naturel.
"""
from __future__ import annotations

from chat.schemas import ExplanationFacts

__all__ = ["build_explanation_facts"]


def build_explanation_facts(context: dict) -> ExplanationFacts:
    return ExplanationFacts(
        case_id=context["case_id"],
        final_decision=context.get("final_decision"),
        decision_summary=list(context.get("decision_summary") or []),
        bounded_by=list(context.get("bounded_by") or []),
        errors=list(context.get("errors") or []),
        alerts=list(context.get("alerts") or []),
    )
