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
    """`missing_information`/`assumptions`/`decisive_factors`/`counterfactuals`/
    `recommended_action`/`evidence_completeness` (Phase 7) sont repris tels
    quels du `context` déjà minimisé par l'API — chaque élément de liste
    arrive comme un `dict` JSON (réponse HTTP déjà sérialisée) et est
    revalidé par Pydantic à la construction d'`ExplanationFacts`, jamais un
    contenu non structuré accepté."""
    return ExplanationFacts(
        case_id=context["case_id"],
        final_decision=context.get("final_decision"),
        decision_summary=list(context.get("decision_summary") or []),
        bounded_by=list(context.get("bounded_by") or []),
        errors=list(context.get("errors") or []),
        alerts=list(context.get("alerts") or []),
        missing_information=list(context.get("missing_information") or []),
        assumptions=list(context.get("assumptions") or []),
        decisive_factors=list(context.get("decisive_factors") or []),
        counterfactuals=list(context.get("counterfactuals") or []),
        recommended_action=context.get("recommended_action") or "",
        evidence_completeness=context.get("evidence_completeness"),
    )
