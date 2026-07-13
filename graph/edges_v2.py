"""Routage conditionnel du graphe V2 — graph/edges_v2.py.

Une seule branche conditionnelle dans tout le graphe V2 (après
`intake_safety`) — le reste du pipeline est strictement séquentiel (plan V2
§2 : `document_understanding`/`eligibility`/`medical_risk` ne bloquent
jamais, `autonomous_decision`/`audit_service`/`finalize` sont toujours
traversés). Contraste volontaire avec V1 (`graph/edges.py`, 12 fonctions de
routage) — la simplification du nombre de branches est un objectif du plan
de refonte, pas un oubli.
"""
from __future__ import annotations

from typing import Literal

from schemas.domain import IntakeSafetyStatus
from state.claim_state_v2 import ClaimStateV2

__all__ = ["Route", "route_intake_safety"]

Route = Literal["continue", "terminal"]


def route_intake_safety(state: ClaimStateV2) -> Route:
    """`ACCEPTED` → poursuite du pipeline (`document_understanding`).
    `BLOCKED`/`QUARANTINED`/`TECHNICAL_FAILURE` → court-circuit direct vers
    `audit_service` (jamais `document_understanding`/`eligibility`/
    `medical_risk`/`autonomous_decision`, qui supposeraient un dossier
    admis)."""
    result = state.get("intake_safety_result")
    if result is not None and result.status is IntakeSafetyStatus.ACCEPTED:
        return "continue"
    return "terminal"
