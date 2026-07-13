"""Modèle de requête d'override humain post-décision (V2) — human_review/override_models.py.

Distinct de `human_review/models.py::HumanDecision` (V1, HITL bloquant,
non modifié — §0 du plan de refonte) : ici, la décision autonome du
pipeline V2 est déjà rendue et persistée (`ClaimStateV2.decision_result`)
— un override n'est jamais une décision *avant* le pipeline, toujours une
correction *après* (voir `services/override_store.py`, Phase V2-0, qui
possède déjà `OverrideAction`/`OverrideRecord` — réutilisés ici par import,
jamais redéfinis).
"""
from __future__ import annotations

from pydantic import Field

from schemas.domain import StrictModel
from services.override_store import OverrideAction

__all__ = ["OverrideRequest"]

CASE_ID_PATTERN = r"^CLM-\d{4,}$"


class OverrideRequest(StrictModel):
    """Requête humaine brute — validée avant construction d'un
    `services.override_store.OverrideRecord`. `case_id` figure dans l'URL
    côté API (Phase V2-9) mais reste ici obligatoire : ce module ne suppose
    aucun transport particulier."""

    case_id: str = Field(..., pattern=CASE_ID_PATTERN)
    actor: str = Field(..., min_length=1, max_length=255)
    action: OverrideAction
    justification: str = Field(..., min_length=1, max_length=1000)
