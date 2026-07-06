"""Modèles de décision humaine — package ``human_review``.

Distinct de ``state.claim_state.HumanDecision`` (``TypedDict`` interne au
``StateGraph`` LangGraph, consommé par
``graph.technical_nodes.node_await_human_review``) : depuis le renommage de
``NEEDS_MORE_INFO`` en ``RETRY`` côté graphe, les deux modules partagent
désormais littéralement les mêmes noms d'action
(``APPROVE``/``MODIFY``/``REJECT``/``RETRY``) — mais restent deux contrats
distincts. Ce module définit la version **Pydantic stricte**
(``extra="forbid"``, justification obligatoire pour toute décision,
validée par ``human_review.service``) ; le ``TypedDict`` du graphe reste une
validation maison (``graph.technical_nodes._validate_human_decision``,
justification/``comment`` optionnel). Le câblage complet des deux (un seul
mécanisme de validation, prévu pour l'étape 14) est hors périmètre de ce
fichier.
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import Field, model_validator

from schemas.domain import StrictModel

CASE_ID_PATTERN = r"^CLM-\d{4,}$"


class ReviewAction(str, Enum):
    """Actions qu'un humain peut choisir lors d'une revue de dossier."""

    APPROVE = "APPROVE"
    MODIFY = "MODIFY"
    REJECT = "REJECT"
    RETRY = "RETRY"


class HumanDecision(StrictModel):
    """Décision humaine validée — justification obligatoire, ``target_node``
    réservé à l'action ``RETRY``."""

    case_id: str = Field(..., pattern=CASE_ID_PATTERN)
    actor: str = Field(..., min_length=1)
    action: ReviewAction
    justification: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description=(
            "Commentaire humain court — jamais un document brut, un texte "
            "OCR complet ou un prompt (bornée à 1000 caractères)."
        ),
    )
    target_node: str | None = None
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _validate_target_node(self) -> "HumanDecision":
        if self.action is ReviewAction.RETRY:
            if self.target_node is None or not self.target_node.strip():
                raise ValueError(
                    "target_node est obligatoire pour l'action RETRY."
                )
        elif self.target_node is not None:
            raise ValueError(
                "target_node n'est autorisé que pour l'action RETRY."
            )
        return self


__all__ = ["ReviewAction", "HumanDecision"]
