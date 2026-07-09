"""Construction du formulaire de décision humaine côté UI — logique pure,
**aucun import Chainlit** (testable indépendamment du runtime).

L'UI est un processus séparé de l'API (voir ``ui/api_client.py``) : elle n'a
jamais accès au ``ClaimState`` serveur, seulement au JSON ``pending_review``
renvoyé par ``GET/POST /claims...`` (``human_review.service.HumanReviewPayload``
sérialisé : ``case_id``/``summary``/``evidence``/``options``). Ce module
reconstruit donc un ``HumanReviewFormView`` à la main à partir de ce JSON,
puis réutilise tel quel l'adaptateur déjà prêt de ``human_review.views``
(``render_for_chainlit_actions``) — jamais une réimplémentation locale du
mapping action → libellé.
"""
from __future__ import annotations

from typing import Any

from human_review.models import ReviewAction
from human_review.views import FORM_FIELDS, HumanReviewFormView, render_for_chainlit_actions

__all__ = [
    "FORM_FIELDS",
    "form_from_pending_review",
    "render_for_chainlit_actions",
    "required_fields_for_action",
]


def form_from_pending_review(pending_review: dict[str, Any]) -> HumanReviewFormView:
    """Reconstruit un ``HumanReviewFormView`` à partir du JSON ``pending_review``
    reçu de l'API. ``recommendation``/``alerts``/``risks``/``disagreements``
    restent aux valeurs par défaut (vides) — l'API ne les expose pas dans
    ``HumanReviewPayload`` aujourd'hui, seuls ``case_id``/``summary``/
    ``evidence``/``options`` le sont."""
    options = pending_review.get("options") or []
    return HumanReviewFormView(
        case_id=pending_review["case_id"],
        summary=list(pending_review.get("summary") or []),
        evidence=dict(pending_review.get("evidence") or {}),
        actions=tuple(ReviewAction(o) for o in options),
    )


def required_fields_for_action(form: HumanReviewFormView, action: ReviewAction) -> list[Any]:
    """Champs à demander pour une action donnée (``justification`` toujours,
    ``target_node`` uniquement pour ``RETRY``) — dérivé de ``form.fields``,
    jamais recopié en dur."""
    return [field for field in form.fields if action in field.applies_to]
