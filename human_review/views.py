"""Vue formulaire HITL — représentation prête à afficher, avant décision.

Framework-agnostique : n'importe ni FastAPI ni Chainlit (aucun des deux
n'est requis pour ce module — Chainlit n'est d'ailleurs pas une dépendance
du projet). Prépare des modèles Pydantic sérialisables en JSON,
consommables tels quels :
  - par une route FastAPI (``response_model=HumanReviewFormView`` ou
    ``JSONResponse(form.model_dump(mode="json"))``) ;
  - par un composant Chainlit (``render_for_chainlit_actions()`` produit une
    liste de dicts au format attendu par ``chainlit.Action``, sans jamais
    importer le paquet).

Ne génère aucun HTML ni template — uniquement des données structurées.
Bâti sur ``human_review/service.py`` (payload minimisé) et
``human_review/models.py`` (contrat de décision) : n'invente aucune
nouvelle source de vérité, ne fait que composer les deux pour l'affichage
et rediriger la soumission vers la validation déjà existante.
"""
from __future__ import annotations

from typing import Any, Mapping

from pydantic import Field, field_validator

from human_review.models import HumanDecision, ReviewAction
from human_review.service import build_human_review_payload, validate_human_decision
from schemas.domain import Recommendation, StrictModel
from schemas.results import DisagreementPoint
from state.claim_state import ClaimState

# ── Champs du formulaire de décision ─────────────────────────────────────────


class EditableField(StrictModel):
    """Un champ modifiable du formulaire de décision humaine.

    Décrit un champ de ``HumanDecision`` — jamais un champ de la réclamation
    elle-même : cette vue n'expose que le formulaire de décision (valider,
    modifier, refuser, relancer), pas un éditeur de données métier.
    """

    name: str
    label: str
    required: bool
    applies_to: tuple[ReviewAction, ...]


JUSTIFICATION_FIELD = EditableField(
    name="justification",
    label="Justification (obligatoire pour toute décision)",
    required=True,
    applies_to=tuple(ReviewAction),
)
"""Toujours affiché et toujours obligatoire, quelle que soit l'action
choisie — voir ``HumanReviewFormView.justification_required`` (verrouillé)."""

TARGET_NODE_FIELD = EditableField(
    name="target_node",
    label="Étape à relancer (obligatoire uniquement pour « Relancer »)",
    required=False,
    applies_to=(ReviewAction.RETRY,),
)
"""N'apparaît que pour l'action RETRY — ``HumanDecision`` interdit ce champ
pour toute autre action (voir ``human_review.models.HumanDecision``)."""

FORM_FIELDS: tuple[EditableField, ...] = (JUSTIFICATION_FIELD, TARGET_NODE_FIELD)

_ACTION_LABELS: dict[ReviewAction, str] = {
    ReviewAction.APPROVE: "Valider",
    ReviewAction.MODIFY: "Modifier",
    ReviewAction.REJECT: "Refuser",
    ReviewAction.RETRY: "Relancer",
}

# ── Formulaire ────────────────────────────────────────────────────────────────


class HumanReviewFormView(StrictModel):
    """Formulaire HITL prêt à afficher.

    Ne porte jamais de décision — uniquement ce qu'un humain doit voir avant
    d'en prendre une : la pré-recommandation non finale, les preuves et
    alertes déjà minimisées (voir ``human_review.service``), les risques et
    contradictions déjà calculés par ``case_reviewer_agent`` (s'ils sont
    disponibles), les quatre actions possibles et les champs à remplir.
    """

    case_id: str
    recommendation: Recommendation | None = Field(
        default=None,
        description=(
            "Pré-recommandation non finale du Case Reviewer, si déjà "
            "disponible — jamais une décision finale."
        ),
    )
    summary: list[str] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)
    alerts: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    disagreements: list[DisagreementPoint] = Field(default_factory=list)
    actions: tuple[ReviewAction, ...] = Field(default_factory=lambda: tuple(ReviewAction))
    fields: tuple[EditableField, ...] = Field(default_factory=lambda: FORM_FIELDS)
    justification_required: bool = True

    @field_validator("justification_required")
    @classmethod
    def _justification_always_required(cls, v: bool) -> bool:
        if not v:
            raise ValueError(
                "justification_required doit toujours être True : aucune "
                "soumission sans justification humaine n'est autorisée."
            )
        return v


def build_human_review_form(state: ClaimState) -> HumanReviewFormView:
    """Compose le formulaire HITL à partir du state.

    Réutilise ``human_review.service.build_human_review_payload`` pour les
    preuves déjà minimisées (jamais un document brut, un texte OCR complet,
    un secret ou un prompt complet) et lit la pré-recommandation/les
    risques/les contradictions déjà calculés par ``case_reviewer_agent``
    (``state["review_result"]``) s'ils sont disponibles — ne recalcule
    jamais rien, ne fait que présenter.
    """
    payload = build_human_review_payload(state)
    review_result = state.get("review_result")
    result_payload = getattr(review_result, "result_payload", None)

    return HumanReviewFormView(
        case_id=payload.case_id,
        recommendation=getattr(result_payload, "recommendation", None),
        summary=payload.summary,
        evidence=payload.evidence,
        alerts=list(state.get("alerts") or []),
        risks=list(getattr(result_payload, "risks", None) or []),
        disagreements=list(getattr(result_payload, "disagreements", None) or []),
    )


# ── Soumission ────────────────────────────────────────────────────────────────


def submit_human_review_decision(raw: Mapping[str, Any]) -> HumanDecision:
    """Point d'entrée de soumission du formulaire.

    Ne fait que déléguer à ``human_review.service.validate_human_decision`` —
    n'introduit aucune règle de validation supplémentaire ni de chemin de
    contournement : une décision sans justification (ou par ailleurs
    invalide) est toujours rejetée avec une ``HumanDecisionValidationError``
    structurée, jamais silencieusement acceptée.
    """
    return validate_human_decision(raw)


# ── Adaptateurs de présentation (FastAPI / Chainlit) ────────────────────────


def render_for_fastapi(form: HumanReviewFormView) -> dict[str, Any]:
    """Représentation JSON directement utilisable comme corps de réponse
    FastAPI. ``HumanReviewFormView`` étant déjà un modèle Pydantic
    (``StrictModel``), ``response_model=HumanReviewFormView`` fonctionne
    aussi nativement sans passer par cette fonction — un raccourci explicite
    pour un usage hors route typée (ex. ``JSONResponse``)."""
    return form.model_dump(mode="json")


def render_for_chainlit_actions(form: HumanReviewFormView) -> list[dict[str, str]]:
    """Représentation des actions au format attendu par ``chainlit.Action``
    (``name``/``value``/``label``) — Chainlit n'est pas une dépendance du
    projet (absent de ``requirements.txt``) : cette fonction ne l'importe
    jamais, elle ne fait que produire une structure de dicts compatible.
    """
    return [
        {"name": action.value, "value": action.value, "label": _ACTION_LABELS[action]}
        for action in form.actions
    ]


__all__ = [
    "EditableField",
    "FORM_FIELDS",
    "HumanReviewFormView",
    "JUSTIFICATION_FIELD",
    "TARGET_NODE_FIELD",
    "build_human_review_form",
    "render_for_chainlit_actions",
    "render_for_fastapi",
    "submit_human_review_decision",
]
