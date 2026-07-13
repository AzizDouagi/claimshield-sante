"""Schémas de requête/réponse — API v2 (pipeline autonome, plan de refonte V2).

Réutilise `schemas.domain.ClaimDecisionV2`/`ReaderRole` et
`services.override_store.OverrideAction` plutôt que de les redéfinir — même
principe de non-duplication que `api/schemas.py` (V1, non modifié).
Contrairement à `api.schemas.ClaimStatusResponse` (V1), aucun champ
``pending_review``/``interrupted`` : le graphe V2 ne bloque jamais (§0 du
plan — voir `graph/workflow_v2.py`).
"""
from __future__ import annotations

from datetime import datetime

from pydantic import Field

from schemas.domain import ClaimDecisionV2, ReaderRole, StrictModel
from services.override_store import OverrideAction

__all__ = [
    "ClaimStatusResponseV2",
    "ClaimSubmissionRequestV2",
    "HealthResponseV2",
    "OverrideRecordResponseV2",
    "OverrideRequestBodyV2",
]


class ClaimSubmissionRequestV2(StrictModel):
    """Corps de ``POST /v2/claims`` — même contrat de dépôt que V1
    (``source_path`` déjà présent sur disque, pas d'upload multipart ici),
    mais sans ``uploaded_files`` (jamais consommé par `intake_safety_agent`,
    qui lit directement le manifeste du répertoire)."""

    case_id: str = Field(..., pattern=r"^CLM-\d{4,}$")
    source_path: str = Field(..., min_length=1, description="Répertoire du dossier, côté serveur")
    required_documents: list[str] = Field(default_factory=list)
    role: ReaderRole = Field(
        default=ReaderRole.ADMINISTRATIVE_MANAGER,
        description="Rôle RBAC pour la vue minimisée produite par document_understanding_agent.",
    )


class ClaimStatusResponseV2(StrictModel):
    """État minimisé d'un dossier V2 — jamais de document brut, de texte OCR
    complet ni de donnée personnelle non déjà pseudonymisée. Contrairement à
    V1, toujours un état *terminal* dès la réponse de soumission (le graphe
    V2 ne s'interrompt jamais)."""

    case_id: str
    current_step: str | None = None
    completed_steps: list[str] = Field(default_factory=list)
    final_decision: ClaimDecisionV2 | None = None
    decision_summary: list[str] = Field(default_factory=list)
    bounded_by: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)


class OverrideRequestBodyV2(StrictModel):
    """Corps de ``POST /v2/claims/{case_id}/override``.

    Revalidé intégralement par
    ``human_review.override_service.validate_and_record_override`` — ce
    schéma ne fait que documenter le contrat pour la génération OpenAPI.
    ``case_id`` n'est pas demandé ici : déjà connu (chemin de l'URL).
    """

    actor: str = Field(..., min_length=1, max_length=255)
    action: OverrideAction
    justification: str = Field(..., min_length=1, max_length=1000)


class OverrideRecordResponseV2(StrictModel):
    """Miroir de ``services.override_store.OverrideRecord`` — même champs,
    schéma API distinct pour ne jamais coupler la forme HTTP au schéma de
    stockage interne."""

    override_id: str
    case_id: str
    actor: str
    action: OverrideAction
    justification: str
    original_decision: str | None = None
    recorded_at: datetime


class HealthResponseV2(StrictModel):
    status: str = "ok"
