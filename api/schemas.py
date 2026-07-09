"""Schémas de requête/réponse de l'API — jamais de contenu brut.

Réutilise les contrats déjà validés ailleurs dans le projet
(``agents.claim_intake_agent.schemas.UploadedFileInfo``,
``human_review.models.ReviewAction``,
``human_review.service.HumanReviewPayload``) plutôt que de les redéfinir —
même principe de non-duplication que le reste du projet.
"""
from __future__ import annotations

from pydantic import Field

from agents.claim_intake_agent.schemas import UploadedFileInfo
from human_review.models import ReviewAction
from human_review.service import HumanReviewPayload
from schemas.domain import ReaderRole, StrictModel


class ClaimSubmissionRequest(StrictModel):
    """Corps de ``POST /claims`` — le dossier doit déjà être déposé sur disque.

    ``source_path`` pointe vers un répertoire déjà présent côté serveur (même
    contrat que ``agents.claim_intake_agent.schemas.ClaimIntakeInput`` — voir
    ``scripts/import_synthea_claimshield_cases.py``/les fixtures de test) :
    cette API ne gère pas l'upload multipart de fichiers, hors périmètre
    d'une exposition minimale du graphe (voir ``api/main.py``).

    ``role`` pilote la politique RBAC DENY-by-default appliquée par
    ``privacy_agent`` pour cette soumission (voir
    ``security.access_policies.ROLE_POLICIES`` — un seul rôle par
    soumission, ``privacy_agent`` ne produit qu'une seule vue minimisée par
    exécution). Défaut ``ADMINISTRATIVE_MANAGER`` (allowlist la plus large)
    si l'appelant ne précise rien — jamais un rôle implicite caché, toujours
    tracé dans ``privacy_result.audit_entry``.
    """

    case_id: str = Field(..., pattern=r"^CLM-\d{4,}$")
    source_path: str = Field(..., min_length=1, description="Répertoire du dossier, côté serveur")
    required_documents: list[str] = Field(default_factory=list)
    uploaded_files: list[UploadedFileInfo] = Field(default_factory=list)
    role: ReaderRole = Field(
        default=ReaderRole.ADMINISTRATIVE_MANAGER,
        description="Rôle RBAC pour la vue minimisée produite par privacy_agent.",
    )


class HumanDecisionRequest(StrictModel):
    """Corps de ``POST /claims/{case_id}/human-decision``.

    Revalidée intégralement par ``human_review.service.validate_and_audit_human_decision``
    (via ``human_review.models.HumanDecision``) au moment de la reprise du
    graphe — ce schéma n'ajoute aucune règle supplémentaire, il ne fait que
    documenter le contrat pour la génération OpenAPI. ``case_id`` n'est pas
    demandé ici : il est déjà connu (chemin de l'URL), complété
    automatiquement par ``graph.technical_nodes.node_await_human_review``.
    """

    actor: str = Field(..., min_length=1)
    action: ReviewAction
    justification: str = Field(..., min_length=1, max_length=1000)
    target_node: str | None = Field(
        default=None, description="Obligatoire uniquement pour l'action RETRY"
    )


class ClaimStatusResponse(StrictModel):
    """État minimisé d'un dossier — jamais de document brut, de texte OCR
    complet ni de donnée personnelle non déjà pseudonymisée par les agents.
    """

    case_id: str
    current_step: str | None = None
    completed_steps: list[str] = Field(default_factory=list)
    final_recommendation: str | None = None
    errors: list[str] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)
    interrupted: bool = False
    pending_review: HumanReviewPayload | None = None


class HealthResponse(StrictModel):
    status: str = "ok"


__all__ = [
    "ClaimStatusResponse",
    "ClaimSubmissionRequest",
    "HealthResponse",
    "HumanDecisionRequest",
]
