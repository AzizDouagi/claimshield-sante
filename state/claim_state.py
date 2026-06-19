"""État partagé du workflow LangGraph — ClaimShield Santé.

ClaimState est le seul objet qui traverse tous les nœuds du graphe.
Règles :
- Minimal : pas de texte OCR brut, pas de contenu PDF, pas de secrets.
- Append-only pour les listes (reducers operator.add) — LangGraph ne permet
  pas d'écraser une liste accumulée sans reducer explicite.
- Sérialisable JSON (tous les types sont des primitives ou des dict Pydantic).
- Versionné : schema_version permet de détecter un state produit par une
  version antérieure du workflow lors d'une reprise sur checkpoint.
"""

from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, TypedDict

from schemas.domain import Recommendation
from schemas.results import (
    AuditEvent,
    CaseReviewerResult,
    ClaimIntakeResult,
    ClinicalConsistencyResult,
    DocumentOcrResult,
    FhirValidatorResult,
    FraudDetectionResult,
    IdentityCoverageResult,
    MedicalCodingResult,
    PrivacyResult,
    SecurityGateResult,
)


# ── Décision humaine (Human-in-the-Loop) ─────────────────────────────────────


class HumanDecision(TypedDict, total=False):
    """Décision saisie par le gestionnaire après interruption LangGraph."""

    actor: str
    decision: str          # "APPROVE" | "REJECT" | "NEEDS_MORE_INFO"
    comment: str
    decided_at: datetime


# ── État principal ────────────────────────────────────────────────────────────


class ClaimState(TypedDict, total=False):
    """État partagé passé à travers tous les nœuds du StateGraph.

    Les champs marqués `Annotated[list, operator.add]` sont append-only :
    chaque nœud ajoute ses éléments sans écraser ceux des nœuds précédents.
    """

    # ── Identité du dossier ───────────────────────────────────────────────────
    case_id: str
    schema_version: str                      # version de ce schéma d'état

    # ── Progression du workflow ───────────────────────────────────────────────
    current_step: str                        # nom du nœud actif
    completed_steps: Annotated[list[str], operator.add]

    # ── Résultats des agents (un par agent, écrasable) ────────────────────────
    intake_result: ClaimIntakeResult | None
    security_result: SecurityGateResult | None
    privacy_result: PrivacyResult | None
    identity_coverage_result: IdentityCoverageResult | None
    fhir_result: FhirValidatorResult | None
    ocr_result: DocumentOcrResult | None
    coding_result: MedicalCodingResult | None
    clinical_result: ClinicalConsistencyResult | None
    fraud_result: FraudDetectionResult | None
    review_result: CaseReviewerResult | None

    # ── Erreurs et alertes (append-only) ─────────────────────────────────────
    errors: Annotated[list[str], operator.add]
    alerts: Annotated[list[str], operator.add]

    # ── Audit (append-only) ───────────────────────────────────────────────────
    audit_trail: Annotated[list[AuditEvent], operator.add]

    # ── Décision humaine (HITL) ───────────────────────────────────────────────
    human_decision: HumanDecision | None

    # ── Recommandation finale ─────────────────────────────────────────────────
    final_recommendation: Recommendation | None
    final_justification: Annotated[list[str], operator.add]
