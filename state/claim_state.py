"""État partagé du workflow LangGraph — ClaimShield Santé.

ClaimState est le seul objet qui traverse tous les nœuds du graphe.
Règles :
- Minimal : pas de texte OCR brut, pas de contenu PDF, pas de secrets.
- Append-only pour les listes (reducers operator.add) — LangGraph ne permet
  pas d'écraser une liste accumulée sans reducer explicite.
- Sérialisable JSON (tous les types sont des primitives ou des dict Pydantic).
- Versionné : schema_version permet de détecter un state produit par une
  version antérieure du workflow lors d'une reprise sur checkpoint.

Contenu autorisé après ingestion :
  case_id · statut ingestion · manifest structuré · métadonnées documents
  chemins relatifs · hashes SHA-256 · alertes · erreurs

Contenu interdit (garanti par validate_state_update) :
  octets bruts · PDF/images en base64 · texte OCR · chemins absolus
  objets fichier ouverts
"""

from __future__ import annotations

import io
import operator
import re
from datetime import datetime
from typing import Annotated, TypedDict

from schemas.domain import IntakeStatus, Recommendation
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

# Détecte les chaînes qui ressemblent à des chemins absolus POSIX, Windows ou UNC.
_ABSOLUTE_PATH_RE = re.compile(r"^(?:/|[A-Za-z]:[/\\]|\\\\)")


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

    Après ingestion, seuls des champs propres (métadonnées, hashes, chemins
    relatifs) doivent figurer dans le state.  intake_input est consommé et
    vidé par le nœud d'ingestion afin que le source_path absolu ne persiste
    pas dans les checkpoints LangGraph.
    """

    # ── Identité du dossier ───────────────────────────────────────────────────
    case_id: str
    schema_version: str                      # version de ce schéma d'état

    # ── Progression du workflow ───────────────────────────────────────────────
    current_step: str                        # nom du nœud actif
    completed_steps: Annotated[list[str], operator.add]

    # ── Entrée d'ingestion — consommée et vidée par le nœud intake ───────────
    # Ce champ peut contenir source_path (chemin absolu temporaire).
    # Le nœud intake le remet à None après traitement ; aucun autre nœud
    # ne doit le lire.
    intake_input: dict | None

    # ── Entrée sécurité — consommée et vidée par le nœud security_gate ───────
    # Contient text_fields et deterministic_injection_flag.
    # Le nœud security_gate le remet à None après traitement.
    security_input: dict | None

    # ── Entrée privacy — consommée et vidée par le nœud privacy ──────────────
    # Contient role, data_classification, contains_real_personal_data, etc.
    # Le nœud privacy le remet à None après traitement.
    privacy_input: dict | None

    # ── Résultats des agents (un par agent, écrasable) ────────────────────────
    intake_result: ClaimIntakeResult | None
    intake_status: IntakeStatus | None       # promu pour le routage du graphe
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


# ── Garde-fou : validation du contenu du state ───────────────────────────────


def _scan_for_forbidden(value: object, breadcrumb: str) -> list[str]:
    """Retourne la liste des violations trouvées dans value (récursif).

    Interdit : octets bruts, objets fichier ouverts, chemins absolus.
    """
    violations: list[str] = []

    if isinstance(value, (bytes, bytearray)):
        violations.append(
            f"{breadcrumb} : contenu binaire interdit (bytes/{type(value).__name__})"
        )
    elif isinstance(value, io.IOBase):
        violations.append(
            f"{breadcrumb} : objet fichier ouvert interdit ({type(value).__name__})"
        )
    elif isinstance(value, str):
        if _ABSOLUTE_PATH_RE.match(value):
            violations.append(
                f"{breadcrumb} : chemin absolu interdit — {value!r}"
            )
    elif isinstance(value, dict):
        for k, v in value.items():
            violations.extend(_scan_for_forbidden(v, f"{breadcrumb}.{k}"))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            violations.extend(_scan_for_forbidden(item, f"{breadcrumb}[{i}]"))
    elif hasattr(value, "model_dump"):
        # Modèle Pydantic — inspecter sa représentation dict
        violations.extend(_scan_for_forbidden(value.model_dump(), breadcrumb))
    # datetime, int, float, bool, None, Enum → autorisés, non inspectés

    return violations


def validate_state_update(updates: dict) -> None:
    """Vérifie qu'une mise à jour du ClaimState ne contient pas de contenu interdit.

    Lève ValueError avec la liste des violations si du contenu binaire,
    un chemin absolu ou un objet fichier ouvert est détecté.

    À appeler par chaque nœud LangGraph avant de retourner ses mises à jour.
    """
    violations: list[str] = []
    for key, value in updates.items():
        # intake_input vidé (None) : ne pas inspecter son contenu précédent
        if key == "intake_input" and value is None:
            continue
        violations.extend(_scan_for_forbidden(value, key))

    if violations:
        raise ValueError(
            "Mise à jour du ClaimState refusée — contenu interdit détecté :\n"
            + "\n".join(f"  • {v}" for v in violations)
        )
