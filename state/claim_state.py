"""État partagé du workflow LangGraph — ClaimShield Santé.

ClaimState est le seul objet qui traverse tous les nœuds du graphe.
Règles :
- Minimal : pas de texte OCR brut, pas de contenu PDF, pas de secrets.
- Append-only pour les listes (reducers operator.add) — LangGraph ne permet
  pas d'écraser une liste accumulée sans reducer explicite.
- Sérialisable JSON (tous les types sont des primitives ou des dict Pydantic).
- Versionné : schema_version permet de détecter un state produit par une
  version antérieure du workflow lors d'une reprise sur checkpoint.

Contenu autorisé après ingestion / OCR :
  case_id · statut ingestion · manifest structuré · métadonnées documents
  chemins relatifs · hashes SHA-256 · alertes · erreurs
  résultats structurés d'agent (Pydantic) · métadonnées de confiance
  références artefacts (artifact_id, artifact_path) · codes raison

Contenu interdit (garanti par validate_state_update) :
  octets bruts · PDF/images en base64 · texte OCR complet (full_text)
  pages OCR brutes (pages[]) · chemins absolus · objets fichier ouverts

Règles spécifiques à l'agent Document/OCR :
  - ocr_result doit être le résultat MINIMISÉ produit par _minimize_for_state().
    full_text et pages doivent être vides avant d'entrer dans le state.
  - ocr_input est consommé et remis à None par le nœud OCR.
  - Le texte OCR complet est écrit dans un artefact externe (artifact_id / artifact_path).
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

    # ── Entrée OCR — consommée et vidée par le nœud document_ocr_agent ───────
    # Contient claim_id, document_id, filename, mime_type, sha256, sanitized_path,
    # security_decision, schema_version, file_index.
    # Le nœud OCR le remet à None après traitement.
    # Invariant : jamais de source_path absolu ni de contenu binaire dans ce champ.
    ocr_input: dict | None

    # ── Entrée identité/couverture — consommée par le nœud identity_coverage ──
    # Contient case_id + fhir_bundle_path (chemin relatif sous incoming/).
    # Les données d'extraction proviennent de ocr_result dans le state.
    identity_coverage_input: dict | None

    # ── Entrée FHIR — consommée par le nœud fhir_validator ───────────────────
    # Contient case_id, fhir_bundle_path (relatif), bundle_expected.
    fhir_input: dict | None

    # ── Entrée codification — consommée par le nœud medical_coding ───────────
    # Contient case_id, procedures (list[str]), medications (list[str]).
    coding_input: dict | None

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


def _check_ocr_result_minimized(result: DocumentOcrResult, breadcrumb: str) -> list[str]:
    """Vérifie que DocumentOcrResult a été minimisé avant d'entrer dans le state.

    Le texte OCR complet et les pages brutes doivent avoir été retirés par
    _minimize_for_state() dans le nœud OCR. Seuls les résultats structurés
    (métadonnées, champs extraits, scores, codes, audit_entry) sont autorisés.

    Violations détectées :
      - full_text non vide → document brut interdit dans le state
      - pages non vide → pages OCR brutes interdites dans le state
      - extraction.full_text non vide → même règle dans la vue détaillée
      - extraction.pages non vide → même règle dans la vue détaillée
    """
    violations: list[str] = []
    if result.full_text:
        violations.append(
            f"{breadcrumb}.full_text : texte OCR complet interdit dans le state "
            f"({len(result.full_text)} caractères) — utiliser _minimize_for_state()"
        )
    if result.pages:
        violations.append(
            f"{breadcrumb}.pages : pages OCR brutes interdites dans le state "
            f"({len(result.pages)} page(s)) — utiliser _minimize_for_state()"
        )
    if result.extraction is not None:
        ext = result.extraction
        if ext.full_text:
            violations.append(
                f"{breadcrumb}.extraction.full_text : texte OCR complet interdit "
                f"({len(ext.full_text)} caractères)"
            )
        if ext.pages:
            violations.append(
                f"{breadcrumb}.extraction.pages : pages OCR brutes interdites "
                f"({len(ext.pages)} page(s))"
            )
    return violations


# Clés consommées (vidées à None) qui sont ignorées dans validate_state_update
_CONSUMED_INPUT_KEYS: frozenset[str] = frozenset({
    "intake_input",
    "security_input",
    "privacy_input",
    "ocr_input",
    "identity_coverage_input",
    "fhir_input",
    "coding_input",
})


def validate_state_update(updates: dict) -> None:
    """Vérifie qu'une mise à jour du ClaimState ne contient pas de contenu interdit.

    Règles appliquées :
      1. Contenu binaire (bytes, bytearray) → interdit.
      2. Objets fichier ouverts (io.IOBase) → interdit.
      3. Chemins absolus dans les chaînes → interdit.
      4. DocumentOcrResult non minimisé (full_text ou pages non vides) → interdit.

    Les clés d'entrée consommées (intake_input, security_input, privacy_input,
    ocr_input) vidées à None sont ignorées sans inspection.

    Lève ValueError avec la liste des violations détectées.
    À appeler par chaque nœud LangGraph avant de retourner ses mises à jour.
    """
    violations: list[str] = []
    for key, value in updates.items():
        if key in _CONSUMED_INPUT_KEYS and value is None:
            continue
        if key == "ocr_result" and isinstance(value, DocumentOcrResult):
            violations.extend(_check_ocr_result_minimized(value, key))
        violations.extend(_scan_for_forbidden(value, key))

    if violations:
        raise ValueError(
            "Mise à jour du ClaimState refusée — contenu interdit détecté :\n"
            + "\n".join(f"  • {v}" for v in violations)
        )
