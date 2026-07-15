"""Schémas de sortie des 5 agents V2 — schemas/v2_results.py.

Réutilise par import les sous-schémas déjà validés de `schemas/results.py`
(`ProcedureCoding`, `ClinicalSignal`, `ClinicalInconsistency`, `FraudSignal`,
`SecurityFinding`, `DocumentExtraction`, `IdentityResult`, `CoverageResult`,
`ClaimManifest`, `StructuredError`, `LlmMetadata`, `DisagreementPoint`) —
jamais dupliqués. Chaque enveloppe V2 documente en docstring le(s) schéma(s)
V1 dont elle dérive (Phase V2-1 du plan de refonte).

Ne modifie aucun fichier V1. Les deux nouveaux enums nécessaires
(`ClaimDecisionV2`, `IntakeSafetyStatus`) ont été ajoutés — de façon
strictement additive — en fin de `schemas/domain.py` (§0 du plan, point
d'intégration autorisé), importés ici comme n'importe quel autre enum du
domaine.
"""
from __future__ import annotations

import re
from enum import Enum

from pydantic import Field, field_validator, model_validator

from schemas.domain import (
    ClaimDecisionV2,
    IntakeSafetyStatus,
    StrictModel,
    VerificationStatus,
)
from schemas.results import (
    ClaimManifest,
    ClinicalInconsistency,
    ClinicalSignal,
    CoverageResult,
    DisagreementPoint,
    DocumentExtraction,
    FraudSignal,
    IdentityResult,
    LlmMetadata,
    ProcedureCoding,
    SecurityFinding,
    StructuredError,
)

__all__ = [
    "AutonomousDecisionResult",
    "DocumentUnderstandingResult",
    "EligibilityResult",
    "EvidenceCompleteness",
    "IntakeSafetyResult",
    "MedicalRiskResult",
    "MedicalRiskResultPayload",
    "RiskLevel",
]

# ── Validateur anti-fuite partagé ──────────────────────────────────────────────
#
# Dupliqué volontairement (jamais importé depuis schemas.results, module privé
# non destiné à l'export cross-fichier) — même convention que
# schemas.audit._reject_security_leak, qui documente explicitement ce choix :
# schemas/ est une couche basse, chaque fichier de schéma reste autonome.

_ABSOLUTE_PATH_RE = re.compile(r"^(?:/|[A-Za-z]:[/\\]|\\\\)")
_SECRET_HINT_RE = re.compile(
    r"(?:api[_-]?key|secret\s*[:=]|password\s*[:=]|token\s*[:=]|bearer\s+[a-z0-9._~+/=-]+)",
    re.IGNORECASE,
)
_MAX_NEWLINES_IN_STRUCTURED_FIELD = 2


def _reject_unstructured_content(value: str, field_name: str) -> str:
    """Interdit chemin absolu, marqueur de secret et contenu multi-lignes —
    jamais un document brut, un texte OCR complet ou un prompt complet dans
    un champ de motif/justification structuré."""
    if _ABSOLUTE_PATH_RE.match(value):
        raise ValueError(f"Chemin absolu interdit dans {field_name}")
    if _SECRET_HINT_RE.search(value):
        raise ValueError(f"Secret potentiel interdit dans {field_name}")
    if value.count("\n") > _MAX_NEWLINES_IN_STRUCTURED_FIELD:
        raise ValueError(
            f"Contenu multi-lignes interdit dans {field_name} — jamais un document "
            "brut, un texte OCR complet ou un prompt complet."
        )
    return value


# ── 1. intake_safety_agent ──────────────────────────────────────────────────────


class IntakeSafetyResult(StrictModel):
    """Fusion de `ClaimIntakeResult` + `SecurityGateResult` (V1) — un seul
    appel LLM (voir agents/intake_safety_agent/agent.py, plan V2 Phase V2-2).

    Dérive de : `schemas.results.ClaimIntakeResult` (manifeste) +
    `schemas.results.SecurityGateResult` (findings, décision de sécurité) —
    combinés ici en une seule sortie d'agent V2.
    """

    case_id: str = Field(..., pattern=r"^CLM-\d{4,}$")
    status: IntakeSafetyStatus
    manifest: ClaimManifest | None = None
    security_findings: list[SecurityFinding] = Field(default_factory=list)
    reasons: list[str] = Field(..., min_length=1)
    errors: list[StructuredError] = Field(default_factory=list)
    llm_trace: LlmMetadata

    @field_validator("reasons")
    @classmethod
    def _reasons_no_raw_content(cls, v: list[str]) -> list[str]:
        return [_reject_unstructured_content(item, "reasons") for item in v]


# ── 2. document_understanding_agent ─────────────────────────────────────────────


class DocumentUnderstandingResult(StrictModel):
    """Fusion de `DocumentOcrResult`/`DocumentExtraction` + `FhirValidatorResult`
    (V1) + vue privacy minimisée (`services.privacy_service`, V2, appelée en
    Phase A) — un seul appel LLM (plan V2 Phase V2-3).

    `privacy_view` est un dict JSON-sérialisable déjà minimisé/pseudonymisé
    (voir `services.privacy_service.PrivacyViewResult.view`) — jamais un
    objet Pydantic dédié : sa structure exacte dépend du rôle, déjà validée
    en amont par `PrivacyService`, non revalidée ici.
    """

    case_id: str = Field(..., pattern=r"^CLM-\d{4,}$")
    status: VerificationStatus
    extraction: DocumentExtraction | None = None
    fhir_summary: dict | None = Field(
        default=None,
        description=(
            "Résumé minimisé de la validation FHIR (status/resource_count/"
            "resource_types) — jamais le bundle FHIR brut."
        ),
    )
    privacy_view: dict | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    errors: list[StructuredError] = Field(default_factory=list)
    llm_trace: LlmMetadata

    @field_validator("reasons")
    @classmethod
    def _reasons_no_raw_content(cls, v: list[str]) -> list[str]:
        return [_reject_unstructured_content(item, "reasons") for item in v]


# ── 3. eligibility_agent ─────────────────────────────────────────────────────────


class EligibilityResult(StrictModel):
    """Porte quasi 1:1 de `schemas.results.IdentityCoverageResult` (V1) —
    renommage seul, logique métier inchangée (plan V2 Phase V2-4).

    `coverage_data_available` (ajouté post-mesure V2-10, AZIZ) distingue
    explicitement « aucune donnée de police/contrat n'a jamais été fournie »
    de « la couverture a été évaluée et jugée invalide » — nécessaire pour
    qu'`autonomous_decision_agent` ne confonde plus une absence de donnée
    avec un risque réel (voir `agents/eligibility_agent/agent.py::run()`)."""

    case_id: str = Field(..., pattern=r"^CLM-\d{4,}$")
    status: VerificationStatus
    identity: IdentityResult
    coverage: CoverageResult
    coverage_data_available: bool = True
    rule_version: str = "1.0.0"
    reasons: list[str] = Field(default_factory=list)
    errors: list[StructuredError] = Field(default_factory=list)
    llm_trace: LlmMetadata

    @field_validator("reasons")
    @classmethod
    def _reasons_no_raw_content(cls, v: list[str]) -> list[str]:
        return [_reject_unstructured_content(item, "reasons") for item in v]


# ── 4. medical_risk_agent ────────────────────────────────────────────────────────


class RiskLevel(str, Enum):
    """Niveau de risque dérivé déterministiquement de `risk_score` — jamais
    choisi librement par le LLM (voir agents/medical_risk_agent/agent.py,
    même patron de bornage que P1-1/P1-2 en V1).

    `CRITICAL` ajouté post-mesure V2-10 (AZIZ) : réservé aux signaux de
    danger *réel* et confirmé (doublon exact de facture, ou score de risque
    réel au-delà du seuil HIGH) — seul niveau qui plafonne directement à
    QUARANTINE dans `autonomous_decision_agent`. `risk_level` n'est plus
    calculé à partir de TOUS les signaux (voir `EvidenceCompleteness`, qui
    porte désormais les signaux de données manquantes)."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EvidenceCompleteness(str, Enum):
    """Complétude des preuves disponibles pour l'évaluation — axe distinct
    et indépendant de `RiskLevel` (ajouté post-mesure V2-10, AZIZ, suite au
    constat que des données manquantes — codification jamais tentée,
    couverture non vérifiable — étaient auparavant comptées comme du risque
    réel et plafonnaient systématiquement à QUARANTINE).

    Dérivé des signaux `IDENTITY_AMBIGUOUS`/`UNRESOLVED_CODING`/
    `LOW_EXTRACTION_CONFIDENCE`/`PREAUTHORIZATION_MISSING` (voir
    `agents/medical_risk_agent/agent.py::_COMPLETENESS_SIGNAL_TYPES`) —
    jamais des signaux de danger réel (`RiskLevel`)."""

    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"


class MedicalRiskResultPayload(StrictModel):
    """Détail métier fusionné : codification + cohérence clinique + fraude.

    Dérive de : `schemas.results.MedicalCodingResult.codings`
    (`ProcedureCoding`), `schemas.results.ClinicalResultPayload`
    (`signals`/`inconsistencies`/compteurs), `schemas.results.FraudResultPayload`
    (`signals` fraude/`risk_score`/`duplicate_invoice`) — réunis en un seul
    détail métier V2, calculé par une seule Phase A puis vérifié par un seul
    appel LLM (plan V2 Phase V2-5, fallback conditionnel V2-5-bis documenté
    dans le plan si la fusion dégrade la qualité mesurée).
    """

    procedure_count: int | None = Field(default=None, ge=0)
    medication_count: int | None = Field(default=None, ge=0)
    codings: list[ProcedureCoding] = Field(default_factory=list)
    clinical_signals: list[ClinicalSignal] = Field(default_factory=list)
    clinical_inconsistencies: list[ClinicalInconsistency] = Field(default_factory=list)
    fraud_signals: list[FraudSignal] = Field(default_factory=list)
    duplicate_invoice: bool | None = None
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_level: RiskLevel = RiskLevel.LOW
    evidence_completeness: EvidenceCompleteness = EvidenceCompleteness.COMPLETE
    threshold_version: str = "1.0.0"
    reasons: list[str] = Field(default_factory=list)

    @field_validator("reasons")
    @classmethod
    def _reasons_no_raw_content(cls, v: list[str]) -> list[str]:
        return [_reject_unstructured_content(item, "reasons") for item in v]


class MedicalRiskResult(StrictModel):
    """Même patron d'enveloppe générique que `ClinicalConsistencyResult`/
    `FraudDetectionResult` (V1) : `status`/`llm_trace`/`confidence`/`errors`/
    `evidence_ids`/`human_review_required` communs, `result_payload` pour le
    détail métier fusionné (voir `MedicalRiskResultPayload`).
    """

    case_id: str = Field(..., pattern=r"^CLM-\d{4,}$")
    status: VerificationStatus
    llm_trace: LlmMetadata
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    errors: list[StructuredError] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    human_review_required: bool = False
    result_payload: MedicalRiskResultPayload = Field(default_factory=MedicalRiskResultPayload)

    @model_validator(mode="after")
    def _evidence_ids_must_reference_real_evidence(self) -> "MedicalRiskResult":
        real_ids = (
            {
                evidence.evidence_id
                for signal in self.result_payload.clinical_signals
                for evidence in signal.evidence
            }
            | {
                evidence.evidence_id
                for inconsistency in self.result_payload.clinical_inconsistencies
                for evidence in inconsistency.evidence
            }
            | {
                evidence.evidence_id
                for signal in self.result_payload.fraud_signals
                for evidence in signal.evidence
            }
        )
        unknown = [i for i in self.evidence_ids if i not in real_ids]
        if unknown:
            raise ValueError(
                f"evidence_ids référence des preuves inexistantes : {unknown} — "
                "jamais un identifiant inventé."
            )
        return self


# ── 5. autonomous_decision_agent ─────────────────────────────────────────────────


class AutonomousDecisionResult(StrictModel):
    """Décision finale bornée — remplace `schemas.results.CaseReviewerResult` (V1).

    Contrairement à `CaseReviewerResult`, `status`/`decision` NE sont PAS
    verrouillés à `NEEDS_REVIEW`/`True` — la V2 supprime la revue humaine
    obligatoire (décision AZIZ « override asynchrone optionnel », voir
    `services/override_store.py` pour la correction post-décision). Les
    bornes qui contraignent réellement `decision` (ex. `risk_level == HIGH`
    plafonne à `{REJECT, QUARANTINE, REQUEST_MORE_INFO}`) sont appliquées en
    Python par `agents/autonomous_decision_agent/agent.py`, jamais dans ce
    schéma — `bounded_by` liste les garde-fous effectivement appliqués, pour
    traçabilité et pour le Chat Reasoning Agent (Phase V2-11a, « explique-moi
    pourquoi »), jamais un motif inventé.
    """

    case_id: str = Field(..., pattern=r"^CLM-\d{4,}$")
    status: VerificationStatus
    decision: ClaimDecisionV2
    justification: list[str] = Field(default_factory=list)
    disagreements: list[DisagreementPoint] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    bounded_by: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    errors: list[StructuredError] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    llm_trace: LlmMetadata

    @field_validator("justification", "risks", "bounded_by")
    @classmethod
    def _no_raw_content(cls, v: list[str], info) -> list[str]:
        return [_reject_unstructured_content(item, info.field_name) for item in v]
