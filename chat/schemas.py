"""Schémas partagés du Chat Reasoning Agent (V2, plan de refonte §6, Phase V2-11a).

Package entièrement nouveau — aucun fichier existant touché (§0 du plan).
Ne réutilise jamais `graph.*`/`agents.*` directement : les seules données
métier qui transitent dans ce module proviennent de réponses HTTP déjà
minimisées de l'API v2 (`api.v2.schemas.ClaimStatusResponseV2`, jamais
revalidées comme un objet Pydantic dédié ici — un simple `dict` JSON déjà
sûr, voir `chat/tools.py`).
"""
from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from schemas.domain import ReaderRole, StrictModel
from schemas.v2_results import (
    DecisionAssumption,
    DecisionCounterfactual,
    DecisionFactor,
    EvidenceCompleteness,
    MissingInformation,
)

__all__ = [
    "AuditSummary",
    "ChatIntent",
    "ChatPlan",
    "ChatPlanAction",
    "CorrectionRecommendation",
    "ExplanationFacts",
    "LlmIntentDecision",
    "NluResult",
    "SimulationChangeRequest",
    "SimulationPatch",
    "SimulationPatchField",
    "SimulationResult",
]


class ChatIntent(str, Enum):
    """Intentions détectables par `chat/nlu.py` — 7 valeurs, alignées sur le
    plan de refonte V2 §6. Toutes détectées dès V2-11a (le NLU ne change
    jamais selon la sous-phase livrée) ; seules `ANALYZE`/`EXPLAIN`/`CORRECT`
    sont exécutables par `chat/planner.py` à ce stade — voir
    `chat.planner._SUPPORTED_INTENTS`. Un intent non encore livré ne
    déclenche jamais de tentative silencieuse, toujours une réponse
    « bientôt disponible » explicite."""

    ANALYZE = "ANALYZE"
    EXPLAIN = "EXPLAIN"
    SIMULATE = "SIMULATE"
    CORRECT = "CORRECT"
    AUDIT = "AUDIT"
    DRAFT_MESSAGE = "DRAFT_MESSAGE"
    CLARIFY_NEEDED = "CLARIFY_NEEDED"


class SimulationPatchField(str, Enum):
    """Liste blanche fermée des champs patchables par une simulation
    **ciblée** (Phase 9, plan de remédiation « autonomie décisionnelle V2 »,
    §7) — jamais un champ arbitraire. Restreinte aux champs d'éligibilité
    déjà calculés (`schemas.results.IdentityResult`/`CoverageResult`) :
    patcher ces champs ne nécessite jamais de retraiter un document (OCR/
    FHIR), contrairement à un changement d'acte/médicament — voir
    `chat/simulation_engine.py::run_targeted_simulation`."""

    IDENTITY_STATUS = "IDENTITY_STATUS"
    COVERAGE_STATUS = "COVERAGE_STATUS"
    CEILING_EXCEEDED = "CEILING_EXCEEDED"
    PREAUTHORIZATION_REQUIRED = "PREAUTHORIZATION_REQUIRED"


_STATUS_PATCH_FIELDS = frozenset(
    {SimulationPatchField.IDENTITY_STATUS, SimulationPatchField.COVERAGE_STATUS}
)
_BOOL_PATCH_FIELDS = frozenset(
    {SimulationPatchField.CEILING_EXCEEDED, SimulationPatchField.PREAUTHORIZATION_REQUIRED}
)
_VALID_STATUS_VALUES = frozenset({"PASS", "NEEDS_REVIEW", "FAIL"})
_VALID_BOOL_VALUES = frozenset({"true", "false"})


class SimulationPatch(StrictModel):
    """Un changement de champ atomique et borné — `field` fixe le type de
    valeur attendu (`value`, toujours une chaîne sérialisée : `PASS`/
    `NEEDS_REVIEW`/`FAIL` pour un statut, `true`/`false` pour un booléen).
    Rejeté à la validation si `value` n'est pas compatible avec `field` —
    jamais une valeur incohérente acceptée puis silencieusement ignorée à
    l'exécution."""

    field: SimulationPatchField
    value: str = Field(..., min_length=1, max_length=20)

    @model_validator(mode="after")
    def _value_matches_field(self) -> "SimulationPatch":
        if self.field in _STATUS_PATCH_FIELDS and self.value not in _VALID_STATUS_VALUES:
            raise ValueError(
                f"SimulationPatch.value doit être PASS/NEEDS_REVIEW/FAIL pour {self.field.value}, "
                f"reçu : {self.value!r}"
            )
        if self.field in _BOOL_PATCH_FIELDS and self.value.lower() not in _VALID_BOOL_VALUES:
            raise ValueError(
                f"SimulationPatch.value doit être true/false pour {self.field.value}, reçu : {self.value!r}"
            )
        return self


class SimulationChangeRequest(StrictModel):
    """Changement hypothétique demandé pour l'intention `SIMULATE` (plan V2
    §6, Phase V2-11b/Phase 9) — trois mécanismes bornés, jamais un contenu
    de document inventé (voir `chat/simulation_engine.py`, limite MVP
    assumée : le chat texte seul ne peut pas fournir un nouveau document).

    `remove_document`/`reader_role` (V2-11b) : simulation **complète**
    (réinvocation du graphe entier). `field_patches` (Phase 9) : simulation
    **ciblée** (réinvocation directe de
    `agents.autonomous_decision_agent.agent.run()`, jamais le graphe entier)
    — mutuellement exclusif avec les deux premiers (un patch de champ n'a de
    sens qu'à partir de l'état réel déjà calculé, jamais combiné à un
    changement de document/rôle qui nécessiterait de retraiter le dossier)."""

    remove_document: str | None = Field(default=None, min_length=1, max_length=200)
    reader_role: ReaderRole | None = None
    field_patches: list[SimulationPatch] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def _field_patches_are_exclusive(self) -> "SimulationChangeRequest":
        if self.field_patches and (self.remove_document is not None or self.reader_role is not None):
            raise ValueError(
                "field_patches est mutuellement exclusif avec remove_document/reader_role — "
                "une simulation ciblée ne peut pas être combinée à un changement de document/rôle."
            )
        return self


class LlmIntentDecision(StrictModel):
    """Sortie brute du LLM d'extraction d'intention (`chat/nlu.py`) —
    autorité réelle mais bornée : `intents` doit toujours contenir au moins
    une valeur (jamais une liste vide silencieuse, `CLARIFY_NEEDED` sert de
    valeur de repli explicite), `case_id` est optionnel et toujours revalidé
    par le pattern `CLM-\\d{4,}` (jamais une valeur inventée acceptée telle
    quelle si elle ne matche pas). `simulation_changes` : uniquement peuplé
    si `SIMULATE` est détecté, toujours borné aux deux opérations connues de
    `SimulationChangeRequest` — jamais un changement de champ arbitraire."""

    intents: list[ChatIntent] = Field(..., min_length=1)
    case_id: str | None = Field(default=None, pattern=r"^CLM-\d{4,}$")
    simulation_changes: SimulationChangeRequest | None = None
    reasoning: str = Field(default="", max_length=500)


class AuditSummary(StrictModel):
    """Résumé d'audit minimisé (`chat/audit_reader.py`, Phase V2-11c) —
    n'expose **jamais** le contenu brut d'un `outcome` (`schemas.audit.
    AuditEvent.outcome`, déjà rédigé/borné à 2000 caractères, mais dont
    l'exposition intégrale au LLM de composition serait redondante et
    risquée) : uniquement des compteurs, types d'événements et acteurs déjà
    connus, jamais un texte libre d'événement."""

    case_id: str = Field(..., pattern=r"^CLM-\d{4,}$")
    event_count: int = Field(ge=0)
    chain_intact: bool
    event_type_counts: dict[str, int] = Field(default_factory=dict)
    actors: list[str] = Field(default_factory=list)
    issues_count: int = Field(default=0, ge=0)


class NluResult(StrictModel):
    """Résultat normalisé de `chat/nlu.py` — toujours au moins un intent.

    `resolved_scenario_id` (Phase 8, plan de remédiation « autonomie
    décisionnelle V2 », §6.3) : identifiant de scénario résolu
    déterministiquement (jamais par le LLM) depuis
    `ConversationSemanticState.resolved_references`, quand le message
    référence explicitement un scénario déjà discuté ("le premier
    scénario"). `None` si aucune référence connue n'a été détectée dans le
    message, ou si aucun état sémantique n'était disponible."""

    intents: list[ChatIntent] = Field(..., min_length=1)
    case_id: str | None = Field(default=None, pattern=r"^CLM-\d{4,}$")
    simulation_changes: SimulationChangeRequest | None = None
    resolved_scenario_id: str | None = Field(default=None, pattern=r"^SCENARIO-\d+$")


class ChatPlanAction(str, Enum):
    """Décision de `chat/planner.py` — jamais laissée au LLM, calculée en
    Python pur à partir de `NluResult`."""

    EXECUTE = "EXECUTE"
    CLARIFY_NEEDED = "CLARIFY_NEEDED"
    NOT_YET_AVAILABLE = "NOT_YET_AVAILABLE"


class ChatPlan(StrictModel):
    """Plan d'exécution — `case_id` non `None` uniquement si `action ==
    EXECUTE` (précondition vérifiée par `chat/planner.py`, jamais reportée
    à l'appelant). `resolved_scenario_id` : simple passthrough de
    `NluResult.resolved_scenario_id` (Phase 8)."""

    action: ChatPlanAction
    intents: list[ChatIntent] = Field(default_factory=list)
    case_id: str | None = Field(default=None, pattern=r"^CLM-\d{4,}$")
    simulation_changes: SimulationChangeRequest | None = None
    unsupported_intents: list[ChatIntent] = Field(default_factory=list)
    resolved_scenario_id: str | None = Field(default=None, pattern=r"^SCENARIO-\d+$")


class ExplanationFacts(StrictModel):
    """Faits groundés extraits de `ClaimStatusResponseV2` (API v2) — jamais
    une nouvelle inférence, uniquement une réorganisation de champs déjà
    produits par le pipeline (`agents.autonomous_decision_agent`).

    `missing_information`/`assumptions`/`decisive_factors`/`counterfactuals`/
    `recommended_action`/`evidence_completeness` (plan de remédiation
    « autonomie décisionnelle V2 », Phase 7 — « explicabilité chat ») :
    réutilisent directement les schémas de `schemas.v2_results` (jamais une
    redéfinition dupliquée) — même source de vérité que la réponse API,
    jamais un nouveau calcul recomposé ici."""

    case_id: str
    final_decision: str | None = None
    decision_summary: list[str] = Field(default_factory=list)
    bounded_by: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)
    missing_information: list[MissingInformation] = Field(default_factory=list)
    assumptions: list[DecisionAssumption] = Field(default_factory=list)
    decisive_factors: list[DecisionFactor] = Field(default_factory=list)
    counterfactuals: list[DecisionCounterfactual] = Field(default_factory=list)
    recommended_action: str = ""
    evidence_completeness: EvidenceCompleteness | None = None


class CorrectionRecommendation(StrictModel):
    """Une recommandation de correction — `action` provient toujours de la
    table déterministe `chat.correction_engine._CORRECTION_TABLE`, jamais
    du LLM (voir `chat/correction_engine.py`)."""

    trigger: str = Field(..., min_length=1, max_length=500)
    action: str = Field(..., min_length=1, max_length=500)


class SimulationResult(StrictModel):
    """Résultat d'une simulation (`chat/simulation_engine.py`, Phase
    V2-11b) — `case_id` est toujours celui du dossier RÉEL (jamais
    l'identifiant synthétique interne utilisé pour isoler la simulation en
    stockage, qui n'est jamais exposé hors de `chat/simulation_engine.py`).
    `applied=False` signifie que la simulation n'a pas pu s'exécuter du
    tout (dossier introuvable, panne technique) — distinct de
    `decision_changed=False`, qui signifie qu'elle s'est bien exécutée mais
    n'a rien changé."""

    case_id: str = Field(..., pattern=r"^CLM-\d{4,}$")
    applied: bool
    original_decision: str | None = None
    simulated_decision: str | None = None
    decision_changed: bool = False
    simulated_reasons: list[str] = Field(default_factory=list)
    error: str | None = None
