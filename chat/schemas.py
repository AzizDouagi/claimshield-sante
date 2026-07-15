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

from pydantic import Field

from schemas.domain import ReaderRole, StrictModel

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


class SimulationChangeRequest(StrictModel):
    """Changement hypothétique demandé pour l'intention `SIMULATE` (plan V2
    §6, Phase V2-11b) — deux opérations bornées seulement, jamais un contenu
    de document inventé (voir `chat/simulation_engine.py`, limite MVP
    assumée : le chat texte seul ne peut pas fournir un nouveau document).

    `remove_document` : retire un document déjà accepté du dossier réel de
    la simulation, par mot-clé insensible à la casse recherché dans son nom
    original (ex. "ordonnance"). `reader_role` : simule avec un rôle de
    lecture différent de celui de la soumission réelle."""

    remove_document: str | None = Field(default=None, min_length=1, max_length=200)
    reader_role: ReaderRole | None = None


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
    """Résultat normalisé de `chat/nlu.py` — toujours au moins un intent."""

    intents: list[ChatIntent] = Field(..., min_length=1)
    case_id: str | None = Field(default=None, pattern=r"^CLM-\d{4,}$")
    simulation_changes: SimulationChangeRequest | None = None


class ChatPlanAction(str, Enum):
    """Décision de `chat/planner.py` — jamais laissée au LLM, calculée en
    Python pur à partir de `NluResult`."""

    EXECUTE = "EXECUTE"
    CLARIFY_NEEDED = "CLARIFY_NEEDED"
    NOT_YET_AVAILABLE = "NOT_YET_AVAILABLE"


class ChatPlan(StrictModel):
    """Plan d'exécution — `case_id` non `None` uniquement si `action ==
    EXECUTE` (précondition vérifiée par `chat/planner.py`, jamais reportée
    à l'appelant)."""

    action: ChatPlanAction
    intents: list[ChatIntent] = Field(default_factory=list)
    case_id: str | None = Field(default=None, pattern=r"^CLM-\d{4,}$")
    simulation_changes: SimulationChangeRequest | None = None
    unsupported_intents: list[ChatIntent] = Field(default_factory=list)


class ExplanationFacts(StrictModel):
    """Faits groundés extraits de `ClaimStatusResponseV2` (API v2) — jamais
    une nouvelle inférence, uniquement une réorganisation de champs déjà
    produits par le pipeline (`agents.autonomous_decision_agent`)."""

    case_id: str
    final_decision: str | None = None
    decision_summary: list[str] = Field(default_factory=list)
    bounded_by: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)


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
