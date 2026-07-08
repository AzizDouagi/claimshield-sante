"""Nœuds techniques du workflow LangGraph — ClaimShield Santé.

Ces nœuds gèrent les transitions d'état du workflow (quarantaine, revue
humaine, échec, finalisation). Ils ne sont PAS des agents métier :

  - Aucun appel LLM.
  - Aucun appel à un outil métier (OCR, FHIR, codes médicaux, stockage…).
  - Aucun accès direct au système de fichiers ou à la base de données.

Exception délibérée : ``node_await_human_review`` appelle
``langgraph.types.interrupt`` — un mécanisme natif du moteur LangGraph, pas
un outil métier — pour suspendre le graphe et attendre une décision humaine.

Responsabilités exclusives
--------------------------
- Mettre à jour ``current_step`` et ``completed_steps``.
- Enregistrer une alerte (non bloquante) ou une erreur (bloquante).
- Fixer ``final_recommendation`` uniquement pour les transitions terminales
  irréversibles (ex. : ``node_failure`` → ``Recommendation.REJECT``).

Distinction avec les nœuds agents (``graph/nodes.py``)
-------------------------------------------------------
| Critère                      | Nœud agent               | Nœud technique           |
|------------------------------|--------------------------|--------------------------|
| Construit via                | ``_make_node()``         | ``_make_technical_node`` |
| Produit un ``*_result``      | Oui                      | Non                      |
| Appelle un agent             | Oui                      | Non                      |
| Registre                     | ``NODE_REGISTRY``        | ``TECHNICAL_NODE_REGISTRY`` |
| Importe depuis ``agents/``   | Oui                      | Non                      |
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from graph.checkpoints import get_thread_id
from human_review.models import ReviewAction
from human_review.service import validate_and_audit_human_decision
from schemas.domain import Recommendation
from state.claim_state import ClaimState, validate_state_update


# ── Configuration immuable d'un nœud technique ───────────────────────────────


@dataclass(frozen=True)
class _TechnicalNodeConfig:
    """Paramètres stables d'un nœud de transition.

    Tous les champs sont lecture seule (frozen=True).  Le placeholder
    ``{case_id}`` est résolu au moment de l'exécution du nœud.
    """

    step_name: str
    """Nom du nœud : utilisé dans ``current_step`` et ``completed_steps``."""

    alert: str | None = None
    """Message d'alerte non bloquant.  Accepte ``{case_id}`` comme placeholder."""

    error: str | None = None
    """Message d'erreur bloquant.  Accepte ``{case_id}`` comme placeholder."""

    final_recommendation: Recommendation | None = None
    """Fixe ``final_recommendation`` pour les transitions terminales irréversibles."""


# ── Factory ───────────────────────────────────────────────────────────────────


def _make_technical_node(config: _TechnicalNodeConfig) -> Callable[[ClaimState], dict]:
    """Génère un nœud de transition pur à partir d'une configuration immuable.

    Le nœud produit **uniquement** des mises à jour de statut et de trace :
    ``current_step``, ``completed_steps``, ``alerts``, ``errors`` et
    ``final_recommendation``.  Aucun champ ``*_result`` n'est jamais écrit.
    """
    def _node(state: ClaimState) -> dict:
        case_id = str(state.get("case_id", "INCONNU"))
        updates: dict[str, Any] = {
            "current_step": config.step_name,
            "completed_steps": [config.step_name],
        }
        if config.alert is not None:
            updates["alerts"] = [config.alert.format(case_id=case_id)]
        if config.error is not None:
            updates["errors"] = [config.error.format(case_id=case_id)]
        if config.final_recommendation is not None:
            updates["final_recommendation"] = config.final_recommendation
        validate_state_update(updates)
        return updates

    _node.__name__ = f"node_{config.step_name}"
    _node.__qualname__ = f"node_{config.step_name}"
    return _node


# ── Nœuds techniques publics ──────────────────────────────────────────────────

node_quarantine = _make_technical_node(_TechnicalNodeConfig(
    step_name="quarantine",
    alert="[workflow] Dossier {case_id} mis en quarantaine — en attente de revue humaine.",
))
"""Enregistre la mise en quarantaine d'un dossier sans décision métier."""

node_needs_review = _make_technical_node(_TechnicalNodeConfig(
    step_name="needs_review",
    alert="[workflow] Dossier {case_id} transféré à la revue humaine.",
))
"""Marque le dossier comme nécessitant une intervention humaine."""

# ── Nœuds de convergence — parallélisation (Phase 2 du plan de remédiation) ──
#
# Purs marqueurs de synchronisation, sans alerte ni erreur : leur seul rôle
# est de servir de point d'arrivée commun à deux branches fanned-out par
# graph.edges.route_privacy_fan_out / route_coding_fan_out
# (graph.add_conditional_edges y renvoie une liste de deux noms de nœuds,
# exécutés dans le même superstep LangGraph). LangGraph n'exécute un nœud de
# convergence qu'une fois que TOUTES les branches programmées dans le
# superstep courant ont terminé — router les deux branches directement vers
# le nœud suivant (sans ce marqueur intermédiaire) risquerait de router sur
# un state partiel si une branche connaît sa propre route conditionnelle
# avant que l'autre ait fini. Les fonctions de routage conditionnelles
# réelles (route_verification_fan_in / route_result_consistency,
# graph/edges.py) ne s'exécutent qu'après ce marqueur, jamais avant.

node_verification_fan_in = _make_technical_node(_TechnicalNodeConfig(
    step_name="verification_fan_in",
))
"""Convergence après le fan-out document_ocr/fhir_validator — voir
``graph.edges.route_verification_fan_in`` pour la décision qui suit."""

node_consistency_fan_in = _make_technical_node(_TechnicalNodeConfig(
    step_name="consistency_fan_in",
))
"""Convergence après le fan-out clinical_consistency/fraud_detection — voir
``graph.edges.route_result_consistency`` pour la décision qui suit."""

# ── Nœud HITL — interruption LangGraph dynamique ─────────────────────────────

ALLOWED_HUMAN_ACTIONS: tuple[str, ...] = tuple(action.value for action in ReviewAction)
"""Actions que l'intervenant peut choisir en réponse à l'interruption —
dérivées directement de ``human_review.models.ReviewAction``
(``APPROVE``/``MODIFY``/``REJECT``/``RETRY``), source unique de vérité :
jamais recopiées en dur ici, pour ne jamais désynchroniser les deux.

``MODIFY`` est traité par ``graph.edges.route_human_review`` comme
``APPROVE`` : un chemin terminal (vers ``audit`` puis ``finalize``), jamais
un chemin de relance — il ne porte donc pas de ``target_node`` (même
validation que ``APPROVE``/``REJECT``, appliquée par
``human_review.models.HumanDecision``). ``RETRY`` (anciennement
``NEEDS_MORE_INFO``, renommé lors de l'unification des deux vocabulaires)
est la seule action à en exiger un — voir ``route_human_review``."""

# Champs *_result inspectés pour construire les preuves minimisées de l'interruption.
_EVIDENCE_RESULT_KEYS: tuple[str, ...] = (
    "intake_result",
    "security_result",
    "privacy_result",
    "ocr_result",
    "fhir_result",
    "identity_coverage_result",
    "coding_result",
    "clinical_result",
    "fraud_result",
    "review_result",
)


def _extract_status_marker(value: Any) -> str | None:
    """Extrait un statut/décision sous forme de chaîne, sans aucun contenu métier."""
    for attr in ("status", "decision"):
        marker = getattr(value, attr, None)
        if marker is not None:
            return str(getattr(marker, "value", marker))
    return None


def _collect_minimized_evidence(state: ClaimState) -> dict[str, str]:
    """Construit les preuves minimisées : statut/décision par résultat d'agent.

    Ne contient jamais de champ métier brut — uniquement les codes de statut
    déjà exposés par les schémas Pydantic des résultats d'agents.
    """
    evidence: dict[str, str] = {}
    for key in _EVIDENCE_RESULT_KEYS:
        marker = _extract_status_marker(state.get(key))
        if marker is not None:
            evidence[key] = marker
    return evidence


def _collect_motifs(state: ClaimState) -> list[str]:
    """Retourne les motifs de revue humaine — alertes puis erreurs, sans doublon."""
    combined: list[str] = [*state.get("alerts", []), *state.get("errors", [])]
    motifs: list[str] = list(dict.fromkeys(combined))
    return motifs or ["Revue humaine requise — aucun motif spécifique enregistré."]


def _collect_review_result_summary(state: ClaimState) -> dict[str, Any] | None:
    """Résumé minimisé de la synthèse ``case_reviewer_agent``, si disponible.

    Ne renvoie jamais le ``CaseReviewerResult`` complet (instance Pydantic) —
    un payload d'interruption doit rester JSON-natif et minimal : uniquement
    la pré-recommandation (toujours révisable), sa justification et les
    risques déjà calculés par l'agent. ``None`` si ``review_result`` n'a pas
    encore été produit pour ce dossier.
    """
    review_result = state.get("review_result")
    payload = getattr(review_result, "result_payload", None)
    if payload is None:
        return None
    recommendation = getattr(payload, "recommendation", None)
    return {
        "recommendation": str(getattr(recommendation, "value", recommendation))
        if recommendation is not None
        else None,
        "justification": list(getattr(payload, "justification", None) or []),
        "risks": list(getattr(payload, "risks", None) or []),
    }


def _extract_thread_id(config: Mapping[str, Any] | None) -> str | None:
    """Extrait le ``thread_id`` de la configuration LangGraph, si présente.

    Retourne ``None`` plutôt que de lever — un appel direct hors contexte de
    graphe (tests, ``config`` absent) ne doit jamais empêcher la construction
    du payload ; c'est ``interrupt()`` lui-même qui échoue dans ce cas
    (``RuntimeError``, aucun runtime de graphe disponible).
    """
    if config is None:
        return None
    try:
        return get_thread_id(config)
    except ValueError:
        return None


def _build_human_review_payload(
    state: ClaimState, *, thread_id: str | None = None
) -> dict[str, Any]:
    """Construit le payload transmis à ``interrupt()`` — aucune donnée brute.

    Contient ``case_id``, ``thread_id`` (identifiant de reprise — nécessaire
    pour retrouver le bon checkpoint via ``Command(resume=...)``), les
    motifs de revue, des preuves minimisées, un résumé de la synthèse
    ``case_reviewer_agent`` (``review_result``, voir
    ``_collect_review_result_summary``) et les actions autorisées.
    """
    return {
        "case_id": str(state.get("case_id", "INCONNU")),
        "thread_id": thread_id,
        "motifs": _collect_motifs(state),
        "preuves_minimisees": _collect_minimized_evidence(state),
        "review_result": _collect_review_result_summary(state),
        "actions_autorisees": list(ALLOWED_HUMAN_ACTIONS),
    }


def _collect_decision_evidence_ids(state: ClaimState) -> list[str]:
    """Identifiants de preuves déjà validées par ``case_reviewer_agent``,
    tracés dans l'audit de la décision humaine — jamais recalculés ici."""
    review_result = state.get("review_result")
    return list(getattr(review_result, "evidence_ids", None) or [])


def node_await_human_review(
    state: ClaimState, config: Optional[RunnableConfig] = None
) -> dict:
    """Suspend le graphe via ``interrupt()`` en attente d'une décision humaine.

    Le payload transmis à ``interrupt()`` contient ``case_id``, ``thread_id``
    (extrait de ``config`` — ``None`` si absent, ex. appel direct hors
    graphe), un résumé minimisé de ``review_result`` (pré-recommandation,
    justification, risques — voir ``_collect_review_result_summary``), les
    motifs de revue, des preuves minimisées (statuts/décisions déjà publics
    des agents, jamais de contenu brut) et les actions autorisées. La
    reprise se fait par ``app.invoke(Command(resume=decision), config=...)``
    en réutilisant obligatoirement le même ``thread_id`` que l'invocation
    initiale — sinon LangGraph ne retrouve pas le checkpoint en attente et
    redémarre le nœud depuis le début plutôt que de reprendre l'exécution
    interrompue.

    La décision reprise est validée par
    ``human_review.service.validate_and_audit_human_decision`` — le même
    modèle Pydantic strict (``human_review.models.HumanDecision``,
    ``extra="forbid"``, ``justification`` obligatoire) que le contrat
    framework-agnostique de ``human_review/`` : ce nœud n'a plus sa propre
    validation maison. ``case_id`` n'a pas besoin d'être fourni explicitement
    dans le payload de reprise — il est complété automatiquement depuis le
    state si absent (déjà connu du graphe, redondant à redemander à
    l'humain). Toute décision invalide (justification absente, action
    inconnue, ``target_node`` manquant/superflu...) lève
    ``HumanDecisionValidationError`` (sous-classe de ``ValueError``) : la
    fonction n'atteint jamais son ``return``, aucune mise à jour de state
    n'est produite, ``human_decision`` n'est jamais fixé, et
    ``graph.edges.route_human_review`` (qui route vers ``END``/``failure``/
    une relance) n'est donc jamais atteint. Le graphe ne peut jamais
    progresser vers ``END`` sur la base d'une décision invalide.

    La décision validée est à la fois stockée dans ``human_decision``
    (``model_dump(mode="json")`` — un dict JSON-natif, pas l'instance
    Pydantic) et **auditée** : l'``AuditEvent`` retourné par
    ``validate_and_audit_human_decision`` (action, justification tronquée,
    auteur, horodatage de la décision, preuves déjà validées par
    ``case_reviewer_agent``) est ajouté à ``audit_trail`` — jamais de
    document brut, de prompt complet ou de texte OCR complet.

    Pour une décision RETRY, incrémente ``correction_attempts`` —
    le compteur que ``graph.edges.route_human_review`` compare à une limite
    configurable (``Settings.claimshield_max_correction_attempts``) avant
    d'autoriser la relance vers ``human_decision.target_node``.
    """
    case_id = str(state.get("case_id", "INCONNU"))
    thread_id = _extract_thread_id(config)
    payload = _build_human_review_payload(state, thread_id=thread_id)
    raw_decision = interrupt(payload)

    if isinstance(raw_decision, Mapping) and "case_id" not in raw_decision:
        raw_decision = {**raw_decision, "case_id": case_id}

    evidence_ids = _collect_decision_evidence_ids(state)
    decision, audit_event = validate_and_audit_human_decision(
        raw_decision, evidence_ids=evidence_ids
    )

    updates: dict[str, Any] = {
        "current_step": "await_human_review",
        "completed_steps": ["await_human_review"],
        "human_decision": decision.model_dump(mode="json"),
        "audit_trail": [audit_event],
        "alerts": [
            f"[workflow] Décision humaine reçue pour {case_id} : {decision.action.value}"
        ],
    }
    if decision.action is ReviewAction.RETRY:
        updates["correction_attempts"] = int(state.get("correction_attempts", 0)) + 1
    validate_state_update(updates)
    return updates


node_failure = _make_technical_node(_TechnicalNodeConfig(
    step_name="failure",
    error="[workflow] Dossier {case_id} en échec — pipeline interrompu.",
    final_recommendation=Recommendation.REJECT,
))
"""Interrompt le pipeline et fixe la recommandation à REJECT."""

node_finalize = _make_technical_node(_TechnicalNodeConfig(
    step_name="finalize",
))
"""Clôt le pipeline sans modifier la recommandation validée avant cette étape."""


# ── Registre ──────────────────────────────────────────────────────────────────

TECHNICAL_NODE_REGISTRY: dict[str, Callable] = {
    "quarantine": node_quarantine,
    "needs_review": node_needs_review,
    "await_human_review": node_await_human_review,
    "failure": node_failure,
    "finalize": node_finalize,
    "verification_fan_in": node_verification_fan_in,
    "consistency_fan_in": node_consistency_fan_in,
}
