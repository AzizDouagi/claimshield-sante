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
from datetime import UTC, datetime
from typing import Any, Callable, Mapping

from langgraph.types import interrupt

from schemas.domain import Recommendation
from state.claim_state import ClaimState, HumanDecision, validate_state_update


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

# ── Nœud HITL — interruption LangGraph dynamique ─────────────────────────────

ALLOWED_HUMAN_ACTIONS: tuple[str, ...] = ("APPROVE", "REJECT", "NEEDS_MORE_INFO")
"""Actions que l'intervenant peut choisir en réponse à l'interruption."""

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


def _build_human_review_payload(state: ClaimState) -> dict[str, Any]:
    """Construit le payload transmis à ``interrupt()`` — aucune donnée brute."""
    return {
        "case_id": str(state.get("case_id", "INCONNU")),
        "motifs": _collect_motifs(state),
        "preuves_minimisees": _collect_minimized_evidence(state),
        "actions_autorisees": list(ALLOWED_HUMAN_ACTIONS),
    }


def _validate_human_decision(raw: Any) -> HumanDecision:
    """Valide la décision fournie à la reprise via ``Command(resume=...)``.

    Lève ``ValueError`` si la décision est absente, mal formée ou hors du
    périmètre autorisé — la reprise n'accepte qu'une décision validée.
    """
    if not isinstance(raw, Mapping):
        raise ValueError(
            "Décision humaine invalide : un mapping est attendu en reprise "
            f"(reçu {type(raw).__name__})"
        )

    actor = raw.get("actor")
    if not isinstance(actor, str) or not actor.strip():
        raise ValueError("Décision humaine invalide : champ 'actor' obligatoire et non vide.")

    decision = raw.get("decision")
    if decision not in ALLOWED_HUMAN_ACTIONS:
        raise ValueError(
            f"Décision humaine invalide : {decision!r} — attendu l'une de "
            f"{ALLOWED_HUMAN_ACTIONS}"
        )

    target_node = raw.get("target_node")
    if decision == "NEEDS_MORE_INFO":
        if not isinstance(target_node, str) or not target_node.strip():
            raise ValueError(
                "Décision humaine invalide : 'target_node' obligatoire et non vide "
                "pour NEEDS_MORE_INFO — le nœud à relancer doit être explicite."
            )
    elif target_node is not None:
        raise ValueError(
            "Décision humaine invalide : 'target_node' n'est autorisé qu'avec "
            "NEEDS_MORE_INFO."
        )

    validated: HumanDecision = {
        "actor": actor.strip(),
        "decision": decision,
        "decided_at": raw.get("decided_at") or datetime.now(UTC),
    }
    if decision == "NEEDS_MORE_INFO":
        validated["target_node"] = target_node.strip()
    comment = raw.get("comment")
    if isinstance(comment, str) and comment.strip():
        validated["comment"] = comment.strip()
    return validated


def node_await_human_review(state: ClaimState) -> dict:
    """Suspend le graphe via ``interrupt()`` en attente d'une décision humaine.

    Le payload transmis à ``interrupt()`` contient ``case_id``, les motifs de
    revue, des preuves minimisées (statuts/décisions déjà publics des agents,
    jamais de contenu brut) et les actions autorisées. La reprise se fait par
    ``app.invoke(Command(resume=decision), config=...)`` en réutilisant
    obligatoirement le même ``thread_id`` que l'invocation initiale — sinon
    LangGraph ne retrouve pas le checkpoint en attente et redémarre le nœud
    depuis le début plutôt que de reprendre l'exécution interrompue.

    Pour une décision NEEDS_MORE_INFO, incrémente ``correction_attempts`` —
    le compteur que ``graph.edges.route_human_review`` compare à une limite
    configurable (``Settings.claimshield_max_correction_attempts``) avant
    d'autoriser la relance vers ``human_decision.target_node``.
    """
    case_id = str(state.get("case_id", "INCONNU"))
    payload = _build_human_review_payload(state)
    raw_decision = interrupt(payload)
    decision = _validate_human_decision(raw_decision)
    decision_label = decision.get("decision", "INCONNU")

    updates: dict[str, Any] = {
        "current_step": "await_human_review",
        "completed_steps": ["await_human_review"],
        "human_decision": decision,
        "alerts": [
            f"[workflow] Décision humaine reçue pour {case_id} : {decision_label}"
        ],
    }
    if decision_label == "NEEDS_MORE_INFO":
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
}
