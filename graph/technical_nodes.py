"""Nœuds techniques du workflow LangGraph — ClaimShield Santé.

Ces nœuds gèrent les transitions d'état du workflow (quarantaine, revue
humaine, échec, finalisation). Ils ne sont PAS des agents métier :

  - Aucun appel LLM.
  - Aucun appel à un outil métier (OCR, FHIR, codes médicaux, stockage…).
  - Aucun accès direct au système de fichiers ou à la base de données.

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
from typing import Any, Callable

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

node_failure = _make_technical_node(_TechnicalNodeConfig(
    step_name="failure",
    error="[workflow] Dossier {case_id} en échec — pipeline interrompu.",
    final_recommendation=Recommendation.REJECT,
))
"""Interrompt le pipeline et fixe la recommandation à REJECT."""

node_finalize = _make_technical_node(_TechnicalNodeConfig(
    step_name="finalize",
))
"""Clôt le pipeline sans modifier la recommandation (déjà fixée par case_reviewer)."""


# ── Registre ──────────────────────────────────────────────────────────────────

TECHNICAL_NODE_REGISTRY: dict[str, Callable] = {
    "quarantine": node_quarantine,
    "needs_review": node_needs_review,
    "failure": node_failure,
    "finalize": node_finalize,
}
