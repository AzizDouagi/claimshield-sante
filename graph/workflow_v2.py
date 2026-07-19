"""Workflow LangGraph V2 — graph/workflow_v2.py (plan de refonte V2, Phase V2-7).

Construit et compile le `StateGraph` complet du pipeline V2. Coexistence
stricte avec V1 (`graph/workflow.py`, non modifié — §0 du plan) : ce module
est entièrement nouveau, n'importe aucun symbole de `graph/workflow.py`.

Topologie
---------
::

    START ──► intake_safety
               ├─[continue]──► document_understanding ──► eligibility ──► medical_risk
               │                                                            ──► recovery
               │                                                            ──► autonomous_decision
               │                                                            ──► audit_service ──► finalize ──► END
               └─[terminal, BLOCKED/QUARANTINED/TECHNICAL_FAILURE]──► audit_service ──► finalize ──► END

Une seule branche conditionnelle dans tout le graphe (après `intake_safety`,
voir `graph.edges_v2.route_intake_safety`) — `document_understanding`,
`eligibility`, `medical_risk` ne bloquent jamais (confiance/statut
non-PASS = simple signal porté dans le résultat, jamais une branche de
routage) ; `audit_service` est toujours traversé avant `finalize`, sur les
deux chemins (aucun raccourci ne le contourne — même garantie que V1 après
la correction de l'étape 13, ici valable dès la conception).

`recovery` (Phase 6 du plan de remédiation « autonomie décisionnelle V2 »,
`graph.recovery_node_v2.make_recovery_node`) est un nœud technique unique,
jamais une boucle LangGraph : il tente, en interne, jusqu'à
`RecoveryPolicy.max_total_attempts` actions bornées de récupération
automatique (appel direct aux fonctions pures des agents, jamais un second
passage dans le graphe) avant que `autonomous_decision` ne synthétise le
dossier — désactivé entièrement si le dossier est déjà dans un état où une
récupération n'apporterait aucune valeur (voir
`graph.recovery_node_v2._recovery_disabled`).

Aucune interruption (`interrupt()`) : le graphe V2 ne bloque jamais sur une
décision humaine (décision AZIZ, plan V2 §0 — voir `services/override_store.py`
pour la correction post-décision, hors de ce graphe).
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

import agents.autonomous_decision_agent.agent as _autonomous_decision
import agents.document_understanding_agent.agent as _document_understanding
import agents.eligibility_agent.agent as _eligibility
import agents.intake_safety_agent.agent as _intake_safety
import agents.medical_risk_agent.agent as _medical_risk
from graph.edges_v2 import route_intake_safety
from graph.nodes_v2 import make_audit_service_node
from graph.recovery_node_v2 import RecoveryPolicy, make_recovery_node
from graph.technical_nodes_v2 import node_finalize_v2
from graph.topology import assert_workflow_topology_is_sound
from services.audit_service import AuditService
from state.claim_state_v2 import ClaimStateV2

__all__ = ["build_workflow_v2", "compile_workflow_v2", "get_workflow_v2_mermaid"]


def build_workflow_v2(
    *,
    audit_service: AuditService | None = None,
    recovery_policy: RecoveryPolicy | None = None,
) -> StateGraph:
    """Construit (sans compiler) le `StateGraph` V2 complet et vérifie sa
    topologie (réutilise `graph.topology.assert_workflow_topology_is_sound`,
    module partagé générique, sans dépendance V1)."""
    graph: StateGraph = StateGraph(ClaimStateV2)

    graph.add_node("intake_safety", _intake_safety.node)
    graph.add_node("document_understanding", _document_understanding.node)
    graph.add_node("eligibility", _eligibility.node)
    graph.add_node("medical_risk", _medical_risk.node)
    graph.add_node("recovery", make_recovery_node(policy=recovery_policy))
    graph.add_node("autonomous_decision", _autonomous_decision.node)
    graph.add_node("audit_service", make_audit_service_node(audit_service))
    graph.add_node("finalize", node_finalize_v2)

    graph.add_edge(START, "intake_safety")
    graph.add_conditional_edges(
        "intake_safety",
        route_intake_safety,
        {"continue": "document_understanding", "terminal": "audit_service"},
    )
    graph.add_edge("document_understanding", "eligibility")
    graph.add_edge("eligibility", "medical_risk")
    graph.add_edge("medical_risk", "recovery")
    graph.add_edge("recovery", "autonomous_decision")
    graph.add_edge("autonomous_decision", "audit_service")
    graph.add_edge("audit_service", "finalize")
    graph.add_edge("finalize", END)

    assert_workflow_topology_is_sound(graph)
    return graph


def compile_workflow_v2(
    checkpointer: Any = None,
    *,
    graph: StateGraph | None = None,
    audit_service: AuditService | None = None,
    recovery_policy: RecoveryPolicy | None = None,
    interrupt_before: list[str] | None = None,
) -> Any:
    """Compile un `StateGraph` V2 en graphe exécutable.

    Args:
        checkpointer: instance de checkpointer LangGraph (`InMemorySaver`,
            `SqliteSaver`…) — voir `graph/checkpoints.py` (réutilisé tel
            quel, non modifié).
        graph: `StateGraph` déjà construit (ex. via `build_workflow_v2()`).
            `None` → un graphe frais est construit avec `audit_service`/
            `recovery_policy`.
        audit_service: voir `build_workflow_v2` (ignoré si `graph` est fourni).
        recovery_policy: voir `build_workflow_v2` (ignoré si `graph` est
            fourni) — `None` retombe sur `graph.recovery_node_v2.DEFAULT_RECOVERY_POLICY`.
        interrupt_before: toujours vide en pratique — le graphe V2 ne
            comporte aucun nœud d'interruption, laissé pour compatibilité de
            signature avec `graph.workflow.compile_workflow` (V1).

    Returns:
        `CompiledStateGraph` prêt pour `.invoke()`/`.stream()`/`.get_state()`.
    """
    if graph is None:
        graph = build_workflow_v2(audit_service=audit_service, recovery_policy=recovery_policy)
    return graph.compile(checkpointer=checkpointer, interrupt_before=interrupt_before or [])


def get_workflow_v2_mermaid(app: Any = None) -> str:
    """Représentation Mermaid du graphe V2 compilé (texte, aucune donnée
    sensible) — `app=None` compile un workflow par défaut."""
    compiled = app if app is not None else compile_workflow_v2()
    return compiled.get_graph().draw_mermaid()
