"""Vérification automatique de la topologie d'un ``StateGraph`` LangGraph.

Module partagé, extrait de ``graph/workflow.py`` (V1) pour permettre à
``graph/workflow_v2.py`` de vérifier la topologie du graphe V2 sans dépendre
de la chaîne d'imports V1 (``graph.nodes``, ``graph.edges``,
``graph.technical_nodes``, ``orchestrator.*``). Fonctions pures, sans aucune
dépendance métier — n'opèrent que sur l'objet ``StateGraph`` fourni.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

__all__ = [
    "find_dangling_transitions",
    "find_isolated_nodes",
    "find_unreachable_nodes",
    "find_dead_end_nodes",
    "assert_workflow_topology_is_sound",
]


def _direct_destinations(graph: StateGraph, node: str) -> set[str]:
    """Destinations directes d'un nœud : arêtes normales + branches conditionnelles."""
    destinations = {dst for src, dst in graph.edges if src == node}
    for branch in graph.branches.get(node, {}).values():
        destinations.update((branch.ends or {}).values())
    return destinations


def find_dangling_transitions(graph: StateGraph) -> list[tuple[str, str]]:
    """Transitions (source, destination) dont la destination n'existe pas.

    Une destination valide est soit ``END``, soit un nœud enregistré via
    ``add_node``. Détecte une faute de frappe ou un nœud oublié dans
    ``add_edge`` / ``add_conditional_edges``.
    """
    known_destinations = set(graph.nodes) | {END}
    dangling: list[tuple[str, str]] = []
    for src, dst in graph.edges:
        if dst not in known_destinations:
            dangling.append((src, dst))
    for src, branches in graph.branches.items():
        for branch in branches.values():
            for dst in (branch.ends or {}).values():
                if dst not in known_destinations:
                    dangling.append((src, dst))
    return dangling


def find_isolated_nodes(graph: StateGraph) -> set[str]:
    """Nœuds enregistrés sans aucune entrée ni sortie valide.

    Un nœud valide a au moins une arête entrante (une autre transition le
    désigne comme destination) ou sortante (il désigne lui-même au moins une
    destination, arête normale ou branche conditionnelle). Un nœud sans l'un
    ni l'autre est câblé via ``add_node`` mais jamais raccordé au graphe —
    plus strict que ``find_unreachable_nodes`` : un nœud avec uniquement une
    sortie (mais aucune entrée) n'est pas isolé ici, alors qu'il resterait
    inaccessible depuis START.
    """
    has_incoming: set[str] = set()
    has_outgoing: set[str] = set()
    for src, dst in graph.edges:
        has_outgoing.add(src)
        has_incoming.add(dst)
    for src, branches in graph.branches.items():
        for branch in branches.values():
            destinations = (branch.ends or {}).values()
            if destinations:
                has_outgoing.add(src)
            has_incoming.update(destinations)
    connected = has_incoming | has_outgoing
    return set(graph.nodes) - connected


def find_unreachable_nodes(graph: StateGraph) -> set[str]:
    """Nœuds enregistrés jamais atteignables depuis START (parcours en largeur).

    Un nœud inaccessible est un nœud mort : câblé via ``add_node`` mais
    jamais raccordé par une arête ou une branche conditionnelle amont.
    """
    visited: set[str] = set()
    frontier: list[str] = [START]
    while frontier:
        current = frontier.pop()
        for dest in _direct_destinations(graph, current):
            if dest == END or dest in visited:
                continue
            visited.add(dest)
            frontier.append(dest)
    return set(graph.nodes) - visited


def find_dead_end_nodes(graph: StateGraph) -> set[str]:
    """Nœuds enregistrés sans aucun chemin possible vers END (impasse).

    Calculé par point fixe : un nœud "peut atteindre END" s'il y va
    directement ou si l'une de ses destinations peut elle-même y aller. Les
    cycles (ex. une route de relance qui revient sur un nœud amont) ne posent
    pas de problème — dès qu'une branche du cycle atteint END, le reste du
    cycle est marqué à l'itération suivante.

    Garantit l'invariant : chaque chemin doit atteindre END, une
    interruption (qui reprend forcément sur un nœud du graphe, donc soumis à
    la même règle) ou un échec explicite qui mène lui-même à END.
    """
    can_reach_end: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in graph.nodes:
            if node in can_reach_end:
                continue
            destinations = _direct_destinations(graph, node)
            if END in destinations or destinations & can_reach_end:
                can_reach_end.add(node)
                changed = True
    return set(graph.nodes) - can_reach_end


def assert_workflow_topology_is_sound(graph: StateGraph) -> None:
    """Vérification automatique exécutée à chaque construction d'un workflow.

    Échoue explicitement, dans cet ordre, si :
      1. une transition pointe vers un nœud absent (dangling) ;
      2. un nœud enregistré n'a aucune entrée ni sortie valide (isolé) ;
      3. un nœud enregistré est inaccessible depuis START (nœud mort) ;
      4. un nœud enregistré n'a aucun chemin possible vers END (impasse).

    L'ordre importe : une transition dangling fausserait les diagnostics
    d'accessibilité (le nom fautif n'existe dans aucun nœud), donc elle est
    signalée en premier avec un message dédié plutôt que de se manifester
    indirectement comme un nœud inaccessible.
    """
    dangling = find_dangling_transitions(graph)
    if dangling:
        details = ", ".join(f"{src!r} → {dst!r}" for src, dst in sorted(dangling))
        raise ValueError(
            f"Topologie du workflow invalide : transition(s) vers un nœud absent : {details}"
        )
    isolated = find_isolated_nodes(graph)
    if isolated:
        raise ValueError(
            "Topologie du workflow invalide : nœud(s) sans entrée ni sortie valide : "
            f"{sorted(isolated)}"
        )
    unreachable = find_unreachable_nodes(graph)
    if unreachable:
        raise ValueError(
            "Topologie du workflow invalide : nœud(s) inaccessible(s) depuis "
            f"START : {sorted(unreachable)}"
        )
    dead_ends = find_dead_end_nodes(graph)
    if dead_ends:
        raise ValueError(
            "Topologie du workflow invalide : nœud(s) sans chemin vers END "
            f"(impasse) : {sorted(dead_ends)}"
        )
