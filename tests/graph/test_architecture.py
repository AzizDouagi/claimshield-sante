"""Test architectural — empêche graph/nodes.py de contourner Orchestrator.

Analyse **statique** (AST du code source, jamais l'exécution) de
``graph/nodes.py`` : vérifie qu'aucun appel direct à un agent
(``agent_module.node(...)``/``agent_module.run(...)``) n'existe en dehors de
la seule exception documentée — l'enregistrement (jamais l'exécution
immédiate) dans ``build_orchestrator()`` (nommée par
``graph.nodes._ORCHESTRATOR_REGISTRATION_FUNCTION``, importée ici plutôt que
recopiée en dur, pour que ce test se désynchronise bruyamment — jamais
silencieusement — d'un renommage de la fonction).

Un contrôle qui se contenterait de vérifier « aucun appel agent nulle part »
serait trivialement satisfait si l'exception légitime disparaissait aussi
(plus aucun agent enregistré, plus aucun graphe fonctionnel). Pour éviter un
contrôle purement décoratif, ce module vérifie donc les deux faces à la
fois : (1) aucun appel direct hors de l'exception documentée, et (2)
l'exception elle-même enregistre bien les 11 agents attendus — jamais zéro,
jamais une liste partielle qui masquerait un oubli silencieux.
"""
from __future__ import annotations

import ast
from pathlib import Path

import graph.nodes as nodes_module
from graph.nodes import _AGENT_MODULE_ALIASES, _ORCHESTRATOR_REGISTRATION_FUNCTION

_DIRECT_CALL_METHODS = frozenset({"node", "run"})


def _module_tree() -> ast.Module:
    source = Path(nodes_module.__file__).read_text(encoding="utf-8")
    return ast.parse(source, filename=nodes_module.__file__)


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosing_function_name(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str | None:
    """Nom de la fonction nommée (``def``) englobant ``node`` — traverse les
    lambdas anonymes sans s'y arrêter, puisqu'un enregistrement dans
    ``agent_registry`` se fait typiquement via ``lambda state: agent.node(state)``."""
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
        current = parents.get(current)
    return None


def _direct_agent_calls(tree: ast.Module) -> list[tuple[ast.Call, str | None]]:
    """Tous les appels ``<alias_agent>.node(...)``/``<alias_agent>.run(...)``
    du module, avec le nom de leur fonction englobante (``None`` si au
    niveau module)."""
    parents = _parent_map(tree)
    matches: list[tuple[ast.Call, str | None]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in _DIRECT_CALL_METHODS:
            continue
        callee = node.func.value
        if isinstance(callee, ast.Name) and callee.id in _AGENT_MODULE_ALIASES:
            matches.append((node, _enclosing_function_name(node, parents)))
    return matches


class TestNoDirectAgentInvocationOutsideOrchestratorRegistration:
    """``graph/nodes.py`` ne doit jamais appeler un agent directement en
    dehors de l'unique exception documentée et nommée
    (``_ORCHESTRATOR_REGISTRATION_FUNCTION``)."""

    def test_direct_agent_calls_exist_only_inside_the_documented_exception(self):
        matches = _direct_agent_calls(_module_tree())

        offending = [
            (call.func.value.id, call.func.attr, enclosing)
            for call, enclosing in matches
            if enclosing != _ORCHESTRATOR_REGISTRATION_FUNCTION
        ]
        assert offending == [], (
            "appel direct à un agent détecté hors de "
            f"{_ORCHESTRATOR_REGISTRATION_FUNCTION!r} : {offending!r} — tout "
            "nœud de graphe doit appeler Orchestrator.execute_agent(), "
            "jamais l'agent directement."
        )

    def test_the_exception_itself_is_not_vacuous_all_eleven_agents_registered(self):
        """Preuve que le test précédent n'est pas trivialement vrai pour une
        mauvaise raison (ex. plus aucun agent référencé nulle part) : les 11
        agents doivent bien apparaître, précisément dans
        ``build_orchestrator``."""
        matches = _direct_agent_calls(_module_tree())
        registered_in_exception = {
            call.func.value.id
            for call, enclosing in matches
            if enclosing == _ORCHESTRATOR_REGISTRATION_FUNCTION
        }
        assert registered_in_exception == set(_AGENT_MODULE_ALIASES)

    def test_documented_exception_function_still_exists(self):
        """``_ORCHESTRATOR_REGISTRATION_FUNCTION`` doit continuer à nommer
        une fonction réellement définie dans le module — sinon la
        documentation et le contrôle architectural auraient divergé du code."""
        tree = _module_tree()
        defined_functions = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        assert _ORCHESTRATOR_REGISTRATION_FUNCTION in defined_functions

    def test_node_closure_only_ever_delegates_to_execute_agent(self):
        """La fonction nœud réellement câblée dans le graphe (``_node``,
        construite par ``_build_node``) ne doit contenir aucun appel direct à
        un agent — sa seule délégation autorisée est
        ``orchestrator.execute_agent()``, appelée exactement une fois."""
        tree = _module_tree()
        build_node_fn = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_build_node"
        )
        node_closure = next(
            node for node in ast.walk(build_node_fn)
            if isinstance(node, ast.FunctionDef) and node.name == "_node"
        )

        direct_agent_calls_inside_closure = [
            call for call in ast.walk(node_closure)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr in _DIRECT_CALL_METHODS
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id in _AGENT_MODULE_ALIASES
        ]
        assert direct_agent_calls_inside_closure == []

        execute_agent_calls = [
            call for call in ast.walk(node_closure)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "execute_agent"
        ]
        assert len(execute_agent_calls) == 1

    def test_translate_outcome_never_calls_an_agent_either(self):
        """``_translate_outcome`` (traduction pure de l'``AgentCallOutcome``)
        ne doit elle non plus jamais invoquer un agent — elle ne fait que
        lire un résultat déjà produit."""
        tree = _module_tree()
        translate_fn = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_translate_outcome"
        )
        calls = [
            call for call in ast.walk(translate_fn)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr in _DIRECT_CALL_METHODS
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id in _AGENT_MODULE_ALIASES
        ]
        assert calls == []
