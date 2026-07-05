"""Workflow LangGraph — ClaimShield Santé.

Construit et compile le StateGraph complet.
Le checkpointer est reçu en paramètre : aucun backend n'est instancié ici.

Topologie
---------
::

    START ──► claim_intake
               ├─[CONTINUE]──► security_gate
               │                ├─[CONTINUE]──► privacy
               │                │                ├─[CONTINUE]──► document_ocr
               │                │                │                ├─[CONTINUE]──► fhir_validator
               │                │                │                │                ├─[CONTINUE]──► identity_coverage
               │                │                │                │                │                ├─[CONTINUE]──► medical_coding
               │                │                │                │                │                │                ├─[CONTINUE]──► clinical_consistency
               │                │                │                │                │                │                │               ──► fraud_detection
               │                │                │                │                │                │                │               ──► case_reviewer
               │                │                │                │                │                │                │               ├─[ROUTE_END]──► audit ──► finalize ──► END
               │                │                │                │                │                │                │               ├─[NEEDS_REVIEW]──► needs_review ──► await_human_review*
               │                │                │                │                │                │                │               └─[FAILURE]──► failure ──► END
               │                │                │                │                │                │                ├─[NEEDS_REVIEW]──► needs_review ──► await_human_review*
               │                │                │                │                │                │                └─[FAILURE/RETRY]──► failure ──► END
               │                │                │                │                │                ├─[NEEDS_REVIEW]──► needs_review ──► await_human_review*
               │                │                │                │                │                ├─[FAILURE/RETRY]──► failure ──► END
               │                │                │                │                ├─[RETRY]──► identity_coverage  (bundle FHIR absent : non bloquant)
               │                │                │                │                ├─[NEEDS_REVIEW]──► needs_review ──► await_human_review*
               │                │                │                │                └─[FAILURE]──► failure ──► END
               │                │                │                ├─[NEEDS_REVIEW]──► needs_review ──► await_human_review*
               │                │                │                └─[FAILURE/RETRY]──► failure ──► END
               │                │                └─[FAILURE]──► failure ──► END
               │                ├─[QUARANTINE]──► quarantine ──► END
               │                └─[FAILURE]──► failure ──► END
               ├─[QUARANTINE]──► quarantine ──► END
               └─[FAILURE/RETRY]──► failure ──► END

    * await_human_review ──► route_human_review (voir graph.edges) :
                 ├─[APPROVE]──────────► END
                 ├─[REJECT]───────────► failure ──► END
                 └─[NEEDS_MORE_INFO]──► relancer (nœud demandé, si sous la
                                         limite) ──► ... └─[au-delà]──► failure ──► END

Règle arêtes
------------
* Arête **conditionnelle** depuis un nœud → aucune arête normale depuis ce même nœud.
* Arête **normale** depuis un nœud → aucune arête conditionnelle depuis ce même nœud.
* Nœuds à arêtes conditionnelles :
  claim_intake · security_gate · privacy · document_ocr · fhir_validator
  identity_coverage · medical_coding · case_reviewer · await_human_review
* Nœuds à arête normale uniquement :
  clinical_consistency · fraud_detection · audit · quarantine · needs_review
  failure · finalize

Nœuds HITL
----------
``needs_review`` : le pipeline pause *avant* l'exécution du nœud
(``interrupt_before``, statique). Pour désactiver cette interruption (tests) :
``compile_workflow(interrupt_before=[])``.

``await_human_review`` : exécuté juste après ``needs_review``, ce nœud
suspend le graphe *pendant* son exécution via ``langgraph.types.interrupt()``
(interruption dynamique). Le payload transmis contient ``case_id``, les
motifs de revue, des preuves minimisées et les actions autorisées
(``APPROVE`` / ``REJECT`` / ``NEEDS_MORE_INFO``). La reprise se fait avec
``app.invoke(Command(resume=decision), config=config)`` en réutilisant
impérativement le ``thread_id`` de l'invocation initiale — voir
``graph.checkpoints.assert_same_thread_id`` / ``CheckpointSession.assert_resume``.

Route de relance (« relancer »)
--------------------------------
Après ``await_human_review``, ``graph.edges.route_human_review`` route selon
``human_decision.decision`` :

- ``APPROVE`` → ``END``.
- ``REJECT`` → ``failure``.
- ``NEEDS_MORE_INFO`` → reprend explicitement au nœud demandé
  (``human_decision.target_node``, doit appartenir à
  ``graph.edges.RELAUNCH_TARGETS``), à condition que ``correction_attempts``
  (compteur minimal incrémenté par ``node_await_human_review`` à chaque
  NEEDS_MORE_INFO — aucun compteur générique existant à réutiliser) ne
  dépasse pas la limite configurable ``max_correction_attempts``
  (``build_workflow(max_correction_attempts=...)`` ou
  ``compile_workflow(max_correction_attempts=...)``, sinon
  ``Settings.claimshield_max_correction_attempts``, défaut 3). Au-delà de
  cette limite → ``failure`` (empêche toute boucle infinie de corrections).
"""
from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from config.settings import get_settings
from graph.edges import (
    CONTINUE,
    END as ROUTE_END,
    FAILURE,
    NEEDS_REVIEW,
    QUARANTINE,
    RELAUNCH_TARGETS,
    RETRY,
    route_coding,
    route_fhir,
    route_human_review,
    route_identity_coverage,
    route_intake,
    route_ocr,
    route_privacy,
    route_review,
    route_security,
)
from graph.nodes import (
    AuditAgentRunnable,
    CaseReviewerRunnable,
    ClinicalConsistencyRunnable,
    FraudDetectionRunnable,
    build_node_registry,
    build_orchestrator,
)
from graph.technical_nodes import (
    node_await_human_review,
    node_failure,
    node_finalize,
    node_needs_review,
    node_quarantine,
)
from orchestrator.executor import Orchestrator
from state.claim_state import ClaimState

# Nœuds où interrompre par défaut pour la validation humaine (HITL).
DEFAULT_INTERRUPT_BEFORE: list[str] = ["needs_review"]

# ── Nœuds agents réels — orchestrateur par défaut ────────────────────────────
#
# Construits une fois, explicitement, via build_orchestrator() (aucune
# instance cachée : visible ici, jamais implicite). Exposés comme noms de
# module — permet à un test de remplacer un nœud précis
# (monkeypatch.setattr(graph.workflow, "node_claim_intake", fake_fn)) sans
# reconstruire tout le graphe, exactement comme avant l'introduction de
# l'orchestrateur. Un ``orchestrator=`` explicite fourni à build_workflow()
# remplace entièrement ces 11 nœuds par défaut (voir plus bas).
_default_orchestrator = build_orchestrator()
_default_agent_nodes = build_node_registry(_default_orchestrator)

node_claim_intake = _default_agent_nodes["claim_intake"]
node_security_gate = _default_agent_nodes["security_gate"]
node_privacy = _default_agent_nodes["privacy"]
node_document_ocr = _default_agent_nodes["document_ocr"]
node_fhir_validator = _default_agent_nodes["fhir_validator"]
node_identity_coverage = _default_agent_nodes["identity_coverage"]
node_medical_coding = _default_agent_nodes["medical_coding"]


# ── Vérification automatique de la topologie ─────────────────────────────────


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
    cycles (ex. la route de relance qui revient sur un nœud amont) ne posent
    pas de problème — dès qu'une branche du cycle atteint END, le reste du
    cycle est marqué à l'itération suivante.

    Garantit l'invariant : chaque chemin doit atteindre END, une
    interruption (qui reprend forcément sur un nœud du graphe, donc soumis à
    la même règle) ou un échec explicite (``failure``, qui mène lui-même à
    END).
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


def _assert_workflow_topology_is_sound(graph: StateGraph) -> None:
    """Vérification automatique exécutée à chaque construction du workflow.

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


def build_workflow(
    *,
    orchestrator: Orchestrator | None = None,
    max_correction_attempts: int | None = None,
    clinical_consistency_impl: ClinicalConsistencyRunnable | None = None,
    fraud_detection_impl: FraudDetectionRunnable | None = None,
    case_reviewer_impl: CaseReviewerRunnable | None = None,
    audit_impl: AuditAgentRunnable | None = None,
) -> StateGraph:
    """Construit (sans compiler) le StateGraph ClaimShield.

    Câble les 16 nœuds et toutes les arêtes/branches conditionnelles, puis
    exécute la vérification automatique de topologie
    (``_assert_workflow_topology_is_sound``) avant de retourner le graphe.
    Aucun checkpointer, aucune interruption : ce sont des préoccupations de
    compilation, traitées par ``compile_workflow``.

    Utile en soi pour l'introspection (nœuds/arêtes, génération Mermaid via
    ``get_workflow_mermaid``) sans dépendre d'un checkpointer.

    Les 11 nœuds agents (7 réels + 4 stubs) appellent désormais
    exclusivement leur agent via ``Orchestrator.execute_agent()``
    (``graph/nodes.py::build_node_registry``) — jamais d'appel direct à
    ``agent_module.node(state)`` depuis ce module.

    Args:
        orchestrator: ``Orchestrator`` (``orchestrator.executor``) injecté —
            construit et détenu par l'appelant. ``None`` → un orchestrateur
            est construit ici via ``graph.nodes.build_orchestrator(...)``
            avec les ``*_impl`` fournis ci-dessous. Si un ``orchestrator``
            est explicitement fourni, les ``*_impl`` sont ignorés pour la
            construction des nœuds agents (son propre ``agent_registry``
            fait déjà foi) — même convention que ``graph=`` pour
            ``compile_workflow``.
        max_correction_attempts: Limite de relances (NEEDS_MORE_INFO) après
            ``await_human_review`` avant de router vers ``failure``.
            ``None`` → ``Settings.claimshield_max_correction_attempts``.
        clinical_consistency_impl: Implémentation injectable de l'agent
            clinical_consistency (étape non encore livrée). ``None`` → stub
            NOT_EVALUATED. Jamais importée en dur : transmise à
            ``build_orchestrator(clinical_consistency_impl=...)`` (ignorée
            si ``orchestrator`` est fourni explicitement).
        fraud_detection_impl: Idem pour fraud_detection (``None`` → stub
            NOT_EVALUATED).
        case_reviewer_impl: Idem pour case_reviewer (``None`` → stub PENDING).
        audit_impl: Idem pour audit (``None`` → stub NOT_EVALUATED).

    Returns:
        ``StateGraph`` non compilé, prêt pour ``compile_workflow(graph=...)``.

    Raises:
        ValueError: Topologie invalide (transition vers un nœud absent,
            nœud isolé, inaccessible depuis START, ou sans chemin vers END).

    Exemple ::

        from graph.workflow import build_workflow, compile_workflow

        graph = build_workflow()
        app = compile_workflow(graph=graph, checkpointer=my_saver)
    """
    if orchestrator is not None:
        # Orchestrateur explicite : remplace entièrement les 11 nœuds agents
        # (y compris les 7 réels) — même convention que ``graph=`` pour
        # ``compile_workflow`` (les paramètres de construction par défaut
        # sont ignorés dès qu'une instance explicite est fournie).
        node_fns = build_node_registry(orchestrator)
    else:
        if any(
            impl is not None
            for impl in (clinical_consistency_impl, fraud_detection_impl, case_reviewer_impl, audit_impl)
        ):
            stub_orchestrator = build_orchestrator(
                clinical_consistency_impl=clinical_consistency_impl,
                fraud_detection_impl=fraud_detection_impl,
                case_reviewer_impl=case_reviewer_impl,
                audit_impl=audit_impl,
            )
            stub_nodes = build_node_registry(stub_orchestrator)
        else:
            stub_nodes = _default_agent_nodes
        # Les 7 agents réels utilisent l'orchestrateur par défaut du module
        # (``_default_orchestrator``) — noms de module monkeypatchables
        # individuellement (ex. tests), exactement comme avant l'introduction
        # de l'orchestrateur. Seuls les 4 agents stubs varient avec *_impl.
        node_fns = {
            "claim_intake": node_claim_intake,
            "security_gate": node_security_gate,
            "privacy": node_privacy,
            "document_ocr": node_document_ocr,
            "fhir_validator": node_fhir_validator,
            "identity_coverage": node_identity_coverage,
            "medical_coding": node_medical_coding,
            "clinical_consistency": stub_nodes["clinical_consistency"],
            "fraud_detection": stub_nodes["fraud_detection"],
            "case_reviewer": stub_nodes["case_reviewer"],
            "audit": stub_nodes["audit"],
        }

    graph = StateGraph(ClaimState)

    # ── Nœuds agents — implémentations réelles ───────────────────────────────
    # Tous appellent orchestrator.execute_agent() — voir graph/nodes.py.

    graph.add_node("claim_intake", node_fns["claim_intake"])
    graph.add_node("security_gate", node_fns["security_gate"])
    graph.add_node("privacy", node_fns["privacy"])
    graph.add_node("document_ocr", node_fns["document_ocr"])
    graph.add_node("fhir_validator", node_fns["fhir_validator"])
    graph.add_node("identity_coverage", node_fns["identity_coverage"])
    graph.add_node("medical_coding", node_fns["medical_coding"])

    # ── Nœuds agents — interfaces injectables (stubs NOT_EVALUATED/PENDING) ──
    #
    # Jamais importés en dur : l'implémentation (*_impl, voir build_orchestrator)
    # est déjà résolue dans orchestrator.agent_registry — ce module ne fait que
    # câbler le nœud générique correspondant.

    graph.add_node("clinical_consistency", node_fns["clinical_consistency"])
    graph.add_node("fraud_detection", node_fns["fraud_detection"])
    graph.add_node("case_reviewer", node_fns["case_reviewer"])
    graph.add_node("audit", node_fns["audit"])

    # ── Nœuds techniques de transition ───────────────────────────────────────

    graph.add_node("quarantine", node_quarantine)
    graph.add_node("needs_review", node_needs_review)
    graph.add_node("await_human_review", node_await_human_review)
    graph.add_node("failure", node_failure)
    graph.add_node("finalize", node_finalize)

    # ── Arête d'entrée ───────────────────────────────────────────────────────

    graph.add_edge(START, "claim_intake")

    # ── Arêtes conditionnelles — agents avec décision de routage ─────────────
    #
    # Règle : si un nœud a add_conditional_edges, il n'a PAS de add_edge.
    # RETRY → "failure" par défaut (évite les boucles infinies sur erreur
    # transitoire), sauf pour fhir_validator où RETRY signifie bundle absent
    # et ne bloque pas le pipeline.

    graph.add_conditional_edges(
        "claim_intake",
        route_intake,
        {
            CONTINUE:  "security_gate",
            QUARANTINE: "quarantine",
            FAILURE:   "failure",
            RETRY:     "failure",
        },
    )

    graph.add_conditional_edges(
        "security_gate",
        route_security,
        {
            CONTINUE:  "privacy",
            QUARANTINE: "quarantine",
            FAILURE:   "failure",
        },
    )

    graph.add_conditional_edges(
        "privacy",
        route_privacy,
        {
            CONTINUE: "document_ocr",
            FAILURE:  "failure",
        },
    )

    graph.add_conditional_edges(
        "document_ocr",
        route_ocr,
        {
            CONTINUE:     "fhir_validator",
            NEEDS_REVIEW: "needs_review",
            FAILURE:      "failure",
            RETRY:        "failure",
        },
    )

    graph.add_conditional_edges(
        "fhir_validator",
        route_fhir,
        {
            CONTINUE:     "identity_coverage",
            NEEDS_REVIEW: "needs_review",
            FAILURE:      "failure",
            RETRY:        "identity_coverage",  # bundle FHIR absent → non bloquant
        },
    )

    graph.add_conditional_edges(
        "identity_coverage",
        route_identity_coverage,
        {
            CONTINUE:     "medical_coding",
            NEEDS_REVIEW: "needs_review",
            FAILURE:      "failure",
            RETRY:        "failure",
        },
    )

    graph.add_conditional_edges(
        "medical_coding",
        route_coding,
        {
            CONTINUE:     "clinical_consistency",
            NEEDS_REVIEW: "needs_review",
            FAILURE:      "failure",
            RETRY:        "failure",
        },
    )

    graph.add_conditional_edges(
        "case_reviewer",
        route_review,
        {
            ROUTE_END:    "audit",
            NEEDS_REVIEW: "needs_review",
            FAILURE:      "failure",
        },
    )

    # Route de relance (« relancer ») : la décision humaine NEEDS_MORE_INFO
    # reprend explicitement au nœud demandé (path_map en identité sur
    # RELAUNCH_TARGETS) tant que le compteur correction_attempts ne dépasse
    # pas max_correction_attempts — au-delà, route_human_review renvoie
    # FAILURE. APPROVE → END, REJECT → failure.
    attempts_limit = (
        max_correction_attempts
        if max_correction_attempts is not None
        else get_settings().claimshield_max_correction_attempts
    )
    graph.add_conditional_edges(
        "await_human_review",
        partial(route_human_review, max_attempts=attempts_limit),
        {
            **{node_name: node_name for node_name in RELAUNCH_TARGETS},
            ROUTE_END: END,
            FAILURE:   "failure",
        },
    )

    # ── Arêtes normales — stubs et nœuds de clôture ──────────────────────────
    #
    # Règle : ces nœuds n'ont PAS de add_conditional_edges.
    # clinical_consistency et fraud_detection sont des stubs NOT_EVALUATED :
    # leur sortie est déterministe — aucune décision de routage requise.

    graph.add_edge("clinical_consistency", "fraud_detection")
    graph.add_edge("fraud_detection", "case_reviewer")

    graph.add_edge("audit", "finalize")
    graph.add_edge("finalize", END)

    graph.add_edge("quarantine", END)
    graph.add_edge("needs_review", "await_human_review")
    graph.add_edge("failure", END)

    # ── Vérification automatique — transitions dangling, nœuds isolés,
    #    inaccessibles et impasses ──────────────────────────────────────────

    _assert_workflow_topology_is_sound(graph)

    return graph


def compile_workflow(
    checkpointer: Any = None,
    *,
    graph: StateGraph | None = None,
    orchestrator: Orchestrator | None = None,
    interrupt_before: list[str] | None = None,
    max_correction_attempts: int | None = None,
    clinical_consistency_impl: ClinicalConsistencyRunnable | None = None,
    fraud_detection_impl: FraudDetectionRunnable | None = None,
    case_reviewer_impl: CaseReviewerRunnable | None = None,
    audit_impl: AuditAgentRunnable | None = None,
) -> Any:
    """Compile un StateGraph ClaimShield en graphe exécutable.

    Le checkpointer est injecté en dépendance : aucun backend n'est créé ici.
    Utiliser ``CheckpointerFactory`` pour obtenir une instance adaptée
    à l'environnement (tests, SQLite, PostgreSQL).

    Args:
        checkpointer: Instance de checkpointer LangGraph (``InMemorySaver``,
            ``SqliteSaver``…).  ``None`` désactive la persistance de state.
        graph: ``StateGraph`` déjà construit (ex. via ``build_workflow()``).
            ``None`` → un graphe frais est construit avec
            ``orchestrator``/``max_correction_attempts`` et les ``*_impl``
            fournis ici. Si un ``graph`` est fourni, ces paramètres de
            construction sont ignorés (le graphe est déjà câblé, orchestrateur
            compris) — seuls ``checkpointer`` et ``interrupt_before``
            s'appliquent à la compilation.
        orchestrator: Voir ``build_workflow`` (ignoré si ``graph`` est fourni).
        interrupt_before: Nœuds HITL où interrompre *avant* exécution.
            ``None`` → ``DEFAULT_INTERRUPT_BEFORE`` (``["needs_review"]``).
            ``[]`` → aucune interruption (tests automatisés).
        max_correction_attempts: Voir ``build_workflow`` (ignoré si ``graph``
            est fourni).
        clinical_consistency_impl: Voir ``build_workflow`` (ignoré si
            ``graph`` est fourni).
        fraud_detection_impl: Voir ``build_workflow`` (ignoré si ``graph``
            est fourni).
        case_reviewer_impl: Voir ``build_workflow`` (ignoré si ``graph`` est
            fourni).
        audit_impl: Voir ``build_workflow`` (ignoré si ``graph`` est fourni).

    Returns:
        ``CompiledStateGraph`` prêt pour ``.invoke()``, ``.stream()`` et
        ``.get_state()``.

    Exemple ::

        from graph.checkpoints import CheckpointerFactory
        from graph.workflow import compile_workflow

        app = compile_workflow(CheckpointerFactory.for_tests().build())
        result = app.invoke(
            initial_state,
            config={"configurable": {"thread_id": "CLM-0001", "checkpoint_ns": ""}},
        )
    """
    if graph is None:
        graph = build_workflow(
            orchestrator=orchestrator,
            max_correction_attempts=max_correction_attempts,
            clinical_consistency_impl=clinical_consistency_impl,
            fraud_detection_impl=fraud_detection_impl,
            case_reviewer_impl=case_reviewer_impl,
            audit_impl=audit_impl,
        )

    interrupts = (
        interrupt_before if interrupt_before is not None else DEFAULT_INTERRUPT_BEFORE
    )

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupts,
    )


def get_workflow_mermaid(app: Any | None = None) -> str:
    """Retourne la représentation Mermaid (texte) du workflow ClaimShield.

    Utilise ``CompiledStateGraph.get_graph().draw_mermaid()`` — fonctionne
    dès que ``langgraph``/``langchain_core`` sont installés (pas de
    dépendance supplémentaire type Graphviz/Playwright, contrairement à
    ``draw_mermaid_png``).

    Args:
        app: Graphe compilé à représenter. ``None`` → compile un workflow
            par défaut (``interrupt_before=[]``, agents futurs en stub) —
            uniquement la topologie, aucune donnée de dossier réel.

    Returns:
        Diagramme Mermaid (``graph TD;...``) sans aucune donnée sensible :
        uniquement des noms de nœuds et de routes, jamais de contenu de
        state, de secret ni de donnée patient.
    """
    if app is None:
        app = compile_workflow(interrupt_before=[])
    return app.get_graph().draw_mermaid()
