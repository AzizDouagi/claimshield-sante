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
               │                │                │                │                │                │                │               ├─[NEEDS_REVIEW]──► needs_review ──► END
               │                │                │                │                │                │                │               └─[FAILURE]──► failure ──► END
               │                │                │                │                │                │                ├─[NEEDS_REVIEW]──► needs_review ──► END
               │                │                │                │                │                │                └─[FAILURE/RETRY]──► failure ──► END
               │                │                │                │                │                ├─[NEEDS_REVIEW]──► needs_review ──► END
               │                │                │                │                │                ├─[FAILURE/RETRY]──► failure ──► END
               │                │                │                │                ├─[RETRY]──► identity_coverage  (bundle FHIR absent : non bloquant)
               │                │                │                │                ├─[NEEDS_REVIEW]──► needs_review ──► END
               │                │                │                │                └─[FAILURE]──► failure ──► END
               │                │                │                ├─[NEEDS_REVIEW]──► needs_review ──► END
               │                │                │                └─[FAILURE/RETRY]──► failure ──► END
               │                │                └─[FAILURE]──► failure ──► END
               │                ├─[QUARANTINE]──► quarantine ──► END
               │                └─[FAILURE]──► failure ──► END
               ├─[QUARANTINE]──► quarantine ──► END
               └─[FAILURE/RETRY]──► failure ──► END

Règle arêtes
------------
* Arête **conditionnelle** depuis un nœud → aucune arête normale depuis ce même nœud.
* Arête **normale** depuis un nœud → aucune arête conditionnelle depuis ce même nœud.
* Nœuds à arêtes conditionnelles :
  claim_intake · security_gate · privacy · document_ocr · fhir_validator
  identity_coverage · medical_coding · case_reviewer
* Nœuds à arête normale uniquement :
  clinical_consistency · fraud_detection · audit · quarantine · needs_review
  failure · finalize

Nœuds HITL (interruption avant exécution par défaut)
-----------------------------------------------------
``needs_review`` : le pipeline pause avant l'exécution du nœud.
L'intervenant fournit sa décision via ``update_state(config, {"human_decision": ...})``.
Pour désactiver l'interruption (tests) : ``build_workflow(interrupt_before=[])``.
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from graph.edges import (
    CONTINUE,
    END as ROUTE_END,
    FAILURE,
    NEEDS_REVIEW,
    QUARANTINE,
    RETRY,
    route_coding,
    route_fhir,
    route_identity_coverage,
    route_intake,
    route_ocr,
    route_privacy,
    route_review,
    route_security,
)
from graph.nodes import (
    node_audit,
    node_case_reviewer,
    node_claim_intake,
    node_clinical_consistency,
    node_document_ocr,
    node_fhir_validator,
    node_fraud_detection,
    node_identity_coverage,
    node_medical_coding,
    node_privacy,
    node_security_gate,
)
from graph.technical_nodes import (
    node_failure,
    node_finalize,
    node_needs_review,
    node_quarantine,
)
from state.claim_state import ClaimState

# Nœuds où interrompre par défaut pour la validation humaine (HITL).
DEFAULT_INTERRUPT_BEFORE: list[str] = ["needs_review"]


def build_workflow(
    checkpointer: Any = None,
    *,
    interrupt_before: list[str] | None = None,
) -> Any:
    """Construit et compile le StateGraph ClaimShield.

    Le checkpointer est injecté en dépendance : aucun backend n'est créé ici.
    Utiliser ``CheckpointerFactory`` pour obtenir une instance adaptée
    à l'environnement (tests, SQLite, PostgreSQL).

    Args:
        checkpointer: Instance de checkpointer LangGraph (``InMemorySaver``,
            ``SqliteSaver``…).  ``None`` désactive la persistance de state.
        interrupt_before: Nœuds HITL où interrompre *avant* exécution.
            ``None`` → ``DEFAULT_INTERRUPT_BEFORE`` (``["needs_review"]``).
            ``[]`` → aucune interruption (tests automatisés).

    Returns:
        ``CompiledStateGraph`` prêt pour ``.invoke()``, ``.stream()`` et
        ``.get_state()``.

    Exemple ::

        from graph.checkpoints import CheckpointerFactory
        from graph.workflow import build_workflow

        app = build_workflow(CheckpointerFactory.for_tests().build())
        result = app.invoke(
            initial_state,
            config={"configurable": {"thread_id": "CLM-0001", "checkpoint_ns": ""}},
        )
    """
    graph = StateGraph(ClaimState)

    # ── Nœuds agents — implémentations réelles ───────────────────────────────

    graph.add_node("claim_intake", node_claim_intake)
    graph.add_node("security_gate", node_security_gate)
    graph.add_node("privacy", node_privacy)
    graph.add_node("document_ocr", node_document_ocr)
    graph.add_node("fhir_validator", node_fhir_validator)
    graph.add_node("identity_coverage", node_identity_coverage)
    graph.add_node("medical_coding", node_medical_coding)

    # ── Nœuds agents — interfaces injectables (stubs NOT_EVALUATED/PENDING) ──

    graph.add_node("clinical_consistency", node_clinical_consistency)
    graph.add_node("fraud_detection", node_fraud_detection)
    graph.add_node("case_reviewer", node_case_reviewer)
    graph.add_node("audit", node_audit)

    # ── Nœuds techniques de transition ───────────────────────────────────────

    graph.add_node("quarantine", node_quarantine)
    graph.add_node("needs_review", node_needs_review)
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
    graph.add_edge("needs_review", END)
    graph.add_edge("failure", END)

    # ── Compilation ──────────────────────────────────────────────────────────

    interrupts = (
        interrupt_before if interrupt_before is not None else DEFAULT_INTERRUPT_BEFORE
    )

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupts,
    )
