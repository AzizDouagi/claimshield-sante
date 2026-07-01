"""Tests — graph/workflow.py — ClaimShield Santé.

Trois niveaux de vérification :
  1. ``TestWorkflowStructure``   : compilation, nœuds, checkpointer injecté.
  2. ``TestWorkflowTopology``    : arêtes normales / conditionnelles, règle de
     non-mélange.
  3. ``TestWorkflowInterrupts``  : DEFAULT_INTERRUPT_BEFORE, override, désactivation.
  4. ``TestWorkflowInvoke``      : exécution complète avec agents mockés.

Stratégie de mock pour les tests d'invocation
----------------------------------------------
Les 7 agents réels nécessitent des fichiers et Ollama.  On les remplace par
des fonctions stub légères patchées dans l'espace de noms ``graph.workflow``
**avant** l'appel à ``build_workflow()``.  LangGraph capture les fonctions
au moment du ``add_node()`` — le patch doit donc précéder la compilation.

Les stubs de stubs (clinical_consistency, fraud_detection, audit) sont
utilisés tels quels ; ils ne dépendent d'aucune ressource externe.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver

import graph.workflow as wf
from graph.workflow import DEFAULT_INTERRUPT_BEFORE, build_workflow
from schemas.domain import (
    IntakeStatus,
    Recommendation,
    SecurityDecision,
    VerificationStatus,
)

# ── Constantes attendues ──────────────────────────────────────────────────────

_AGENT_NODES = {
    "claim_intake", "security_gate", "privacy",
    "document_ocr", "fhir_validator", "identity_coverage", "medical_coding",
    "clinical_consistency", "fraud_detection", "case_reviewer", "audit",
}
_TECHNICAL_NODES = {"quarantine", "needs_review", "failure", "finalize"}
_ALL_NODES = _AGENT_NODES | _TECHNICAL_NODES | {"__start__"}

# Nœuds ayant des arêtes conditionnelles (jamais d'arête normale en plus).
_CONDITIONAL_SOURCES = {
    "claim_intake", "security_gate", "privacy",
    "document_ocr", "fhir_validator", "identity_coverage", "medical_coding",
    "case_reviewer",
}
# Arêtes normales attendues (source, destination).
_EXPECTED_UNCONDITIONAL = {
    ("__start__", "claim_intake"),
    ("clinical_consistency", "fraud_detection"),
    ("fraud_detection", "case_reviewer"),
    ("audit", "finalize"),
    ("finalize", "__end__"),
    ("quarantine", "__end__"),
    ("needs_review", "__end__"),
    ("failure", "__end__"),
}


# ── TestWorkflowStructure ─────────────────────────────────────────────────────


class TestWorkflowStructure:
    def test_build_returns_non_none(self):
        app = build_workflow(interrupt_before=[])
        assert app is not None

    def test_build_is_callable_graph(self):
        app = build_workflow(interrupt_before=[])
        assert callable(app.invoke)

    def test_all_nodes_present(self):
        app = build_workflow(interrupt_before=[])
        assert set(app.nodes) == _ALL_NODES

    def test_start_node_present(self):
        app = build_workflow(interrupt_before=[])
        assert "__start__" in app.nodes

    def test_all_agent_nodes_present(self):
        app = build_workflow(interrupt_before=[])
        assert _AGENT_NODES <= set(app.nodes)

    def test_all_technical_nodes_present(self):
        app = build_workflow(interrupt_before=[])
        assert _TECHNICAL_NODES <= set(app.nodes)

    def test_no_checkpointer_accepted(self):
        app = build_workflow(None, interrupt_before=[])
        assert app.checkpointer is None

    def test_injected_checkpointer_stored(self):
        saver = InMemorySaver()
        app = build_workflow(saver, interrupt_before=[])
        assert app.checkpointer is saver

    def test_different_savers_stored_independently(self):
        saver_a = InMemorySaver()
        saver_b = InMemorySaver()
        app_a = build_workflow(saver_a, interrupt_before=[])
        app_b = build_workflow(saver_b, interrupt_before=[])
        assert app_a.checkpointer is saver_a
        assert app_b.checkpointer is saver_b
        assert app_a.checkpointer is not app_b.checkpointer


# ── TestWorkflowInterrupts ────────────────────────────────────────────────────


class TestWorkflowInterrupts:
    def test_default_interrupt_constant_contains_needs_review(self):
        assert "needs_review" in DEFAULT_INTERRUPT_BEFORE

    def test_default_interrupt_applied_when_none_passed(self):
        saver = InMemorySaver()
        app = build_workflow(saver)
        assert "needs_review" in app.interrupt_before_nodes

    def test_none_interrupt_uses_default(self):
        saver = InMemorySaver()
        app = build_workflow(saver, interrupt_before=None)
        assert app.interrupt_before_nodes == DEFAULT_INTERRUPT_BEFORE

    def test_empty_list_disables_all_interrupts(self):
        saver = InMemorySaver()
        app = build_workflow(saver, interrupt_before=[])
        assert app.interrupt_before_nodes == []

    def test_custom_interrupt_before_respected(self):
        saver = InMemorySaver()
        app = build_workflow(saver, interrupt_before=["quarantine"])
        assert "quarantine" in app.interrupt_before_nodes
        assert "needs_review" not in app.interrupt_before_nodes

    def test_multiple_interrupts_respected(self):
        saver = InMemorySaver()
        app = build_workflow(saver, interrupt_before=["needs_review", "quarantine"])
        assert set(app.interrupt_before_nodes) == {"needs_review", "quarantine"}


# ── TestWorkflowTopology ──────────────────────────────────────────────────────


class TestWorkflowTopology:
    def _app(self):
        return build_workflow(interrupt_before=[])

    def test_start_to_claim_intake_is_unconditional(self):
        app = self._app()
        assert ("__start__", "claim_intake") in app.builder.edges

    def test_clinical_consistency_to_fraud_unconditional(self):
        app = self._app()
        assert ("clinical_consistency", "fraud_detection") in app.builder.edges

    def test_fraud_to_case_reviewer_unconditional(self):
        app = self._app()
        assert ("fraud_detection", "case_reviewer") in app.builder.edges

    def test_audit_to_finalize_unconditional(self):
        app = self._app()
        assert ("audit", "finalize") in app.builder.edges

    def test_finalize_to_end(self):
        app = self._app()
        assert ("finalize", "__end__") in app.builder.edges

    def test_failure_to_end(self):
        app = self._app()
        assert ("failure", "__end__") in app.builder.edges

    def test_quarantine_to_end(self):
        app = self._app()
        assert ("quarantine", "__end__") in app.builder.edges

    def test_needs_review_to_end(self):
        app = self._app()
        assert ("needs_review", "__end__") in app.builder.edges

    def test_all_expected_unconditional_edges_present(self):
        app = self._app()
        assert _EXPECTED_UNCONDITIONAL <= app.builder.edges

    def test_claim_intake_has_conditional_edges(self):
        app = self._app()
        assert "claim_intake" in app.builder.branches

    def test_security_gate_has_conditional_edges(self):
        app = self._app()
        assert "security_gate" in app.builder.branches

    def test_privacy_has_conditional_edges(self):
        app = self._app()
        assert "privacy" in app.builder.branches

    def test_document_ocr_has_conditional_edges(self):
        app = self._app()
        assert "document_ocr" in app.builder.branches

    def test_fhir_validator_has_conditional_edges(self):
        app = self._app()
        assert "fhir_validator" in app.builder.branches

    def test_identity_coverage_has_conditional_edges(self):
        app = self._app()
        assert "identity_coverage" in app.builder.branches

    def test_medical_coding_has_conditional_edges(self):
        app = self._app()
        assert "medical_coding" in app.builder.branches

    def test_case_reviewer_has_conditional_edges(self):
        app = self._app()
        assert "case_reviewer" in app.builder.branches

    def test_all_conditional_sources_have_branches(self):
        app = self._app()
        assert _CONDITIONAL_SOURCES <= set(app.builder.branches.keys())

    def test_no_node_has_both_edge_types(self):
        app = self._app()
        unconditional_sources = {src for src, _ in app.builder.edges}
        conditional_sources = set(app.builder.branches.keys())
        overlap = unconditional_sources & conditional_sources
        # __start__ n'est jamais une source de conditional_edges — pas d'overlap.
        assert not overlap, f"Nœuds avec les deux types d'arête : {overlap}"

    def test_stub_nodes_have_no_conditional_edges(self):
        app = self._app()
        stub_nodes = {"clinical_consistency", "fraud_detection"}
        assert not (stub_nodes & set(app.builder.branches.keys()))


# ── Helpers d'invocation ──────────────────────────────────────────────────────


def _initial_state() -> dict:
    """État initial minimal passé au workflow pour les tests d'invocation."""
    return {
        "case_id": "CLM-WF-TEST",
        "schema_version": "1.0.0",
        "current_step": "initial",
        "completed_steps": [],
        "errors": [],
        "alerts": [],
        "final_justification": [],
    }


def _make_mock_agents(monkeypatch) -> None:
    """Patche les 7 agents réels avec des stubs légers sans I/O.

    Chaque stub retourne uniquement les champs que le routeur suivant lit,
    plus les champs de traçabilité (current_step, completed_steps).
    Les champs *_result sont des SimpleNamespace — ils passent les routers
    sans validation Pydantic (les mocks remplacent la fonction de nœud
    complète, pas seulement l'agent sous-jacent).
    """

    def mock_claim_intake(state: dict) -> dict:
        return {
            "intake_status": IntakeStatus.ACCEPTED,
            "intake_input": None,
            "current_step": "claim_intake",
            "completed_steps": ["claim_intake"],
        }

    def mock_security_gate(state: dict) -> dict:
        return {
            "security_result": SimpleNamespace(decision=SecurityDecision.ALLOW),
            "security_input": None,
            "current_step": "security_gate",
            "completed_steps": ["security_gate"],
        }

    def mock_privacy(state: dict) -> dict:
        from schemas.domain import PrivacyDecision
        return {
            "privacy_result": SimpleNamespace(decision=PrivacyDecision.ALLOW),
            "privacy_input": None,
            "current_step": "privacy",
            "completed_steps": ["privacy"],
        }

    def mock_document_ocr(state: dict) -> dict:
        return {
            "ocr_result": SimpleNamespace(status=VerificationStatus.PASS),
            "ocr_input": None,
            "current_step": "document_ocr",
            "completed_steps": ["document_ocr"],
        }

    def mock_fhir_validator(state: dict) -> dict:
        return {
            "fhir_result": SimpleNamespace(status=VerificationStatus.PASS),
            "fhir_input": None,
            "current_step": "fhir_validator",
            "completed_steps": ["fhir_validator"],
        }

    def mock_identity_coverage(state: dict) -> dict:
        return {
            "identity_coverage_result": SimpleNamespace(
                identity=SimpleNamespace(status=VerificationStatus.PASS),
                coverage=SimpleNamespace(status=VerificationStatus.PASS),
            ),
            "identity_coverage_input": None,
            "current_step": "identity_coverage",
            "completed_steps": ["identity_coverage"],
        }

    def mock_medical_coding(state: dict) -> dict:
        return {
            "coding_result": SimpleNamespace(status=VerificationStatus.PASS),
            "coding_input": None,
            "current_step": "medical_coding",
            "completed_steps": ["medical_coding"],
        }

    monkeypatch.setattr(wf, "node_claim_intake", mock_claim_intake)
    monkeypatch.setattr(wf, "node_security_gate", mock_security_gate)
    monkeypatch.setattr(wf, "node_privacy", mock_privacy)
    monkeypatch.setattr(wf, "node_document_ocr", mock_document_ocr)
    monkeypatch.setattr(wf, "node_fhir_validator", mock_fhir_validator)
    monkeypatch.setattr(wf, "node_identity_coverage", mock_identity_coverage)
    monkeypatch.setattr(wf, "node_medical_coding", mock_medical_coding)


# ── TestWorkflowInvoke ────────────────────────────────────────────────────────


class TestWorkflowInvoke:
    # Les tests de flux utilisent checkpointer=None : LangGraph n'essaie pas de
    # sérialiser le state, ce qui évite les conflits avec les objets mock.
    # Seul test_invoke_checkpoints_state utilise InMemorySaver (chemin court
    # sans objets non-sérialisables).

    def test_invoke_reaches_needs_review_via_stub_case_reviewer(self, monkeypatch):
        """Pipeline nominal jusqu'à case_reviewer (stub PENDING → needs_review)."""
        _make_mock_agents(monkeypatch)
        app = build_workflow(None, interrupt_before=[])
        result = app.invoke(_initial_state())
        steps = result.get("completed_steps", [])
        assert "needs_review" in steps

    def test_invoke_includes_all_agents_before_case_reviewer(self, monkeypatch):
        """Tous les agents s'exécutent avant la revue (PENDING path)."""
        _make_mock_agents(monkeypatch)
        app = build_workflow(None, interrupt_before=[])
        result = app.invoke(_initial_state())
        steps = set(result.get("completed_steps", []))
        expected = {
            "claim_intake", "security_gate", "privacy",
            "document_ocr", "fhir_validator", "identity_coverage",
            "medical_coding", "clinical_consistency", "fraud_detection",
            "case_reviewer",
        }
        assert expected <= steps

    def test_invoke_finalize_path_with_approve(self, monkeypatch):
        """Pipeline complet jusqu'à finalize avec case_reviewer APPROVE."""
        _make_mock_agents(monkeypatch)

        def mock_case_reviewer(state: dict) -> dict:
            return {
                "review_result": SimpleNamespace(
                    recommendation=Recommendation.APPROVE,
                    human_review_required=False,
                    justification=["Approuvé — test."],
                ),
                "final_recommendation": Recommendation.APPROVE,
                "final_justification": ["Approuvé — test."],
                "current_step": "case_reviewer",
                "completed_steps": ["case_reviewer"],
            }

        monkeypatch.setattr(wf, "node_case_reviewer", mock_case_reviewer)
        app = build_workflow(None, interrupt_before=[])
        result = app.invoke(_initial_state())
        steps = result.get("completed_steps", [])
        assert "audit" in steps
        assert "finalize" in steps

    def test_invoke_failure_path_on_security_block(self, monkeypatch):
        """security_gate BLOCK → failure → final_recommendation = REJECT."""
        _make_mock_agents(monkeypatch)

        def mock_security_block(state: dict) -> dict:
            return {
                "security_result": SimpleNamespace(decision=SecurityDecision.BLOCK),
                "security_input": None,
                "current_step": "security_gate",
                "completed_steps": ["security_gate"],
            }

        monkeypatch.setattr(wf, "node_security_gate", mock_security_block)
        app = build_workflow(None, interrupt_before=[])
        result = app.invoke(_initial_state())
        steps = result.get("completed_steps", [])
        assert "failure" in steps
        assert result.get("final_recommendation") == Recommendation.REJECT

    def test_invoke_quarantine_path_on_security_quarantine(self, monkeypatch):
        """security_gate QUARANTINE → quarantine node exécuté."""
        _make_mock_agents(monkeypatch)

        def mock_security_quarantine(state: dict) -> dict:
            return {
                "security_result": SimpleNamespace(decision=SecurityDecision.QUARANTINE),
                "security_input": None,
                "current_step": "security_gate",
                "completed_steps": ["security_gate"],
            }

        monkeypatch.setattr(wf, "node_security_gate", mock_security_quarantine)
        app = build_workflow(None, interrupt_before=[])
        result = app.invoke(_initial_state())
        steps = result.get("completed_steps", [])
        assert "quarantine" in steps

    def test_invoke_checkpoints_state(self, monkeypatch):
        """Vérifie que le checkpoint est écrit avec InMemorySaver.

        Le dossier est bloqué dès claim_intake (BLOCKED) : le failure node
        n'écrit que des primitives (strings, enum) → aucun objet non-sérialisable.
        """
        # claim_intake retourne BLOCKED → failure → END (chemin court, tout sérialisable)
        def mock_claim_blocked(state: dict) -> dict:
            return {
                "intake_status": IntakeStatus.BLOCKED,
                "intake_input": None,
                "current_step": "claim_intake",
                "completed_steps": ["claim_intake"],
            }

        monkeypatch.setattr(wf, "node_claim_intake", mock_claim_blocked)
        saver = InMemorySaver()
        app = build_workflow(saver, interrupt_before=[])
        thread_cfg = {"configurable": {"thread_id": "CLM-WF-CP", "checkpoint_ns": ""}}
        app.invoke(_initial_state(), config=thread_cfg)
        checkpoint = saver.get(thread_cfg)
        assert checkpoint is not None
        assert "channel_values" in checkpoint
        assert "failure" in checkpoint["channel_values"].get("completed_steps", [])
