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
**avant** l'appel à ``compile_workflow()`` (qui construit le graphe via
``build_workflow()`` en interne, sauf ``graph`` explicite).  LangGraph
capture les fonctions au moment du ``add_node()`` — le patch doit donc
précéder la construction.

Les agents aval injectables restent déterministes en test ; audit ne dépend
pas d'une ressource externe tant que son implémentation réelle n'est pas
câblée.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

import graph.workflow as wf
from graph.workflow import (
    DEFAULT_INTERRUPT_BEFORE,
    build_workflow,
    compile_workflow,
    find_dangling_transitions,
    find_dead_end_nodes,
    find_isolated_nodes,
    find_unreachable_nodes,
    get_workflow_mermaid,
)
from schemas.domain import (
    IntakeStatus,
    PrivacyDecision,
    Recommendation,
    SecurityDecision,
    VerificationStatus,
)
from schemas.results import CaseReviewerResult, LlmMetadata


@dataclass
class _StubResult:
    """Résultat d'agent mocké sérialisable (dataclass) — requis pour les tests
    utilisant un vrai checkpointer (``InMemorySaver``), contrairement à
    ``SimpleNamespace`` qui n'est pas sérialisable par le serde msgpack de
    LangGraph."""

    decision: Any = None
    status: Any = None

# ── Constantes attendues ──────────────────────────────────────────────────────

_AGENT_NODES = {
    "claim_intake", "security_gate", "privacy",
    "document_ocr", "fhir_validator", "identity_coverage", "medical_coding",
    "clinical_consistency", "fraud_detection", "case_reviewer", "audit",
}
_TECHNICAL_NODES = {
    "quarantine", "needs_review", "await_human_review", "failure", "finalize",
}
_ALL_NODES = _AGENT_NODES | _TECHNICAL_NODES | {"__start__"}

# Nœuds ayant des arêtes conditionnelles (jamais d'arête normale en plus).
_CONDITIONAL_SOURCES = {
    "claim_intake", "security_gate", "privacy",
    "document_ocr", "fhir_validator", "identity_coverage", "medical_coding",
    "case_reviewer", "await_human_review",
}
# Arêtes normales attendues (source, destination).
_EXPECTED_UNCONDITIONAL = {
    ("__start__", "claim_intake"),
    ("clinical_consistency", "fraud_detection"),
    ("fraud_detection", "case_reviewer"),
    ("audit", "finalize"),
    ("finalize", "__end__"),
    ("quarantine", "__end__"),
    ("needs_review", "await_human_review"),
    ("failure", "__end__"),
}


# ── TestWorkflowStructure ─────────────────────────────────────────────────────


class TestWorkflowStructure:
    def test_build_returns_non_none(self):
        app = compile_workflow(interrupt_before=[])
        assert app is not None

    def test_build_is_callable_graph(self):
        app = compile_workflow(interrupt_before=[])
        assert callable(app.invoke)

    def test_all_nodes_present(self):
        app = compile_workflow(interrupt_before=[])
        assert set(app.nodes) == _ALL_NODES

    def test_start_node_present(self):
        app = compile_workflow(interrupt_before=[])
        assert "__start__" in app.nodes

    def test_all_agent_nodes_present(self):
        app = compile_workflow(interrupt_before=[])
        assert _AGENT_NODES <= set(app.nodes)

    def test_all_technical_nodes_present(self):
        app = compile_workflow(interrupt_before=[])
        assert _TECHNICAL_NODES <= set(app.nodes)

    def test_no_checkpointer_accepted(self):
        app = compile_workflow(None, interrupt_before=[])
        assert app.checkpointer is None

    def test_injected_checkpointer_stored(self):
        saver = InMemorySaver()
        app = compile_workflow(saver, interrupt_before=[])
        assert app.checkpointer is saver

    def test_different_savers_stored_independently(self):
        saver_a = InMemorySaver()
        saver_b = InMemorySaver()
        app_a = compile_workflow(saver_a, interrupt_before=[])
        app_b = compile_workflow(saver_b, interrupt_before=[])
        assert app_a.checkpointer is saver_a
        assert app_b.checkpointer is saver_b
        assert app_a.checkpointer is not app_b.checkpointer


# ── TestWorkflowInterrupts ────────────────────────────────────────────────────


class TestWorkflowInterrupts:
    def test_default_interrupt_constant_contains_needs_review(self):
        assert "needs_review" in DEFAULT_INTERRUPT_BEFORE

    def test_default_interrupt_applied_when_none_passed(self):
        saver = InMemorySaver()
        app = compile_workflow(saver)
        assert "needs_review" in app.interrupt_before_nodes

    def test_none_interrupt_uses_default(self):
        saver = InMemorySaver()
        app = compile_workflow(saver, interrupt_before=None)
        assert app.interrupt_before_nodes == DEFAULT_INTERRUPT_BEFORE

    def test_empty_list_disables_all_interrupts(self):
        saver = InMemorySaver()
        app = compile_workflow(saver, interrupt_before=[])
        assert app.interrupt_before_nodes == []

    def test_custom_interrupt_before_respected(self):
        saver = InMemorySaver()
        app = compile_workflow(saver, interrupt_before=["quarantine"])
        assert "quarantine" in app.interrupt_before_nodes
        assert "needs_review" not in app.interrupt_before_nodes

    def test_multiple_interrupts_respected(self):
        saver = InMemorySaver()
        app = compile_workflow(saver, interrupt_before=["needs_review", "quarantine"])
        assert set(app.interrupt_before_nodes) == {"needs_review", "quarantine"}


# ── TestWorkflowTopology ──────────────────────────────────────────────────────


class TestWorkflowTopology:
    def _app(self):
        return compile_workflow(interrupt_before=[])

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

    def test_needs_review_to_await_human_review(self):
        app = self._app()
        assert ("needs_review", "await_human_review") in app.builder.edges

    def test_await_human_review_has_conditional_edges(self):
        app = self._app()
        assert "await_human_review" in app.builder.branches

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
        "case_id": "CLM-0008",
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

    def test_invoke_reaches_needs_review_via_real_case_reviewer(self, monkeypatch):
        """Pipeline nominal jusqu'au reviewer réel, puis HITL obligatoire."""
        _make_mock_agents(monkeypatch)
        app = compile_workflow(None, interrupt_before=[])
        result = app.invoke(_initial_state())
        steps = result.get("completed_steps", [])
        assert "needs_review" in steps

    def test_invoke_includes_all_agents_before_case_reviewer(self, monkeypatch):
        """Tous les agents s'exécutent avant la revue (PENDING path)."""
        _make_mock_agents(monkeypatch)
        app = compile_workflow(None, interrupt_before=[])
        result = app.invoke(_initial_state())
        steps = set(result.get("completed_steps", []))
        expected = {
            "claim_intake", "security_gate", "privacy",
            "document_ocr", "fhir_validator", "identity_coverage",
            "medical_coding", "clinical_consistency", "fraud_detection",
            "case_reviewer",
        }
        assert expected <= steps

    def test_invoke_approve_path_interrupts_for_human_review(self, monkeypatch):
        """Pipeline nominal : case_reviewer APPROVE reste non final.

        ``case_reviewer`` traverse désormais l'orchestrateur (voir
        ``graph/nodes.py``) : l'injection se fait via ``case_reviewer_impl=``
        (mécanisme officiel), jamais en remplaçant une factory de nœud qui
        n'existe plus dans ``graph.workflow``."""
        _make_mock_agents(monkeypatch)

        app = compile_workflow(None, interrupt_before=[], case_reviewer_impl=_FakeCaseReviewer())
        result = app.invoke(_initial_state())
        steps = result.get("completed_steps", [])
        assert "__interrupt__" in result
        assert "needs_review" in steps
        assert "audit" not in steps
        assert "finalize" not in steps

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
        app = compile_workflow(None, interrupt_before=[])
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
        app = compile_workflow(None, interrupt_before=[])
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
        app = compile_workflow(saver, interrupt_before=[])
        thread_cfg = {"configurable": {"thread_id": "CLM-0003", "checkpoint_ns": ""}}
        app.invoke(_initial_state(), config=thread_cfg)
        checkpoint = saver.get(thread_cfg)
        assert checkpoint is not None
        assert "channel_values" in checkpoint
        assert "failure" in checkpoint["channel_values"].get("completed_steps", [])


# ── TestWorkflowHumanReview ───────────────────────────────────────────────────


def _make_short_needs_review_agents(monkeypatch) -> None:
    """Patche un chemin court et 100% sérialisable jusqu'à needs_review.

    document_ocr renvoie NEEDS_REVIEW, qui route directement vers
    needs_review sans passer par fhir_validator/identity_coverage/
    medical_coding/case_reviewer. Les résultats sont des ``_StubResult``
    (dataclass) plutôt que des ``SimpleNamespace`` : un vrai checkpointer
    (``InMemorySaver``) doit pouvoir sérialiser chaque étape intermédiaire,
    pas seulement l'état final.
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
            "security_result": _StubResult(decision=SecurityDecision.ALLOW),
            "security_input": None,
            "current_step": "security_gate",
            "completed_steps": ["security_gate"],
        }

    def mock_privacy(state: dict) -> dict:
        return {
            "privacy_result": _StubResult(decision=PrivacyDecision.ALLOW),
            "privacy_input": None,
            "current_step": "privacy",
            "completed_steps": ["privacy"],
        }

    def mock_document_ocr(state: dict) -> dict:
        return {
            "ocr_result": _StubResult(status=VerificationStatus.NEEDS_REVIEW),
            "ocr_input": None,
            "current_step": "document_ocr",
            "completed_steps": ["document_ocr"],
            "alerts": ["[document_ocr] confiance limite — revue requise."],
        }

    monkeypatch.setattr(wf, "node_claim_intake", mock_claim_intake)
    monkeypatch.setattr(wf, "node_security_gate", mock_security_gate)
    monkeypatch.setattr(wf, "node_privacy", mock_privacy)
    monkeypatch.setattr(wf, "node_document_ocr", mock_document_ocr)


class TestWorkflowHumanReview:
    """Vérifie le nœud await_human_review dans le pipeline complet.

    Chemin : claim_intake → security_gate → privacy → document_ocr
    (NEEDS_REVIEW) → needs_review → await_human_review.
    """

    def test_pipeline_interrupts_at_await_human_review(self, monkeypatch):
        _make_short_needs_review_agents(monkeypatch)
        saver = InMemorySaver()
        app = compile_workflow(saver, interrupt_before=[])
        config = {"configurable": {"thread_id": "CLM-0004", "checkpoint_ns": ""}}

        result = app.invoke(_initial_state(), config=config)

        assert "__interrupt__" in result
        payload = result["__interrupt__"][0].value
        assert payload["case_id"] == "CLM-0008"
        assert "[document_ocr] confiance limite — revue requise." in payload["motifs"]
        assert payload["actions_autorisees"] == ["APPROVE", "REJECT", "NEEDS_MORE_INFO"]
        assert "needs_review" in result.get("completed_steps", [])
        assert "await_human_review" not in result.get("completed_steps", [])

    def test_resume_with_same_thread_id_completes_pipeline(self, monkeypatch):
        from langgraph.types import Command

        _make_short_needs_review_agents(monkeypatch)
        saver = InMemorySaver()
        app = compile_workflow(saver, interrupt_before=[])
        config = {"configurable": {"thread_id": "CLM-0005", "checkpoint_ns": ""}}
        app.invoke(_initial_state(), config=config)

        result = app.invoke(
            Command(resume={"actor": "reviewer@example.com", "decision": "APPROVE"}),
            config=config,
        )

        assert "__interrupt__" not in result
        assert result["human_decision"]["decision"] == "APPROVE"
        assert "await_human_review" in result["completed_steps"]

    def test_resume_with_different_thread_id_does_not_resume(self, monkeypatch):
        """La reprise doit obligatoirement réutiliser le thread_id initial.

        Avec un thread_id différent, LangGraph ne retrouve aucun checkpoint
        en attente : il redémarre une exécution indépendante depuis START,
        qui réinterrompt aussitôt sur un dossier vierge — la décision fournie
        n'est jamais appliquée au dossier interrompu.
        """
        from langgraph.types import Command

        _make_short_needs_review_agents(monkeypatch)
        saver = InMemorySaver()
        app = compile_workflow(saver, interrupt_before=[])
        config = {"configurable": {"thread_id": "CLM-0006", "checkpoint_ns": ""}}
        app.invoke(_initial_state(), config=config)

        other_config = {"configurable": {"thread_id": "CLM-0007", "checkpoint_ns": ""}}
        result = app.invoke(
            Command(resume={"actor": "reviewer@example.com", "decision": "APPROVE"}),
            config=other_config,
        )
        assert "__interrupt__" in result
        assert "human_decision" not in result


# ── TestWorkflowRelaunchRoute ─────────────────────────────────────────────────


class TestWorkflowRelaunchRoute:
    """Vérifie la route « relancer » : reprise au nœud explicitement demandé,
    compteur de corrections et limite configurable.
    """

    def test_successful_correction_resumes_at_requested_node(self, monkeypatch):
        """NEEDS_MORE_INFO(target_node=document_ocr) relance bien document_ocr
        (2e appel observé) et le pipeline progresse au-delà de needs_review —
        ici jusqu'à fhir_validator, mocké pour échouer, ce qui prouve que la
        relance a repris l'exécution normale du graphe plutôt que de rester
        bloquée sur l'interruption.
        """
        from langgraph.types import Command

        call_count = {"ocr": 0}

        def mock_claim_intake(state: dict) -> dict:
            return {
                "intake_status": IntakeStatus.ACCEPTED,
                "intake_input": None,
                "current_step": "claim_intake",
                "completed_steps": ["claim_intake"],
            }

        def mock_security_gate(state: dict) -> dict:
            return {
                "security_result": _StubResult(decision=SecurityDecision.ALLOW),
                "security_input": None,
                "current_step": "security_gate",
                "completed_steps": ["security_gate"],
            }

        def mock_privacy(state: dict) -> dict:
            return {
                "privacy_result": _StubResult(decision=PrivacyDecision.ALLOW),
                "privacy_input": None,
                "current_step": "privacy",
                "completed_steps": ["privacy"],
            }

        def mock_document_ocr(state: dict) -> dict:
            call_count["ocr"] += 1
            if call_count["ocr"] == 1:
                return {
                    "ocr_result": _StubResult(status=VerificationStatus.NEEDS_REVIEW),
                    "ocr_input": None,
                    "current_step": "document_ocr",
                    "completed_steps": ["document_ocr"],
                    "alerts": ["[document_ocr] confiance limite — revue requise."],
                }
            # Correction appliquée par l'humain : le second passage réussit.
            return {
                "ocr_result": _StubResult(status=VerificationStatus.PASS),
                "ocr_input": None,
                "current_step": "document_ocr",
                "completed_steps": ["document_ocr"],
            }

        def mock_fhir_validator(state: dict) -> dict:
            return {
                "fhir_result": _StubResult(status=VerificationStatus.FAIL),
                "fhir_input": None,
                "current_step": "fhir_validator",
                "completed_steps": ["fhir_validator"],
            }

        monkeypatch.setattr(wf, "node_claim_intake", mock_claim_intake)
        monkeypatch.setattr(wf, "node_security_gate", mock_security_gate)
        monkeypatch.setattr(wf, "node_privacy", mock_privacy)
        monkeypatch.setattr(wf, "node_document_ocr", mock_document_ocr)
        monkeypatch.setattr(wf, "node_fhir_validator", mock_fhir_validator)

        saver = InMemorySaver()
        app = compile_workflow(saver, interrupt_before=[])
        config = {"configurable": {"thread_id": "CLM-0002", "checkpoint_ns": ""}}

        first = app.invoke(_initial_state(), config=config)
        assert "__interrupt__" in first
        assert call_count["ocr"] == 1

        result = app.invoke(
            Command(
                resume={
                    "actor": "reviewer@example.com",
                    "decision": "NEEDS_MORE_INFO",
                    "target_node": "document_ocr",
                }
            ),
            config=config,
        )

        assert call_count["ocr"] == 2
        assert result.get("correction_attempts") == 1
        assert "__interrupt__" not in result
        assert "document_ocr" in result.get("completed_steps", [])
        assert "failure" in result.get("completed_steps", [])
        assert result.get("final_recommendation") == Recommendation.REJECT

    def test_relaunch_beyond_limit_routes_to_failure(self, monkeypatch):
        """Deux demandes NEEDS_MORE_INFO successives avec max_correction_attempts=1 :
        la première est acceptée (attempts=1 <= 1), document_ocr signale de
        nouveau NEEDS_REVIEW ; la seconde dépasse la limite (attempts=2 > 1)
        et est routée vers failure sans relancer une 3e fois document_ocr.
        """
        from langgraph.types import Command

        call_count = {"ocr": 0}

        def mock_claim_intake(state: dict) -> dict:
            return {
                "intake_status": IntakeStatus.ACCEPTED,
                "intake_input": None,
                "current_step": "claim_intake",
                "completed_steps": ["claim_intake"],
            }

        def mock_security_gate(state: dict) -> dict:
            return {
                "security_result": _StubResult(decision=SecurityDecision.ALLOW),
                "security_input": None,
                "current_step": "security_gate",
                "completed_steps": ["security_gate"],
            }

        def mock_privacy(state: dict) -> dict:
            return {
                "privacy_result": _StubResult(decision=PrivacyDecision.ALLOW),
                "privacy_input": None,
                "current_step": "privacy",
                "completed_steps": ["privacy"],
            }

        def mock_document_ocr(state: dict) -> dict:
            call_count["ocr"] += 1
            return {
                "ocr_result": _StubResult(status=VerificationStatus.NEEDS_REVIEW),
                "ocr_input": None,
                "current_step": "document_ocr",
                "completed_steps": ["document_ocr"],
                "alerts": ["[document_ocr] confiance limite — revue requise."],
            }

        monkeypatch.setattr(wf, "node_claim_intake", mock_claim_intake)
        monkeypatch.setattr(wf, "node_security_gate", mock_security_gate)
        monkeypatch.setattr(wf, "node_privacy", mock_privacy)
        monkeypatch.setattr(wf, "node_document_ocr", mock_document_ocr)

        saver = InMemorySaver()
        app = compile_workflow(saver, interrupt_before=[], max_correction_attempts=1)
        config = {"configurable": {"thread_id": "CLM-0001", "checkpoint_ns": ""}}

        app.invoke(_initial_state(), config=config)
        assert call_count["ocr"] == 1

        resume_payload = Command(
            resume={
                "actor": "reviewer@example.com",
                "decision": "NEEDS_MORE_INFO",
                "target_node": "document_ocr",
            }
        )

        # 1re relance : attempts=1 <= max_attempts(1) → autorisée, document_ocr
        # signale de nouveau NEEDS_REVIEW → nouvelle interruption.
        second = app.invoke(resume_payload, config=config)
        assert call_count["ocr"] == 2
        assert second.get("correction_attempts") == 1
        assert "__interrupt__" in second

        # 2e relance : attempts=2 > max_attempts(1) → refusée, route vers failure
        # sans réexécuter document_ocr.
        third = app.invoke(resume_payload, config=config)
        assert call_count["ocr"] == 2
        assert third.get("correction_attempts") == 2
        assert "__interrupt__" not in third
        assert "failure" in third.get("completed_steps", [])
        assert third.get("final_recommendation") == Recommendation.REJECT


# ── TestWorkflowFutureAgentInjection ──────────────────────────────────────────


class _FakeCaseReviewer:
    """Implémentation factice conforme à CaseReviewerRunnable (méthode run())."""

    def run(self, state: dict) -> CaseReviewerResult:
        return CaseReviewerResult(
            case_id=str(state.get("case_id", "UNKNOWN")),
            recommendation=Recommendation.APPROVE,
            justification=["Approuvé par implémentation injectée."],
            human_review_required=False,
            human_review_reasons=[],
            llm_metadata=LlmMetadata(model_name="test-llm", prompt_version="test"),
        )


class TestWorkflowFutureAgentInjection:
    """Les agents futurs (clinical_consistency, fraud_detection, case_reviewer,
    audit) ne sont jamais importés en dur dans build_workflow() : ils sont
    construits via make_node_<nom>(impl) — None → stub, sinon l'implémentation
    injectée. Ce test prouve l'injection réelle de bout en bout (pas un
    monkeypatch de test) via case_reviewer_impl.
    """

    def test_default_impl_none_uses_real_reviewer_and_requires_human_review(self, monkeypatch):
        _make_mock_agents(monkeypatch)
        app = compile_workflow(None, interrupt_before=[])
        result = app.invoke(_initial_state())
        # Reviewer réel par défaut : pré-recommandation non finale → HITL.
        assert "needs_review" in result.get("completed_steps", [])

    def test_injected_case_reviewer_impl_is_used(self, monkeypatch):
        _make_mock_agents(monkeypatch)
        app = compile_workflow(
            None, interrupt_before=[], case_reviewer_impl=_FakeCaseReviewer()
        )
        result = app.invoke(_initial_state())
        assert result.get("final_recommendation") == Recommendation.APPROVE
        assert "needs_review" in result.get("completed_steps", [])
        assert "finalize" not in result.get("completed_steps", [])

    def test_workflow_module_has_no_default_future_agent_nodes(self):
        """Les 4 nœuds futurs ne sont jamais des instances de nœud
        pré-construites au niveau du module — leur nœud dépend de
        l'implémentation injectée, résolue à chaque construction via
        ``build_orchestrator``/``build_node_registry``."""
        for name in ("node_clinical_consistency", "node_fraud_detection",
                     "node_case_reviewer", "node_audit"):
            assert not hasattr(wf, name), f"{name} ne doit pas être importé en dur dans workflow.py"

    def test_workflow_module_imports_orchestrator_factories_not_hardcoded_nodes(self):
        """Les agents (7 réels + 4 stubs) ne sont jamais câblés en dur : ils
        passent par ``build_orchestrator``/``build_node_registry``
        (``graph/nodes.py``), jamais par une factory ``make_node_<nom>``
        dédiée par agent (mécanisme retiré avec l'introduction de
        l'orchestrateur)."""
        assert callable(getattr(wf, "build_orchestrator", None))
        assert callable(getattr(wf, "build_node_registry", None))
        for name in ("make_node_clinical_consistency", "make_node_fraud_detection",
                     "make_node_case_reviewer", "make_node_audit"):
            assert not hasattr(wf, name), f"{name} ne doit plus exister dans workflow.py"


# ── TestWorkflowBusinessOrder ─────────────────────────────────────────────────


class TestWorkflowBusinessOrder:
    """Épingle l'ordre métier du pipeline nominal (feuille de route)."""

    EXPECTED_NOMINAL_ORDER = [
        "claim_intake",
        "security_gate",
        "privacy",
        "document_ocr",
        "fhir_validator",
        "identity_coverage",
        "medical_coding",
        "clinical_consistency",
        "fraud_detection",
        "case_reviewer",
        "needs_review",
    ]

    def test_nominal_path_follows_business_order(self, monkeypatch):
        _make_mock_agents(monkeypatch)

        app = compile_workflow(None, interrupt_before=[], case_reviewer_impl=_FakeCaseReviewer())
        result = app.invoke(_initial_state())
        steps = result.get("completed_steps", [])

        # Chaque étape attendue avant validation humaine doit apparaître, dans
        # l'ordre métier défini par la feuille de route.
        observed_order = [step for step in steps if step in self.EXPECTED_NOMINAL_ORDER]
        assert observed_order == self.EXPECTED_NOMINAL_ORDER
        assert "audit" not in steps
        assert "finalize" not in steps


# ── TestWorkflowTopologyChecks ────────────────────────────────────────────────


class TestWorkflowTopologyChecks:
    """Vérification automatique des nœuds inaccessibles et des impasses."""

    def test_production_graph_has_no_unreachable_nodes(self):
        graph = build_workflow()
        assert find_unreachable_nodes(graph) == set()

    def test_production_graph_has_no_dead_end_nodes(self):
        graph = build_workflow()
        assert find_dead_end_nodes(graph) == set()

    def test_production_graph_has_no_isolated_nodes(self):
        graph = build_workflow()
        assert find_isolated_nodes(graph) == set()

    def test_production_graph_has_no_dangling_transitions(self):
        graph = build_workflow()
        assert find_dangling_transitions(graph) == []

    def test_detects_isolated_node(self):
        """Un nœud sans aucune entrée ni sortie est isolé — même s'il n'est
        pas nécessairement « inaccessible » au sens strict si on ne compte
        que les entrées (ici il n'a ni l'un ni l'autre)."""
        from langgraph.graph import END, START, StateGraph
        from state.claim_state import ClaimState

        graph = StateGraph(ClaimState)
        graph.add_node("claim_intake", lambda state: {})
        graph.add_node("island", lambda state: {})
        graph.add_edge(START, "claim_intake")
        graph.add_edge("claim_intake", END)

        assert find_isolated_nodes(graph) == {"island"}

    def test_node_with_only_outgoing_edge_is_not_isolated(self):
        """Une sortie valide suffit — même sans entrée, le nœud n'est pas
        isolé (mais il resterait inaccessible depuis START, un problème
        distinct détecté par find_unreachable_nodes)."""
        from langgraph.graph import END, START, StateGraph
        from state.claim_state import ClaimState

        graph = StateGraph(ClaimState)
        graph.add_node("claim_intake", lambda state: {})
        graph.add_node("floating", lambda state: {})
        graph.add_edge(START, "claim_intake")
        graph.add_edge("claim_intake", END)
        graph.add_edge("floating", END)  # sortie valide, aucune entrée

        assert find_isolated_nodes(graph) == set()
        assert find_unreachable_nodes(graph) == {"floating"}

    def test_detects_dangling_normal_edge(self):
        from langgraph.graph import START, StateGraph
        from state.claim_state import ClaimState

        graph = StateGraph(ClaimState)
        graph.add_node("claim_intake", lambda state: {})
        graph.add_edge(START, "claim_intake")
        graph.add_edge("claim_intake", "cliam_intake_typo")  # nœud jamais enregistré

        assert find_dangling_transitions(graph) == [("claim_intake", "cliam_intake_typo")]

    def test_detects_dangling_conditional_edge(self):
        from langgraph.graph import END, START, StateGraph
        from state.claim_state import ClaimState

        def _route(state: dict) -> str:
            return "ok"

        graph = StateGraph(ClaimState)
        graph.add_node("claim_intake", lambda state: {})
        graph.add_edge(START, "claim_intake")
        graph.add_conditional_edges(
            "claim_intake", _route, {"ok": END, "missing": "does_not_exist"}
        )

        assert find_dangling_transitions(graph) == [("claim_intake", "does_not_exist")]

    def test_build_workflow_raises_on_dangling_transition(self, monkeypatch):
        """build_workflow() échoue explicitement — pas de KeyError LangGraph
        opaque à la compilation, mais un ValueError clair à la construction.

        Simule une régression du path_map de la route de relance (un nom de
        nœud inexistant apparaîtrait dans RELAUNCH_TARGETS).
        """
        broken_targets = frozenset({"claim_intake", "nonexistent_node"})
        monkeypatch.setattr(wf, "RELAUNCH_TARGETS", broken_targets)

        with pytest.raises(ValueError, match="nœud absent"):
            build_workflow()

    def test_detects_orphan_unreachable_node(self):
        from langgraph.graph import END, START, StateGraph
        from state.claim_state import ClaimState

        graph = StateGraph(ClaimState)
        graph.add_node("claim_intake", lambda state: {})
        graph.add_node("orphan", lambda state: {})
        graph.add_edge(START, "claim_intake")
        graph.add_edge("claim_intake", END)
        # "orphan" est enregistré mais jamais raccordé par une arête.

        assert find_unreachable_nodes(graph) == {"orphan"}

    def test_detects_dead_end_node(self):
        from langgraph.graph import START, StateGraph
        from state.claim_state import ClaimState

        graph = StateGraph(ClaimState)
        graph.add_node("claim_intake", lambda state: {})
        graph.add_node("trap", lambda state: {})
        graph.add_edge(START, "claim_intake")
        graph.add_edge("claim_intake", "trap")
        # "trap" ne mène nulle part — et par transitivité, claim_intake non
        # plus, puisque sa seule destination est elle-même une impasse.

        assert find_dead_end_nodes(graph) == {"claim_intake", "trap"}

    def test_cycle_with_an_exit_is_not_a_dead_end(self):
        """Un cycle (ex. route de relance) n'est pas une impasse tant qu'une
        branche du cycle atteint END."""
        from langgraph.graph import END, START, StateGraph
        from state.claim_state import ClaimState

        def _route(state: dict) -> str:
            return state.get("route", "loop")

        graph = StateGraph(ClaimState)
        graph.add_node("a", lambda state: {})
        graph.add_node("b", lambda state: {})
        graph.add_edge(START, "a")
        graph.add_conditional_edges("a", _route, {"loop": "b", "exit": END})
        graph.add_edge("b", "a")  # cycle a -> b -> a

        assert find_dead_end_nodes(graph) == set()
        assert find_unreachable_nodes(graph) == set()

    def test_cycle_without_exit_is_a_dead_end(self):
        from langgraph.graph import START, StateGraph
        from state.claim_state import ClaimState

        graph = StateGraph(ClaimState)
        graph.add_node("a", lambda state: {})
        graph.add_node("b", lambda state: {})
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", "a")  # cycle sans issue

        assert find_dead_end_nodes(graph) == {"a", "b"}
        assert find_unreachable_nodes(graph) == set()

    def test_build_workflow_raises_on_unreachable_node(self, monkeypatch):
        """La vérification automatique est appelée à chaque construction du graphe."""
        from graph import workflow as workflow_module

        def _broken_assert(graph):
            raise ValueError("Topologie du workflow invalide : nœud(s) inaccessible(s)")

        monkeypatch.setattr(workflow_module, "_assert_workflow_topology_is_sound", _broken_assert)
        try:
            build_workflow()
            assert False, "build_workflow() aurait dû lever ValueError"
        except ValueError as exc:
            assert "inaccessible" in str(exc)


# ── TestBuildCompileSplit ──────────────────────────────────────────────────────


class TestBuildCompileSplit:
    """build_workflow() construit (StateGraph non compilé) ; compile_workflow()
    compile un graphe (construit sur place, ou fourni via graph=...)."""

    def test_build_workflow_returns_state_graph(self):
        from langgraph.graph import StateGraph

        graph = build_workflow()
        assert isinstance(graph, StateGraph)

    def test_build_workflow_graph_has_all_nodes(self):
        graph = build_workflow()
        assert set(graph.nodes) == _ALL_NODES - {"__start__"}

    def test_build_workflow_graph_is_not_compiled(self):
        graph = build_workflow()
        assert not hasattr(graph, "invoke")

    def test_compile_workflow_without_graph_builds_one(self):
        app = compile_workflow(interrupt_before=[])
        assert callable(app.invoke)
        assert set(app.nodes) == _ALL_NODES

    def test_compile_workflow_reuses_provided_graph(self):
        graph = build_workflow()
        app = compile_workflow(graph=graph, interrupt_before=[])
        assert app.builder is graph

    def test_compile_workflow_applies_checkpointer_to_provided_graph(self):
        graph = build_workflow()
        saver = InMemorySaver()
        app = compile_workflow(saver, graph=graph, interrupt_before=[])
        assert app.checkpointer is saver

    def test_compile_workflow_ignores_construction_params_when_graph_given(self, monkeypatch):
        """max_correction_attempts / *_impl ne s'appliquent qu'à la
        construction — un graph déjà construit les ignore silencieusement."""
        build_calls = {"count": 0}
        original_build_workflow = wf.build_workflow

        def _counting_build_workflow(**kwargs):
            build_calls["count"] += 1
            return original_build_workflow(**kwargs)

        monkeypatch.setattr(wf, "build_workflow", _counting_build_workflow)
        graph = build_workflow()
        compile_workflow(graph=graph, interrupt_before=[], max_correction_attempts=1)
        assert build_calls["count"] == 0


# ── TestWorkflowMermaid ────────────────────────────────────────────────────────


class TestWorkflowMermaid:
    def test_returns_mermaid_text(self):
        mermaid = get_workflow_mermaid()
        assert isinstance(mermaid, str)
        assert "graph TD" in mermaid

    def test_contains_all_node_names(self):
        mermaid = get_workflow_mermaid()
        for node in _ALL_NODES - {"__start__"}:
            assert node in mermaid

    def test_accepts_precompiled_app(self):
        app = compile_workflow(interrupt_before=[])
        mermaid = get_workflow_mermaid(app)
        assert "claim_intake" in mermaid

    def test_contains_no_sensitive_data(self):
        """Le diagramme ne décrit que la structure — jamais de case_id, de
        chemin de fichier, de clé ou de donnée patient."""
        mermaid = get_workflow_mermaid()
        forbidden_markers = ("CLM-", "PSE-", "PAT-", "storage/", "pseudonymization")
        for marker in forbidden_markers:
            assert marker not in mermaid
