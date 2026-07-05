"""Interruption dynamique et reprise — await_human_review — ClaimShield Santé.

Scénario HITL complet, avec un vrai checkpointer (``InMemorySaver``) :
  1. Le pipeline s'interrompt sur ``await_human_review`` (via
     ``langgraph.types.interrupt()``, déclenché par un statut NEEDS_REVIEW).
  2. La reprise se fait avec ``Command(resume=...)`` et **exactement** le
     même ``thread_id`` que l'invocation initiale.
  3. Les étapes déjà validées (claim_intake, security_gate, privacy,
     document_ocr) ne sont jamais rejouées lors de la reprise — vérifié par
     compteurs d'appels sur de faux agents déterministes (aucun LLM).
  4. La décision humaine est intégrée dans le state et détermine
     effectivement la suite du pipeline (APPROVE → END, REJECT → failure) —
     la reprise ne suit pas un chemin fixe indépendant de la décision.
  5. Un ``thread_id`` différent ne récupère jamais le state interrompu : il
     redémarre une exécution indépendante depuis START, sans toucher au
     dossier original persistant sous l'ancien thread_id.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import graph.workflow as wf
from graph.checkpoints import make_thread_config
from graph.workflow import compile_workflow
from schemas.domain import (
    IntakeStatus,
    PrivacyDecision,
    Recommendation,
    SecurityDecision,
    VerificationStatus,
)

_UPSTREAM_NODES: tuple[str, ...] = ("claim_intake", "security_gate", "privacy", "document_ocr")


@dataclass
class _StubResult:
    decision: Any = None
    status: Any = None


def _initial_state(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "schema_version": "1.0.0",
        "current_step": "initial",
        "completed_steps": [],
        "errors": [],
        "alerts": [],
        "final_justification": [],
    }


def _install_agents_reaching_needs_review(monkeypatch, call_counts: dict[str, int]) -> None:
    """Patche 4 faux agents déterministes menant à needs_review — jamais de LLM.

    ``call_counts`` est incrémenté à chaque exécution réelle : sert de preuve
    indépendante qu'une étape n'est pas rejouée après une reprise.
    """

    def mock_claim_intake(state: dict) -> dict:
        call_counts["claim_intake"] += 1
        return {
            "intake_status": IntakeStatus.ACCEPTED,
            "intake_input": None,
            "current_step": "claim_intake",
            "completed_steps": ["claim_intake"],
        }

    def mock_security_gate(state: dict) -> dict:
        call_counts["security_gate"] += 1
        return {
            "security_result": _StubResult(decision=SecurityDecision.ALLOW),
            "security_input": None,
            "current_step": "security_gate",
            "completed_steps": ["security_gate"],
        }

    def mock_privacy(state: dict) -> dict:
        call_counts["privacy"] += 1
        return {
            "privacy_result": _StubResult(decision=PrivacyDecision.ALLOW),
            "privacy_input": None,
            "current_step": "privacy",
            "completed_steps": ["privacy"],
        }

    def mock_document_ocr(state: dict) -> dict:
        call_counts["document_ocr"] += 1
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


def _build_app(monkeypatch) -> tuple:
    call_counts = {name: 0 for name in _UPSTREAM_NODES}
    _install_agents_reaching_needs_review(monkeypatch, call_counts)
    saver = InMemorySaver()
    app = compile_workflow(saver, interrupt_before=[])
    return app, saver, call_counts


class TestInterruptAndResume:
    """Interruption sur await_human_review, reprise avec le même thread_id."""

    def test_interrupts_at_await_human_review(self, monkeypatch):
        app, _, call_counts = _build_app(monkeypatch)
        config = make_thread_config("CLM-0001")

        result = app.invoke(_initial_state("CLM-0001"), config=config)

        assert "__interrupt__" in result
        assert "needs_review" in result.get("completed_steps", [])
        assert "await_human_review" not in result.get("completed_steps", [])
        for name in _UPSTREAM_NODES:
            assert call_counts[name] == 1

    def test_resume_same_thread_id_does_not_replay_validated_steps(self, monkeypatch):
        app, _, call_counts = _build_app(monkeypatch)
        config = make_thread_config("CLM-0002")

        app.invoke(_initial_state("CLM-0002"), config=config)
        counts_before_resume = dict(call_counts)

        result = app.invoke(
            Command(resume={"actor": "reviewer@example.com", "decision": "APPROVE"}),
            config=config,
        )

        # Aucune étape déjà validée n'est rejouée : les compteurs d'appels
        # restent strictement identiques après la reprise.
        assert call_counts == counts_before_resume
        for name in _UPSTREAM_NODES:
            assert call_counts[name] == 1

        # completed_steps ne contient chaque étape amont qu'une seule fois —
        # pas de doublon dû à une relecture.
        for name in _UPSTREAM_NODES:
            assert result["completed_steps"].count(name) == 1

        assert "__interrupt__" not in result

    def test_human_decision_is_integrated_before_continuation(self, monkeypatch):
        """La décision humaine détermine effectivement la suite : APPROVE
        termine directement à END, sans passer par failure."""
        app, _, _ = _build_app(monkeypatch)
        config = make_thread_config("CLM-0003")
        app.invoke(_initial_state("CLM-0003"), config=config)

        result = app.invoke(
            Command(resume={"actor": "reviewer@example.com", "decision": "APPROVE"}),
            config=config,
        )

        assert result["human_decision"]["decision"] == "APPROVE"
        assert result["human_decision"]["actor"] == "reviewer@example.com"
        assert "await_human_review" in result["completed_steps"]
        assert "__interrupt__" not in result
        assert "failure" not in result["completed_steps"]

    def test_reject_decision_routes_to_failure_instead_of_approve(self, monkeypatch):
        """Preuve complémentaire que la décision pilote réellement la suite :
        une décision différente (REJECT) produit un chemin différent."""
        app, _, _ = _build_app(monkeypatch)
        config = make_thread_config("CLM-0004")
        app.invoke(_initial_state("CLM-0004"), config=config)

        result = app.invoke(
            Command(resume={"actor": "reviewer@example.com", "decision": "REJECT"}),
            config=config,
        )

        assert result["human_decision"]["decision"] == "REJECT"
        assert "failure" in result["completed_steps"]
        assert result.get("final_recommendation") == Recommendation.REJECT

    def test_different_thread_id_does_not_recover_previous_state(self, monkeypatch):
        app, saver, _ = _build_app(monkeypatch)
        original_config = make_thread_config("CLM-0005")
        app.invoke(_initial_state("CLM-0005"), config=original_config)

        other_config = make_thread_config("CLM-0006")
        result = app.invoke(
            Command(resume={"actor": "reviewer@example.com", "decision": "APPROVE"}),
            config=other_config,
        )

        # Nouveau thread : aucun checkpoint en attente n'est retrouvé —
        # LangGraph redémarre une exécution neuve depuis START, qui
        # réinterrompt aussitôt sur un dossier vierge (case_id "INCONNU").
        assert "__interrupt__" in result
        assert "human_decision" not in result
        payload = result["__interrupt__"][0].value
        assert payload["case_id"] == "INCONNU"

        # Le dossier original, persistant sous son propre thread_id, reste
        # inchangé : ni la décision fournie, ni l'exécution sur l'autre
        # thread ne l'ont affecté.
        original_checkpoint = saver.get(original_config)
        assert original_checkpoint is not None
        assert original_checkpoint["channel_values"].get("case_id") == "CLM-0005"
        assert original_checkpoint["channel_values"].get("human_decision") is None
