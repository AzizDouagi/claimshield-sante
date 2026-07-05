"""Panne agent non gérée (exception simulant une panne LLM) — graph/workflow.py.

Les agents réels gèrent déjà en interne l'indisponibilité du LLM par des
replis contrôlés (voir ``tests/agents/test_*_llm.py`` — BLOCK conservateur,
FAIL, NEEDS_REVIEW selon l'agent). Ce fichier couvre le cas plus grave où un
agent **lève une exception non gérée** (panne LLM non rattrapée en interne,
bug, timeout réseau non catégorisé, etc.) — le filet de sécurité testé ici
est celui de ``graph/nodes.py::_make_node``, exercé au niveau du graphe
complet plutôt qu'en isolation (voir ``tests/graph/test_nodes.py`` pour la
version unitaire du même mécanisme).

Vérifie :
  - le pipeline ne plante jamais : ``app.invoke()`` ne lève pas ;
  - l'erreur capturée est structurée (préfixe ``[<agent>]``, type et message
    de l'exception) et associée au bon nœud, pas à un autre ;
  - le workflow se termine explicitement (route ``failure`` → ``END``,
    ``final_recommendation = REJECT``) — jamais de blocage silencieux ;
  - le dernier checkpoint valide (juste avant la panne) reste disponible et
    intact dans l'historique du checkpointer ;
  - l'agent en panne n'est appelé qu'une seule fois pour ce scénario précis
    (``RuntimeError`` — panne non catégorisée) : le retry technique
    automatique introduit à l'étape 11 (``graph/nodes.py::_invoke_agent_node``)
    ne couvre que les erreurs réseau/timeout transitoires
    (``_TRANSIENT_NODE_EXCEPTIONS``) et laisse ce scénario inchangé — voir
    ``tests/graph/test_nodes.py::TestNodeRetryIntegration`` pour le cas
    couvert par le retry.
"""
from __future__ import annotations

from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver

import graph.workflow as wf
from graph.checkpoints import make_thread_config
from graph.workflow import compile_workflow
from schemas.domain import IntakeStatus, Recommendation

_OLLAMA_FAILURE_MESSAGE = "Ollama indisponible : connection refused"


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


def _mock_claim_intake_success(state: dict) -> dict:
    return {
        "intake_status": IntakeStatus.ACCEPTED,
        "intake_input": None,
        "current_step": "claim_intake",
        "completed_steps": ["claim_intake"],
    }


class TestLlmFailurePath:
    """security_gate_agent lève une exception non gérée (panne LLM simulée).

    claim_intake reste un faux agent déterministe (succès, aucun LLM) pour
    garantir qu'un checkpoint valide existe avant la panne.
    """

    def _build_app(self, monkeypatch) -> tuple:
        monkeypatch.setattr(wf, "node_claim_intake", _mock_claim_intake_success)
        saver = InMemorySaver()
        app = compile_workflow(saver, interrupt_before=[])
        return app, saver

    def test_invoke_does_not_raise_and_terminates_via_failure(self, monkeypatch):
        app, _ = self._build_app(monkeypatch)
        case_id = "CLM-0001"
        config = make_thread_config(case_id)

        with patch(
            "agents.security_gate_agent.agent.node",
            side_effect=RuntimeError(_OLLAMA_FAILURE_MESSAGE),
        ):
            result = app.invoke(_initial_state(case_id), config=config)  # ne doit jamais lever

        # Terminaison explicite : ni interruption silencieuse, ni blocage.
        assert "__interrupt__" not in result
        assert result.get("current_step") == "failure"
        assert "failure" in result.get("completed_steps", [])
        assert result.get("final_recommendation") == Recommendation.REJECT

    def test_error_is_structured_and_names_the_failing_node(self, monkeypatch):
        app, _ = self._build_app(monkeypatch)
        case_id = "CLM-0002"
        config = make_thread_config(case_id)

        with patch(
            "agents.security_gate_agent.agent.node",
            side_effect=RuntimeError(_OLLAMA_FAILURE_MESSAGE),
        ):
            result = app.invoke(_initial_state(case_id), config=config)

        errors = result.get("errors", [])
        agent_errors = [e for e in errors if "security_gate" in e]

        assert len(agent_errors) == 1, f"Erreur security_gate absente ou dupliquée : {errors}"
        message = agent_errors[0]
        assert "RuntimeError" in message
        assert _OLLAMA_FAILURE_MESSAGE in message
        # Associée au bon nœud uniquement — pas confondue avec claim_intake.
        assert "claim_intake" not in message

    def test_last_valid_checkpoint_before_failure_remains_available(self, monkeypatch):
        app, _ = self._build_app(monkeypatch)
        case_id = "CLM-0003"
        config = make_thread_config(case_id)

        with patch(
            "agents.security_gate_agent.agent.node",
            side_effect=RuntimeError(_OLLAMA_FAILURE_MESSAGE),
        ):
            app.invoke(_initial_state(case_id), config=config)

        history = list(app.get_state_history(config))
        assert history, "Aucun checkpoint retrouvé après la panne"

        # Le checkpoint le plus récent (historique en ordre décroissant) est
        # l'état de panne — toujours valide et chargeable, non corrompu.
        latest = history[0]
        assert latest.values.get("current_step") == "failure"
        assert latest.values.get("intake_status") == IntakeStatus.ACCEPTED

        # Le checkpoint juste avant la panne (après claim_intake, avant
        # security_gate) reste disponible et intact — c'est le point de
        # reprise qu'utiliserait un retry (le retry technique de l'étape 11
        # rejoue l'appel agent en interne, sans nouveau checkpoint
        # intermédiaire ; ce checkpoint reste la garantie de reprise pour
        # toute relance de plus haut niveau, ex. HITL « relancer »).
        last_good = next(
            (snap for snap in history if snap.values.get("current_step") == "claim_intake"),
            None,
        )
        assert last_good is not None, "Le checkpoint post-claim_intake a disparu"
        assert last_good.values.get("intake_status") == IntakeStatus.ACCEPTED
        assert last_good.values.get("security_result") is None
        assert "failure" not in last_good.values.get("completed_steps", [])
        assert "security_gate" not in last_good.values.get("completed_steps", [])

    def test_failing_agent_is_not_retried_automatically(self, monkeypatch):
        """``RuntimeError`` est une panne non catégorisée : hors du périmètre
        du retry technique de l'étape 11 (``_TRANSIENT_NODE_EXCEPTIONS``),
        donc toujours aucun retry automatique ici."""
        app, _ = self._build_app(monkeypatch)
        case_id = "CLM-0004"
        config = make_thread_config(case_id)
        call_count = {"security_gate": 0}

        def _raising_node(state: dict):
            call_count["security_gate"] += 1
            raise RuntimeError(_OLLAMA_FAILURE_MESSAGE)

        with patch("agents.security_gate_agent.agent.node", side_effect=_raising_node):
            app.invoke(_initial_state(case_id), config=config)

        assert call_count["security_gate"] == 1
