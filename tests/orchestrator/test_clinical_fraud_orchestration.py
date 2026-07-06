"""Orchestration bout-en-bout — clinical_consistency_agent et fraud_detection_agent.

Complète les tests génériques déjà existants
(``test_policies.py::TestRefusalScenariosEndToEnd``,
``test_execution.py``/``test_executor.py``) par des preuves concrètes,
nommément pour ces deux agents (les seuls, avec ``medical_coding_agent`` et
``document_ocr_agent``, à exiger ``ModelCapability.TOOL_CALLING``) que le
contrat commun de l'orchestrateur s'applique bien à leur enregistrement réel
(``AgentName``, ``AGENT_RESULT_MODELS``, ``AGENT_REQUIRED_CAPABILITIES``,
``ALLOWED_TOOLS_PER_AGENT``) :

  1. outil refusé  → agent jamais appelé (fail-closed, ``NO_AUTHORIZED_TOOLS``) ;
  2. sortie invalide → jamais acceptée comme résultat (``AGENT_RESULT_INVALID``/
     ``AGENT_RESULT_UNSTRUCTURED``) ;
  3. panne modèle  → jamais avalée silencieusement (``AGENT_EXECUTION_FAILED``,
     retry selon ``RetryPolicy`` puis échec visible, ou modèle incompatible
     refusé avant tout appel).

Chaque appel passe exclusivement par ``Orchestrator.execute_agent()`` — ni
``agents.clinical_consistency_agent.agent.run()``/``.node()`` ni
``agents.fraud_detection_agent.agent.run()``/``.node()`` n'est jamais appelé
directement ici (voir aussi la vérification statique
``tests/graph/test_architecture.py``, qui garantit qu'aucun appel direct
n'existe non plus dans ``graph/nodes.py`` en production).
"""
from __future__ import annotations

import httpx
import pytest

from orchestrator.executor import Orchestrator, RetryPolicy
from orchestrator.model_registry import (
    ModelCapability,
    ModelRegistry,
    ModelSpec,
    build_default_registry,
)
from orchestrator.orchestrator import AgentCallRequest, AgentName
from orchestrator.policies import PolicyDecision, PolicyEffect, build_authorized_tools

# ── Helpers ────────────────────────────────────────────────────────────────────

AGENT_CASES = [
    (AgentName.CLINICAL_CONSISTENCY, "medical_coding", "ClinicalConsistencyResult", "clinical_result"),
    (AgentName.FRAUD_DETECTION, "clinical_consistency", "FraudDetectionResult", "fraud_result"),
]
AGENT_IDS = [case[0].value for case in AGENT_CASES]


def _request(agent_name: AgentName, current_step: str, requested_model: str) -> AgentCallRequest:
    return AgentCallRequest(
        agent_name=agent_name,
        case_id="CLM-0001",
        current_step=current_step,
        requested_model=requested_model,
        attempt=1,
    )


def _allow(code: str = "OK") -> PolicyDecision:
    from schemas.results import StructuredError

    return PolicyDecision(effect=PolicyEffect.ALLOW, reason=StructuredError(code=code, message="ok"))


def _refusal_events(outcome) -> list:
    return [event for event in outcome.audit_events if event.action == "refusal"]


# ── 1. Outil refusé — fail-closed ────────────────────────────────────────────


class TestToolRefusedFailsClosed:
    @pytest.mark.parametrize(("agent_name", "step", "model_name", "_field"), AGENT_CASES, ids=AGENT_IDS)
    def test_agent_never_called_without_authorized_tool(self, agent_name, step, model_name, _field):
        """Sans outil autorisé, l'agent (qui exige TOOL_CALLING) n'est
        jamais appelé — refus fail-closed avant tout appel."""
        registry = build_default_registry()
        model_id = registry.list_models()[0].model_id
        calls: list[str] = []
        orchestrator = Orchestrator(
            model_registry=registry,
            agent_registry={agent_name: lambda state: calls.append("called") or {}},
            preconditions_check=lambda state, request: _allow(),
            tools_check=lambda name: (),
        )
        request = _request(agent_name, step, model_name)
        state = {"case_id": "CLM-0001", "current_step": step}

        outcome = orchestrator.execute_agent(request, state, model_id=model_id)

        assert outcome.success is False
        assert outcome.error.code == "NO_AUTHORIZED_TOOLS"
        assert calls == [], "l'agent n'aurait jamais dû être appelé sans outil autorisé"
        refusals = _refusal_events(outcome)
        assert refusals
        assert refusals[-1].details["policy"] == "NO_AUTHORIZED_TOOLS"
        assert refusals[-1].actor == agent_name.value

    @pytest.mark.parametrize(("agent_name", "step", "model_name", "_field"), AGENT_CASES, ids=AGENT_IDS)
    def test_real_authorized_tools_are_never_empty(self, agent_name, step, model_name, _field):
        """Contrôle de cohérence : en configuration réelle (pas de double),
        l'agent dispose bien d'au moins un outil autorisé — le refus
        ci-dessus provient donc bien du double injecté, pas d'un oubli de
        câblage réel."""
        assert build_authorized_tools(agent_name), (
            f"{agent_name.value} devrait avoir au moins un outil autorisé en configuration réelle"
        )


# ── 2. Sortie invalide — jamais acceptée ─────────────────────────────────────


class TestInvalidOutputNeverAccepted:
    def _permissive_orchestrator(self, agent_registry: dict, *, retry_policy: RetryPolicy | None = None) -> Orchestrator:
        return Orchestrator(
            model_registry=ModelRegistry(),
            agent_registry=agent_registry,
            preconditions_check=lambda state, request: _allow(),
            model_check=lambda registry, agent_name, model_id: _allow(),
            tools_check=lambda agent_name: build_authorized_tools(agent_name),
            retry_policy=retry_policy or RetryPolicy(),
        )

    @pytest.mark.parametrize(("agent_name", "step", "model_name", "field"), AGENT_CASES, ids=AGENT_IDS)
    def test_incomplete_dict_is_rejected(self, agent_name, step, model_name, field):
        """Un dict incomplet (llm_trace manquant) n'est jamais accepté tel
        quel comme résultat final."""
        orchestrator = self._permissive_orchestrator(
            {agent_name: lambda state: {field: {"case_id": state["case_id"], "status": "PASS"}}}
        )
        request = _request(agent_name, step, model_name)
        state = {"case_id": "CLM-0001", "current_step": step}

        outcome = orchestrator.execute_agent(request, state, model_id="whatever")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_RESULT_INVALID"
        assert outcome.result_payload is None

    @pytest.mark.parametrize(("agent_name", "step", "model_name", "field"), AGENT_CASES, ids=AGENT_IDS)
    def test_free_text_output_is_rejected(self, agent_name, step, model_name, field):
        """Un texte libre à la place d'un dict/instance Pydantic n'est
        jamais accepté — jamais tenté en model_validate()."""
        orchestrator = self._permissive_orchestrator(
            {agent_name: lambda state: {field: "le dossier semble correct"}}
        )
        request = _request(agent_name, step, model_name)
        state = {"case_id": "CLM-0001", "current_step": step}

        outcome = orchestrator.execute_agent(request, state, model_id="whatever")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_RESULT_UNSTRUCTURED"
        assert outcome.result_payload is None

    @pytest.mark.parametrize(("agent_name", "step", "model_name", "field"), AGENT_CASES, ids=AGENT_IDS)
    def test_missing_result_field_is_rejected(self, agent_name, step, model_name, field):
        """L'agent retourne un dict qui ne contient même pas le champ
        résultat attendu — jamais un succès fabriqué à partir de rien."""
        orchestrator = self._permissive_orchestrator({agent_name: lambda state: {}})
        request = _request(agent_name, step, model_name)
        state = {"case_id": "CLM-0001", "current_step": step}

        outcome = orchestrator.execute_agent(request, state, model_id="whatever")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_RESULT_MISSING"


# ── 3. Panne modèle — jamais avalée silencieusement ──────────────────────────


class TestModelFailureNeverSwallowed:
    def _permissive_orchestrator(self, agent_registry: dict, *, retry_policy: RetryPolicy) -> Orchestrator:
        return Orchestrator(
            model_registry=ModelRegistry(),
            agent_registry=agent_registry,
            preconditions_check=lambda state, request: _allow(),
            model_check=lambda registry, agent_name, model_id: _allow(),
            tools_check=lambda agent_name: build_authorized_tools(agent_name),
            retry_policy=retry_policy,
        )

    @pytest.mark.parametrize(("agent_name", "step", "model_name", "_field"), AGENT_CASES, ids=AGENT_IDS)
    def test_transient_failure_is_retried_then_surfaced(self, agent_name, step, model_name, _field):
        """Panne transitoire (ex. Ollama injoignable) : rejouée selon
        RetryPolicy, puis visible telle quelle si elle persiste — jamais
        convertie en succès fabriqué ni masquée."""
        calls = {"count": 0}

        def _agent(state: dict) -> dict:
            calls["count"] += 1
            raise httpx.ConnectError("connexion refusée : Ollama indisponible")

        orchestrator = self._permissive_orchestrator(
            {agent_name: _agent}, retry_policy=RetryPolicy(max_attempts=3)
        )
        request = _request(agent_name, step, model_name)
        state = {"case_id": "CLM-0001", "current_step": step}

        outcome = orchestrator.execute_agent(request, state, model_id="whatever")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_EXECUTION_FAILED"
        assert "ConnectError" in outcome.error.message
        assert calls["count"] == 3, "doit être rejouée jusqu'à épuisement de RetryPolicy"

    @pytest.mark.parametrize(("agent_name", "step", "model_name", "_field"), AGENT_CASES, ids=AGENT_IDS)
    def test_non_transient_failure_is_never_retried(self, agent_name, step, model_name, _field):
        """Une panne non catégorisée (bug, valeur invalide) n'est jamais
        rejouée — un seul appel malgré une limite de retry élevée."""
        calls = {"count": 0}

        def _agent(state: dict) -> dict:
            calls["count"] += 1
            raise RuntimeError("erreur de programmation inattendue")

        orchestrator = self._permissive_orchestrator(
            {agent_name: _agent}, retry_policy=RetryPolicy(max_attempts=5)
        )
        request = _request(agent_name, step, model_name)
        state = {"case_id": "CLM-0001", "current_step": step}

        outcome = orchestrator.execute_agent(request, state, model_id="whatever")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_EXECUTION_FAILED"
        assert calls["count"] == 1

    @pytest.mark.parametrize(("agent_name", "step", "model_name", "_field"), AGENT_CASES, ids=AGENT_IDS)
    def test_incompatible_model_blocks_before_any_call(self, agent_name, step, model_name, _field):
        """Modèle incompatible (ne déclare pas TOOL_CALLING) : ni le client
        du modèle ni l'agent ne sont jamais invoqués."""
        client_factory_calls: list[str] = []

        def _client_factory():
            client_factory_calls.append("called")
            return object()

        registry = ModelRegistry()
        registry.register(
            ModelSpec(
                model_id="structured-only",
                provider="ollama",
                capabilities=frozenset({ModelCapability.STRUCTURED_OUTPUT}),
                client_factory=_client_factory,
            )
        )
        agent_calls: list[str] = []
        orchestrator = Orchestrator(
            model_registry=registry,
            agent_registry={agent_name: lambda state: agent_calls.append("called") or {}},
            preconditions_check=lambda state, request: _allow(),
        )
        request = _request(agent_name, step, model_name)
        state = {"case_id": "CLM-0001", "current_step": step}

        outcome = orchestrator.execute_agent(request, state, model_id="structured-only")

        assert outcome.success is False
        assert outcome.error.code == "MODEL_INCOMPATIBLE"
        assert client_factory_calls == [], "le client du modèle refusé n'aurait jamais dû être instancié"
        assert agent_calls == [], "l'agent n'aurait jamais dû être appelé"
