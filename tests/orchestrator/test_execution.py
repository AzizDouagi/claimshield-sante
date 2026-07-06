"""Tests des erreurs d'exécution — orchestrator/executor.py.

Trois natures de panne à l'exécution d'un agent, distinctes des refus
d'autorisation (voir ``test_policies.py::TestRefusalScenariosEndToEnd``) :
  1. sortie Pydantic invalide (dict incomplet ou texte libre) ;
  2. panne modèle (exception levée par l'agent en train d'appeler son LLM —
     transitoire ou non — et, en amont, indisponibilité du modèle demandé
     avec ou sans fallback autorisé) ;
  3. exception d'un outil authentiquement autorisé (confirmé via
     ``evaluate_tool_authorization``/``get_authorized_tool`` avant d'être
     exercé), levée pendant son exécution par l'agent.

Pour chacune : le comportement de retry (``RetryPolicy``) et sa limite sont
vérifiés, le fallback modèle est exercé quand il est permis, le dernier échec
reste visible (jamais remplacé par un résumé générique) et correctement
attribué (agent, case_id, tentative), et aucune exception n'est jamais
avalée ni convertie en un faux résultat métier.
"""
from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from orchestrator.executor import Orchestrator, RetryPolicy
from orchestrator.model_registry import ModelCapability, ModelRegistry, ModelSpec, build_default_registry
from orchestrator.orchestrator import AgentCallRequest, AgentName
from orchestrator.policies import (
    PolicyDecision,
    PolicyEffect,
    build_authorized_tools,
    evaluate_tool_authorization,
    get_authorized_tool,
)
from schemas.results import LlmMetadata, MedicalCodingResult, SecurityGateResult, StructuredError


# ── Helpers ────────────────────────────────────────────────────────────────────


def _request(agent_name: AgentName, current_step: str, requested_model: str, **overrides) -> AgentCallRequest:
    fields = {
        "agent_name": agent_name,
        "case_id": "CLM-0001",
        "current_step": current_step,
        "requested_model": requested_model,
        "attempt": 1,
    }
    fields.update(overrides)
    return AgentCallRequest(**fields)


def _allow(code: str = "OK") -> PolicyDecision:
    return PolicyDecision(effect=PolicyEffect.ALLOW, reason=StructuredError(code=code, message="ok"))


def _permissive_orchestrator(agent_registry: dict, *, retry_policy: RetryPolicy) -> Orchestrator:
    """Orchestrateur dont les trois contrôles amont (préconditions, modèle,
    outils) autorisent toujours — permet d'isoler exclusivement le
    comportement d'exécution (retry, panne, fallback) de l'agent lui-même."""
    return Orchestrator(
        model_registry=ModelRegistry(),
        agent_registry=agent_registry,
        preconditions_check=lambda state, request: _allow(),
        model_check=lambda registry, agent_name, model_id: _allow(),
        tools_check=lambda agent_name: (),
        retry_policy=retry_policy,
    )


def _permissive_orchestrator_with_real_tools(agent_registry: dict, *, retry_policy: RetryPolicy) -> Orchestrator:
    """Comme ``_permissive_orchestrator``, mais avec les outils réellement
    autorisés (``build_authorized_tools``) — nécessaire pour exercer un
    agent ``TOOL_CALLING`` (ex. medical_coding) sans le bloquer sur
    ``NO_AUTHORIZED_TOOLS``."""
    return Orchestrator(
        model_registry=ModelRegistry(),
        agent_registry=agent_registry,
        preconditions_check=lambda state, request: _allow(),
        model_check=lambda registry, agent_name, model_id: _allow(),
        tools_check=build_authorized_tools,
        retry_policy=retry_policy,
    )


def _valid_security_result(state: dict) -> dict:
    return {"security_result": SecurityGateResult(claim_id=state["case_id"], decision="ALLOW", reasons=["ok"])}


def _valid_coding_result(state: dict) -> dict:
    return {
        "coding_result": MedicalCodingResult(
            case_id=state["case_id"],
            status="PASS",
            llm_metadata=LlmMetadata(model_name="test-llm", prompt_version="test"),
        )
    }


# ── 1. Sortie Pydantic invalide ───────────────────────────────────────────────


class TestInvalidPydanticOutput:
    def test_incomplete_dict_is_retried_then_succeeds(self):
        calls = {"count": 0}

        def flaky(state):
            calls["count"] += 1
            if calls["count"] == 1:
                return {"security_result": {"claim_id": state["case_id"], "decision": "ALLOW"}}  # champ manquant
            return _valid_security_result(state)

        orchestrator = _permissive_orchestrator(
            {AgentName.SECURITY_GATE: flaky}, retry_policy=RetryPolicy(max_attempts=2)
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is True
        assert calls["count"] == 2
        assert outcome.result_payload["decision"] == "ALLOW"

    def test_free_text_output_is_retried_then_succeeds(self):
        calls = {"count": 0}

        def flaky(state):
            calls["count"] += 1
            if calls["count"] == 1:
                return {"security_result": "ALLOW, tout va bien, aucune anomalie."}
            return _valid_security_result(state)

        orchestrator = _permissive_orchestrator(
            {AgentName.SECURITY_GATE: flaky}, retry_policy=RetryPolicy(max_attempts=2)
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is True
        assert calls["count"] == 2

    def test_incomplete_dict_exhausts_retries_and_final_error_is_invalid_not_generic(self):
        calls = {"count": 0}

        def always_incomplete(state):
            calls["count"] += 1
            return {"security_result": {"claim_id": state["case_id"], "decision": "ALLOW"}}

        orchestrator = _permissive_orchestrator(
            {AgentName.SECURITY_GATE: always_incomplete}, retry_policy=RetryPolicy(max_attempts=3)
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_RESULT_INVALID"
        assert calls["count"] == 3
        assert outcome.result_payload is None

    def test_free_text_output_exhausts_retries_with_unstructured_code(self):
        calls = {"count": 0}

        def always_free_text(state):
            calls["count"] += 1
            return {"security_result": ["ALLOW"]}  # jamais un dict ni une instance validée

        orchestrator = _permissive_orchestrator(
            {AgentName.SECURITY_GATE: always_free_text}, retry_policy=RetryPolicy(max_attempts=2)
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_RESULT_UNSTRUCTURED"
        assert calls["count"] == 2

    def test_missing_result_field_is_never_retried_by_default(self):
        """AGENT_RESULT_MISSING n'appartient pas à ``retryable_error_codes``
        par défaut : une seule tentative, jamais un résultat fabriqué pour
        combler l'absence."""
        calls = {"count": 0}

        def returns_nothing(state):
            calls["count"] += 1
            return {}

        orchestrator = _permissive_orchestrator(
            {AgentName.SECURITY_GATE: returns_nothing}, retry_policy=RetryPolicy(max_attempts=5)
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_RESULT_MISSING"
        assert calls["count"] == 1
        assert outcome.result_payload is None


# ── 2. Panne modèle ────────────────────────────────────────────────────────────


class TestModelFailureDuringExecution:
    """Panne survenant pendant l'appel du modèle par l'agent lui-même —
    l'orchestrateur ne connaît le LLM qu'à travers l'exception levée par
    ``agent_registry[agent_name](state)``."""

    def test_transient_model_failure_is_retried_then_succeeds(self):
        calls = {"count": 0}

        def flaky(state):
            calls["count"] += 1
            if calls["count"] == 1:
                raise httpx.ConnectError("Ollama injoignable")
            return _valid_security_result(state)

        orchestrator = _permissive_orchestrator(
            {AgentName.SECURITY_GATE: flaky}, retry_policy=RetryPolicy(max_attempts=2)
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is True
        assert calls["count"] == 2

    def test_transient_model_failure_exceeds_retry_limit_and_stays_visible(self):
        calls = {"count": 0}

        def always_down(state):
            calls["count"] += 1
            raise ConnectionError("connexion au modèle refusée")

        orchestrator = _permissive_orchestrator(
            {AgentName.SECURITY_GATE: always_down}, retry_policy=RetryPolicy(max_attempts=3)
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_EXECUTION_FAILED"
        assert "ConnectionError" in outcome.error.message
        assert "connexion au modèle refusée" in outcome.error.message
        assert calls["count"] == 3

    def test_non_transient_model_error_is_not_retried(self):
        """Une exception non catégorisée (bug de parsing de prompt, valeur
        invalide) n'est jamais assimilée à une panne transitoire du modèle —
        aucun retry, échec immédiat."""
        calls = {"count": 0}

        def buggy(state):
            calls["count"] += 1
            raise ValueError("réponse LLM non parsable")

        orchestrator = _permissive_orchestrator(
            {AgentName.SECURITY_GATE: buggy}, retry_policy=RetryPolicy(max_attempts=5)
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert calls["count"] == 1
        assert "ValueError" in outcome.error.message

    def test_model_fallback_allows_execution_to_proceed(self):
        """Le modèle demandé est désactivé, mais un autre modèle compatible
        est enregistré : le fallback est accepté et l'agent s'exécute bel et
        bien — la panne initiale n'empêche pas l'exécution quand un
        remplacement est permis."""
        registry = ModelRegistry()
        registry.register(
            ModelSpec(
                model_id="primary",
                provider="ollama",
                capabilities=frozenset({ModelCapability.STRUCTURED_OUTPUT}),
                client_factory=lambda: object(),
                enabled=False,
            )
        )
        registry.register(
            ModelSpec(
                model_id="secondary",
                provider="ollama",
                capabilities=frozenset({ModelCapability.STRUCTURED_OUTPUT}),
                client_factory=lambda: object(),
            )
        )
        calls = {"count": 0}

        def runner(state):
            calls["count"] += 1
            return _valid_security_result(state)

        orchestrator = Orchestrator(model_registry=registry, agent_registry={AgentName.SECURITY_GATE: runner})
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake", "intake_result": object()}

        outcome = orchestrator.execute_agent(request, state, model_id="primary")

        assert outcome.success is True
        assert calls["count"] == 1
        assert outcome.metadata["model_id"] == "secondary"
        assert outcome.metadata["model_fallback_from"] == "primary"

    def test_model_failure_without_fallback_surfaces_the_original_cause(self):
        """Aucun modèle de repli disponible : la panne d'origine (modèle
        désactivé) reste visible telle quelle, jamais masquée par une
        erreur générique — et l'agent n'est jamais appelé."""
        registry = ModelRegistry()
        registry.register(
            ModelSpec(
                model_id="only-one",
                provider="ollama",
                capabilities=frozenset({ModelCapability.STRUCTURED_OUTPUT}),
                client_factory=lambda: object(),
                enabled=False,
            )
        )
        calls = {"count": 0}
        orchestrator = Orchestrator(
            model_registry=registry,
            agent_registry={AgentName.SECURITY_GATE: lambda state: calls.__setitem__("count", calls["count"] + 1) or {}},
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake", "intake_result": object()}

        outcome = orchestrator.execute_agent(request, state, model_id="only-one")

        assert outcome.success is False
        assert outcome.error.code == "MODEL_DISABLED"
        assert calls["count"] == 0


# ── 3. Exception d'un outil autorisé ──────────────────────────────────────────


class TestAuthorizedToolException:
    """``rechercher_code`` (medical_coding) est un outil réellement autorisé
    (confirmé ci-dessous) — sa panne, provoquée en amont dans
    ``tools/medical_coding.py``, doit se comporter exactement comme une
    panne d'agent ordinaire : jamais absorbée silencieusement."""

    def _agent_calling_the_authorized_tool(self, state: dict) -> dict:
        tool = get_authorized_tool(AgentName.MEDICAL_CODING, "rechercher_code")
        tool.invoke({"description": "consultation générale", "section": "procedures"})
        return _valid_coding_result(state)  # jamais atteint si l'outil lève

    def test_tool_is_genuinely_authorized_before_being_exercised(self):
        decision = evaluate_tool_authorization(AgentName.MEDICAL_CODING, "rechercher_code")
        assert decision.effect is PolicyEffect.ALLOW

    def test_transient_tool_exception_is_retried_then_succeeds(self):
        calls = {"count": 0}

        def flaky_agent(state):
            calls["count"] += 1
            with patch(
                "agents.medical_coding_agent.tools.lookup_code",
                side_effect=ConnectionError("table de codes indisponible"),
            ):
                if calls["count"] == 1:
                    self._agent_calling_the_authorized_tool(state)
            return _valid_coding_result(state)

        orchestrator = _permissive_orchestrator_with_real_tools(
            {AgentName.MEDICAL_CODING: flaky_agent}, retry_policy=RetryPolicy(max_attempts=2)
        )
        request = _request(AgentName.MEDICAL_CODING, "identity_coverage", "MedicalCodingResult")
        state = {"case_id": "CLM-0001", "current_step": "identity_coverage"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is True
        assert calls["count"] == 2

    def test_authorized_tool_exception_propagates_as_execution_failure(self):
        def agent_with_failing_tool(state):
            with patch(
                "agents.medical_coding_agent.tools.lookup_code",
                side_effect=RuntimeError("panne définitive de la table de codes"),
            ):
                self._agent_calling_the_authorized_tool(state)
            return _valid_coding_result(state)  # jamais atteint

        orchestrator = _permissive_orchestrator_with_real_tools(
            {AgentName.MEDICAL_CODING: agent_with_failing_tool}, retry_policy=RetryPolicy(max_attempts=1)
        )
        request = _request(AgentName.MEDICAL_CODING, "identity_coverage", "MedicalCodingResult")
        state = {"case_id": "CLM-0001", "current_step": "identity_coverage"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_EXECUTION_FAILED"
        assert "RuntimeError" in outcome.error.message
        assert "panne définitive de la table de codes" in outcome.error.message
        assert outcome.agent_name is AgentName.MEDICAL_CODING

    def test_non_transient_tool_exception_is_not_retried(self):
        calls = {"count": 0}

        def agent_with_buggy_tool(state):
            calls["count"] += 1
            with patch(
                "agents.medical_coding_agent.tools.lookup_code",
                side_effect=KeyError("section inconnue"),
            ):
                self._agent_calling_the_authorized_tool(state)
            return _valid_coding_result(state)

        orchestrator = _permissive_orchestrator_with_real_tools(
            {AgentName.MEDICAL_CODING: agent_with_buggy_tool}, retry_policy=RetryPolicy(max_attempts=4)
        )
        request = _request(AgentName.MEDICAL_CODING, "identity_coverage", "MedicalCodingResult")
        state = {"case_id": "CLM-0001", "current_step": "identity_coverage"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert calls["count"] == 1


# ── 4. Limites de retry et interaction avec le fallback modèle ──────────────


class TestRetryLimitsAndFallbackIntegration:
    def test_retry_count_never_exceeds_max_attempts(self):
        calls = {"count": 0}

        def always_flaky(state):
            calls["count"] += 1
            raise httpx.TimeoutException("délai dépassé")

        orchestrator = _permissive_orchestrator(
            {AgentName.SECURITY_GATE: always_flaky}, retry_policy=RetryPolicy(max_attempts=4)
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert calls["count"] == 4
        assert outcome.attempt == 4
        assert outcome.success is False

    def test_default_retry_policy_never_retries(self):
        """``RetryPolicy()`` par défaut (``max_attempts=1``) : une seule
        tentative, même pour une panne transitoire — comportement inchangé
        tant qu'aucune politique n'est explicitement injectée."""
        calls = {"count": 0}

        def always_flaky(state):
            calls["count"] += 1
            raise ConnectionError("en panne")

        orchestrator = _permissive_orchestrator({AgentName.SECURITY_GATE: always_flaky}, retry_policy=RetryPolicy())
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert calls["count"] == 1
        assert outcome.success is False

    def test_model_fallback_is_resolved_once_not_once_per_retry_attempt(self):
        """Le fallback modèle intervient une seule fois, avant la boucle de
        retry de l'appel agent — il ne doit jamais être réévalué à chaque
        tentative d'appel de l'agent."""
        registry = ModelRegistry()
        model_check_calls: list[str] = []
        real_registry = build_default_registry()
        fallback_model_id = real_registry.list_models()[0].model_id
        registry.register(
            ModelSpec(
                model_id="primary",
                provider="ollama",
                capabilities=frozenset({ModelCapability.STRUCTURED_OUTPUT}),
                client_factory=lambda: object(),
                enabled=False,
            )
        )
        registry.register(
            ModelSpec(
                model_id=fallback_model_id,
                provider="ollama",
                capabilities=frozenset({ModelCapability.STRUCTURED_OUTPUT}),
                client_factory=lambda: object(),
            )
        )

        def counting_model_check(registry, agent_name, model_id):
            model_check_calls.append(model_id)
            from orchestrator.policies import evaluate_model_authorization

            return evaluate_model_authorization(registry, agent_name, model_id)

        agent_calls = {"count": 0}

        def flaky_agent(state):
            agent_calls["count"] += 1
            if agent_calls["count"] < 3:
                raise httpx.ConnectError("indisponible")
            return _valid_security_result(state)

        orchestrator = Orchestrator(
            model_registry=registry,
            agent_registry={AgentName.SECURITY_GATE: flaky_agent},
            model_check=counting_model_check,
            retry_policy=RetryPolicy(max_attempts=3),
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake", "intake_result": object()}

        outcome = orchestrator.execute_agent(request, state, model_id="primary")

        assert outcome.success is True
        assert agent_calls["count"] == 3, "l'agent doit être rejoué à chaque tentative"
        # model_check est appelé une fois pour "primary" (refusé) puis une
        # fois pour le candidat de repli — jamais une fois par tentative
        # d'appel de l'agent (3 tentatives, seulement 2 appels de model_check).
        assert model_check_calls == ["primary", fallback_model_id]


# ── 5. Visibilité et attribution du dernier échec ─────────────────────────────


class TestFinalFailureVisibilityAndAttribution:
    def test_final_error_reflects_the_last_attempt_not_the_first(self):
        messages = ["panne n°1", "panne n°2", "panne n°3 — dernière"]
        calls = {"count": 0}

        def flaky(state):
            message = messages[calls["count"]]
            calls["count"] += 1
            raise ConnectionError(message)

        orchestrator = _permissive_orchestrator(
            {AgentName.SECURITY_GATE: flaky}, retry_policy=RetryPolicy(max_attempts=3)
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert "panne n°3 — dernière" in outcome.error.message
        assert "panne n°1" not in outcome.error.message
        assert calls["count"] == 3

    def test_outcome_attempt_matches_the_final_attempt_number(self):
        calls = {"count": 0}

        def always_flaky(state):
            calls["count"] += 1
            raise ConnectionError("x")

        orchestrator = _permissive_orchestrator(
            {AgentName.SECURITY_GATE: always_flaky}, retry_policy=RetryPolicy(max_attempts=3)
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.attempt == 3
        assert outcome.case_id == "CLM-0001"
        assert outcome.agent_name is AgentName.SECURITY_GATE

    def test_audit_result_event_reflects_the_final_failure_code(self):
        def always_flaky(state):
            raise ConnectionError("toujours en panne")

        orchestrator = _permissive_orchestrator(
            {AgentName.SECURITY_GATE: always_flaky}, retry_policy=RetryPolicy(max_attempts=2)
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        result_events = [e for e in outcome.audit_events if e.action == "result"]
        assert len(result_events) == 1, "un seul événement 'result' final, pas un par tentative"
        assert result_events[0].details["final_status"] == "AGENT_EXECUTION_FAILED"
        assert result_events[0].details["attempt"] == "2"

    def test_retry_events_recorded_for_each_replayed_attempt(self):
        calls = {"count": 0}

        def flaky(state):
            calls["count"] += 1
            if calls["count"] < 3:
                raise httpx.ConnectError("x")
            return _valid_security_result(state)

        orchestrator = _permissive_orchestrator(
            {AgentName.SECURITY_GATE: flaky}, retry_policy=RetryPolicy(max_attempts=3)
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        retry_events = [e for e in outcome.audit_events if e.action == "retry"]
        call_events = [e for e in outcome.audit_events if e.action == "call"]
        assert len(retry_events) == 2
        assert len(call_events) == 3
        assert outcome.success is True


# ── 6. Aucune exception avalée, aucun faux résultat métier ───────────────────


class TestNoSilentSwallowing:
    def test_exception_type_and_message_are_preserved_verbatim(self):
        def buggy(state):
            raise RuntimeError("détail précis de la panne")

        orchestrator = _permissive_orchestrator(
            {AgentName.SECURITY_GATE: buggy}, retry_policy=RetryPolicy(max_attempts=1)
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert "RuntimeError" in outcome.error.message
        assert "détail précis de la panne" in outcome.error.message

    @pytest.mark.parametrize(
        "agent_runner,expected_code",
        [
            (lambda state: {}, "AGENT_RESULT_MISSING"),
            (lambda state: {"security_result": "texte libre"}, "AGENT_RESULT_UNSTRUCTURED"),
            (lambda state: {"security_result": {"claim_id": state["case_id"]}}, "AGENT_RESULT_INVALID"),
        ],
    )
    def test_failure_never_carries_a_result_payload(self, agent_runner, expected_code):
        orchestrator = _permissive_orchestrator(
            {AgentName.SECURITY_GATE: agent_runner}, retry_policy=RetryPolicy(max_attempts=1)
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert outcome.error.code == expected_code
        assert outcome.result_payload is None, "jamais de résultat métier, même partiel, en cas d'échec"

    def test_agent_exception_never_raises_out_of_execute_agent(self):
        """Quelle que soit la panne (bug, réseau, outil), ``execute_agent``
        ne lève jamais — toujours un ``AgentCallOutcome`` structuré."""

        def crashes(state):
            raise Exception("panne quelconque")

        orchestrator = _permissive_orchestrator(
            {AgentName.SECURITY_GATE: crashes}, retry_policy=RetryPolicy(max_attempts=1)
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert outcome.error is not None

    def test_partial_business_result_is_never_accepted_as_success(self):
        """Un dict partiellement correct (champ obligatoire manquant) n'est
        jamais accepté tel quel comme un succès — même s'il « ressemble » à
        un résultat valide."""

        def partial(state):
            return {
                "security_result": {
                    "claim_id": state["case_id"],
                    # "decision" manquant : jamais un ALLOW implicite.
                    "reasons": ["ambigu"],
                }
            }

        orchestrator = _permissive_orchestrator(
            {AgentName.SECURITY_GATE: partial}, retry_policy=RetryPolicy(max_attempts=1)
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_RESULT_INVALID"
