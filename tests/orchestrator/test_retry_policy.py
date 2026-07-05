"""Tests de la politique de retry — orchestrator/executor.py::RetryPolicy.

Couvre la politique elle-même (RetryPolicy.is_retryable) et son intégration
dans Orchestrator.execute_agent : succès après retry, panne permanente
(jamais rejouée) et dépassement de la limite de tentatives — ainsi que la
garantie qu'un refus de permission ou une précondition non satisfaite n'est
jamais soumis au retry, et que le contexte (case_id, state) est préservé
d'une tentative à l'autre.
"""
from __future__ import annotations

import httpx
import pytest

from orchestrator.executor import DEFAULT_RETRYABLE_ERROR_CODES, Orchestrator, RetryPolicy
from orchestrator.model_registry import ModelRegistry, build_default_registry
from orchestrator.orchestrator import AgentCallRequest, AgentName
from orchestrator.policies import PolicyDecision, PolicyEffect
from schemas.results import SecurityGateResult, StructuredError


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


def _deny(code: str) -> PolicyDecision:
    return PolicyDecision(effect=PolicyEffect.DENY, reason=StructuredError(code=code, message="refusé"))


def _valid_security_result(state: dict) -> dict:
    return {
        "security_result": SecurityGateResult(
            claim_id=state["case_id"], decision="ALLOW", reasons=["nominal"]
        )
    }


def _fully_permissive_orchestrator(agent_registry, *, max_attempts: int, retryable_error_codes=None) -> Orchestrator:
    kwargs = {}
    if retryable_error_codes is not None:
        kwargs["retryable_error_codes"] = retryable_error_codes
    return Orchestrator(
        model_registry=ModelRegistry(),
        agent_registry=agent_registry,
        preconditions_check=lambda state, request: _allow(),
        model_check=lambda registry, agent_name, model_id: _allow(),
        tools_check=lambda agent_name: (),
        retry_policy=RetryPolicy(max_attempts=max_attempts, **kwargs),
    )


# ── 1. RetryPolicy — configuration et logique d'éligibilité ──────────────────


class TestRetryPolicyConfiguration:
    def test_default_max_attempts_is_one(self):
        assert RetryPolicy().max_attempts == 1

    def test_max_attempts_configurable(self):
        assert RetryPolicy(max_attempts=5).max_attempts == 5

    def test_max_attempts_below_one_rejected(self):
        with pytest.raises(ValueError):
            RetryPolicy(max_attempts=0)

    def test_max_attempts_negative_rejected(self):
        with pytest.raises(ValueError):
            RetryPolicy(max_attempts=-1)

    def test_default_retryable_codes_exclude_permission_and_missing(self):
        forbidden = {
            "CASE_ID_MISMATCH",
            "STEP_DECLARATION_MISMATCH",
            "STEP_MISMATCH",
            "PRECONDITION_RESULT_MISSING",
            "MODEL_NOT_FOUND",
            "MODEL_DISABLED",
            "MODEL_INCOMPATIBLE",
            "NO_AUTHORIZED_TOOLS",
            "AGENT_NOT_REGISTERED",
            "TOOL_NOT_AUTHORIZED_FOR_AGENT",
            "AGENT_RESULT_MISSING",
        }
        assert DEFAULT_RETRYABLE_ERROR_CODES.isdisjoint(forbidden)


class TestRetryPolicyIsRetryable:
    def test_transient_exception_is_retryable(self):
        policy = RetryPolicy()
        assert policy.is_retryable(
            error_code="AGENT_EXECUTION_FAILED", exc=httpx.ConnectError("refused")
        )

    def test_connection_error_is_retryable(self):
        policy = RetryPolicy()
        assert policy.is_retryable(error_code="AGENT_EXECUTION_FAILED", exc=ConnectionError("x"))

    def test_non_transient_exception_is_not_retryable(self):
        policy = RetryPolicy()
        assert not policy.is_retryable(error_code="AGENT_EXECUTION_FAILED", exc=RuntimeError("bug"))

    def test_execution_failed_without_exception_context_is_not_retryable(self):
        policy = RetryPolicy()
        assert not policy.is_retryable(error_code="AGENT_EXECUTION_FAILED", exc=None)

    def test_result_invalid_is_retryable_by_default(self):
        policy = RetryPolicy()
        assert policy.is_retryable(error_code="AGENT_RESULT_INVALID", exc=None)

    def test_result_unstructured_is_retryable_by_default(self):
        policy = RetryPolicy()
        assert policy.is_retryable(error_code="AGENT_RESULT_UNSTRUCTURED", exc=None)

    def test_result_missing_is_not_retryable_by_default(self):
        policy = RetryPolicy()
        assert not policy.is_retryable(error_code="AGENT_RESULT_MISSING", exc=None)

    @pytest.mark.parametrize(
        "code",
        [
            "CASE_ID_MISMATCH",
            "STEP_DECLARATION_MISMATCH",
            "STEP_MISMATCH",
            "PRECONDITION_RESULT_MISSING",
            "MODEL_NOT_FOUND",
            "MODEL_DISABLED",
            "MODEL_INCOMPATIBLE",
            "NO_AUTHORIZED_TOOLS",
            "AGENT_NOT_REGISTERED",
            "TOOL_NOT_AUTHORIZED_FOR_AGENT",
        ],
    )
    def test_permission_and_precondition_codes_never_retryable(self, code):
        policy = RetryPolicy()
        assert not policy.is_retryable(error_code=code, exc=None)

    def test_retryable_error_codes_are_configurable(self):
        policy = RetryPolicy(retryable_error_codes=frozenset({"CUSTOM_CODE"}))
        assert policy.is_retryable(error_code="CUSTOM_CODE", exc=None)
        assert not policy.is_retryable(error_code="AGENT_RESULT_INVALID", exc=None)

    def test_transient_exceptions_are_configurable(self):
        policy = RetryPolicy(transient_exceptions=(ValueError,))
        assert policy.is_retryable(error_code="AGENT_EXECUTION_FAILED", exc=ValueError("x"))
        assert not policy.is_retryable(
            error_code="AGENT_EXECUTION_FAILED", exc=ConnectionError("x")
        )


# ── 2. Succès après retry ─────────────────────────────────────────────────────


class TestSuccessAfterRetry:
    def test_transient_failure_then_success(self):
        calls = {"count": 0}

        def flaky(state):
            calls["count"] += 1
            if calls["count"] < 2:
                raise ConnectionError("connexion refusée")
            return _valid_security_result(state)

        orchestrator = _fully_permissive_orchestrator(
            {AgentName.SECURITY_GATE: flaky}, max_attempts=3
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is True
        assert calls["count"] == 2
        assert outcome.attempt == 2

    def test_repairable_output_then_valid_output(self):
        """Sortie explicitement réparable (AGENT_RESULT_INVALID) suivie
        d'une sortie valide — pas une exception, une simple sortie malformée
        rejouée avec succès."""
        calls = {"count": 0}

        def flaky_output(state):
            calls["count"] += 1
            if calls["count"] < 2:
                return {"security_result": {"claim_id": state["case_id"], "decision": "ALLOW"}}
            return _valid_security_result(state)

        orchestrator = _fully_permissive_orchestrator(
            {AgentName.SECURITY_GATE: flaky_output}, max_attempts=3
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is True
        assert calls["count"] == 2

    def test_success_on_first_attempt_never_retries(self):
        calls = {"count": 0}

        def always_ok(state):
            calls["count"] += 1
            return _valid_security_result(state)

        orchestrator = _fully_permissive_orchestrator(
            {AgentName.SECURITY_GATE: always_ok}, max_attempts=5
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is True
        assert calls["count"] == 1
        assert outcome.attempt == 1


# ── 3. Panne permanente — jamais rejouée ──────────────────────────────────────


class TestPermanentFailureNeverRetried:
    def test_non_transient_exception_attempted_once_despite_higher_limit(self):
        calls = {"count": 0}

        def buggy(state):
            calls["count"] += 1
            raise RuntimeError("bug non catégorisé")

        orchestrator = _fully_permissive_orchestrator(
            {AgentName.SECURITY_GATE: buggy}, max_attempts=5
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_EXECUTION_FAILED"
        assert calls["count"] == 1

    def test_missing_result_field_attempted_once_despite_higher_limit(self):
        calls = {"count": 0}

        def empty(state):
            calls["count"] += 1
            return {}

        orchestrator = _fully_permissive_orchestrator(
            {AgentName.SECURITY_GATE: empty}, max_attempts=5
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_RESULT_MISSING"
        assert calls["count"] == 1


class TestPermissionRefusalNeverRetried:
    """Un refus de permission ou une précondition non satisfaite n'est
    jamais soumis à la politique de retry — l'agent n'est même pas appelé
    une seule fois."""

    def test_precondition_denial_never_calls_agent_even_with_high_max_attempts(self):
        calls = {"count": 0}

        def spy(state):
            calls["count"] += 1
            return _valid_security_result(state)

        orchestrator = Orchestrator(
            model_registry=ModelRegistry(),
            agent_registry={AgentName.SECURITY_GATE: spy},
            preconditions_check=lambda state, request: _deny("STEP_MISMATCH"),
            model_check=lambda registry, agent_name, model_id: _allow(),
            tools_check=lambda agent_name: (),
            retry_policy=RetryPolicy(max_attempts=10),
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert outcome.error.code == "STEP_MISMATCH"
        assert calls["count"] == 0

    def test_model_denial_never_calls_agent_even_with_high_max_attempts(self):
        calls = {"count": 0}

        def spy(state):
            calls["count"] += 1
            return _valid_security_result(state)

        orchestrator = Orchestrator(
            model_registry=ModelRegistry(),
            agent_registry={AgentName.SECURITY_GATE: spy},
            preconditions_check=lambda state, request: _allow(),
            model_check=lambda registry, agent_name, model_id: _deny("MODEL_INCOMPATIBLE"),
            tools_check=lambda agent_name: (),
            retry_policy=RetryPolicy(max_attempts=10),
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert outcome.error.code == "MODEL_INCOMPATIBLE"
        assert calls["count"] == 0

    def test_no_authorized_tools_never_calls_agent_even_with_high_max_attempts(self):
        calls = {"count": 0}

        def spy(state):
            calls["count"] += 1
            return {}

        orchestrator = Orchestrator(
            model_registry=ModelRegistry(),
            agent_registry={AgentName.MEDICAL_CODING: spy},
            preconditions_check=lambda state, request: _allow(),
            model_check=lambda registry, agent_name, model_id: _allow(),
            tools_check=lambda agent_name: (),
            retry_policy=RetryPolicy(max_attempts=10),
        )
        request = _request(
            AgentName.MEDICAL_CODING, "identity_coverage", "MedicalCodingResult"
        )
        state = {"case_id": "CLM-0001", "current_step": "identity_coverage"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert outcome.error.code == "NO_AUTHORIZED_TOOLS"
        assert calls["count"] == 0

    def test_real_incompatible_model_via_registry_never_retried(self):
        """Intégration bout en bout avec un vrai ModelRegistry : le refus
        de modèle reste définitif quel que soit max_attempts."""
        from orchestrator.model_registry import ModelCapability, ModelSpec

        registry = ModelRegistry()
        registry.register(
            ModelSpec(
                model_id="structured-only",
                provider="ollama",
                capabilities=frozenset({ModelCapability.STRUCTURED_OUTPUT}),
                client_factory=lambda: object(),
            )
        )
        calls = {"count": 0}
        orchestrator = Orchestrator(
            model_registry=registry,
            agent_registry={
                AgentName.MEDICAL_CODING: lambda state: calls.__setitem__(
                    "count", calls["count"] + 1
                )
                or {}
            },
            retry_policy=RetryPolicy(max_attempts=10),
        )
        request = _request(
            AgentName.MEDICAL_CODING, "identity_coverage", "MedicalCodingResult"
        )
        state = {
            "case_id": "CLM-0001",
            "current_step": "identity_coverage",
            "identity_coverage_result": object(),
        }

        outcome = orchestrator.execute_agent(request, state, model_id="structured-only")

        assert outcome.success is False
        assert outcome.error.code == "MODEL_INCOMPATIBLE"
        assert calls["count"] == 0


# ── 4. Dépassement de la limite de tentatives ─────────────────────────────────


class TestRetryLimitExceeded:
    def test_always_transient_failure_stops_at_max_attempts(self):
        calls = {"count": 0}

        def always_flaky(state):
            calls["count"] += 1
            raise httpx.ConnectError("toujours en panne")

        orchestrator = _fully_permissive_orchestrator(
            {AgentName.SECURITY_GATE: always_flaky}, max_attempts=3
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_EXECUTION_FAILED"
        assert calls["count"] == 3
        assert outcome.attempt == 3

    def test_always_repairable_invalid_output_stops_at_max_attempts(self):
        calls = {"count": 0}

        def always_invalid(state):
            calls["count"] += 1
            return {"security_result": {"claim_id": state["case_id"], "decision": "ALLOW"}}

        orchestrator = _fully_permissive_orchestrator(
            {AgentName.SECURITY_GATE: always_invalid}, max_attempts=4
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_RESULT_INVALID"
        assert calls["count"] == 4
        assert outcome.attempt == 4

    def test_default_policy_never_retries_even_a_transient_failure(self):
        """RetryPolicy() par défaut (max_attempts=1) : comportement
        inchangé — aucune relance, même pour une panne transitoire."""
        calls = {"count": 0}

        def always_flaky(state):
            calls["count"] += 1
            raise ConnectionError("en panne")

        orchestrator = _fully_permissive_orchestrator(
            {AgentName.SECURITY_GATE: always_flaky}, max_attempts=1
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert calls["count"] == 1
        assert outcome.attempt == 1


# ── 5. Même exécution, même case_id, même contexte entre tentatives ──────────


class TestSameContextAcrossAttempts:
    def test_same_state_object_passed_to_every_attempt(self):
        seen_states = []

        def flaky(state):
            seen_states.append(state)
            if len(seen_states) < 3:
                raise ConnectionError("x")
            return _valid_security_result(state)

        orchestrator = _fully_permissive_orchestrator(
            {AgentName.SECURITY_GATE: flaky}, max_attempts=3
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        orchestrator.execute_agent(request, state, model_id="m")

        assert len(seen_states) == 3
        assert all(s is state for s in seen_states)

    def test_case_id_and_agent_identity_unchanged_across_attempts(self):
        seen_case_ids = []
        seen_agents = []

        def flaky(state):
            if len(seen_case_ids) < 2:
                seen_case_ids.append(state["case_id"])
                seen_agents.append("security_gate")
                raise ConnectionError("x")
            seen_case_ids.append(state["case_id"])
            return _valid_security_result(state)

        orchestrator = _fully_permissive_orchestrator(
            {AgentName.SECURITY_GATE: flaky}, max_attempts=3
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert set(seen_case_ids) == {"CLM-0001"}
        assert outcome.case_id == "CLM-0001"
        assert outcome.agent_name is AgentName.SECURITY_GATE

    def test_final_outcome_attempt_reflects_last_try(self):
        calls = {"count": 0}

        def flaky(state):
            calls["count"] += 1
            if calls["count"] < 3:
                raise ConnectionError("x")
            return _valid_security_result(state)

        orchestrator = _fully_permissive_orchestrator(
            {AgentName.SECURITY_GATE: flaky}, max_attempts=5
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult", attempt=1)
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.attempt == 3
        assert outcome.success is True

    def test_original_request_attempt_field_left_untouched(self):
        """La requête fournie par l'appelant n'est jamais mutée : chaque
        tentative construit sa propre copie (attempt différent), l'original
        reste attempt=1."""

        def always_ok(state):
            return _valid_security_result(state)

        orchestrator = _fully_permissive_orchestrator(
            {AgentName.SECURITY_GATE: always_ok}, max_attempts=3
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult", attempt=1)
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        orchestrator.execute_agent(request, state, model_id="m")

        assert request.attempt == 1


# ── 6. Intégration avec le registre de modèles réel ───────────────────────────


class TestRetryPolicyWithRealRegistry:
    def test_success_after_retry_with_real_default_registry(self):
        registry = build_default_registry()
        model_id = registry.list_models()[0].model_id
        calls = {"count": 0}

        def flaky(state):
            calls["count"] += 1
            if calls["count"] < 2:
                raise httpx.TimeoutException("timeout")
            return _valid_security_result(state)

        orchestrator = Orchestrator(
            model_registry=registry,
            agent_registry={AgentName.SECURITY_GATE: flaky},
            retry_policy=RetryPolicy(max_attempts=3),
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {
            "case_id": "CLM-0001",
            "current_step": "claim_intake",
            "intake_result": object(),
        }

        outcome = orchestrator.execute_agent(request, state, model_id=model_id)

        assert outcome.success is True
        assert calls["count"] == 2
