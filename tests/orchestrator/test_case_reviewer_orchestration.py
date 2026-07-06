"""Orchestration bout-en-bout — case_reviewer_agent.

Complète les tests génériques déjà existants (``test_policies.py::
TestRefusalScenariosEndToEnd``, ``test_execution.py``/``test_executor.py``,
``test_clinical_fraud_orchestration.py`` pour clinical_consistency/
fraud_detection) par des preuves concrètes, nommément pour
``case_reviewer_agent`` :

  1. sortie invalide     → jamais acceptée comme résultat
     (``AGENT_RESULT_INVALID``/``AGENT_RESULT_UNSTRUCTURED``/
     ``AGENT_RESULT_MISSING``) ;
  2. panne modèle        → jamais avalée silencieusement
     (``AGENT_EXECUTION_FAILED``, retry selon ``RetryPolicy`` puis échec
     visible, ou modèle incompatible refusé avant tout appel) ;
  3. recommandation auto-approuvée → jamais acceptée : toute tentative de
     sortie avec ``status`` ≠ ``NEEDS_REVIEW`` ou ``human_review_required``
     = ``False`` échoue à la validation Pydantic elle-même (verrouillage de
     schéma), donc à ``validate_agent_result``/``execute_agent`` en amont de
     toute logique métier de l'orchestrateur ;
  4. traçabilité d'audit → ``llm_call_id``/``prompt_version``/preuves
     (``evidence_ids``)/statut systématiquement présents et attribués.

Chaque appel passe exclusivement par ``Orchestrator.execute_agent()`` — ni
``agents.case_reviewer_agent.agent.run()`` ni ``.node()`` n'est jamais
appelé directement ici. La garantie que ``graph/nodes.py`` (production)
n'appelle lui non plus jamais ``case_reviewer_agent`` directement est
vérifiée statiquement par ``tests/graph/test_architecture.py`` (``_case_reviewer``
fait partie de ``_AGENT_MODULE_ALIASES``, seul ``build_orchestrator()`` peut
y appeler ``.node``/``.run``).
"""
from __future__ import annotations

import httpx
import pytest
from pydantic import ValidationError

from orchestrator.executor import Orchestrator, RetryPolicy
from orchestrator.model_registry import (
    ModelCapability,
    ModelRegistry,
    ModelSpec,
)
from orchestrator.orchestrator import AgentCallRequest, AgentName, validate_agent_result
from orchestrator.policies import build_authorized_tools
from schemas.results import CaseReviewerResult, CaseReviewerResultPayload, LlmMetadata

_AGENT = AgentName.CASE_REVIEWER
_STEP = "fraud_detection"
_MODEL_NAME = "CaseReviewerResult"


def _request() -> AgentCallRequest:
    return AgentCallRequest(
        agent_name=_AGENT,
        case_id="CLM-0001",
        current_step=_STEP,
        requested_model=_MODEL_NAME,
        attempt=1,
    )


def _valid_result() -> CaseReviewerResult:
    return CaseReviewerResult(
        case_id="CLM-0001",
        llm_trace=LlmMetadata(model_name="gemma4:latest", prompt_version="1.1.0"),
        result_payload=CaseReviewerResultPayload(
            recommendation="PENDING",
            human_review_reasons=["Validation humaine obligatoire avant toute décision finale."],
        ),
    )


# ── 1. Sortie invalide — jamais acceptée ─────────────────────────────────────


class TestInvalidOutputNeverAccepted:
    def _permissive_orchestrator(self, agent_registry: dict, *, retry_policy: RetryPolicy | None = None) -> Orchestrator:
        from orchestrator.policies import PolicyDecision, PolicyEffect
        from schemas.results import StructuredError

        def _allow(*_args, **_kwargs) -> PolicyDecision:
            return PolicyDecision(effect=PolicyEffect.ALLOW, reason=StructuredError(code="OK", message="ok"))

        return Orchestrator(
            model_registry=ModelRegistry(),
            agent_registry=agent_registry,
            preconditions_check=lambda state, request: _allow(),
            model_check=lambda registry, agent_name, model_id: _allow(),
            tools_check=lambda agent_name: build_authorized_tools(agent_name),
            retry_policy=retry_policy or RetryPolicy(),
        )

    def test_incomplete_dict_is_rejected(self):
        """Un dict incomplet (llm_trace/result_payload manquants) n'est
        jamais accepté tel quel comme résultat final."""
        orchestrator = self._permissive_orchestrator(
            {_AGENT: lambda state: {"review_result": {"case_id": state["case_id"]}}}
        )
        state = {"case_id": "CLM-0001", "current_step": _STEP}

        outcome = orchestrator.execute_agent(_request(), state, model_id="whatever")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_RESULT_INVALID"
        assert outcome.result_payload is None

    def test_free_text_output_is_rejected(self):
        """Un texte libre à la place d'un dict/instance Pydantic n'est
        jamais accepté — jamais tenté en model_validate()."""
        orchestrator = self._permissive_orchestrator(
            {_AGENT: lambda state: {"review_result": "le dossier semble approuvable"}}
        )
        state = {"case_id": "CLM-0001", "current_step": _STEP}

        outcome = orchestrator.execute_agent(_request(), state, model_id="whatever")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_RESULT_UNSTRUCTURED"
        assert outcome.result_payload is None

    def test_missing_result_field_is_rejected(self):
        """L'agent retourne un dict qui ne contient même pas le champ
        résultat attendu — jamais un succès fabriqué à partir de rien."""
        orchestrator = self._permissive_orchestrator({_AGENT: lambda state: {}})
        state = {"case_id": "CLM-0001", "current_step": _STEP}

        outcome = orchestrator.execute_agent(_request(), state, model_id="whatever")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_RESULT_MISSING"

    def test_valid_result_is_accepted(self):
        """Contre-preuve : un résultat réellement conforme est bien accepté,
        la politique ci-dessus ne bloque pas un cas valide."""
        orchestrator = self._permissive_orchestrator(
            {_AGENT: lambda state: {"review_result": _valid_result()}}
        )
        state = {"case_id": "CLM-0001", "current_step": _STEP}

        outcome = orchestrator.execute_agent(_request(), state, model_id="whatever")

        assert outcome.success is True
        assert outcome.result_payload["result_payload"]["recommendation"] == "PENDING"


# ── 2. Recommandation auto-approuvée — jamais acceptée ───────────────────────


class TestAutoApprovedRecommendationNeverAccepted:
    """Le verrouillage de schéma (``CaseReviewerResult.status``/
    ``human_review_required``) s'applique bien avant toute logique de
    l'orchestrateur — une tentative de contournement échoue à la
    construction Pydantic elle-même, jamais acceptée par erreur."""

    def test_schema_rejects_human_review_required_false_directly(self):
        with pytest.raises(ValidationError):
            CaseReviewerResult(
                case_id="CLM-0001",
                llm_trace=LlmMetadata(model_name="test", prompt_version="test"),
                human_review_required=False,
                result_payload=CaseReviewerResultPayload(
                    recommendation="APPROVE",
                    human_review_reasons=["Motif."],
                ),
            )

    def test_schema_rejects_final_status_directly(self):
        with pytest.raises(ValidationError):
            CaseReviewerResult(
                case_id="CLM-0001",
                status="PASS",
                llm_trace=LlmMetadata(model_name="test", prompt_version="test"),
                result_payload=CaseReviewerResultPayload(
                    recommendation="APPROVE",
                    human_review_reasons=["Motif."],
                ),
            )

    def test_validate_agent_result_rejects_auto_approved_dict(self):
        """Même en repassant par validate_agent_result (revalidation
        systématique de toute sortie brute d'agent), une tentative de
        décision finale automatique échoue avec AGENT_RESULT_INVALID."""
        from orchestrator.orchestrator import AgentResultValidationError

        raw = {
            "case_id": "CLM-0001",
            "llm_trace": {"model_name": "test", "prompt_version": "test"},
            "human_review_required": False,
            "result_payload": {
                "recommendation": "APPROVE",
                "human_review_reasons": ["Motif."],
            },
        }
        with pytest.raises(AgentResultValidationError) as exc_info:
            validate_agent_result(_AGENT, raw)
        assert exc_info.value.structured.code == "AGENT_RESULT_INVALID"

    def test_execute_agent_never_accepts_auto_approval_attempt(self):
        """Preuve bout-en-bout via un vrai Orchestrator : un agent qui tente
        de renvoyer une décision finale automatique (dict brut contournant
        le constructeur Pydantic) est refusé par execute_agent, jamais un
        succès fabriqué."""
        from orchestrator.policies import PolicyDecision, PolicyEffect
        from schemas.results import StructuredError

        def _allow(*_args, **_kwargs) -> PolicyDecision:
            return PolicyDecision(effect=PolicyEffect.ALLOW, reason=StructuredError(code="OK", message="ok"))

        malicious_payload = {
            "case_id": "CLM-0001",
            "llm_trace": {"model_name": "test", "prompt_version": "test"},
            "human_review_required": False,
            "result_payload": {
                "recommendation": "APPROVE",
                "human_review_reasons": ["Motif."],
            },
        }
        orchestrator = Orchestrator(
            model_registry=ModelRegistry(),
            agent_registry={_AGENT: lambda state: {"review_result": malicious_payload}},
            preconditions_check=lambda state, request: _allow(),
            model_check=lambda registry, agent_name, model_id: _allow(),
            tools_check=lambda agent_name: build_authorized_tools(agent_name),
            retry_policy=RetryPolicy(),
        )
        state = {"case_id": "CLM-0001", "current_step": _STEP}

        outcome = orchestrator.execute_agent(_request(), state, model_id="whatever")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_RESULT_INVALID"
        assert outcome.result_payload is None


# ── 3. Panne modèle — jamais avalée silencieusement ──────────────────────────


class TestModelFailureNeverSwallowed:
    def _permissive_orchestrator(self, agent_registry: dict, *, retry_policy: RetryPolicy) -> Orchestrator:
        from orchestrator.policies import PolicyDecision, PolicyEffect
        from schemas.results import StructuredError

        def _allow(*_args, **_kwargs) -> PolicyDecision:
            return PolicyDecision(effect=PolicyEffect.ALLOW, reason=StructuredError(code="OK", message="ok"))

        return Orchestrator(
            model_registry=ModelRegistry(),
            agent_registry=agent_registry,
            preconditions_check=lambda state, request: _allow(),
            model_check=lambda registry, agent_name, model_id: _allow(),
            tools_check=lambda agent_name: build_authorized_tools(agent_name),
            retry_policy=retry_policy,
        )

    def test_transient_failure_is_retried_then_surfaced(self):
        calls = {"count": 0}

        def _agent(state: dict) -> dict:
            calls["count"] += 1
            raise httpx.ConnectError("connexion refusée : Ollama indisponible")

        orchestrator = self._permissive_orchestrator(
            {_AGENT: _agent}, retry_policy=RetryPolicy(max_attempts=3)
        )
        state = {"case_id": "CLM-0001", "current_step": _STEP}

        outcome = orchestrator.execute_agent(_request(), state, model_id="whatever")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_EXECUTION_FAILED"
        assert "ConnectError" in outcome.error.message
        assert calls["count"] == 3

    def test_non_transient_failure_is_never_retried(self):
        calls = {"count": 0}

        def _agent(state: dict) -> dict:
            calls["count"] += 1
            raise RuntimeError("erreur de programmation inattendue")

        orchestrator = self._permissive_orchestrator(
            {_AGENT: _agent}, retry_policy=RetryPolicy(max_attempts=5)
        )
        state = {"case_id": "CLM-0001", "current_step": _STEP}

        outcome = orchestrator.execute_agent(_request(), state, model_id="whatever")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_EXECUTION_FAILED"
        assert calls["count"] == 1

    def test_incompatible_model_blocks_before_any_call(self):
        """case_reviewer exige STRUCTURED_OUTPUT : un modèle qui ne déclare
        que TOOL_CALLING est incompatible — ni son client ni l'agent ne
        sont jamais invoqués."""
        from orchestrator.policies import PolicyDecision, PolicyEffect
        from schemas.results import StructuredError

        def _allow(*_args, **_kwargs) -> PolicyDecision:
            return PolicyDecision(effect=PolicyEffect.ALLOW, reason=StructuredError(code="OK", message="ok"))

        client_factory_calls: list[str] = []

        def _client_factory():
            client_factory_calls.append("called")
            return object()

        registry = ModelRegistry()
        registry.register(
            ModelSpec(
                model_id="tool-only",
                provider="ollama",
                capabilities=frozenset({ModelCapability.TOOL_CALLING}),
                client_factory=_client_factory,
            )
        )
        agent_calls: list[str] = []
        orchestrator = Orchestrator(
            model_registry=registry,
            agent_registry={_AGENT: lambda state: agent_calls.append("called") or {}},
            preconditions_check=lambda state, request: _allow(),
        )
        state = {"case_id": "CLM-0001", "current_step": _STEP}

        outcome = orchestrator.execute_agent(_request(), state, model_id="tool-only")

        assert outcome.success is False
        assert outcome.error.code == "MODEL_INCOMPATIBLE"
        assert client_factory_calls == []
        assert agent_calls == []


# ── 4. Traçabilité d'audit — llm_call_id, prompt_version, preuves, statut ────


class TestAuditTrailFields:
    def test_success_produces_call_and_result_audit_events(self):
        from orchestrator.policies import PolicyDecision, PolicyEffect
        from schemas.results import StructuredError

        def _allow(*_args, **_kwargs) -> PolicyDecision:
            return PolicyDecision(effect=PolicyEffect.ALLOW, reason=StructuredError(code="OK", message="ok"))

        orchestrator = Orchestrator(
            model_registry=ModelRegistry(),
            agent_registry={_AGENT: lambda state: {"review_result": _valid_result()}},
            preconditions_check=lambda state, request: _allow(),
            model_check=lambda registry, agent_name, model_id: _allow(),
            tools_check=lambda agent_name: build_authorized_tools(agent_name),
            retry_policy=RetryPolicy(),
        )
        state = {"case_id": "CLM-0001", "current_step": _STEP}

        outcome = orchestrator.execute_agent(_request(), state, model_id="whatever")

        assert outcome.success is True
        natures = [event.action for event in outcome.audit_events]
        assert "call" in natures
        assert "result" in natures
        result_event = next(e for e in outcome.audit_events if e.action == "result")
        assert result_event.details["final_status"]

    def test_node_level_audit_carries_llm_call_id_prompt_version_evidence_and_status(self):
        """Au niveau du nœud (agents/case_reviewer_agent/agent.py), l'audit
        porte bien llm_call_id, prompt_version, preuves (evidence_ids) et
        statut — vérifié aussi à ce niveau, pas seulement via l'orchestrateur
        générique (qui n'a pas de connaissance du contenu métier)."""
        from agents.case_reviewer_agent import agent as case_reviewer_agent_module

        updates = case_reviewer_agent_module.make_node()({"case_id": "CLM-0001"})
        details = updates["audit_trail"][0].details
        assert details["llm_call_id"]
        assert details["prompt_version"]
        assert "evidence_ids" in details
        assert details["status"] == "NEEDS_REVIEW"
