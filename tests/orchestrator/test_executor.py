"""Tests de l'exécution contrôlée d'un agent — orchestrator/executor.py.

Le test exigé — vérifier l'ordre exact des contrôles sur le chemin nominal —
est ``TestExecuteAgentOrder::test_nominal_path_calls_checks_in_exact_order``.
Les classes suivantes vérifient en complément qu'aucun contournement n'est
possible : chaque étape refusée empêche définitivement l'appel de l'agent.
"""
from __future__ import annotations

from orchestrator.executor import Orchestrator, RetryPolicy
from orchestrator.model_registry import ModelCapability, ModelRegistry, ModelSpec, build_default_registry
from orchestrator.orchestrator import AgentCallRequest, AgentName
from orchestrator.policies import PolicyDecision, PolicyEffect
from agents.audit_agent.schemas import LlmAuditNormalizedEvent
from schemas.audit import AuditEventType, RedactionStatus
from schemas.domain import DataClassification
from schemas.results import SecurityGateResult, StructuredError
from services.audit_store import AuditStore


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


def _deny(code: str = "DENIED") -> PolicyDecision:
    return PolicyDecision(effect=PolicyEffect.DENY, reason=StructuredError(code=code, message="refusé"))


def _fake_security_gate_runner(state: dict) -> dict:
    return {
        "security_result": SecurityGateResult(
            claim_id=state["case_id"], decision="ALLOW", reasons=["nominal"]
        )
    }


def _audit_normalizer(calls: list[dict]):
    def _normalize(event: dict) -> LlmAuditNormalizedEvent:
        calls.append(event)
        final_status = event["details"].get("final_status") or event["outcome"]
        outcome = f"{event['action']}:{final_status}"
        return LlmAuditNormalizedEvent(
            event_type=AuditEventType(event["candidate_event_type"]),
            actor=str(event["actor"]),
            outcome=outcome,
            summary=f"Evenement orchestrateur normalise : {event['action']}.",
            redaction_status=RedactionStatus.FULLY_REDACTED,
            classification=DataClassification.CONFIDENTIAL,
            anomalies=[] if final_status in {"", "IN_PROGRESS", "SUCCESS"} else [str(final_status)],
            redactions=["Payload agent non repris."],
            agent_name=str(event["actor"]),
            tool_calls=[
                name for name in str(event["details"].get("tools") or "").split(",") if name
            ],
            evidence_ids=[event["action"]],
            reasons=["Normalisation audit de test."],
        )

    return _normalize


# ── 1. Ordre exact des contrôles — test nominal exigé ─────────────────────────


class TestExecuteAgentOrder:
    def test_nominal_path_calls_checks_in_exact_order(self):
        """Sur le chemin nominal (tout autorisé), les quatre étapes doivent
        s'exécuter dans l'ordre : préconditions -> modèle -> outils -> agent.
        Aucune étape sautée, aucune réordonnée."""
        call_order: list[str] = []

        def fake_preconditions(state, request):
            call_order.append("preconditions")
            return _allow()

        def fake_model_check(registry, agent_name, model_id):
            call_order.append("model")
            return _allow()

        def fake_tools_check(agent_name):
            call_order.append("tools")
            return ()

        def fake_agent_runner(state):
            call_order.append("agent")
            return _fake_security_gate_runner(state)

        orchestrator = Orchestrator(
            model_registry=ModelRegistry(),
            agent_registry={AgentName.SECURITY_GATE: fake_agent_runner},
            preconditions_check=fake_preconditions,
            model_check=fake_model_check,
            tools_check=fake_tools_check,
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="whatever-model")

        assert call_order == ["preconditions", "model", "tools", "agent"]
        assert outcome.success is True
        assert outcome.result_payload["decision"] == "ALLOW"

    def test_nominal_path_with_real_default_checks_and_registry(self):
        """Même propriété, mais avec les contrôles réels par défaut
        (aucun override) et un ModelRegistry réel — vérifie que le câblage
        par défaut de l'orchestrateur fonctionne de bout en bout."""
        registry = build_default_registry()
        model_id = registry.list_models()[0].model_id
        orchestrator = Orchestrator(
            model_registry=registry,
            agent_registry={AgentName.SECURITY_GATE: _fake_security_gate_runner},
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {
            "case_id": "CLM-0001",
            "current_step": "claim_intake",
            "intake_result": object(),
        }

        outcome = orchestrator.execute_agent(request, state, model_id=model_id)

        assert outcome.success is True
        assert outcome.agent_name is AgentName.SECURITY_GATE
        assert outcome.attempt == 1


# ── 2. Refus de contournement — chaque étape bloque définitivement ───────────


class TestExecuteAgentBypassRefused:
    def test_precondition_denial_stops_before_model_or_agent(self):
        calls: list[str] = []

        orchestrator = Orchestrator(
            model_registry=ModelRegistry(),
            agent_registry={
                AgentName.SECURITY_GATE: lambda state: calls.append("agent") or _fake_security_gate_runner(state)
            },
            preconditions_check=lambda state, request: _deny("PRECONDITION_DENIED"),
            model_check=lambda registry, agent_name, model_id: calls.append("model") or _allow(),
            tools_check=lambda agent_name: calls.append("tools") or (),
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert outcome.error.code == "PRECONDITION_DENIED"
        assert calls == []

    def test_model_denial_stops_before_tools_or_agent(self):
        calls: list[str] = []

        orchestrator = Orchestrator(
            model_registry=ModelRegistry(),
            agent_registry={
                AgentName.SECURITY_GATE: lambda state: calls.append("agent") or _fake_security_gate_runner(state)
            },
            preconditions_check=lambda state, request: _allow(),
            model_check=lambda registry, agent_name, model_id: _deny("MODEL_DENIED"),
            tools_check=lambda agent_name: calls.append("tools") or (),
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert outcome.error.code == "MODEL_DENIED"
        assert calls == []

    def test_no_authorized_tools_for_tool_calling_agent_blocks_execution(self):
        """medical_coding requiert TOOL_CALLING : une allowlist vide doit
        bloquer l'appel, même si préconditions et modèle sont autorisés."""
        calls: list[str] = []
        orchestrator = Orchestrator(
            model_registry=ModelRegistry(),
            agent_registry={
                AgentName.MEDICAL_CODING: lambda state: calls.append("agent") or {}
            },
            preconditions_check=lambda state, request: _allow(),
            model_check=lambda registry, agent_name, model_id: _allow(),
            tools_check=lambda agent_name: (),
        )
        request = _request(
            AgentName.MEDICAL_CODING, "identity_coverage", "MedicalCodingResult"
        )
        state = {"case_id": "CLM-0001", "current_step": "identity_coverage"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert outcome.error.code == "NO_AUTHORIZED_TOOLS"
        assert calls == []

    def test_structured_output_agent_not_blocked_by_empty_tools(self):
        """security_gate ne requiert pas TOOL_CALLING : une liste d'outils
        vide ne doit pas bloquer son exécution."""
        orchestrator = Orchestrator(
            model_registry=ModelRegistry(),
            agent_registry={AgentName.SECURITY_GATE: _fake_security_gate_runner},
            preconditions_check=lambda state, request: _allow(),
            model_check=lambda registry, agent_name, model_id: _allow(),
            tools_check=lambda agent_name: (),
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is True

    def test_unregistered_agent_denied_without_exception(self):
        orchestrator = Orchestrator(
            model_registry=ModelRegistry(),
            agent_registry={},
            preconditions_check=lambda state, request: _allow(),
            model_check=lambda registry, agent_name, model_id: _allow(),
            tools_check=lambda agent_name: (),
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_NOT_REGISTERED"

    def test_agent_runner_exception_is_wrapped_not_raised(self):
        def _raises(state):
            raise RuntimeError("panne agent")

        orchestrator = Orchestrator(
            model_registry=ModelRegistry(),
            agent_registry={AgentName.SECURITY_GATE: _raises},
            preconditions_check=lambda state, request: _allow(),
            model_check=lambda registry, agent_name, model_id: _allow(),
            tools_check=lambda agent_name: (),
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_EXECUTION_FAILED"
        assert "panne agent" in outcome.error.message

    def test_agent_runner_missing_result_field_denied(self):
        orchestrator = Orchestrator(
            model_registry=ModelRegistry(),
            agent_registry={AgentName.SECURITY_GATE: lambda state: {}},
            preconditions_check=lambda state, request: _allow(),
            model_check=lambda registry, agent_name, model_id: _allow(),
            tools_check=lambda agent_name: (),
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_RESULT_MISSING"

    def test_agent_runner_returning_free_text_result_denied_not_raised(self):
        """Un agent qui retournerait du texte libre au lieu du modèle
        Pydantic attendu ne fait jamais planter execute_agent : refus
        structuré, jamais un dict brut ou du texte accepté comme résultat."""
        orchestrator = Orchestrator(
            model_registry=ModelRegistry(),
            agent_registry={
                AgentName.SECURITY_GATE: lambda state: {
                    "security_result": "ALLOW, tout va bien, aucun souci détecté."
                }
            },
            preconditions_check=lambda state, request: _allow(),
            model_check=lambda registry, agent_name, model_id: _allow(),
            tools_check=lambda agent_name: (),
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_RESULT_UNSTRUCTURED"
        assert outcome.result_payload is None

    def test_agent_runner_returning_incomplete_dict_denied_not_raised(self):
        """Un dict incomplet (champ requis manquant) n'est jamais accepté
        comme résultat final — refusé, pas de crash."""
        orchestrator = Orchestrator(
            model_registry=ModelRegistry(),
            agent_registry={
                AgentName.SECURITY_GATE: lambda state: {
                    "security_result": {"claim_id": state["case_id"], "decision": "ALLOW"}
                }
            },
            preconditions_check=lambda state, request: _allow(),
            model_check=lambda registry, agent_name, model_id: _allow(),
            tools_check=lambda agent_name: (),
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_RESULT_INVALID"

    def test_incompatible_model_denied_with_real_registry(self):
        """Intégration avec le vrai ModelRegistry : un modèle sans la
        capacité requise est refusé avant tout appel de l'agent."""
        registry = ModelRegistry()
        registry.register(
            ModelSpec(
                model_id="structured-only",
                provider="ollama",
                capabilities=frozenset({ModelCapability.STRUCTURED_OUTPUT}),
                client_factory=lambda: object(),
            )
        )
        calls: list[str] = []
        orchestrator = Orchestrator(
            model_registry=registry,
            agent_registry={
                AgentName.MEDICAL_CODING: lambda state: calls.append("agent") or {}
            },
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
        assert calls == []


# ── 3. Fallback modèle — succès, refus, absence de candidat ──────────────────


class TestModelFallback:
    def test_successful_fallback_switches_to_authorized_replacement(self, caplog):
        """Le modèle demandé (désactivé) est refusé ; le seul autre modèle du
        registre, compatible et activé, est retenu automatiquement — même
        schéma de sortie, mêmes outils, agent bien appelé."""
        registry = ModelRegistry()
        registry.register(
            ModelSpec(
                model_id="primary",
                provider="ollama",
                capabilities=frozenset(
                    {ModelCapability.STRUCTURED_OUTPUT, ModelCapability.TOOL_CALLING}
                ),
                client_factory=lambda: object(),
                enabled=False,
            )
        )
        registry.register(
            ModelSpec(
                model_id="secondary",
                provider="ollama",
                capabilities=frozenset(
                    {ModelCapability.STRUCTURED_OUTPUT, ModelCapability.TOOL_CALLING}
                ),
                client_factory=lambda: object(),
            )
        )
        orchestrator = Orchestrator(
            model_registry=registry,
            agent_registry={AgentName.SECURITY_GATE: _fake_security_gate_runner},
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {
            "case_id": "CLM-0001",
            "current_step": "claim_intake",
            "intake_result": object(),
        }

        with caplog.at_level("WARNING"):
            outcome = orchestrator.execute_agent(request, state, model_id="primary")

        assert outcome.success is True
        assert outcome.result_payload["decision"] == "ALLOW"
        assert outcome.metadata["model_id"] == "secondary"
        assert outcome.metadata["model_fallback_from"] == "primary"
        log_text = " ".join(record.getMessage() for record in caplog.records)
        assert "primary" in log_text
        assert "MODEL_DISABLED" in log_text
        assert "secondary" in log_text

    def test_forbidden_fallback_keeps_original_error_without_masking(self):
        """Un candidat existe bien dans le registre (compatible, activé),
        mais la politique de vérification le refuse explicitement — le
        fallback n'est jamais imposé, l'erreur d'origine reste inchangée."""
        registry = ModelRegistry()
        registry.register(
            ModelSpec(
                model_id="primary",
                provider="ollama",
                capabilities=frozenset({ModelCapability.STRUCTURED_OUTPUT}),
                client_factory=lambda: object(),
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
        calls: list[str] = []

        def _forbid_all(registry, agent_name, model_id):
            calls.append(model_id)
            return _deny("MODEL_DISABLED" if model_id == "primary" else "MODEL_FORBIDDEN_BY_POLICY")

        orchestrator = Orchestrator(
            model_registry=registry,
            agent_registry={
                AgentName.SECURITY_GATE: lambda state: (_ for _ in ()).throw(
                    AssertionError("l'agent ne doit jamais être appelé")
                )
            },
            model_check=_forbid_all,
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {
            "case_id": "CLM-0001",
            "current_step": "claim_intake",
            "intake_result": object(),
        }

        outcome = orchestrator.execute_agent(request, state, model_id="primary")

        assert outcome.success is False
        assert outcome.error.code == "MODEL_DISABLED"
        assert calls == ["primary", "secondary"]

    def test_no_model_available_returns_original_error(self):
        """Aucun autre modèle n'est enregistré : l'erreur d'origine (modèle
        désactivé) est retournée telle quelle, sans jamais appeler l'agent."""
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
        calls: list[str] = []
        orchestrator = Orchestrator(
            model_registry=registry,
            agent_registry={
                AgentName.SECURITY_GATE: lambda state: calls.append("agent") or _fake_security_gate_runner(state)
            },
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {
            "case_id": "CLM-0001",
            "current_step": "claim_intake",
            "intake_result": object(),
        }

        outcome = orchestrator.execute_agent(request, state, model_id="only-one")

        assert outcome.success is False
        assert outcome.error.code == "MODEL_DISABLED"
        assert calls == []


# ── 4. Événements d'audit — présence et minimisation ─────────────────────────


ALL_EVENT_DETAIL_KEYS = {"model_id", "tools", "policy", "attempt", "final_status"}
ALL_EVENT_ACTIONS = {"authorization", "refusal", "call", "retry", "fallback", "result"}


def _two_model_registry(*, primary_enabled: bool = True) -> ModelRegistry:
    registry = ModelRegistry()
    registry.register(
        ModelSpec(
            model_id="primary",
            provider="ollama",
            capabilities=frozenset({ModelCapability.STRUCTURED_OUTPUT, ModelCapability.TOOL_CALLING}),
            client_factory=lambda: object(),
            enabled=primary_enabled,
        )
    )
    registry.register(
        ModelSpec(
            model_id="secondary",
            provider="ollama",
            capabilities=frozenset({ModelCapability.STRUCTURED_OUTPUT, ModelCapability.TOOL_CALLING}),
            client_factory=lambda: object(),
        )
    )
    return registry


class TestAuditEvents:
    def test_nominal_success_emits_authorization_call_and_result_events(self):
        registry = _two_model_registry()
        orchestrator = Orchestrator(
            model_registry=registry,
            agent_registry={AgentName.SECURITY_GATE: _fake_security_gate_runner},
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {
            "case_id": "CLM-0001",
            "current_step": "claim_intake",
            "intake_result": object(),
        }

        outcome = orchestrator.execute_agent(request, state, model_id="primary")

        assert outcome.success is True
        actions = [event.action for event in outcome.audit_events]
        assert actions == ["authorization", "authorization", "authorization", "call", "result"]
        for event in outcome.audit_events:
            assert event.case_id == "CLM-0001"
            assert event.actor == "security_gate"
            assert set(event.details.keys()) == ALL_EVENT_DETAIL_KEYS
        assert outcome.audit_events[-1].outcome == "SUCCESS"
        assert outcome.audit_events[-1].details["final_status"] == "SUCCESS"
        assert outcome.audit_events[-1].details["attempt"] == "1"

    def test_precondition_refusal_emits_single_refusal_event(self):
        orchestrator = Orchestrator(
            model_registry=ModelRegistry(),
            agent_registry={AgentName.SECURITY_GATE: _fake_security_gate_runner},
            preconditions_check=lambda state, request: _deny("PRECONDITION_DENIED"),
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}

        outcome = orchestrator.execute_agent(request, state, model_id="m")

        assert outcome.success is False
        assert len(outcome.audit_events) == 1
        event = outcome.audit_events[0]
        assert event.action == "refusal"
        assert event.outcome == "DENY"
        assert event.details["policy"] == "PRECONDITION_DENIED"
        assert event.details["final_status"] == "PRECONDITION_DENIED"

    def test_model_fallback_success_emits_fallback_event(self):
        registry = _two_model_registry(primary_enabled=False)
        orchestrator = Orchestrator(
            model_registry=registry,
            agent_registry={AgentName.SECURITY_GATE: _fake_security_gate_runner},
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {
            "case_id": "CLM-0001",
            "current_step": "claim_intake",
            "intake_result": object(),
        }

        outcome = orchestrator.execute_agent(request, state, model_id="primary")

        assert outcome.success is True
        actions = [event.action for event in outcome.audit_events]
        assert actions == ["authorization", "fallback", "authorization", "call", "result"]
        fallback_event = outcome.audit_events[1]
        assert fallback_event.outcome == "APPLIED"
        assert fallback_event.details["model_id"] == "secondary"

    def test_model_fallback_forbidden_emits_fallback_rejected_then_refusal(self):
        registry = ModelRegistry()
        registry.register(
            ModelSpec(
                model_id="primary",
                provider="ollama",
                capabilities=frozenset({ModelCapability.STRUCTURED_OUTPUT}),
                client_factory=lambda: object(),
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

        def _forbid_all(registry, agent_name, model_id):
            return _deny("MODEL_DISABLED" if model_id == "primary" else "MODEL_FORBIDDEN_BY_POLICY")

        orchestrator = Orchestrator(
            model_registry=registry,
            agent_registry={AgentName.SECURITY_GATE: _fake_security_gate_runner},
            model_check=_forbid_all,
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {
            "case_id": "CLM-0001",
            "current_step": "claim_intake",
            "intake_result": object(),
        }

        outcome = orchestrator.execute_agent(request, state, model_id="primary")

        assert outcome.success is False
        actions = [event.action for event in outcome.audit_events]
        assert actions == ["authorization", "fallback", "refusal"]
        assert outcome.audit_events[1].outcome == "REJECTED"
        assert outcome.audit_events[2].outcome == "DENY"
        assert outcome.audit_events[2].details["policy"] == "MODEL_DISABLED"

    def test_retry_then_success_emits_call_retry_call_result_events(self):
        registry = _two_model_registry()
        calls = {"count": 0}

        def _flaky_runner(state):
            calls["count"] += 1
            if calls["count"] == 1:
                raise ConnectionError("Ollama injoignable")
            return _fake_security_gate_runner(state)

        orchestrator = Orchestrator(
            model_registry=registry,
            agent_registry={AgentName.SECURITY_GATE: _flaky_runner},
            retry_policy=RetryPolicy(max_attempts=2),
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {
            "case_id": "CLM-0001",
            "current_step": "claim_intake",
            "intake_result": object(),
        }

        outcome = orchestrator.execute_agent(request, state, model_id="primary")

        assert outcome.success is True
        assert calls["count"] == 2
        actions = [event.action for event in outcome.audit_events]
        assert actions == [
            "authorization", "authorization", "authorization",
            "call", "retry", "call", "result",
        ]
        retry_event = outcome.audit_events[4]
        assert retry_event.action == "retry"
        assert retry_event.details["attempt"] == "2"
        result_event = outcome.audit_events[-1]
        assert result_event.details["attempt"] == "2"
        assert result_event.details["final_status"] == "SUCCESS"

    def test_events_never_leak_secrets_prompts_documents_or_ocr_text(self):
        """Le message d'exception peut contenir un extrait sensible côté
        ``outcome.error`` (comportement existant, hors périmètre de cette
        tâche) — mais les événements d'audit eux-mêmes ne doivent jamais
        reprendre ce texte brut : seuls des identifiants/codes/compteurs y
        figurent (``model_id``, ``tools``, ``policy``, ``attempt``,
        ``final_status`` — jamais le message d'exception)."""
        forbidden_markers = (
            "sk-super-secret-key",
            "PROMPT SYSTEME COMPLET: tu es un assistant...",
            "texte OCR complet de la facture",
            "document brut base64...",
        )

        def _leaky_runner(state):
            raise RuntimeError("panne inattendue contenant " + " ".join(forbidden_markers))

        registry = _two_model_registry()
        orchestrator = Orchestrator(
            model_registry=registry,
            agent_registry={AgentName.SECURITY_GATE: _leaky_runner},
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {
            "case_id": "CLM-0001",
            "current_step": "claim_intake",
            "intake_result": object(),
        }

        outcome = orchestrator.execute_agent(request, state, model_id="primary")

        assert outcome.success is False
        assert outcome.error.code == "AGENT_EXECUTION_FAILED"
        assert len(outcome.audit_events) > 0
        for event in outcome.audit_events:
            serialized = " ".join(event.details.values()) + event.action + event.outcome
            for marker in forbidden_markers:
                assert marker not in serialized
        assert outcome.audit_events[-1].details["final_status"] == "AGENT_EXECUTION_FAILED"

    def test_all_events_expose_only_the_documented_fields(self):
        """Chaque événement expose exactement les mêmes clés de ``details``
        (``model_id``, ``tools``, ``policy``, ``attempt``, ``final_status``)
        — jamais de champ ad hoc, jamais de contenu métier — et une action
        parmi les six natures documentées."""
        registry = _two_model_registry()
        orchestrator = Orchestrator(
            model_registry=registry,
            agent_registry={AgentName.SECURITY_GATE: _fake_security_gate_runner},
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {
            "case_id": "CLM-0001",
            "current_step": "claim_intake",
            "intake_result": object(),
        }

        outcome = orchestrator.execute_agent(request, state, model_id="primary")

        assert len(outcome.audit_events) > 0
        for event in outcome.audit_events:
            assert set(event.details.keys()) == ALL_EVENT_DETAIL_KEYS
            assert event.case_id == "CLM-0001"
            assert event.actor == "security_gate"
            assert event.action in ALL_EVENT_ACTIONS

    def test_audit_events_are_appendable_to_existing_audit_trail(self):
        """``audit_events`` doit pouvoir être ajouté tel quel à
        ``ClaimState.audit_trail`` (interface append-only existante, déjà
        utilisée par ``agents/privacy_agent``) sans transformation ni perte —
        même schéma ``AuditEvent``, aucune nouvelle interface créée."""
        from schemas.results import AuditEvent

        registry = _two_model_registry()
        orchestrator = Orchestrator(
            model_registry=registry,
            agent_registry={AgentName.SECURITY_GATE: _fake_security_gate_runner},
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {
            "case_id": "CLM-0001",
            "current_step": "claim_intake",
            "intake_result": object(),
        }

        outcome = orchestrator.execute_agent(request, state, model_id="primary")

        audit_trail: list[AuditEvent] = []
        audit_trail = audit_trail + list(outcome.audit_events)
        assert all(isinstance(event, AuditEvent) for event in audit_trail)
        assert len(audit_trail) == len(outcome.audit_events)


# ── 5. Persistance AuditStore via normalizer Audit Agent ─────────────────────


class TestAuditStorePersistence:
    def test_agent_call_is_normalized_then_persisted_append_only(self):
        store = AuditStore()
        normalized_calls: list[dict] = []
        orchestrator = Orchestrator(
            model_registry=_two_model_registry(),
            agent_registry={AgentName.SECURITY_GATE: _fake_security_gate_runner},
            audit_store=store,
            audit_normalizer=_audit_normalizer(normalized_calls),
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {
            "case_id": "CLM-0001",
            "current_step": "claim_intake",
            "intake_result": object(),
        }

        outcome = orchestrator.execute_agent(request, state, model_id="primary")

        persisted = store.read_by_case_id("CLM-0001")
        assert outcome.success is True
        assert len(persisted) == len(outcome.audit_events) == len(normalized_calls)
        assert any(call["action"] == "call" for call in normalized_calls)
        assert store.verify_claim_integrity("CLM-0001").intact is True

    def test_model_failure_is_normalized_and_persisted_as_error(self):
        store = AuditStore()
        normalized_calls: list[dict] = []

        def model_down(state):
            raise ConnectionError("modele indisponible")

        orchestrator = Orchestrator(
            model_registry=_two_model_registry(),
            agent_registry={AgentName.SECURITY_GATE: model_down},
            audit_store=store,
            audit_normalizer=_audit_normalizer(normalized_calls),
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {
            "case_id": "CLM-0001",
            "current_step": "claim_intake",
            "intake_result": object(),
        }

        outcome = orchestrator.execute_agent(request, state, model_id="primary")

        persisted = store.read_by_case_id("CLM-0001")
        assert outcome.success is False
        assert outcome.error.code == "AGENT_EXECUTION_FAILED"
        assert len(persisted) == len(outcome.audit_events)
        assert any(event.outcome == "result:AGENT_EXECUTION_FAILED" for event in persisted)
        assert store.verify_claim_integrity("CLM-0001").intact is True

    def test_invalid_output_is_normalized_and_persisted_as_error(self):
        store = AuditStore()
        normalized_calls: list[dict] = []

        def invalid_output(state):
            return {"security_result": {"claim_id": state["case_id"], "decision": "ALLOW"}}

        orchestrator = Orchestrator(
            model_registry=_two_model_registry(),
            agent_registry={AgentName.SECURITY_GATE: invalid_output},
            audit_store=store,
            audit_normalizer=_audit_normalizer(normalized_calls),
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {
            "case_id": "CLM-0001",
            "current_step": "claim_intake",
            "intake_result": object(),
        }

        outcome = orchestrator.execute_agent(request, state, model_id="primary")

        persisted = store.read_by_case_id("CLM-0001")
        assert outcome.success is False
        assert outcome.error.code == "AGENT_RESULT_INVALID"
        assert len(persisted) == len(outcome.audit_events)
        assert any(event.outcome == "result:AGENT_RESULT_INVALID" for event in persisted)
        assert store.verify_claim_integrity("CLM-0001").intact is True

    def test_retry_is_normalized_and_persisted(self):
        store = AuditStore()
        normalized_calls: list[dict] = []
        calls = {"count": 0}

        def flaky(state):
            calls["count"] += 1
            if calls["count"] == 1:
                raise ConnectionError("modele temporairement indisponible")
            return _fake_security_gate_runner(state)

        orchestrator = Orchestrator(
            model_registry=_two_model_registry(),
            agent_registry={AgentName.SECURITY_GATE: flaky},
            retry_policy=RetryPolicy(max_attempts=2),
            audit_store=store,
            audit_normalizer=_audit_normalizer(normalized_calls),
        )
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        state = {
            "case_id": "CLM-0001",
            "current_step": "claim_intake",
            "intake_result": object(),
        }

        outcome = orchestrator.execute_agent(request, state, model_id="primary")

        persisted = store.read_by_case_id("CLM-0001")
        assert outcome.success is True
        assert calls["count"] == 2
        assert any(event.event_type is AuditEventType.RETRY for event in persisted)
        assert any(call["action"] == "retry" for call in normalized_calls)
        assert len(persisted) == len(outcome.audit_events)
        assert store.verify_claim_integrity("CLM-0001").intact is True
