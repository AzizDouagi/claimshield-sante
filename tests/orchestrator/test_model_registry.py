"""Tests du registre injectable de modèles — orchestrator/model_registry.py.

Couvre : inscription (register), sélection (select_for_agent), refus
explicite (modèle absent/désactivé/incompatible) et indisponibilité
(client_factory qui échoue à l'appel).
"""
from __future__ import annotations

from dataclasses import fields

import pytest

from orchestrator.model_registry import (
    AGENT_REQUIRED_CAPABILITIES,
    ModelCapability,
    ModelDisabledError,
    ModelIncompatibleError,
    ModelNotFoundError,
    ModelRegistry,
    ModelRegistryError,
    ModelSpec,
    ModelUnavailableError,
    build_default_registry,
)
from orchestrator.orchestrator import AgentName


class _FakeClient:
    """Client factice représentant un ChatOllama instancié."""


def _spec(
    model_id: str = "ollama-gemma4",
    *,
    capabilities: frozenset[ModelCapability] | None = None,
    enabled: bool = True,
    client_factory=None,
) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        provider="ollama",
        capabilities=capabilities
        if capabilities is not None
        else frozenset({ModelCapability.STRUCTURED_OUTPUT, ModelCapability.TOOL_CALLING}),
        client_factory=client_factory or _FakeClient,
        enabled=enabled,
    )


# ── 1. Inscription (register) ─────────────────────────────────────────────────


class TestRegistration:
    def test_register_valid_spec_succeeds(self):
        registry = ModelRegistry()
        registry.register(_spec())
        assert registry.is_registered("ollama-gemma4")

    def test_registered_model_appears_in_list_models(self):
        registry = ModelRegistry()
        registry.register(_spec("model-a"))
        registry.register(_spec("model-b"))
        ids = {m.model_id for m in registry.list_models()}
        assert ids == {"model-a", "model-b"}

    def test_duplicate_model_id_rejected(self):
        registry = ModelRegistry()
        registry.register(_spec("dup"))
        with pytest.raises(ValueError, match="déjà enregistré"):
            registry.register(_spec("dup"))

    def test_fresh_registry_starts_empty(self):
        registry = ModelRegistry()
        assert registry.list_models() == ()
        assert not registry.is_registered("anything")

    def test_two_registries_do_not_share_state(self):
        """Aucune instance globale cachée : chaque registre est indépendant."""
        registry_a = ModelRegistry()
        registry_b = ModelRegistry()
        registry_a.register(_spec("only-in-a"))
        assert registry_a.is_registered("only-in-a")
        assert not registry_b.is_registered("only-in-a")

    def test_empty_model_id_rejected(self):
        with pytest.raises(ValueError):
            _spec(model_id="")

    def test_blank_model_id_rejected(self):
        with pytest.raises(ValueError):
            _spec(model_id="   ")

    def test_non_callable_client_factory_rejected(self):
        with pytest.raises(TypeError):
            ModelSpec(
                model_id="x",
                provider="ollama",
                capabilities=frozenset(),
                client_factory="not_a_callable",  # type: ignore[arg-type]
            )

    def test_capabilities_must_be_frozenset(self):
        with pytest.raises(TypeError):
            ModelSpec(
                model_id="x",
                provider="ollama",
                capabilities=[ModelCapability.STRUCTURED_OUTPUT],  # type: ignore[arg-type]
                client_factory=_FakeClient,
            )

    def test_model_spec_never_stores_secret_or_url_fields(self):
        """Garantie structurelle : ModelSpec ne peut pas dériver vers un
        champ api_key/secret/token/base_url/url/client."""
        names = {f.name for f in fields(ModelSpec)}
        forbidden = {"api_key", "secret", "token", "password", "base_url", "url", "client"}
        assert names & forbidden == set()
        assert names == {"model_id", "provider", "capabilities", "client_factory", "enabled"}


# ── 2. Sélection (select_for_agent) ───────────────────────────────────────────


class TestSelection:
    def test_select_compatible_model_returns_spec(self):
        registry = ModelRegistry()
        registry.register(_spec())
        spec = registry.select_for_agent(AgentName.SECURITY_GATE, "ollama-gemma4")
        assert spec.model_id == "ollama-gemma4"

    def test_agent_required_capabilities_has_one_entry_per_agent(self):
        assert set(AGENT_REQUIRED_CAPABILITIES.keys()) == set(AgentName)

    def test_structured_output_agents_require_structured_output(self):
        for agent in (
            AgentName.CLAIM_INTAKE,
            AgentName.SECURITY_GATE,
            AgentName.PRIVACY,
            AgentName.FHIR_VALIDATOR,
        ):
            assert ModelCapability.STRUCTURED_OUTPUT in AGENT_REQUIRED_CAPABILITIES[agent]

    def test_tool_calling_agents_require_tool_calling(self):
        for agent in (AgentName.MEDICAL_CODING, AgentName.DOCUMENT_OCR):
            assert ModelCapability.TOOL_CALLING in AGENT_REQUIRED_CAPABILITIES[agent]

    def test_stub_agents_require_no_capability(self):
        for agent in (
            AgentName.IDENTITY_COVERAGE,
            AgentName.CLINICAL_CONSISTENCY,
            AgentName.FRAUD_DETECTION,
            AgentName.CASE_REVIEWER,
            AgentName.AUDIT,
        ):
            assert AGENT_REQUIRED_CAPABILITIES[agent] == frozenset()

    def test_default_registry_serves_all_llm_backed_agents(self):
        registry = build_default_registry()
        model_id = registry.list_models()[0].model_id
        for agent in (
            AgentName.CLAIM_INTAKE,
            AgentName.SECURITY_GATE,
            AgentName.PRIVACY,
            AgentName.FHIR_VALIDATOR,
            AgentName.MEDICAL_CODING,
            AgentName.DOCUMENT_OCR,
        ):
            spec = registry.select_for_agent(agent, model_id)
            assert spec.model_id == model_id

    def test_default_registry_has_exactly_one_model(self):
        registry = build_default_registry()
        assert len(registry.list_models()) == 1


# ── 3. Refus explicite ────────────────────────────────────────────────────────


class TestExplicitRefusal:
    def test_unknown_model_id_raises_not_found(self):
        registry = ModelRegistry()
        with pytest.raises(ModelNotFoundError) as exc_info:
            registry.get("does-not-exist")
        assert exc_info.value.model_id == "does-not-exist"
        assert exc_info.value.structured.code == "MODEL_NOT_FOUND"

    def test_disabled_model_raises_disabled(self):
        registry = ModelRegistry()
        registry.register(_spec("off", enabled=False))
        with pytest.raises(ModelDisabledError) as exc_info:
            registry.get("off")
        assert exc_info.value.structured.code == "MODEL_DISABLED"

    def test_incompatible_model_raises_incompatible(self):
        registry = ModelRegistry()
        registry.register(
            _spec("structured-only", capabilities=frozenset({ModelCapability.STRUCTURED_OUTPUT}))
        )
        with pytest.raises(ModelIncompatibleError) as exc_info:
            registry.select_for_agent(AgentName.MEDICAL_CODING, "structured-only")
        assert exc_info.value.missing_capabilities == frozenset({ModelCapability.TOOL_CALLING})
        assert exc_info.value.structured.code == "MODEL_INCOMPATIBLE"

    def test_select_for_agent_on_unknown_model_raises_not_found(self):
        registry = ModelRegistry()
        with pytest.raises(ModelNotFoundError):
            registry.select_for_agent(AgentName.SECURITY_GATE, "nope")

    def test_select_for_agent_on_disabled_model_raises_disabled_not_incompatible(self):
        registry = ModelRegistry()
        registry.register(_spec("off", enabled=False))
        with pytest.raises(ModelDisabledError):
            registry.select_for_agent(AgentName.SECURITY_GATE, "off")

    def test_all_registry_errors_are_value_errors_with_structured_payload(self):
        registry = ModelRegistry()
        try:
            registry.get("nope")
        except ModelRegistryError as exc:
            assert isinstance(exc, ValueError)
            assert exc.structured.message
        else:
            pytest.fail("ModelNotFoundError attendue")

    def test_disabled_model_still_reports_incompatibility_reason_via_get(self):
        """Un modèle désactivé est refusé avant même la vérification de
        compatibilité — le message reflète bien 'désactivé', pas 'incompatible'."""
        registry = ModelRegistry()
        registry.register(
            _spec("off", enabled=False, capabilities=frozenset())
        )
        with pytest.raises(ModelDisabledError):
            registry.select_for_agent(AgentName.AUDIT, "off")


# ── 4. Indisponibilité (client_factory échoue) ────────────────────────────────


class TestUnavailability:
    def test_client_factory_success_returns_client(self):
        registry = ModelRegistry()
        registry.register(_spec("ok", client_factory=_FakeClient))
        client = registry.get_client("ok")
        assert isinstance(client, _FakeClient)

    def test_client_factory_failure_wrapped_in_unavailable_error(self):
        def _raises():
            raise ConnectionError("Ollama injoignable")

        registry = ModelRegistry()
        registry.register(_spec("flaky", client_factory=_raises))
        with pytest.raises(ModelUnavailableError) as exc_info:
            registry.get_client("flaky")
        assert exc_info.value.structured.code == "MODEL_UNAVAILABLE"
        assert isinstance(exc_info.value.__cause__, ConnectionError)

    def test_unavailable_error_preserves_original_exception_message(self):
        def _raises():
            raise RuntimeError("connection refused")

        registry = ModelRegistry()
        registry.register(_spec("flaky", client_factory=_raises))
        with pytest.raises(ModelUnavailableError, match="connection refused"):
            registry.get_client("flaky")

    def test_get_client_on_unknown_model_raises_not_found_not_unavailable(self):
        registry = ModelRegistry()
        with pytest.raises(ModelNotFoundError):
            registry.get_client("ghost")

    def test_get_client_on_disabled_model_raises_disabled_not_unavailable(self):
        registry = ModelRegistry()
        registry.register(_spec("off", enabled=False))
        with pytest.raises(ModelDisabledError):
            registry.get_client("off")

    def test_get_client_for_agent_success(self):
        registry = ModelRegistry()
        registry.register(_spec("ok"))
        client = registry.get_client_for_agent(AgentName.SECURITY_GATE, "ok")
        assert isinstance(client, _FakeClient)

    def test_get_client_for_agent_incompatible_never_calls_factory(self):
        calls = {"count": 0}

        def _factory():
            calls["count"] += 1
            return _FakeClient()

        registry = ModelRegistry()
        registry.register(
            _spec(
                "structured-only",
                capabilities=frozenset({ModelCapability.STRUCTURED_OUTPUT}),
                client_factory=_factory,
            )
        )
        with pytest.raises(ModelIncompatibleError):
            registry.get_client_for_agent(AgentName.MEDICAL_CODING, "structured-only")
        assert calls["count"] == 0

    def test_get_client_for_agent_unavailable(self):
        def _raises():
            raise ConnectionError("refused")

        registry = ModelRegistry()
        registry.register(_spec("flaky", client_factory=_raises))
        with pytest.raises(ModelUnavailableError):
            registry.get_client_for_agent(AgentName.SECURITY_GATE, "flaky")

    def test_default_registry_client_factory_is_get_llm(self):
        from llm.factory import get_llm

        registry = build_default_registry()
        spec = registry.get(registry.list_models()[0].model_id)
        assert spec.client_factory is get_llm


# ── 5. Fallback (find_fallback) ────────────────────────────────────────────


class TestFindFallback:
    def test_returns_first_other_compatible_enabled_model(self):
        registry = ModelRegistry()
        registry.register(_spec("primary"))
        registry.register(_spec("secondary"))
        fallback = registry.find_fallback(AgentName.SECURITY_GATE, exclude_model_id="primary")
        assert fallback is not None
        assert fallback.model_id == "secondary"

    def test_excludes_the_failing_model_itself(self):
        registry = ModelRegistry()
        registry.register(_spec("only-one"))
        fallback = registry.find_fallback(AgentName.SECURITY_GATE, exclude_model_id="only-one")
        assert fallback is None

    def test_skips_disabled_candidates(self):
        registry = ModelRegistry()
        registry.register(_spec("primary"))
        registry.register(_spec("disabled-secondary", enabled=False))
        fallback = registry.find_fallback(AgentName.SECURITY_GATE, exclude_model_id="primary")
        assert fallback is None

    def test_skips_incompatible_candidates(self):
        registry = ModelRegistry()
        registry.register(_spec("primary"))
        registry.register(
            _spec(
                "structured-only-secondary",
                capabilities=frozenset({ModelCapability.STRUCTURED_OUTPUT}),
            )
        )
        fallback = registry.find_fallback(AgentName.MEDICAL_CODING, exclude_model_id="primary")
        assert fallback is None

    def test_returns_none_when_registry_empty(self):
        registry = ModelRegistry()
        assert registry.find_fallback(AgentName.SECURITY_GATE, exclude_model_id="ghost") is None

    def test_never_raises_even_for_unknown_exclude_model_id(self):
        registry = ModelRegistry()
        registry.register(_spec("only-one"))
        fallback = registry.find_fallback(AgentName.SECURITY_GATE, exclude_model_id="ghost")
        assert fallback is not None
        assert fallback.model_id == "only-one"
