"""Tests unitaires des adaptateurs de nœuds — graph/nodes.py.

Depuis le câblage de l'orchestrateur (``orchestrator/executor.py``), chaque
nœud agent appelle exclusivement ``Orchestrator.execute_agent()`` — plus
aucun appel direct à ``agent_module.node(state)`` depuis ce module. Les
agents réels restent mockés via ``unittest.mock.patch`` sur leur module
(``agents.<pkg>.agent.node``) : l'orchestrateur résout ce nom à l'appel
(fermeture, jamais capturé à la construction), donc le patch reste efficace
même une fois l'agent enregistré dans ``orchestrator.agent_registry``.

``TestOrchestratorSpy`` est le test exigé : il espionne
``Orchestrator.execute_agent`` pour prouver que le nœud le traverse
réellement (jamais un raccourci direct vers l'agent).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError

from graph.nodes import (
    _AGENT_CONFIGS,
    _AgentConfig,
    _build_request,
    _exception_fallback,
    _graph_preconditions_check,
    _translate_outcome,
    _validate_result_type,
    build_node_registry,
    build_orchestrator,
)
from orchestrator.executor import Orchestrator, RetryPolicy
from orchestrator.model_registry import ModelCapability, ModelRegistry, ModelSpec, build_default_registry
from orchestrator.orchestrator import AgentCallOutcome, AgentCallRequest, AgentName
from orchestrator.policies import build_authorized_tools
from schemas.domain import IntakeStatus, SecurityDecision, VerificationStatus
from schemas.results import (
    ClaimIntakeResult,
    ClaimManifest,
    DocumentOcrResult,
    FhirValidatorResult,
    IdentityCoverageResult,
    IdentityResult,
    CoverageResult,
    MedicalCodingResult,
    PrivacyResult,
    SecurityGateResult,
    StructuredError,
)
from schemas.domain import DataClassification, ExtractionStatus, DocumentType, OcrSource


# ── Fixtures et helpers ────────────────────────────────────────────────────────


def _state(case_id: str = "CLM-0001", **extra) -> dict:
    return {"case_id": case_id, **extra}


def _valid_result(name: str, case_id: str = "CLM-0001"):
    """Instance valide minimale du modèle de résultat attendu par ``name``."""
    if name == "claim_intake":
        return ClaimIntakeResult(
            claim_id=case_id,
            status=IntakeStatus.ACCEPTED,
            manifest=ClaimManifest(
                claim_id=case_id, file_count=1, total_size_bytes=10, status=IntakeStatus.ACCEPTED
            ),
            accepted_count=1,
            quarantined_count=0,
        )
    if name == "security_gate":
        return SecurityGateResult(claim_id=case_id, decision=SecurityDecision.ALLOW, reasons=["ok"])
    if name == "privacy":
        return PrivacyResult(
            case_id=case_id,
            status=VerificationStatus.PASS,
            data_classification=DataClassification.SYNTHETIC_TEST_DATA,
            contains_real_personal_data=False,
        )
    if name == "fhir_validator":
        return FhirValidatorResult(case_id=case_id, status=VerificationStatus.PASS, bundle_expected=True)
    if name == "medical_coding":
        return MedicalCodingResult(case_id=case_id, status=VerificationStatus.PASS)
    if name == "document_ocr":
        return DocumentOcrResult(
            claim_id=case_id,
            file_path="incoming/CLM-0001/facture.pdf",
            sha256="a" * 64,
            mime_type="application/pdf",
            extraction_status=ExtractionStatus.SUCCESS,
            status=VerificationStatus.PASS,
            document_type=DocumentType.INVOICE,
            ocr_source=OcrSource.PDF_TEXT,
            artifact_id="ocr-artifact-1",
            artifact_path="artifacts/document_ocr/CLM-0001/facture.json",
        )
    if name == "identity_coverage":
        return IdentityCoverageResult(
            case_id=case_id,
            identity=IdentityResult(status=VerificationStatus.PASS),
            coverage=CoverageResult(status=VerificationStatus.PASS),
        )
    raise ValueError(f"agent inconnu : {name!r}")


_REAL_AGENTS: tuple[tuple[str, str, str], ...] = (
    ("claim_intake", "agents.claim_intake_agent.agent.node", "intake_result"),
    ("security_gate", "agents.security_gate_agent.agent.node", "security_result"),
    ("privacy", "agents.privacy_agent.agent.node", "privacy_result"),
    ("fhir_validator", "agents.fhir_validator_agent.agent.node", "fhir_result"),
    ("medical_coding", "agents.medical_coding_agent.agent.node", "coding_result"),
    ("document_ocr", "agents.document_ocr_agent.agent.node", "ocr_result"),
    ("identity_coverage", "agents.identity_coverage_agent.agent.node", "identity_coverage_result"),
)
"""(nom, cible du patch, clé du résultat) — les 7 agents réels."""


def _agent_update(name: str, step: str, case_id: str = "CLM-0001") -> dict:
    """Mise à jour ClaimState simulée par un agent qui réussit — même forme
    que celle réellement retournée par ``agent_module.node`` (étape 10)."""
    result_key = {n: k for n, _, k in _REAL_AGENTS}[name]
    return {
        result_key: _valid_result(name, case_id),
        "current_step": step,
        "completed_steps": [step],
    }


# ── 1. Helpers internes (inchangés) ───────────────────────────────────────────


class TestValidateResultType:
    def test_none_is_accepted(self):
        _validate_result_type({}, "key", ClaimIntakeResult)

    def test_correct_instance_passes(self):
        instance = _valid_result("claim_intake")
        _validate_result_type({"key": instance}, "key", ClaimIntakeResult)

    def test_wrong_type_raises_type_error(self):
        with pytest.raises(TypeError, match="invalide"):
            _validate_result_type({"key": "bad_value"}, "key", ClaimIntakeResult)

    def test_wrong_type_name_appears_in_message(self):
        with pytest.raises(TypeError) as exc_info:
            _validate_result_type({"key": 42}, "key", SecurityGateResult)
        assert "int" in str(exc_info.value)
        assert "SecurityGateResult" in str(exc_info.value)

    def test_key_absent_is_accepted(self):
        _validate_result_type({"other_key": "value"}, "key", ClaimIntakeResult)

    def test_invalid_dict_raises_validation_error(self):
        with pytest.raises(ValidationError):
            _validate_result_type({"key": {"bad": "structure"}}, "key", SecurityGateResult)


class TestExceptionFallback:
    CFG = _AgentConfig(
        agent_name="test_agent",
        agent_enum=AgentName.CLAIM_INTAKE,
        step_name="test_step",
        result_key="test_result",
        result_model=ClaimIntakeResult,
        input_key="test_input",
    )

    def test_returns_dict(self):
        result = _exception_fallback(_state(), self.CFG, RuntimeError("boom"))
        assert isinstance(result, dict)

    def test_errors_list_populated(self):
        result = _exception_fallback(_state(), self.CFG, RuntimeError("boom"))
        assert len(result["errors"]) == 1

    def test_error_contains_agent_name(self):
        result = _exception_fallback(_state(), self.CFG, RuntimeError("boom"))
        assert "test_agent" in result["errors"][0]

    def test_error_contains_exception_type(self):
        result = _exception_fallback(_state(), self.CFG, ValueError("bad"))
        assert "ValueError" in result["errors"][0]

    def test_error_contains_exception_message(self):
        result = _exception_fallback(_state(), self.CFG, RuntimeError("detail error"))
        assert "detail error" in result["errors"][0]

    def test_input_key_set_to_none(self):
        result = _exception_fallback(_state(), self.CFG, RuntimeError("x"))
        assert result["test_input"] is None

    def test_completed_steps_set(self):
        result = _exception_fallback(_state(), self.CFG, RuntimeError("x"))
        assert "test_step" in result["completed_steps"]

    def test_current_step_set(self):
        result = _exception_fallback(_state(), self.CFG, RuntimeError("x"))
        assert result["current_step"] == "test_step"

    def test_no_input_key_config(self):
        cfg = _AgentConfig(
            agent_name="no_input",
            agent_enum=AgentName.CLAIM_INTAKE,
            step_name="step",
            result_key="result",
            result_model=ClaimIntakeResult,
            input_key=None,
        )
        result = _exception_fallback(_state(), cfg, RuntimeError("x"))
        assert "test_input" not in result
        assert "errors" in result


# ── 2. _build_request ─────────────────────────────────────────────────────────


class TestBuildRequest:
    CFG = _AGENT_CONFIGS["security_gate"]

    def test_builds_request_from_state(self):
        request = _build_request(_state(current_step="claim_intake"), self.CFG)
        assert request.agent_name is AgentName.SECURITY_GATE
        assert request.case_id == "CLM-0001"
        assert request.current_step == "claim_intake"
        assert request.requested_model == "SecurityGateResult"

    def test_missing_current_step_falls_back_to_step_name(self):
        request = _build_request(_state(), self.CFG)
        assert request.current_step == self.CFG.step_name

    def test_invalid_case_id_raises_validation_error(self):
        with pytest.raises(ValidationError):
            _build_request({}, self.CFG)


# ── 3. _graph_preconditions_check ─────────────────────────────────────────────


class TestGraphPreconditionsCheck:
    def test_matching_case_id_is_allowed(self):
        request = AgentCallRequest(
            agent_name=AgentName.SECURITY_GATE,
            case_id="CLM-0001",
            current_step="claim_intake",
            requested_model="SecurityGateResult",
        )
        decision = _graph_preconditions_check(_state(), request)
        assert decision.allowed is True

    def test_mismatched_case_id_is_denied(self):
        request = AgentCallRequest(
            agent_name=AgentName.SECURITY_GATE,
            case_id="CLM-0002",
            current_step="claim_intake",
            requested_model="SecurityGateResult",
        )
        decision = _graph_preconditions_check(_state("CLM-0001"), request)
        assert decision.allowed is False
        assert decision.reason.code == "CASE_ID_MISMATCH"

    def test_absent_state_case_id_is_allowed(self):
        """Aucune référence à comparer : l'ordre est garanti par la
        topologie du graphe, pas par ce contrôle allégé."""
        request = AgentCallRequest(
            agent_name=AgentName.SECURITY_GATE,
            case_id="CLM-0001",
            current_step="claim_intake",
            requested_model="SecurityGateResult",
        )
        decision = _graph_preconditions_check({}, request)
        assert decision.allowed is True


# ── 4. _translate_outcome ──────────────────────────────────────────────────────


class TestTranslateOutcome:
    CFG = _AGENT_CONFIGS["security_gate"]

    def _request(self) -> AgentCallRequest:
        return AgentCallRequest(
            agent_name=AgentName.SECURITY_GATE,
            case_id="CLM-0001",
            current_step="claim_intake",
            requested_model="SecurityGateResult",
        )

    def test_success_preserves_state_updates_from_agent(self):
        result = _valid_result("security_gate")
        outcome = AgentCallOutcome.from_request(
            self._request(),
            success=True,
            result_payload=result.model_dump(),
            state_updates={
                "security_result": result,
                "security_input": None,
                "current_step": "security_gate",
                "completed_steps": ["security_gate"],
            },
        )
        updates = _translate_outcome(self.CFG, outcome)
        assert updates["current_step"] == "security_gate"
        assert updates["completed_steps"] == ["security_gate"]
        assert updates["security_input"] is None

    def test_success_result_key_is_reinstantiated_and_validated(self):
        result = _valid_result("security_gate")
        outcome = AgentCallOutcome.from_request(
            self._request(),
            success=True,
            result_payload=result.model_dump(),
            state_updates={"security_result": result},
        )
        updates = _translate_outcome(self.CFG, outcome)
        assert isinstance(updates["security_result"], SecurityGateResult)
        assert updates["security_result"].decision == SecurityDecision.ALLOW

    def test_success_appends_audit_events_to_existing_audit_trail(self):
        from schemas.results import AuditEvent

        result = _valid_result("security_gate")
        own_event = AuditEvent(event_id="own", case_id="CLM-0001", actor="security_gate", action="view", outcome="ok")
        orchestrator_event = AuditEvent(
            event_id="orch", case_id="CLM-0001", actor="security_gate", action="result", outcome="SUCCESS"
        )
        outcome = AgentCallOutcome.from_request(
            self._request(),
            success=True,
            result_payload=result.model_dump(),
            state_updates={"security_result": result, "audit_trail": [own_event]},
            audit_events=[orchestrator_event],
        )
        updates = _translate_outcome(self.CFG, outcome)
        assert updates["audit_trail"] == [own_event, orchestrator_event]

    def test_failure_produces_structured_error(self):
        outcome = AgentCallOutcome.from_request(
            self._request(),
            success=False,
            error=StructuredError(code="PRECONDITION_RESULT_MISSING", message="résultat requis absent"),
        )
        updates = _translate_outcome(self.CFG, outcome)
        assert updates["errors"] == ["[security_gate] PRECONDITION_RESULT_MISSING : résultat requis absent"]
        assert updates["completed_steps"] == ["security_gate"]
        assert updates["current_step"] == "security_gate"
        assert updates["security_input"] is None

    def test_failure_includes_audit_events(self):
        from schemas.results import AuditEvent

        event = AuditEvent(event_id="1", case_id="CLM-0001", actor="security_gate", action="refusal", outcome="DENY")
        outcome = AgentCallOutcome.from_request(
            self._request(),
            success=False,
            error=StructuredError(code="MODEL_DISABLED", message="modèle désactivé"),
            audit_events=[event],
        )
        updates = _translate_outcome(self.CFG, outcome)
        assert updates["audit_trail"] == [event]


# ── 5. Nœuds réels — délégation, isolation, invalidité (via l'orchestrateur) ─


@pytest.fixture
def orchestrator() -> Orchestrator:
    return build_orchestrator()


@pytest.fixture
def node_registry(orchestrator):
    return build_node_registry(orchestrator)


class TestRealAgentNodes:
    @pytest.mark.parametrize("name,patch_path,result_key", _REAL_AGENTS)
    def test_delegates_to_agent_and_preserves_bookkeeping(self, node_registry, name, patch_path, result_key):
        expected = _agent_update(name, step=name)
        with patch(patch_path, return_value=expected) as mock_node:
            result = node_registry[name](_state())
        mock_node.assert_called_once()
        assert result["current_step"] == expected["current_step"]
        assert result["completed_steps"] == expected["completed_steps"]
        assert result[result_key] == expected[result_key]
        assert "audit_trail" in result and len(result["audit_trail"]) > 0

    @pytest.mark.parametrize("name,patch_path,result_key", _REAL_AGENTS)
    def test_agent_exception_returns_structured_error(self, node_registry, name, patch_path, result_key):
        with patch(patch_path, side_effect=RuntimeError("panne agent")):
            result = node_registry[name](_state())
        assert result["errors"]
        assert name in result["errors"][0]
        assert "panne agent" in result["errors"][0]

    @pytest.mark.parametrize("name,patch_path,result_key", _REAL_AGENTS)
    def test_wrong_result_type_returns_structured_error(self, node_registry, name, patch_path, result_key):
        with patch(patch_path, return_value={result_key: "not_a_valid_model"}):
            result = node_registry[name](_state())
        assert result["errors"]

    @pytest.mark.parametrize("name,patch_path,result_key", _REAL_AGENTS)
    def test_missing_result_field_returns_structured_error(self, node_registry, name, patch_path, result_key):
        with patch(patch_path, return_value={"current_step": name}):
            result = node_registry[name](_state())
        assert result["errors"]

    @pytest.mark.parametrize("name,patch_path,result_key", _REAL_AGENTS)
    def test_invalid_state_never_raises(self, node_registry, name, patch_path, result_key):
        """State sans case_id valide : jamais d'exception propagée, jamais
        l'agent appelé (échec dès la construction de la requête)."""
        with patch(patch_path) as mock_node:
            result = node_registry[name]({})
        mock_node.assert_not_called()
        assert isinstance(result, dict)
        assert result.get("errors")


# ── 6. build_orchestrator / build_node_registry ───────────────────────────────


class TestBuildNodeRegistry:
    EXPECTED_KEYS = {
        "claim_intake", "security_gate", "privacy", "fhir_validator",
        "medical_coding", "document_ocr", "identity_coverage",
        "clinical_consistency", "fraud_detection", "case_reviewer", "audit",
    }

    def test_registry_contains_all_expected_keys(self, node_registry):
        assert set(node_registry.keys()) == self.EXPECTED_KEYS

    def test_all_entries_are_callable(self, node_registry):
        for name, fn in node_registry.items():
            assert callable(fn), f"{name} n'est pas callable"

    def test_node_names_match_registry_keys(self, node_registry):
        for key, fn in node_registry.items():
            assert fn.__name__ == f"node_{key}"

    def test_build_orchestrator_never_shares_state_across_calls(self):
        """Aucune instance globale cachée : deux appels retournent deux
        orchestrateurs indépendants."""
        first = build_orchestrator()
        second = build_orchestrator()
        assert first is not second
        assert first.agent_registry is not second.agent_registry

    def test_build_orchestrator_uses_default_model_registry(self):
        orch = build_orchestrator()
        default = build_default_registry()
        assert {m.model_id for m in orch.model_registry.list_models()} == {
            m.model_id for m in default.list_models()
        }

    def test_build_orchestrator_retry_policy_uses_settings_by_default(self):
        from config.settings import get_settings

        orch = build_orchestrator()
        assert orch.retry_policy.max_attempts == get_settings().claimshield_max_node_retry_attempts

    def test_build_orchestrator_accepts_explicit_retry_policy(self):
        custom = RetryPolicy(max_attempts=7)
        orch = build_orchestrator(retry_policy=custom)
        assert orch.retry_policy is custom


# ── 7. Retry technique (désormais porté par RetryPolicy de l'orchestrateur) ──


class TestNodeRetryIntegration:
    """Le retry sur erreur transitoire est désormais celui de
    ``Orchestrator.retry_policy`` — configuré explicitement par test plutôt
    que par monkeypatch de ``get_settings`` (l'orchestrateur est construit
    une fois, en dehors de tout appel de nœud)."""

    def test_transient_error_retried_then_succeeds(self):
        orch = build_orchestrator(retry_policy=RetryPolicy(max_attempts=3))
        registry = build_node_registry(orch)
        expected = _agent_update("security_gate", step="security_gate")
        mock_node = MagicMock(side_effect=[httpx.ConnectError("connexion refusée"), expected])

        with patch("agents.security_gate_agent.agent.node", mock_node):
            result = registry["security_gate"](_state())

        assert mock_node.call_count == 2
        assert result["security_result"] == expected["security_result"]

    def test_transient_error_exhausts_retries_then_structured_fallback(self):
        orch = build_orchestrator(retry_policy=RetryPolicy(max_attempts=2))
        registry = build_node_registry(orch)
        mock_node = MagicMock(side_effect=ConnectionError("connexion refusée"))

        with patch("agents.security_gate_agent.agent.node", mock_node):
            result = registry["security_gate"](_state())

        assert mock_node.call_count == 2
        assert result.get("errors")
        assert "security_gate" in result["errors"][0]
        assert "ConnectionError" in result["errors"][0]

    def test_non_transient_exception_never_retried(self):
        """Comportement inchangé depuis l'étape 10 : une panne non
        catégorisée (RuntimeError) n'est jamais retentée automatiquement,
        quel que soit ``max_attempts``."""
        orch = build_orchestrator(retry_policy=RetryPolicy(max_attempts=3))
        registry = build_node_registry(orch)
        mock_node = MagicMock(side_effect=RuntimeError("Ollama indisponible"))

        with patch("agents.security_gate_agent.agent.node", mock_node):
            result = registry["security_gate"](_state())

        assert mock_node.call_count == 1
        assert "RuntimeError" in result["errors"][0]

    def test_max_attempts_of_one_calls_once(self):
        orch = build_orchestrator(retry_policy=RetryPolicy(max_attempts=1))
        registry = build_node_registry(orch)
        mock_node = MagicMock(side_effect=httpx.ConnectError("connexion refusée"))

        with patch("agents.claim_intake_agent.agent.node", mock_node):
            result = registry["claim_intake"](_state())

        assert mock_node.call_count == 1
        assert result.get("errors")

    def test_injected_stub_impl_retries_transient_error_then_succeeds(self):
        calls = {"count": 0}

        class _FlakyThenPass:
            def run(self, state):
                calls["count"] += 1
                if calls["count"] < 2:
                    raise ConnectionError("connexion refusée")
                from schemas.results import ClinicalConsistencyResult

                return ClinicalConsistencyResult(
                    case_id=str(state.get("case_id", "CLM-0000")),
                    status=VerificationStatus.PASS,
                    confidence=0.9,
                    reasons=["test: retry réussi"],
                )

        orch = build_orchestrator(
            clinical_consistency_impl=_FlakyThenPass(), retry_policy=RetryPolicy(max_attempts=3)
        )
        registry = build_node_registry(orch)

        updates = registry["clinical_consistency"](_state())

        assert calls["count"] == 2
        assert updates["clinical_result"].status is VerificationStatus.PASS

    def test_injected_stub_impl_non_transient_exception_not_retried(self):
        calls = {"count": 0}

        class _AlwaysCrashes:
            def run(self, state):
                calls["count"] += 1
                raise RuntimeError("boom")

        orch = build_orchestrator(fraud_detection_impl=_AlwaysCrashes(), retry_policy=RetryPolicy(max_attempts=3))
        registry = build_node_registry(orch)

        updates = registry["fraud_detection"](_state())

        assert calls["count"] == 1
        assert updates.get("errors")


# ── 8. Spy — preuve que l'orchestrateur est bien traversé (test exigé) ──────


def _spy_on_execute_agent(orchestrator: Orchestrator) -> MagicMock:
    """Espionne ``Orchestrator.execute_agent`` sans altérer son comportement.

    ``Orchestrator`` est un dataclass gelé (``frozen=True``) : impossible
    d'assigner ``execute_agent`` sur l'instance elle-même
    (``patch.object(orchestrator, ...)`` lèverait ``FrozenInstanceError``).
    Le spy patche donc la méthode au niveau de la **classe** — un
    ``MagicMock`` n'étant pas un descripteur, ``orchestrator.execute_agent``
    résout alors vers ce même mock pour toute instance, appelé sans ``self``
    implicite ; le spy rappelle explicitement la méthode d'origine liée à
    ``orchestrator`` pour préserver le comportement réel.
    """
    original = Orchestrator.execute_agent

    def _side_effect(request, state, *, model_id):
        return original(orchestrator, request, state, model_id=model_id)

    return MagicMock(side_effect=_side_effect)


class TestOrchestratorSpy:
    """Espionne ``Orchestrator.execute_agent`` pour prouver qu'un nœud le
    traverse réellement — jamais un raccourci direct vers l'agent qui
    contournerait préconditions/modèle/outils/audit."""

    def test_node_calls_orchestrator_execute_agent_with_correct_agent_name(self, orchestrator, node_registry):
        spy = _spy_on_execute_agent(orchestrator)
        with patch.object(Orchestrator, "execute_agent", spy):
            expected = _agent_update("security_gate", step="security_gate")
            with patch("agents.security_gate_agent.agent.node", return_value=expected):
                node_registry["security_gate"](_state())

        spy.assert_called_once()
        request = spy.call_args.args[0]
        assert request.agent_name is AgentName.SECURITY_GATE
        assert spy.call_args.kwargs["model_id"]

    def test_agent_never_called_before_orchestrator_authorizes_it(self, orchestrator, node_registry):
        """L'agent réel n'est atteint qu'après que le spy — substitué à
        ``execute_agent`` — a lui-même été invoqué : preuve qu'aucun chemin
        de code n'atteint l'agent sans passer par l'orchestrateur."""
        call_order: list[str] = []
        original = Orchestrator.execute_agent

        def _tracking_side_effect(request, state, *, model_id):
            call_order.append("orchestrator")
            return original(orchestrator, request, state, model_id=model_id)

        spy = MagicMock(side_effect=_tracking_side_effect)

        def _tracked_agent(state):
            call_order.append("agent")
            return _agent_update("security_gate", step="security_gate")

        with patch.object(Orchestrator, "execute_agent", spy):
            with patch("agents.security_gate_agent.agent.node", side_effect=_tracked_agent):
                node_registry["security_gate"](_state())

        assert call_order == ["orchestrator", "agent"]

    def test_spy_records_one_call_per_node_invocation(self, orchestrator, node_registry):
        spy = _spy_on_execute_agent(orchestrator)
        with patch.object(Orchestrator, "execute_agent", spy):
            expected = _agent_update("claim_intake", step="claim_intake")
            with patch("agents.claim_intake_agent.agent.node", return_value=expected):
                node_registry["claim_intake"](_state())
                node_registry["claim_intake"](_state())

        assert spy.call_count == 2

    def test_stub_agent_node_also_traverses_orchestrator(self, orchestrator, node_registry):
        """Les 4 agents stubs (clinical_consistency, fraud_detection,
        case_reviewer, audit) traversent eux aussi l'orchestrateur — même
        mécanisme quel que soit l'agent."""
        spy = _spy_on_execute_agent(orchestrator)
        with patch.object(Orchestrator, "execute_agent", spy):
            node_registry["clinical_consistency"](_state())

        spy.assert_called_once()
        request = spy.call_args.args[0]
        assert request.agent_name is AgentName.CLINICAL_CONSISTENCY


# ── 9. Intégration bout en bout — nœud de graphe via l'orchestrateur ────────
#
# Un faux modèle (jamais un ChatOllama réel), un faux agent (fonction
# déterministe) et les outils réellement autorisés (déterministes,
# `rechercher_code`) sont injectés dans un ``Orchestrator`` construit à la
# main, puis câblés dans un nœud de graphe via ``build_node_registry`` —
# exactement le chemin emprunté en production par ``graph/workflow.py``.


class _FakeModelClient:
    """Faux client de modèle déterministe — jamais instancié dans ces tests
    (l'orchestrateur ne fait que vérifier l'autorisation du modèle, il
    n'appelle jamais lui-même ``client_factory``) : sa présence prouve
    seulement qu'aucun ``ChatOllama`` réel n'est nécessaire."""


def _fake_model_registry(*, enabled: bool = True) -> ModelRegistry:
    """Registre avec un unique faux modèle, enregistré sous l'identifiant
    réellement configuré (``Settings.claimshield_llm_model``) — le nœud le
    résout à l'appel via ``get_settings()`` ; l'enregistrer sous ce même
    identifiant garantit qu'il est utilisé directement, sans passer par le
    mécanisme de fallback modèle (hors périmètre de ce test d'intégration)."""
    from config.settings import get_settings

    registry = ModelRegistry()
    registry.register(
        ModelSpec(
            model_id=get_settings().claimshield_llm_model,
            provider="fake",
            capabilities=frozenset({ModelCapability.STRUCTURED_OUTPUT, ModelCapability.TOOL_CALLING}),
            client_factory=_FakeModelClient,
            enabled=enabled,
        )
    )
    return registry


def _fake_medical_coding_agent(calls: list[str]):
    """Faux agent déterministe — jamais un appel LLM, jamais un import
    d'``agents.medical_coding_agent``."""

    def _runner(state: dict) -> dict:
        calls.append("agent_called")
        return {
            "coding_result": MedicalCodingResult(
                case_id=state["case_id"],
                status=VerificationStatus.PASS,
                reasons=["faux agent déterministe : correspondance simulée"],
            ),
            "coding_input": None,
            "current_step": "medical_coding",
            "completed_steps": ["medical_coding"],
        }

    return _runner


class TestGraphNodeIntegrationThroughOrchestrator:
    """Test d'intégration exigé : exécute un nœud de graphe réel
    (``build_node_registry``) via un ``Orchestrator`` entièrement injecté —
    faux modèle, faux agent, outils déterministes réellement autorisés."""

    def test_node_execution_exercises_policies_invocation_validation_state_and_audit(self):
        calls: list[str] = []
        orchestrator = Orchestrator(
            model_registry=_fake_model_registry(),
            agent_registry={AgentName.MEDICAL_CODING: _fake_medical_coding_agent(calls)},
            preconditions_check=_graph_preconditions_check,
            tools_check=build_authorized_tools,  # outils réellement autorisés, déterministes
            retry_policy=RetryPolicy(max_attempts=1),
        )
        node_fn = build_node_registry(orchestrator)["medical_coding"]
        state = {
            "case_id": "CLM-0001",
            "current_step": "identity_coverage",
            "coding_input": {"acts": ["consultation"]},
        }

        updates = node_fn(state)

        # Invocation : le faux agent a bien été appelé — une seule fois.
        assert calls == ["agent_called"]

        # Validation : le résultat est une véritable instance Pydantic
        # revalidée par l'orchestrateur, jamais le dict brut de l'agent.
        assert isinstance(updates["coding_result"], MedicalCodingResult)
        assert updates["coding_result"].status is VerificationStatus.PASS

        # Mise à jour du state : même forme qu'à l'étape 10 — bookkeeping
        # complet préservé, entrée consommée.
        assert updates["current_step"] == "medical_coding"
        assert updates["completed_steps"] == ["medical_coding"]
        assert updates["coding_input"] is None

        # Audit : les trois politiques (préconditions, modèle, outils) et
        # l'appel de l'agent sont tous tracés, dans l'ordre.
        actions = [event.action for event in updates["audit_trail"]]
        assert actions == ["authorization", "authorization", "authorization", "call", "result"]
        for event in updates["audit_trail"]:
            assert event.case_id == "CLM-0001"
            assert event.actor == "medical_coding"
        assert updates["audit_trail"][-1].outcome == "SUCCESS"

    def test_refusal_never_executes_the_targeted_agent(self):
        """Modèle interdit (désactivé) : le nœud reste bloqué avant tout
        appel — le faux agent ciblé n'est jamais exécuté, l'outil autorisé
        n'est jamais résolu pour lui, et le refus reste visible et attribué."""
        calls: list[str] = []
        orchestrator = Orchestrator(
            model_registry=_fake_model_registry(enabled=False),
            agent_registry={AgentName.MEDICAL_CODING: _fake_medical_coding_agent(calls)},
            preconditions_check=_graph_preconditions_check,
            tools_check=build_authorized_tools,
            retry_policy=RetryPolicy(max_attempts=1),
        )
        node_fn = build_node_registry(orchestrator)["medical_coding"]
        state = {
            "case_id": "CLM-0001",
            "current_step": "identity_coverage",
            "coding_input": {"acts": ["consultation"]},
        }

        updates = node_fn(state)

        # Aucune exécution réelle de l'agent ciblé.
        assert calls == []
        assert "coding_result" not in updates

        # Le refus reste visible et correctement attribué.
        assert updates["errors"]
        assert "medical_coding" in updates["errors"][0]
        assert "MODEL_DISABLED" in updates["errors"][0]
        assert updates["current_step"] == "medical_coding"
        assert updates["coding_input"] is None

        refusal_events = [e for e in updates["audit_trail"] if e.action == "refusal"]
        assert refusal_events
        assert refusal_events[-1].details["policy"] == "MODEL_DISABLED"
        assert refusal_events[-1].actor == "medical_coding"


# ── 10. Non-régression — checkpoints et reprise (étape 10) ───────────────────
#
# L'intégration de l'orchestrateur (ce module) ne doit jamais casser la
# persistance de state ni la reprise après interruption HITL — ces
# mécanismes vivent dans graph/checkpoints.py et graph/workflow.py, testés
# en détail dans tests/graph/test_checkpoints.py et
# tests/graph/test_workflow_interrupt_resume.py. Ce test minimal confirme,
# depuis test_nodes.py, qu'un nœud reconstruit via build_node_registry reste
# entièrement compatible avec un state sérialisable par un vrai checkpointer.


class TestCheckpointCompatibility:
    def test_node_output_is_checkpoint_serializable(self):
        """La mise à jour produite par un nœud (résultat validé + audit_trail)
        doit rester sérialisable par un vrai checkpointer LangGraph — condition
        nécessaire pour que les tests de checkpoint/reprise de l'étape 10
        (``test_checkpoints.py``, ``test_workflow_interrupt_resume.py``)
        continuent de fonctionner sans changement."""
        from graph.checkpoints import CheckpointSession, get_checkpointer

        calls: list[str] = []
        orchestrator = Orchestrator(
            model_registry=_fake_model_registry(),
            agent_registry={AgentName.MEDICAL_CODING: _fake_medical_coding_agent(calls)},
            preconditions_check=_graph_preconditions_check,
            tools_check=build_authorized_tools,
            retry_policy=RetryPolicy(max_attempts=1),
        )
        node_fn = build_node_registry(orchestrator)["medical_coding"]
        state = {"case_id": "CLM-0001", "current_step": "identity_coverage"}

        updates = node_fn(state)

        checkpointer = get_checkpointer(backend="memory")
        session = CheckpointSession("CLM-0001")
        full_state = {
            "case_id": "CLM-0001",
            "schema_version": "1.0.0",
            "completed_steps": [],
            "errors": [],
            "alerts": [],
            "final_justification": [],
            **updates,
        }

        session.save(checkpointer, full_state, step=1)
        restored = session.load(checkpointer)

        assert restored is not None
        assert restored["coding_result"].status is VerificationStatus.PASS
        assert restored["current_step"] == "medical_coding"
