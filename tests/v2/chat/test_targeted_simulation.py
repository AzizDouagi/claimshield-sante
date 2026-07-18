"""Tests de la simulation ciblée (Phase 9, plan de remédiation « autonomie
décisionnelle V2 », §7) — `chat.schemas.SimulationPatch`/`SimulationPatchField`
et `chat.simulation_engine.run_targeted_simulation`.

Utilise un graphe stub minimal (`_StubGraph`) plutôt qu'un vrai graphe
compilé pour les tests unitaires ciblés — `agents.autonomous_decision_agent.agent.run`
est monkeypatché, jamais un vrai appel LLM. Un test d'intégration bout en
bout (vrai graphe compilé, vrai dossier soumis) complète la couverture, même
patron que `tests/v2/chat/test_simulation_engine.py`."""
from __future__ import annotations

from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from chat.schemas import SimulationChangeRequest, SimulationPatch, SimulationPatchField
from chat.simulation_engine import run_simulation, run_targeted_simulation
from schemas.domain import ClaimDecisionV2, ReaderRole, VerificationStatus
from schemas.results import CoverageResult, IdentityResult, LlmMetadata
from schemas.v2_results import AutonomousDecisionResult, EligibilityResult


def _llm_trace() -> LlmMetadata:
    return LlmMetadata(model_name="gemma4:latest", prompt_version="1.0.0")


def _eligibility_result(
    *,
    identity_status: VerificationStatus = VerificationStatus.NEEDS_REVIEW,
    coverage_status: VerificationStatus = VerificationStatus.NEEDS_REVIEW,
    ceiling_exceeded: bool | None = None,
    preauthorization_required: bool | None = None,
) -> EligibilityResult:
    return EligibilityResult(
        case_id="CLM-7501",
        status=VerificationStatus.NEEDS_REVIEW,
        identity=IdentityResult(status=identity_status),
        coverage=CoverageResult(
            status=coverage_status,
            ceiling_exceeded=ceiling_exceeded,
            preauthorization_required=preauthorization_required,
        ),
        llm_trace=_llm_trace(),
    )


def _decision_result(decision: ClaimDecisionV2, *, justification: list[str] | None = None) -> AutonomousDecisionResult:
    return AutonomousDecisionResult(
        case_id="CLM-7501",
        status=VerificationStatus.PASS,
        decision=decision,
        justification=justification or [],
        llm_trace=_llm_trace(),
    )


class _StubSnapshot:
    def __init__(self, values: dict | None) -> None:
        self.values = values or {}


class _StubGraph:
    def __init__(self, values: dict | None) -> None:
        self._values = values

    def get_state(self, config):
        return _StubSnapshot(self._values)


# ── Schéma SimulationPatch / SimulationChangeRequest ──────────────────────────


class TestSimulationPatchSchema:
    def test_status_field_accepts_valid_verification_status_values(self):
        for value in ("PASS", "NEEDS_REVIEW", "FAIL"):
            patch = SimulationPatch(field=SimulationPatchField.IDENTITY_STATUS, value=value)
            assert patch.value == value

    def test_status_field_rejects_invalid_value(self):
        with pytest.raises(ValidationError):
            SimulationPatch(field=SimulationPatchField.COVERAGE_STATUS, value="MAYBE")

    def test_bool_field_accepts_true_false(self):
        patch = SimulationPatch(field=SimulationPatchField.CEILING_EXCEEDED, value="true")
        assert patch.value == "true"

    def test_bool_field_rejects_non_boolean_value(self):
        with pytest.raises(ValidationError):
            SimulationPatch(field=SimulationPatchField.CEILING_EXCEEDED, value="PASS")

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            SimulationPatch(field="INVENTED_FIELD", value="PASS")


class TestSimulationChangeRequestExclusivity:
    def test_field_patches_alone_is_valid(self):
        request = SimulationChangeRequest(
            field_patches=[SimulationPatch(field=SimulationPatchField.COVERAGE_STATUS, value="PASS")]
        )
        assert len(request.field_patches) == 1

    def test_field_patches_with_remove_document_rejected(self):
        with pytest.raises(ValidationError):
            SimulationChangeRequest(
                remove_document="ordonnance",
                field_patches=[SimulationPatch(field=SimulationPatchField.COVERAGE_STATUS, value="PASS")],
            )

    def test_field_patches_with_reader_role_rejected(self):
        with pytest.raises(ValidationError):
            SimulationChangeRequest(
                reader_role=ReaderRole.FRAUD_ANALYST,
                field_patches=[SimulationPatch(field=SimulationPatchField.COVERAGE_STATUS, value="PASS")],
            )

    def test_remove_document_and_reader_role_alone_still_work(self):
        """Non-régression V2-11b — les deux mécanismes existants restent
        utilisables indépendamment de `field_patches`."""
        request = SimulationChangeRequest(remove_document="ordonnance", reader_role=ReaderRole.AUDITOR)
        assert request.field_patches == []

    def test_max_five_patches(self):
        with pytest.raises(ValidationError):
            SimulationChangeRequest(
                field_patches=[
                    SimulationPatch(field=SimulationPatchField.COVERAGE_STATUS, value="PASS") for _ in range(6)
                ]
            )


# ── run_targeted_simulation ────────────────────────────────────────────────────


class TestRunTargetedSimulation:
    def test_unknown_case_returns_not_applied(self):
        graph = _StubGraph(None)
        result = run_targeted_simulation(
            "CLM-7099",
            [SimulationPatch(field=SimulationPatchField.COVERAGE_STATUS, value="PASS")],
            compiled_graph=graph,
        )
        assert result.applied is False
        assert result.error is not None

    def test_missing_eligibility_result_returns_not_applied(self):
        graph = _StubGraph({"final_decision": ClaimDecisionV2.REJECT})
        result = run_targeted_simulation(
            "CLM-7501",
            [SimulationPatch(field=SimulationPatchField.COVERAGE_STATUS, value="PASS")],
            compiled_graph=graph,
        )
        assert result.applied is False
        assert "éligibilité" in result.error.lower()

    def test_identity_status_patch_reaches_autonomous_decision_agent(self, monkeypatch):
        eligibility = _eligibility_result(identity_status=VerificationStatus.FAIL)
        graph = _StubGraph({"final_decision": ClaimDecisionV2.REJECT, "eligibility_result": eligibility})

        run_spy = Mock(return_value=_decision_result(ClaimDecisionV2.APPROVE))
        monkeypatch.setattr("chat.simulation_engine.run_autonomous_decision", run_spy)

        result = run_targeted_simulation(
            "CLM-7501",
            [SimulationPatch(field=SimulationPatchField.IDENTITY_STATUS, value="PASS")],
            compiled_graph=graph,
        )

        assert result.applied is True
        assert result.original_decision == "REJECT"
        assert result.simulated_decision == "APPROVE"
        assert result.decision_changed is True
        run_spy.assert_called_once()
        patched_state = run_spy.call_args[0][1]
        assert patched_state["eligibility_result"].identity.status is VerificationStatus.PASS
        # Le reste de l'objet reste inchangé (seul `identity.status` patché).
        assert patched_state["eligibility_result"].coverage.status is eligibility.coverage.status

    def test_real_eligibility_result_never_mutated(self, monkeypatch):
        eligibility = _eligibility_result(coverage_status=VerificationStatus.FAIL)
        graph = _StubGraph({"final_decision": ClaimDecisionV2.REJECT, "eligibility_result": eligibility})
        monkeypatch.setattr(
            "chat.simulation_engine.run_autonomous_decision",
            Mock(return_value=_decision_result(ClaimDecisionV2.APPROVE)),
        )

        run_targeted_simulation(
            "CLM-7501",
            [SimulationPatch(field=SimulationPatchField.COVERAGE_STATUS, value="PASS")],
            compiled_graph=graph,
        )

        assert eligibility.coverage.status is VerificationStatus.FAIL

    def test_ceiling_exceeded_bool_patch_applied(self, monkeypatch):
        eligibility = _eligibility_result(ceiling_exceeded=False)
        graph = _StubGraph({"final_decision": ClaimDecisionV2.REJECT, "eligibility_result": eligibility})
        run_spy = Mock(return_value=_decision_result(ClaimDecisionV2.PARTIAL_APPROVE))
        monkeypatch.setattr("chat.simulation_engine.run_autonomous_decision", run_spy)

        result = run_targeted_simulation(
            "CLM-7501",
            [SimulationPatch(field=SimulationPatchField.CEILING_EXCEEDED, value="true")],
            compiled_graph=graph,
        )

        assert result.applied is True
        patched_state = run_spy.call_args[0][1]
        assert patched_state["eligibility_result"].coverage.ceiling_exceeded is True

    def test_multiple_patches_all_applied(self, monkeypatch):
        eligibility = _eligibility_result()
        graph = _StubGraph({"final_decision": ClaimDecisionV2.REJECT, "eligibility_result": eligibility})
        run_spy = Mock(return_value=_decision_result(ClaimDecisionV2.APPROVE))
        monkeypatch.setattr("chat.simulation_engine.run_autonomous_decision", run_spy)

        run_targeted_simulation(
            "CLM-7501",
            [
                SimulationPatch(field=SimulationPatchField.IDENTITY_STATUS, value="PASS"),
                SimulationPatch(field=SimulationPatchField.COVERAGE_STATUS, value="PASS"),
                SimulationPatch(field=SimulationPatchField.PREAUTHORIZATION_REQUIRED, value="false"),
            ],
            compiled_graph=graph,
        )

        patched = run_spy.call_args[0][1]["eligibility_result"]
        assert patched.identity.status is VerificationStatus.PASS
        assert patched.coverage.status is VerificationStatus.PASS
        assert patched.coverage.preauthorization_required is False

    def test_agent_exception_returns_not_applied_never_raises(self, monkeypatch):
        eligibility = _eligibility_result()
        graph = _StubGraph({"final_decision": ClaimDecisionV2.REJECT, "eligibility_result": eligibility})
        monkeypatch.setattr(
            "chat.simulation_engine.run_autonomous_decision", Mock(side_effect=RuntimeError("panne"))
        )

        result = run_targeted_simulation(
            "CLM-7501",
            [SimulationPatch(field=SimulationPatchField.COVERAGE_STATUS, value="PASS")],
            compiled_graph=graph,
        )

        assert result.applied is False
        assert "panne" in result.error


class TestRunSimulationDispatch:
    """`run_simulation` délègue à `run_targeted_simulation` dès que
    `field_patches` est non vide — jamais le graphe entier réinvoqué dans ce
    cas (vérifié par l'absence d'appel à `graph.invoke`)."""

    def test_dispatches_to_targeted_simulation(self, monkeypatch):
        spy = Mock(return_value=Mock(applied=True))
        monkeypatch.setattr("chat.simulation_engine.run_targeted_simulation", spy)
        changes = SimulationChangeRequest(
            field_patches=[SimulationPatch(field=SimulationPatchField.COVERAGE_STATUS, value="PASS")]
        )
        run_simulation("CLM-7501", changes, compiled_graph=Mock())
        spy.assert_called_once()

    def test_never_dispatches_without_field_patches(self, monkeypatch):
        spy = Mock()
        monkeypatch.setattr("chat.simulation_engine.run_targeted_simulation", spy)
        graph = _StubGraph(None)
        run_simulation("CLM-7099", SimulationChangeRequest(remove_document="ordonnance"), compiled_graph=graph)
        spy.assert_not_called()
