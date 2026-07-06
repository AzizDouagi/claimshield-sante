"""Tests des interfaces injectables — agents ClaimShield Santé.

Couvre :
  - Conformité Protocol (runtime_checkable) — agents injectables et audit stub.
  - Stub audit : valeur NOT_EVALUATED, jamais de résultat métier inventé.
  - ``make_node`` : délégation, mise à jour state, erreurs/alertes.
  - Injection déterministe : implémentations de test pilotables pour les 4 agents.
  - ``make_node_<name>`` dans graph/nodes.py : isolation d'exceptions.
  - NODE_REGISTRY : 11 entrées, noms stables.

Les tests exerçant le comportement RÉEL par défaut de clinical_consistency_agent
et fraud_detection_agent (Phase A déterministe + Phase B LLM) vivent dans
``tests/agents/test_clinical_consistency_agent.py`` et
``tests/agents/test_fraud_detection_agent.py`` — ce module ne couvre plus que
le mécanisme d'injection générique, identique pour les 4 agents.
"""
from __future__ import annotations


import pytest

from agents.audit_agent.agent import (
    AuditAgentRunnable,
    _NotImplementedStub as AuditStub,
    make_node as make_audit_node,
)
from agents.case_reviewer_agent.agent import (
    CaseReviewerRunnable,
    _DEFAULT_IMPL as DefaultCaseReviewerImpl,
    make_node as make_case_reviewer_node,
)
from agents.clinical_consistency_agent.agent import (
    ClinicalConsistencyRunnable,
    make_node as make_clinical_node,
)
from agents.fraud_detection_agent.agent import (
    FraudDetectionRunnable,
    make_node as make_fraud_node,
)
from graph.nodes import build_node_registry, build_orchestrator
from schemas.domain import Recommendation, VerificationStatus
from schemas.results import (
    AuditResult,
    CaseReviewerResult,
    CaseReviewerResultPayload,
    ClinicalConsistencyResult,
    ClinicalResultPayload,
    FraudDetectionResult,
    FraudResultPayload,
    LlmMetadata,
)

_LLM_TRACE = LlmMetadata(model_name="test-llm", prompt_version="test")


# ── Helpers ────────────────────────────────────────────────────────────────────


def _state(case_id: str = "CLM-0001") -> dict:
    return {"case_id": case_id}


# ── Implémentations déterministes injectables pour les tests ──────────────────


class _ClinicalPass:
    def run(self, state):
        return ClinicalConsistencyResult(
            case_id=str(state.get("case_id", "CLM-0000")),
            status=VerificationStatus.PASS,
            llm_trace=_LLM_TRACE,
            confidence=0.95,
            result_payload=ClinicalResultPayload(reasons=["test: pass déterministe"]),
        )


class _ClinicalFail:
    def run(self, state):
        return ClinicalConsistencyResult(
            case_id=str(state.get("case_id", "CLM-0000")),
            status=VerificationStatus.FAIL,
            llm_trace=_LLM_TRACE,
            result_payload=ClinicalResultPayload(reasons=["test: incohérence de date"]),
        )


class _ClinicalNeedsReview:
    def run(self, state):
        return ClinicalConsistencyResult(
            case_id=str(state.get("case_id", "CLM-0000")),
            status=VerificationStatus.NEEDS_REVIEW,
            llm_trace=_LLM_TRACE,
            result_payload=ClinicalResultPayload(reasons=["test: ordonnance ambiguë"]),
        )


class _FraudPass:
    def run(self, state):
        return FraudDetectionResult(
            case_id=str(state.get("case_id", "CLM-0000")),
            status=VerificationStatus.PASS,
            llm_trace=_LLM_TRACE,
            result_payload=FraudResultPayload(risk_score=0.05, reasons=["test: aucun signal"]),
        )


class _FraudFail:
    def run(self, state):
        return FraudDetectionResult(
            case_id=str(state.get("case_id", "CLM-0000")),
            status=VerificationStatus.FAIL,
            llm_trace=_LLM_TRACE,
            result_payload=FraudResultPayload(
                risk_score=0.92,
                duplicate_invoice=True,
                reasons=["test: doublon détecté"],
            ),
        )


class _ReviewApprove:
    def run(self, state):
        return CaseReviewerResult(
            case_id=str(state.get("case_id", "CLM-0000")),
            llm_trace=LlmMetadata(model_name="test-llm", prompt_version="test"),
            result_payload=CaseReviewerResultPayload(
                recommendation=Recommendation.APPROVE,
                justification=["test: tous les contrôles passés"],
                human_review_reasons=["test: validation humaine requise"],
            ),
        )


class _ReviewReject:
    def run(self, state):
        return CaseReviewerResult(
            case_id=str(state.get("case_id", "CLM-0000")),
            llm_trace=LlmMetadata(model_name="test-llm", prompt_version="test"),
            result_payload=CaseReviewerResultPayload(
                recommendation=Recommendation.REJECT,
                justification=["test: doublon confirmé"],
                human_review_reasons=["test: validation humaine requise"],
            ),
        )


class _AuditPass:
    def run(self, state):
        return AuditResult(
            case_id=str(state.get("case_id", "CLM-0000")),
            status=VerificationStatus.PASS,
            events_count=3,
        )


# ── 1. Conformité Protocol ─────────────────────────────────────────────────────


class TestProtocolConformance:
    """runtime_checkable : isinstance() fonctionne sans héritage."""

    def test_case_reviewer_default_impl_is_runnable(self):
        assert isinstance(DefaultCaseReviewerImpl, CaseReviewerRunnable)

    def test_audit_stub_is_runnable(self):
        assert isinstance(AuditStub(), AuditAgentRunnable)

    def test_deterministic_impl_satisfies_clinical_protocol(self):
        assert isinstance(_ClinicalPass(), ClinicalConsistencyRunnable)

    def test_deterministic_impl_satisfies_fraud_protocol(self):
        assert isinstance(_FraudPass(), FraudDetectionRunnable)

    def test_deterministic_impl_satisfies_case_reviewer_protocol(self):
        assert isinstance(_ReviewApprove(), CaseReviewerRunnable)

    def test_deterministic_impl_satisfies_audit_protocol(self):
        assert isinstance(_AuditPass(), AuditAgentRunnable)


# ── 2. Stub audit par défaut — valeur NON métier ──────────────────────────────


class TestStubDefaults:
    """Le stub audit ne retourne jamais PASS."""

    def test_audit_stub_returns_not_evaluated(self):
        result = AuditStub().run(_state())
        assert result.status is VerificationStatus.NOT_EVALUATED

    def test_audit_stub_events_empty_in_result(self):
        result = AuditStub().run(_state())
        assert result.events == []

    def test_audit_stub_counts_existing_trail(self):
        from schemas.results import AuditEvent
        events = [
            AuditEvent(event_id="1", case_id="CLM-0001", actor="a", action="x", outcome="ok"),
            AuditEvent(event_id="2", case_id="CLM-0001", actor="b", action="y", outcome="ok"),
        ]
        state = {"case_id": "CLM-0001", "audit_trail": events}
        result = AuditStub().run(state)
        assert result.events_count == 2

    def test_stub_reason_contains_stub_marker(self):
        # AuditResult n'a pas de champ reasons/justification par conception.
        result = AuditStub().run(_state())
        assert result.status is VerificationStatus.NOT_EVALUATED

    def test_audit_stub_not_evaluated_is_marker(self):
        result = AuditStub().run(_state())
        assert result.status is VerificationStatus.NOT_EVALUATED

    def test_stub_case_id_comes_from_state(self):
        for stub_cls in (AuditStub,):
            result = stub_cls().run({"case_id": "CLM-9999"})
            assert result.case_id == "CLM-9999"


# ── 3. make_node — nœuds avec implémentation par défaut ───────────────────────


class TestMakeNodeDefault:
    def test_clinical_node_sets_clinical_result(self):
        node_fn = make_clinical_node()
        updates = node_fn(_state())
        assert "clinical_result" in updates

    def test_clinical_node_default_needs_review_without_upstream_data(self):
        """Étape 12 : l'implémentation par défaut est réelle, plus un stub.
        Sans ocr_result ni coding_result, elle signale NEEDS_REVIEW plutôt
        que d'inventer un PASS ou de continuer à retourner NOT_EVALUATED."""
        node_fn = make_clinical_node()
        updates = node_fn(_state())
        assert updates["clinical_result"].status is VerificationStatus.NEEDS_REVIEW

    def test_fraud_node_sets_fraud_result(self):
        node_fn = make_fraud_node()
        updates = node_fn(_state())
        assert "fraud_result" in updates

    def test_fraud_node_default_needs_review_without_upstream_data(self):
        """Étape 12 : idem pour fraud_detection — preuves insuffisantes,
        jamais un PASS ou un NOT_EVALUATED silencieux."""
        node_fn = make_fraud_node()
        updates = node_fn(_state())
        assert updates["fraud_result"].status is VerificationStatus.NEEDS_REVIEW

    def test_case_reviewer_node_sets_review_result(self):
        node_fn = make_case_reviewer_node()
        updates = node_fn(_state())
        assert "review_result" in updates

    def test_case_reviewer_node_recommendation_pending(self):
        node_fn = make_case_reviewer_node()
        updates = node_fn(_state())
        assert updates["review_result"].result_payload.recommendation is Recommendation.PENDING

    def test_case_reviewer_node_sets_final_recommendation(self):
        node_fn = make_case_reviewer_node()
        updates = node_fn(_state())
        assert updates["final_recommendation"] is Recommendation.PENDING

    def test_case_reviewer_node_requires_human_review(self):
        node_fn = make_case_reviewer_node()
        updates = node_fn(_state())
        assert updates["review_result"].human_review_required is True

    def test_audit_node_sets_audit_result(self):
        node_fn = make_audit_node()
        updates = node_fn(_state())
        assert "audit_result" in updates

    def test_audit_node_status_not_evaluated(self):
        node_fn = make_audit_node()
        updates = node_fn(_state())
        assert updates["audit_result"].status is VerificationStatus.NOT_EVALUATED


# ── 4. make_node — injection d'implémentations déterministes ──────────────────


class TestMakeNodeInjection:
    def test_clinical_injected_pass_result(self):
        node_fn = make_clinical_node(_ClinicalPass())
        updates = node_fn(_state())
        assert updates["clinical_result"].status is VerificationStatus.PASS

    def test_clinical_injected_fail_adds_errors(self):
        node_fn = make_clinical_node(_ClinicalFail())
        updates = node_fn(_state())
        assert updates.get("errors")
        assert "clinical_consistency_agent" in updates["errors"][0]

    def test_clinical_injected_needs_review_adds_alerts(self):
        node_fn = make_clinical_node(_ClinicalNeedsReview())
        updates = node_fn(_state())
        assert updates.get("alerts")

    def test_fraud_injected_pass_result(self):
        node_fn = make_fraud_node(_FraudPass())
        updates = node_fn(_state())
        assert updates["fraud_result"].status is VerificationStatus.PASS

    def test_fraud_injected_fail_adds_errors(self):
        node_fn = make_fraud_node(_FraudFail())
        updates = node_fn(_state())
        assert updates.get("errors")

    def test_case_reviewer_injected_approve(self):
        node_fn = make_case_reviewer_node(_ReviewApprove())
        updates = node_fn(_state())
        assert updates["review_result"].result_payload.recommendation is Recommendation.APPROVE
        assert updates["final_recommendation"] is Recommendation.APPROVE
        assert updates["review_result"].human_review_required is True

    def test_case_reviewer_injected_reject_adds_errors(self):
        node_fn = make_case_reviewer_node(_ReviewReject())
        updates = node_fn(_state())
        assert updates.get("errors")

    def test_audit_injected_pass_result(self):
        node_fn = make_audit_node(_AuditPass())
        updates = node_fn(_state())
        assert updates["audit_result"].status is VerificationStatus.PASS

    def test_case_reviewer_approve_forces_human_alert(self):
        node_fn = make_case_reviewer_node(_ReviewApprove())
        updates = node_fn(_state())
        assert updates["review_result"].human_review_required is True
        assert updates.get("alerts")

    def test_completed_steps_set_by_injected_node(self):
        for node_fn, step in [
            (make_clinical_node(_ClinicalPass()), "clinical_consistency"),
            (make_fraud_node(_FraudPass()), "fraud_detection"),
            (make_case_reviewer_node(_ReviewApprove()), "case_reviewer"),
            (make_audit_node(_AuditPass()), "audit"),
        ]:
            updates = node_fn(_state())
            assert step in updates["completed_steps"], f"{step} absent de completed_steps"


# ── 5. graph/nodes.py — nœuds par défaut (via l'orchestrateur) ──────────────


class TestGraphNodeStubDefaults:
    def test_node_clinical_consistency_default_needs_review(self):
        """Étape 12 : évaluation réelle par défaut — sans ocr_result ni
        coding_result, NEEDS_REVIEW (preuves insuffisantes), plus NOT_EVALUATED."""
        node_fn = build_node_registry(build_orchestrator())["clinical_consistency"]
        updates = node_fn(_state())
        assert updates["clinical_result"].status is VerificationStatus.NEEDS_REVIEW

    def test_node_fraud_detection_default_needs_review(self):
        node_fn = build_node_registry(build_orchestrator())["fraud_detection"]
        updates = node_fn(_state())
        assert updates["fraud_result"].status is VerificationStatus.NEEDS_REVIEW

    def test_node_case_reviewer_default_requires_human_review(self):
        node_fn = build_node_registry(build_orchestrator())["case_reviewer"]
        updates = node_fn(_state())
        assert updates["review_result"].human_review_required is True
        assert updates["review_result"].llm_trace is not None

    def test_node_audit_not_evaluated(self):
        node_fn = build_node_registry(build_orchestrator())["audit"]
        updates = node_fn(_state())
        assert updates["audit_result"].status is VerificationStatus.NOT_EVALUATED


# ── 6. graph/nodes.py — injection d'implémentation via build_orchestrator ───


class TestGraphNodeFactories:
    def test_clinical_with_impl_returns_pass(self):
        orchestrator = build_orchestrator(clinical_consistency_impl=_ClinicalPass())
        node_fn = build_node_registry(orchestrator)["clinical_consistency"]
        updates = node_fn(_state())
        assert updates["clinical_result"].status is VerificationStatus.PASS

    def test_fraud_with_impl_returns_pass(self):
        orchestrator = build_orchestrator(fraud_detection_impl=_FraudPass())
        node_fn = build_node_registry(orchestrator)["fraud_detection"]
        updates = node_fn(_state())
        assert updates["fraud_result"].status is VerificationStatus.PASS

    def test_case_reviewer_with_impl_returns_approve(self):
        orchestrator = build_orchestrator(case_reviewer_impl=_ReviewApprove())
        node_fn = build_node_registry(orchestrator)["case_reviewer"]
        updates = node_fn(_state())
        assert updates["review_result"].result_payload.recommendation is Recommendation.APPROVE

    def test_audit_with_impl_returns_pass(self):
        orchestrator = build_orchestrator(audit_impl=_AuditPass())
        node_fn = build_node_registry(orchestrator)["audit"]
        updates = node_fn(_state())
        assert updates["audit_result"].status is VerificationStatus.PASS

    def test_none_impl_uses_real_default_implementation(self):
        """``None`` (étape 12) retombe sur l'évaluation réelle par défaut,
        sans repasser par un stub NOT_EVALUATED."""
        orchestrator = build_orchestrator(clinical_consistency_impl=None)
        node_fn = build_node_registry(orchestrator)["clinical_consistency"]
        updates = node_fn(_state())
        assert updates["clinical_result"].status is VerificationStatus.NEEDS_REVIEW

    def test_exception_isolated(self):
        class Crasher:
            def run(self, state):
                raise RuntimeError("boom")

        orchestrator = build_orchestrator(clinical_consistency_impl=Crasher())
        node_fn = build_node_registry(orchestrator)["clinical_consistency"]
        updates = node_fn(_state())
        assert updates.get("errors")
        assert "clinical_consistency" in updates["errors"][0]

    def test_fraud_exception_isolated(self):
        class Crasher:
            def run(self, state):
                raise ValueError("bad input")

        orchestrator = build_orchestrator(fraud_detection_impl=Crasher())
        node_fn = build_node_registry(orchestrator)["fraud_detection"]
        updates = node_fn(_state())
        assert updates.get("errors")

    def test_case_reviewer_exception_isolated(self):
        class Crasher:
            def run(self, state):
                raise OSError("network down")

        orchestrator = build_orchestrator(case_reviewer_impl=Crasher())
        node_fn = build_node_registry(orchestrator)["case_reviewer"]
        updates = node_fn(_state())
        assert updates.get("errors")


# ── 7. build_node_registry — complétude et stabilité des noms ───────────────


class TestNodeRegistryCompleteness:
    ALL_EXPECTED = {
        "claim_intake", "security_gate", "privacy",
        "fhir_validator", "medical_coding", "document_ocr",
        "identity_coverage",
        "clinical_consistency", "fraud_detection", "case_reviewer", "audit",
    }

    def test_registry_has_eleven_entries(self):
        registry = build_node_registry(build_orchestrator())
        assert len(registry) == 11

    def test_registry_contains_all_expected_keys(self):
        registry = build_node_registry(build_orchestrator())
        assert set(registry.keys()) == self.ALL_EXPECTED

    @pytest.mark.parametrize("key", list(ALL_EXPECTED))
    def test_all_entries_are_callable(self, key):
        registry = build_node_registry(build_orchestrator())
        assert callable(registry[key])

    def test_stub_nodes_names_stable(self):
        registry = build_node_registry(build_orchestrator())
        assert registry["clinical_consistency"].__name__ == "node_clinical_consistency"
        assert registry["fraud_detection"].__name__ == "node_fraud_detection"
        assert registry["case_reviewer"].__name__ == "node_case_reviewer"
        assert registry["audit"].__name__ == "node_audit"

    @pytest.mark.parametrize("key", list(ALL_EXPECTED))
    def test_each_node_returns_dict_on_empty_state(self, key):
        registry = build_node_registry(build_orchestrator())
        result = registry[key]({})
        assert isinstance(result, dict)
