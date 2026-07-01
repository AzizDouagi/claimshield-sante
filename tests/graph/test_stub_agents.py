"""Tests des interfaces injectables — agents stubs ClaimShield Santé.

Couvre :
  - Conformité Protocol (runtime_checkable)
  - Stub : valeurs NOT_EVALUATED / PENDING, jamais de résultat métier inventé
  - ``make_node`` : délégation, mise à jour state, erreurs/alertes
  - Injection déterministe : implémentations de test pilotables
  - ``make_node_<name>`` dans graph/nodes.py : isolation d'exceptions
  - NODE_REGISTRY : 11 entrées, noms stables
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.audit_agent.agent import (
    AuditAgentRunnable,
    _NotImplementedStub as AuditStub,
    make_node as make_audit_node,
)
from agents.case_reviewer_agent.agent import (
    CaseReviewerRunnable,
    _NotImplementedStub as CaseReviewerStub,
    make_node as make_case_reviewer_node,
)
from agents.clinical_consistency_agent.agent import (
    ClinicalConsistencyRunnable,
    _NotImplementedStub as ClinicalStub,
    make_node as make_clinical_node,
)
from agents.fraud_detection_agent.agent import (
    FraudDetectionRunnable,
    _NotImplementedStub as FraudStub,
    make_node as make_fraud_node,
)
from graph.nodes import (
    NODE_REGISTRY,
    make_node_audit,
    make_node_case_reviewer,
    make_node_clinical_consistency,
    make_node_fraud_detection,
    node_audit,
    node_case_reviewer,
    node_clinical_consistency,
    node_fraud_detection,
)
from schemas.domain import Recommendation, VerificationStatus
from schemas.results import (
    AuditResult,
    CaseReviewerResult,
    ClinicalConsistencyResult,
    FraudDetectionResult,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _state(case_id: str = "CLM-0001") -> dict:
    return {"case_id": case_id}


# ── Implémentations déterministes injectables pour les tests ──────────────────


class _ClinicalPass:
    def run(self, state):
        return ClinicalConsistencyResult(
            case_id=str(state.get("case_id", "CLM-0000")),
            status=VerificationStatus.PASS,
            confidence=0.95,
            reasons=["test: pass déterministe"],
        )


class _ClinicalFail:
    def run(self, state):
        return ClinicalConsistencyResult(
            case_id=str(state.get("case_id", "CLM-0000")),
            status=VerificationStatus.FAIL,
            reasons=["test: incohérence de date"],
        )


class _ClinicalNeedsReview:
    def run(self, state):
        return ClinicalConsistencyResult(
            case_id=str(state.get("case_id", "CLM-0000")),
            status=VerificationStatus.NEEDS_REVIEW,
            reasons=["test: ordonnance ambiguë"],
        )


class _FraudPass:
    def run(self, state):
        return FraudDetectionResult(
            case_id=str(state.get("case_id", "CLM-0000")),
            status=VerificationStatus.PASS,
            risk_score=0.05,
            reasons=["test: aucun signal"],
        )


class _FraudFail:
    def run(self, state):
        return FraudDetectionResult(
            case_id=str(state.get("case_id", "CLM-0000")),
            status=VerificationStatus.FAIL,
            risk_score=0.92,
            duplicate_invoice=True,
            reasons=["test: doublon détecté"],
        )


class _ReviewApprove:
    def run(self, state):
        return CaseReviewerResult(
            case_id=str(state.get("case_id", "CLM-0000")),
            recommendation=Recommendation.APPROVE,
            justification=["test: tous les contrôles passés"],
            human_review_required=False,
        )


class _ReviewReject:
    def run(self, state):
        return CaseReviewerResult(
            case_id=str(state.get("case_id", "CLM-0000")),
            recommendation=Recommendation.REJECT,
            justification=["test: doublon confirmé"],
            human_review_required=False,
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

    def test_clinical_stub_is_runnable(self):
        assert isinstance(ClinicalStub(), ClinicalConsistencyRunnable)

    def test_fraud_stub_is_runnable(self):
        assert isinstance(FraudStub(), FraudDetectionRunnable)

    def test_case_reviewer_stub_is_runnable(self):
        assert isinstance(CaseReviewerStub(), CaseReviewerRunnable)

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


# ── 2. Stubs par défaut — valeurs NON métier ──────────────────────────────────


class TestStubDefaults:
    """Le stub ne retourne jamais PASS, APPROVE ou REJECT."""

    def test_clinical_stub_returns_not_evaluated(self):
        result = ClinicalStub().run(_state())
        assert result.status is VerificationStatus.NOT_EVALUATED

    def test_clinical_stub_never_returns_pass(self):
        result = ClinicalStub().run(_state())
        assert result.status is not VerificationStatus.PASS

    def test_clinical_stub_never_returns_fail(self):
        result = ClinicalStub().run(_state())
        assert result.status is not VerificationStatus.FAIL

    def test_fraud_stub_returns_not_evaluated(self):
        result = FraudStub().run(_state())
        assert result.status is VerificationStatus.NOT_EVALUATED

    def test_fraud_stub_risk_score_is_zero(self):
        result = FraudStub().run(_state())
        assert result.risk_score == 0.0

    def test_fraud_stub_no_duplicate_signal(self):
        result = FraudStub().run(_state())
        assert result.duplicate_invoice is None

    def test_case_reviewer_stub_returns_pending(self):
        result = CaseReviewerStub().run(_state())
        assert result.recommendation is Recommendation.PENDING

    def test_case_reviewer_stub_never_approve(self):
        result = CaseReviewerStub().run(_state())
        assert result.recommendation is not Recommendation.APPROVE

    def test_case_reviewer_stub_never_reject(self):
        result = CaseReviewerStub().run(_state())
        assert result.recommendation is not Recommendation.REJECT

    def test_case_reviewer_stub_human_review_required(self):
        result = CaseReviewerStub().run(_state())
        assert result.human_review_required is True

    def test_audit_stub_returns_not_evaluated(self):
        result = AuditStub().run(_state())
        assert result.status is VerificationStatus.NOT_EVALUATED

    def test_audit_stub_events_empty_in_result(self):
        result = AuditStub().run(_state())
        assert result.events == []

    def test_audit_stub_counts_existing_trail(self):
        from schemas.results import AuditEvent
        from datetime import datetime, UTC
        events = [
            AuditEvent(event_id="1", case_id="CLM-0001", actor="a", action="x", outcome="ok"),
            AuditEvent(event_id="2", case_id="CLM-0001", actor="b", action="y", outcome="ok"),
        ]
        state = {"case_id": "CLM-0001", "audit_trail": events}
        result = AuditStub().run(state)
        assert result.events_count == 2

    def test_stub_reason_contains_stub_marker(self):
        # AuditResult n'a pas de champ reasons/justification par conception.
        for stub_cls in (ClinicalStub, FraudStub, CaseReviewerStub):
            instance = stub_cls()
            result = instance.run(_state())
            reasons = getattr(result, "reasons", None) or getattr(result, "justification", [])
            assert any("[stub]" in r for r in reasons), f"{stub_cls.__name__} sans marqueur [stub]"

    def test_audit_stub_not_evaluated_is_marker(self):
        result = AuditStub().run(_state())
        assert result.status is VerificationStatus.NOT_EVALUATED

    def test_stub_case_id_comes_from_state(self):
        for stub_cls in (ClinicalStub, FraudStub, CaseReviewerStub, AuditStub):
            result = stub_cls().run({"case_id": "CLM-9999"})
            assert result.case_id == "CLM-9999"


# ── 3. make_node — nœud avec stub par défaut ──────────────────────────────────


class TestMakeNodeDefault:
    def test_clinical_node_sets_clinical_result(self):
        node_fn = make_clinical_node()
        updates = node_fn(_state())
        assert "clinical_result" in updates

    def test_clinical_node_status_not_evaluated(self):
        node_fn = make_clinical_node()
        updates = node_fn(_state())
        assert updates["clinical_result"].status is VerificationStatus.NOT_EVALUATED

    def test_fraud_node_sets_fraud_result(self):
        node_fn = make_fraud_node()
        updates = node_fn(_state())
        assert "fraud_result" in updates

    def test_fraud_node_status_not_evaluated(self):
        node_fn = make_fraud_node()
        updates = node_fn(_state())
        assert updates["fraud_result"].status is VerificationStatus.NOT_EVALUATED

    def test_case_reviewer_node_sets_review_result(self):
        node_fn = make_case_reviewer_node()
        updates = node_fn(_state())
        assert "review_result" in updates

    def test_case_reviewer_node_recommendation_pending(self):
        node_fn = make_case_reviewer_node()
        updates = node_fn(_state())
        assert updates["review_result"].recommendation is Recommendation.PENDING

    def test_case_reviewer_node_sets_final_recommendation(self):
        node_fn = make_case_reviewer_node()
        updates = node_fn(_state())
        assert updates["final_recommendation"] is Recommendation.PENDING

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
        assert updates["review_result"].recommendation is Recommendation.APPROVE
        assert updates["final_recommendation"] is Recommendation.APPROVE

    def test_case_reviewer_injected_reject_adds_errors(self):
        node_fn = make_case_reviewer_node(_ReviewReject())
        updates = node_fn(_state())
        assert updates.get("errors")

    def test_audit_injected_pass_result(self):
        node_fn = make_audit_node(_AuditPass())
        updates = node_fn(_state())
        assert updates["audit_result"].status is VerificationStatus.PASS

    def test_case_reviewer_approve_no_human_no_alert(self):
        node_fn = make_case_reviewer_node(_ReviewApprove())
        updates = node_fn(_state())
        assert not updates.get("alerts")

    def test_completed_steps_set_by_injected_node(self):
        for node_fn, step in [
            (make_clinical_node(_ClinicalPass()), "clinical_consistency"),
            (make_fraud_node(_FraudPass()), "fraud_detection"),
            (make_case_reviewer_node(_ReviewApprove()), "case_reviewer"),
            (make_audit_node(_AuditPass()), "audit"),
        ]:
            updates = node_fn(_state())
            assert step in updates["completed_steps"], f"{step} absent de completed_steps"


# ── 5. graph/nodes.py — nœuds stubs par défaut ────────────────────────────────


class TestGraphNodeStubDefaults:
    def test_node_clinical_consistency_not_evaluated(self):
        updates = node_clinical_consistency(_state())
        assert updates["clinical_result"].status is VerificationStatus.NOT_EVALUATED

    def test_node_fraud_detection_not_evaluated(self):
        updates = node_fraud_detection(_state())
        assert updates["fraud_result"].status is VerificationStatus.NOT_EVALUATED

    def test_node_case_reviewer_pending(self):
        updates = node_case_reviewer(_state())
        assert updates["review_result"].recommendation is Recommendation.PENDING

    def test_node_audit_not_evaluated(self):
        updates = node_audit(_state())
        assert updates["audit_result"].status is VerificationStatus.NOT_EVALUATED


# ── 6. graph/nodes.py — make_node_<name> factories ───────────────────────────


class TestGraphNodeFactories:
    def test_make_node_clinical_with_impl_returns_pass(self):
        node_fn = make_node_clinical_consistency(_ClinicalPass())
        updates = node_fn(_state())
        assert updates["clinical_result"].status is VerificationStatus.PASS

    def test_make_node_fraud_with_impl_returns_pass(self):
        node_fn = make_node_fraud_detection(_FraudPass())
        updates = node_fn(_state())
        assert updates["fraud_result"].status is VerificationStatus.PASS

    def test_make_node_case_reviewer_with_impl_returns_approve(self):
        node_fn = make_node_case_reviewer(_ReviewApprove())
        updates = node_fn(_state())
        assert updates["review_result"].recommendation is Recommendation.APPROVE

    def test_make_node_audit_with_impl_returns_pass(self):
        node_fn = make_node_audit(_AuditPass())
        updates = node_fn(_state())
        assert updates["audit_result"].status is VerificationStatus.PASS

    def test_make_node_none_impl_uses_stub(self):
        node_fn = make_node_clinical_consistency(None)
        updates = node_fn(_state())
        assert updates["clinical_result"].status is VerificationStatus.NOT_EVALUATED

    def test_make_node_exception_isolated(self):
        class Crasher:
            def run(self, state):
                raise RuntimeError("boom")

        node_fn = make_node_clinical_consistency(Crasher())
        updates = node_fn(_state())
        assert updates.get("errors")
        assert "clinical_consistency" in updates["errors"][0]

    def test_make_node_fraud_exception_isolated(self):
        class Crasher:
            def run(self, state):
                raise ValueError("bad input")

        node_fn = make_node_fraud_detection(Crasher())
        updates = node_fn(_state())
        assert updates.get("errors")

    def test_make_node_case_reviewer_exception_isolated(self):
        class Crasher:
            def run(self, state):
                raise OSError("network down")

        node_fn = make_node_case_reviewer(Crasher())
        updates = node_fn(_state())
        assert updates.get("errors")


# ── 7. NODE_REGISTRY — complétude et stabilité des noms ──────────────────────


class TestNodeRegistryCompleteness:
    ALL_EXPECTED = {
        "claim_intake", "security_gate", "privacy",
        "fhir_validator", "medical_coding", "document_ocr",
        "identity_coverage",
        "clinical_consistency", "fraud_detection", "case_reviewer", "audit",
    }

    def test_registry_has_eleven_entries(self):
        assert len(NODE_REGISTRY) == 11

    def test_registry_contains_all_expected_keys(self):
        assert set(NODE_REGISTRY.keys()) == self.ALL_EXPECTED

    @pytest.mark.parametrize("key", list(ALL_EXPECTED))
    def test_all_entries_are_callable(self, key):
        assert callable(NODE_REGISTRY[key])

    def test_stub_nodes_names_stable(self):
        assert NODE_REGISTRY["clinical_consistency"].__name__ == "node_clinical_consistency"
        assert NODE_REGISTRY["fraud_detection"].__name__ == "node_fraud_detection"
        assert NODE_REGISTRY["case_reviewer"].__name__ == "node_case_reviewer"
        assert NODE_REGISTRY["audit"].__name__ == "node_audit"

    @pytest.mark.parametrize("key", list(ALL_EXPECTED))
    def test_each_node_returns_dict_on_empty_state(self, key):
        result = NODE_REGISTRY[key]({})
        assert isinstance(result, dict)
