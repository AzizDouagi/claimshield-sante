"""Tests unitaires des fonctions de routage — graph/edges.py.

Chaque classe couvre une fonction de routage.  Les résultats d'agents sont
simulés via ``types.SimpleNamespace`` : le routage ne dépend que des champs
``status``, ``decision``, ``recommendation`` et ``human_review_required``.
Aucun agent réel n'est instancié.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from graph.edges import (
    ALL_ROUTES,
    CONTINUE,
    END,
    FAILURE,
    NEEDS_REVIEW,
    QUARANTINE,
    RETRY,
    route_coding,
    route_fhir,
    route_identity_coverage,
    route_intake,
    route_ocr,
    route_privacy,
    route_review,
    route_security,
    route_verification_fan_in,
)
from schemas.domain import (
    IntakeStatus,
    PrivacyDecision,
    Recommendation,
    SecurityDecision,
    VerificationStatus,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _intake_state(status: str | None, *, via_result: bool = False) -> dict:
    """Construit un state minimal pour route_intake."""
    if status is None:
        return {}
    if via_result:
        return {"intake_result": SimpleNamespace(status=status)}
    return {"intake_status": status}


def _security_state(decision: str | None) -> dict:
    if decision is None:
        return {}
    return {"security_result": SimpleNamespace(decision=decision)}


def _privacy_state(decision: str | None) -> dict:
    if decision is None:
        return {}
    return {"privacy_result": SimpleNamespace(decision=decision)}


def _single_result_state(key: str, status: str | None) -> dict:
    if status is None:
        return {}
    return {key: SimpleNamespace(status=status)}


def _id_cov_state(id_status: str, cov_status: str) -> dict:
    return {
        "identity_coverage_result": SimpleNamespace(
            identity=SimpleNamespace(status=id_status),
            coverage=SimpleNamespace(status=cov_status),
        )
    }


def _fan_in_state(
    *,
    ocr: str | None = None,
    fhir: str | None = None,
    coding: str | None = None,
    id_status: str | None = None,
    cov_status: str | None = None,
) -> dict:
    state: dict = {}
    if ocr is not None:
        state["ocr_result"] = SimpleNamespace(status=ocr)
    if fhir is not None:
        state["fhir_result"] = SimpleNamespace(status=fhir)
    if coding is not None:
        state["coding_result"] = SimpleNamespace(status=coding)
    if id_status is not None and cov_status is not None:
        state["identity_coverage_result"] = SimpleNamespace(
            identity=SimpleNamespace(status=id_status),
            coverage=SimpleNamespace(status=cov_status),
        )
    return state


def _review_state(recommendation: str | None, *, human: bool = False) -> dict:
    if recommendation is None:
        return {}
    return {
        "review_result": SimpleNamespace(
            recommendation=recommendation,
            human_review_required=human,
        )
    }


# ── 1. Constantes ──────────────────────────────────────────────────────────────


class TestRouteConstants:
    def test_all_routes_are_distinct(self):
        values = [CONTINUE, QUARANTINE, NEEDS_REVIEW, RETRY, FAILURE, END]
        assert len(set(values)) == len(values)

    def test_all_routes_in_set(self):
        assert ALL_ROUTES == {CONTINUE, QUARANTINE, NEEDS_REVIEW, RETRY, FAILURE, END}

    def test_all_routes_are_strings(self):
        for r in ALL_ROUTES:
            assert isinstance(r, str)


# ── 2. route_intake ────────────────────────────────────────────────────────────


class TestRouteIntake:
    def test_accepted_via_promoted_status(self):
        assert route_intake(_intake_state(IntakeStatus.ACCEPTED)) == CONTINUE

    def test_quarantined_via_promoted_status(self):
        assert route_intake(_intake_state(IntakeStatus.QUARANTINED)) == QUARANTINE

    def test_blocked_via_promoted_status(self):
        assert route_intake(_intake_state(IntakeStatus.BLOCKED)) == FAILURE

    def test_error_via_promoted_status(self):
        assert route_intake(_intake_state(IntakeStatus.ERROR)) == RETRY

    def test_accepted_via_intake_result_fallback(self):
        assert route_intake(_intake_state(IntakeStatus.ACCEPTED, via_result=True)) == CONTINUE

    def test_quarantined_via_intake_result_fallback(self):
        assert route_intake(_intake_state(IntakeStatus.QUARANTINED, via_result=True)) == QUARANTINE

    def test_blocked_via_intake_result_fallback(self):
        assert route_intake(_intake_state(IntakeStatus.BLOCKED, via_result=True)) == FAILURE

    def test_error_via_intake_result_fallback(self):
        assert route_intake(_intake_state(IntakeStatus.ERROR, via_result=True)) == RETRY

    def test_promoted_status_takes_priority_over_result(self):
        state = {
            "intake_status": IntakeStatus.ACCEPTED,
            "intake_result": SimpleNamespace(status=IntakeStatus.BLOCKED),
        }
        assert route_intake(state) == CONTINUE

    def test_no_status_no_result_returns_failure(self):
        assert route_intake({}) == FAILURE

    def test_none_status_falls_through_to_result(self):
        state = {
            "intake_status": None,
            "intake_result": SimpleNamespace(status=IntakeStatus.ACCEPTED),
        }
        assert route_intake(state) == CONTINUE

    def test_invalid_status_string_returns_failure(self):
        assert route_intake({"intake_status": "UNKNOWN_VALUE"}) == FAILURE


# ── 3. route_security ─────────────────────────────────────────────────────────


class TestRouteSecurity:
    def test_allow_returns_continue(self):
        assert route_security(_security_state(SecurityDecision.ALLOW)) == CONTINUE

    def test_quarantine_returns_quarantine(self):
        assert route_security(_security_state(SecurityDecision.QUARANTINE)) == QUARANTINE

    def test_block_returns_failure(self):
        assert route_security(_security_state(SecurityDecision.BLOCK)) == FAILURE

    def test_missing_result_returns_failure(self):
        assert route_security({}) == FAILURE

    def test_invalid_decision_returns_failure(self):
        assert route_security({"security_result": SimpleNamespace(decision="INVALID")}) == FAILURE


# ── 4. route_privacy ──────────────────────────────────────────────────────────


class TestRoutePrivacy:
    def test_allow_returns_continue(self):
        assert route_privacy(_privacy_state(PrivacyDecision.ALLOW)) == CONTINUE

    def test_block_returns_failure(self):
        assert route_privacy(_privacy_state(PrivacyDecision.BLOCK)) == FAILURE

    def test_missing_result_returns_failure(self):
        assert route_privacy({}) == FAILURE

    def test_invalid_decision_returns_failure(self):
        assert route_privacy({"privacy_result": SimpleNamespace(decision="INVALID")}) == FAILURE


# ── 5. route_ocr ──────────────────────────────────────────────────────────────


class TestRouteOcr:
    def test_pass_returns_continue(self):
        assert route_ocr(_single_result_state("ocr_result", VerificationStatus.PASS)) == CONTINUE

    def test_needs_review_returns_needs_review(self):
        assert route_ocr(_single_result_state("ocr_result", VerificationStatus.NEEDS_REVIEW)) == NEEDS_REVIEW

    def test_fail_returns_failure(self):
        assert route_ocr(_single_result_state("ocr_result", VerificationStatus.FAIL)) == FAILURE

    def test_pending_returns_retry(self):
        assert route_ocr(_single_result_state("ocr_result", VerificationStatus.PENDING)) == RETRY

    def test_not_evaluated_returns_retry(self):
        assert route_ocr(_single_result_state("ocr_result", VerificationStatus.NOT_EVALUATED)) == RETRY

    def test_missing_result_returns_failure(self):
        assert route_ocr({}) == FAILURE

    def test_invalid_status_returns_failure(self):
        assert route_ocr({"ocr_result": SimpleNamespace(status="BOGUS")}) == FAILURE


# ── 6. route_fhir ─────────────────────────────────────────────────────────────


class TestRouteFhir:
    def test_pass_returns_continue(self):
        assert route_fhir(_single_result_state("fhir_result", VerificationStatus.PASS)) == CONTINUE

    def test_needs_review_returns_needs_review(self):
        assert route_fhir(_single_result_state("fhir_result", VerificationStatus.NEEDS_REVIEW)) == NEEDS_REVIEW

    def test_fail_returns_failure(self):
        assert route_fhir(_single_result_state("fhir_result", VerificationStatus.FAIL)) == FAILURE

    def test_pending_returns_retry(self):
        assert route_fhir(_single_result_state("fhir_result", VerificationStatus.PENDING)) == RETRY

    def test_not_evaluated_returns_retry(self):
        assert route_fhir(_single_result_state("fhir_result", VerificationStatus.NOT_EVALUATED)) == RETRY

    def test_missing_result_returns_failure(self):
        assert route_fhir({}) == FAILURE


# ── 7. route_coding ───────────────────────────────────────────────────────────


class TestRouteCoding:
    def test_pass_returns_continue(self):
        assert route_coding(_single_result_state("coding_result", VerificationStatus.PASS)) == CONTINUE

    def test_needs_review_returns_needs_review(self):
        assert route_coding(_single_result_state("coding_result", VerificationStatus.NEEDS_REVIEW)) == NEEDS_REVIEW

    def test_fail_returns_failure(self):
        assert route_coding(_single_result_state("coding_result", VerificationStatus.FAIL)) == FAILURE

    def test_pending_returns_retry(self):
        assert route_coding(_single_result_state("coding_result", VerificationStatus.PENDING)) == RETRY

    def test_not_evaluated_returns_retry(self):
        assert route_coding(_single_result_state("coding_result", VerificationStatus.NOT_EVALUATED)) == RETRY

    def test_missing_result_returns_failure(self):
        assert route_coding({}) == FAILURE


# ── 8. route_identity_coverage ────────────────────────────────────────────────


class TestRouteIdentityCoverage:
    def test_both_pass_returns_continue(self):
        assert route_identity_coverage(_id_cov_state("PASS", "PASS")) == CONTINUE

    def test_identity_fail_returns_failure(self):
        assert route_identity_coverage(_id_cov_state("FAIL", "PASS")) == FAILURE

    def test_coverage_fail_returns_failure(self):
        assert route_identity_coverage(_id_cov_state("PASS", "FAIL")) == FAILURE

    def test_both_fail_returns_failure(self):
        assert route_identity_coverage(_id_cov_state("FAIL", "FAIL")) == FAILURE

    def test_identity_needs_review_returns_needs_review(self):
        assert route_identity_coverage(_id_cov_state("NEEDS_REVIEW", "PASS")) == NEEDS_REVIEW

    def test_coverage_needs_review_returns_needs_review(self):
        assert route_identity_coverage(_id_cov_state("PASS", "NEEDS_REVIEW")) == NEEDS_REVIEW

    def test_both_needs_review_returns_needs_review(self):
        assert route_identity_coverage(_id_cov_state("NEEDS_REVIEW", "NEEDS_REVIEW")) == NEEDS_REVIEW

    def test_fail_takes_priority_over_needs_review(self):
        assert route_identity_coverage(_id_cov_state("FAIL", "NEEDS_REVIEW")) == FAILURE

    def test_identity_pending_returns_retry(self):
        assert route_identity_coverage(_id_cov_state("PENDING", "PASS")) == RETRY

    def test_coverage_not_evaluated_returns_retry(self):
        assert route_identity_coverage(_id_cov_state("PASS", "NOT_EVALUATED")) == RETRY

    def test_missing_result_returns_failure(self):
        assert route_identity_coverage({}) == FAILURE

    def test_invalid_status_returns_failure(self):
        state = {
            "identity_coverage_result": SimpleNamespace(
                identity=SimpleNamespace(status="BAD"),
                coverage=SimpleNamespace(status="PASS"),
            )
        }
        assert route_identity_coverage(state) == FAILURE


# ── 9. route_verification_fan_in ──────────────────────────────────────────────


class TestRouteVerificationFanIn:
    def test_all_pass_returns_continue(self):
        state = _fan_in_state(ocr="PASS", fhir="PASS", coding="PASS")
        assert route_verification_fan_in(state) == CONTINUE

    def test_any_fail_returns_failure(self):
        state = _fan_in_state(ocr="PASS", fhir="FAIL", coding="PASS")
        assert route_verification_fan_in(state) == FAILURE

    def test_any_needs_review_returns_needs_review(self):
        state = _fan_in_state(ocr="PASS", fhir="PASS", coding="NEEDS_REVIEW")
        assert route_verification_fan_in(state) == NEEDS_REVIEW

    def test_fail_takes_priority_over_needs_review(self):
        state = _fan_in_state(ocr="FAIL", fhir="NEEDS_REVIEW", coding="PASS")
        assert route_verification_fan_in(state) == FAILURE

    def test_identity_fail_triggers_failure(self):
        state = _fan_in_state(ocr="PASS", id_status="FAIL", cov_status="PASS")
        assert route_verification_fan_in(state) == FAILURE

    def test_coverage_needs_review_triggers_needs_review(self):
        state = _fan_in_state(ocr="PASS", id_status="PASS", cov_status="NEEDS_REVIEW")
        assert route_verification_fan_in(state) == NEEDS_REVIEW

    def test_not_evaluated_ignored_with_pass_returns_continue(self):
        state = _fan_in_state(ocr="PASS", fhir="NOT_EVALUATED", coding="PASS")
        assert route_verification_fan_in(state) == CONTINUE

    def test_all_not_evaluated_returns_failure(self):
        state = _fan_in_state(ocr="NOT_EVALUATED", fhir="NOT_EVALUATED")
        assert route_verification_fan_in(state) == FAILURE

    def test_no_results_returns_failure(self):
        assert route_verification_fan_in({}) == FAILURE

    def test_single_pass_result_returns_continue(self):
        state = _fan_in_state(ocr="PASS")
        assert route_verification_fan_in(state) == CONTINUE

    def test_single_fail_result_returns_failure(self):
        state = _fan_in_state(coding="FAIL")
        assert route_verification_fan_in(state) == FAILURE

    def test_identity_coverage_both_pass_contributes_continue(self):
        state = _fan_in_state(ocr="PASS", id_status="PASS", cov_status="PASS")
        assert route_verification_fan_in(state) == CONTINUE

    def test_invalid_status_in_any_result_returns_failure(self):
        state = {"ocr_result": SimpleNamespace(status="INVALID")}
        assert route_verification_fan_in(state) == FAILURE


# ── 10. route_review ──────────────────────────────────────────────────────────


class TestRouteReview:
    def test_approve_no_human_returns_end(self):
        assert route_review(_review_state(Recommendation.APPROVE, human=False)) == END

    def test_approve_with_human_returns_needs_review(self):
        assert route_review(_review_state(Recommendation.APPROVE, human=True)) == NEEDS_REVIEW

    def test_reject_returns_end(self):
        assert route_review(_review_state(Recommendation.REJECT)) == END

    def test_reject_with_human_still_returns_end(self):
        assert route_review(_review_state(Recommendation.REJECT, human=True)) == END

    def test_pending_returns_needs_review(self):
        assert route_review(_review_state(Recommendation.PENDING)) == NEEDS_REVIEW

    def test_missing_result_returns_failure(self):
        assert route_review({}) == FAILURE

    def test_invalid_recommendation_returns_failure(self):
        state = {"review_result": SimpleNamespace(recommendation="INVALID", human_review_required=False)}
        assert route_review(state) == FAILURE


# ── 11. Invariants transversaux ───────────────────────────────────────────────


class TestRouteInvariants:
    """Toutes les fonctions retournent exclusivement des routes du registre."""

    FUNCTIONS = [
        route_intake,
        route_security,
        route_privacy,
        route_ocr,
        route_fhir,
        route_coding,
        route_review,
        route_verification_fan_in,
        route_identity_coverage,
    ]

    @pytest.mark.parametrize("fn", FUNCTIONS)
    def test_empty_state_returns_known_route(self, fn):
        result = fn({})
        assert result in ALL_ROUTES, f"{fn.__name__} a retourné '{result}' hors du registre"

    @pytest.mark.parametrize("fn", FUNCTIONS)
    def test_return_type_is_string(self, fn):
        result = fn({})
        assert isinstance(result, str)
