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
    RELAUNCH_RESULT_FIELDS,
    RELAUNCH_TARGETS,
    RETRY,
    route_after_audit,
    route_coding,
    route_fhir,
    route_human_review,
    route_identity_coverage,
    route_intake,
    route_ocr,
    route_privacy,
    route_result_consistency,
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
            result_payload=SimpleNamespace(recommendation=recommendation),
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


# ── 9bis. route_result_consistency ────────────────────────────────────────────


class TestRouteResultConsistency:
    def test_agreement_returns_continue(self):
        state = {
            "fhir_result": SimpleNamespace(status="PASS"),
            "coding_result": SimpleNamespace(status="PASS"),
        }
        assert route_result_consistency(state) == CONTINUE

    def test_no_results_returns_continue(self):
        """Rien à comparer n'est pas une raison de bloquer le pipeline."""
        assert route_result_consistency({}) == CONTINUE

    def test_minor_disagreement_returns_continue(self):
        """Un écart d'un cran (PASS vs NEEDS_REVIEW) est signalé mais ne
        bloque pas — seul un désaccord critique route vers needs_review."""
        state = {
            "fhir_result": SimpleNamespace(status="PASS"),
            "coding_result": SimpleNamespace(status="NEEDS_REVIEW"),
        }
        assert route_result_consistency(state) == CONTINUE

    def test_critical_disagreement_returns_needs_review(self):
        state = {
            "fhir_result": SimpleNamespace(status="PASS"),
            "coding_result": SimpleNamespace(status="FAIL"),
        }
        assert route_result_consistency(state) == NEEDS_REVIEW

    def test_critical_disagreement_route_never_alters_source_results(self):
        """Le routage se contente de signaler — il ne modifie jamais les
        résultats source ni ne choisit lequel est correct."""
        fhir_result = SimpleNamespace(status="PASS")
        coding_result = SimpleNamespace(status="FAIL")
        state = {"fhir_result": fhir_result, "coding_result": coding_result}

        route = route_result_consistency(state)

        assert route == NEEDS_REVIEW
        assert fhir_result.status == "PASS"
        assert coding_result.status == "FAIL"


# ── 10. route_review ──────────────────────────────────────────────────────────


class TestRouteReview:
    def test_approve_no_human_returns_end(self):
        assert route_review(_review_state(Recommendation.APPROVE, human=False)) == END

    def test_approve_with_human_returns_needs_review(self):
        assert route_review(_review_state(Recommendation.APPROVE, human=True)) == NEEDS_REVIEW

    def test_reject_no_human_returns_end(self):
        """Chemin défensif/legacy — l'implémentation réelle ne produit
        jamais human_review_required=False (voir test_reject_with_human_requires_review)."""
        assert route_review(_review_state(Recommendation.REJECT, human=False)) == END

    def test_reject_with_human_requires_review(self):
        """Un REJECT reste une décision de dossier : jamais finalisé
        (END/audit) sans revue humaine, exactement comme APPROVE."""
        assert route_review(_review_state(Recommendation.REJECT, human=True)) == NEEDS_REVIEW

    def test_pending_returns_needs_review(self):
        assert route_review(_review_state(Recommendation.PENDING)) == NEEDS_REVIEW

    def test_missing_result_returns_failure(self):
        assert route_review({}) == FAILURE

    def test_invalid_recommendation_returns_failure(self):
        state = {
            "review_result": SimpleNamespace(
                result_payload=SimpleNamespace(recommendation="INVALID"),
                human_review_required=False,
            )
        }
        assert route_review(state) == FAILURE


# ── 11bis. route_human_review — route de relance (« relancer ») ──────────────


def _human_review_state(
    decision: str | None,
    *,
    target_node: str | None = None,
    correction_attempts: int = 0,
    target_has_run: bool = True,
) -> dict:
    """``target_has_run`` place un ``*_result`` non None pour ``target_node``,
    simulant qu'il a déjà été exécuté pour ce dossier — précondition exigée
    par ``route_human_review`` avant toute relance."""
    if decision is None:
        return {"correction_attempts": correction_attempts}
    human_decision = {"actor": "reviewer@example.com", "action": decision}
    if target_node is not None:
        human_decision["target_node"] = target_node
    state = {"human_decision": human_decision, "correction_attempts": correction_attempts}
    if target_node is not None and target_has_run:
        result_field = RELAUNCH_RESULT_FIELDS.get(target_node)
        if result_field is not None:
            state[result_field] = SimpleNamespace(status="NEEDS_REVIEW")
    return state


class TestRouteHumanReview:
    def test_approve_returns_audit(self):
        """Aucun chemin terminal ne contourne l'audit — APPROVE y passe
        d'abord, route_after_audit décide ensuite finalize/failure."""
        state = _human_review_state("APPROVE")
        assert route_human_review(state, max_attempts=3) == "audit"

    def test_modify_returns_audit(self):
        state = _human_review_state("MODIFY")
        assert route_human_review(state, max_attempts=3) == "audit"

    def test_reject_returns_audit(self):
        """REJECT passe aussi par audit — un rejet contrôlé, jamais un
        contournement direct vers failure."""
        state = _human_review_state("REJECT")
        assert route_human_review(state, max_attempts=3) == "audit"

    def test_needs_more_info_under_limit_returns_target_node(self):
        state = _human_review_state(
            "RETRY", target_node="document_ocr", correction_attempts=1
        )
        assert route_human_review(state, max_attempts=3) == "document_ocr"

    def test_needs_more_info_at_limit_returns_target_node(self):
        """attempts == max_attempts est encore autorisé (limite inclusive)."""
        state = _human_review_state(
            "RETRY", target_node="fhir_validator", correction_attempts=3
        )
        assert route_human_review(state, max_attempts=3) == "fhir_validator"

    def test_needs_more_info_beyond_limit_returns_failure(self):
        state = _human_review_state(
            "RETRY", target_node="document_ocr", correction_attempts=4
        )
        assert route_human_review(state, max_attempts=3) == FAILURE

    def test_needs_more_info_unknown_target_node_returns_failure(self):
        state = _human_review_state(
            "RETRY", target_node="finalize", correction_attempts=1
        )
        assert route_human_review(state, max_attempts=3) == FAILURE

    def test_needs_more_info_missing_target_node_returns_failure(self):
        state = _human_review_state("RETRY", correction_attempts=1)
        assert route_human_review(state, max_attempts=3) == FAILURE

    def test_needs_more_info_never_ran_returns_failure(self):
        """Précondition : on ne relance jamais un nœud sans résultat existant
        pour ce dossier — même s'il appartient à RELAUNCH_TARGETS et que le
        compteur est sous la limite."""
        state = _human_review_state(
            "RETRY",
            target_node="medical_coding",
            correction_attempts=1,
            target_has_run=False,
        )
        assert route_human_review(state, max_attempts=3) == FAILURE

    @pytest.mark.parametrize(
        "target_node", ["clinical_consistency", "fraud_detection", "case_reviewer"]
    )
    def test_needs_more_info_relaunches_agents_reimplemented_since_stage_12(self, target_node):
        """clinical_consistency/fraud_detection/case_reviewer sont désormais
        des agents réels (non des stubs) : relançables dès qu'ils ont déjà
        produit un résultat pour ce dossier, comme les 7 agents historiques."""
        state = _human_review_state(
            "RETRY", target_node=target_node, correction_attempts=1
        )
        assert route_human_review(state, max_attempts=3) == target_node

    @pytest.mark.parametrize(
        "target_node", ["clinical_consistency", "fraud_detection", "case_reviewer"]
    )
    def test_needs_more_info_refuses_reimplemented_agent_never_run(self, target_node):
        """Même précondition « déjà exécuté » pour les agents réintégrés :
        aucune relance possible sans résultat existant pour ce dossier."""
        state = _human_review_state(
            "RETRY",
            target_node=target_node,
            correction_attempts=1,
            target_has_run=False,
        )
        assert route_human_review(state, max_attempts=3) == FAILURE

    def test_needs_more_info_audit_is_never_a_valid_target(self):
        """audit_agent reste un stub — jamais une cible de relance valide,
        même si un humain la demande explicitement."""
        state = _human_review_state(
            "RETRY", target_node="audit", correction_attempts=1
        )
        assert route_human_review(state, max_attempts=3) == FAILURE

    def test_missing_human_decision_returns_failure(self):
        assert route_human_review({}, max_attempts=3) == FAILURE

    def test_unknown_decision_returns_failure(self):
        state = _human_review_state("MAYBE")
        assert route_human_review(state, max_attempts=3) == FAILURE

    def test_relaunch_result_fields_cover_all_targets(self):
        assert set(RELAUNCH_RESULT_FIELDS.keys()) == RELAUNCH_TARGETS

    def test_all_relaunch_targets_are_agent_nodes(self):
        expected = {
            "claim_intake", "security_gate", "privacy", "document_ocr",
            "fhir_validator", "identity_coverage", "medical_coding",
            "clinical_consistency", "fraud_detection", "case_reviewer",
        }
        assert RELAUNCH_TARGETS == expected

    def test_audit_is_never_a_relaunch_target(self):
        """audit_agent reste un stub jamais évalué avant la revue humaine —
        aucun audit_result ne peut donc jamais satisfaire sa précondition de
        relance ; l'exclure de RELAUNCH_TARGETS est une garantie explicite,
        pas un oubli (voir audit.md, constat historique « relance limitée »,
        résolu pour les autres agents mais volontairement pas pour audit)."""
        assert "audit" not in RELAUNCH_TARGETS
        assert "audit" not in RELAUNCH_RESULT_FIELDS


# ── 11ter. route_after_audit — clôture contrôlée par l'audit ─────────────────


class TestRouteAfterAudit:
    def test_reject_routes_to_failure(self):
        state = {"human_decision": {"actor": "a", "action": "REJECT"}}
        assert route_after_audit(state) == FAILURE

    def test_approve_routes_to_end(self):
        state = {"human_decision": {"actor": "a", "action": "APPROVE"}}
        assert route_after_audit(state) == END

    def test_modify_routes_to_end(self):
        state = {"human_decision": {"actor": "a", "action": "MODIFY"}}
        assert route_after_audit(state) == END

    def test_missing_human_decision_defaults_to_end(self):
        """Défensif uniquement : audit n'est jamais atteint sans qu'une
        décision humaine ait déjà été enregistrée par route_human_review ;
        en son absence, le défaut est le chemin nominal, jamais un blocage
        silencieux."""
        assert route_after_audit({}) == END


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
        route_result_consistency,
        route_after_audit,
    ]

    @pytest.mark.parametrize("fn", FUNCTIONS)
    def test_empty_state_returns_known_route(self, fn):
        result = fn({})
        assert result in ALL_ROUTES, f"{fn.__name__} a retourné '{result}' hors du registre"

    @pytest.mark.parametrize("fn", FUNCTIONS)
    def test_return_type_is_string(self, fn):
        result = fn({})
        assert isinstance(result, str)
