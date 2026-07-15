"""Tests de agents/autonomous_decision_agent (V2) — Phase V2-6.

`TestBoundedAuthority` est le cœur de cette suite (garde-fou non
négociable du plan V2 §6) : pour chaque borne, une réponse LLM qui tente
de la dépasser doit toujours être neutralisée, quel que soit le contenu du
prompt.
"""
from __future__ import annotations

from unittest.mock import Mock

from agents.autonomous_decision_agent.agent import node, run
from agents.autonomous_decision_agent.schemas import LlmAutonomousDecision
from schemas.domain import ClaimDecisionV2, IntakeSafetyStatus, VerificationStatus
from schemas.results import LlmMetadata, ProcedureCoding
from schemas.v2_results import (
    EligibilityResult,
    EvidenceCompleteness,
    IntakeSafetyResult,
    MedicalRiskResult,
    MedicalRiskResultPayload,
    RiskLevel,
)
from schemas.results import CoverageResult, IdentityResult


def _llm_trace() -> LlmMetadata:
    return LlmMetadata(model_name="gemma4:latest", prompt_version="1.0.0")


def _intake_safety(status: IntakeSafetyStatus = IntakeSafetyStatus.ACCEPTED) -> IntakeSafetyResult:
    return IntakeSafetyResult(
        case_id="CLM-6001", status=status, reasons=["motif"], llm_trace=_llm_trace()
    )


def _eligibility(status: VerificationStatus = VerificationStatus.PASS) -> EligibilityResult:
    return EligibilityResult(
        case_id="CLM-6001",
        status=status,
        identity=IdentityResult(status=status),
        coverage=CoverageResult(status=status),
        llm_trace=_llm_trace(),
    )


def _medical_risk(
    risk_level: RiskLevel = RiskLevel.LOW,
    status: VerificationStatus = VerificationStatus.PASS,
    codings: list[ProcedureCoding] | None = None,
    evidence_completeness: EvidenceCompleteness = EvidenceCompleteness.COMPLETE,
) -> MedicalRiskResult:
    return MedicalRiskResult(
        case_id="CLM-6001",
        status=status,
        llm_trace=_llm_trace(),
        result_payload=MedicalRiskResultPayload(
            risk_level=risk_level, codings=codings or [], evidence_completeness=evidence_completeness
        ),
    )


def _decision(**overrides) -> LlmAutonomousDecision:
    defaults = {
        "decision": "APPROVE",
        "summary": "Dossier conforme.",
        "reasons": [],
        "confidence": 0.9,
    }
    defaults.update(overrides)
    return LlmAutonomousDecision(**defaults)


def _state(**results) -> dict:
    state = {
        "case_id": "CLM-6001",
        "schema_version": "2.0.0",
        "current_step": "medical_risk",
        "completed_steps": [],
        "intake_safety_result": _intake_safety(),
        "document_understanding_result": None,
        "eligibility_result": _eligibility(),
        "medical_risk_result": _medical_risk(),
    }
    state.update(results)
    return state


class TestBoundedAuthority:
    def test_blocked_intake_forces_reject_without_llm_call(self, monkeypatch):
        spy = Mock()
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision", spy
        )
        state = _state(intake_safety_result=_intake_safety(IntakeSafetyStatus.BLOCKED))
        result = run("CLM-6001", state)
        assert result.decision is ClaimDecisionV2.REJECT
        spy.assert_not_called()

    def test_quarantined_intake_forces_quarantine_without_llm_call(self, monkeypatch):
        spy = Mock()
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision", spy
        )
        state = _state(intake_safety_result=_intake_safety(IntakeSafetyStatus.QUARANTINED))
        result = run("CLM-6001", state)
        assert result.decision is ClaimDecisionV2.QUARANTINE
        spy.assert_not_called()

    def test_high_risk_never_produces_approve_even_if_llm_tries(self, monkeypatch):
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=_decision(decision="APPROVE")),
        )
        state = _state(medical_risk_result=_medical_risk(risk_level=RiskLevel.HIGH))
        result = run("CLM-6001", state)
        assert result.decision is not ClaimDecisionV2.APPROVE
        # Correctif post-mesure V2-10 : REQUEST_MORE_INFO retiré du plafond HIGH
        # (un danger réel confirmé n'est pas résolu par une demande d'info).
        assert result.decision in (ClaimDecisionV2.REJECT, ClaimDecisionV2.QUARANTINE)
        assert any("hors des bornes" in b for b in result.bounded_by)

    def test_critical_risk_forces_quarantine_without_llm_call(self, monkeypatch):
        spy = Mock()
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision", spy
        )
        state = _state(medical_risk_result=_medical_risk(risk_level=RiskLevel.CRITICAL))
        result = run("CLM-6001", state)
        assert result.decision is ClaimDecisionV2.QUARANTINE
        spy.assert_not_called()
        assert any("CRITICAL" in b for b in result.bounded_by)

    def test_insufficient_evidence_forces_request_more_info_never_quarantine(self, monkeypatch):
        spy = Mock()
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision", spy
        )
        state = _state(
            medical_risk_result=_medical_risk(
                risk_level=RiskLevel.LOW, evidence_completeness=EvidenceCompleteness.INSUFFICIENT
            )
        )
        result = run("CLM-6001", state)
        assert result.decision is ClaimDecisionV2.REQUEST_MORE_INFO
        assert result.decision is not ClaimDecisionV2.QUARANTINE
        spy.assert_not_called()

    def test_fallback_prefers_request_more_info_over_quarantine(self, monkeypatch):
        """`_merge_decision` — correctif post-mesure V2-10 : quand la
        proposition LLM sort de `allowed`, le repli ne doit plus retomber
        systématiquement sur QUARANTINE. Ici `allowed` par défaut inclut
        REQUEST_MORE_INFO, jamais QUARANTINE (retiré de l'ensemble par
        défaut) — donc même une décision LLM invalide ('QUARANTINE' n'étant
        de toute façon plus dans `allowed`) retombe sur REQUEST_MORE_INFO."""
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=_decision(decision="QUARANTINE")),
        )
        result = run("CLM-6001", _state())
        assert result.decision is ClaimDecisionV2.REQUEST_MORE_INFO
        assert result.decision is not ClaimDecisionV2.QUARANTINE

    def test_high_risk_never_produces_partial_approve(self, monkeypatch):
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=_decision(decision="PARTIAL_APPROVE")),
        )
        state = _state(medical_risk_result=_medical_risk(risk_level=RiskLevel.HIGH))
        result = run("CLM-6001", state)
        assert result.decision is not ClaimDecisionV2.PARTIAL_APPROVE

    def test_partial_approve_rejected_without_mixed_codings_condition(self, monkeypatch):
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=_decision(decision="PARTIAL_APPROVE")),
        )
        # Aucun coding mixte PASS/non-PASS -> condition PARTIAL_APPROVE non réunie.
        state = _state(medical_risk_result=_medical_risk(codings=[]))
        result = run("CLM-6001", state)
        assert result.decision is not ClaimDecisionV2.PARTIAL_APPROVE

    def test_partial_approve_allowed_with_mixed_codings_condition(self, monkeypatch):
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=_decision(decision="PARTIAL_APPROVE")),
        )
        mixed_codings = [
            ProcedureCoding(original_description="a", status=VerificationStatus.PASS, proposed_code="123"),
            ProcedureCoding(original_description="b", status=VerificationStatus.NEEDS_REVIEW),
        ]
        state = _state(medical_risk_result=_medical_risk(codings=mixed_codings))
        result = run("CLM-6001", state)
        assert result.decision is ClaimDecisionV2.PARTIAL_APPROVE

    def test_technical_failure_never_selectable_by_llm(self, monkeypatch):
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=_decision(decision="TECHNICAL_FAILURE")),
        )
        state = _state()
        result = run("CLM-6001", state)
        # TECHNICAL_FAILURE n'appartient jamais à l'ensemble autorisé -> repli.
        assert result.decision is not ClaimDecisionV2.TECHNICAL_FAILURE


class TestNominal:
    def test_approve_accepted_when_within_bounds(self, monkeypatch):
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=_decision(decision="APPROVE")),
        )
        result = run("CLM-6001", _state())
        assert result.decision is ClaimDecisionV2.APPROVE
        assert result.status is VerificationStatus.PASS

    def test_reject_accepted_when_within_bounds(self, monkeypatch):
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=_decision(decision="REJECT")),
        )
        result = run("CLM-6001", _state())
        assert result.decision is ClaimDecisionV2.REJECT


class TestLlmFailClosed:
    def test_llm_unavailable_forces_technical_failure(self, monkeypatch):
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=None),
        )
        result = run("CLM-6001", _state())
        assert result.decision is ClaimDecisionV2.TECHNICAL_FAILURE
        assert result.status is VerificationStatus.FAIL
        assert any(e.code == "LLM_UNAVAILABLE" for e in result.errors)


class TestDisagreements:
    def test_disagreement_detected_between_document_and_eligibility(self, monkeypatch):
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=_decision()),
        )
        from schemas.v2_results import DocumentUnderstandingResult

        doc_result = DocumentUnderstandingResult(
            case_id="CLM-6001", status=VerificationStatus.FAIL, llm_trace=_llm_trace()
        )
        state = _state(
            document_understanding_result=doc_result,
            eligibility_result=_eligibility(VerificationStatus.PASS),
        )
        result = run("CLM-6001", state)
        assert len(result.disagreements) >= 1


class TestNodeIntegration:
    def test_node_updates_state(self, monkeypatch):
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=_decision(decision="APPROVE")),
        )
        updates = node(_state())  # type: ignore[arg-type]
        assert updates["current_step"] == "autonomous_decision"
        assert updates["final_decision"] is ClaimDecisionV2.APPROVE
        assert "errors" not in updates

    def test_node_reports_errors_on_reject(self, monkeypatch):
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=_decision(decision="REJECT")),
        )
        updates = node(_state())  # type: ignore[arg-type]
        assert updates["final_decision"] is ClaimDecisionV2.REJECT
        assert updates["errors"]
