"""Tests de agents/autonomous_decision_agent (V2) — plan de remédiation
« autonomie décisionnelle V2 », phase 4.

`TestBoundedAuthority` reste le cœur de cette suite (garde-fou non
négociable) : pour chaque borne, une réponse LLM qui tente de la dépasser
doit toujours être neutralisée, quel que soit le contenu du prompt. La
matrice ne lit plus jamais `risk_level`/`evidence_completeness` agrégés —
uniquement la classification explicite des signaux
(`agents.autonomous_decision_agent.policy.classify_risk_signals`) et la
politique d'acceptation minimale (`evaluate_acceptance_requirements`).
"""
from __future__ import annotations

from unittest.mock import Mock

from agents.autonomous_decision_agent.agent import node, run
from agents.autonomous_decision_agent.schemas import LlmAutonomousDecision
from schemas.domain import ClaimDecisionV2, IntakeSafetyStatus, VerificationStatus
from schemas.results import (
    CoverageResult,
    FraudEvidence,
    FraudEvidenceSource,
    FraudSignal,
    IdentityResult,
    LlmMetadata,
    ClinicalEvidence,
    ClinicalEvidenceSource,
    ClinicalSignal,
    ProcedureCoding,
)
from schemas.domain import SeverityLevel
from schemas.v2_results import (
    DocumentUnderstandingResult,
    EligibilityResult,
    IntakeSafetyResult,
    MedicalRiskResult,
    MedicalRiskResultPayload,
)


def _llm_trace() -> LlmMetadata:
    return LlmMetadata(model_name="gemma4:latest", prompt_version="1.0.0")


def _intake_safety(status: IntakeSafetyStatus = IntakeSafetyStatus.ACCEPTED) -> IntakeSafetyResult:
    return IntakeSafetyResult(case_id="CLM-6001", status=status, reasons=["motif"], llm_trace=_llm_trace())


def _eligibility(
    identity_status: VerificationStatus = VerificationStatus.PASS,
    coverage_status: VerificationStatus = VerificationStatus.PASS,
    *,
    ceiling_exceeded: bool = False,
    coverage_data_available: bool = True,
) -> EligibilityResult:
    return EligibilityResult(
        case_id="CLM-6001",
        status=VerificationStatus.PASS if identity_status is coverage_status is VerificationStatus.PASS else VerificationStatus.NEEDS_REVIEW,
        identity=IdentityResult(status=identity_status),
        coverage=CoverageResult(status=coverage_status, ceiling_exceeded=ceiling_exceeded),
        coverage_data_available=coverage_data_available,
        llm_trace=_llm_trace(),
    )


def _document_understanding(
    status: VerificationStatus = VerificationStatus.PASS, confidence: float = 0.9
) -> DocumentUnderstandingResult:
    return DocumentUnderstandingResult(
        case_id="CLM-6001", status=status, confidence=confidence, llm_trace=_llm_trace()
    )


def _fraud_signal(signal_type: str, *, risk_contribution: float = 0.2) -> FraudSignal:
    return FraudSignal(
        signal_type=signal_type,
        description=f"Signal {signal_type}.",
        risk_contribution=risk_contribution,
        evidence=[FraudEvidence(source=FraudEvidenceSource.IDENTITY_COVERAGE, field="x", value="y")],
    )


def _clinical_signal(severity: SeverityLevel) -> ClinicalSignal:
    return ClinicalSignal(
        signal_type="MISSING_PRESCRIPTION_REFERENCE",
        description="Médicament facturé sans ordonnance.",
        fields_compared=["medication_count"],
        severity=severity,
        evidence=[
            ClinicalEvidence(source=ClinicalEvidenceSource.OCR_EXTRACTION, field="medication_count", value="1")
        ],
    )


def _medical_risk(
    *,
    status: VerificationStatus = VerificationStatus.PASS,
    codings: list[ProcedureCoding] | None = None,
    fraud_signals: list[FraudSignal] | None = None,
    clinical_signals: list[ClinicalSignal] | None = None,
    procedure_count: int = 1,
    medication_count: int = 0,
) -> MedicalRiskResult:
    default_codings = codings if codings is not None else [
        ProcedureCoding(original_description="acte résolu", status=VerificationStatus.PASS, proposed_code="123")
    ]
    return MedicalRiskResult(
        case_id="CLM-6001",
        status=status,
        llm_trace=_llm_trace(),
        result_payload=MedicalRiskResultPayload(
            procedure_count=procedure_count,
            medication_count=medication_count,
            codings=default_codings,
            fraud_signals=fraud_signals or [],
            clinical_signals=clinical_signals or [],
        ),
    )


def _decision(**overrides) -> LlmAutonomousDecision:
    defaults = {
        "recommended_decision": "APPROVE",
        "reasoning_summary": "Dossier conforme.",
        "supporting_factor_ids": [],
        "adverse_factor_ids": [],
        "confidence_adjustment": 0.0,
    }
    if "decision" in overrides:
        overrides["recommended_decision"] = overrides.pop("decision")
    defaults.update(overrides)
    return LlmAutonomousDecision(**defaults)


def _state(**results) -> dict:
    """État de base — dossier favorable, les 4 résultats amont présents
    (identité/couverture confirmées, au moins un acte résolu) : APPROVE
    reste reachable par défaut. Les tests qui veulent tester une branche
    forcée/un repli surchargent le(s) résultat(s) concerné(s)."""
    state = {
        "case_id": "CLM-6001",
        "schema_version": "2.0.0",
        "current_step": "medical_risk",
        "completed_steps": [],
        "intake_safety_result": _intake_safety(),
        "document_understanding_result": _document_understanding(),
        "eligibility_result": _eligibility(),
        "medical_risk_result": _medical_risk(),
    }
    state.update(results)
    return state


class TestUpstreamResultsPresence:
    """Cas 11 — panne réelle empêchant toute analyse."""

    def test_missing_document_understanding_result_forces_technical_failure_without_llm_call(
        self, monkeypatch
    ):
        spy = Mock()
        monkeypatch.setattr("agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision", spy)
        state = _state(document_understanding_result=None)
        result = run("CLM-6001", state)
        assert result.decision is ClaimDecisionV2.TECHNICAL_FAILURE
        spy.assert_not_called()

    def test_all_four_upstream_results_present_never_forces_technical_failure(self, monkeypatch):
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=_decision(decision="APPROVE")),
        )
        result = run("CLM-6001", _state())
        assert result.decision is not ClaimDecisionV2.TECHNICAL_FAILURE


class TestBoundedAuthority:
    def test_intake_technical_failure_forces_technical_failure_without_llm_call(self, monkeypatch):
        spy = Mock()
        monkeypatch.setattr("agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision", spy)
        state = _state(intake_safety_result=_intake_safety(IntakeSafetyStatus.TECHNICAL_FAILURE))
        result = run("CLM-6001", state)
        assert result.decision is ClaimDecisionV2.TECHNICAL_FAILURE
        spy.assert_not_called()

    def test_blocked_intake_forces_reject_without_llm_call(self, monkeypatch):
        spy = Mock()
        monkeypatch.setattr("agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision", spy)
        state = _state(intake_safety_result=_intake_safety(IntakeSafetyStatus.BLOCKED))
        result = run("CLM-6001", state)
        assert result.decision is ClaimDecisionV2.REJECT
        spy.assert_not_called()

    def test_quarantined_intake_forces_quarantine_without_llm_call(self, monkeypatch):
        spy = Mock()
        monkeypatch.setattr("agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision", spy)
        state = _state(intake_safety_result=_intake_safety(IntakeSafetyStatus.QUARANTINED))
        result = run("CLM-6001", state)
        assert result.decision is ClaimDecisionV2.QUARANTINE
        spy.assert_not_called()

    def test_confirmed_critical_clinical_signal_forces_quarantine_without_llm_call(self, monkeypatch):
        spy = Mock()
        monkeypatch.setattr("agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision", spy)
        state = _state(
            medical_risk_result=_medical_risk(clinical_signals=[_clinical_signal(SeverityLevel.CRITICAL)])
        )
        result = run("CLM-6001", state)
        assert result.decision is ClaimDecisionV2.QUARANTINE
        spy.assert_not_called()

    def test_confirmed_exact_duplicate_fraud_forces_quarantine_without_llm_call(self, monkeypatch):
        spy = Mock()
        monkeypatch.setattr("agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision", spy)
        state = _state(medical_risk_result=_medical_risk(fraud_signals=[_fraud_signal("EXACT_DUPLICATE_INVOICE")]))
        result = run("CLM-6001", state)
        assert result.decision is ClaimDecisionV2.QUARANTINE
        spy.assert_not_called()

    def test_confirmed_identity_mismatch_forces_reject_without_llm_call(self, monkeypatch):
        spy = Mock()
        monkeypatch.setattr("agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision", spy)
        state = _state(
            eligibility_result=_eligibility(identity_status=VerificationStatus.FAIL),
            medical_risk_result=_medical_risk(fraud_signals=[_fraud_signal("IDENTITY_MISMATCH")]),
        )
        result = run("CLM-6001", state)
        assert result.decision is ClaimDecisionV2.REJECT
        spy.assert_not_called()

    def test_confirmed_coverage_exclusion_forces_reject_without_llm_call(self, monkeypatch):
        spy = Mock()
        monkeypatch.setattr("agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision", spy)
        state = _state(
            eligibility_result=_eligibility(coverage_status=VerificationStatus.FAIL, coverage_data_available=True),
            medical_risk_result=_medical_risk(fraud_signals=[_fraud_signal("COVERAGE_INACTIVE_OR_EXPIRED")]),
        )
        result = run("CLM-6001", state)
        assert result.decision is ClaimDecisionV2.REJECT
        spy.assert_not_called()

    def test_suspected_fraud_risk_alone_never_forces_quarantine_or_reject(self, monkeypatch):
        """Correctif central du point 1 d'AZIZ : une combinaison
        `NEAR_DUPLICATE_INVOICE` (similarité) + `CEILING_EXCEEDED` (fait
        confirmé mais pas un danger) — qui aurait historiquement atteint
        `risk_level == HIGH` — ne force plus jamais QUARANTINE/REJECT. APPROVE
        et PARTIAL_APPROVE restent atteignables."""
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=_decision(decision="APPROVE")),
        )
        state = _state(
            eligibility_result=_eligibility(ceiling_exceeded=True),
            medical_risk_result=_medical_risk(
                fraud_signals=[_fraud_signal("NEAR_DUPLICATE_INVOICE"), _fraud_signal("CEILING_EXCEEDED")]
            ),
        )
        result = run("CLM-6001", state)
        assert result.decision is ClaimDecisionV2.APPROVE
        assert result.decision not in (ClaimDecisionV2.QUARANTINE, ClaimDecisionV2.REJECT)

    def test_partial_approve_rejected_without_mixed_codings_or_ceiling_condition(self, monkeypatch):
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=_decision(decision="PARTIAL_APPROVE")),
        )
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

    def test_partial_approve_allowed_with_confirmed_ceiling_exceeded(self, monkeypatch):
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=_decision(decision="PARTIAL_APPROVE")),
        )
        state = _state(
            eligibility_result=_eligibility(ceiling_exceeded=True),
            medical_risk_result=_medical_risk(codings=[]),
        )
        result = run("CLM-6001", state)
        assert result.decision is ClaimDecisionV2.PARTIAL_APPROVE

    def test_technical_failure_never_selectable_by_llm(self, monkeypatch):
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=_decision(decision="TECHNICAL_FAILURE")),
        )
        result = run("CLM-6001", _state())
        assert result.decision is not ClaimDecisionV2.TECHNICAL_FAILURE

    def test_llm_out_of_bounds_falls_back_to_evidence_based_tie_break(self, monkeypatch):
        """`_merge_decision` — le repli n'est plus un ordre statique
        (`_FALLBACK_PRIORITY`) mais `choose_accept_or_reject_from_available_evidence`,
        fondé sur les preuves disponibles. Ici le dossier est favorable
        (identité/couverture confirmées, acte résolu) — QUARANTINE proposé
        par le LLM est hors bornes (jamais offert sans signal confirmé) ->
        repli sur APPROVE (score/politique d'acceptation favorables)."""
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=_decision(decision="QUARANTINE")),
        )
        result = run("CLM-6001", _state())
        assert result.decision is ClaimDecisionV2.APPROVE
        assert any("hors des bornes" in b for b in result.bounded_by)


class TestAcceptanceRequirementsGate:
    """Point 4 d'AZIZ — une couverture UNKNOWN/NEEDS_REVIEW seule (sans
    aucun autre signal positif confirmé) ne suffit jamais à autoriser
    APPROVE, y compris quand le LLM le propose directement."""

    def test_coverage_unknown_alone_without_positive_signal_never_allows_approve(self, monkeypatch):
        spy = Mock(return_value=_decision(decision="APPROVE"))
        monkeypatch.setattr("agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision", spy)
        state = _state(
            eligibility_result=_eligibility(
                identity_status=VerificationStatus.NEEDS_REVIEW,
                coverage_status=VerificationStatus.NEEDS_REVIEW,
                coverage_data_available=False,
            ),
            medical_risk_result=_medical_risk(codings=[]),
        )
        result = run("CLM-6001", state)
        assert result.decision is not ClaimDecisionV2.APPROVE
        assert result.decision is ClaimDecisionV2.REJECT
        # APPROVE n'a jamais été proposé au LLM — jamais un contournement possible.
        spy.assert_not_called()

    def test_identity_confirmed_alone_allows_approve_despite_coverage_gap(self, monkeypatch):
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=_decision(decision="APPROVE")),
        )
        state = _state(
            eligibility_result=_eligibility(
                identity_status=VerificationStatus.PASS,
                coverage_status=VerificationStatus.NEEDS_REVIEW,
                coverage_data_available=False,
            ),
            medical_risk_result=_medical_risk(codings=[]),
        )
        result = run("CLM-6001", state)
        assert result.decision is ClaimDecisionV2.APPROVE
        assert result.confidence >= 0.0  # confiance non négative, simple garde-fou

    def test_missing_information_populated_on_unresolved_coding_gap(self, monkeypatch):
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=_decision(decision="APPROVE")),
        )
        state = _state(medical_risk_result=_medical_risk(fraud_signals=[_fraud_signal("UNRESOLVED_CODING")]))
        result = run("CLM-6001", state)
        assert any(m.code == "UNRESOLVED_CODING" for m in result.missing_information)
        assert result.decision in (ClaimDecisionV2.APPROVE, ClaimDecisionV2.PARTIAL_APPROVE, ClaimDecisionV2.REJECT)
        assert result.decision not in (ClaimDecisionV2.QUARANTINE,)


class TestNominal:
    def test_approve_accepted_when_within_bounds(self, monkeypatch):
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=_decision(decision="APPROVE")),
        )
        result = run("CLM-6001", _state())
        assert result.decision is ClaimDecisionV2.APPROVE
        assert result.status is VerificationStatus.PASS

    def test_explainability_fields_populated_on_approve(self, monkeypatch):
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=_decision(decision="APPROVE")),
        )
        result = run("CLM-6001", _state())
        assert result.risk_signal_classification == []  # dossier propre, aucun signal
        assert any(f.code == "IDENTITY_CONFIRMED" for f in result.supporting_factors)
        assert any(f.code == "COVERAGE_CONFIRMED" for f in result.supporting_factors)
        assert result.adverse_factors == []
        assert result.decisive_factors  # supporting_factors repris comme décisifs sur un APPROVE

    def test_explainability_fields_populated_on_forced_quarantine(self, monkeypatch):
        spy = Mock()
        monkeypatch.setattr("agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision", spy)
        state = _state(medical_risk_result=_medical_risk(fraud_signals=[_fraud_signal("EXACT_DUPLICATE_INVOICE")]))
        result = run("CLM-6001", state)
        assert any(s.signal_type == "EXACT_DUPLICATE_INVOICE" for s in result.risk_signal_classification)
        assert any(f.code == "EXACT_DUPLICATE_INVOICE" for f in result.adverse_factors)
        assert result.decisive_factors == result.adverse_factors

    def test_reject_accepted_when_within_bounds(self, monkeypatch):
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=_decision(decision="REJECT")),
        )
        result = run("CLM-6001", _state())
        assert result.decision is ClaimDecisionV2.REJECT


class TestLlmFailClosed:
    """Phase 5 — panne LLM isolée ≠ TECHNICAL_FAILURE : quand les 4 résultats
    amont existent, une panne du LLM final retombe sur
    `choose_accept_or_reject_from_available_evidence`, jamais un
    `TECHNICAL_FAILURE` automatique. `TECHNICAL_FAILURE` reste réservé au
    gate structurel (résultat amont absent, `TestUpstreamResultsPresence`)."""

    def test_llm_unavailable_produces_grounded_decision_not_technical_failure(self, monkeypatch):
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=None),
        )
        result = run("CLM-6001", _state())  # dossier favorable : identité/couverture confirmées
        assert result.decision is not ClaimDecisionV2.TECHNICAL_FAILURE
        assert result.decision is ClaimDecisionV2.APPROVE
        assert any(e.code == "LLM_UNAVAILABLE" for e in result.errors)
        assert result.confidence <= 0.4  # confiance plafonnée basse, LLM absent

    def test_llm_unavailable_with_missing_upstream_result_still_technical_failure(self, monkeypatch):
        """Distinction Cas 11 vs Cas 12/13 : une panne LLM combinée à une
        vraie absence de résultat amont reste TECHNICAL_FAILURE (le gate
        structurel prime, LLM jamais même consulté)."""
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=None),
        )
        state = _state(document_understanding_result=None)
        result = run("CLM-6001", state)
        assert result.decision is ClaimDecisionV2.TECHNICAL_FAILURE


class TestDisagreements:
    def test_disagreement_detected_between_document_and_eligibility(self, monkeypatch):
        monkeypatch.setattr(
            "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
            Mock(return_value=_decision()),
        )
        doc_result = DocumentUnderstandingResult(
            case_id="CLM-6001", status=VerificationStatus.FAIL, llm_trace=_llm_trace()
        )
        state = _state(
            document_understanding_result=doc_result,
            eligibility_result=_eligibility(VerificationStatus.PASS, VerificationStatus.PASS),
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
