"""Tests de agents/medical_risk_agent (V2) — Phase V2-5.

Fusion de medical_coding_agent + clinical_consistency_agent +
fraud_detection_agent (V1) en un seul agent, une seule Phase A combinée,
un seul appel LLM. Ces tests couvrent la logique déterministe et le
comportement fail-closed — la mesure du critère (a)/(c) (divergence vs
les 3 agents V1 séparés, sur les 37 fixtures réelles) est effectuée
séparément par `scripts/evaluate_medical_risk_fusion.py`, avec un vrai
LLM (Ollama), hors suite pytest.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from agents.medical_risk_agent.agent import node, run
from agents.medical_risk_agent.schemas import LlmMedicalRiskDecision
from schemas.domain import VerificationStatus
from schemas.v2_results import RiskLevel
from services.duplicate_index import DuplicateIndex


def _decision(**overrides) -> LlmMedicalRiskDecision:
    defaults = {
        "coding_resolved": [],
        "coding_rationale": "",
        "clinical_context": "",
        "clinical_severity_assessments": [],
        "clinical_referenced_evidence_ids": [],
        "clinical_acknowledged_inconsistencies": [],
        "fraud_rationale": "",
        "fraud_signal_assessments": [],
        "fraud_referenced_signal_types": [],
        "reasons": [],
    }
    defaults.update(overrides)
    return LlmMedicalRiskDecision(**defaults)


class TestNoUpstreamData:
    def test_no_inputs_at_all_needs_review_never_crashes(self, monkeypatch):
        monkeypatch.setattr(
            "agents.medical_risk_agent.agent._invoke_llm_medical_risk",
            Mock(return_value=_decision()),
        )
        result = run(case_id="CLM-5001")
        assert result.status is not VerificationStatus.PASS
        assert result.result_payload.risk_level is RiskLevel.LOW
        assert result.result_payload.codings == []


class TestCodingFusion:
    def test_exact_match_procedure_pass(self, monkeypatch):
        monkeypatch.setattr(
            "agents.medical_risk_agent.agent._invoke_llm_medical_risk",
            Mock(return_value=_decision()),
        )
        result = run(case_id="CLM-5002", procedures=["Consultation dentaire"])
        codings = result.result_payload.codings
        assert len(codings) == 1
        assert codings[0].status is VerificationStatus.PASS
        assert codings[0].proposed_code == "34043003"

    def test_llm_can_resolve_needs_review_coding_from_bounded_candidates(self, monkeypatch):
        # "Consultation generale" produit un palier fuzzy_candidates_found (NEEDS_REVIEW)
        from tools.medical_coding import code_procedures

        candidates = code_procedures(["Consultation generale"])[0].alternatives
        assert candidates, "le test suppose au moins un candidat flou proposé"
        monkeypatch.setattr(
            "agents.medical_risk_agent.agent._invoke_llm_medical_risk",
            Mock(
                return_value=_decision(
                    coding_resolved=[
                        {
                            "description": "Consultation generale",
                            "proposed_code": candidates[0],
                            "rationale": "Correspondance la plus proche.",
                        }
                    ]
                )
            ),
        )
        result = run(case_id="CLM-5003", procedures=["Consultation generale"])
        coding = result.result_payload.codings[0]
        # Même garantie que V1 : un candidat flou confirmé reste NEEDS_REVIEW,
        # jamais un PASS automatique.
        assert coding.status is VerificationStatus.NEEDS_REVIEW
        assert coding.rule_applied == "fuzzy_match_llm_selected"


class TestClinicalFusion:
    def test_missing_prescription_reference_signal(self, monkeypatch):
        monkeypatch.setattr(
            "agents.medical_risk_agent.agent._invoke_llm_medical_risk",
            Mock(return_value=_decision()),
        )
        ocr_result = SimpleNamespace(
            extracted_fields={
                "medication_count": "2",
                "procedure_count": "0",
                "service_date": "2026-01-10",
            },
            confidence_score=0.9,
            document_type=None,
            sha256=None,
        )
        result = run(case_id="CLM-5010", ocr_result=ocr_result)
        signal_types = {s.signal_type for s in result.result_payload.clinical_signals}
        assert "MISSING_PRESCRIPTION_REFERENCE" in signal_types
        assert result.status is VerificationStatus.FAIL  # signal CRITICAL

    def test_llm_severity_adjustment_bounded_to_one_notch(self, monkeypatch):
        ocr_result = SimpleNamespace(
            extracted_fields={
                "medication_count": "2",
                "procedure_count": "0",
                "service_date": "2026-01-10",
            },
            confidence_score=0.9,
            document_type=None,
            sha256=None,
        )
        # CRITICAL -> LOW est un écart de 3 crans, hors borne : ignoré.
        monkeypatch.setattr(
            "agents.medical_risk_agent.agent._invoke_llm_medical_risk",
            Mock(
                return_value=_decision(
                    clinical_severity_assessments=[
                        {
                            "signal_type": "MISSING_PRESCRIPTION_REFERENCE",
                            "severity_override": "LOW",
                            "rationale": "tentative hors borne",
                        }
                    ]
                )
            ),
        )
        result = run(case_id="CLM-5011", ocr_result=ocr_result)
        assert result.status is VerificationStatus.FAIL  # sévérité inchangée, toujours CRITICAL


class TestFraudFusion:
    def test_duplicate_detection_produces_signal(self, monkeypatch):
        monkeypatch.setattr(
            "agents.medical_risk_agent.agent._invoke_llm_medical_risk",
            Mock(return_value=_decision()),
        )
        from agents.privacy_agent.schemas import FraudView

        index = DuplicateIndex()
        fraud_view = FraudView(
            patient_pseudonym="PAT-ABCDEF123456",
            document_hashes={"invoice": "a" * 64},
            amount_requested="100.00",
            service_date="2026-01-10",
        )
        # Premier passage : enregistre l'empreinte.
        run(case_id="CLM-5020", fraud_view=fraud_view, duplicate_index=index)
        # Second passage, même empreinte (même patient/montant/hash) : doublon exact détecté.
        result = run(case_id="CLM-5021", fraud_view=fraud_view, duplicate_index=index)
        signal_types = {s.signal_type for s in result.result_payload.fraud_signals}
        assert "EXACT_DUPLICATE_INVOICE" in signal_types
        assert result.result_payload.duplicate_invoice is True

    def test_signal_never_without_evidence(self, monkeypatch):
        monkeypatch.setattr(
            "agents.medical_risk_agent.agent._invoke_llm_medical_risk",
            Mock(return_value=_decision()),
        )
        identity_like = SimpleNamespace(
            identity=SimpleNamespace(status=VerificationStatus.FAIL),
            coverage=SimpleNamespace(status=VerificationStatus.PASS, ceiling_exceeded=False, preauthorization_required=False),
        )
        result = run(case_id="CLM-5022", identity_coverage_result=identity_like)
        for signal in result.result_payload.fraud_signals:
            assert len(signal.evidence) >= 1


class TestLlmFailClosed:
    def test_llm_unavailable_keeps_deterministic_results(self, monkeypatch):
        monkeypatch.setattr(
            "agents.medical_risk_agent.agent._invoke_llm_medical_risk",
            Mock(return_value=None),
        )
        result = run(case_id="CLM-5030", procedures=["Consultation dentaire"])
        assert result.result_payload.codings[0].status is VerificationStatus.PASS
        assert any(e.code == "LLM_UNAVAILABLE" for e in result.errors)
        assert any("indisponible" in r for r in result.result_payload.reasons)


class TestNodeIntegration:
    def test_node_updates_state(self, monkeypatch):
        monkeypatch.setattr(
            "agents.medical_risk_agent.agent._invoke_llm_medical_risk",
            Mock(return_value=_decision()),
        )
        state = {
            "case_id": "CLM-5040",
            "schema_version": "2.0.0",
            "current_step": "eligibility",
            "completed_steps": ["intake_safety", "document_understanding", "eligibility"],
        }
        updates = node(state)  # type: ignore[arg-type]
        assert updates["current_step"] == "medical_risk"
        assert updates["medical_risk_result"] is not None
