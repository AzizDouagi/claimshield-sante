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

from agents.medical_risk_agent.agent import _evidence_completeness, node, run
from agents.medical_risk_agent.schemas import LlmMedicalRiskDecision
from schemas.domain import VerificationStatus
from schemas.v2_results import EvidenceCompleteness, RiskLevel
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
        assert result.result_payload.evidence_completeness is EvidenceCompleteness.INSUFFICIENT


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


class TestEvidenceCompletenessVsRisk:
    """Correctif post-mesure V2-10 (AZIZ) : `UNRESOLVED_CODING`/données
    manquantes ne doivent jamais, seules, produire un `risk_level` HIGH/
    CRITICAL — uniquement dégrader `evidence_completeness`. Un signal de
    danger réel (identité non concordante confirmée, doublon exact) doit
    au contraire continuer à produire HIGH/CRITICAL, sans changement de
    poids par rapport à V1 (`agents.fraud_detection_agent.agent._collect_signals`,
    non modifiée)."""

    def _clean_identity_coverage(self) -> SimpleNamespace:
        return SimpleNamespace(
            identity=SimpleNamespace(status=VerificationStatus.PASS),
            coverage=SimpleNamespace(
                status=VerificationStatus.PASS,
                ceiling_exceeded=False,
                preauthorization_required=False,
            ),
        )

    def test_unresolved_coding_alone_never_reaches_high_or_critical(self, monkeypatch):
        monkeypatch.setattr(
            "agents.medical_risk_agent.agent._invoke_llm_medical_risk",
            Mock(return_value=_decision()),
        )
        ocr_result = SimpleNamespace(
            extracted_fields={}, confidence_score=0.95, document_type=None, sha256=None
        )
        result = run(
            case_id="CLM-5050",
            procedures=[],
            medications=[],
            ocr_result=ocr_result,
            identity_coverage_result=self._clean_identity_coverage(),
        )
        signal_types = {s.signal_type for s in result.result_payload.fraud_signals}
        assert "UNRESOLVED_CODING" in signal_types
        assert result.result_payload.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM)
        assert result.result_payload.risk_score == 0.0  # aucun signal de risque réel

    def test_missing_procedure_evidence_reduces_completeness_not_risk(self, monkeypatch):
        monkeypatch.setattr(
            "agents.medical_risk_agent.agent._invoke_llm_medical_risk",
            Mock(return_value=_decision()),
        )
        result = run(
            case_id="CLM-5051",
            procedures=[],
            medications=[],
            identity_coverage_result=self._clean_identity_coverage(),
        )
        assert result.result_payload.evidence_completeness is EvidenceCompleteness.INSUFFICIENT
        assert result.result_payload.risk_level is not RiskLevel.HIGH
        assert result.result_payload.risk_level is not RiskLevel.CRITICAL

    def test_confirmed_identity_mismatch_reaches_high_risk(self, monkeypatch):
        monkeypatch.setattr(
            "agents.medical_risk_agent.agent._invoke_llm_medical_risk",
            Mock(return_value=_decision()),
        )
        identity_like = SimpleNamespace(
            identity=SimpleNamespace(status=VerificationStatus.FAIL),
            coverage=SimpleNamespace(
                status=VerificationStatus.PASS, ceiling_exceeded=False, preauthorization_required=False
            ),
        )
        result = run(case_id="CLM-5052", identity_coverage_result=identity_like)
        assert result.result_payload.risk_level is RiskLevel.HIGH

    def test_exact_duplicate_forces_critical_regardless_of_score(self, monkeypatch):
        monkeypatch.setattr(
            "agents.medical_risk_agent.agent._invoke_llm_medical_risk",
            Mock(return_value=_decision()),
        )
        from agents.privacy_agent.schemas import FraudView

        index = DuplicateIndex()
        fraud_view = FraudView(
            patient_pseudonym="PAT-ABCDEF123456",
            document_hashes={"invoice": "b" * 64},
            amount_requested="100.00",
            service_date="2026-01-10",
        )
        run(case_id="CLM-5053", fraud_view=fraud_view, duplicate_index=index)
        result = run(case_id="CLM-5054", fraud_view=fraud_view, duplicate_index=index)
        assert result.result_payload.risk_level is RiskLevel.CRITICAL

    def test_evidence_completeness_partial_with_single_completeness_signal(self, monkeypatch):
        monkeypatch.setattr(
            "agents.medical_risk_agent.agent._invoke_llm_medical_risk",
            Mock(return_value=_decision()),
        )
        # Un seul signal de complétude (confiance OCR faible), coding/identité/
        # couverture propres — jamais d'absence structurelle (procedures fourni).
        ocr_result = SimpleNamespace(
            extracted_fields={}, confidence_score=0.2, document_type=None, sha256=None
        )
        result = run(
            case_id="CLM-5055",
            procedures=["Consultation dentaire"],
            ocr_result=ocr_result,
            identity_coverage_result=self._clean_identity_coverage(),
        )
        assert result.result_payload.evidence_completeness is EvidenceCompleteness.PARTIAL
        assert result.result_payload.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM)

    def test_two_completeness_signals_stay_partial_not_insufficient(self, monkeypatch):
        """Régression V2-10 (seuil assoupli, décision AZIZ sur mesure réelle) :
        le cas le plus fréquent sur les fixtures réelles — nom patient absent
        de l'OCR (`IDENTITY_AMBIGUOUS`) + code médicament en correspondance
        approximative (`UNRESOLVED_CODING`) — doit rester `PARTIAL` (LLM
        consulté, `APPROVE` atteignable), jamais `INSUFFICIENT` (`REQUEST_MORE_INFO`
        forcé sans consultation du LLM). Avant l'assouplissement (seuil `>= 2`),
        ces deux mêmes signaux forçaient `INSUFFICIENT` à tort."""
        monkeypatch.setattr(
            "agents.medical_risk_agent.agent._invoke_llm_medical_risk",
            Mock(return_value=_decision()),
        )
        identity_ambiguous = SimpleNamespace(
            identity=SimpleNamespace(status=VerificationStatus.NEEDS_REVIEW),
            coverage=SimpleNamespace(
                status=VerificationStatus.PASS, ceiling_exceeded=False, preauthorization_required=False
            ),
        )
        # Un acte fourni (jamais d'absence structurelle) dont le code ne sera
        # pas résolu exactement (référentiel local sans correspondance) —
        # produit UNRESOLVED_CODING côté fraude, en plus d'IDENTITY_AMBIGUOUS.
        result = run(
            case_id="CLM-5056",
            procedures=["Acte non référencé xyz123"],
            identity_coverage_result=identity_ambiguous,
        )
        signal_types = {s.signal_type for s in result.result_payload.fraud_signals}
        assert "IDENTITY_AMBIGUOUS" in signal_types
        assert "UNRESOLVED_CODING" in signal_types
        assert len(signal_types & {"IDENTITY_AMBIGUOUS", "PREAUTHORIZATION_MISSING", "UNRESOLVED_CODING", "LOW_EXTRACTION_CONFIDENCE"}) == 2
        assert result.result_payload.evidence_completeness is EvidenceCompleteness.PARTIAL

    def test_evidence_completeness_helper_threshold_boundaries(self):
        """Test unitaire direct de `_evidence_completeness` — seuil `>= 3`."""
        assert (
            _evidence_completeness(completeness_signals=[], structural_absence=False)
            is EvidenceCompleteness.COMPLETE
        )
        assert (
            _evidence_completeness(completeness_signals=[object()], structural_absence=False)
            is EvidenceCompleteness.PARTIAL
        )
        assert (
            _evidence_completeness(completeness_signals=[object(), object()], structural_absence=False)
            is EvidenceCompleteness.PARTIAL
        )
        assert (
            _evidence_completeness(
                completeness_signals=[object(), object(), object()], structural_absence=False
            )
            is EvidenceCompleteness.INSUFFICIENT
        )
        assert (
            _evidence_completeness(completeness_signals=[], structural_absence=True)
            is EvidenceCompleteness.INSUFFICIENT
        )


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
