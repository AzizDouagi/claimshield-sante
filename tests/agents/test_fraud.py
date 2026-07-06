"""Suite ciblée — Fraud Detection Agent — ClaimShield Santé.

Complète (sans les remplacer) les suites exhaustives
``test_fraud_detection_agent.py``/``test_fraud_detection_schemas.py`` par un
jeu de scénarios resserré, pensé pour être exécuté seul :

    pytest tests/agents/test_clinical_consistency.py tests/agents/test_fraud.py -q

Couvre : doublon exact, quasi-doublon, preuve manquante, score jamais opaque
(toujours attribué à des signaux avec preuve), sortie LLM invalide,
``llm_trace`` obligatoire, ``evidence_ids`` jamais inventés, validation
Pydantic stricte (``extra='forbid'``).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agents.fraud_detection_agent import agent
from agents.fraud_detection_agent.schemas import LlmFraudDecision
from agents.privacy_agent.schemas import FraudView
from schemas.domain import VerificationStatus
from schemas.results import LlmMetadata
from services.duplicate_index import DuplicateIndex


# ── Fixtures locales ──────────────────────────────────────────────────────────


def _fraud_view(**overrides) -> FraudView:
    payload = {
        "patient_pseudonym": "PAT-AAAAAAAAAAAA",
        "document_hashes": {"invoice": "a" * 64},
        "amount_requested": "100.00",
        "service_date": "2024-01-15",
        "invoice_reference": "***1234",
    }
    payload.update(overrides)
    return FraudView(**payload)


# ── Doublon exact ─────────────────────────────────────────────────────────────


class TestExactDuplicate:
    def test_identical_document_hash_is_flagged(self):
        index = DuplicateIndex()
        agent.run("CLM-0001", fraud_view=_fraud_view(document_hashes={"invoice": "a" * 64}), duplicate_index=index)

        result = agent.run(
            "CLM-0002",
            fraud_view=_fraud_view(document_hashes={"invoice": "a" * 64}),
            duplicate_index=index,
        )

        assert result.result_payload.duplicate_invoice is True
        assert any(s.signal_type == "EXACT_DUPLICATE_INVOICE" for s in result.result_payload.signals)

    def test_exact_duplicate_signal_is_critical_and_attributed(self):
        index = DuplicateIndex()
        agent.run("CLM-0001", fraud_view=_fraud_view(), duplicate_index=index)
        result = agent.run(
            "CLM-0002", fraud_view=_fraud_view(document_hashes={"invoice": "a" * 64}), duplicate_index=index
        )
        signal = next(s for s in result.result_payload.signals if s.signal_type == "EXACT_DUPLICATE_INVOICE")
        assert signal.severity.value == "CRITICAL"
        assert len(signal.evidence) >= 1


# ── Quasi-doublon ─────────────────────────────────────────────────────────────


class TestNearDuplicate:
    def test_close_amount_date_and_description_same_patient_is_flagged(self):
        index = DuplicateIndex()
        agent.run(
            "CLM-0001",
            fraud_view=_fraud_view(
                document_hashes={"invoice": "a" * 64}, amount_requested="100.00", service_date="2024-01-15"
            ),
            duplicate_index=index,
        )

        result = agent.run(
            "CLM-0002",
            fraud_view=_fraud_view(
                document_hashes={"invoice": "b" * 64}, amount_requested="100.50", service_date="2024-01-16"
            ),
            duplicate_index=index,
        )

        assert result.result_payload.duplicate_invoice is True
        assert any(s.signal_type == "NEAR_DUPLICATE_INVOICE" for s in result.result_payload.signals)

    def test_different_patient_is_never_a_near_duplicate(self):
        index = DuplicateIndex()
        agent.run(
            "CLM-0001",
            fraud_view=_fraud_view(patient_pseudonym="PAT-AAAAAAAAAAAA"),
            duplicate_index=index,
        )
        result = agent.run(
            "CLM-0002",
            fraud_view=_fraud_view(
                patient_pseudonym="PAT-CCCCCCCCCCCC", document_hashes={"invoice": "b" * 64}
            ),
            duplicate_index=index,
        )
        assert result.result_payload.duplicate_invoice is False


# ── Preuve manquante ──────────────────────────────────────────────────────────


class TestMissingEvidenceRejected:
    def test_fraud_signal_without_evidence_is_rejected(self):
        """Un signal de fraude ne peut jamais être une affirmation non
        appuyée — refusé par le schéma, jamais un signal fantôme."""
        from schemas.results import FraudSignal

        with pytest.raises(ValidationError):
            FraudSignal(signal_type="X", description="d", risk_contribution=0.4, evidence=[])

    def test_fraud_signal_requires_evidence_field_present(self):
        from schemas.results import FraudSignal

        with pytest.raises(ValidationError):
            FraudSignal(signal_type="X", description="d", risk_contribution=0.4)


# ── Score jamais opaque ───────────────────────────────────────────────────────


class TestScoreNeverOpaque:
    def test_every_signal_contributing_to_risk_score_has_evidence(self):
        from agents.fraud_detection_agent.agent import (
            _check_duplicate,
            _collect_signals,
        )

        identity_coverage = None
        signals, _ = _collect_signals(identity_coverage, None, None)
        signal, _, _ = _check_duplicate(
            "CLM-0001", _fraud_view(), None, DuplicateIndex()
        )
        all_signals = signals + ([signal] if signal else [])
        for s in all_signals:
            assert s.evidence, f"{s.signal_type} sans preuve — score opaque interdit"

    def test_risk_score_matches_sum_of_signal_contributions(self):
        result = agent.run(
            "CLM-0001",
            fraud_view=_fraud_view(document_hashes={"invoice": "a" * 64}),
            duplicate_index=DuplicateIndex(),
        )
        expected = min(1.0, sum(s.risk_contribution for s in result.result_payload.signals))
        assert result.result_payload.risk_score == pytest.approx(expected)


# ── Sortie LLM invalide ───────────────────────────────────────────────────────


class TestInvalidLlmOutput:
    def test_llm_decision_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            LlmFraudDecision(rationale="x", is_fraud=True)

    def test_llm_decision_rejects_accusatory_language(self):
        with pytest.raises(ValidationError):
            LlmFraudDecision(rationale="Fraude confirmée sur ce dossier.")

    def test_llm_unavailable_or_invalid_falls_back_fail_closed(self):
        with patch.object(agent, "_invoke_llm_fraud", return_value=None):
            result = agent.run("CLM-0001")
        assert len(result.errors) == 1
        assert result.errors[0].code == "LLM_UNAVAILABLE"
        assert any("LLM indisponible" in r for r in result.result_payload.reasons)


# ── llm_trace obligatoire ─────────────────────────────────────────────────────


class TestLlmTrace:
    def test_llm_trace_is_always_present_and_non_null(self):
        result = agent.run("CLM-0001")
        assert result.llm_trace is not None
        assert result.llm_trace.model_name
        assert result.llm_trace.prompt_version

    def test_llm_trace_is_a_required_non_optional_field(self):
        from schemas.results import FraudDetectionResult

        assert FraudDetectionResult.model_fields["llm_trace"].is_required()
        with pytest.raises(ValidationError):
            FraudDetectionResult(case_id="CLM-0001", status=VerificationStatus.PASS)


# ── evidence_ids jamais inventés ─────────────────────────────────────────────


class TestEvidenceIds:
    def test_evidence_ids_reference_only_real_evidence(self):
        index = DuplicateIndex()
        agent.run("CLM-0001", fraud_view=_fraud_view(), duplicate_index=index)
        result = agent.run(
            "CLM-0002", fraud_view=_fraud_view(document_hashes={"invoice": "a" * 64}), duplicate_index=index
        )
        real_ids = {e.evidence_id for s in result.result_payload.signals for e in s.evidence}
        assert result.evidence_ids
        assert set(result.evidence_ids) <= real_ids

    def test_invented_evidence_id_is_rejected_by_schema(self):
        from schemas.results import FraudDetectionResult

        with pytest.raises(ValidationError):
            FraudDetectionResult(
                case_id="CLM-0001",
                status=VerificationStatus.PASS,
                llm_trace=LlmMetadata(model_name="test-llm", prompt_version="test"),
                evidence_ids=["EVID-invented"],
            )


# ── Validation Pydantic stricte ────────────────────────────────────────────────


class TestPydanticValidation:
    def test_result_forbids_unknown_fields(self):
        from schemas.results import FraudDetectionResult

        with pytest.raises(ValidationError):
            FraudDetectionResult(
                case_id="CLM-0001",
                status=VerificationStatus.PASS,
                llm_trace=LlmMetadata(model_name="test-llm", prompt_version="test"),
                unexpected=True,
            )

    def test_result_round_trips_through_dump_and_validate(self):
        from schemas.results import FraudDetectionResult

        result = agent.run("CLM-0001")
        restored = FraudDetectionResult.model_validate(result.model_dump())
        assert restored == result
