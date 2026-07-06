"""Suite ciblée — Clinical Consistency Agent — ClaimShield Santé.

Complète (sans les remplacer) les suites exhaustives
``test_clinical_consistency_agent.py``/``test_clinical_consistency_schemas.py``
par un jeu de scénarios resserré, pensé pour être exécuté seul :

    pytest tests/agents/test_clinical_consistency.py tests/agents/test_fraud.py -q

Couvre : dates impossibles, acte absent, preuve manquante, score jamais
opaque (toujours attribué à des signaux avec preuve), sortie LLM invalide,
``llm_trace`` obligatoire, ``evidence_ids`` jamais inventés, validation
Pydantic stricte (``extra='forbid'``).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agents.clinical_consistency_agent import agent
from agents.clinical_consistency_agent.schemas import LlmClinicalDecision
from schemas.domain import DocumentType, ExtractionStatus, OcrSource, VerificationStatus
from schemas.results import DocumentOcrResult, ExtractedField, LlmMetadata, MedicalCodingResult, ProcedureCoding


# ── Fixtures locales ──────────────────────────────────────────────────────────


def _ocr_result(**fields: str) -> DocumentOcrResult:
    extracted = {
        name: ExtractedField(field_name=name, value=value, confidence=0.9)
        for name, value in fields.items()
    }
    return DocumentOcrResult(
        claim_id="CLM-0001",
        file_path="facture.pdf",
        sha256="a" * 64,
        mime_type="application/pdf",
        extraction_status=ExtractionStatus.SUCCESS,
        status=VerificationStatus.PASS,
        document_type=DocumentType.INVOICE,
        ocr_source=OcrSource.PDF_TEXT,
        extracted_fields=extracted,
    )


def _coding_result(count: int, status: VerificationStatus = VerificationStatus.PASS) -> MedicalCodingResult:
    codings = [
        ProcedureCoding(original_description=f"acte {i}", proposed_code=f"C{i}", status=VerificationStatus.PASS)
        for i in range(count)
    ]
    return MedicalCodingResult(
        case_id="CLM-0001",
        status=status,
        codings=codings,
        llm_metadata=LlmMetadata(model_name="test-llm", prompt_version="test"),
    )


# ── Dates impossibles ─────────────────────────────────────────────────────────


class TestImpossibleDates:
    def test_invalid_calendar_date_produces_impossible_date_signal(self):
        """Un 30 février n'existe pas : signalé, jamais accepté silencieusement."""
        ocr = _ocr_result(procedure_count="1", care_date="2024-02-30", service_date="2024-01-10")
        result = agent.run("CLM-0001", ocr_result=ocr, coding_result=_coding_result(1))
        assert any(s.signal_type == "IMPOSSIBLE_DATE" for s in result.result_payload.signals)

    def test_ambiguous_date_format_produces_impossible_date_signal(self):
        ocr = _ocr_result(procedure_count="1", care_date="03/04/2024", service_date="2024-01-10")
        result = agent.run("CLM-0001", ocr_result=ocr, coding_result=_coding_result(1))
        assert any(s.signal_type == "IMPOSSIBLE_DATE" for s in result.result_payload.signals)

    def test_impossible_date_signal_is_critical_and_attributed(self):
        ocr = _ocr_result(procedure_count="1", care_date="2024-02-30", service_date="2024-01-10")
        result = agent.run("CLM-0001", ocr_result=ocr, coding_result=_coding_result(1))
        signal = next(s for s in result.result_payload.signals if s.signal_type == "IMPOSSIBLE_DATE")
        assert signal.severity.value == "CRITICAL"
        assert len(signal.evidence) >= 1
        assert all(e.evidence_id for e in signal.evidence)


# ── Acte absent ───────────────────────────────────────────────────────────────


class TestMissingProcedure:
    def test_dated_care_without_any_coded_procedure_is_signaled(self):
        ocr = _ocr_result(procedure_count="0", care_date="2024-01-10", service_date="2024-01-10")
        result = agent.run("CLM-0001", ocr_result=ocr, coding_result=_coding_result(0))
        assert any(
            s.signal_type == "MISSING_PROCEDURE_EVIDENCE" for s in result.result_payload.signals
        )

    def test_absence_of_coded_procedure_never_invented_when_coding_unavailable(self):
        """Codification non encore disponible (``coding_result=None``) :
        jamais un signal fabriqué à partir d'une hypothèse."""
        ocr = _ocr_result(procedure_count="1", care_date="2024-01-10", service_date="2024-01-10")
        result = agent.run("CLM-0001", ocr_result=ocr, coding_result=None)
        assert not any(
            s.signal_type == "MISSING_PROCEDURE_EVIDENCE" for s in result.result_payload.signals
        )


# ── Preuve manquante ──────────────────────────────────────────────────────────


class TestMissingEvidenceRejected:
    def test_clinical_signal_without_any_reference_is_rejected(self):
        """Un signal sans champ ni document comparé est une sortie libre non
        structurée — refusée par le schéma, jamais acceptée."""
        from schemas.results import ClinicalSignal

        with pytest.raises(ValidationError):
            ClinicalSignal(signal_type="VAGUE", description="Quelque chose ne va pas.")

    def test_clinical_inconsistency_without_evidence_is_rejected(self):
        from schemas.results import ClinicalInconsistency

        with pytest.raises(ValidationError):
            ClinicalInconsistency(
                inconsistency_type="X", expected="a", observed="b", evidence=[]
            )


# ── Score jamais opaque ───────────────────────────────────────────────────────


class TestScoreNeverOpaque:
    def test_every_signal_contributing_to_status_has_evidence(self):
        """La confiance/le statut ne sont jamais un nombre nu : chaque signal
        qui les influence porte au moins une preuve attribuée."""
        ocr = _ocr_result(medication_count="2", care_date="2024-01-10", service_date="2024-01-10")
        result = agent.run("CLM-0001", ocr_result=ocr)
        assert result.result_payload.signals
        for signal in result.result_payload.signals:
            assert signal.evidence, f"{signal.signal_type} sans preuve — score opaque interdit"

    def test_confidence_decreases_with_signal_count_never_a_bare_default(self):
        clean_ocr = _ocr_result(procedure_count="1", service_date="2024-01-15")
        clean_result = agent.run("CLM-0001", ocr_result=clean_ocr, coding_result=_coding_result(1))

        noisy_ocr = _ocr_result(medication_count="2", care_date="2024-01-10", service_date="2024-01-10")
        noisy_result = agent.run("CLM-0001", ocr_result=noisy_ocr)

        assert noisy_result.confidence < clean_result.confidence


# ── Sortie LLM invalide ───────────────────────────────────────────────────────


class TestInvalidLlmOutput:
    def test_llm_decision_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            LlmClinicalDecision(clinical_context="x", diagnosis="grippe")

    def test_llm_decision_rejects_free_text_reasons(self):
        with pytest.raises(ValidationError):
            LlmClinicalDecision(reasons="un seul motif en texte libre")

    def test_llm_unavailable_or_invalid_falls_back_fail_closed(self):
        """LLM indisponible/réponse invalide : statut déterministe conservé,
        erreur structurée ajoutée à l'enveloppe — jamais un succès inventé."""
        with patch.object(agent, "_invoke_llm_clinical", return_value=None):
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
        from schemas.results import ClinicalConsistencyResult

        assert ClinicalConsistencyResult.model_fields["llm_trace"].is_required()
        with pytest.raises(ValidationError):
            ClinicalConsistencyResult(case_id="CLM-0001", status=VerificationStatus.PASS)


# ── evidence_ids jamais inventés ─────────────────────────────────────────────


class TestEvidenceIds:
    def test_evidence_ids_reference_only_real_evidence(self):
        ocr = _ocr_result(medication_count="2", care_date="2024-01-10", service_date="2024-01-10")
        result = agent.run("CLM-0001", ocr_result=ocr)
        real_ids = {
            e.evidence_id for s in result.result_payload.signals for e in s.evidence
        } | {
            e.evidence_id for i in result.result_payload.inconsistencies for e in i.evidence
        }
        assert result.evidence_ids
        assert set(result.evidence_ids) <= real_ids

    def test_invented_evidence_id_is_rejected_by_schema(self):
        from schemas.results import ClinicalConsistencyResult

        with pytest.raises(ValidationError):
            ClinicalConsistencyResult(
                case_id="CLM-0001",
                status=VerificationStatus.PASS,
                llm_trace=LlmMetadata(model_name="test-llm", prompt_version="test"),
                evidence_ids=["EVID-invented"],
            )


# ── Validation Pydantic stricte ────────────────────────────────────────────────


class TestPydanticValidation:
    def test_result_forbids_unknown_fields(self):
        from schemas.results import ClinicalConsistencyResult

        with pytest.raises(ValidationError):
            ClinicalConsistencyResult(
                case_id="CLM-0001",
                status=VerificationStatus.PASS,
                llm_trace=LlmMetadata(model_name="test-llm", prompt_version="test"),
                unexpected=True,
            )

    def test_result_round_trips_through_dump_and_validate(self):
        from schemas.results import ClinicalConsistencyResult

        result = agent.run("CLM-0001")
        restored = ClinicalConsistencyResult.model_validate(result.model_dump())
        assert restored == result
