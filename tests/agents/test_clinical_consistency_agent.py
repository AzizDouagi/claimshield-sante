"""Tests du Clinical Consistency Agent — ClaimShield Santé (étape 12).

Couvre :
  - Phase A déterministe : signaux, compteurs, statut PASS/NEEDS_REVIEW/FAIL,
    chronologie ordonnance/soin et acte absent (``tools.date_checks``).
  - Phase B agent ReAct LLM (appel obligatoire, outil autorisé
    ``verifier_chronologie``) : contexte explicatif, jamais d'autorité sur
    le statut, jamais de diagnostic/décision finale/affirmation non prouvée.
  - Fallback LLM indisponible : statut déterministe conservé.
  - Résumés minimisés transmis au LLM : FHIR (``_fhir_summary``) et vue
    médicale (``_medical_view_summary``/``_extract_medical_view``, déjà
    pseudonymisée par ``privacy_agent`` — jamais reconstruite ici).
  - node() : mise à jour du state, audit_trail, errors/alerts.
  - Interface injectable préservée (ClinicalConsistencyRunnable, make_node).
  - Enveloppe générique (llm_trace, confidence, evidence_ids,
    human_review_required, result_payload) introduite pour rendre la trace
    LLM obligatoire et interdire tout contenu brut/OCR complet/prompt complet.
"""
from __future__ import annotations

from unittest.mock import patch

from agents.clinical_consistency_agent import agent
from agents.clinical_consistency_agent.schemas import ClinicalSignalAssessment, LlmClinicalDecision
from agents.privacy_agent.schemas import MedicalView
from schemas.domain import (
    DataClassification,
    DocumentType,
    ExtractionStatus,
    OcrSource,
    SeverityLevel,
    VerificationStatus,
)
from schemas.results import (
    AuditEvent,
    ClinicalResultPayload,
    DocumentOcrResult,
    ExtractedField,
    FhirValidatorResult,
    LlmMetadata,
    MedicalCodingResult,
    PrivacyResult,
    ProcedureCoding,
)


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
        ProcedureCoding(
            original_description=f"acte {i}",
            proposed_code=f"C{i}",
            status=VerificationStatus.PASS,
        )
        for i in range(count)
    ]
    return MedicalCodingResult(
        case_id="CLM-0001",
        status=status,
        codings=codings,
        llm_metadata=LlmMetadata(model_name="test-llm", prompt_version="test"),
    )


# ── Phase A — pas de données ────────────────────────────────────────────────


class TestNoUpstreamData:
    def test_no_ocr_no_coding_is_needs_review(self):
        result = agent.run("CLM-0001")
        assert result.status is VerificationStatus.NEEDS_REVIEW

    def test_no_ocr_no_coding_has_no_signals(self):
        result = agent.run("CLM-0001")
        assert result.result_payload.signals == []

    def test_no_ocr_no_coding_explains_reason(self):
        result = agent.run("CLM-0001")
        assert any("non vérifiable" in r for r in result.result_payload.reasons)


# ── Phase A — cas cohérent ───────────────────────────────────────────────────


class TestConsistentCase:
    def test_matching_counts_no_medication_is_pass(self):
        ocr = _ocr_result(procedure_count="2", service_date="2024-01-15")
        coding = _coding_result(2)
        result = agent.run("CLM-0001", ocr_result=ocr, coding_result=coding)
        assert result.status is VerificationStatus.PASS
        assert result.result_payload.signals == []

    def test_medication_with_prescription_no_signal(self):
        ocr = _ocr_result(
            procedure_count="1",
            medication_count="2",
            prescription_number="RX-123",
            service_date="2024-01-15",
        )
        coding = _coding_result(1)
        result = agent.run("CLM-0001", ocr_result=ocr, coding_result=coding)
        assert result.status is VerificationStatus.PASS
        assert result.result_payload.prescription_required is True

    def test_prescription_required_false_when_medication_count_zero(self):
        ocr = _ocr_result(procedure_count="1", medication_count="0", service_date="2024-01-15")
        coding = _coding_result(1)
        result = agent.run("CLM-0001", ocr_result=ocr, coding_result=coding)
        assert result.result_payload.prescription_required is False

    def test_counts_populated_from_ocr(self):
        ocr = _ocr_result(procedure_count="3", medication_count="1", service_date="2024-01-15")
        coding = _coding_result(3)
        result = agent.run("CLM-0001", ocr_result=ocr, coding_result=coding)
        assert result.result_payload.procedure_count == 3
        assert result.result_payload.medication_count == 1


# ── Phase A — signaux critiques ──────────────────────────────────────────────


class TestCriticalSignals:
    def test_medication_without_prescription_is_fail(self):
        ocr = _ocr_result(medication_count="2", service_date="2024-01-15")
        result = agent.run("CLM-0001", ocr_result=ocr)
        assert result.status is VerificationStatus.FAIL
        assert any(
            s.signal_type == "MISSING_PRESCRIPTION_REFERENCE" for s in result.result_payload.signals
        )

    def test_zero_coded_procedures_is_fail(self):
        ocr = _ocr_result(procedure_count="4", service_date="2024-01-15")
        coding = _coding_result(0)
        result = agent.run("CLM-0001", ocr_result=ocr, coding_result=coding)
        assert result.status is VerificationStatus.FAIL
        assert any(
            s.signal_type == "PROCEDURE_CODING_COUNT_MISMATCH" for s in result.result_payload.signals
        )


# ── Phase A — signaux mineurs (NEEDS_REVIEW) ────────────────────────────────


class TestMinorSignals:
    def test_procedure_coding_mismatch_partial_is_needs_review(self):
        ocr = _ocr_result(procedure_count="5", service_date="2024-01-15")
        coding = _coding_result(3)
        result = agent.run("CLM-0001", ocr_result=ocr, coding_result=coding)
        assert result.status is VerificationStatus.NEEDS_REVIEW
        assert any(
            s.signal_type == "PROCEDURE_CODING_COUNT_MISMATCH" for s in result.result_payload.signals
        )

    def test_missing_service_date_is_needs_review(self):
        ocr = _ocr_result(procedure_count="2")
        coding = _coding_result(2)
        result = agent.run("CLM-0001", ocr_result=ocr, coding_result=coding)
        assert result.status is VerificationStatus.NEEDS_REVIEW
        assert any(s.signal_type == "MISSING_SERVICE_DATE" for s in result.result_payload.signals)

    def test_unresolved_coding_status_is_needs_review(self):
        ocr = _ocr_result(procedure_count="2", service_date="2024-01-15")
        coding = _coding_result(2, status=VerificationStatus.NEEDS_REVIEW)
        result = agent.run("CLM-0001", ocr_result=ocr, coding_result=coding)
        assert result.status is VerificationStatus.NEEDS_REVIEW
        assert any(s.signal_type == "UPSTREAM_CODING_UNRESOLVED" for s in result.result_payload.signals)


# ── Phase B — LLM ne peut jamais changer le statut ──────────────────────────


class TestLlmNeverOverridesStatus:
    def test_llm_fallback_keeps_deterministic_status(self):
        with patch.object(agent, "_invoke_llm_clinical", return_value=None):
            ocr = _ocr_result(procedure_count="2", service_date="2024-01-15")
            coding = _coding_result(2)
            result = agent.run("CLM-0001", ocr_result=ocr, coding_result=coding)
        assert result.status is VerificationStatus.PASS
        assert any("LLM indisponible" in r for r in result.result_payload.reasons)

    def test_llm_decision_has_no_status_field(self):
        """LlmClinicalDecision ne porte aucun champ de statut — garantie
        structurelle qu'aucune décision médicale ne peut venir du LLM."""
        assert "status" not in LlmClinicalDecision.model_fields
        assert "recommended_status" not in LlmClinicalDecision.model_fields

    def test_llm_context_appended_to_reasons(self):
        decision = LlmClinicalDecision(clinical_context="Contexte de test.", reasons=["motif LLM"])
        with patch.object(agent, "_invoke_llm_clinical", return_value=decision):
            result = agent.run("CLM-0001")
        assert "Contexte de test." in result.result_payload.reasons
        assert "motif LLM" in result.result_payload.reasons


# ── P1-2 — autorité LLM bornée sur la sévérité des signaux ─────────────────


class TestBoundedSeverityAssessments:
    """Le LLM ne peut jamais fixer lui-même le statut (toujours vrai, voir
    TestLlmNeverOverridesStatus) mais peut désormais proposer, pour un
    signal déjà calculé et déjà attribué à une preuve, un ajustement de
    sévérité borné à un cran maximum sur le lattice SeverityLevel — recalculé
    déterministiquement, jamais laissé au LLM."""

    def _critical_signal_ocr(self):
        # Médicament facturé sans numéro d'ordonnance → MISSING_PRESCRIPTION_REFERENCE,
        # toujours CRITICAL en Phase A → statut FAIL.
        return _ocr_result(medication_count="1", service_date="2024-01-15")

    def test_downgrade_within_bound_can_flip_fail_to_needs_review(self):
        ocr = self._critical_signal_ocr()
        with patch.object(agent, "_invoke_llm_clinical", return_value=None):
            baseline = agent.run("CLM-0001", ocr_result=ocr)
        assert baseline.status is VerificationStatus.FAIL

        decision = LlmClinicalDecision(
            severity_assessments=[
                ClinicalSignalAssessment(
                    signal_type="MISSING_PRESCRIPTION_REFERENCE",
                    severity_override="HIGH",
                    rationale="Contexte atténuant documenté.",
                )
            ]
        )
        with patch.object(agent, "_invoke_llm_clinical", return_value=decision):
            downgraded = agent.run("CLM-0001", ocr_result=ocr)

        assert downgraded.status is VerificationStatus.NEEDS_REVIEW
        assert downgraded.result_payload.signals[0].severity == SeverityLevel.HIGH

    def test_adjustment_emits_structured_log(self, caplog):
        """P3-2 : le point de décision autonome (ajustement de sévérité
        effectivement appliqué) est journalisé pour traçabilité."""
        ocr = self._critical_signal_ocr()
        decision = LlmClinicalDecision(
            severity_assessments=[
                ClinicalSignalAssessment(
                    signal_type="MISSING_PRESCRIPTION_REFERENCE",
                    severity_override="HIGH",
                    rationale="Contexte atténuant documenté.",
                )
            ]
        )
        with caplog.at_level("INFO"):
            with patch.object(agent, "_invoke_llm_clinical", return_value=decision):
                agent.run("CLM-0001", ocr_result=ocr)

        messages = [r.getMessage() for r in caplog.records]
        assert any(
            "clinical_consistency_severity_adjusted" in m and "CLM-0001" in m for m in messages
        )

    def test_adjustment_beyond_one_notch_is_ignored(self):
        """MEDIUM → CRITICAL est un écart de deux crans — hors borne,
        toujours ignoré, jamais partiellement appliqué."""
        ocr = _ocr_result(procedure_count="3", service_date="2024-01-15")
        coding = _coding_result(2)
        with patch.object(agent, "_invoke_llm_clinical", return_value=None):
            baseline = agent.run("CLM-0001", ocr_result=ocr, coding_result=coding)
        assert baseline.status is VerificationStatus.NEEDS_REVIEW
        assert baseline.result_payload.signals[0].severity == SeverityLevel.MEDIUM

        decision = LlmClinicalDecision(
            severity_assessments=[
                ClinicalSignalAssessment(
                    signal_type="PROCEDURE_CODING_COUNT_MISMATCH",
                    severity_override="CRITICAL",
                    rationale="Tentative de saut de deux crans.",
                )
            ]
        )
        with patch.object(agent, "_invoke_llm_clinical", return_value=decision):
            result = agent.run("CLM-0001", ocr_result=ocr, coding_result=coding)

        assert result.status is VerificationStatus.NEEDS_REVIEW
        assert result.result_payload.signals[0].severity == SeverityLevel.MEDIUM
        assert any("hors borne autorisée" in r for r in result.result_payload.reasons)

    def test_upgrade_within_bound_changes_severity_without_status_flip(self):
        ocr = _ocr_result(procedure_count="3", service_date="2024-01-15")
        coding = _coding_result(2)
        decision = LlmClinicalDecision(
            severity_assessments=[
                ClinicalSignalAssessment(
                    signal_type="PROCEDURE_CODING_COUNT_MISMATCH",
                    severity_override="HIGH",
                    rationale="Contexte aggravant identifié.",
                )
            ]
        )
        with patch.object(agent, "_invoke_llm_clinical", return_value=decision):
            result = agent.run("CLM-0001", ocr_result=ocr, coding_result=coding)

        assert result.result_payload.signals[0].severity == SeverityLevel.HIGH
        assert result.status is VerificationStatus.NEEDS_REVIEW  # HIGH n'est pas CRITICAL

    def test_unknown_signal_type_is_silently_ignored(self):
        ocr = self._critical_signal_ocr()
        with patch.object(agent, "_invoke_llm_clinical", return_value=None):
            baseline = agent.run("CLM-0001", ocr_result=ocr)

        decision = LlmClinicalDecision(
            severity_assessments=[
                ClinicalSignalAssessment(
                    signal_type="COMPLETELY_INVENTED_SIGNAL",
                    severity_override="LOW",
                    rationale="Signal jamais calculé par la Phase A.",
                )
            ]
        )
        with patch.object(agent, "_invoke_llm_clinical", return_value=decision):
            result = agent.run("CLM-0001", ocr_result=ocr)

        assert result.status == baseline.status
        assert len(result.result_payload.signals) == len(baseline.result_payload.signals)
        assert not any(
            s.signal_type == "COMPLETELY_INVENTED_SIGNAL" for s in result.result_payload.signals
        )

    def test_evidence_unchanged_after_adjustment(self):
        """Seule la sévérité change — la preuve attribuée reste strictement
        identique. Comparaison au sein d'une seule collecte de signaux
        (``evidence_id`` auto-généré à chaque appel de ``_collect_signals``)."""
        ocr = self._critical_signal_ocr()
        signals, _, _, _, _, _, _ = agent._collect_signals(ocr, None)
        baseline_signal = next(s for s in signals if s.signal_type == "MISSING_PRESCRIPTION_REFERENCE")

        adjusted_signals, notes, changed = agent._apply_signal_assessments(
            signals,
            [
                ClinicalSignalAssessment(
                    signal_type="MISSING_PRESCRIPTION_REFERENCE",
                    severity_override="HIGH",
                    rationale="Contexte atténuant documenté.",
                )
            ],
        )
        adjusted_signal = next(
            s for s in adjusted_signals if s.signal_type == "MISSING_PRESCRIPTION_REFERENCE"
        )

        assert adjusted_signal.evidence == baseline_signal.evidence
        assert adjusted_signal.severity != baseline_signal.severity
        assert changed is True
        assert notes

    def test_adjustment_rationale_appears_in_reasons(self):
        ocr = self._critical_signal_ocr()
        decision = LlmClinicalDecision(
            severity_assessments=[
                ClinicalSignalAssessment(
                    signal_type="MISSING_PRESCRIPTION_REFERENCE",
                    severity_override="HIGH",
                    rationale="Contexte atténuant documenté dans les données transmises.",
                )
            ]
        )
        with patch.object(agent, "_invoke_llm_clinical", return_value=decision):
            result = agent.run("CLM-0001", ocr_result=ocr)

        assert any(
            "Contexte atténuant documenté dans les données transmises." in r
            for r in result.result_payload.reasons
        )


# ── Enveloppe générique — llm_trace, confidence, evidence_ids, revue ───────


class TestGenericEnvelope:
    def test_llm_trace_is_always_present(self):
        result = agent.run("CLM-0001")
        assert result.llm_trace is not None
        assert result.llm_trace.model_name

    def test_confidence_within_bounds(self):
        result = agent.run("CLM-0001")
        assert 0.0 <= result.confidence <= 1.0

    def test_evidence_ids_reference_real_evidence(self):
        ocr = _ocr_result(medication_count="2", service_date="2024-01-15")
        result = agent.run("CLM-0001", ocr_result=ocr)
        assert result.evidence_ids
        all_evidence_ids = {
            evidence.evidence_id
            for signal in result.result_payload.signals
            for evidence in signal.evidence
        } | {
            evidence.evidence_id
            for inconsistency in result.result_payload.inconsistencies
            for evidence in inconsistency.evidence
        }
        assert set(result.evidence_ids) <= all_evidence_ids

    def test_human_review_required_true_when_not_pass(self):
        ocr = _ocr_result(medication_count="2", service_date="2024-01-15")
        result = agent.run("CLM-0001", ocr_result=ocr)
        assert result.status is VerificationStatus.FAIL
        assert result.human_review_required is True

    def test_human_review_not_required_when_pass(self):
        ocr = _ocr_result(procedure_count="1", service_date="2024-01-15")
        coding = _coding_result(1)
        result = agent.run("CLM-0001", ocr_result=ocr, coding_result=coding)
        assert result.status is VerificationStatus.PASS
        assert result.human_review_required is False

    def test_errors_defaults_to_empty_list(self):
        result = agent.run("CLM-0001")
        assert result.errors == []


# ── node() — intégration state ───────────────────────────────────────────────


class TestNode:
    def test_node_reads_ocr_and_coding_from_state(self):
        ocr = _ocr_result(medication_count="1", service_date="2024-01-15")
        state = {"case_id": "CLM-0001", "ocr_result": ocr}
        updates = agent.node(state)
        assert updates["clinical_result"].status is VerificationStatus.FAIL

    def test_node_sets_bookkeeping_fields(self):
        updates = agent.node({"case_id": "CLM-0001"})
        assert updates["current_step"] == "clinical_consistency"
        assert updates["completed_steps"] == ["clinical_consistency"]

    def test_node_fail_populates_errors(self):
        ocr = _ocr_result(medication_count="1", service_date="2024-01-15")
        updates = agent.node({"case_id": "CLM-0001", "ocr_result": ocr})
        assert updates.get("errors")
        assert "clinical_consistency_agent" in updates["errors"][0]

    def test_node_needs_review_populates_alerts(self):
        updates = agent.node({"case_id": "CLM-0001"})
        assert updates.get("alerts")

    def test_node_pass_has_no_errors_or_alerts(self):
        ocr = _ocr_result(procedure_count="1", service_date="2024-01-15")
        coding = _coding_result(1)
        state = {"case_id": "CLM-0001", "ocr_result": ocr, "coding_result": coding}
        updates = agent.node(state)
        assert not updates.get("errors")
        assert not updates.get("alerts")

    def test_node_produces_audit_event(self):
        updates = agent.node({"case_id": "CLM-0001"})
        assert "audit_trail" in updates
        assert len(updates["audit_trail"]) == 1
        assert isinstance(updates["audit_trail"][0], AuditEvent)
        assert updates["audit_trail"][0].actor == "clinical_consistency_agent"

    def test_audit_event_carries_llm_traceability_fields(self):
        """Audite llm_call_id, model_name, prompt_version, tools et erreurs —
        traçabilité obligatoire de l'appel LLM et de l'outil autorisé."""
        updates = agent.node({"case_id": "CLM-0001"})
        details = updates["audit_trail"][0].details
        assert details["llm_call_id"]
        assert details["model_name"]
        assert details["prompt_version"]
        assert details["tools"] == "verifier_chronologie"
        assert "errors" in details

    def test_llm_call_id_is_unique_per_execution(self):
        first = agent.node({"case_id": "CLM-0001"})["audit_trail"][0].details["llm_call_id"]
        second = agent.node({"case_id": "CLM-0001"})["audit_trail"][0].details["llm_call_id"]
        assert first != second

    def test_node_never_raises_on_empty_state(self):
        updates = agent.node({})
        assert isinstance(updates, dict)


class TestEnvelopeErrorsOnLlmFailure:
    def test_llm_unavailable_populates_envelope_errors(self):
        with patch.object(agent, "_invoke_llm_clinical", return_value=None):
            result = agent.run("CLM-0001")
        assert len(result.errors) == 1
        assert result.errors[0].code == "LLM_UNAVAILABLE"

    def test_llm_available_leaves_envelope_errors_empty(self):
        decision = LlmClinicalDecision(clinical_context="Contexte.")
        with patch.object(agent, "_invoke_llm_clinical", return_value=decision):
            result = agent.run("CLM-0001")
        assert result.errors == []

    def test_audit_errors_reflect_envelope_errors(self):
        with patch.object(agent, "_invoke_llm_clinical", return_value=None):
            updates = agent.node({"case_id": "CLM-0001"})
        assert updates["audit_trail"][0].details["errors"] == "LLM_UNAVAILABLE"


# ── Interface injectable préservée ───────────────────────────────────────────


class TestInjectableInterfacePreserved:
    def test_protocol_still_exists_and_is_runtime_checkable(self):
        assert isinstance(agent._DEFAULT_IMPL, agent.ClinicalConsistencyRunnable)

    def test_make_node_accepts_custom_impl(self):
        class _Custom:
            def run(self, state):
                from schemas.results import ClinicalConsistencyResult

                return ClinicalConsistencyResult(
                    case_id=str(state.get("case_id")),
                    status=VerificationStatus.PASS,
                    llm_trace=LlmMetadata(model_name="test-llm", prompt_version="test"),
                    result_payload=ClinicalResultPayload(reasons=["custom"]),
                )

        node_fn = agent.make_node(_Custom())
        updates = node_fn({"case_id": "CLM-0001"})
        assert updates["clinical_result"].result_payload.reasons == ["custom"]


# ── Phase A — chronologie ordonnance/soin et acte absent ────────────────────


class TestChronologySignals:
    def test_prescription_before_care_is_detected(self):
        ocr = _ocr_result(
            procedure_count="1",
            medication_count="1",
            prescription_number="RX-1",
            care_date="2024-01-10",
            prescription_date="2024-01-05",
            service_date="2024-01-10",
        )
        coding = _coding_result(1)
        result = agent.run("CLM-0001", ocr_result=ocr, coding_result=coding)
        assert any(
            s.signal_type == "PRESCRIPTION_BEFORE_CARE" for s in result.result_payload.signals
        )
        assert result.status is VerificationStatus.FAIL

    def test_impossible_date_is_detected(self):
        ocr = _ocr_result(
            procedure_count="1",
            care_date="2024-02-30",
            service_date="2024-01-10",
        )
        coding = _coding_result(1)
        result = agent.run("CLM-0001", ocr_result=ocr, coding_result=coding)
        assert any(s.signal_type == "IMPOSSIBLE_DATE" for s in result.result_payload.signals)

    def test_missing_procedure_evidence_is_detected(self):
        ocr = _ocr_result(procedure_count="0", care_date="2024-01-10", service_date="2024-01-10")
        coding = _coding_result(0)
        result = agent.run("CLM-0001", ocr_result=ocr, coding_result=coding)
        assert any(
            s.signal_type == "MISSING_PROCEDURE_EVIDENCE" for s in result.result_payload.signals
        )

    def test_evidence_ids_include_chronology_evidence(self):
        ocr = _ocr_result(
            procedure_count="1",
            medication_count="1",
            prescription_number="RX-1",
            care_date="2024-01-10",
            prescription_date="2024-01-05",
            service_date="2024-01-10",
        )
        coding = _coding_result(1)
        result = agent.run("CLM-0001", ocr_result=ocr, coding_result=coding)
        chronology_signal = next(
            s for s in result.result_payload.signals if s.signal_type == "PRESCRIPTION_BEFORE_CARE"
        )
        assert {e.evidence_id for e in chronology_signal.evidence} <= set(result.evidence_ids)


# ── Résumés minimisés transmis au LLM ────────────────────────────────────────


class TestMinimizedSummaries:
    def test_fhir_summary_never_exposes_raw_bundle(self):
        fhir_result = FhirValidatorResult(
            case_id="CLM-0001",
            status=VerificationStatus.PASS,
            bundle_expected=True,
            resource_count=3,
            resource_types=["Patient", "Claim"],
            llm_metadata=LlmMetadata(model_name="test-llm", prompt_version="test"),
        )
        summary = agent._fhir_summary(fhir_result)
        assert summary == {
            "status": "PASS",
            "resource_count": 3,
            "resource_types": ["Patient", "Claim"],
        }

    def test_fhir_summary_none_when_absent(self):
        assert agent._fhir_summary(None) is None

    def test_medical_view_summary_reads_minimized_fields(self):
        view = MedicalView(
            patient_pseudonym="PAT-abc123",
            service_date="2024-01-10",
            procedures=["consultation"],
            prescription_names=["paracetamol"],
            diagnosis_codes=["J06.9"],
            encounter_class="ambulatory",
        )
        summary = agent._medical_view_summary(view)
        assert summary["patient_pseudonym"] == "PAT-abc123"
        assert summary["procedures"] == ["consultation"]

    def test_medical_view_summary_none_for_non_medical_view(self):
        assert agent._medical_view_summary({"not": "a medical view"}) is None
        assert agent._medical_view_summary(None) is None

    def test_extract_medical_view_reads_privacy_result_view(self):
        view_dict = MedicalView(
            patient_pseudonym="PAT-abc123",
            procedures=["consultation"],
        ).model_dump()
        privacy_result = PrivacyResult(
            case_id="CLM-0001",
            status=VerificationStatus.PASS,
            data_classification=DataClassification.SYNTHETIC_TEST_DATA,
            contains_real_personal_data=False,
            view=view_dict,
            view_role="MEDICAL_REVIEWER",
        )
        medical_view = agent._extract_medical_view(privacy_result)
        assert isinstance(medical_view, MedicalView)
        assert medical_view.patient_pseudonym == "PAT-abc123"

    def test_extract_medical_view_none_for_other_role(self):
        privacy_result = PrivacyResult(
            case_id="CLM-0001",
            status=VerificationStatus.PASS,
            data_classification=DataClassification.SYNTHETIC_TEST_DATA,
            contains_real_personal_data=False,
            view={"claim_id": "CLM-0001", "actor": "x", "action": "y", "outcome": "z"},
            view_role="AUDITOR",
        )
        assert agent._extract_medical_view(privacy_result) is None

    def test_extract_medical_view_none_when_absent(self):
        assert agent._extract_medical_view(None) is None


# ── Phase B — outil autorisé et intégration ─────────────────────────────────


class TestAuthorizedToolIntegration:
    def test_run_accepts_fhir_result_and_medical_view(self):
        """run() accepte les nouveaux paramètres sans lever — l'intégration
        des résumés minimisés ne casse jamais l'appel existant."""
        fhir_result = FhirValidatorResult(
            case_id="CLM-0001",
            status=VerificationStatus.PASS,
            bundle_expected=True,
            llm_metadata=LlmMetadata(model_name="test-llm", prompt_version="test"),
        )
        medical_view = MedicalView(patient_pseudonym="PAT-abc123")
        result = agent.run(
            "CLM-0001",
            fhir_result=fhir_result,
            medical_view=medical_view,
        )
        assert result.status in (VerificationStatus.PASS, VerificationStatus.NEEDS_REVIEW)

    def test_verifier_chronologie_tool_is_the_only_authorized_tool(self):
        from orchestrator.orchestrator import AgentName
        from orchestrator.policies import ALLOWED_TOOLS_PER_AGENT

        assert ALLOWED_TOOLS_PER_AGENT[AgentName.CLINICAL_CONSISTENCY] == frozenset(
            {"verifier_chronologie"}
        )

    def test_llm_decision_has_no_diagnostic_or_decision_field(self):
        """Garantie structurelle : ni diagnostic, ni décision finale, ni champ
        d'affirmation libre non rattaché à des signaux ne peut jamais être
        introduit par le LLM."""
        forbidden_fields = {
            "diagnosis",
            "diagnostic",
            "decision",
            "final_decision",
            "recommendation",
            "status",
        }
        assert forbidden_fields.isdisjoint(LlmClinicalDecision.model_fields.keys())


# ── _merge_llm_decision — preuves/incohérences citées, confiance, revue ─────


class TestMergeLlmDecision:
    def test_valid_referenced_evidence_produces_no_warning(self):
        reasons = agent._merge_llm_decision(
            LlmClinicalDecision(
                clinical_context="Contexte.", referenced_evidence_ids=["EVID-real1"]
            ),
            [],
            known_evidence_ids={"EVID-real1"},
            known_inconsistency_types=set(),
        )
        assert not any("ignorées" in r for r in reasons)

    def test_unknown_referenced_evidence_is_ignored_with_warning(self):
        """Garantie anti-hallucination : une preuve inventée par le LLM est
        signalée comme ignorée, jamais acceptée telle quelle."""
        reasons = agent._merge_llm_decision(
            LlmClinicalDecision(referenced_evidence_ids=["EVID-invented"]),
            [],
            known_evidence_ids={"EVID-real1"},
            known_inconsistency_types=set(),
        )
        assert any("ignorées" in r for r in reasons)

    def test_unknown_acknowledged_inconsistency_is_ignored_with_warning(self):
        reasons = agent._merge_llm_decision(
            LlmClinicalDecision(acknowledged_inconsistencies=["INVENTED_TYPE"]),
            [],
            known_evidence_ids=set(),
            known_inconsistency_types={"PROCEDURE_CODING_COUNT_MISMATCH"},
        )
        assert any("ignorées" in r for r in reasons)

    def test_suggests_human_review_appends_informational_note_only(self):
        reasons = agent._merge_llm_decision(
            LlmClinicalDecision(suggests_human_review=True),
            [],
            known_evidence_ids=set(),
            known_inconsistency_types=set(),
        )
        assert any("non contraignante" in r for r in reasons)

    def test_llm_confidence_never_overrides_deterministic_confidence(self):
        """llm_confidence est purement informatif : le résultat final garde
        sa confiance déterministe, jamais celle suggérée par le LLM."""
        ocr = _ocr_result(procedure_count="1", service_date="2024-01-15")
        coding = _coding_result(1)
        deterministic_confidence = 1.0  # aucun signal -> confiance max

        with patch.object(
            agent,
            "_invoke_llm_clinical",
            return_value=LlmClinicalDecision(llm_confidence=0.01),
        ):
            result = agent.run("CLM-0001", ocr_result=ocr, coding_result=coding)

        assert result.confidence == deterministic_confidence
        assert any("Confiance perçue par le LLM" in r for r in result.result_payload.reasons)

    def test_suggests_human_review_never_overrides_deterministic_flag(self):
        """Même si le LLM suggère une revue, un dossier PASS reste
        human_review_required=False (dérivé uniquement du statut)."""
        ocr = _ocr_result(procedure_count="1", service_date="2024-01-15")
        coding = _coding_result(1)

        with patch.object(
            agent,
            "_invoke_llm_clinical",
            return_value=LlmClinicalDecision(suggests_human_review=True),
        ):
            result = agent.run("CLM-0001", ocr_result=ocr, coding_result=coding)

        assert result.status is VerificationStatus.PASS
        assert result.human_review_required is False

    def test_llm_decision_none_still_produces_deterministic_result(self):
        reasons = agent._merge_llm_decision(
            None, ["motif initial"], known_evidence_ids=set(), known_inconsistency_types=set()
        )
        assert "motif initial" in reasons
        assert any("LLM indisponible" in r for r in reasons)
