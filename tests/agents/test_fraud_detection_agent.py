"""Tests du Fraud Detection Agent — ClaimShield Santé (étape 12).

Couvre :
  - Phase A déterministe : signaux pondérés, risk_score, seuils de statut.
  - Doublons : détection exacte/quasi-doublon via services.duplicate_index
    et tools.statistics, à partir de la vue antifraude minimisée
    (FraudView, historique pseudonymisé seulement) — jamais un entrepôt brut.
  - Phase B agent ReAct LLM (appel obligatoire, outil autorisé
    verifier_doublon) : interprétation des doublons/montants/signaux,
    jamais d'autorité sur le score/statut, jamais d'accusation, de blocage
    définitif ni de décision sans revue humaine.
  - Fallback LLM indisponible : score déterministe conservé.
  - node() : mise à jour du state, audit_trail, errors/alerts.
  - Interface injectable préservée (FraudDetectionRunnable, make_node).
  - Enveloppe générique (llm_trace, confidence, evidence_ids,
    human_review_required, result_payload) : chaque signal doit être
    référencé par au moins une preuve structurée (FraudEvidence).
"""
from __future__ import annotations

from unittest.mock import patch

from agents.fraud_detection_agent import agent
from agents.fraud_detection_agent.schemas import LlmFraudDecision, SignalAssessment
from agents.privacy_agent.schemas import FraudView
from schemas.domain import DataClassification, DocumentType, ExtractionStatus, OcrSource, VerificationStatus
from schemas.results import (
    AuditEvent,
    CoverageResult,
    DocumentOcrResult,
    FraudResultPayload,
    IdentityCoverageResult,
    IdentityResult,
    LlmMetadata,
    MedicalCodingResult,
    PrivacyResult,
)
from services.duplicate_index import DuplicateIndex


def _identity_coverage(
    identity_status: VerificationStatus = VerificationStatus.PASS,
    coverage_status: VerificationStatus = VerificationStatus.PASS,
    ceiling_exceeded: bool = False,
    preauthorization_required: bool = False,
    preauthorization_status: str | None = None,
) -> IdentityCoverageResult:
    return IdentityCoverageResult(
        case_id="CLM-0001",
        identity=IdentityResult(status=identity_status),
        coverage=CoverageResult(
            status=coverage_status,
            ceiling_exceeded=ceiling_exceeded,
            preauthorization_required=preauthorization_required,
            preauthorization_status=preauthorization_status,
        ),
    )


def _coding_result(status: VerificationStatus = VerificationStatus.PASS) -> MedicalCodingResult:
    return MedicalCodingResult(
        case_id="CLM-0001",
        status=status,
        codings=[],
        llm_metadata=LlmMetadata(model_name="test-llm", prompt_version="test"),
    )


def _ocr_result(confidence_score: float = 0.9) -> DocumentOcrResult:
    return DocumentOcrResult(
        claim_id="CLM-0001",
        file_path="facture.pdf",
        sha256="a" * 64,
        mime_type="application/pdf",
        extraction_status=ExtractionStatus.SUCCESS,
        status=VerificationStatus.PASS,
        document_type=DocumentType.INVOICE,
        ocr_source=OcrSource.PDF_TEXT,
        confidence_score=confidence_score,
    )


def _fraud_view(**overrides) -> FraudView:
    payload = {
        "patient_pseudonym": "PAT-AAAAAAAAAAAA",
        "document_hashes": {"invoice": "a" * 64},
        "amount_requested": "100.00",
        "total_billed": "100.00",
        "service_date": "2024-01-15",
        "invoice_reference": "***1234",
        "provider_reference": "PRV-BBBBBBBBBBBB",
    }
    payload.update(overrides)
    return FraudView(**payload)


def _privacy_result_with_view(view: FraudView) -> PrivacyResult:
    return PrivacyResult(
        case_id="CLM-0001",
        status=VerificationStatus.PASS,
        data_classification=DataClassification.SYNTHETIC_TEST_DATA,
        contains_real_personal_data=False,
        view=view.model_dump(),
        view_role="FRAUD_ANALYST",
    )


# ── Phase A — pas de données ────────────────────────────────────────────────


class TestNoUpstreamData:
    def test_no_data_is_needs_review(self):
        result = agent.run("CLM-0001")
        assert result.status is VerificationStatus.NEEDS_REVIEW

    def test_no_data_risk_score_zero(self):
        result = agent.run("CLM-0001")
        assert result.result_payload.risk_score == 0.0

    def test_no_data_explains_reason(self):
        result = agent.run("CLM-0001")
        assert any("Aucune preuve" in r for r in result.result_payload.reasons)


# ── Phase A — cas propre ──────────────────────────────────────────────────────


class TestCleanCase:
    def test_all_pass_no_signals_is_pass(self):
        identity_coverage = _identity_coverage()
        coding = _coding_result()
        ocr = _ocr_result()
        result = agent.run(
            "CLM-0001",
            identity_coverage_result=identity_coverage,
            coding_result=coding,
            ocr_result=ocr,
        )
        assert result.status is VerificationStatus.PASS
        assert result.result_payload.signals == []
        assert result.result_payload.risk_score == 0.0

    def test_duplicate_invoice_none_without_fraud_view(self):
        """Sans vue antifraude minimisée, la vérification de doublon n'a
        même pas pu être menée : jamais une valeur inventée."""
        result = agent.run("CLM-0001")
        assert result.result_payload.duplicate_invoice is None


# ── Phase A — signaux individuels ────────────────────────────────────────────


class TestIndividualSignals:
    def test_identity_fail_adds_signal(self):
        identity_coverage = _identity_coverage(identity_status=VerificationStatus.FAIL)
        result = agent.run("CLM-0001", identity_coverage_result=identity_coverage)
        assert any(s.signal_type == "IDENTITY_MISMATCH" for s in result.result_payload.signals)

    def test_coverage_fail_adds_signal(self):
        identity_coverage = _identity_coverage(coverage_status=VerificationStatus.FAIL)
        result = agent.run("CLM-0001", identity_coverage_result=identity_coverage)
        assert any(
            s.signal_type == "COVERAGE_INACTIVE_OR_EXPIRED" for s in result.result_payload.signals
        )

    def test_ceiling_exceeded_adds_signal(self):
        identity_coverage = _identity_coverage(ceiling_exceeded=True)
        result = agent.run("CLM-0001", identity_coverage_result=identity_coverage)
        assert any(s.signal_type == "CEILING_EXCEEDED" for s in result.result_payload.signals)

    def test_preauthorization_missing_adds_signal(self):
        identity_coverage = _identity_coverage(
            preauthorization_required=True, preauthorization_status="missing"
        )
        result = agent.run("CLM-0001", identity_coverage_result=identity_coverage)
        assert any(
            s.signal_type == "PREAUTHORIZATION_MISSING" for s in result.result_payload.signals
        )

    def test_preauthorization_approved_no_signal(self):
        identity_coverage = _identity_coverage(
            preauthorization_required=True, preauthorization_status="approved"
        )
        result = agent.run("CLM-0001", identity_coverage_result=identity_coverage)
        assert not any(
            s.signal_type == "PREAUTHORIZATION_MISSING" for s in result.result_payload.signals
        )

    def test_unresolved_coding_adds_signal(self):
        coding = _coding_result(status=VerificationStatus.NEEDS_REVIEW)
        result = agent.run("CLM-0001", coding_result=coding)
        assert any(s.signal_type == "UNRESOLVED_CODING" for s in result.result_payload.signals)

    def test_low_ocr_confidence_adds_signal(self):
        ocr = _ocr_result(confidence_score=0.2)
        result = agent.run("CLM-0001", ocr_result=ocr)
        assert any(
            s.signal_type == "LOW_EXTRACTION_CONFIDENCE" for s in result.result_payload.signals
        )

    def test_high_ocr_confidence_no_signal(self):
        ocr = _ocr_result(confidence_score=0.95)
        result = agent.run("CLM-0001", ocr_result=ocr)
        assert not any(
            s.signal_type == "LOW_EXTRACTION_CONFIDENCE" for s in result.result_payload.signals
        )

    def test_every_signal_carries_at_least_one_evidence(self):
        """Un signal de fraude ne peut jamais être une affirmation non
        appuyée — voir FraudSignal.evidence (min_length=1)."""
        identity_coverage = _identity_coverage(
            identity_status=VerificationStatus.FAIL,
            coverage_status=VerificationStatus.FAIL,
            ceiling_exceeded=True,
            preauthorization_required=True,
            preauthorization_status="missing",
        )
        coding = _coding_result(status=VerificationStatus.FAIL)
        ocr = _ocr_result(confidence_score=0.1)
        result = agent.run(
            "CLM-0001",
            identity_coverage_result=identity_coverage,
            coding_result=coding,
            ocr_result=ocr,
        )
        assert result.result_payload.signals
        assert all(len(s.evidence) >= 1 for s in result.result_payload.signals)


# ── Phase A — seuils de statut ────────────────────────────────────────────────


class TestStatusThresholds:
    def test_high_combined_risk_is_fail(self):
        identity_coverage = _identity_coverage(
            identity_status=VerificationStatus.FAIL,
            coverage_status=VerificationStatus.FAIL,
        )
        result = agent.run("CLM-0001", identity_coverage_result=identity_coverage)
        assert result.result_payload.risk_score >= 0.7
        assert result.status is VerificationStatus.FAIL

    def test_moderate_risk_is_needs_review(self):
        identity_coverage = _identity_coverage(identity_status=VerificationStatus.NEEDS_REVIEW)
        coding = _coding_result(status=VerificationStatus.NEEDS_REVIEW)
        result = agent.run("CLM-0001", identity_coverage_result=identity_coverage, coding_result=coding)
        assert 0.3 <= result.result_payload.risk_score < 0.7
        assert result.status is VerificationStatus.NEEDS_REVIEW

    def test_risk_score_capped_at_one(self):
        identity_coverage = _identity_coverage(
            identity_status=VerificationStatus.FAIL,
            coverage_status=VerificationStatus.FAIL,
            ceiling_exceeded=True,
            preauthorization_required=True,
            preauthorization_status="missing",
        )
        coding = _coding_result(status=VerificationStatus.FAIL)
        ocr = _ocr_result(confidence_score=0.1)
        result = agent.run(
            "CLM-0001",
            identity_coverage_result=identity_coverage,
            coding_result=coding,
            ocr_result=ocr,
        )
        assert result.result_payload.risk_score <= 1.0


# ── Phase B — LLM ne peut jamais changer le score/statut ────────────────────


class TestLlmNeverOverridesRisk:
    def test_llm_fallback_keeps_deterministic_score(self):
        with patch.object(agent, "_invoke_llm_fraud", return_value=None):
            result = agent.run("CLM-0001")
        assert result.result_payload.risk_score == 0.0
        assert any("LLM indisponible" in r for r in result.result_payload.reasons)

    def test_llm_decision_has_no_status_or_score_field(self):
        assert "status" not in LlmFraudDecision.model_fields
        assert "risk_score" not in LlmFraudDecision.model_fields

    def test_llm_rationale_appended_to_reasons(self):
        decision = LlmFraudDecision(rationale="Justification de test.", reasons=["motif LLM"])
        with patch.object(agent, "_invoke_llm_fraud", return_value=decision):
            result = agent.run("CLM-0001")
        assert "Justification de test." in result.result_payload.reasons
        assert "motif LLM" in result.result_payload.reasons


# ── P1-1 — autorité LLM bornée sur la pondération des signaux ──────────────


class TestBoundedSignalAssessments:
    """Le LLM ne peut jamais fixer lui-même score/statut (toujours vrai,
    voir TestLlmNeverOverridesRisk) mais peut désormais proposer, pour un
    signal déjà calculé et déjà attribué à une preuve, un ajustement borné
    (DOWNGRADE/NEUTRAL/UPGRADE) — recalculé déterministiquement, jamais
    laissé au LLM."""

    def _moderate_risk_identity_coverage(self):
        return _identity_coverage(
            identity_status=VerificationStatus.NEEDS_REVIEW,
            coverage_status=VerificationStatus.FAIL,
        )

    def test_upgrade_increases_risk_score_and_can_change_status(self):
        identity_coverage = self._moderate_risk_identity_coverage()
        with patch.object(agent, "_invoke_llm_fraud", return_value=None):
            baseline = agent.run("CLM-0001", identity_coverage_result=identity_coverage)
        assert baseline.status is VerificationStatus.NEEDS_REVIEW

        decision = LlmFraudDecision(
            signal_assessments=[
                SignalAssessment(
                    signal_type="COVERAGE_INACTIVE_OR_EXPIRED",
                    severity_adjustment="UPGRADE",
                    rationale="Contexte aggravant identifié.",
                )
            ]
        )
        with patch.object(agent, "_invoke_llm_fraud", return_value=decision):
            upgraded = agent.run("CLM-0001", identity_coverage_result=identity_coverage)

        assert upgraded.result_payload.risk_score > baseline.result_payload.risk_score
        assert upgraded.status is VerificationStatus.FAIL
        assert upgraded.human_review_required is True

    def test_adjustment_emits_structured_log(self, caplog):
        """P3-2 : le point de décision autonome (ajustement de pondération
        effectivement appliqué) est journalisé pour traçabilité."""
        identity_coverage = self._moderate_risk_identity_coverage()
        decision = LlmFraudDecision(
            signal_assessments=[
                SignalAssessment(
                    signal_type="COVERAGE_INACTIVE_OR_EXPIRED",
                    severity_adjustment="UPGRADE",
                    rationale="Contexte aggravant identifié.",
                )
            ]
        )
        with caplog.at_level("INFO"):
            with patch.object(agent, "_invoke_llm_fraud", return_value=decision):
                agent.run("CLM-0001", identity_coverage_result=identity_coverage)

        messages = [r.getMessage() for r in caplog.records]
        assert any("fraud_detection_signal_weight_adjusted" in m and "CLM-0001" in m for m in messages)

    def test_downgrade_decreases_risk_score(self):
        identity_coverage = self._moderate_risk_identity_coverage()
        with patch.object(agent, "_invoke_llm_fraud", return_value=None):
            baseline = agent.run("CLM-0001", identity_coverage_result=identity_coverage)

        decision = LlmFraudDecision(
            signal_assessments=[
                SignalAssessment(
                    signal_type="COVERAGE_INACTIVE_OR_EXPIRED",
                    severity_adjustment="DOWNGRADE",
                    rationale="Contexte atténuant identifié.",
                )
            ]
        )
        with patch.object(agent, "_invoke_llm_fraud", return_value=decision):
            downgraded = agent.run("CLM-0001", identity_coverage_result=identity_coverage)

        assert downgraded.result_payload.risk_score < baseline.result_payload.risk_score

    def test_neutral_adjustment_has_no_effect(self):
        identity_coverage = self._moderate_risk_identity_coverage()
        with patch.object(agent, "_invoke_llm_fraud", return_value=None):
            baseline = agent.run("CLM-0001", identity_coverage_result=identity_coverage)

        decision = LlmFraudDecision(
            signal_assessments=[
                SignalAssessment(signal_type="COVERAGE_INACTIVE_OR_EXPIRED", severity_adjustment="NEUTRAL")
            ]
        )
        with patch.object(agent, "_invoke_llm_fraud", return_value=decision):
            neutral = agent.run("CLM-0001", identity_coverage_result=identity_coverage)

        assert neutral.result_payload.risk_score == baseline.result_payload.risk_score
        assert neutral.status == baseline.status

    def test_unknown_signal_type_is_silently_ignored(self):
        """Une référence à un signal_type jamais calculé par la Phase A —
        garantie anti-hallucination : aucun nouveau signal ne peut être créé
        via signal_assessments, seule une pondération de signal réel peut
        changer."""
        identity_coverage = self._moderate_risk_identity_coverage()
        with patch.object(agent, "_invoke_llm_fraud", return_value=None):
            baseline = agent.run("CLM-0001", identity_coverage_result=identity_coverage)

        decision = LlmFraudDecision(
            signal_assessments=[
                SignalAssessment(
                    signal_type="COMPLETELY_INVENTED_SIGNAL",
                    severity_adjustment="UPGRADE",
                    rationale="Signal jamais calculé par la Phase A.",
                )
            ]
        )
        with patch.object(agent, "_invoke_llm_fraud", return_value=decision):
            result = agent.run("CLM-0001", identity_coverage_result=identity_coverage)

        assert result.result_payload.risk_score == baseline.result_payload.risk_score
        assert len(result.result_payload.signals) == len(baseline.result_payload.signals)
        assert not any(
            s.signal_type == "COMPLETELY_INVENTED_SIGNAL" for s in result.result_payload.signals
        )

    def test_adjustment_never_creates_a_new_signal(self):
        identity_coverage = self._moderate_risk_identity_coverage()
        decision = LlmFraudDecision(
            signal_assessments=[
                SignalAssessment(
                    signal_type="COVERAGE_INACTIVE_OR_EXPIRED",
                    severity_adjustment="UPGRADE",
                    rationale="Contexte aggravant identifié.",
                )
            ]
        )
        with patch.object(agent, "_invoke_llm_fraud", return_value=decision):
            result = agent.run("CLM-0001", identity_coverage_result=identity_coverage)

        assert len(result.result_payload.signals) == 2  # inchangé : IDENTITY_AMBIGUOUS + COVERAGE_*

    def test_evidence_unchanged_after_adjustment(self):
        """Seule la pondération numérique change — la preuve attribuée
        (evidence_id, source, field, valeur) reste strictement identique.

        Compare au sein d'une seule collecte de signaux (``evidence_id`` est
        auto-généré à chaque appel de ``_collect_signals`` — comparer deux
        appels séparés de ``agent.run()`` donnerait à tort des evidence_id
        différents, sans rapport avec l'ajustement testé ici)."""
        identity_coverage = self._moderate_risk_identity_coverage()
        signals, _ = agent._collect_signals(identity_coverage, None, None)
        baseline_signal = next(s for s in signals if s.signal_type == "COVERAGE_INACTIVE_OR_EXPIRED")

        adjusted_signals, notes = agent._apply_signal_assessments(
            signals,
            [
                SignalAssessment(
                    signal_type="COVERAGE_INACTIVE_OR_EXPIRED",
                    severity_adjustment="UPGRADE",
                    rationale="Contexte aggravant identifié.",
                )
            ],
        )
        adjusted_signal = next(
            s for s in adjusted_signals if s.signal_type == "COVERAGE_INACTIVE_OR_EXPIRED"
        )

        assert adjusted_signal.evidence == baseline_signal.evidence
        assert adjusted_signal.risk_contribution != baseline_signal.risk_contribution
        assert notes  # motif d'ajustement produit

    def test_adjustment_rationale_appears_in_reasons(self):
        identity_coverage = self._moderate_risk_identity_coverage()
        decision = LlmFraudDecision(
            signal_assessments=[
                SignalAssessment(
                    signal_type="COVERAGE_INACTIVE_OR_EXPIRED",
                    severity_adjustment="UPGRADE",
                    rationale="Contexte aggravant identifié dans les données transmises.",
                )
            ]
        )
        with patch.object(agent, "_invoke_llm_fraud", return_value=decision):
            result = agent.run("CLM-0001", identity_coverage_result=identity_coverage)

        assert any(
            "Contexte aggravant identifié dans les données transmises." in r
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
        identity_coverage = _identity_coverage(identity_status=VerificationStatus.FAIL)
        result = agent.run("CLM-0001", identity_coverage_result=identity_coverage)
        assert result.evidence_ids
        all_evidence_ids = {
            evidence.evidence_id
            for signal in result.result_payload.signals
            for evidence in signal.evidence
        }
        assert set(result.evidence_ids) <= all_evidence_ids

    def test_human_review_required_true_when_not_pass(self):
        identity_coverage = _identity_coverage(
            identity_status=VerificationStatus.FAIL,
            coverage_status=VerificationStatus.FAIL,
        )
        result = agent.run("CLM-0001", identity_coverage_result=identity_coverage)
        assert result.status is VerificationStatus.FAIL
        assert result.human_review_required is True

    def test_human_review_not_required_when_pass(self):
        result = agent.run(
            "CLM-0001",
            identity_coverage_result=_identity_coverage(),
            coding_result=_coding_result(),
            ocr_result=_ocr_result(),
        )
        assert result.status is VerificationStatus.PASS
        assert result.human_review_required is False

    def test_errors_defaults_to_empty_list(self):
        result = agent.run("CLM-0001")
        assert result.errors == []


# ── node() — intégration state ───────────────────────────────────────────────


class TestNode:
    def test_node_reads_all_upstream_results_from_state(self):
        identity_coverage = _identity_coverage(identity_status=VerificationStatus.FAIL)
        state = {"case_id": "CLM-0001", "identity_coverage_result": identity_coverage}
        updates = agent.node(state)
        assert updates["fraud_result"].status is not VerificationStatus.PASS

    def test_node_sets_bookkeeping_fields(self):
        updates = agent.node({"case_id": "CLM-0001"})
        assert updates["current_step"] == "fraud_detection"
        assert updates["completed_steps"] == ["fraud_detection"]

    def test_node_fail_populates_errors(self):
        identity_coverage = _identity_coverage(
            identity_status=VerificationStatus.FAIL,
            coverage_status=VerificationStatus.FAIL,
        )
        updates = agent.node({"case_id": "CLM-0001", "identity_coverage_result": identity_coverage})
        assert updates.get("errors")
        assert "fraud_detection_agent" in updates["errors"][0]

    def test_node_needs_review_populates_alerts(self):
        updates = agent.node({"case_id": "CLM-0001"})
        assert updates.get("alerts")

    def test_node_pass_has_no_errors_or_alerts(self):
        state = {
            "case_id": "CLM-0001",
            "identity_coverage_result": _identity_coverage(),
            "coding_result": _coding_result(),
            "ocr_result": _ocr_result(),
        }
        updates = agent.node(state)
        assert not updates.get("errors")
        assert not updates.get("alerts")

    def test_node_produces_audit_event(self):
        updates = agent.node({"case_id": "CLM-0001"})
        assert "audit_trail" in updates
        assert len(updates["audit_trail"]) == 1
        assert isinstance(updates["audit_trail"][0], AuditEvent)
        assert updates["audit_trail"][0].actor == "fraud_detection_agent"

    def test_audit_event_carries_llm_traceability_fields(self):
        """Audite llm_call_id, model_name, prompt_version, tools et erreurs —
        traçabilité obligatoire de l'appel LLM et de l'outil autorisé."""
        updates = agent.node({"case_id": "CLM-0001"})
        details = updates["audit_trail"][0].details
        assert details["llm_call_id"]
        assert details["model_name"]
        assert details["prompt_version"]
        assert details["tools"] == "verifier_doublon"
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
        with patch.object(agent, "_invoke_llm_fraud", return_value=None):
            result = agent.run("CLM-0001")
        assert len(result.errors) == 1
        assert result.errors[0].code == "LLM_UNAVAILABLE"

    def test_llm_available_leaves_envelope_errors_empty(self):
        decision = LlmFraudDecision(rationale="Justification.")
        with patch.object(agent, "_invoke_llm_fraud", return_value=decision):
            result = agent.run("CLM-0001")
        assert result.errors == []

    def test_audit_errors_reflect_envelope_errors(self):
        with patch.object(agent, "_invoke_llm_fraud", return_value=None):
            updates = agent.node({"case_id": "CLM-0001"})
        assert updates["audit_trail"][0].details["errors"] == "LLM_UNAVAILABLE"


# ── Interface injectable préservée ───────────────────────────────────────────


class TestInjectableInterfacePreserved:
    def test_protocol_still_exists_and_is_runtime_checkable(self):
        assert isinstance(agent._DEFAULT_IMPL, agent.FraudDetectionRunnable)

    def test_make_node_accepts_custom_impl(self):
        class _Custom:
            def run(self, state):
                from schemas.results import FraudDetectionResult

                return FraudDetectionResult(
                    case_id=str(state.get("case_id")),
                    status=VerificationStatus.PASS,
                    llm_trace=LlmMetadata(model_name="test-llm", prompt_version="test"),
                    result_payload=FraudResultPayload(reasons=["custom"]),
                )

        node_fn = agent.make_node(_Custom())
        updates = node_fn({"case_id": "CLM-0001"})
        assert updates["fraud_result"].result_payload.reasons == ["custom"]


# ── Doublons — historique pseudonymisé seulement ─────────────────────────────


class TestDuplicateDetection:
    def test_no_fraud_view_no_duplicate_check(self):
        index = DuplicateIndex()
        result = agent.run("CLM-0001", fraud_view=None, duplicate_index=index)
        assert result.result_payload.duplicate_invoice is None
        assert len(index) == 0

    def test_empty_history_reports_false_not_none(self):
        """Un index vide mais interrogeable donne une réponse « non trouvé »
        (False), pas une absence de vérification (None)."""
        index = DuplicateIndex()
        result = agent.run("CLM-0001", fraud_view=_fraud_view(), duplicate_index=index)
        assert result.result_payload.duplicate_invoice is False

    def test_exact_duplicate_detected_via_hash(self):
        index = DuplicateIndex()
        agent.run(
            "CLM-0001",
            fraud_view=_fraud_view(document_hashes={"invoice": "a" * 64}),
            duplicate_index=index,
        )
        result = agent.run(
            "CLM-0002",
            identity_coverage_result=_identity_coverage(),
            coding_result=_coding_result(),
            ocr_result=_ocr_result(),
            fraud_view=_fraud_view(document_hashes={"invoice": "a" * 64}),
            duplicate_index=index,
        )
        assert result.result_payload.duplicate_invoice is True
        assert any(
            s.signal_type == "EXACT_DUPLICATE_INVOICE" for s in result.result_payload.signals
        )
        assert result.status is not VerificationStatus.PASS

    def test_near_duplicate_detected_via_similarity(self):
        index = DuplicateIndex()
        agent.run(
            "CLM-0001",
            fraud_view=_fraud_view(
                document_hashes={"invoice": "a" * 64},
                amount_requested="100.00",
                service_date="2024-01-15",
            ),
            duplicate_index=index,
        )
        result = agent.run(
            "CLM-0002",
            fraud_view=_fraud_view(
                document_hashes={"invoice": "b" * 64},
                amount_requested="100.50",
                service_date="2024-01-16",
            ),
            duplicate_index=index,
        )
        assert result.result_payload.duplicate_invoice is True
        assert any(
            s.signal_type == "NEAR_DUPLICATE_INVOICE" for s in result.result_payload.signals
        )

    def test_different_patient_is_not_a_duplicate(self):
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

    def test_duplicate_signal_carries_attributed_evidence(self):
        index = DuplicateIndex()
        agent.run("CLM-0001", fraud_view=_fraud_view(), duplicate_index=index)
        result = agent.run(
            "CLM-0002",
            fraud_view=_fraud_view(document_hashes={"invoice": "a" * 64}),
            duplicate_index=index,
        )
        signal = next(
            s for s in result.result_payload.signals if s.signal_type == "EXACT_DUPLICATE_INVOICE"
        )
        assert len(signal.evidence) >= 1
        assert all(e.evidence_id for e in signal.evidence)

    def test_missing_amount_and_hash_gives_none(self):
        index = DuplicateIndex()
        result = agent.run(
            "CLM-0001",
            fraud_view=_fraud_view(document_hashes={}, amount_requested=None, total_billed=None),
            duplicate_index=index,
        )
        assert result.result_payload.duplicate_invoice is None

    def test_defaults_to_module_level_shared_index_when_not_injected(self):
        """Sans duplicate_index explicite, run() retombe sur l'index partagé
        par défaut (visible, jamais un singleton caché)."""
        from agents.fraud_detection_agent.tools import _DEFAULT_DUPLICATE_INDEX

        before = len(_DEFAULT_DUPLICATE_INDEX)
        agent.run("CLM-DEFAULT-IDX-TEST", fraud_view=_fraud_view(document_hashes={"invoice": "f" * 64}))
        assert len(_DEFAULT_DUPLICATE_INDEX) == before + 1


class TestExtractFraudView:
    def test_extracts_view_for_fraud_analyst_role(self):
        privacy_result = _privacy_result_with_view(_fraud_view())
        view = agent._extract_fraud_view(privacy_result)
        assert isinstance(view, FraudView)
        assert view.patient_pseudonym == "PAT-AAAAAAAAAAAA"

    def test_none_for_other_role(self):
        privacy_result = PrivacyResult(
            case_id="CLM-0001",
            status=VerificationStatus.PASS,
            data_classification=DataClassification.SYNTHETIC_TEST_DATA,
            contains_real_personal_data=False,
            view={"claim_id": "CLM-0001", "actor": "x", "action": "y", "outcome": "z"},
            view_role="AUDITOR",
        )
        assert agent._extract_fraud_view(privacy_result) is None

    def test_none_when_privacy_result_absent(self):
        assert agent._extract_fraud_view(None) is None


# ── Phase B — outil autorisé et interdictions ────────────────────────────────


class TestAuthorizedToolAndProhibitions:
    def test_llm_decision_has_no_accusation_or_blocking_field(self):
        """Garantie structurelle : ni accusation, ni blocage, ni décision
        finale ne peut jamais être introduit par le LLM."""
        forbidden_fields = {
            "is_fraud",
            "fraud_confirmed",
            "accusation",
            "block",
            "blocked",
            "final_decision",
            "decision",
            "status",
            "risk_score",
        }
        assert forbidden_fields.isdisjoint(LlmFraudDecision.model_fields.keys())

    def test_fail_status_always_requires_human_review(self):
        """Aucun blocage définitif : un statut FAIL implique toujours une
        revue humaine, jamais une décision autonome de l'agent ou du LLM."""
        identity_coverage = _identity_coverage(
            identity_status=VerificationStatus.FAIL, coverage_status=VerificationStatus.FAIL
        )
        result = agent.run("CLM-0001", identity_coverage_result=identity_coverage)
        assert result.status is VerificationStatus.FAIL
        assert result.human_review_required is True


# ── _merge_llm_decision — signaux cités, risque perçu, revue ────────────────


class TestMergeLlmDecision:
    def test_valid_referenced_signal_produces_no_warning(self):
        reasons = agent._merge_llm_decision(
            LlmFraudDecision(rationale="Justification.", referenced_signal_types=["IDENTITY_MISMATCH"]),
            [],
            known_signal_types={"IDENTITY_MISMATCH"},
        )
        assert not any("ignorées" in r for r in reasons)

    def test_unknown_referenced_signal_is_ignored_with_warning(self):
        reasons = agent._merge_llm_decision(
            LlmFraudDecision(referenced_signal_types=["INVENTED_SIGNAL"]),
            [],
            known_signal_types={"IDENTITY_MISMATCH"},
        )
        assert any("ignorées" in r for r in reasons)

    def test_suggests_human_review_appends_informational_note_only(self):
        reasons = agent._merge_llm_decision(
            LlmFraudDecision(suggests_human_review=True), [], known_signal_types=set()
        )
        assert any("non contraignante" in r for r in reasons)

    def test_llm_risk_perception_never_overrides_deterministic_risk_score(self):
        with patch.object(
            agent, "_invoke_llm_fraud", return_value=LlmFraudDecision(llm_risk_perception=0.99)
        ):
            result = agent.run("CLM-0001")
        assert result.result_payload.risk_score == 0.0
        assert any("Risque perçu par le LLM" in r for r in result.result_payload.reasons)

    def test_suggests_human_review_never_overrides_deterministic_flag(self):
        with patch.object(
            agent, "_invoke_llm_fraud", return_value=LlmFraudDecision(suggests_human_review=True)
        ):
            result = agent.run(
                "CLM-0001",
                identity_coverage_result=_identity_coverage(),
                coding_result=_coding_result(),
                ocr_result=_ocr_result(),
            )
        assert result.status is VerificationStatus.PASS
        assert result.human_review_required is False

    def test_llm_decision_none_still_produces_deterministic_result(self):
        reasons = agent._merge_llm_decision(None, ["motif initial"], known_signal_types=set())
        assert "motif initial" in reasons
        assert any("LLM indisponible" in r for r in reasons)
