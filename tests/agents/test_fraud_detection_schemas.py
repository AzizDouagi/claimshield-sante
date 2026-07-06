"""Tests de validation des schémas Pydantic du Fraud Detection Agent.

Couvre ``agents/fraud_detection_agent/schemas.py`` (re-export public) et les
modèles sources de ``schemas/results.py`` : ``FraudDetectionResult``
(enveloppe générique), ``FraudResultPayload`` (détail métier),
``FraudSignal``, ``FraudEvidence``, ``FraudEvidenceSource``.

Même patron que ``tests/agents/test_clinical_consistency_schemas.py`` — voir
sa docstring pour le détail des garanties vérifiées (extra='forbid' à tous
les niveaux, aucune sortie libre non structurée, aucun document brut/OCR
complet/prompt complet, trace LLM obligatoire, evidence_ids jamais inventés,
round-trip JSON).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from agents.fraud_detection_agent.schemas import (
    FraudDetectionResult,
    FraudEvidence,
    FraudEvidenceSource,
    FraudResultPayload,
    FraudSignal,
    LlmFraudDecision,
)
from schemas.domain import SeverityLevel, VerificationStatus
from schemas.results import LlmMetadata, StructuredError


def _evidence(**overrides) -> FraudEvidence:
    payload = {
        "source": FraudEvidenceSource.IDENTITY_COVERAGE,
        "field": "identity.status",
        "document_reference": "identity_coverage_result",
        "value": "FAIL",
    }
    payload.update(overrides)
    return FraudEvidence(**payload)


def _signal(**overrides) -> FraudSignal:
    payload = {
        "signal_type": "IDENTITY_MISMATCH",
        "description": "Identité patient non concordante.",
        "risk_contribution": 0.4,
        "evidence": [_evidence()],
    }
    payload.update(overrides)
    return FraudSignal(**payload)


def _llm_trace() -> LlmMetadata:
    return LlmMetadata(model_name="test-llm", prompt_version="test")


# ── FraudEvidence (preuves) ──────────────────────────────────────────────────


def test_evidence_accepts_minimal_valid_payload():
    evidence = _evidence()
    assert evidence.source is FraudEvidenceSource.IDENTITY_COVERAGE
    assert evidence.field == "identity.status"
    assert evidence.value == "FAIL"


def test_evidence_document_reference_is_optional():
    evidence = FraudEvidence(source=FraudEvidenceSource.MEDICAL_CODING, field="status", value="FAIL")
    assert evidence.document_reference is None


def test_evidence_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        _evidence(unexpected="oops")


def test_evidence_rejects_empty_field():
    with pytest.raises(ValidationError):
        _evidence(field="")


def test_evidence_rejects_empty_value():
    with pytest.raises(ValidationError):
        _evidence(value="")


def test_evidence_rejects_unknown_source():
    with pytest.raises(ValidationError):
        _evidence(source="not_a_real_source")


def test_evidence_accepts_all_four_sources():
    assert _evidence(source=FraudEvidenceSource.OCR_EXTRACTION, field="confidence_score", value="0.2")
    assert _evidence(source=FraudEvidenceSource.MEDICAL_CODING, field="status", value="FAIL")
    assert _evidence(source=FraudEvidenceSource.IDENTITY_COVERAGE, field="coverage.status", value="FAIL")
    assert _evidence(
        source=FraudEvidenceSource.DUPLICATE_INDEX, field="matched_case_id", value="CLM-0001"
    )


def test_evidence_has_an_auto_generated_id():
    evidence = _evidence()
    assert evidence.evidence_id
    assert evidence.evidence_id != _evidence().evidence_id  # jamais deux id identiques


def test_evidence_is_json_serializable():
    dumped = _evidence().model_dump(mode="json")
    assert dumped["source"] == "identity_coverage"
    assert dumped["evidence_id"]


def test_evidence_round_trips_through_dump_and_validate():
    evidence = _evidence()
    restored = FraudEvidence.model_validate(evidence.model_dump())
    assert restored == evidence


# ── FraudEvidence — jamais de document brut, OCR complet ou prompt ──────────


class TestEvidenceRejectsUnstructuredContent:
    def test_rejects_absolute_path_in_value(self):
        with pytest.raises(ValidationError):
            _evidence(value="/etc/passwd")

    def test_rejects_secret_hint_in_value(self):
        with pytest.raises(ValidationError):
            _evidence(value="api_key: sk-1234567890")

    def test_rejects_multiline_dump_in_value(self):
        raw_dump = "ligne 1\n\nligne 2\n\nligne 3\n\nligne 4"
        with pytest.raises(ValidationError):
            _evidence(value=raw_dump)

    def test_accepts_short_single_line_value(self):
        evidence = _evidence(value="valeur courte")
        assert evidence.value == "valeur courte"


# ── FraudSignal (signaux) ─────────────────────────────────────────────────────


def test_signal_accepts_valid_payload():
    signal = _signal()
    assert signal.signal_type == "IDENTITY_MISMATCH"
    assert signal.severity is SeverityLevel.MEDIUM  # défaut
    assert len(signal.evidence) == 1


def test_signal_rejects_empty_evidence():
    """Garantie centrale : un signal de fraude ne peut jamais être une
    affirmation non appuyée — il combine uniquement des preuves déjà
    validées par d'autres agents."""
    with pytest.raises(ValidationError):
        FraudSignal(
            signal_type="IDENTITY_MISMATCH",
            description="d",
            risk_contribution=0.4,
            evidence=[],
        )


def test_signal_requires_evidence_field_present():
    with pytest.raises(ValidationError):
        FraudSignal(signal_type="X", description="d", risk_contribution=0.4)


def test_signal_rejects_empty_signal_type():
    with pytest.raises(ValidationError):
        _signal(signal_type="")


def test_signal_rejects_empty_description():
    with pytest.raises(ValidationError):
        _signal(description="")


def test_signal_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        _signal(unexpected_field=True)


def test_signal_rejects_risk_contribution_out_of_bounds():
    with pytest.raises(ValidationError):
        _signal(risk_contribution=1.5)
    with pytest.raises(ValidationError):
        _signal(risk_contribution=-0.1)


def test_signal_severity_rejects_free_text_value():
    with pytest.raises(ValidationError):
        _signal(severity="WARNING")


def test_signal_accepts_explicit_severity():
    signal = _signal(severity=SeverityLevel.CRITICAL)
    assert signal.severity is SeverityLevel.CRITICAL


def test_signal_is_json_serializable():
    dumped = _signal(severity=SeverityLevel.CRITICAL).model_dump(mode="json")
    assert dumped["severity"] == "CRITICAL"
    assert dumped["evidence"][0]["value"] == "FAIL"


# ── FraudResultPayload (détail métier) ───────────────────────────────────────


def test_payload_defaults_are_empty_and_safe():
    payload = FraudResultPayload()
    assert payload.signals == []
    assert payload.reasons == []
    assert payload.duplicate_invoice is None
    assert payload.risk_score == 0.0
    assert payload.threshold_version == "1.0.0"


def test_payload_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        FraudResultPayload(unexpected=True)


def test_payload_rejects_risk_score_out_of_bounds():
    with pytest.raises(ValidationError):
        FraudResultPayload(risk_score=1.5)
    with pytest.raises(ValidationError):
        FraudResultPayload(risk_score=-0.1)


def test_payload_reasons_reject_raw_document_dump():
    raw_dump = "\n\n".join(f"ligne {i}" for i in range(10))
    with pytest.raises(ValidationError):
        FraudResultPayload(reasons=[raw_dump])


def test_payload_reasons_reject_secret_hint():
    with pytest.raises(ValidationError):
        FraudResultPayload(reasons=["password: hunter2"])


# ── FraudDetectionResult (enveloppe) ─────────────────────────────────────────


def test_fraud_result_accepts_full_nested_structure():
    signal = _signal(severity=SeverityLevel.CRITICAL)
    result = FraudDetectionResult(
        case_id="CLM-0001",
        status=VerificationStatus.FAIL,
        llm_trace=_llm_trace(),
        confidence=0.6,
        evidence_ids=[signal.evidence[0].evidence_id],
        result_payload=FraudResultPayload(
            risk_score=0.9,
            signals=[signal],
            reasons=["Risque de fraude élevé."],
        ),
    )
    assert result.result_payload.signals[0].signal_type == "IDENTITY_MISMATCH"
    assert result.result_payload.risk_score == 0.9
    assert result.model_dump_json()


def test_fraud_result_defaults_payload_confidence_and_errors():
    result = FraudDetectionResult(
        case_id="CLM-0001", status=VerificationStatus.PASS, llm_trace=_llm_trace()
    )
    assert result.result_payload.signals == []
    assert result.result_payload.risk_score == 0.0
    assert result.result_payload.duplicate_invoice is None
    assert result.confidence == 1.0
    assert result.errors == []
    assert result.evidence_ids == []
    assert result.human_review_required is False


def test_fraud_result_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        FraudDetectionResult(
            case_id="CLM-0001", status=VerificationStatus.PASS, llm_trace=_llm_trace(), unexpected=True
        )


def test_fraud_result_forbids_unknown_fields_in_nested_payload():
    with pytest.raises(ValidationError):
        FraudDetectionResult(
            case_id="CLM-0001",
            status=VerificationStatus.PASS,
            llm_trace=_llm_trace(),
            result_payload={"unexpected": True},
        )


def test_fraud_result_rejects_confidence_out_of_bounds():
    with pytest.raises(ValidationError):
        FraudDetectionResult(
            case_id="CLM-0001", status=VerificationStatus.PASS, llm_trace=_llm_trace(), confidence=1.5
        )
    with pytest.raises(ValidationError):
        FraudDetectionResult(
            case_id="CLM-0001", status=VerificationStatus.PASS, llm_trace=_llm_trace(), confidence=-0.1
        )


def test_fraud_result_rejects_risk_score_out_of_bounds_via_payload():
    with pytest.raises(ValidationError):
        FraudDetectionResult(
            case_id="CLM-0001",
            status=VerificationStatus.PASS,
            llm_trace=_llm_trace(),
            result_payload=FraudResultPayload(risk_score=1.5),
        )


class TestLlmTraceIsMandatory:
    """Règle projet fail-closed : un résultat sans trace LLM n'est jamais
    une exécution valide (voir tests/orchestrator/test_llm_trace_contract.py)."""

    def test_llm_trace_is_a_required_field(self):
        assert FraudDetectionResult.model_fields["llm_trace"].is_required()

    def test_llm_trace_does_not_accept_none(self):
        with pytest.raises(ValidationError):
            FraudDetectionResult(case_id="CLM-0001", status=VerificationStatus.PASS, llm_trace=None)

    def test_construction_without_llm_trace_is_rejected(self):
        with pytest.raises(ValidationError):
            FraudDetectionResult(case_id="CLM-0001", status=VerificationStatus.PASS)


class TestErrorsField:
    def test_errors_accepts_structured_error(self):
        result = FraudDetectionResult(
            case_id="CLM-0001",
            status=VerificationStatus.FAIL,
            llm_trace=_llm_trace(),
            errors=[StructuredError(code="LLM_UNAVAILABLE", message="LLM indisponible.")],
        )
        assert result.errors[0].code == "LLM_UNAVAILABLE"

    def test_errors_rejects_free_text(self):
        with pytest.raises(ValidationError):
            FraudDetectionResult(
                case_id="CLM-0001",
                status=VerificationStatus.FAIL,
                llm_trace=_llm_trace(),
                errors=["texte libre non structuré"],
            )


class TestEvidenceIdsMustBeReal:
    def test_accepts_evidence_ids_present_in_payload(self):
        signal = _signal()
        result = FraudDetectionResult(
            case_id="CLM-0001",
            status=VerificationStatus.NEEDS_REVIEW,
            llm_trace=_llm_trace(),
            evidence_ids=[signal.evidence[0].evidence_id],
            result_payload=FraudResultPayload(signals=[signal]),
        )
        assert result.evidence_ids == [signal.evidence[0].evidence_id]

    def test_rejects_invented_evidence_id(self):
        """Garantie anti-hallucination : evidence_ids ne peut jamais
        référencer une preuve qui n'existe pas dans result_payload."""
        with pytest.raises(ValidationError):
            FraudDetectionResult(
                case_id="CLM-0001",
                status=VerificationStatus.PASS,
                llm_trace=_llm_trace(),
                evidence_ids=["EVID-invented0"],
            )


class TestHumanReviewRequired:
    def test_defaults_to_false(self):
        result = FraudDetectionResult(
            case_id="CLM-0001", status=VerificationStatus.PASS, llm_trace=_llm_trace()
        )
        assert result.human_review_required is False

    def test_can_be_set_true(self):
        result = FraudDetectionResult(
            case_id="CLM-0001",
            status=VerificationStatus.FAIL,
            llm_trace=_llm_trace(),
            human_review_required=True,
        )
        assert result.human_review_required is True


def test_fraud_result_round_trips_with_nested_models():
    """Round-trip requis par l'orchestrateur (revalidation d'un dict déjà
    issu de model_dump()) et par les checkpoints LangGraph."""
    signal = _signal()
    result = FraudDetectionResult(
        case_id="CLM-0001",
        status=VerificationStatus.NEEDS_REVIEW,
        llm_trace=_llm_trace(),
        evidence_ids=[signal.evidence[0].evidence_id],
        result_payload=FraudResultPayload(risk_score=0.4, signals=[signal]),
    )
    restored = FraudDetectionResult.model_validate(result.model_dump())
    assert restored == result


# ── LlmFraudDecision — jamais de sortie libre non structurée ────────────────


def test_llm_decision_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        LlmFraudDecision(rationale="x", status="PASS")


def test_llm_decision_has_no_authority_field():
    """Vérifie structurellement qu'aucun champ ne pourrait donner au LLM une
    autorité sur le score de risque ou le statut déterministes."""
    forbidden_fields = {"status", "risk_score", "severity", "recommended_status"}
    assert forbidden_fields.isdisjoint(LlmFraudDecision.model_fields.keys())


def test_llm_decision_reasons_must_be_a_list_not_free_text():
    with pytest.raises(ValidationError):
        LlmFraudDecision(reasons="un seul motif en texte libre")


# ── LlmFraudDecision — interdiction des termes accusatoires ─────────────────


class TestAccusatoryLanguageRejected:
    @pytest.mark.parametrize(
        "rationale",
        [
            "La fraude est confirmée sur ce dossier.",
            "Cette fraude est clairement avérée.",
            "Fraude établie sans ambiguïté.",
            "This is a confirmed fraud case.",
            "Fraud confirmed beyond doubt.",
            "Le patient est coupable.",
            "C'est un escroc notoire.",
            "Ce dossier révèle un fraudeur.",
        ],
    )
    def test_rejects_accusatory_rationale(self, rationale):
        with pytest.raises(ValidationError):
            LlmFraudDecision(rationale=rationale)

    def test_rejects_accusatory_reasons(self):
        with pytest.raises(ValidationError):
            LlmFraudDecision(reasons=["Fraude confirmée par recoupement des signaux."])

    @pytest.mark.parametrize(
        "rationale",
        [
            "Risque de fraude potentielle à surveiller.",
            "Signal de fraude non confirmé, à vérifier.",
            "Aucune fraude avérée à ce stade, incertitude élevée.",
            "Pas de fraude prouvée pour le moment.",
            "Le montant demandé semble atypique pour ce type de soin.",
        ],
    )
    def test_accepts_legitimate_risk_language(self, rationale):
        decision = LlmFraudDecision(rationale=rationale)
        assert decision.rationale == rationale

    def test_error_message_names_the_offending_phrase(self):
        with pytest.raises(ValidationError, match="confirmée"):
            LlmFraudDecision(rationale="Fraude confirmée.")
