"""Tests de validation des schémas Pydantic du Clinical Consistency Agent.

Couvre ``agents/clinical_consistency_agent/schemas.py`` (re-export public) et
les modèles sources de ``schemas/results.py`` : ``ClinicalConsistencyResult``
(enveloppe générique), ``ClinicalResultPayload`` (détail métier),
``ClinicalSignal``, ``ClinicalInconsistency``, ``ClinicalEvidence``,
``ClinicalEvidenceSource``.

Vérifie en particulier :
  - extra='forbid' sur tous les modèles, à tous les niveaux d'imbrication
    (enveloppe, payload, signal, incohérence, preuve) — aucun champ inconnu
    n'est jamais accepté.
  - qu'un ``ClinicalSignal``/``ClinicalInconsistency`` ne peut jamais être une
    sortie libre non structurée — chaque signal référence des champs ou des
    documents comparés, chaque incohérence porte au moins une preuve.
  - qu'aucun document brut, texte OCR complet ou prompt complet ne peut être
    injecté dans une preuve ou un motif (garde-fou multi-lignes/secret/chemin
    absolu, partagé avec ``FraudEvidence``).
  - que ``llm_trace`` est obligatoire et non optionnel (règle projet
    fail-closed : un résultat sans trace LLM ne représente jamais une
    exécution valide — voir ``tests/orchestrator/test_llm_trace_contract.py``).
  - que ``evidence_ids`` ne peut jamais référencer une preuve inexistante
    (jamais un identifiant inventé).
  - la sérialisation JSON (``model_dump(mode="json")``) et le round-trip
    (``model_validate(model_dump())``), nécessaires à l'orchestrateur et aux
    checkpoints LangGraph.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from agents.clinical_consistency_agent.schemas import (
    ClinicalConsistencyResult,
    ClinicalEvidence,
    ClinicalEvidenceSource,
    ClinicalInconsistency,
    ClinicalResultPayload,
    ClinicalSignal,
    LlmClinicalDecision,
)
from schemas.domain import SeverityLevel, VerificationStatus
from schemas.results import LlmMetadata, StructuredError


def _evidence(**overrides) -> ClinicalEvidence:
    payload = {
        "source": ClinicalEvidenceSource.OCR_EXTRACTION,
        "field": "procedure_count",
        "document_reference": "INVOICE",
        "value": "3",
    }
    payload.update(overrides)
    return ClinicalEvidence(**payload)


def _llm_trace() -> LlmMetadata:
    return LlmMetadata(model_name="test-llm", prompt_version="test")


# ── ClinicalEvidence (preuves) ────────────────────────────────────────────────


def test_evidence_accepts_minimal_valid_payload():
    evidence = _evidence()
    assert evidence.source is ClinicalEvidenceSource.OCR_EXTRACTION
    assert evidence.field == "procedure_count"
    assert evidence.value == "3"


def test_evidence_document_reference_is_optional():
    evidence = ClinicalEvidence(
        source=ClinicalEvidenceSource.MEDICAL_CODING, field="status", value="FAIL"
    )
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


def test_evidence_has_an_auto_generated_id():
    evidence = _evidence()
    assert evidence.evidence_id
    assert evidence.evidence_id != _evidence().evidence_id  # jamais deux id identiques


def test_evidence_is_json_serializable():
    dumped = _evidence().model_dump(mode="json")
    assert dumped["source"] == "ocr_extraction"
    assert dumped["document_reference"] == "INVOICE"
    assert dumped["evidence_id"]


def test_evidence_round_trips_through_dump_and_validate():
    evidence = _evidence()
    restored = ClinicalEvidence.model_validate(evidence.model_dump())
    assert restored == evidence


# ── ClinicalEvidence — jamais de document brut, OCR complet ou prompt ───────


class TestEvidenceRejectsUnstructuredContent:
    def test_rejects_absolute_path_in_value(self):
        with pytest.raises(ValidationError):
            _evidence(value="/etc/passwd")

    def test_rejects_secret_hint_in_value(self):
        with pytest.raises(ValidationError):
            _evidence(value="api_key: sk-1234567890")

    def test_rejects_multiline_dump_in_value(self):
        """Un texte OCR complet ou un document brut se manifeste comme un
        contenu multi-lignes — jamais accepté dans une valeur de preuve."""
        raw_dump = "ligne 1\n\nligne 2\n\nligne 3\n\nligne 4"
        with pytest.raises(ValidationError):
            _evidence(value=raw_dump)

    def test_accepts_short_single_line_value(self):
        evidence = _evidence(value="ligne unique courte")
        assert evidence.value == "ligne unique courte"


# ── ClinicalSignal (signaux) ──────────────────────────────────────────────────


def test_signal_accepts_fields_compared_only():
    signal = ClinicalSignal(
        signal_type="MISSING_SERVICE_DATE",
        description="Date de service absente.",
        fields_compared=["service_date"],
    )
    assert signal.severity is SeverityLevel.MEDIUM  # défaut
    assert signal.documents_compared == []


def test_signal_accepts_documents_compared_only():
    signal = ClinicalSignal(
        signal_type="DOCUMENT_TYPE_MISMATCH",
        description="Type de document inattendu.",
        documents_compared=["INVOICE", "PRESCRIPTION"],
    )
    assert signal.fields_compared == []


def test_signal_rejects_no_fields_and_no_documents_compared():
    """Garantie centrale : un signal ne peut jamais être une description
    libre sans référence structurée à ce qui a été comparé."""
    with pytest.raises(ValidationError):
        ClinicalSignal(
            signal_type="VAGUE_SIGNAL",
            description="Quelque chose ne va pas.",
        )


def test_signal_rejects_empty_signal_type():
    with pytest.raises(ValidationError):
        ClinicalSignal(signal_type="", description="d", fields_compared=["x"])


def test_signal_rejects_empty_description():
    with pytest.raises(ValidationError):
        ClinicalSignal(signal_type="X", description="", fields_compared=["x"])


def test_signal_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        ClinicalSignal(
            signal_type="X",
            description="d",
            fields_compared=["x"],
            unexpected_field=True,
        )


def test_signal_severity_rejects_free_text_value():
    """La sévérité est un SeverityLevel contrôlé — jamais une chaîne libre
    telle que l'ancien 'WARNING' non structuré."""
    with pytest.raises(ValidationError):
        ClinicalSignal(
            signal_type="X",
            description="d",
            fields_compared=["x"],
            severity="WARNING",
        )


def test_signal_accepts_evidence_referencing_compared_fields():
    signal = ClinicalSignal(
        signal_type="PROCEDURE_CODING_COUNT_MISMATCH",
        description="Écart entre actes facturés et codes résolus.",
        fields_compared=["procedure_count", "coding_result.codings"],
        evidence=[
            _evidence(field="procedure_count", value="5"),
            _evidence(source=ClinicalEvidenceSource.MEDICAL_CODING, field="codings", value="3"),
        ],
        severity=SeverityLevel.CRITICAL,
    )
    assert len(signal.evidence) == 2
    assert signal.evidence[0].field in signal.fields_compared


def test_signal_is_json_serializable():
    signal = ClinicalSignal(
        signal_type="X", description="d", fields_compared=["x"], severity=SeverityLevel.CRITICAL
    )
    dumped = signal.model_dump(mode="json")
    assert dumped["severity"] == "CRITICAL"


# ── ClinicalInconsistency (incohérences) ─────────────────────────────────────


def test_inconsistency_accepts_valid_payload_with_evidence():
    inconsistency = ClinicalInconsistency(
        inconsistency_type="PROCEDURE_COUNT_MISMATCH",
        expected="5",
        observed="3",
        severity=SeverityLevel.CRITICAL,
        evidence=[_evidence()],
    )
    assert inconsistency.expected == "5"
    assert inconsistency.observed == "3"


def test_inconsistency_rejects_empty_evidence():
    """Une incohérence ne peut jamais être affirmée sans au moins une preuve
    structurée — jamais un simple constat non appuyé."""
    with pytest.raises(ValidationError):
        ClinicalInconsistency(
            inconsistency_type="X", expected="a", observed="b", evidence=[]
        )


def test_inconsistency_requires_evidence_field_present():
    with pytest.raises(ValidationError):
        ClinicalInconsistency(inconsistency_type="X", expected="a", observed="b")


def test_inconsistency_rejects_empty_expected_or_observed():
    with pytest.raises(ValidationError):
        ClinicalInconsistency(
            inconsistency_type="X", expected="", observed="b", evidence=[_evidence()]
        )
    with pytest.raises(ValidationError):
        ClinicalInconsistency(
            inconsistency_type="X", expected="a", observed="", evidence=[_evidence()]
        )


def test_inconsistency_rejects_raw_document_in_expected_or_observed():
    raw_dump = "a\n\nb\n\nc\n\nd"
    with pytest.raises(ValidationError):
        ClinicalInconsistency(
            inconsistency_type="X", expected=raw_dump, observed="b", evidence=[_evidence()]
        )


def test_inconsistency_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        ClinicalInconsistency(
            inconsistency_type="X",
            expected="a",
            observed="b",
            evidence=[_evidence()],
            unexpected="oops",
        )


def test_inconsistency_default_severity_is_medium():
    inconsistency = ClinicalInconsistency(
        inconsistency_type="X", expected="a", observed="b", evidence=[_evidence()]
    )
    assert inconsistency.severity is SeverityLevel.MEDIUM


def test_inconsistency_is_json_serializable():
    inconsistency = ClinicalInconsistency(
        inconsistency_type="X", expected="a", observed="b", evidence=[_evidence()]
    )
    dumped = inconsistency.model_dump(mode="json")
    assert dumped["evidence"][0]["value"] == "3"


# ── ClinicalResultPayload (détail métier) ────────────────────────────────────


def test_payload_defaults_are_empty_and_safe():
    payload = ClinicalResultPayload()
    assert payload.signals == []
    assert payload.inconsistencies == []
    assert payload.reasons == []
    assert payload.procedure_count is None


def test_payload_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        ClinicalResultPayload(unexpected=True)


def test_payload_rejects_negative_counts():
    with pytest.raises(ValidationError):
        ClinicalResultPayload(procedure_count=-1)
    with pytest.raises(ValidationError):
        ClinicalResultPayload(medication_count=-1)


def test_payload_reasons_reject_raw_document_dump():
    raw_dump = "\n\n".join(f"ligne {i}" for i in range(10))
    with pytest.raises(ValidationError):
        ClinicalResultPayload(reasons=[raw_dump])


def test_payload_reasons_reject_secret_hint():
    with pytest.raises(ValidationError):
        ClinicalResultPayload(reasons=["password: hunter2"])


# ── ClinicalConsistencyResult (enveloppe) ────────────────────────────────────


def test_clinical_result_accepts_full_nested_structure():
    signal = ClinicalSignal(
        signal_type="MISSING_PRESCRIPTION_REFERENCE",
        description="Médicament facturé sans ordonnance.",
        fields_compared=["medication_count", "prescription_number"],
        evidence=[_evidence(field="medication_count", value="2")],
        severity=SeverityLevel.CRITICAL,
    )
    inconsistency = ClinicalInconsistency(
        inconsistency_type="MISSING_PRESCRIPTION_REFERENCE",
        expected="prescription_number renseigné",
        observed="prescription_number absent",
        severity=SeverityLevel.CRITICAL,
        evidence=[_evidence(field="prescription_number", value="absent")],
    )
    result = ClinicalConsistencyResult(
        case_id="CLM-0001",
        status=VerificationStatus.FAIL,
        llm_trace=_llm_trace(),
        confidence=0.7,
        result_payload=ClinicalResultPayload(
            procedure_count=1,
            medication_count=2,
            prescription_required=True,
            signals=[signal],
            inconsistencies=[inconsistency],
            reasons=["Incohérence clinique critique détectée."],
        ),
    )
    assert result.result_payload.signals[0].signal_type == "MISSING_PRESCRIPTION_REFERENCE"
    assert result.result_payload.inconsistencies[0].observed == "prescription_number absent"
    assert result.model_dump_json()


def test_clinical_result_defaults_payload_confidence_and_errors():
    result = ClinicalConsistencyResult(
        case_id="CLM-0001", status=VerificationStatus.PASS, llm_trace=_llm_trace()
    )
    assert result.result_payload.signals == []
    assert result.result_payload.inconsistencies == []
    assert result.confidence == 1.0
    assert result.errors == []
    assert result.evidence_ids == []
    assert result.human_review_required is False


def test_clinical_result_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        ClinicalConsistencyResult(
            case_id="CLM-0001", status=VerificationStatus.PASS, llm_trace=_llm_trace(), unexpected=True
        )


def test_clinical_result_forbids_unknown_fields_in_nested_payload():
    """Un champ inconnu dans result_payload doit aussi être refusé — l'enveloppe
    ne doit jamais devenir une échappatoire à extra='forbid'."""
    with pytest.raises(ValidationError):
        ClinicalConsistencyResult(
            case_id="CLM-0001",
            status=VerificationStatus.PASS,
            llm_trace=_llm_trace(),
            result_payload={"unexpected": True},
        )


def test_clinical_result_rejects_confidence_out_of_bounds():
    with pytest.raises(ValidationError):
        ClinicalConsistencyResult(
            case_id="CLM-0001", status=VerificationStatus.PASS, llm_trace=_llm_trace(), confidence=1.5
        )
    with pytest.raises(ValidationError):
        ClinicalConsistencyResult(
            case_id="CLM-0001", status=VerificationStatus.PASS, llm_trace=_llm_trace(), confidence=-0.1
        )


def test_clinical_result_rejects_negative_counts_via_payload():
    with pytest.raises(ValidationError):
        ClinicalConsistencyResult(
            case_id="CLM-0001",
            status=VerificationStatus.PASS,
            llm_trace=_llm_trace(),
            result_payload=ClinicalResultPayload(procedure_count=-1),
        )


class TestLlmTraceIsMandatory:
    """Règle projet fail-closed : un résultat sans trace LLM n'est jamais
    une exécution valide (voir tests/orchestrator/test_llm_trace_contract.py)."""

    def test_llm_trace_is_a_required_field(self):
        assert ClinicalConsistencyResult.model_fields["llm_trace"].is_required()

    def test_llm_trace_does_not_accept_none(self):
        with pytest.raises(ValidationError):
            ClinicalConsistencyResult(
                case_id="CLM-0001", status=VerificationStatus.PASS, llm_trace=None
            )

    def test_construction_without_llm_trace_is_rejected(self):
        with pytest.raises(ValidationError):
            ClinicalConsistencyResult(case_id="CLM-0001", status=VerificationStatus.PASS)


class TestErrorsField:
    def test_errors_accepts_structured_error(self):
        result = ClinicalConsistencyResult(
            case_id="CLM-0001",
            status=VerificationStatus.FAIL,
            llm_trace=_llm_trace(),
            errors=[StructuredError(code="LLM_UNAVAILABLE", message="LLM indisponible.")],
        )
        assert result.errors[0].code == "LLM_UNAVAILABLE"

    def test_errors_rejects_free_text(self):
        with pytest.raises(ValidationError):
            ClinicalConsistencyResult(
                case_id="CLM-0001",
                status=VerificationStatus.FAIL,
                llm_trace=_llm_trace(),
                errors=["texte libre non structuré"],
            )


class TestEvidenceIdsMustBeReal:
    def test_accepts_evidence_ids_present_in_payload(self):
        evidence = _evidence()
        signal = ClinicalSignal(
            signal_type="X", description="d", fields_compared=["x"], evidence=[evidence]
        )
        result = ClinicalConsistencyResult(
            case_id="CLM-0001",
            status=VerificationStatus.NEEDS_REVIEW,
            llm_trace=_llm_trace(),
            evidence_ids=[evidence.evidence_id],
            result_payload=ClinicalResultPayload(signals=[signal]),
        )
        assert result.evidence_ids == [evidence.evidence_id]

    def test_rejects_invented_evidence_id(self):
        """Garantie anti-hallucination : evidence_ids ne peut jamais
        référencer une preuve qui n'existe pas dans result_payload."""
        with pytest.raises(ValidationError):
            ClinicalConsistencyResult(
                case_id="CLM-0001",
                status=VerificationStatus.PASS,
                llm_trace=_llm_trace(),
                evidence_ids=["EVID-invented0"],
            )


class TestHumanReviewRequired:
    def test_defaults_to_false(self):
        result = ClinicalConsistencyResult(
            case_id="CLM-0001", status=VerificationStatus.PASS, llm_trace=_llm_trace()
        )
        assert result.human_review_required is False

    def test_can_be_set_true(self):
        result = ClinicalConsistencyResult(
            case_id="CLM-0001",
            status=VerificationStatus.FAIL,
            llm_trace=_llm_trace(),
            human_review_required=True,
        )
        assert result.human_review_required is True


def test_clinical_result_round_trips_with_nested_models():
    """Round-trip requis par l'orchestrateur (revalidation d'un dict déjà
    issu de model_dump()) et par les checkpoints LangGraph."""
    evidence = _evidence(field="service_date", value="absent")
    inconsistency_evidence = _evidence()
    result = ClinicalConsistencyResult(
        case_id="CLM-0001",
        status=VerificationStatus.NEEDS_REVIEW,
        llm_trace=_llm_trace(),
        evidence_ids=[evidence.evidence_id, inconsistency_evidence.evidence_id],
        result_payload=ClinicalResultPayload(
            signals=[
                ClinicalSignal(
                    signal_type="MISSING_SERVICE_DATE",
                    description="d",
                    fields_compared=["service_date"],
                    evidence=[evidence],
                )
            ],
            inconsistencies=[
                ClinicalInconsistency(
                    inconsistency_type="X",
                    expected="a",
                    observed="b",
                    evidence=[inconsistency_evidence],
                )
            ],
        ),
    )
    restored = ClinicalConsistencyResult.model_validate(result.model_dump())
    assert restored == result


# ── LlmClinicalDecision — jamais de sortie libre non structurée ──────────────


def test_llm_decision_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        LlmClinicalDecision(clinical_context="x", status="PASS")


def test_llm_decision_has_no_authority_field():
    """Vérifie structurellement qu'aucun champ ne pourrait donner au LLM
    une autorité sur le statut ou la sévérité déterministes."""
    forbidden_fields = {"status", "severity", "recommended_status", "confidence"}
    assert forbidden_fields.isdisjoint(LlmClinicalDecision.model_fields.keys())


def test_llm_decision_reasons_must_be_a_list_not_free_text():
    with pytest.raises(ValidationError):
        LlmClinicalDecision(reasons="un seul motif en texte libre")


# ── LlmClinicalDecision — preuves, incohérences, confiance, revue ───────────


def test_llm_decision_defaults_are_empty_and_safe():
    decision = LlmClinicalDecision()
    assert decision.referenced_evidence_ids == []
    assert decision.acknowledged_inconsistencies == []
    assert decision.llm_confidence is None
    assert decision.suggests_human_review is False


def test_llm_decision_accepts_referenced_evidence_and_inconsistencies():
    decision = LlmClinicalDecision(
        referenced_evidence_ids=["EVID-abc123"],
        acknowledged_inconsistencies=["PROCEDURE_CODING_COUNT_MISMATCH"],
        llm_confidence=0.8,
        suggests_human_review=True,
    )
    assert decision.referenced_evidence_ids == ["EVID-abc123"]
    assert decision.suggests_human_review is True


def test_llm_decision_rejects_confidence_out_of_bounds():
    with pytest.raises(ValidationError):
        LlmClinicalDecision(llm_confidence=1.5)
    with pytest.raises(ValidationError):
        LlmClinicalDecision(llm_confidence=-0.1)


def test_llm_decision_acknowledged_inconsistencies_reject_secret_hint():
    with pytest.raises(ValidationError):
        LlmClinicalDecision(acknowledged_inconsistencies=["password: hunter2"])


def test_llm_decision_acknowledged_inconsistencies_reject_absolute_path():
    with pytest.raises(ValidationError):
        LlmClinicalDecision(acknowledged_inconsistencies=["/etc/passwd"])


def test_llm_decision_has_no_treatment_or_document_field():
    """Vérifie structurellement qu'aucun champ ne pourrait porter une
    recommandation de traitement ou un document inventé."""
    forbidden_fields = {"treatment", "recommended_treatment", "document", "documents", "diagnosis"}
    assert forbidden_fields.isdisjoint(LlmClinicalDecision.model_fields.keys())
