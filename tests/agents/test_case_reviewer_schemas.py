"""Tests de validation des schémas Pydantic du Case Reviewer Agent.

Couvre ``schemas/results.py`` : ``CaseReviewerResult`` (enveloppe générique,
même patron que ``ClinicalConsistencyResult``/``FraudDetectionResult``) et
``CaseReviewerResultPayload`` (détail métier : recommandation, justification,
contradictions, risques, questions humain).

Vérifie en particulier :
  - extra='forbid' sur les deux modèles, y compris dans le sous-modèle imbriqué.
  - que ``llm_trace`` est obligatoire et non optionnel (règle projet
    fail-closed, même garantie que Clinical/Fraud — voir
    ``tests/orchestrator/test_llm_trace_contract.py``).
  - qu'``evidence_ids``/``justification``/``risks``/``human_review_reasons``
    ne peuvent jamais porter de document brut, de texte OCR complet ou de
    secret (garde-fou partagé ``_reject_unstructured_content``).
  - **qu'aucune décision finale automatique n'est possible dans le schéma** :
    ``status`` est verrouillé à ``NEEDS_REVIEW`` et ``human_review_required``
    est verrouillé à ``True`` — toute tentative de construction avec une
    autre valeur lève une ``ValidationError``, quelle que soit
    l'implémentation qui tente de la construire.
  - que ``human_review_reasons`` (questions humain) ne peut jamais être vide :
    une revue humaine toujours obligatoire doit toujours porter un motif.
  - la sérialisation JSON et le round-trip (``model_dump()``/``model_validate()``),
    nécessaires à l'orchestrateur et aux checkpoints LangGraph.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.domain import Recommendation, VerificationStatus
from schemas.results import (
    CaseReviewerResult,
    CaseReviewerResultPayload,
    DisagreementPoint,
    LlmMetadata,
    StructuredError,
)


def _llm_trace() -> LlmMetadata:
    return LlmMetadata(model_name="test-llm", prompt_version="test")


def _payload(**overrides) -> CaseReviewerResultPayload:
    fields = {
        "recommendation": Recommendation.PENDING,
        "human_review_reasons": ["Validation humaine obligatoire avant toute décision finale."],
    }
    fields.update(overrides)
    return CaseReviewerResultPayload(**fields)


# ── CaseReviewerResultPayload (détail métier) ────────────────────────────────


def test_payload_accepts_minimal_valid_payload():
    payload = _payload()
    assert payload.recommendation is Recommendation.PENDING
    assert payload.justification == []
    assert payload.disagreements == []
    assert payload.risks == []


def test_payload_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        CaseReviewerResultPayload(
            recommendation=Recommendation.APPROVE,
            human_review_reasons=["Motif."],
            unexpected=True,
        )


def test_payload_requires_recommendation():
    with pytest.raises(ValidationError):
        CaseReviewerResultPayload(human_review_reasons=["Motif."])


def test_payload_rejects_empty_human_review_reasons():
    """Questions humain : une revue toujours obligatoire doit toujours porter
    au moins un motif explicite — jamais silencieuse."""
    with pytest.raises(ValidationError):
        CaseReviewerResultPayload(recommendation=Recommendation.APPROVE, human_review_reasons=[])


def test_payload_requires_human_review_reasons_field():
    with pytest.raises(ValidationError):
        CaseReviewerResultPayload(recommendation=Recommendation.APPROVE)


def test_payload_accepts_disagreements():
    payload = _payload(
        disagreements=[
            DisagreementPoint(agent="fhir_validator", field="status", expected="PASS", observed="FAIL")
        ]
    )
    assert payload.disagreements[0].agent == "fhir_validator"


def test_payload_accepts_risks():
    payload = _payload(risks=["Plafond de couverture dépassé.", "Score de risque de fraude élevé (0.80)."])
    assert len(payload.risks) == 2


def test_payload_justification_rejects_raw_document_dump():
    raw_dump = "\n\n".join(f"ligne {i}" for i in range(10))
    with pytest.raises(ValidationError):
        _payload(justification=[raw_dump])


def test_payload_risks_reject_secret_hint():
    with pytest.raises(ValidationError):
        _payload(risks=["password: hunter2"])


def test_payload_human_review_reasons_reject_absolute_path():
    with pytest.raises(ValidationError):
        CaseReviewerResultPayload(
            recommendation=Recommendation.APPROVE,
            human_review_reasons=["/etc/passwd contient le motif"],
        )


# ── CaseReviewerResult (enveloppe) ───────────────────────────────────────────


def test_case_reviewer_result_accepts_full_nested_structure():
    result = CaseReviewerResult(
        case_id="CLM-0001",
        llm_trace=_llm_trace(),
        confidence=0.6,
        evidence_ids=["EVID-abc0000001"],
        result_payload=_payload(
            recommendation=Recommendation.REJECT,
            justification=["Score de fraude critique."],
            risks=["Score de risque de fraude élevé (0.90)."],
        ),
    )
    assert result.status is VerificationStatus.NEEDS_REVIEW
    assert result.human_review_required is True
    assert result.result_payload.recommendation is Recommendation.REJECT
    assert result.model_dump_json()


def test_case_reviewer_result_defaults_status_confidence_and_errors():
    result = CaseReviewerResult(case_id="CLM-0001", llm_trace=_llm_trace(), result_payload=_payload())
    assert result.status is VerificationStatus.NEEDS_REVIEW
    assert result.confidence == 1.0
    assert result.errors == []
    assert result.evidence_ids == []
    assert result.human_review_required is True


def test_case_reviewer_result_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        CaseReviewerResult(
            case_id="CLM-0001", llm_trace=_llm_trace(), result_payload=_payload(), unexpected=True
        )


def test_case_reviewer_result_forbids_unknown_fields_in_nested_payload():
    """Un champ inconnu dans result_payload doit aussi être refusé — l'enveloppe
    ne doit jamais devenir une échappatoire à extra='forbid'."""
    with pytest.raises(ValidationError):
        CaseReviewerResult(
            case_id="CLM-0001",
            llm_trace=_llm_trace(),
            result_payload={"recommendation": "APPROVE", "human_review_reasons": ["m"], "unexpected": True},
        )


def test_case_reviewer_result_rejects_confidence_out_of_bounds():
    with pytest.raises(ValidationError):
        CaseReviewerResult(
            case_id="CLM-0001", llm_trace=_llm_trace(), result_payload=_payload(), confidence=1.5
        )
    with pytest.raises(ValidationError):
        CaseReviewerResult(
            case_id="CLM-0001", llm_trace=_llm_trace(), result_payload=_payload(), confidence=-0.1
        )


def test_case_reviewer_result_requires_result_payload():
    with pytest.raises(ValidationError):
        CaseReviewerResult(case_id="CLM-0001", llm_trace=_llm_trace())


class TestLlmTraceIsMandatory:
    """Règle projet fail-closed : un résultat sans trace LLM n'est jamais
    une exécution valide (voir tests/orchestrator/test_llm_trace_contract.py)."""

    def test_llm_trace_is_a_required_field(self):
        assert CaseReviewerResult.model_fields["llm_trace"].is_required()

    def test_llm_trace_does_not_accept_none(self):
        with pytest.raises(ValidationError):
            CaseReviewerResult(case_id="CLM-0001", llm_trace=None, result_payload=_payload())

    def test_construction_without_llm_trace_is_rejected(self):
        with pytest.raises(ValidationError):
            CaseReviewerResult(case_id="CLM-0001", result_payload=_payload())


class TestErrorsField:
    def test_errors_accepts_structured_error(self):
        result = CaseReviewerResult(
            case_id="CLM-0001",
            llm_trace=_llm_trace(),
            result_payload=_payload(),
            errors=[StructuredError(code="LLM_UNAVAILABLE", message="LLM indisponible.")],
        )
        assert result.errors[0].code == "LLM_UNAVAILABLE"

    def test_errors_rejects_free_text(self):
        with pytest.raises(ValidationError):
            CaseReviewerResult(
                case_id="CLM-0001",
                llm_trace=_llm_trace(),
                result_payload=_payload(),
                errors=["texte libre non structuré"],
            )


class TestEvidenceIdsField:
    """CaseReviewerResult ne porte aucun objet de preuve propre — evidence_ids
    agrège des identifiants déjà validés en amont (voir agent.py::_collect_evidence_ids).
    Seul le contenu (pas la référence croisée) est validé ici."""

    def test_accepts_plain_evidence_ids(self):
        result = CaseReviewerResult(
            case_id="CLM-0001",
            llm_trace=_llm_trace(),
            evidence_ids=["EVID-abc0000001", "EVID-def0000002"],
            result_payload=_payload(),
        )
        assert len(result.evidence_ids) == 2

    def test_rejects_secret_hint_in_evidence_ids(self):
        with pytest.raises(ValidationError):
            CaseReviewerResult(
                case_id="CLM-0001",
                llm_trace=_llm_trace(),
                evidence_ids=["api_key=sk-secret"],
                result_payload=_payload(),
            )


# ── Interdiction structurelle de toute décision finale automatique ──────────


class TestNoAutomaticFinalDecisionInSchema:
    """Cœur de la garantie demandée : ni ``status`` ni ``human_review_required``
    ne peuvent jamais représenter une décision finale automatique, quelle que
    soit l'implémentation (réelle, injectée, ou de test) qui construit l'objet."""

    def test_status_defaults_to_needs_review(self):
        result = CaseReviewerResult(case_id="CLM-0001", llm_trace=_llm_trace(), result_payload=_payload())
        assert result.status is VerificationStatus.NEEDS_REVIEW

    @pytest.mark.parametrize(
        "status",
        [
            VerificationStatus.PASS,
            VerificationStatus.FAIL,
            VerificationStatus.PENDING,
            VerificationStatus.NOT_EVALUATED,
        ],
    )
    def test_status_rejects_any_value_other_than_needs_review(self, status):
        with pytest.raises(ValidationError):
            CaseReviewerResult(
                case_id="CLM-0001", status=status, llm_trace=_llm_trace(), result_payload=_payload()
            )

    def test_human_review_required_defaults_to_true(self):
        result = CaseReviewerResult(case_id="CLM-0001", llm_trace=_llm_trace(), result_payload=_payload())
        assert result.human_review_required is True

    def test_human_review_required_rejects_false(self):
        with pytest.raises(ValidationError):
            CaseReviewerResult(
                case_id="CLM-0001",
                llm_trace=_llm_trace(),
                human_review_required=False,
                result_payload=_payload(),
            )

    @pytest.mark.parametrize("recommendation", [Recommendation.APPROVE, Recommendation.REJECT])
    def test_no_recommendation_can_bypass_the_lock(self, recommendation):
        """Même une pré-recommandation tranchée (APPROVE/REJECT) ne débloque
        jamais status/human_review_required — la révisabilité est
        inconditionnelle, indépendante du contenu de result_payload."""
        result = CaseReviewerResult(
            case_id="CLM-0001",
            llm_trace=_llm_trace(),
            result_payload=_payload(recommendation=recommendation),
        )
        assert result.status is VerificationStatus.NEEDS_REVIEW
        assert result.human_review_required is True


def test_case_reviewer_result_round_trips_with_nested_models():
    """Round-trip requis par l'orchestrateur (revalidation d'un dict déjà
    issu de model_dump()) et par les checkpoints LangGraph."""
    result = CaseReviewerResult(
        case_id="CLM-0001",
        llm_trace=_llm_trace(),
        confidence=0.55,
        evidence_ids=["EVID-abc0000001"],
        errors=[StructuredError(code="LLM_UNAVAILABLE", message="LLM indisponible.")],
        result_payload=_payload(
            recommendation=Recommendation.PENDING,
            justification=["Synthèse en attente."],
            disagreements=[
                DisagreementPoint(agent="coding", field="status", expected="PASS", observed="NEEDS_REVIEW")
            ],
            risks=["Pré-autorisation requise — à confirmer par l'humain."],
        ),
    )
    dumped = result.model_dump(mode="json")
    restored = CaseReviewerResult.model_validate(dumped)
    assert restored == result
