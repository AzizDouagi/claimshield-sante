"""Tests de agents/eligibility_agent (V2) — Phase V2-4.

`eligibility_agent.run()` délègue directement à
`agents.identity_coverage_agent.agent.run()` (V1, non modifié) — ces tests
vérifient la parité fonctionnelle (mêmes verdicts identité/couverture) et
la traduction correcte vers `schemas.v2_results.EligibilityResult`.
"""
from __future__ import annotations

from unittest.mock import Mock

from agents.eligibility_agent.agent import _normalize_extracted_fields, node, run
from schemas.domain import VerificationStatus
from schemas.results import ExtractedField

_LEGACY_CONTRACT = {
    "patient_id": "PAT-001",
    "payer_name": "Cigna Health",
    "coverage_rate": "0.80",
    "policy_active": True,
    "coverage_start_date": "2026-01-01",
    "coverage_end_date": "2026-12-31",
    "ceiling_remaining": "500.00",
    "preauthorization_required": False,
}

_STRUCTURED_CONTRACT = {
    "patient_id": "PAT-001",
    "status": "active",
    "currency": "USD",
    "start_date": "2026-01-01",
    "end_date": "2026-12-31",
    "annual_limit": "5000.00",
    "covered_procedure_codes": ["PROC-CONSULT"],
    "excluded_procedure_codes": [],
}

_EXTRACTED_FIELDS = {
    "patient_id": "PAT-001",
    "patient_name": "Jane Doe",
    "payer_name": "Cigna Health",
    "amount_requested": "100.00",
    "service_date": "2026-06-15",
}


class TestParityWithV1:
    def test_legacy_coverage_path_pass(self):
        result = run(
            case_id="CLM-4001",
            dossier_patient_id="PAT-001",
            contract=_LEGACY_CONTRACT,
            extracted_fields=_EXTRACTED_FIELDS,
        )
        assert result.identity.status is VerificationStatus.PASS
        assert result.coverage.status is VerificationStatus.PASS
        assert result.status is VerificationStatus.PASS

    def test_structured_coverage_path_active(self):
        result = run(
            case_id="CLM-4002",
            dossier_patient_id="PAT-001",
            contract=_STRUCTURED_CONTRACT,
            procedure_codes=["PROC-CONSULT"],
            extracted_fields={
                **_EXTRACTED_FIELDS,
                "total_amount": "150.00",
                "currency": "USD",
            },
        )
        assert result.identity.status is VerificationStatus.PASS
        assert result.coverage.status is VerificationStatus.PASS
        assert result.status is VerificationStatus.PASS

    def test_run_delegates_to_v1_agent(self, monkeypatch):
        spy = Mock(wraps=None)
        import agents.identity_coverage_agent.agent as v1_agent

        original = v1_agent.run
        spy.side_effect = original
        monkeypatch.setattr("agents.eligibility_agent.agent.v1_agent.run", spy)

        run(case_id="CLM-4003", dossier_patient_id="PAT-001", contract=_LEGACY_CONTRACT)
        spy.assert_called_once()


class TestStatusAggregation:
    def test_overall_status_is_worst_of_identity_and_coverage(self):
        mismatched_contract = {**_LEGACY_CONTRACT, "patient_id": "PAT-999-OTHER"}
        result = run(
            case_id="CLM-4010",
            dossier_patient_id="PAT-001",
            contract=mismatched_contract,
            extracted_fields=_EXTRACTED_FIELDS,
        )
        assert result.status is not VerificationStatus.PASS
        worse_rank = {"PASS": 0, "NEEDS_REVIEW": 1, "FAIL": 2}
        assert worse_rank[result.status.value] >= max(
            worse_rank[result.identity.status.value], worse_rank[result.coverage.status.value]
        )

    def test_no_contract_no_policy_never_crashes(self):
        result = run(case_id="CLM-4011", extracted_fields=_EXTRACTED_FIELDS)
        assert result.status in (
            VerificationStatus.PASS,
            VerificationStatus.NEEDS_REVIEW,
            VerificationStatus.FAIL,
        )
        assert result.reasons


class TestMissingCoverageDataDowngrade:
    """Correctif post-mesure V2-10 (AZIZ) : l'absence de donnée de police/
    assureur ne doit jamais produire un `coverage.status = FAIL` automatique
    — seulement `NEEDS_REVIEW`, avec `coverage_data_available=False` pour que
    `autonomous_decision_agent` distingue explicitement « donnée manquante »
    de « couverture réellement invalide »."""

    def test_missing_payer_name_downgrades_to_needs_review_not_fail(self):
        # Aucun contrat, aucun policy_number, aucun payer_name dans les champs
        # extraits — exactement la situation réelle du graphe V2 (bug diagnostiqué
        # post-mesure V2-10 : eligibility_agent.node() ne transmet jamais
        # contract=/policy_number=, et payer_name n'est souvent pas extrait).
        result = run(
            case_id="CLM-4012",
            dossier_patient_id="PAT-001",
            extracted_fields={"patient_id": "PAT-001", "amount_requested": "100.00"},
        )
        assert result.coverage.status is VerificationStatus.NEEDS_REVIEW
        assert result.coverage.status is not VerificationStatus.FAIL
        assert result.coverage_data_available is False

    def test_missing_amount_requested_downgrades_to_needs_review_not_fail(self):
        # Généralisation post-mesure V2-10 (CLM-0002, échantillon à 5 dossiers) :
        # payer_name présent mais aucun montant demandé nulle part — second
        # motif à item unique de verify_coverage, non couvert par le premier
        # correctif (payer_name seul).
        result = run(
            case_id="CLM-4016",
            dossier_patient_id="PAT-001",
            extracted_fields={"patient_id": "PAT-001", "payer_name": "Cigna Health"},
        )
        assert result.coverage.status is VerificationStatus.NEEDS_REVIEW
        assert result.coverage.status is not VerificationStatus.FAIL

    def test_coverage_data_available_true_when_contract_provided(self):
        result = run(
            case_id="CLM-4013",
            dossier_patient_id="PAT-001",
            contract=_LEGACY_CONTRACT,
            extracted_fields=_EXTRACTED_FIELDS,
        )
        assert result.coverage_data_available is True

    def test_coverage_data_available_true_when_payer_name_present_without_contract(self):
        result = run(
            case_id="CLM-4014",
            dossier_patient_id="PAT-001",
            extracted_fields={"patient_id": "PAT-001", "payer_name": "Cigna Health"},
        )
        assert result.coverage_data_available is True

    def test_real_fail_reason_is_never_downgraded(self):
        # Contrat réellement fourni, réellement inactif — un FAIL confirmé
        # par une vraie donnée de contrat ne doit jamais être adouci.
        inactive_contract = {**_LEGACY_CONTRACT, "policy_active": False}
        result = run(
            case_id="CLM-4015",
            dossier_patient_id="PAT-001",
            contract=inactive_contract,
            extracted_fields=_EXTRACTED_FIELDS,
        )
        assert result.coverage.status is VerificationStatus.FAIL
        assert result.coverage_data_available is True


class TestErrorsMapping:
    def test_low_extraction_confidence_populates_errors(self):
        result = run(
            case_id="CLM-4020",
            dossier_patient_id="PAT-001",
            contract=_LEGACY_CONTRACT,
            extracted_fields=_EXTRACTED_FIELDS,
            extraction_confidence=0.3,
        )
        assert any(e.code == "LOW_EXTRACTION_CONFIDENCE" for e in result.errors)


class TestNodeIntegration:
    def test_node_reads_extraction_and_updates_state(self):
        from schemas.domain import DocumentType, ExtractionStatus
        from schemas.results import DocumentClassification, DocumentExtraction, ExtractedField

        extraction = DocumentExtraction(
            claim_id="CLM-4030",
            document_id="CLM-4030-doc-0",
            classification=DocumentClassification(
                document_type=DocumentType.INVOICE,
                confidence=0.9,
                classification_source="filename",
            ),
            extraction_status=ExtractionStatus.SUCCESS,
            confidence_score=0.9,
            is_readable=True,
            fields={
                "patient_id": ExtractedField(field_name="patient_id", value="PAT-001", confidence=0.9),
            },
        )

        class _FakeDocResult:
            def __init__(self, extraction):
                self.extraction = extraction

        state = {
            "case_id": "CLM-4030",
            "schema_version": "2.0.0",
            "current_step": "document_understanding",
            "completed_steps": ["intake_safety", "document_understanding"],
            "document_understanding_result": _FakeDocResult(extraction),
        }
        updates = node(state)  # type: ignore[arg-type]
        assert updates["current_step"] == "eligibility"
        assert updates["completed_steps"] == ["eligibility"]
        assert updates["eligibility_result"].identity is not None

    def test_node_handles_missing_document_understanding_result(self):
        state = {
            "case_id": "CLM-4031",
            "schema_version": "2.0.0",
            "current_step": "document_understanding",
            "completed_steps": [],
        }
        updates = node(state)  # type: ignore[arg-type]
        assert updates["eligibility_result"] is not None


class TestServiceDateNormalization:
    """Régression V2-10 : un dossier réel (graphe V2 strictement séquentiel,
    jamais court-circuité avant `eligibility`, contrairement à V1) transmet
    `service_date` au format brut OCR (ex. `JJ/MM/AAAA`), jamais normalisé
    en amont pour ce champ — `identity_coverage_agent.agent.run()` (V1, non
    modifié) le refusait alors avec une `pydantic.ValidationError` non
    gérée, jamais observé en V1 car ce chemin n'y est jamais exercé
    bout en bout avec de vrais documents."""

    def test_day_first_slash_format_never_crashes(self):
        result = run(
            case_id="CLM-4040",
            dossier_patient_id="PAT-001",
            contract=_LEGACY_CONTRACT,
            extracted_fields={**_EXTRACTED_FIELDS, "service_date": "19/05/1976"},
        )
        assert result.status in (
            VerificationStatus.PASS,
            VerificationStatus.NEEDS_REVIEW,
            VerificationStatus.FAIL,
        )

    def test_extracted_field_object_with_slash_date_never_crashes(self):
        result = run(
            case_id="CLM-4041",
            dossier_patient_id="PAT-001",
            contract=_LEGACY_CONTRACT,
            extracted_fields={
                **_EXTRACTED_FIELDS,
                "service_date": ExtractedField(
                    field_name="service_date", value="19/05/1976", confidence=0.9
                ),
            },
        )
        assert result.status in (
            VerificationStatus.PASS,
            VerificationStatus.NEEDS_REVIEW,
            VerificationStatus.FAIL,
        )

    def test_explicit_service_date_kwarg_with_slash_format_never_crashes(self):
        result = run(
            case_id="CLM-4042",
            dossier_patient_id="PAT-001",
            contract=_LEGACY_CONTRACT,
            service_date="19/05/1976",
        )
        assert result.status in (
            VerificationStatus.PASS,
            VerificationStatus.NEEDS_REVIEW,
            VerificationStatus.FAIL,
        )

    def test_ambiguous_date_dropped_not_invented(self):
        normalized = _normalize_extracted_fields({"service_date": "05/06/2026"})
        assert "service_date" not in normalized

    def test_iso_date_passes_through_unaffected(self):
        normalized = _normalize_extracted_fields({"service_date": "2026-06-15"})
        assert normalized["service_date"] == "2026-06-15"

    def test_unambiguous_slash_date_normalized_to_iso(self):
        normalized = _normalize_extracted_fields({"service_date": "19/05/1976"})
        assert normalized["service_date"] == "1976-05-19"

    def test_none_and_missing_extracted_fields_untouched(self):
        assert _normalize_extracted_fields(None) is None
        assert _normalize_extracted_fields({}) == {}
        assert _normalize_extracted_fields({"patient_id": "PAT-001"}) == {"patient_id": "PAT-001"}
