"""Tests de tools/fhir_medical_items.py — plan de remédiation « autonomie
décisionnelle V2 », phase 3.

Fonctions pures, bundles FHIR synthétiques minimaux — jamais de fichier réel
sur disque ni de validation structurelle complète (déjà couverte par
tests/tools/test_fhir_validation.py).
"""
from __future__ import annotations

from schemas.v2_results import MedicalItemType
from tools.fhir_medical_items import (
    extract_medical_items_from_bundle,
    extract_payer_hint_from_coverage,
)


def _bundle(*resources: dict) -> dict:
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": r} for r in resources],
    }


class TestExtractMedicalItemsFromBundle:
    def test_procedure_resource_extracted_as_procedure(self):
        bundle = _bundle(
            {
                "resourceType": "Procedure",
                "id": "proc-1",
                "code": {"text": "Consultation ophtalmologique"},
            }
        )
        items = extract_medical_items_from_bundle(bundle)
        assert len(items) == 1
        assert items[0].item_type is MedicalItemType.PROCEDURE
        assert items[0].description == "Consultation ophtalmologique"
        assert items[0].source_document_id == "proc-1"
        assert items[0].classification_method == "fhir_resource_type"
        assert items[0].confidence == 1.0

    def test_medication_request_extracted_as_medication(self):
        bundle = _bundle(
            {
                "resourceType": "MedicationRequest",
                "id": "medreq-1",
                "medicationCodeableConcept": {
                    "coding": [{"display": "Metformine 500 mg"}],
                },
            }
        )
        items = extract_medical_items_from_bundle(bundle)
        assert len(items) == 1
        assert items[0].item_type is MedicalItemType.MEDICATION
        assert items[0].description == "Metformine 500 mg"

    def test_medication_statement_also_extracted_as_medication(self):
        bundle = _bundle(
            {
                "resourceType": "MedicationStatement",
                "medicationCodeableConcept": {"text": "Salbutamol inhalateur"},
            }
        )
        items = extract_medical_items_from_bundle(bundle)
        assert len(items) == 1
        assert items[0].item_type is MedicalItemType.MEDICATION

    def test_text_field_preferred_over_coding_display(self):
        bundle = _bundle(
            {
                "resourceType": "Procedure",
                "code": {
                    "text": "Libellé préféré",
                    "coding": [{"display": "Libellé secondaire"}],
                },
            }
        )
        items = extract_medical_items_from_bundle(bundle)
        assert items[0].description == "Libellé préféré"

    def test_other_resource_types_are_ignored(self):
        bundle = _bundle(
            {"resourceType": "Patient", "id": "p1"},
            {"resourceType": "Coverage", "id": "cov-1"},
            {"resourceType": "Claim", "id": "claim-1"},
        )
        assert extract_medical_items_from_bundle(bundle) == []

    def test_procedure_without_exploitable_code_is_skipped(self):
        bundle = _bundle({"resourceType": "Procedure", "id": "proc-empty", "code": {}})
        assert extract_medical_items_from_bundle(bundle) == []

    def test_empty_bundle_returns_empty_list(self):
        assert extract_medical_items_from_bundle(_bundle()) == []

    def test_multiple_resources_all_extracted(self):
        bundle = _bundle(
            {"resourceType": "Procedure", "id": "p1", "code": {"text": "Acte A"}},
            {
                "resourceType": "MedicationRequest",
                "id": "m1",
                "medicationCodeableConcept": {"text": "Médicament B"},
            },
        )
        items = extract_medical_items_from_bundle(bundle)
        assert len(items) == 2
        assert {i.item_type for i in items} == {MedicalItemType.PROCEDURE, MedicalItemType.MEDICATION}

    def test_never_invents_a_description(self):
        """Aucun champ FHIR standard exploitable -> aucun élément produit,
        jamais une description inventée."""
        bundle = _bundle({"resourceType": "MedicationRequest", "id": "m1"})
        assert extract_medical_items_from_bundle(bundle) == []


class TestExtractPayerHintFromCoverage:
    def test_payer_display_and_policy_number_extracted(self):
        bundle = _bundle(
            {
                "resourceType": "Coverage",
                "payer": [{"display": "Assureur Santé Plus"}],
                "identifier": [{"value": "POLICY-12345"}],
            }
        )
        hint = extract_payer_hint_from_coverage(bundle)
        assert hint == {"payer_name": "Assureur Santé Plus", "policy_number": "POLICY-12345"}

    def test_no_coverage_resource_returns_none(self):
        bundle = _bundle({"resourceType": "Patient", "id": "p1"})
        assert extract_payer_hint_from_coverage(bundle) is None

    def test_coverage_without_exploitable_fields_returns_none(self):
        bundle = _bundle({"resourceType": "Coverage", "id": "cov-empty"})
        assert extract_payer_hint_from_coverage(bundle) is None

    def test_payer_only_without_identifier_still_returns_hint(self):
        bundle = _bundle({"resourceType": "Coverage", "payer": [{"display": "Assureur X"}]})
        hint = extract_payer_hint_from_coverage(bundle)
        assert hint == {"payer_name": "Assureur X", "policy_number": None}

    def test_first_coverage_resource_wins(self):
        bundle = _bundle(
            {"resourceType": "Coverage", "payer": [{"display": "Premier assureur"}]},
            {"resourceType": "Coverage", "payer": [{"display": "Second assureur"}]},
        )
        hint = extract_payer_hint_from_coverage(bundle)
        assert hint["payer_name"] == "Premier assureur"
