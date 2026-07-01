"""Tests unitaires de la validation schéma FHIR — tools/fhir_validation.py.

Couvrent uniquement la couche validate_resource_schema et _is_rule_enabled.
Les tests s'appuient sur la bibliothèque fhir.resources installée localement.
Aucun appel réseau. Le bundle source n'est jamais modifié.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from tools.fhir_validation import (
    _FHIR_LIB_AVAILABLE,
    _is_rule_enabled,
    validate_fhir_bundle,
    validate_resource_schema,
)

pytestmark = pytest.mark.skipif(
    not _FHIR_LIB_AVAILABLE,
    reason="fhir.resources non installée — tests de validation schéma ignorés",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_PATIENT_VALID = {
    "resourceType": "Patient",
    "id": "p1",
}

_PATIENT_MALFORMED = {
    # name doit être une liste d'objets HumanName, pas une chaîne
    "resourceType": "Patient",
    "id": "p2",
    "name": "Jean Dupont",
}

_COVERAGE_VALID = {
    "resourceType": "Coverage",
    "id": "cov1",
    "status": "active",
    "beneficiary": {"reference": "Patient/p1"},
}

_ENCOUNTER_MALFORMED = {
    # class doit être une liste en R4B, pas un objet
    "resourceType": "Encounter",
    "id": "enc1",
    "status": "finished",
    "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "AMB"},
}


def _make_bundle(*resources: dict) -> dict:
    """Construit un bundle minimal autour de la liste de ressources fournie."""
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": r} for r in resources],
    }


# ── Tests _is_rule_enabled ────────────────────────────────────────────────────


class TestIsRuleEnabled:
    def test_rule_present_returns_true(self) -> None:
        rules = {"rules": [{"id": "FHIR_RESOURCE_SCHEMA_VALID", "enabled": True}]}
        assert _is_rule_enabled(rules, "FHIR_RESOURCE_SCHEMA_VALID") is True

    def test_rule_absent_returns_false(self) -> None:
        rules = {"rules": [{"id": "FHIR_BUNDLE_STRUCTURE_VALID", "enabled": True}]}
        assert _is_rule_enabled(rules, "FHIR_RESOURCE_SCHEMA_VALID") is False

    def test_empty_rules_list(self) -> None:
        assert _is_rule_enabled({"rules": []}, "FHIR_RESOURCE_SCHEMA_VALID") is False

    def test_missing_rules_key(self) -> None:
        assert _is_rule_enabled({}, "FHIR_RESOURCE_SCHEMA_VALID") is False

    def test_non_dict_entry_skipped(self) -> None:
        rules = {"rules": ["FHIR_RESOURCE_SCHEMA_VALID", None]}
        assert _is_rule_enabled(rules, "FHIR_RESOURCE_SCHEMA_VALID") is False


# ── Tests validate_resource_schema ────────────────────────────────────────────


class TestValidateResourceSchemaValid:
    def test_valid_patient_no_warnings(self) -> None:
        errors, warnings = validate_resource_schema(_make_bundle(_PATIENT_VALID))
        assert errors == []
        assert not any("Patient/p1" in w and "SCHEMA_VALID" in w for w in warnings)

    def test_valid_coverage_no_warnings(self) -> None:
        errors, warnings = validate_resource_schema(_make_bundle(_COVERAGE_VALID))
        assert errors == []

    def test_empty_bundle_no_issues(self) -> None:
        bundle = {"resourceType": "Bundle", "type": "collection", "entry": []}
        errors, warnings = validate_resource_schema(bundle)
        assert errors == []
        assert warnings == []

    def test_errors_always_empty(self) -> None:
        """validate_resource_schema ne produit jamais d'erreurs bloquantes."""
        errors, _ = validate_resource_schema(_make_bundle(_PATIENT_MALFORMED))
        assert errors == []


class TestValidateResourceSchemaUnknownType:
    def test_unknown_type_produces_warning(self) -> None:
        resource = {"resourceType": "ConcoursSpecial", "id": "x1"}
        _, warnings = validate_resource_schema(_make_bundle(resource))
        assert len(warnings) == 1
        w = warnings[0]
        assert "FHIR_RESOURCE_TYPE_SUPPORTED" in w
        assert "ConcoursSpecial" in w
        assert "entry[0]" in w

    def test_unknown_type_message_has_no_field_values(self) -> None:
        resource = {"resourceType": "Ghost", "id": "g1", "secret": "CONFIDENTIEL"}
        _, warnings = validate_resource_schema(_make_bundle(resource))
        assert all("CONFIDENTIEL" not in w for w in warnings)

    def test_unknown_type_no_errors(self) -> None:
        resource = {"resourceType": "Unicorn", "id": "u1"}
        errors, _ = validate_resource_schema(_make_bundle(resource))
        assert errors == []


class TestValidateResourceSchemaMalformed:
    def test_malformed_patient_name_string(self) -> None:
        """Patient.name doit être une liste — une chaîne produit un avertissement."""
        _, warnings = validate_resource_schema(_make_bundle(_PATIENT_MALFORMED))
        schema_warns = [w for w in warnings if "FHIR_RESOURCE_SCHEMA_VALID" in w]
        assert len(schema_warns) == 1
        w = schema_warns[0]
        assert "entry[0]" in w
        assert "Patient" in w
        assert "name" in w  # chemin de champ inclus

    def test_malformed_encounter_class_not_list(self) -> None:
        """Encounter.class doit être une liste en R4B."""
        _, warnings = validate_resource_schema(_make_bundle(_ENCOUNTER_MALFORMED))
        schema_warns = [w for w in warnings if "FHIR_RESOURCE_SCHEMA_VALID" in w]
        assert len(schema_warns) == 1
        assert "class" in schema_warns[0]

    def test_issue_count_in_message(self) -> None:
        """Le nombre de problèmes détectés apparaît dans le message."""
        _, warnings = validate_resource_schema(_make_bundle(_PATIENT_MALFORMED))
        schema_warns = [w for w in warnings if "FHIR_RESOURCE_SCHEMA_VALID" in w]
        assert any("problème(s)" in w for w in schema_warns)

    def test_field_values_never_in_message(self) -> None:
        """Les valeurs des champs (potentiellement PII) ne figurent pas dans les messages."""
        resource = {
            "resourceType": "Patient",
            "id": "pii1",
            "name": "MARIE CURIE PII_DATA",  # valeur PII fictive
        }
        _, warnings = validate_resource_schema(_make_bundle(resource))
        assert all("MARIE CURIE" not in w for w in warnings)
        assert all("PII_DATA" not in w for w in warnings)

    def test_location_format_entry_prefix(self) -> None:
        """Le préfixe entry[N] (ResourceType/id) est présent."""
        _, warnings = validate_resource_schema(_make_bundle(_PATIENT_MALFORMED))
        schema_warns = [w for w in warnings if "FHIR_RESOURCE_SCHEMA_VALID" in w]
        assert any("entry[0]" in w for w in schema_warns)
        assert any("Patient/p2" in w for w in schema_warns)

    def test_multiple_resources_independent_warnings(self) -> None:
        """Chaque ressource malformée produit son propre avertissement."""
        r1 = {"resourceType": "Patient", "id": "p1", "name": "mauvais"}
        r2 = {"resourceType": "Patient", "id": "p2", "name": "aussi_mauvais"}
        _, warnings = validate_resource_schema(_make_bundle(r1, r2))
        schema_warns = [w for w in warnings if "FHIR_RESOURCE_SCHEMA_VALID" in w]
        assert len(schema_warns) == 2
        assert any("entry[0]" in w for w in schema_warns)
        assert any("entry[1]" in w for w in schema_warns)


class TestValidateResourceSchemaUnsupportedVersion:
    def test_stu3_version_returns_warning(self) -> None:
        _, warnings = validate_resource_schema(_make_bundle(_PATIENT_VALID), fhir_version="STU3")
        assert len(warnings) == 1
        w = warnings[0]
        assert "FHIR_RESOURCE_SCHEMA_VALID" in w
        assert "STU3" in w
        assert "non prise en charge" in w

    def test_unsupported_version_no_resource_check(self) -> None:
        """Avec version non supportée, aucune ressource n'est validée."""
        r = {"resourceType": "Patient", "id": "p1", "name": "mauvais"}
        _, warnings = validate_resource_schema(_make_bundle(r), fhir_version="DSTU2")
        assert len(warnings) == 1  # uniquement l'avertissement de version

    def test_unsupported_version_no_errors(self) -> None:
        errors, _ = validate_resource_schema(_make_bundle(_PATIENT_VALID), fhir_version="STU3")
        assert errors == []

    @pytest.mark.parametrize("version", ["R4", "R4B", "r4", "r4b"])
    def test_supported_versions_proceed(self, version: str) -> None:
        """R4 et R4B (casse indifférente) ne déclenchent pas l'avertissement de version."""
        _, warnings = validate_resource_schema(_make_bundle(_PATIENT_VALID), fhir_version=version)
        version_warns = [w for w in warnings if "non prise en charge" in w]
        assert version_warns == []


class TestValidateResourceSchemaInternalError:
    def test_internal_error_produces_warning(self) -> None:
        """Une erreur interne inattendue du validateur produit un warning non bloquant."""
        from fhir.resources import get_fhir_model_class

        def raise_runtime(*args, **kwargs):  # noqa: ANN001, ANN202
            raise RuntimeError("erreur simulée interne")

        with patch.object(
            get_fhir_model_class("Patient"),
            "model_validate",
            side_effect=raise_runtime,
        ):
            _, warnings = validate_resource_schema(_make_bundle(_PATIENT_VALID))

        internal_warns = [w for w in warnings if "erreur interne" in w]
        assert len(internal_warns) == 1
        assert "RuntimeError" in internal_warns[0]

    def test_internal_error_no_exception_propagated(self) -> None:
        """Une erreur inattendue de _fhir_get_model ne remonte pas en exception."""
        with patch("tools.fhir_validation._fhir_get_model", side_effect=RuntimeError("boom")):
            errors, warnings = validate_resource_schema(_make_bundle(_PATIENT_VALID))
        assert errors == []
        assert len(warnings) == 1
        assert "erreur interne" in warnings[0]
        assert "RuntimeError" in warnings[0]


class TestValidateResourceSchemaLibUnavailable:
    def test_lib_unavailable_single_warning(self) -> None:
        """Si fhir.resources est absente, un seul avertissement est produit."""
        with patch("tools.fhir_validation._FHIR_LIB_AVAILABLE", False):
            errors, warnings = validate_resource_schema(_make_bundle(_PATIENT_VALID))
        assert errors == []
        assert len(warnings) == 1
        assert "absente" in warnings[0]

    def test_lib_unavailable_no_resource_validated(self) -> None:
        """Aucune ressource n'est validée si la bibliothèque est absente."""
        with patch("tools.fhir_validation._FHIR_LIB_AVAILABLE", False):
            _, warnings = validate_resource_schema(
                _make_bundle(_PATIENT_MALFORMED, _ENCOUNTER_MALFORMED)
            )
        assert len(warnings) == 1  # uniquement l'avertissement d'absence de lib


# ── Tests d'intégration avec validate_fhir_bundle ────────────────────────────


class TestValidateFhirBundleWithSchema:
    """Intégration : validate_resource_schema est appelée si la règle est présente."""

    _RULES_WITH_SCHEMA: dict = {
        "rules": [{"id": "FHIR_RESOURCE_SCHEMA_VALID", "enabled": True}],
        "profile": "R4",
        "required_resource_types": ["Patient"],
        "optional_resource_types": [],
        "bundle_types_accepted": ["collection"],
        "supported_resource_types": [],
        "min_cardinalities": {},
        "resource_required_fields": {},
        "coverage_status_values": ["active"],
        "reference_fields": {},
        "supported_profiles": ["R4"],
    }

    _RULES_WITHOUT_SCHEMA: dict = {
        "rules": [],
        "profile": "R4",
        "required_resource_types": ["Patient"],
        "optional_resource_types": [],
        "bundle_types_accepted": ["collection"],
        "supported_resource_types": [],
        "min_cardinalities": {},
        "resource_required_fields": {},
        "coverage_status_values": ["active"],
        "reference_fields": {},
        "supported_profiles": ["R4"],
    }

    _FIXTURE = "datasets/fixtures/valid/CLM-0001/input/patient_fhir_bundle.json"

    def test_schema_rule_present_triggers_validation(self) -> None:
        """Avec la règle FHIR_RESOURCE_SCHEMA_VALID, la validation schéma est déclenchée."""
        status, errors, warnings, _ = validate_fhir_bundle(
            self._FIXTURE,
            rules=self._RULES_WITH_SCHEMA,
        )
        # La validation structurelle passe ; les warnings R4/R4B peuvent apparaître
        assert errors == []
        # Le statut peut être PASS ou NEEDS_REVIEW selon les warnings R4B
        assert status.value in {"PASS", "NEEDS_REVIEW"}

    def test_schema_rule_absent_skips_validation(self) -> None:
        """Sans la règle FHIR_RESOURCE_SCHEMA_VALID, la validation schéma est ignorée."""
        status, errors, warnings, _ = validate_fhir_bundle(
            self._FIXTURE,
            rules=self._RULES_WITHOUT_SCHEMA,
        )
        schema_warns = [w for w in warnings if "FHIR_RESOURCE_SCHEMA_VALID" in w]
        assert schema_warns == []

    def test_bundle_not_copied_in_result(self) -> None:
        """Le bundle complet n'apparaît pas dans les résultats retournés."""
        import json as _json

        bundle_path = self._FIXTURE
        bundle_data = _json.load(open(bundle_path))
        first_entry_str = str(bundle_data["entry"][0])[:30]

        _, errors, warnings, _ = validate_fhir_bundle(
            bundle_path,
            rules=self._RULES_WITH_SCHEMA,
        )
        combined = " ".join(errors + warnings)
        assert first_entry_str not in combined

    def test_path_none_bundle_expected_returns_fail(self) -> None:
        status, errors, _, _ = validate_fhir_bundle(
            None,
            bundle_expected=True,
            rules=self._RULES_WITH_SCHEMA,
        )
        assert status.value == "FAIL"
        assert errors

    def test_malformed_resource_in_bundle_produces_warning(self) -> None:
        """Un bundle avec ressource malformée produit un warning, pas une erreur."""
        import json as _json
        import tempfile
        from pathlib import Path

        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "p1", "name": "chaîne"}},
            ],
        }
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False, encoding="utf-8"
        ) as f:
            _json.dump(bundle, f)
            tmp_path = f.name

        try:
            status, errors, warnings, _ = validate_fhir_bundle(
                tmp_path, rules=self._RULES_WITH_SCHEMA
            )
            assert errors == []
            schema_warns = [w for w in warnings if "FHIR_RESOURCE_SCHEMA_VALID" in w]
            assert len(schema_warns) >= 1
        finally:
            Path(tmp_path).unlink(missing_ok=True)
