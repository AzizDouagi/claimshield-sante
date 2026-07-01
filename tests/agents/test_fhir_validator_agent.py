"""Tests du FHIR Validator Agent — agents/fhir_validator_agent/agent.py.

Couvre :
  - bundle valide (CLM-0001 fixture et bundle tmp minimal)
  - bundle absent / NOT_PROVIDED
  - JSON invalide (tronqué, vide, tableau, chaîne brute)
  - racine invalide (non-Bundle, clé resourceType manquante)
  - resourceType absent ou incorrect dans une entrée
  - ressources Patient / Coverage absentes (obligatoire vs optionnel)
  - ressource mal formée (warning non bloquant, PII absente des messages)
  - référence interne valide résolue
  - référence interne non résolue (FAIL)
  - cardinalité minimale invalide (0 Patient)
  - profil inconnu (warning, non bloquant)
  - version FHIR non prise en charge dans validate_resource_schema (warning)
  - hash SHA-256 correct → confirmé ; incorrect → FAIL avant toute validation
  - bundle original inchangé sur disque après validation (immuabilité)
  - bundle non retourné dans FhirValidatorResult (secret non exposé)
  - déterminisme (même entrée → mêmes sorties)
  - Security Gate ALLOW / BLOCK / absent → exécution / NOT_EVALUATED
  - nœud LangGraph : fhir_input absent ou invalide, consommation à None
  - nœud LangGraph : audit_trail produit sans contenu FHIR brut
  - nœud LangGraph : SHA-256 extrait depuis le manifest d'ingestion
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agents.fhir_validator_agent.agent import node, run
from schemas.domain import SecurityDecision, VerificationStatus
from schemas.results import AuditEvent, FhirValidatorResult
from tools.file_inspection import compute_sha256


# ── Fixtures et helpers ───────────────────────────────────────────────────────

CLM_FIXTURE = "datasets/fixtures/valid/CLM-0001/input/patient_fhir_bundle.json"


def _write(tmp_path: Path, bundle: dict, name: str = "bundle.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(bundle), encoding="utf-8")
    return p


def _bundle(*extra: dict) -> dict:
    """Bundle minimal contenant un Patient et des ressources optionnelles."""
    entries = [{"resource": {"resourceType": "Patient", "id": "PAT-001"}}]
    entries += [{"resource": r} for r in extra]
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


def _coverage(ref: str = "Patient/PAT-001") -> dict:
    return {
        "resourceType": "Coverage",
        "id": "COV-001",
        "status": "active",
        "beneficiary": {"reference": ref},
    }


def _mock_security(decision: SecurityDecision) -> MagicMock:
    m = MagicMock()
    m.decision = decision
    return m


def _mock_intake(filename: str, sha256: str) -> MagicMock:
    f = MagicMock()
    f.original_name = filename
    f.sha256 = sha256
    manifest = MagicMock()
    manifest.files = [f]
    intake = MagicMock()
    intake.manifest = manifest
    return intake


# ── Bundle valide ─────────────────────────────────────────────────────────────


class TestBundleValide:
    def test_fixture_clm0001_passe(self) -> None:
        """La fixture CLM-0001 passe la validation structurelle."""
        result = run("CLM-0001", CLM_FIXTURE)
        assert isinstance(result, FhirValidatorResult)
        assert result.status in (VerificationStatus.PASS, VerificationStatus.NEEDS_REVIEW)
        assert result.errors == []
        assert "Patient" in result.resource_types
        assert result.resource_count >= 1

    def test_bundle_minimal_valid(self, tmp_path: Path) -> None:
        p = _write(tmp_path, _bundle())
        result = run("CLM-TEST", str(p))
        assert result.status in (VerificationStatus.PASS, VerificationStatus.NEEDS_REVIEW)
        assert result.errors == []
        assert result.bundle_expected is True
        assert result.references_checked is True

    def test_bundle_valide_avec_coverage_reference_resolue(self, tmp_path: Path) -> None:
        """Coverage.beneficiary → Patient présent → pas d'erreur de référence."""
        p = _write(tmp_path, _bundle(_coverage("Patient/PAT-001")))
        result = run("CLM-TEST", str(p))
        assert result.errors == []

    def test_rule_version_et_scope(self, tmp_path: Path) -> None:
        p = _write(tmp_path, _bundle())
        result = run("CLM-TEST", str(p))
        assert result.rule_version == "1.0.0"
        assert result.validation_scope == "STRUCTURAL_ONLY"

    def test_raison_disclaimer_structurel(self, tmp_path: Path) -> None:
        p = _write(tmp_path, _bundle())
        result = run("CLM-TEST", str(p))
        assert any("structurelle uniquement" in r for r in result.reasons)

    def test_profile_checked_r4(self, tmp_path: Path) -> None:
        p = _write(tmp_path, _bundle())
        result = run("CLM-TEST", str(p))
        assert result.profile_checked == "R4"

    def test_resource_types_extraits(self, tmp_path: Path) -> None:
        p = _write(tmp_path, _bundle(_coverage()))
        result = run("CLM-TEST", str(p))
        assert "Patient" in result.resource_types
        assert "Coverage" in result.resource_types
        assert result.resource_count == 2


# ── Bundle absent / NOT_PROVIDED ─────────────────────────────────────────────


class TestBundleAbsent:
    def test_not_provided_pass(self) -> None:
        """bundle_expected=False + path=None → PASS (NOT_PROVIDED)."""
        result = run("CLM-TEST", None, bundle_expected=False)
        assert result.status == VerificationStatus.PASS
        assert result.bundle_expected is False
        assert any("NOT_PROVIDED" in r for r in result.reasons)
        assert result.errors == []

    def test_not_provided_validation_ignoree(self) -> None:
        result = run("CLM-TEST", None, bundle_expected=False)
        assert any("Validation structurelle ignorée" in r for r in result.reasons)

    def test_bundle_attendu_sans_chemin_fail(self) -> None:
        """bundle_expected=True + path=None → FAIL."""
        result = run("CLM-TEST", None, bundle_expected=True)
        assert result.status == VerificationStatus.FAIL
        assert result.errors

    def test_chemin_inexistant_fail(self, tmp_path: Path) -> None:
        result = run("CLM-TEST", str(tmp_path / "inexistant.json"))
        assert result.status == VerificationStatus.FAIL

    def test_not_provided_references_checked_false(self) -> None:
        result = run("CLM-TEST", None, bundle_expected=False)
        assert result.references_checked is False


# ── JSON invalide ─────────────────────────────────────────────────────────────


class TestJsonInvalide:
    def test_json_tronque_fail(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text('{"resourceType": "Bundle"', encoding="utf-8")
        result = run("CLM-TEST", str(p))
        assert result.status == VerificationStatus.FAIL
        assert any("JSON malformé" in e for e in result.errors)

    def test_fichier_vide_fail(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.json"
        p.write_text("", encoding="utf-8")
        result = run("CLM-TEST", str(p))
        assert result.status == VerificationStatus.FAIL

    def test_json_tableau_fail(self, tmp_path: Path) -> None:
        """Un tableau JSON n'est pas un dict → FAIL (racine invalide)."""
        p = tmp_path / "array.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        result = run("CLM-TEST", str(p))
        assert result.status == VerificationStatus.FAIL

    def test_json_chaine_brute_fail(self, tmp_path: Path) -> None:
        """Une chaîne JSON n'est pas un dict → FAIL."""
        p = tmp_path / "string.json"
        p.write_text('"ceci est une chaîne"', encoding="utf-8")
        result = run("CLM-TEST", str(p))
        assert result.status == VerificationStatus.FAIL


# ── Racine invalide ───────────────────────────────────────────────────────────


class TestRacineInvalide:
    def test_resource_type_incorrect_fail(self, tmp_path: Path) -> None:
        """Un objet JSON sans resourceType='Bundle' → FAIL."""
        p = _write(tmp_path, {"resourceType": "Patient", "id": "p1"})
        result = run("CLM-TEST", str(p))
        assert result.status == VerificationStatus.FAIL

    def test_resource_type_manquant_fail(self, tmp_path: Path) -> None:
        p = _write(tmp_path, {"type": "collection", "entry": []})
        result = run("CLM-TEST", str(p))
        assert result.status == VerificationStatus.FAIL

    def test_objet_vide_fail(self, tmp_path: Path) -> None:
        p = _write(tmp_path, {})
        result = run("CLM-TEST", str(p))
        assert result.status == VerificationStatus.FAIL

    def test_entry_non_liste_fail(self, tmp_path: Path) -> None:
        """entry doit être une liste, pas un dict."""
        p = _write(
            tmp_path,
            {"resourceType": "Bundle", "type": "collection", "entry": {"resource": {}}},
        )
        result = run("CLM-TEST", str(p))
        assert result.status == VerificationStatus.FAIL

    def test_erreur_contient_bundle_resource_type(self, tmp_path: Path) -> None:
        p = _write(tmp_path, {"resourceType": "Collection", "type": "collection", "entry": []})
        result = run("CLM-TEST", str(p))
        assert result.status == VerificationStatus.FAIL
        assert any("Bundle" in e for e in result.errors)


# ── resourceType absent ou incorrect dans une entrée ─────────────────────────


class TestResourceTypeEntree:
    def test_entree_sans_resource_type_fail(self, tmp_path: Path) -> None:
        """Une entrée sans resourceType produit une erreur de champ obligatoire."""
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [{"resource": {"id": "p1"}}],
        }
        p = _write(tmp_path, bundle)
        result = run("CLM-TEST", str(p))
        assert result.status == VerificationStatus.FAIL
        assert any("resourceType" in e for e in result.errors)

    def test_entree_resource_type_null_fail(self, tmp_path: Path) -> None:
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [{"resource": {"resourceType": None, "id": "p1"}}],
        }
        p = _write(tmp_path, bundle)
        result = run("CLM-TEST", str(p))
        assert result.status == VerificationStatus.FAIL

    def test_resource_type_non_supporte_warning(self, tmp_path: Path) -> None:
        """Observation (hors supported_resource_types) → warning non bloquant."""
        p = _write(tmp_path, _bundle({"resourceType": "Observation", "id": "OBS-001"}))
        result = run("CLM-TEST", str(p))
        assert any("non supporté localement" in w for w in result.warnings)

    def test_resource_type_non_supporte_ne_bloque_pas(self, tmp_path: Path) -> None:
        p = _write(tmp_path, _bundle({"resourceType": "Observation", "id": "OBS-001"}))
        result = run("CLM-TEST", str(p))
        assert result.errors == []


# ── Ressources obligatoires absentes ─────────────────────────────────────────


class TestRessourcesObligatoires:
    def test_sans_patient_fail(self, tmp_path: Path) -> None:
        """Bundle sans Patient → FAIL (ressource obligatoire)."""
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [{"resource": _coverage()}],
        }
        p = _write(tmp_path, bundle)
        result = run("CLM-TEST", str(p))
        assert result.status == VerificationStatus.FAIL
        assert any("Patient" in e for e in result.errors)

    def test_entry_vide_fail(self, tmp_path: Path) -> None:
        p = _write(tmp_path, {"resourceType": "Bundle", "type": "collection", "entry": []})
        result = run("CLM-TEST", str(p))
        assert result.status == VerificationStatus.FAIL

    def test_coverage_optionnel_absence_ne_bloque_pas(self, tmp_path: Path) -> None:
        """Coverage est optionnel — son absence n'est pas bloquante."""
        p = _write(tmp_path, _bundle())
        result = run("CLM-TEST", str(p))
        assert result.errors == []

    def test_claim_optionnel_absence_ne_bloque_pas(self, tmp_path: Path) -> None:
        p = _write(tmp_path, _bundle())
        result = run("CLM-TEST", str(p))
        assert result.errors == []


# ── Ressource mal formée ──────────────────────────────────────────────────────


class TestRessourceMalFormee:
    def test_patient_name_chaine_warning(self, tmp_path: Path) -> None:
        """Patient.name doit être une liste HumanName — une chaîne → warning R4B."""
        b = _bundle()
        b["entry"][0]["resource"]["name"] = "Jean Dupont"
        p = _write(tmp_path, b)
        result = run("CLM-TEST", str(p))
        assert result.errors == []
        schema_warns = [w for w in result.warnings if "FHIR_RESOURCE_SCHEMA_VALID" in w]
        assert len(schema_warns) >= 1
        assert any("name" in w for w in schema_warns)

    def test_malformation_ne_fail_pas(self, tmp_path: Path) -> None:
        """Ressource mal formée → PASS ou NEEDS_REVIEW, jamais FAIL."""
        b = _bundle()
        b["entry"][0]["resource"]["name"] = "mauvais_type"
        p = _write(tmp_path, b)
        result = run("CLM-TEST", str(p))
        assert result.status in (VerificationStatus.PASS, VerificationStatus.NEEDS_REVIEW)

    def test_pii_absente_des_messages(self, tmp_path: Path) -> None:
        """Les valeurs PII ne doivent jamais apparaître dans erreurs ou warnings."""
        b = _bundle()
        b["entry"][0]["resource"]["name"] = "VALEUR_PII_CONFIDENTIELLE"
        p = _write(tmp_path, b)
        result = run("CLM-TEST", str(p))
        combined = " ".join(result.errors + result.warnings)
        assert "VALEUR_PII_CONFIDENTIELLE" not in combined

    def test_chemin_de_champ_inclus_dans_warning(self, tmp_path: Path) -> None:
        """Le chemin du champ mal formé (ex: 'name') apparaît dans le warning."""
        b = _bundle()
        b["entry"][0]["resource"]["name"] = "mauvais"
        p = _write(tmp_path, b)
        result = run("CLM-TEST", str(p))
        schema_warns = [w for w in result.warnings if "FHIR_RESOURCE_SCHEMA_VALID" in w]
        assert any("name" in w for w in schema_warns)

    def test_plusieurs_ressources_malformees_warnings_independants(
        self, tmp_path: Path
    ) -> None:
        """Chaque ressource malformée produit son propre warning."""
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "p1", "name": "mauvais"}},
                {"resource": {"resourceType": "Patient", "id": "p2", "name": "aussi_mauvais"}},
            ],
        }
        p = _write(tmp_path, bundle)
        result = run("CLM-TEST", str(p))
        schema_warns = [w for w in result.warnings if "FHIR_RESOURCE_SCHEMA_VALID" in w]
        assert len(schema_warns) == 2
        assert any("entry[0]" in w for w in schema_warns)
        assert any("entry[1]" in w for w in schema_warns)


# ── Références internes ───────────────────────────────────────────────────────


class TestReferencesInternes:
    def test_reference_resolue_pas_erreur(self, tmp_path: Path) -> None:
        """Coverage.beneficiary.reference = 'Patient/PAT-001' + Patient présent → OK."""
        p = _write(tmp_path, _bundle(_coverage("Patient/PAT-001")))
        result = run("CLM-TEST", str(p))
        assert result.errors == []

    def test_reference_non_resolue_fail(self, tmp_path: Path) -> None:
        """Coverage.beneficiary.reference → Patient absent → FAIL."""
        p = _write(tmp_path, _bundle(_coverage("Patient/PAT-MISSING")))
        result = run("CLM-TEST", str(p))
        assert result.status == VerificationStatus.FAIL
        assert any("référence interne non résolue" in e for e in result.errors)
        assert any("PAT-MISSING" in e for e in result.errors)

    def test_references_checked_true_si_valide(self, tmp_path: Path) -> None:
        p = _write(tmp_path, _bundle())
        result = run("CLM-TEST", str(p))
        assert result.references_checked is True

    def test_references_checked_false_si_bundle_non_charge(self) -> None:
        """Si le bundle ne peut pas être chargé, references_checked est False."""
        result = run("CLM-TEST", None, bundle_expected=True)
        assert result.references_checked is False

    def test_reference_interne_emplacement_precise(self, tmp_path: Path) -> None:
        """L'entrée et le chemin de la référence apparaissent dans l'erreur."""
        p = _write(tmp_path, _bundle(_coverage("Patient/ABSENT")))
        result = run("CLM-TEST", str(p))
        ref_errors = [e for e in result.errors if "référence interne non résolue" in e]
        assert ref_errors
        assert any("beneficiary" in e for e in ref_errors)


# ── Cardinalité minimale invalide ─────────────────────────────────────────────


class TestCardinaliteInvalide:
    def test_zero_patient_viole_cardinalite_min(self, tmp_path: Path) -> None:
        """min_cardinalities.Patient=1 — bundle sans Patient → FAIL."""
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Coverage",
                        "id": "COV-001",
                        "status": "active",
                        "beneficiary": {"reference": "Patient/PAT-001"},
                    }
                }
            ],
        }
        p = _write(tmp_path, bundle)
        result = run("CLM-TEST", str(p))
        assert result.status == VerificationStatus.FAIL
        assert any("Patient" in e for e in result.errors)

    def test_un_patient_respecte_cardinalite_min(self, tmp_path: Path) -> None:
        p = _write(tmp_path, _bundle())
        result = run("CLM-TEST", str(p))
        assert result.errors == []

    def test_deux_patients_respecte_cardinalite_min(self, tmp_path: Path) -> None:
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "PAT-001"}},
                {"resource": {"resourceType": "Patient", "id": "PAT-002"}},
            ],
        }
        p = _write(tmp_path, bundle)
        result = run("CLM-TEST", str(p))
        assert result.errors == []


# ── Profil inconnu ────────────────────────────────────────────────────────────


class TestProfilInconnu:
    def test_profil_inconnu_warning(self, tmp_path: Path) -> None:
        b = _bundle()
        b["entry"][0]["resource"]["meta"] = {
            "profile": ["http://example.test/fhir/CustomProfile"]
        }
        p = _write(tmp_path, b)
        result = run("CLM-TEST", str(p))
        assert any("profil non supporté localement" in w for w in result.warnings)

    def test_profil_inconnu_ne_bloque_pas(self, tmp_path: Path) -> None:
        b = _bundle()
        b["entry"][0]["resource"]["meta"] = {
            "profile": ["http://example.test/fhir/CustomProfile"]
        }
        p = _write(tmp_path, b)
        result = run("CLM-TEST", str(p))
        assert result.status in (VerificationStatus.PASS, VerificationStatus.NEEDS_REVIEW)
        assert result.errors == []

    def test_profil_r4_connu_pas_de_warning(self, tmp_path: Path) -> None:
        """Un profil contenant 'R4' est supporté — pas de warning de profil."""
        b = _bundle()
        b["entry"][0]["resource"]["meta"] = {"profile": ["R4"]}
        p = _write(tmp_path, b)
        result = run("CLM-TEST", str(p))
        assert not any("profil non supporté" in w for w in result.warnings)

    def test_profil_inconnu_url_dans_warning(self, tmp_path: Path) -> None:
        """L'URL du profil inconnu apparaît dans le warning."""
        url = "http://example.test/fhir/UnknownProfile"
        b = _bundle()
        b["entry"][0]["resource"]["meta"] = {"profile": [url]}
        p = _write(tmp_path, b)
        result = run("CLM-TEST", str(p))
        assert any(url in w for w in result.warnings)


# ── Version FHIR non prise en charge ─────────────────────────────────────────


class TestVersionFhir:
    def test_r4_default(self, tmp_path: Path) -> None:
        p = _write(tmp_path, _bundle())
        result = run("CLM-TEST", str(p))
        assert result.profile_checked == "R4"

    def test_r4b_supporte(self, tmp_path: Path) -> None:
        """fhir_version='R4B' est supporté — pas d'erreur de version."""
        p = _write(tmp_path, _bundle())
        result = run("CLM-TEST", str(p), fhir_version="R4B")
        assert result.errors == []

    def test_version_non_supportee_warning_schema(self, tmp_path: Path) -> None:
        """fhir_version non reconnue → validate_resource_schema produit un warning."""
        p = _write(tmp_path, _bundle())
        from tools.fhir_validation import _FHIR_LIB_AVAILABLE, validate_resource_schema

        if not _FHIR_LIB_AVAILABLE:
            pytest.skip("fhir.resources non installée")
        _, warns = validate_resource_schema(
            json.loads(p.read_text(encoding="utf-8")), fhir_version="STU3"
        )
        assert any("FHIR_RESOURCE_SCHEMA_VALID" in w for w in warns)
        assert any("non prise en charge" in w for w in warns)

    def test_version_stu3_unique_warning(self, tmp_path: Path) -> None:
        """Avec version STU3, seul l'avertissement de version est émis."""
        from tools.fhir_validation import _FHIR_LIB_AVAILABLE, validate_resource_schema

        if not _FHIR_LIB_AVAILABLE:
            pytest.skip("fhir.resources non installée")
        _, warns = validate_resource_schema(
            json.loads(_write(tmp_path, _bundle()).read_text("utf-8")),
            fhir_version="STU3",
        )
        assert len(warns) == 1


# ── Vérification SHA-256 ──────────────────────────────────────────────────────


class TestSha256:
    def test_hash_correct_passe(self, tmp_path: Path) -> None:
        p = _write(tmp_path, _bundle())
        sha = compute_sha256(p)
        result = run("CLM-TEST", str(p), expected_sha256=sha)
        assert result.status in (VerificationStatus.PASS, VerificationStatus.NEEDS_REVIEW)
        assert any("SHA-256 confirmée" in r for r in result.reasons)

    def test_hash_incorrect_fail(self, tmp_path: Path) -> None:
        """expected_sha256 incorrect → FAIL avant toute validation FHIR."""
        p = _write(tmp_path, _bundle())
        result = run("CLM-TEST", str(p), expected_sha256="a" * 64)
        assert result.status == VerificationStatus.FAIL
        assert any("intégrité échouée" in e for e in result.errors)

    def test_hash_incorrect_stoppe_validation(self, tmp_path: Path) -> None:
        """Échec SHA-256 → resource_types vide (validation FHIR non exécutée)."""
        p = _write(tmp_path, _bundle())
        result = run("CLM-TEST", str(p), expected_sha256="b" * 64)
        assert result.resource_types == []
        assert result.resource_count == 0

    def test_hash_absent_passe_sans_verif(self, tmp_path: Path) -> None:
        """expected_sha256=None → aucune erreur d'intégrité."""
        p = _write(tmp_path, _bundle())
        result = run("CLM-TEST", str(p), expected_sha256=None)
        assert not any("intégrité" in e for e in result.errors)
        assert any("hash attendu absent" in r for r in result.reasons)

    def test_message_hash_tronque(self, tmp_path: Path) -> None:
        """Le message d'erreur ne révèle que les 16 premiers caractères de chaque hash."""
        p = _write(tmp_path, _bundle())
        wrong = "c" * 64
        result = run("CLM-TEST", str(p), expected_sha256=wrong)
        sha_err = next(e for e in result.errors if "intégrité" in e)
        assert wrong not in sha_err          # hash complet absent
        assert wrong[:16] in sha_err         # préfixe présent


# ── Immuabilité du bundle original ───────────────────────────────────────────


class TestImmutabilite:
    def test_fichier_inchange_apres_run(self, tmp_path: Path) -> None:
        """run() ne modifie pas le fichier bundle sur disque."""
        p = _write(tmp_path, _bundle())
        sha_avant = compute_sha256(p)
        run("CLM-TEST", str(p))
        sha_apres = compute_sha256(p)
        assert sha_avant == sha_apres

    def test_bundle_content_absent_du_resultat(self, tmp_path: Path) -> None:
        """La valeur des champs FHIR ne doit pas apparaître dans FhirValidatorResult.

        Note : le NOM d'un champ mal formé peut figurer dans les warnings Pydantic
        (chemin du problème), mais sa VALEUR ne doit jamais être exposée.
        """
        b = _bundle()
        b["entry"][0]["resource"]["secret_field"] = "VALEUR_PII_SECRETE_12345"
        p = _write(tmp_path, b)
        result = run("CLM-TEST", str(p))
        result_json = result.model_dump_json()
        # La valeur PII ne doit jamais apparaître
        assert "VALEUR_PII_SECRETE_12345" not in result_json

    def test_full_text_bundle_absent_du_resultat(self) -> None:
        """Le contenu brut du fichier FHIR n'apparaît pas dans FhirValidatorResult."""
        result = run("CLM-0001", CLM_FIXTURE)
        raw = Path(CLM_FIXTURE).read_text(encoding="utf-8")
        first_chunk = raw[:50]
        result_json = result.model_dump_json()
        assert first_chunk not in result_json


# ── Déterminisme ──────────────────────────────────────────────────────────────


class TestDeterminisme:
    def test_meme_entree_meme_sortie_pass(self, tmp_path: Path) -> None:
        p = _write(tmp_path, _bundle())
        r1 = run("CLM-TEST", str(p))
        r2 = run("CLM-TEST", str(p))
        assert r1.status == r2.status
        assert r1.errors == r2.errors
        assert r1.warnings == r2.warnings
        assert r1.resource_types == r2.resource_types

    def test_meme_entree_meme_sortie_fail(self, tmp_path: Path) -> None:
        p = _write(tmp_path, _bundle(_coverage("Patient/ABSENT")))
        r1 = run("CLM-TEST", str(p))
        r2 = run("CLM-TEST", str(p))
        assert r1.status == r2.status == VerificationStatus.FAIL
        assert r1.errors == r2.errors

    def test_not_provided_deterministe(self) -> None:
        r1 = run("CLM-TEST", None, bundle_expected=False)
        r2 = run("CLM-TEST", None, bundle_expected=False)
        assert r1.status == r2.status == VerificationStatus.PASS
        assert r1.reasons == r2.reasons


# ── Security Gate ─────────────────────────────────────────────────────────────


class TestSecurityGate:
    def test_security_non_allow_fail(self, tmp_path: Path) -> None:
        p = _write(tmp_path, _bundle())
        result = run("CLM-TEST", str(p), security_allowed=False)
        assert result.status == VerificationStatus.FAIL
        assert result.validation_scope == "NOT_EVALUATED"
        assert any("Security Gate non ALLOW" in e for e in result.errors)

    def test_security_non_allow_aucune_erreur_fhir(self, tmp_path: Path) -> None:
        """Avec security_allowed=False, aucune erreur de contenu FHIR."""
        p = _write(tmp_path, _bundle())
        result = run("CLM-TEST", str(p), security_allowed=False)
        assert result.resource_types == []
        assert result.resource_count == 0

    def test_security_allow_validation_normale(self, tmp_path: Path) -> None:
        p = _write(tmp_path, _bundle())
        result = run("CLM-TEST", str(p), security_allowed=True)
        assert result.status in (VerificationStatus.PASS, VerificationStatus.NEEDS_REVIEW)

    def test_security_non_allow_raison_contient_message(self, tmp_path: Path) -> None:
        p = _write(tmp_path, _bundle())
        result = run("CLM-TEST", str(p), security_allowed=False)
        assert any("Validation FHIR non exécutée" in r for r in result.reasons)


# ── Nœud LangGraph ───────────────────────────────────────────────────────────


class TestNode:
    def test_fhir_input_consomme_a_none(self) -> None:
        updates = node({
            "case_id": "CLM-0001",
            "fhir_input": {"case_id": "CLM-0001", "fhir_bundle_path": None, "bundle_expected": False},
        })
        assert updates["fhir_input"] is None

    def test_fhir_result_ecrit(self) -> None:
        updates = node({
            "case_id": "CLM-0001",
            "fhir_input": {"case_id": "CLM-0001", "fhir_bundle_path": None, "bundle_expected": False},
        })
        assert isinstance(updates["fhir_result"], FhirValidatorResult)

    def test_current_step_et_completed_steps(self) -> None:
        updates = node({
            "case_id": "CLM-0001",
            "fhir_input": {"case_id": "CLM-0001", "fhir_bundle_path": None, "bundle_expected": False},
        })
        assert updates["current_step"] == "fhir_validation"
        assert "fhir_validation" in updates["completed_steps"]

    def test_audit_event_produit(self) -> None:
        updates = node({
            "case_id": "CLM-0001",
            "fhir_input": {"case_id": "CLM-0001", "fhir_bundle_path": None, "bundle_expected": False},
        })
        assert len(updates["audit_trail"]) == 1
        audit = updates["audit_trail"][0]
        assert isinstance(audit, AuditEvent)
        assert audit.actor == "fhir_validator_agent"
        assert audit.action == "fhir_validation"
        assert audit.case_id == "CLM-0001"

    def test_audit_sans_contenu_fhir_brut(self) -> None:
        """L'audit ne doit pas contenir le contenu des entrées FHIR."""
        updates = node({
            "case_id": "CLM-0001",
            "fhir_input": {
                "case_id": "CLM-0001",
                "fhir_bundle_path": CLM_FIXTURE,
                "bundle_expected": True,
            },
            "security_result": _mock_security(SecurityDecision.ALLOW),
        })
        audit_json = json.dumps(updates["audit_trail"][0].model_dump(mode="json"))
        assert "resourceType" not in audit_json
        assert '"entry"' not in audit_json

    def test_fhir_input_absent_fail(self) -> None:
        """fhir_input absent du state → FAIL NOT_EVALUATED."""
        updates = node({"case_id": "CLM-0001"})
        result = updates["fhir_result"]
        assert result.status == VerificationStatus.FAIL
        assert updates["fhir_input"] is None
        assert updates["audit_trail"]

    def test_fhir_input_invalide_fail(self) -> None:
        """fhir_input avec champ inconnu ou case_id vide → FAIL Pydantic."""
        updates = node({
            "case_id": "CLM-0001",
            "fhir_input": {"case_id": "", "unknown_field": "oops"},
        })
        assert updates["fhir_result"].status == VerificationStatus.FAIL
        assert updates["fhir_input"] is None

    def test_node_security_allow_valide(self) -> None:
        """Security Gate ALLOW + bundle CLM-0001 → validation exécutée."""
        updates = node({
            "case_id": "CLM-0001",
            "fhir_input": {
                "case_id": "CLM-0001",
                "fhir_bundle_path": CLM_FIXTURE,
                "bundle_expected": True,
            },
            "security_result": _mock_security(SecurityDecision.ALLOW),
        })
        result = updates["fhir_result"]
        assert result.status in (VerificationStatus.PASS, VerificationStatus.NEEDS_REVIEW)

    def test_node_security_block_not_evaluated(self) -> None:
        """Security Gate BLOCK → NOT_EVALUATED FAIL sans erreur FHIR."""
        updates = node({
            "case_id": "CLM-0001",
            "fhir_input": {
                "case_id": "CLM-0001",
                "fhir_bundle_path": CLM_FIXTURE,
                "bundle_expected": True,
            },
            "security_result": _mock_security(SecurityDecision.BLOCK),
        })
        result = updates["fhir_result"]
        assert result.status == VerificationStatus.FAIL
        assert result.validation_scope == "NOT_EVALUATED"

    def test_node_security_absent_not_evaluated(self) -> None:
        """Pas de security_result dans le state → NOT_EVALUATED."""
        updates = node({
            "case_id": "CLM-0001",
            "fhir_input": {
                "case_id": "CLM-0001",
                "fhir_bundle_path": CLM_FIXTURE,
                "bundle_expected": True,
            },
        })
        result = updates["fhir_result"]
        assert result.status == VerificationStatus.FAIL
        assert result.validation_scope == "NOT_EVALUATED"

    def test_node_sha256_manifest_correct(self) -> None:
        """Nœud extrait SHA-256 du manifest et confirme si correct."""
        sha = compute_sha256(Path(CLM_FIXTURE))
        intake = _mock_intake("patient_fhir_bundle.json", sha)
        updates = node({
            "case_id": "CLM-0001",
            "fhir_input": {
                "case_id": "CLM-0001",
                "fhir_bundle_path": CLM_FIXTURE,
                "bundle_expected": True,
            },
            "security_result": _mock_security(SecurityDecision.ALLOW),
            "intake_result": intake,
        })
        result = updates["fhir_result"]
        assert not any("intégrité échouée" in e for e in result.errors)
        assert any("SHA-256 confirmée" in r for r in result.reasons)

    def test_node_sha256_manifest_incorrect_fail(self) -> None:
        """SHA-256 incorrect dans le manifest → FAIL d'intégrité."""
        intake = _mock_intake("patient_fhir_bundle.json", "d" * 64)
        updates = node({
            "case_id": "CLM-0001",
            "fhir_input": {
                "case_id": "CLM-0001",
                "fhir_bundle_path": CLM_FIXTURE,
                "bundle_expected": True,
            },
            "security_result": _mock_security(SecurityDecision.ALLOW),
            "intake_result": intake,
        })
        assert updates["fhir_result"].status == VerificationStatus.FAIL
        assert any("intégrité échouée" in e for e in updates["fhir_result"].errors)

    def test_node_resultat_json_serialisable(self) -> None:
        """FhirValidatorResult produit par le nœud est JSON-sérialisable."""
        updates = node({
            "case_id": "CLM-0001",
            "fhir_input": {
                "case_id": "CLM-0001",
                "fhir_bundle_path": CLM_FIXTURE,
                "bundle_expected": True,
            },
            "security_result": _mock_security(SecurityDecision.ALLOW),
        })
        dumped = updates["fhir_result"].model_dump(mode="json")
        assert isinstance(dumped, dict)
        assert dumped["case_id"] == "CLM-0001"
        json.dumps(dumped)  # doit passer sans exception

    def test_node_not_provided_pass(self) -> None:
        """bundle_expected=False dans fhir_input → PASS NOT_PROVIDED."""
        updates = node({
            "case_id": "CLM-TEST",
            "fhir_input": {"case_id": "CLM-TEST", "fhir_bundle_path": None, "bundle_expected": False},
            "security_result": _mock_security(SecurityDecision.ALLOW),
        })
        assert updates["fhir_result"].status == VerificationStatus.PASS
        assert updates["fhir_result"].bundle_expected is False
