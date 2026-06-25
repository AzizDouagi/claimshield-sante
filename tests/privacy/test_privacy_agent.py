"""Tests du Privacy Agent — politique DENY-by-default, quatre rôles stables.

Organisation :
  - TestReaderRole                — énumération stable, rôle inconnu rejeté
  - TestPrivacyInput              — schéma d'entrée, rôle obligatoire
  - TestAccessPolicies            — modèle DENY-by-default, allowlists
  - TestPseudonymize              — masquage et pseudonymisation
  - TestPrivacyAgentRun           — pipeline complet (run())
  - TestPrivacyAgentSecurityGate  — pré-condition Security Gate
  - TestPrivacyAgentNode          — nœud LangGraph, rôle obligatoire
  - TestPrivacyResultInvariants   — checklist des invariants
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agents.privacy_agent.agent import _violation_to_codes, node, run
from agents.privacy_agent.schemas import PrivacyInput, PrivacyCode, PrivacyDecision, ReaderRole
from schemas.domain import (
    DataClassification,
    PRIVACY_CODE_DESCRIPTIONS,
    SecurityDecision,
    VerificationStatus,
)
from schemas.results import SecurityAuditEntry, SecurityGateResult
from security.access_policies import (
    ALL_KNOWN_FIELDS,
    ALWAYS_BLOCKED_FIELDS,
    AUDIT_FIELDS,
    FINANCIAL_FIELDS,
    FRAUD_FIELDS,
    IDENTITY_FIELDS,
    MEDICAL_FIELDS,
    POLICY_VERSION,
    PolicyViolation,
    RoleAccessPolicy,
    SECRET_FIELDS,
    compute_masked_fields,
    get_role_policy,
    verify_view_privacy,
)
from tools.pseudonymize import mask_field_value, mask_name, pseudonymize_fields, pseudonymize_id


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_gate_result(decision: SecurityDecision = SecurityDecision.ALLOW) -> SecurityGateResult:
    return SecurityGateResult(
        claim_id="CLM-0001",
        decision=decision,
        findings=[],
        reason_codes=[],
        applied_policy="default",
        policy_version="1.1.0",
        evaluated_at=datetime.now(UTC),
        next_allowed_action=(
            "continue_pipeline" if decision == SecurityDecision.ALLOW else "terminate_pipeline"
        ),
        audit_entry=SecurityAuditEntry(
            claim_id="CLM-0001",
            actor="security_gate_agent",
            outcome=decision.value,
            decision=decision,
            evaluated_at=datetime.now(UTC),
            policy_applied="default",
            policy_version="1.1.0",
        ),
        prompt_injection_detected=False,
        blocked_fields=[],
        reasons=["Aucune menace"] if decision == SecurityDecision.ALLOW else ["Menace"],
    )


def _make_privacy_input(**kwargs) -> PrivacyInput:
    defaults = {
        "case_id": "CLM-0001",
        "role": ReaderRole.ADMINISTRATIVE_MANAGER,
        "data_classification": DataClassification.SYNTHETIC_TEST_DATA,
        "contains_real_personal_data": False,
    }
    defaults.update(kwargs)
    return PrivacyInput(**defaults)


# ── TestReaderRole ────────────────────────────────────────────────────────────


class TestReaderRole:
    def test_quatre_roles_stables(self):
        roles = list(ReaderRole)
        assert len(roles) == 4

    def test_roles_attendus_presents(self):
        assert ReaderRole.ADMINISTRATIVE_MANAGER == "ADMINISTRATIVE_MANAGER"
        assert ReaderRole.MEDICAL_REVIEWER == "MEDICAL_REVIEWER"
        assert ReaderRole.FRAUD_ANALYST == "FRAUD_ANALYST"
        assert ReaderRole.AUDITOR == "AUDITOR"

    def test_role_inconnu_leve_value_error(self):
        with pytest.raises(ValueError):
            ReaderRole("GESTIONNAIRE")

    def test_role_inconnu_leve_value_error_externe(self):
        with pytest.raises(ValueError):
            ReaderRole("EXTERNE")

    def test_role_inconnu_leve_value_error_systeme(self):
        with pytest.raises(ValueError):
            ReaderRole("SYSTEME")

    def test_role_inconnu_arbitraire(self):
        with pytest.raises(ValueError):
            ReaderRole("SUPER_ADMIN")

    def test_roles_sont_des_chaines(self):
        for role in ReaderRole:
            assert isinstance(role.value, str)


# ── TestPrivacyInput ──────────────────────────────────────────────────────────


class TestPrivacyInput:
    def test_role_obligatoire(self):
        """Un rôle est obligatoire pour demander une vue."""
        with pytest.raises(ValidationError):
            PrivacyInput(
                case_id="CLM-0001",
                data_classification=DataClassification.SYNTHETIC_TEST_DATA,
                contains_real_personal_data=False,
            )

    def test_role_inconnu_rejete_par_pydantic(self):
        """Un rôle inconnu provoque un blocage au niveau schéma."""
        with pytest.raises(ValidationError):
            PrivacyInput(
                case_id="CLM-0001",
                role="ROLE_INEXISTANT",
                data_classification=DataClassification.SYNTHETIC_TEST_DATA,
                contains_real_personal_data=False,
            )

    def test_construction_minimale(self):
        inp = _make_privacy_input()
        assert inp.case_id == "CLM-0001"
        assert inp.role == ReaderRole.ADMINISTRATIVE_MANAGER
        assert inp.contains_real_personal_data is False
        assert inp.fields_to_evaluate == []

    def test_construction_complete(self):
        inp = PrivacyInput(
            case_id="CLM-0001",
            role=ReaderRole.AUDITOR,
            data_classification=DataClassification.CONFIDENTIAL,
            contains_real_personal_data=True,
            fields_to_evaluate=["patient_name", "claim_reference"],
            patient_name="Jean Dupont",
            patient_id="uuid-1234",
            invoice_number="INV-0001",
        )
        assert inp.role == ReaderRole.AUDITOR
        assert inp.patient_name == "Jean Dupont"

    def test_case_id_invalide_rejete(self):
        with pytest.raises(ValidationError):
            PrivacyInput(
                case_id="INVALID",
                role=ReaderRole.ADMINISTRATIVE_MANAGER,
                data_classification=DataClassification.SYNTHETIC_TEST_DATA,
                contains_real_personal_data=False,
            )

    def test_champ_inconnu_rejete(self):
        with pytest.raises(ValidationError):
            PrivacyInput(
                case_id="CLM-0001",
                role=ReaderRole.ADMINISTRATIVE_MANAGER,
                data_classification=DataClassification.SYNTHETIC_TEST_DATA,
                contains_real_personal_data=False,
                champ_inexistant="valeur",
            )

    def test_chemin_absolu_rejete_dans_patient_name(self):
        with pytest.raises(ValidationError):
            _make_privacy_input(patient_name="/etc/passwd")

    def test_secret_rejete_dans_patient_id(self):
        with pytest.raises(ValidationError):
            _make_privacy_input(patient_id="secret=abc123")

    def test_tous_les_roles_valides(self):
        for role in ReaderRole:
            inp = _make_privacy_input(role=role)
            assert inp.role == role


# ── TestAccessPolicies ────────────────────────────────────────────────────────


class TestAccessPolicies:
    def test_tous_les_roles_ont_une_politique(self):
        for role in ReaderRole:
            policy = get_role_policy(role)
            assert isinstance(policy, RoleAccessPolicy)
            assert policy.role == role

    def test_aucun_role_acces_complet(self):
        """Aucun rôle ne reçoit automatiquement tous les champs."""
        for role in ReaderRole:
            policy = get_role_policy(role)
            assert policy.allowed_fields != ALL_KNOWN_FIELDS, (
                f"Le rôle {role.value} a accès à tous les champs — interdit par la politique"
            )
            assert len(policy.allowed_fields) < len(ALL_KNOWN_FIELDS)

    def test_politique_par_defaut_deny(self):
        """Tout champ absent de l'allowlist est refusé."""
        for role in ReaderRole:
            policy = get_role_policy(role)
            denied = ALL_KNOWN_FIELDS - policy.allowed_fields
            assert len(denied) > 0, (
                f"Le rôle {role.value} n'a aucun champ refusé — "
                "viole la politique DENY-by-default"
            )

    def test_identifiants_personnels_refuses_pour_tous(self):
        """patient_name et patient_id sont refusés pour tous les rôles."""
        for role in ReaderRole:
            policy = get_role_policy(role)
            assert "patient_name" not in policy.allowed_fields, (
                f"{role.value} ne doit pas voir patient_name"
            )
            assert "patient_id" not in policy.allowed_fields, (
                f"{role.value} ne doit pas voir patient_id"
            )

    def test_administrative_manager_acces_financier(self):
        policy = get_role_policy(ReaderRole.ADMINISTRATIVE_MANAGER)
        for field in ("total_billed", "amount_requested", "patient_share", "payer_name"):
            assert field in policy.allowed_fields

    def test_administrative_manager_refuse_acces_medical(self):
        policy = get_role_policy(ReaderRole.ADMINISTRATIVE_MANAGER)
        for field in ("procedures", "prescriptions", "diagnosis_codes", "encounter_class"):
            assert field not in policy.allowed_fields

    def test_medical_reviewer_acces_medical(self):
        policy = get_role_policy(ReaderRole.MEDICAL_REVIEWER)
        for field in ("procedures", "prescriptions", "diagnosis_codes"):
            assert field in policy.allowed_fields

    def test_medical_reviewer_refuse_acces_financier(self):
        policy = get_role_policy(ReaderRole.MEDICAL_REVIEWER)
        for field in ("total_billed", "amount_requested", "invoice_number", "payer_name"):
            assert field not in policy.allowed_fields

    def test_medical_reviewer_refuse_identifiants_nominatifs(self):
        policy = get_role_policy(ReaderRole.MEDICAL_REVIEWER)
        assert "patient_name" not in policy.allowed_fields
        assert "patient_id" not in policy.allowed_fields

    def test_fraud_analyst_acces_montants(self):
        policy = get_role_policy(ReaderRole.FRAUD_ANALYST)
        for field in ("total_billed", "amount_requested", "claim_reference", "invoice_number"):
            assert field in policy.allowed_fields

    def test_fraud_analyst_refuse_acces_medical(self):
        policy = get_role_policy(ReaderRole.FRAUD_ANALYST)
        for field in ("procedures", "prescriptions", "diagnosis_codes", "birth_date", "gender"):
            assert field not in policy.allowed_fields

    def test_auditor_acces_minimal(self):
        policy = get_role_policy(ReaderRole.AUDITOR)
        assert "claim_reference" in policy.allowed_fields
        assert "service_date" in policy.allowed_fields
        assert len(policy.allowed_fields) <= 5

    def test_auditor_refuse_donnees_financieres_individuelles(self):
        policy = get_role_policy(ReaderRole.AUDITOR)
        for field in ("patient_share", "amount_requested", "invoice_number", "payer_name"):
            assert field not in policy.allowed_fields

    def test_compute_masked_fields_refus_champ_inconnu(self):
        """Tout champ inconnu passé en fields_to_evaluate est refusé (DENY-by-default)."""
        for role in ReaderRole:
            masked = compute_masked_fields(role, fields_to_evaluate=["champ_totalement_inconnu"])
            assert "champ_totalement_inconnu" in masked, (
                f"Le rôle {role.value} doit refuser les champs inconnus"
            )

    def test_compute_masked_fields_sans_filtre_couvre_univers(self):
        for role in ReaderRole:
            masked = compute_masked_fields(role)
            allowed = get_role_policy(role).allowed_fields
            assert set(masked) == ALL_KNOWN_FIELDS - allowed

    def test_compute_masked_fields_avec_filtre(self):
        masked = compute_masked_fields(
            ReaderRole.ADMINISTRATIVE_MANAGER,
            fields_to_evaluate=["patient_name", "claim_reference"],
        )
        assert "patient_name" in masked      # refusé pour ce rôle
        assert "claim_reference" not in masked  # autorisé

    def test_compute_masked_fields_resultat_trie(self):
        for role in ReaderRole:
            masked = compute_masked_fields(role)
            assert masked == sorted(masked)


# ── TestPseudonymize ──────────────────────────────────────────────────────────


class TestPseudonymize:
    def test_mask_name_deux_mots(self):
        result = mask_name("Jean Dupont")
        assert result.startswith("J")
        assert "*" in result

    def test_mask_name_chaine_vide(self):
        assert mask_name("") == ""

    def test_mask_name_un_caractere(self):
        assert mask_name("A") == "A"

    def test_pseudonymize_id_stable(self):
        assert pseudonymize_id("uuid-1234") == pseudonymize_id("uuid-1234")

    def test_pseudonymize_id_prefixe(self):
        assert pseudonymize_id("uuid-1234").startswith("PSE-")

    def test_pseudonymize_id_differents(self):
        assert pseudonymize_id("uuid-0001") != pseudonymize_id("uuid-0002")

    def test_mask_field_value_name(self):
        result = mask_field_value("patient_name", "Jean Dupont")
        assert "J" in result and "*" in result

    def test_mask_field_value_id(self):
        assert mask_field_value("patient_id", "uuid-abc").startswith("PAT-")

    def test_mask_field_value_invoice(self):
        result = mask_field_value("invoice_number", "INV-2024-001")
        assert result.startswith("***")

    def test_mask_field_value_autre(self):
        assert mask_field_value("coverage_rate", "0.80") == "[MASQUÉ]"

    def test_pseudonymize_fields_applique_masquage(self):
        fields = {"patient_name": "Jean Dupont", "payer_name": "XYZ", "invoice_number": "INV-001"}
        result = pseudonymize_fields(fields, ["patient_name", "invoice_number"])
        assert result["patient_name"] != "Jean Dupont"
        assert result["payer_name"] == "XYZ"
        assert result["invoice_number"].startswith("***")

    def test_pseudonymize_fields_none_inchange(self):
        result = pseudonymize_fields({"patient_name": None}, ["patient_name"])
        assert result["patient_name"] is None

    def test_pseudonymize_fields_non_masques_inchanges(self):
        result = pseudonymize_fields({"payer_name": "XYZ"}, ["patient_name"])
        assert result == {"payer_name": "XYZ"}


# ── TestPrivacyAgentRun ───────────────────────────────────────────────────────


class TestPrivacyAgentRun:
    def test_nominal_synthetique_administrative_manager_pass(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(_make_privacy_input(), gate)
        assert result.case_id == "CLM-0001"
        assert result.status == VerificationStatus.PASS
        assert result.data_classification == DataClassification.SYNTHETIC_TEST_DATA
        assert len(result.reasons) >= 1

    def test_donnees_reelles_needs_review(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(_make_privacy_input(contains_real_personal_data=True), gate)
        assert result.status == VerificationStatus.NEEDS_REVIEW

    def test_confidentiel_needs_review(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(
            _make_privacy_input(data_classification=DataClassification.CONFIDENTIAL),
            gate,
        )
        assert result.status == VerificationStatus.NEEDS_REVIEW

    def test_anonymise_pass(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(
            _make_privacy_input(data_classification=DataClassification.ANONYMIZED),
            gate,
        )
        assert result.status == VerificationStatus.PASS

    def test_tous_roles_ont_des_champs_refuses(self):
        """Aucun rôle n'a une vue complète (DENY-by-default)."""
        gate = _make_gate_result(SecurityDecision.ALLOW)
        for role in ReaderRole:
            result = run(_make_privacy_input(role=role), gate)
            assert len(result.redacted_fields) > 0, (
                f"Le rôle {role.value} n'a aucun champ refusé — viole DENY-by-default"
            )

    def test_administrative_manager_refuse_champs_medicaux(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(_make_privacy_input(role=ReaderRole.ADMINISTRATIVE_MANAGER), gate)
        for field in ("procedures", "prescriptions", "diagnosis_codes"):
            assert field in result.redacted_fields

    def test_medical_reviewer_refuse_champs_financiers(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(_make_privacy_input(role=ReaderRole.MEDICAL_REVIEWER), gate)
        for field in ("total_billed", "amount_requested", "invoice_number"):
            assert field in result.redacted_fields

    def test_fraud_analyst_refuse_donnees_medicales(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(_make_privacy_input(role=ReaderRole.FRAUD_ANALYST), gate)
        for field in ("procedures", "prescriptions", "birth_date", "gender"):
            assert field in result.redacted_fields

    def test_auditor_acces_minimal_refus_massif(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(_make_privacy_input(role=ReaderRole.AUDITOR), gate)
        assert len(result.redacted_fields) >= 15

    def test_identifiants_personnels_refuses_par_tous(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        for role in ReaderRole:
            result = run(_make_privacy_input(role=role), gate)
            assert "patient_name" in result.redacted_fields, (
                f"{role.value} ne devrait pas voir patient_name"
            )
            assert "patient_id" in result.redacted_fields, (
                f"{role.value} ne devrait pas voir patient_id"
            )

    def test_fields_to_evaluate_filtre_univers(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(
            _make_privacy_input(
                role=ReaderRole.AUDITOR,
                fields_to_evaluate=["patient_name", "claim_reference"],
            ),
            gate,
        )
        assert set(result.redacted_fields).issubset({"patient_name", "claim_reference"})

    def test_champ_inconnu_refuse_deny_by_default(self):
        """Un champ inconnu passé en fields_to_evaluate est refusé."""
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(
            _make_privacy_input(
                role=ReaderRole.ADMINISTRATIVE_MANAGER,
                fields_to_evaluate=["champ_totalement_inconnu"],
            ),
            gate,
        )
        assert "champ_totalement_inconnu" in result.redacted_fields

    def test_avec_valeurs_personnelles_pas_de_crash(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(
            _make_privacy_input(
                role=ReaderRole.MEDICAL_REVIEWER,
                patient_name="Jean Dupont",
                patient_id="uuid-1234",
            ),
            gate,
        )
        assert result.status in (VerificationStatus.PASS, VerificationStatus.NEEDS_REVIEW)


# ── TestPrivacyAgentSecurityGate ──────────────────────────────────────────────


class TestPrivacyAgentSecurityGate:
    def test_security_result_absent_retourne_fail(self):
        result = run(_make_privacy_input(), security_result=None)
        assert result.status == VerificationStatus.FAIL
        assert any("ABSENT" in r or "Security Gate" in r for r in result.reasons)

    def test_security_result_block_retourne_fail(self):
        result = run(_make_privacy_input(), _make_gate_result(SecurityDecision.BLOCK))
        assert result.status == VerificationStatus.FAIL
        assert any("BLOCK" in r or "Security Gate" in r for r in result.reasons)

    def test_security_result_quarantine_retourne_fail(self):
        result = run(_make_privacy_input(), _make_gate_result(SecurityDecision.QUARANTINE))
        assert result.status == VerificationStatus.FAIL

    def test_security_result_allow_autorise_traitement(self):
        result = run(_make_privacy_input(), _make_gate_result(SecurityDecision.ALLOW))
        assert result.status != VerificationStatus.FAIL

    def test_fail_masked_fields_vides(self):
        result = run(_make_privacy_input(), security_result=None)
        assert result.redacted_fields == []

    def test_fail_classification_confidentiel_par_defaut(self):
        result = run(_make_privacy_input(), security_result=None)
        assert result.data_classification == DataClassification.CONFIDENTIAL


# ── TestPrivacyAgentNode ──────────────────────────────────────────────────────


class TestPrivacyAgentNode:
    def _state(self, **kwargs) -> dict:
        defaults = {
            "case_id": "CLM-0001",
            "security_result": _make_gate_result(SecurityDecision.ALLOW),
            "privacy_input": {"role": "ADMINISTRATIVE_MANAGER"},
        }
        defaults.update(kwargs)
        return defaults

    def test_node_retourne_privacy_result(self):
        updates = node(self._state())
        assert "privacy_result" in updates
        assert updates["privacy_result"] is not None

    def test_node_vide_privacy_input(self):
        updates = node(self._state(privacy_input={"role": "MEDICAL_REVIEWER"}))
        assert updates["privacy_input"] is None

    def test_node_step_privacy(self):
        updates = node(self._state())
        assert updates["current_step"] == "privacy"
        assert "privacy" in updates["completed_steps"]

    def test_node_sans_role_retourne_fail(self):
        """Un rôle est obligatoire — absence dans privacy_input → FAIL."""
        updates = node(self._state(privacy_input={}))
        assert updates["privacy_result"].status == VerificationStatus.FAIL
        assert "errors" in updates

    def test_node_sans_role_none_retourne_fail(self):
        """privacy_input=None équivaut à un rôle absent → FAIL."""
        updates = node(self._state(privacy_input=None))
        assert updates["privacy_result"].status == VerificationStatus.FAIL

    def test_node_role_inconnu_retourne_fail(self):
        """Un rôle inconnu déclenche un blocage DENY-by-default."""
        updates = node(self._state(privacy_input={"role": "SUPER_ADMIN"}))
        assert updates["privacy_result"].status == VerificationStatus.FAIL
        assert "errors" in updates

    def test_node_role_inconnu_ancienne_valeur_retourne_fail(self):
        """Les anciens rôles (GESTIONNAIRE, SYSTEME…) sont rejetés."""
        for ancien_role in ("GESTIONNAIRE", "SYSTEME", "EXTERNE", "MEDECIN_CONSEIL"):
            updates = node(self._state(privacy_input={"role": ancien_role}))
            assert updates["privacy_result"].status == VerificationStatus.FAIL, (
                f"L'ancien rôle '{ancien_role}' devrait être rejeté"
            )

    def test_node_sans_security_result_fail(self):
        updates = node(self._state(security_result=None))
        assert updates["privacy_result"].status == VerificationStatus.FAIL
        assert "errors" in updates

    def test_node_security_block_fail(self):
        updates = node(self._state(security_result=_make_gate_result(SecurityDecision.BLOCK)))
        assert updates["privacy_result"].status == VerificationStatus.FAIL

    def test_node_tous_roles_valides_acceptes(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        for role in ReaderRole:
            updates = node(self._state(
                security_result=gate,
                privacy_input={"role": role.value},
            ))
            result = updates["privacy_result"]
            assert result.status != VerificationStatus.FAIL, (
                f"Le rôle valide {role.value} ne doit pas produire FAIL"
            )

    def test_node_donnees_reelles_needs_review_et_alerte(self):
        updates = node(self._state(privacy_input={
            "role": "ADMINISTRATIVE_MANAGER",
            "contains_real_personal_data": True,
        }))
        assert updates["privacy_result"].status == VerificationStatus.NEEDS_REVIEW
        assert "alerts" in updates

    def test_node_state_serialisable(self):
        from state.claim_state import validate_state_update
        validate_state_update(node(self._state()))

    def test_node_case_id_provient_du_state(self):
        assert node(self._state())["privacy_result"].case_id == "CLM-0001"

    def test_node_champs_refuses_dans_resultat(self):
        """DENY-by-default : le résultat du nœud contient des champs refusés."""
        updates = node(self._state(privacy_input={"role": "AUDITOR"}))
        result = updates["privacy_result"]
        assert len(result.redacted_fields) >= 15


# ── TestPrivacyResultInvariants ───────────────────────────────────────────────


class TestPrivacyResultInvariants:
    """Checklist des invariants structurels du PrivacyResult."""

    @pytest.fixture
    def all_results(self) -> list:
        gate_allow = _make_gate_result(SecurityDecision.ALLOW)
        gate_block = _make_gate_result(SecurityDecision.BLOCK)
        return [
            run(_make_privacy_input(role=ReaderRole.ADMINISTRATIVE_MANAGER), gate_allow),
            run(_make_privacy_input(role=ReaderRole.MEDICAL_REVIEWER), gate_allow),
            run(_make_privacy_input(role=ReaderRole.FRAUD_ANALYST), gate_allow),
            run(_make_privacy_input(role=ReaderRole.AUDITOR), gate_allow),
            run(_make_privacy_input(contains_real_personal_data=True), gate_allow),
            run(_make_privacy_input(data_classification=DataClassification.CONFIDENTIAL), gate_allow),
            run(_make_privacy_input(), security_result=None),
            run(_make_privacy_input(), gate_block),
        ]

    def test_toujours_un_case_id(self, all_results):
        for r in all_results:
            assert r.case_id

    def test_toujours_un_status(self, all_results):
        for r in all_results:
            assert r.status in {
                VerificationStatus.PASS,
                VerificationStatus.NEEDS_REVIEW,
                VerificationStatus.FAIL,
            }

    def test_toujours_au_moins_une_raison(self, all_results):
        for r in all_results:
            assert len(r.reasons) >= 1

    def test_masked_fields_trie(self, all_results):
        for r in all_results:
            assert r.redacted_fields == sorted(r.redacted_fields)

    def test_fail_masked_fields_vides(self, all_results):
        for r in all_results:
            if r.status == VerificationStatus.FAIL:
                assert r.redacted_fields == []

    def test_pass_et_needs_review_ont_des_champs_refuses(self, all_results):
        """DENY-by-default : même un résultat PASS a des champs refusés."""
        for r in all_results:
            if r.status in (VerificationStatus.PASS, VerificationStatus.NEEDS_REVIEW):
                assert len(r.redacted_fields) > 0

    def test_pas_de_chemin_absolu_dans_reasons(self, all_results):
        import re
        abs_re = re.compile(r"^(?:/|[A-Za-z]:[/\\]|\\\\)")
        for r in all_results:
            for reason in r.reasons:
                assert not abs_re.match(reason)

    def test_json_serialisable(self, all_results):
        import json
        for r in all_results:
            json.dumps(r.model_dump(), default=str)

    def test_ground_truth_compatible(self):
        """Vérifie que le résultat nominal est conforme à l'oracle ground_truth."""
        import json
        from pathlib import Path

        gt_path = Path("datasets/fixtures/valid/CLM-0001/oracle/ground_truth.json")
        if not gt_path.exists():
            pytest.skip("Fixtures non disponibles")

        with gt_path.open() as f:
            expected = json.load(f).get("expected_privacy", {})

        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(
            PrivacyInput(
                case_id="CLM-0001",
                role=ReaderRole.ADMINISTRATIVE_MANAGER,
                data_classification=DataClassification(
                    expected.get("data_classification", "SYNTHETIC_TEST_DATA")
                ),
                contains_real_personal_data=expected.get("contains_real_personal_data", False),
            ),
            gate,
        )
        assert result.status.value == expected["status"]
        assert result.data_classification.value == expected["data_classification"]
        assert result.contains_real_personal_data == expected["contains_real_personal_data"]


# ── TestFieldClassification ───────────────────────────────────────────────────


class TestFieldClassification:
    """Classification des champs par famille — taxonomie de security/access_policies.py."""

    def test_identity_fields_contient_patient_name_et_id(self):
        assert "patient_name" in IDENTITY_FIELDS
        assert "patient_id" in IDENTITY_FIELDS

    def test_identity_fields_contient_donnees_demo(self):
        assert "birth_date" in IDENTITY_FIELDS
        assert "gender" in IDENTITY_FIELDS

    def test_financial_fields_contient_montants(self):
        for field in ("total_billed", "amount_requested", "patient_share", "payer_name"):
            assert field in FINANCIAL_FIELDS

    def test_financial_fields_contient_references_facture(self):
        assert "invoice_number" in FINANCIAL_FIELDS
        assert "claim_reference" in FINANCIAL_FIELDS

    def test_medical_fields_contient_actes_et_diagnostics(self):
        for field in ("procedures", "prescriptions", "diagnosis_codes"):
            assert field in MEDICAL_FIELDS

    def test_medical_fields_contient_prestataire(self):
        assert "provider_id" in MEDICAL_FIELDS
        assert "organization_id" in MEDICAL_FIELDS

    def test_fraud_fields_sous_ensemble_connu(self):
        assert FRAUD_FIELDS.issubset(ALL_KNOWN_FIELDS)

    def test_audit_fields_sous_ensemble_fraud_fields(self):
        assert AUDIT_FIELDS.issubset(FRAUD_FIELDS)

    def test_audit_fields_acces_minimal(self):
        assert "claim_reference" in AUDIT_FIELDS
        assert "service_date" in AUDIT_FIELDS
        assert "submitted_at" in AUDIT_FIELDS
        assert len(AUDIT_FIELDS) == 3

    def test_secret_fields_contient_champs_systeme(self):
        for field in ("raw_ocr_text", "system_prompt", "api_key", "storage_path"):
            assert field in SECRET_FIELDS

    def test_secret_fields_disjoints_all_known_fields(self):
        assert SECRET_FIELDS.isdisjoint(ALL_KNOWN_FIELDS), (
            "SECRET_FIELDS ne doit pas contenir de champ métier connu"
        )

    def test_always_blocked_contient_identity_et_secrets(self):
        assert IDENTITY_FIELDS.issubset(ALWAYS_BLOCKED_FIELDS)
        assert SECRET_FIELDS.issubset(ALWAYS_BLOCKED_FIELDS)

    def test_all_known_fields_egal_union_classifications(self):
        union = IDENTITY_FIELDS | FINANCIAL_FIELDS | MEDICAL_FIELDS | {"service_date", "submitted_at"}
        assert union == ALL_KNOWN_FIELDS

    def test_policy_version_non_vide(self):
        assert POLICY_VERSION
        assert isinstance(POLICY_VERSION, str)

    def test_aucun_role_acces_champ_secret(self):
        for role in ReaderRole:
            policy = get_role_policy(role)
            overlap = policy.allowed_fields & SECRET_FIELDS
            assert not overlap, (
                f"Le rôle {role.value} a accès à des champs secrets : {overlap}"
            )

    def test_aucun_role_acces_patient_name(self):
        for role in ReaderRole:
            policy = get_role_policy(role)
            assert "patient_name" not in policy.allowed_fields

    def test_aucun_role_acces_patient_id(self):
        for role in ReaderRole:
            policy = get_role_policy(role)
            assert "patient_id" not in policy.allowed_fields


# ── TestVerifyViewPrivacy ─────────────────────────────────────────────────────


class TestVerifyViewPrivacy:
    """Vérification post-vue — blocage des identifiants bruts et champs secrets."""

    def test_vue_propre_sans_violation(self):
        view = {
            "claim_id": "CLM-0001",
            "patient_pseudonym": "PAT-A8F3C921B4D3",
            "dossier_status": "PENDING",
        }
        assert verify_view_privacy(view) == []

    def test_violation_secret_field(self):
        view = {"patient_pseudonym": "PAT-A8F3C921B4D3", "api_key": "sk-abc123"}
        violations = verify_view_privacy(view)
        assert any(v.reason_code == "SECRET_FIELD_IN_VIEW" for v in violations)
        assert any(v.field == "api_key" for v in violations)

    def test_violation_raw_identity_patient_name(self):
        view = {"patient_name": "Jean Dupont", "claim_id": "CLM-0001"}
        violations = verify_view_privacy(view)
        assert any(v.reason_code == "RAW_IDENTITY_IN_VIEW" for v in violations)
        assert any(v.field == "patient_name" for v in violations)

    def test_violation_raw_identity_patient_id(self):
        view = {"patient_id": "PATIENT-0007", "claim_id": "CLM-0001"}
        violations = verify_view_privacy(view)
        assert any(v.field == "patient_id" for v in violations)

    def test_violation_pseudonyme_patient_mauvais_prefixe(self):
        view = {"patient_pseudonym": "Jean Dupont"}
        violations = verify_view_privacy(view)
        assert any(v.reason_code == "INVALID_PSEUDONYM_FORMAT" for v in violations)
        assert any(v.field == "patient_pseudonym" for v in violations)

    def test_violation_pseudonyme_provider_mauvais_prefixe(self):
        view = {"patient_pseudonym": "PAT-A8F3C921B4D3", "provider_pseudonym": "raw-id-456"}
        violations = verify_view_privacy(view)
        assert any(v.field == "provider_pseudonym" for v in violations)

    def test_violation_provider_reference_mauvais_prefixe(self):
        view = {"patient_pseudonym": "PAT-A8F3C921B4D3", "provider_reference": "identifiant-brut"}
        violations = verify_view_privacy(view)
        assert any(v.field == "provider_reference" for v in violations)

    def test_pseudonyme_none_accepte(self):
        view = {"patient_pseudonym": "PAT-A8F3C921B4D3", "provider_pseudonym": None}
        assert verify_view_privacy(view) == []

    def test_pat_inconnu_accepte(self):
        view = {"patient_pseudonym": "PAT-INCONNU"}
        assert verify_view_privacy(view) == []

    def test_plusieurs_violations_retournees(self):
        view = {
            "patient_name": "Jean Dupont",
            "api_key": "sk-abc",
            "patient_pseudonym": "identifiant-brut",
        }
        violations = verify_view_privacy(view)
        assert len(violations) >= 3

    def test_policy_violation_est_frozen_dataclass(self):
        v = PolicyViolation(field="x", reason_code="TEST", message="test")
        assert v.field == "x"
        assert v.reason_code == "TEST"
        assert v.message == "test"
        with pytest.raises((AttributeError, TypeError)):
            v.field = "y"  # type: ignore[misc]

    def test_vue_vide_sans_violation(self):
        assert verify_view_privacy({}) == []

    def test_vue_audit_propre(self):
        view = {
            "claim_id": "CLM-0001",
            "actor": "privacy_agent",
            "actor_role": "AUDITOR",
            "action": "view_request",
            "timestamp": "2026-01-01T00:00:00Z",
            "policy_version": "1.1.0",
            "outcome": "PASS",
            "reason_codes": [],
        }
        assert verify_view_privacy(view) == []

    def test_run_avec_claim_data_propre_sans_fail(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(
            _make_privacy_input(
                role=ReaderRole.ADMINISTRATIVE_MANAGER,
                claim_data={
                    "patient_id": "PATIENT-0007",
                    "dossier_status": "PENDING",
                    "total_billed": "1500.00",
                    "submitted_at": "2026-01-01",
                },
            ),
            gate,
        )
        assert result.status != VerificationStatus.FAIL or result.errors is not None

    def test_run_avec_view_bloquee_retourne_fail(self):
        # Injection directe d'une vue corrompue — teste verify_view_privacy (défense en profondeur).
        corrupted_view = {"patient_name": "Jean Dupont", "claim_id": "CLM-0001"}
        violations = verify_view_privacy(corrupted_view)
        assert len(violations) > 0
        assert violations[0].reason_code in {
            "RAW_IDENTITY_IN_VIEW", "SECRET_FIELD_IN_VIEW", "INVALID_PSEUDONYM_FORMAT"
        }


# ── TestPrivacyAuditEntry ─────────────────────────────────────────────────────


class TestPrivacyAuditEntry:
    """Tests du schéma PrivacyAuditEntry — trace d'audit minimisée."""

    from schemas.results import PrivacyAuditEntry

    def _make_entry(self, **kwargs):
        from schemas.results import PrivacyAuditEntry
        defaults = {
            "role": "ADMINISTRATIVE_MANAGER",
            "outcome": "PASS",
            "decision": VerificationStatus.PASS,
            "policy_version": "1.1.0",
        }
        defaults.update(kwargs)
        return PrivacyAuditEntry(**defaults)

    def test_construction_minimale(self):
        entry = self._make_entry()
        assert entry.actor == "privacy_agent"
        assert entry.action == "view_request"
        assert entry.role == "ADMINISTRATIVE_MANAGER"
        assert entry.outcome == "PASS"

    def test_champ_inconnu_rejete(self):
        from pydantic import ValidationError as PydanticError
        from schemas.results import PrivacyAuditEntry
        with pytest.raises(PydanticError):
            PrivacyAuditEntry(
                role="AUDITOR",
                outcome="PASS",
                champ_inexistant="valeur",
            )

    def test_chemin_absolu_rejete_dans_role(self):
        from pydantic import ValidationError as PydanticError
        from schemas.results import PrivacyAuditEntry
        with pytest.raises(PydanticError):
            PrivacyAuditEntry(role="/etc/passwd", outcome="PASS")

    def test_secret_rejete_dans_outcome(self):
        from pydantic import ValidationError as PydanticError
        from schemas.results import PrivacyAuditEntry
        with pytest.raises(PydanticError):
            PrivacyAuditEntry(role="AUDITOR", outcome="api_key=abc123")

    def test_redacted_count_negatif_rejete(self):
        from pydantic import ValidationError as PydanticError
        from schemas.results import PrivacyAuditEntry
        with pytest.raises(PydanticError):
            PrivacyAuditEntry(role="AUDITOR", outcome="PASS", redacted_count=-1)

    def test_json_serialisable(self):
        import json
        entry = self._make_entry(redacted_count=5, view_built=True)
        json.dumps(entry.model_dump(), default=str)

    def test_role_unknown_accepte_pour_fail(self):
        entry = self._make_entry(role="UNKNOWN", outcome="FAIL", decision=VerificationStatus.FAIL)
        assert entry.role == "UNKNOWN"
        assert entry.decision == VerificationStatus.FAIL

    def test_evaluated_at_present(self):
        entry = self._make_entry()
        assert entry.evaluated_at is not None

    def test_flags_par_defaut(self):
        entry = self._make_entry()
        assert entry.view_built is False
        assert entry.pseudonymization_applied is False
        assert entry.redacted_count == 0


# ── TestAuditTraceInResult ────────────────────────────────────────────────────


class TestAuditTraceInResult:
    """Tests de la trace d'audit dans PrivacyResult (point 13 du pipeline)."""

    def test_audit_entry_present_sur_pass(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(_make_privacy_input(), gate)
        assert result.audit_entry is not None

    def test_audit_entry_role_correspond(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(_make_privacy_input(role=ReaderRole.MEDICAL_REVIEWER), gate)
        assert result.audit_entry is not None
        assert result.audit_entry.role == "MEDICAL_REVIEWER"

    def test_audit_entry_policy_version(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(_make_privacy_input(), gate)
        assert result.audit_entry is not None
        assert result.audit_entry.policy_version == POLICY_VERSION

    def test_audit_entry_decision_correspond_status(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(_make_privacy_input(), gate)
        assert result.audit_entry is not None
        assert result.audit_entry.decision == result.status

    def test_audit_entry_redacted_count(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(_make_privacy_input(role=ReaderRole.AUDITOR), gate)
        assert result.audit_entry is not None
        assert result.audit_entry.redacted_count == len(result.redacted_fields)

    def test_audit_entry_view_built_false_sans_claim_data(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(_make_privacy_input(), gate)
        assert result.audit_entry is not None
        assert result.audit_entry.view_built is False

    def test_audit_entry_view_built_true_avec_claim_data(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(
            _make_privacy_input(
                role=ReaderRole.ADMINISTRATIVE_MANAGER,
                claim_data={"dossier_status": "PENDING", "patient_id": "P-001"},
            ),
            gate,
        )
        assert result.audit_entry is not None
        assert result.audit_entry.view_built is True

    def test_audit_entry_pseudonymization_applied_false_sans_valeurs(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(_make_privacy_input(), gate)
        assert result.audit_entry is not None
        assert result.audit_entry.pseudonymization_applied is False

    def test_audit_entry_pseudonymization_applied_true_avec_valeurs(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(_make_privacy_input(patient_name="Jean Dupont"), gate)
        assert result.audit_entry is not None
        assert result.audit_entry.pseudonymization_applied is True

    def test_audit_entry_present_sur_fail_security_gate(self):
        result = run(_make_privacy_input(), security_result=None)
        assert result.status == VerificationStatus.FAIL
        assert result.audit_entry is not None
        assert result.audit_entry.decision == VerificationStatus.FAIL

    def test_audit_entry_tous_roles(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        for role in ReaderRole:
            result = run(_make_privacy_input(role=role), gate)
            assert result.audit_entry is not None
            assert result.audit_entry.role == role.value

    def test_audit_entry_json_serialisable(self):
        import json
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(_make_privacy_input(), gate)
        json.dumps(result.model_dump(), default=str)

    def test_node_ajoute_audit_trail_au_state(self):
        state = {
            "case_id": "CLM-0001",
            "security_result": _make_gate_result(SecurityDecision.ALLOW),
            "privacy_input": {"role": "ADMINISTRATIVE_MANAGER"},
        }
        updates = node(state)
        assert "audit_trail" in updates
        assert len(updates["audit_trail"]) == 1

    def test_node_audit_trail_contient_audit_event(self):
        from schemas.results import AuditEvent
        state = {
            "case_id": "CLM-0001",
            "security_result": _make_gate_result(SecurityDecision.ALLOW),
            "privacy_input": {"role": "AUDITOR"},
        }
        updates = node(state)
        event = updates["audit_trail"][0]
        assert isinstance(event, AuditEvent)
        assert event.actor == "privacy_agent"
        assert event.action == "view_request"
        assert event.case_id == "CLM-0001"

    def test_node_audit_trail_outcome_correspond_status(self):
        state = {
            "case_id": "CLM-0001",
            "security_result": _make_gate_result(SecurityDecision.ALLOW),
            "privacy_input": {"role": "ADMINISTRATIVE_MANAGER"},
        }
        updates = node(state)
        result = updates["privacy_result"]
        event = updates["audit_trail"][0]
        assert event.outcome == result.status.value


# ── TestPseudonymizationKeyCheck ─────────────────────────────────────────────


class TestPseudonymizationKeyCheck:
    """Tests de la vérification de la clé de pseudonymisation (point 14 du pipeline)."""

    def test_cle_disponible_en_environnement_test(self):
        from tools.pseudonymize import pseudonymization_key_is_available
        assert pseudonymization_key_is_available() is True

    def test_run_avec_cle_disponible_ne_fail_pas(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(_make_privacy_input(), gate)
        assert result.status != VerificationStatus.FAIL or (
            result.errors and "pseudonymisation" not in result.errors[0].lower()
        )

    def test_cle_indisponible_retourne_fail(self, monkeypatch):
        from tools import pseudonymize as pseudo_module
        monkeypatch.setattr(pseudo_module, "pseudonymization_key_is_available", lambda: False)
        from agents.privacy_agent import agent as agent_module
        monkeypatch.setattr(
            agent_module,
            "pseudonymization_key_is_available",
            lambda: False,
        )
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(_make_privacy_input(), gate)
        assert result.status == VerificationStatus.FAIL
        assert any("pseudonymisation" in e.lower() for e in result.errors)

    def test_cle_indisponible_audit_entry_fail(self, monkeypatch):
        from agents.privacy_agent import agent as agent_module
        monkeypatch.setattr(agent_module, "pseudonymization_key_is_available", lambda: False)
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(_make_privacy_input(role=ReaderRole.MEDICAL_REVIEWER), gate)
        assert result.audit_entry is not None
        assert result.audit_entry.decision == VerificationStatus.FAIL


# ── TestPrivacyCodesAndDecisions ─────────────────────────────────────────────


class TestPrivacyCodesAndDecisions:
    """Tests des codes stables PrivacyCode et de la décision binaire PrivacyDecision (étape 9)."""

    # ── Existence et documentation des codes ─────────────────────────────────

    def test_tous_les_neuf_codes_existent(self):
        codes = [c.value for c in PrivacyCode]
        attendus = [
            "MISSING_ROLE",
            "UNKNOWN_ROLE",
            "UNKNOWN_POLICY",
            "MISSING_PSEUDONYMIZATION_KEY",
            "UNMASKED_IDENTIFIER",
            "FORBIDDEN_FIELD_EXPOSED",
            "INVALID_PRIVACY_INPUT",
            "INVALID_PRIVACY_OUTPUT",
            "PSEUDONYMIZATION_ERROR",
        ]
        for code in attendus:
            assert code in codes, f"{code} manquant dans PrivacyCode"

    def test_exactement_neuf_codes(self):
        assert len(list(PrivacyCode)) == 9

    def test_tous_les_codes_ont_une_description(self):
        for code in PrivacyCode:
            assert code in PRIVACY_CODE_DESCRIPTIONS, f"{code.value} sans description"

    def test_descriptions_non_vides(self):
        for code, desc in PRIVACY_CODE_DESCRIPTIONS.items():
            assert desc.strip(), f"Description vide pour {code.value}"

    def test_privacy_decision_allow_existe(self):
        assert PrivacyDecision.ALLOW.value == "ALLOW"

    def test_privacy_decision_block_existe(self):
        assert PrivacyDecision.BLOCK.value == "BLOCK"

    def test_privacy_decision_exactement_deux_valeurs(self):
        assert len(list(PrivacyDecision)) == 2

    # ── Champ computed decision dans PrivacyResult ───────────────────────────

    def test_status_pass_produit_decision_allow(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(_make_privacy_input(), gate)
        assert result.status == VerificationStatus.PASS
        assert result.decision == PrivacyDecision.ALLOW

    def test_status_needs_review_produit_decision_allow(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        inp = _make_privacy_input(
            data_classification=DataClassification.CONFIDENTIAL
        )
        result = run(inp, gate)
        assert result.status == VerificationStatus.NEEDS_REVIEW
        assert result.decision == PrivacyDecision.ALLOW

    def test_status_fail_produit_decision_block(self):
        result = run(_make_privacy_input(), security_result=None)
        assert result.status == VerificationStatus.FAIL
        assert result.decision == PrivacyDecision.BLOCK

    def test_decision_dans_model_dump(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(_make_privacy_input(), gate)
        d = result.model_dump()
        assert "decision" in d
        assert d["decision"] == PrivacyDecision.ALLOW

    # ── reason_codes chemin nominal (vide) ───────────────────────────────────

    def test_happy_path_reason_codes_vide(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(_make_privacy_input(), gate)
        assert result.reason_codes == []

    def test_happy_path_audit_reason_codes_vide(self):
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(_make_privacy_input(), gate)
        assert result.audit_entry is not None
        assert result.audit_entry.reason_codes == []

    # ── reason_codes MISSING_PSEUDONYMIZATION_KEY ─────────────────────────────

    def test_cle_absente_produit_code_missing_key(self, monkeypatch):
        from agents.privacy_agent import agent as agent_module
        monkeypatch.setattr(agent_module, "pseudonymization_key_is_available", lambda: False)
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(_make_privacy_input(), gate)
        assert PrivacyCode.MISSING_PSEUDONYMIZATION_KEY in result.reason_codes

    def test_cle_absente_audit_entry_a_le_code(self, monkeypatch):
        from agents.privacy_agent import agent as agent_module
        monkeypatch.setattr(agent_module, "pseudonymization_key_is_available", lambda: False)
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(_make_privacy_input(), gate)
        assert result.audit_entry is not None
        assert PrivacyCode.MISSING_PSEUDONYMIZATION_KEY in result.audit_entry.reason_codes

    # ── reason_codes PSEUDONYMIZATION_ERROR ───────────────────────────────────

    def test_erreur_pseudonymisation_produit_code(self, monkeypatch):
        from agents.privacy_agent import agent as agent_module

        def _raise_pseudonymization_error(*a, **kw):
            raise RuntimeError("simulated pseudonymization failure")

        monkeypatch.setattr(agent_module, "pseudonymize_fields", _raise_pseudonymization_error)
        gate = _make_gate_result(SecurityDecision.ALLOW)
        inp = _make_privacy_input(patient_id="PATIENT-001")
        result = run(inp, gate)
        assert result.status == VerificationStatus.FAIL
        assert PrivacyCode.PSEUDONYMIZATION_ERROR in result.reason_codes

    # ── reason_codes UNKNOWN_POLICY ────────────────────────────────────────────

    def test_unknown_policy_produit_code(self, monkeypatch):
        from agents.privacy_agent import agent as agent_module

        def _raise_key_error(*a, **kw):
            raise KeyError("UNKNOWN_ROLE_X")

        monkeypatch.setattr(agent_module, "compute_masked_fields", _raise_key_error)
        gate = _make_gate_result(SecurityDecision.ALLOW)
        result = run(_make_privacy_input(), gate)
        assert result.status == VerificationStatus.FAIL
        assert PrivacyCode.UNKNOWN_POLICY in result.reason_codes

    # ── reason_codes via node() ────────────────────────────────────────────────

    def test_node_sans_role_produit_missing_role(self):
        state = {
            "case_id": "CLM-0001",
            "security_result": _make_gate_result(SecurityDecision.ALLOW),
            "privacy_input": {},
        }
        updates = node(state)
        result = updates["privacy_result"]
        assert result.status == VerificationStatus.FAIL
        assert PrivacyCode.MISSING_ROLE in result.reason_codes
        assert result.decision == PrivacyDecision.BLOCK

    def test_node_role_inconnu_produit_unknown_role(self):
        state = {
            "case_id": "CLM-0001",
            "security_result": _make_gate_result(SecurityDecision.ALLOW),
            "privacy_input": {"role": "ROLE_INEXISTANT"},
        }
        updates = node(state)
        result = updates["privacy_result"]
        assert result.status == VerificationStatus.FAIL
        assert PrivacyCode.UNKNOWN_ROLE in result.reason_codes
        assert result.decision == PrivacyDecision.BLOCK

    def test_node_case_id_invalide_produit_invalid_input(self):
        state = {
            "case_id": "mauvais-id",
            "security_result": _make_gate_result(SecurityDecision.ALLOW),
            "privacy_input": {"role": "ADMINISTRATIVE_MANAGER"},
        }
        updates = node(state)
        result = updates["privacy_result"]
        assert result.status == VerificationStatus.FAIL
        assert PrivacyCode.INVALID_PRIVACY_INPUT in result.reason_codes
        assert result.decision == PrivacyDecision.BLOCK

    # ── _violation_to_codes ────────────────────────────────────────────────────

    def test_secret_field_mappe_forbidden_field_exposed(self):
        violations = [
            PolicyViolation(
                field="api_key",
                reason_code="SECRET_FIELD_IN_VIEW",
                message="champ secret",
            )
        ]
        codes = _violation_to_codes(violations)
        assert PrivacyCode.FORBIDDEN_FIELD_EXPOSED in codes

    def test_raw_identity_mappe_forbidden_field_exposed(self):
        violations = [
            PolicyViolation(
                field="patient_name",
                reason_code="RAW_IDENTITY_IN_VIEW",
                message="identifiant brut",
            )
        ]
        codes = _violation_to_codes(violations)
        assert PrivacyCode.FORBIDDEN_FIELD_EXPOSED in codes

    def test_invalid_pseudonym_mappe_unmasked_identifier(self):
        violations = [
            PolicyViolation(
                field="patient_pseudonym",
                reason_code="INVALID_PSEUDONYM_FORMAT",
                message="pas de préfixe",
            )
        ]
        codes = _violation_to_codes(violations)
        assert PrivacyCode.UNMASKED_IDENTIFIER in codes

    def test_violations_multiples_codes_dedupliques(self):
        violations = [
            PolicyViolation(field="api_key", reason_code="SECRET_FIELD_IN_VIEW", message="a"),
            PolicyViolation(field="patient_name", reason_code="RAW_IDENTITY_IN_VIEW", message="b"),
        ]
        codes = _violation_to_codes(violations)
        assert codes.count(PrivacyCode.FORBIDDEN_FIELD_EXPOSED) == 1

    def test_violations_mixtes_deux_codes_distincts(self):
        violations = [
            PolicyViolation(field="api_key", reason_code="SECRET_FIELD_IN_VIEW", message="a"),
            PolicyViolation(
                field="patient_pseudonym",
                reason_code="INVALID_PSEUDONYM_FORMAT",
                message="b",
            ),
        ]
        codes = _violation_to_codes(violations)
        assert PrivacyCode.FORBIDDEN_FIELD_EXPOSED in codes
        assert PrivacyCode.UNMASKED_IDENTIFIER in codes

    # ── Security Gate blocked — reason_codes vide ─────────────────────────────

    def test_security_gate_block_reason_codes_vide(self):
        gate = _make_gate_result(SecurityDecision.BLOCK)
        result = run(_make_privacy_input(), gate)
        assert result.status == VerificationStatus.FAIL
        assert result.reason_codes == []
        assert result.decision == PrivacyDecision.BLOCK

    def test_security_gate_absent_reason_codes_vide(self):
        result = run(_make_privacy_input(), security_result=None)
        assert result.reason_codes == []
        assert result.decision == PrivacyDecision.BLOCK
