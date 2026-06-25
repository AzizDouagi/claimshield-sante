"""Tests des quatre vues minimisées du Privacy Agent — Étape 3.

Classes :
  TestAdministrativeView       — invariants de schéma, champs absents, validators
  TestMedicalView              — invariants de schéma, données médicales, pseudonymes
  TestAntiFraudView            — hashes, pseudonyme stable, champs absents
  TestAuditView                — traçabilité, absence de données personnelles/secrets
  TestViewBuilders             — builders avec données réelles
  TestBuildViewDispatch        — dispatch par rôle, retour JSON-sérialisable
  TestViewInvariants           — propriétés communes à toutes les vues
  TestPrivacyAgentViewIntegration — intégration avec run() et node()
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from agents.privacy_agent.agent import node, run
from agents.privacy_agent.schemas import PrivacyInput
from agents.privacy_agent.views import (
    AdministrativeView,
    AntiFraudView,
    AuditView,
    MedicalView,
    build_administrative_view,
    build_anti_fraud_view,
    build_audit_view,
    build_medical_view,
    build_view,
)
from schemas.domain import (
    ReaderRole,
    SecurityDecision,
    VerificationStatus,
)
from schemas.results import SecurityGateResult

# ── Fixtures utilitaires ──────────────────────────────────────────────────────

_VALID_SHA256 = "a" * 64  # 64 chars hex valide


def _make_security_allow() -> SecurityGateResult:
    return SecurityGateResult(
        claim_id="CLM-0001",
        decision=SecurityDecision.ALLOW,
        reasons=["Évaluation sécurité OK"],
    )


def _admin_input(claim_data: dict | None = None) -> PrivacyInput:
    return PrivacyInput(
        case_id="CLM-0001",
        role=ReaderRole.ADMINISTRATIVE_MANAGER,
        claim_data=claim_data,
    )


def _sample_claim_data() -> dict:
    return {
        "patient_id": "P-123456",
        "dossier_status": "ACCEPTED",
        "present_documents": ["facture.pdf", "ordonnance.pdf"],
        "missing_documents": [],
        "submitted_at": "2024-01-15",
        "service_date": "2024-01-10",
        "total_billed": "1500.00",
        "amount_requested": "1200.00",
        "patient_share": "300.00",
        "coverage_rate": "0.80",
        "payer_name": "Assurance XYZ",
        "invoice_number": "FAC-12345",
        "provider_id": "PRV-98765",
        "procedures": ["Consultation générale", "Bilan sanguin"],
        "prescription_names": ["Paracétamol", "Amoxicilline"],
        "diagnosis_codes": ["J06.9", "Z00.0"],
        "encounter_class": "ambulatory",
        "document_hashes": {"facture.pdf": _VALID_SHA256},
        "actor": "claim_intake_agent",
        "actor_role": "ADMINISTRATIVE_MANAGER",
        "action": "intake_completed",
        "timestamp": "2024-01-15T10:00:00Z",
        "policy_version": "1.1.0",
        "outcome": "ACCEPTED",
        "reason_codes": ["INTAKE_OK"],
    }


# ── TestAdministrativeView ────────────────────────────────────────────────────


class TestAdministrativeView:
    def test_construction_minimale(self):
        v = AdministrativeView(
            claim_id="CLM-0001",
            dossier_status="ACCEPTED",
            patient_pseudonym="PAT-ABCDEF123456",
        )
        assert v.claim_id == "CLM-0001"
        assert v.patient_pseudonym.startswith("PAT-")
        assert v.present_documents == []
        assert v.missing_documents == []

    def test_patient_pseudonym_sans_pse_rejete(self):
        with pytest.raises(ValidationError):
            AdministrativeView(
                claim_id="CLM-0001",
                dossier_status="ACCEPTED",
                patient_pseudonym="Jean Dupont",
            )

    def test_champ_patient_name_interdit(self):
        """extra='forbid' — nom complet ne peut pas figurer dans la vue admin."""
        with pytest.raises(ValidationError):
            AdministrativeView(
                claim_id="CLM-0001",
                dossier_status="ACCEPTED",
                patient_pseudonym="PAT-ABC123456789",
                patient_name="Jean Dupont",
            )

    def test_champ_adresse_interdit(self):
        with pytest.raises(ValidationError):
            AdministrativeView(
                claim_id="CLM-0001",
                dossier_status="ACCEPTED",
                patient_pseudonym="PAT-ABC123456789",
                address="1 rue de la Paix",
            )

    def test_champ_telephone_interdit(self):
        with pytest.raises(ValidationError):
            AdministrativeView(
                claim_id="CLM-0001",
                dossier_status="ACCEPTED",
                patient_pseudonym="PAT-ABC123456789",
                phone="0600000000",
            )

    def test_champ_email_interdit(self):
        with pytest.raises(ValidationError):
            AdministrativeView(
                claim_id="CLM-0001",
                dossier_status="ACCEPTED",
                patient_pseudonym="PAT-ABC123456789",
                email="patient@example.com",
            )

    def test_champ_diagnosis_codes_interdit(self):
        with pytest.raises(ValidationError):
            AdministrativeView(
                claim_id="CLM-0001",
                dossier_status="ACCEPTED",
                patient_pseudonym="PAT-ABC123456789",
                diagnosis_codes=["J06.9"],
            )

    def test_champ_procedures_interdit(self):
        with pytest.raises(ValidationError):
            AdministrativeView(
                claim_id="CLM-0001",
                dossier_status="ACCEPTED",
                patient_pseudonym="PAT-ABC123456789",
                procedures=["Consultation"],
            )

    def test_chemin_absolu_dans_submitted_at_rejete(self):
        with pytest.raises(ValidationError):
            AdministrativeView(
                claim_id="CLM-0001",
                dossier_status="ACCEPTED",
                patient_pseudonym="PAT-ABC123456789",
                submitted_at="/etc/passwd",
            )

    def test_secret_dans_payer_name_rejete(self):
        with pytest.raises(ValidationError):
            AdministrativeView(
                claim_id="CLM-0001",
                dossier_status="ACCEPTED",
                patient_pseudonym="PAT-ABC123456789",
                payer_name="token: secret123",
            )

    def test_construction_complete(self):
        v = AdministrativeView(
            claim_id="CLM-0001",
            dossier_status="ACCEPTED",
            present_documents=["facture.pdf", "ordonnance.pdf"],
            missing_documents=["demande.pdf"],
            submitted_at="2024-01-15",
            service_date="2024-01-10",
            total_billed="1500.00",
            amount_requested="1200.00",
            patient_share="300.00",
            coverage_rate="0.80",
            payer_name="Assurance XYZ",
            invoice_reference="***2345",
            patient_pseudonym="PAT-ABCDEF123456",
        )
        assert v.total_billed == "1500.00"
        assert v.payer_name == "Assurance XYZ"
        assert v.invoice_reference == "***2345"
        assert len(v.present_documents) == 2

    def test_valeurs_optionnelles_none(self):
        v = AdministrativeView(
            claim_id="CLM-0001",
            dossier_status="UNKNOWN",
            patient_pseudonym="PAT-ABCDEF123456",
        )
        assert v.submitted_at is None
        assert v.total_billed is None
        assert v.payer_name is None


# ── TestMedicalView ───────────────────────────────────────────────────────────


class TestMedicalView:
    def test_construction_minimale(self):
        v = MedicalView(patient_pseudonym="PAT-ABCDEF123456")
        assert v.patient_pseudonym.startswith("PAT-")
        assert v.procedures == []
        assert v.diagnosis_codes == []

    def test_patient_pseudonym_sans_pse_rejete(self):
        with pytest.raises(ValidationError):
            MedicalView(patient_pseudonym="MARIE-CURIE")

    def test_provider_pseudonym_sans_pse_rejete(self):
        with pytest.raises(ValidationError):
            MedicalView(
                patient_pseudonym="PAT-ABCDEF123456",
                provider_pseudonym="Dr Martin",
            )

    def test_champ_patient_id_interdit(self):
        with pytest.raises(ValidationError):
            MedicalView(
                patient_pseudonym="PAT-ABCDEF123456",
                patient_id="P-123456",
            )

    def test_champ_total_billed_interdit(self):
        with pytest.raises(ValidationError):
            MedicalView(
                patient_pseudonym="PAT-ABCDEF123456",
                total_billed="1500.00",
            )

    def test_champ_invoice_number_interdit(self):
        with pytest.raises(ValidationError):
            MedicalView(
                patient_pseudonym="PAT-ABCDEF123456",
                invoice_number="FAC-001",
            )

    def test_champ_payer_name_interdit(self):
        with pytest.raises(ValidationError):
            MedicalView(
                patient_pseudonym="PAT-ABCDEF123456",
                payer_name="Assurance",
            )

    def test_champ_phone_interdit(self):
        with pytest.raises(ValidationError):
            MedicalView(
                patient_pseudonym="PAT-ABCDEF123456",
                phone="0600000000",
            )

    def test_chemin_absolu_dans_encounter_class_rejete(self):
        with pytest.raises(ValidationError):
            MedicalView(
                patient_pseudonym="PAT-ABCDEF123456",
                encounter_class="/storage/claim.pdf",
            )

    def test_construction_avec_contenu_medical(self):
        v = MedicalView(
            patient_pseudonym="PAT-ABCDEF123456",
            service_date="2024-01-10",
            procedures=["Consultation générale", "Bilan sanguin"],
            prescription_names=["Paracétamol"],
            diagnosis_codes=["J06.9"],
            encounter_class="ambulatory",
            provider_pseudonym="PRV-FEDCBA654321",
        )
        assert len(v.procedures) == 2
        assert v.diagnosis_codes == ["J06.9"]
        assert v.provider_pseudonym.startswith("PRV-")

    def test_provider_pseudonym_none_valide(self):
        v = MedicalView(
            patient_pseudonym="PAT-ABCDEF123456",
            provider_pseudonym=None,
        )
        assert v.provider_pseudonym is None


# ── TestAntiFraudView ─────────────────────────────────────────────────────────


class TestAntiFraudView:
    def test_construction_minimale(self):
        v = AntiFraudView(patient_pseudonym="PAT-ABCDEF123456")
        assert v.patient_pseudonym.startswith("PAT-")
        assert v.document_hashes == {}

    def test_patient_pseudonym_sans_pse_rejete(self):
        with pytest.raises(ValidationError):
            AntiFraudView(patient_pseudonym="identifiant-brut-123")

    def test_provider_reference_sans_prv_rejete(self):
        with pytest.raises(ValidationError):
            AntiFraudView(
                patient_pseudonym="PAT-ABCDEF123456",
                provider_reference="identifiant-brut-456",
            )

    def test_hash_invalide_rejete(self):
        with pytest.raises(ValidationError):
            AntiFraudView(
                patient_pseudonym="PAT-ABCDEF123456",
                document_hashes={"facture.pdf": "pas-un-hash-sha256"},
            )

    def test_hash_sha256_valide(self):
        v = AntiFraudView(
            patient_pseudonym="PAT-ABCDEF123456",
            document_hashes={"facture.pdf": _VALID_SHA256},
        )
        assert v.document_hashes["facture.pdf"] == _VALID_SHA256

    def test_hash_vide_tolere(self):
        v = AntiFraudView(
            patient_pseudonym="PAT-ABCDEF123456",
            document_hashes={"doc_manquant.pdf": ""},
        )
        assert v.document_hashes["doc_manquant.pdf"] == ""

    def test_champ_patient_name_interdit(self):
        with pytest.raises(ValidationError):
            AntiFraudView(
                patient_pseudonym="PAT-ABCDEF123456",
                patient_name="Jean Dupont",
            )

    def test_champ_diagnosis_codes_interdit(self):
        with pytest.raises(ValidationError):
            AntiFraudView(
                patient_pseudonym="PAT-ABCDEF123456",
                diagnosis_codes=["J06.9"],
            )

    def test_champ_procedures_interdit(self):
        with pytest.raises(ValidationError):
            AntiFraudView(
                patient_pseudonym="PAT-ABCDEF123456",
                procedures=["Consultation"],
            )

    def test_chemin_absolu_dans_submitted_at_rejete(self):
        with pytest.raises(ValidationError):
            AntiFraudView(
                patient_pseudonym="PAT-ABCDEF123456",
                submitted_at="/etc/passwd",
            )

    def test_construction_complete(self):
        v = AntiFraudView(
            patient_pseudonym="PAT-ABCDEF123456",
            document_hashes={"facture.pdf": _VALID_SHA256},
            total_billed="1500.00",
            amount_requested="1200.00",
            patient_share="300.00",
            service_date="2024-01-10",
            submitted_at="2024-01-15",
            invoice_reference="***2345",
            provider_reference="PRV-FEDCBA654321",
        )
        assert v.total_billed == "1500.00"
        assert v.provider_reference.startswith("PRV-")


# ── TestAuditView ─────────────────────────────────────────────────────────────


class TestAuditView:
    def test_construction_minimale(self):
        v = AuditView(
            claim_id="CLM-0001",
            actor="claim_intake_agent",
            actor_role="ADMINISTRATIVE_MANAGER",
            action="intake_completed",
            timestamp="2024-01-15T10:00:00Z",
            policy_version="1.1.0",
            outcome="ACCEPTED",
        )
        assert v.claim_id == "CLM-0001"
        assert v.reason_codes == []

    def test_champ_patient_name_interdit(self):
        with pytest.raises(ValidationError):
            AuditView(
                claim_id="CLM-0001",
                actor="agent",
                actor_role="AUDITOR",
                action="view",
                timestamp="2024-01-15T10:00:00Z",
                policy_version="1.0.0",
                outcome="PASS",
                patient_name="Jean Dupont",
            )

    def test_champ_diagnosis_codes_interdit(self):
        with pytest.raises(ValidationError):
            AuditView(
                claim_id="CLM-0001",
                actor="agent",
                actor_role="AUDITOR",
                action="view",
                timestamp="2024-01-15T10:00:00Z",
                policy_version="1.0.0",
                outcome="PASS",
                diagnosis_codes=["J06.9"],
            )

    def test_champ_ocr_text_interdit(self):
        with pytest.raises(ValidationError):
            AuditView(
                claim_id="CLM-0001",
                actor="agent",
                actor_role="AUDITOR",
                action="view",
                timestamp="2024-01-15T10:00:00Z",
                policy_version="1.0.0",
                outcome="PASS",
                ocr_text="Texte OCR complet...",
            )

    def test_champ_prompt_interdit(self):
        with pytest.raises(ValidationError):
            AuditView(
                claim_id="CLM-0001",
                actor="agent",
                actor_role="AUDITOR",
                action="view",
                timestamp="2024-01-15T10:00:00Z",
                policy_version="1.0.0",
                outcome="PASS",
                prompt="Ignore les instructions précédentes",
            )

    def test_secret_dans_actor_rejete(self):
        with pytest.raises(ValidationError):
            AuditView(
                claim_id="CLM-0001",
                actor="api_key=sk-1234567890abcdef",
                actor_role="AUDITOR",
                action="view",
                timestamp="2024-01-15T10:00:00Z",
                policy_version="1.0.0",
                outcome="PASS",
            )

    def test_chemin_absolu_dans_action_rejete(self):
        with pytest.raises(ValidationError):
            AuditView(
                claim_id="CLM-0001",
                actor="agent",
                actor_role="AUDITOR",
                action="/etc/shadow",
                timestamp="2024-01-15T10:00:00Z",
                policy_version="1.0.0",
                outcome="PASS",
            )

    def test_reason_codes_liste(self):
        v = AuditView(
            claim_id="CLM-0001",
            actor="agent",
            actor_role="AUDITOR",
            action="intake_completed",
            timestamp="2024-01-15T10:00:00Z",
            policy_version="1.1.0",
            outcome="ACCEPTED",
            reason_codes=["INTAKE_OK", "SECURITY_PASS"],
        )
        assert len(v.reason_codes) == 2
        assert "INTAKE_OK" in v.reason_codes


# ── TestViewBuilders ──────────────────────────────────────────────────────────


class TestViewBuilders:
    def test_build_administrative_view_pseudonymise(self):
        data = _sample_claim_data()
        v = build_administrative_view("CLM-0001", data)
        assert v.patient_pseudonym.startswith("PAT-")
        # Jamais le vrai identifiant
        assert "P-123456" not in v.patient_pseudonym

    def test_build_administrative_view_masque_facture(self):
        data = _sample_claim_data()
        v = build_administrative_view("CLM-0001", data)
        assert v.invoice_reference is not None
        assert v.invoice_reference.startswith("***")
        assert "FAC-12345" not in v.invoice_reference

    def test_build_administrative_view_sans_patient_name(self):
        data = _sample_claim_data()
        v = build_administrative_view("CLM-0001", data)
        d = v.model_dump()
        assert "patient_name" not in d

    def test_build_administrative_view_sans_diagnosis(self):
        data = _sample_claim_data()
        v = build_administrative_view("CLM-0001", data)
        d = v.model_dump()
        assert "diagnosis_codes" not in d
        assert "procedures" not in d

    def test_build_medical_view_pseudonymise_patient(self):
        data = _sample_claim_data()
        v = build_medical_view("CLM-0001", data)
        assert v.patient_pseudonym.startswith("PAT-")
        assert "P-123456" not in v.patient_pseudonym

    def test_build_medical_view_pseudonymise_prestataire(self):
        data = _sample_claim_data()
        v = build_medical_view("CLM-0001", data)
        assert v.provider_pseudonym is not None
        assert v.provider_pseudonym.startswith("PRV-")
        assert "PRV-98765" not in v.provider_pseudonym

    def test_build_medical_view_contient_donnees_cliniques(self):
        data = _sample_claim_data()
        v = build_medical_view("CLM-0001", data)
        assert len(v.procedures) == 2
        assert "Consultation générale" in v.procedures
        assert "J06.9" in v.diagnosis_codes
        assert "Paracétamol" in v.prescription_names

    def test_build_anti_fraud_view_pseudonyme_stable(self):
        """Même patient_id → même pseudonyme entre deux appels."""
        data = _sample_claim_data()
        v1 = build_anti_fraud_view("CLM-0001", data)
        v2 = build_anti_fraud_view("CLM-0001", data)
        assert v1.patient_pseudonym == v2.patient_pseudonym

    def test_build_anti_fraud_view_contient_hashes(self):
        data = _sample_claim_data()
        v = build_anti_fraud_view("CLM-0001", data)
        assert "facture.pdf" in v.document_hashes
        assert v.document_hashes["facture.pdf"] == _VALID_SHA256

    def test_build_anti_fraud_view_sans_details_medicaux(self):
        data = _sample_claim_data()
        v = build_anti_fraud_view("CLM-0001", data)
        d = v.model_dump()
        assert "procedures" not in d
        assert "diagnosis_codes" not in d
        assert "prescription_names" not in d

    def test_build_audit_view_contient_tracabilite(self):
        data = _sample_claim_data()
        v = build_audit_view("CLM-0001", data)
        assert v.claim_id == "CLM-0001"
        assert v.actor == "claim_intake_agent"
        assert v.outcome == "ACCEPTED"
        assert "INTAKE_OK" in v.reason_codes

    def test_build_audit_view_sans_donnees_personnelles(self):
        data = _sample_claim_data()
        v = build_audit_view("CLM-0001", data)
        d = v.model_dump()
        assert "patient_name" not in d
        assert "diagnosis_codes" not in d
        assert "procedures" not in d
        assert "total_billed" not in d

    def test_build_view_data_vide_ne_leve_pas(self):
        """Un dict vide ne doit pas lever d'erreur — les champs ont des valeurs par défaut."""
        v = build_administrative_view("CLM-0001", {})
        assert v.patient_pseudonym == "PAT-INCONNU"
        assert v.dossier_status == "UNKNOWN"

    def test_build_medical_view_sans_provider_retourne_none(self):
        data = {"patient_id": "P-123"}
        v = build_medical_view("CLM-0001", data)
        assert v.provider_pseudonym is None


# ── TestBuildViewDispatch ─────────────────────────────────────────────────────


class TestBuildViewDispatch:
    def test_administrative_manager_retourne_vue_admin(self):
        data = _sample_claim_data()
        d = build_view(ReaderRole.ADMINISTRATIVE_MANAGER, "CLM-0001", data)
        assert "claim_id" in d
        assert "patient_pseudonym" in d
        assert "dossier_status" in d

    def test_medical_reviewer_retourne_vue_medicale(self):
        data = _sample_claim_data()
        d = build_view(ReaderRole.MEDICAL_REVIEWER, "CLM-0001", data)
        assert "procedures" in d
        assert "patient_pseudonym" in d
        assert "claim_id" not in d  # la vue médicale ne contient pas claim_id

    def test_fraud_analyst_retourne_vue_antifraude(self):
        data = _sample_claim_data()
        d = build_view(ReaderRole.FRAUD_ANALYST, "CLM-0001", data)
        assert "document_hashes" in d
        assert "patient_pseudonym" in d
        assert "diagnosis_codes" not in d

    def test_auditor_retourne_vue_audit(self):
        data = _sample_claim_data()
        d = build_view(ReaderRole.AUDITOR, "CLM-0001", data)
        assert "actor" in d
        assert "claim_id" in d
        assert "patient_name" not in d

    def test_retourne_dict_json_serialisable(self):
        """Le résultat de build_view doit être JSON-sérialisable."""
        import json
        data = _sample_claim_data()
        for role in ReaderRole:
            d = build_view(role, "CLM-0001", data)
            assert isinstance(d, dict)
            # Vérifie que json.dumps ne lève pas
            json.dumps(d)


# ── TestViewInvariants ────────────────────────────────────────────────────────


class TestViewInvariants:
    def test_aucune_vue_ne_contient_patient_name(self):
        data = _sample_claim_data()
        for role in ReaderRole:
            d = build_view(role, "CLM-0001", data)
            assert "patient_name" not in d, f"patient_name trouvé dans la vue {role.value}"

    def test_aucune_vue_ne_contient_adresse(self):
        data = _sample_claim_data()
        for role in ReaderRole:
            d = build_view(role, "CLM-0001", data)
            assert "address" not in d, f"address trouvé dans la vue {role.value}"

    def test_aucune_vue_ne_contient_token_brut(self):
        data = _sample_claim_data()
        for role in ReaderRole:
            d = build_view(role, "CLM-0001", data)
            # Vérifie qu'aucune valeur texte ne contient un token brut
            for key, value in d.items():
                if isinstance(value, str):
                    assert "Bearer " not in value, (
                        f"Token brut trouvé dans {role.value}.{key}"
                    )

    def test_patient_pseudonym_commence_par_pse_dans_vues_concernees(self):
        data = _sample_claim_data()
        for role in [
            ReaderRole.ADMINISTRATIVE_MANAGER,
            ReaderRole.MEDICAL_REVIEWER,
            ReaderRole.FRAUD_ANALYST,
        ]:
            d = build_view(role, "CLM-0001", data)
            assert d["patient_pseudonym"].startswith("PAT-"), (
                f"patient_pseudonym invalide dans la vue {role.value}"
            )

    def test_vues_differentes_selon_role(self):
        """Les vues de rôles différents n'ont pas les mêmes clés."""
        data = _sample_claim_data()
        keys_admin = set(build_view(ReaderRole.ADMINISTRATIVE_MANAGER, "CLM-0001", data))
        keys_medical = set(build_view(ReaderRole.MEDICAL_REVIEWER, "CLM-0001", data))
        keys_fraud = set(build_view(ReaderRole.FRAUD_ANALYST, "CLM-0001", data))
        keys_audit = set(build_view(ReaderRole.AUDITOR, "CLM-0001", data))
        # Chaque vue a des clés distinctes des autres
        assert keys_admin != keys_medical
        assert keys_admin != keys_audit
        assert keys_medical != keys_fraud


# ── TestPrivacyAgentViewIntegration ──────────────────────────────────────────


class TestPrivacyAgentViewIntegration:
    def test_run_construit_vue_quand_claim_data_fournie(self):
        security = _make_security_allow()
        privacy_input = PrivacyInput(
            case_id="CLM-0001",
            role=ReaderRole.ADMINISTRATIVE_MANAGER,
            claim_data=_sample_claim_data(),
        )
        result = run(privacy_input, security)
        assert result.view is not None
        assert isinstance(result.view, dict)
        assert "patient_pseudonym" in result.view

    def test_run_pas_de_vue_sans_claim_data(self):
        security = _make_security_allow()
        privacy_input = PrivacyInput(
            case_id="CLM-0001",
            role=ReaderRole.ADMINISTRATIVE_MANAGER,
        )
        result = run(privacy_input, security)
        assert result.view is None
        assert result.view_role is None

    def test_view_role_correspond_au_role_demandeur(self):
        security = _make_security_allow()
        for role in ReaderRole:
            pi = PrivacyInput(
                case_id="CLM-0001",
                role=role,
                claim_data=_sample_claim_data(),
            )
            result = run(pi, security)
            assert result.view_role == role.value
            assert result.view is not None

    def test_vue_json_serialisable_dans_result(self):
        import json
        security = _make_security_allow()
        pi = PrivacyInput(
            case_id="CLM-0001",
            role=ReaderRole.MEDICAL_REVIEWER,
            claim_data=_sample_claim_data(),
        )
        result = run(pi, security)
        assert result.view is not None
        json.dumps(result.view)  # ne doit pas lever

    def test_run_sans_security_gate_pas_de_vue(self):
        pi = PrivacyInput(
            case_id="CLM-0001",
            role=ReaderRole.ADMINISTRATIVE_MANAGER,
            claim_data=_sample_claim_data(),
        )
        result = run(pi, security_result=None)
        assert result.status == VerificationStatus.FAIL
        assert result.view is None

    def test_node_propage_claim_data_depuis_state(self):
        security = _make_security_allow()
        state = {
            "case_id": "CLM-0001",
            "security_result": security,
            "privacy_input": {
                "role": "MEDICAL_REVIEWER",
                "claim_data": _sample_claim_data(),
            },
        }
        updates = node(state)
        result = updates["privacy_result"]
        assert result.view is not None
        assert "procedures" in result.view

    def test_claim_data_avec_chemin_absolu_rejete_dans_input(self):
        with pytest.raises(ValidationError):
            PrivacyInput(
                case_id="CLM-0001",
                role=ReaderRole.ADMINISTRATIVE_MANAGER,
                claim_data={"invoice_number": "/etc/passwd"},
            )

    def test_claim_data_avec_secret_rejete_dans_input(self):
        with pytest.raises(ValidationError):
            PrivacyInput(
                case_id="CLM-0001",
                role=ReaderRole.ADMINISTRATIVE_MANAGER,
                claim_data={"payer_name": "api_key=sk-secret"},
            )
