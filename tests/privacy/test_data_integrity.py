"""Tests d'intégrité des données du Privacy Agent — Étape 10.

Données sources : dict structuré représentant un ClaimSubmission.
Le branchement sur les données OCR brutes est réservé à l'étape 7.

Organisation :
  TestRoleIsolation         — chaque rôle reçoit uniquement sa vue
  TestPseudonymConsistency  — même patient → même pseudonyme
  TestDataLeakagePrevention — données sensibles absentes des vues non autorisées
  TestMaskingFunctions      — adresse, téléphone et e-mail correctement masqués
  TestErrorHandling         — rôle inconnu, politique manquante, clé absente, fuite
  TestSourceImmutability    — objet source strictement inchangé après run()
  TestJSONSerializability   — toutes les sorties sérialisables en JSON
  TestNoSecretInOutputs     — aucun secret dans les traces d'audit
"""
from __future__ import annotations

import copy
import json
from datetime import UTC, datetime


from agents.privacy_agent.agent import run
from agents.privacy_agent.agent import node
from agents.privacy_agent.schemas import PrivacyCode, PrivacyDecision, PrivacyInput, ReaderRole
from schemas.domain import SecurityDecision, VerificationStatus
from schemas.results import SecurityAuditEntry, SecurityGateResult
from security.access_policies import verify_view_privacy
from tools.pseudonymize import mask_email, mask_name, mask_phone, mask_postal_address

# ── Données structurées représentatives d'un ClaimSubmission ──────────────────

_CASE_ID = "CLM-0001"
_PATIENT_ID = "PATIENT-SYNTH-0001"
_PATIENT_ID_2 = "PATIENT-SYNTH-0002"
_PATIENT_NAME = "Jean Dupont"
_PROVIDER_ID = "PROV-SYNTH-0099"

# SHA-256 fictifs valides (64 caractères hexadécimaux)
_SHA_FACTURE = "a" * 64
_SHA_ORDONNANCE = "b" * 64

_CLAIM_DATA: dict = {
    # Identification personnelle — jamais exposée en clair dans une vue minimisée
    "patient_id": _PATIENT_ID,
    "patient_name": _PATIENT_NAME,
    "birth_date": "1985-03-22",
    "gender": "M",
    # Financier et facturation
    "total_billed": "1250.00",
    "amount_requested": "1000.00",
    "patient_share": "250.00",
    "coverage_rate": "0.80",
    "payer_name": "Mutuelle TOPAZ",
    "invoice_number": "INV-2026-0042",
    "prescription_number": "ORD-2026-0017",
    "claim_reference": _CASE_ID,
    # Médical et clinique
    "procedures": ["Consultation généraliste", "Radiographie thoracique"],
    "prescription_names": ["Amoxicilline 500mg", "Paracétamol 1000mg"],
    "diagnosis_codes": ["J18.9", "R05"],
    "encounter_class": "ambulatory",
    "provider_id": _PROVIDER_ID,
    "organization_id": "ORG-0042",
    # Temporel et logistique
    "service_date": "2026-06-01",
    "submitted_at": "2026-06-05T10:30:00",
    # Structurel — nécessaire pour la vue administrative
    "dossier_status": "ACCEPTED",
    "present_documents": [
        "facture_CLM-0001.pdf",
        "ordonnance_CLM-0001.pdf",
        "demande_CLM-0001.pdf",
    ],
    "missing_documents": [],
    # Hashes documentaires — nécessaires pour la vue antifraude
    "document_hashes": {
        "facture": _SHA_FACTURE,
        "ordonnance": _SHA_ORDONNANCE,
    },
    # Trace de traitement — nécessaire pour la vue audit
    "actor": "system",
    "actor_role": "ADMINISTRATIVE_MANAGER",
    "action": "view_request",
    "timestamp": "2026-06-25T09:00:00",
    "policy_version": "1.1.0",
    "outcome": "PASS",
    "reason_codes": [],
}


# ── Helpers de tests ──────────────────────────────────────────────────────────


def _gate_allow(case_id: str = _CASE_ID) -> SecurityGateResult:
    return SecurityGateResult(
        claim_id=case_id,
        decision=SecurityDecision.ALLOW,
        findings=[],
        reason_codes=[],
        applied_policy="default",
        policy_version="1.1.0",
        evaluated_at=datetime.now(UTC),
        next_allowed_action="continue_pipeline",
        audit_entry=SecurityAuditEntry(
            claim_id=case_id,
            actor="security_gate_agent",
            outcome="ALLOW",
            decision=SecurityDecision.ALLOW,
            policy_applied="default",
            policy_version="1.1.0",
        ),
        reasons=["Évaluation réussie — aucune anomalie détectée"],
    )


def _gate_block(case_id: str = _CASE_ID) -> SecurityGateResult:
    return SecurityGateResult(
        claim_id=case_id,
        decision=SecurityDecision.BLOCK,
        findings=[],
        reason_codes=[],
        applied_policy="default",
        policy_version="1.1.0",
        evaluated_at=datetime.now(UTC),
        next_allowed_action="terminate_pipeline",
        audit_entry=None,
        reasons=["Injection de prompt détectée"],
    )


def _make_input(
    role: ReaderRole,
    *,
    case_id: str = _CASE_ID,
    claim_data: dict | None = None,
    **kwargs,
) -> PrivacyInput:
    return PrivacyInput(
        case_id=case_id,
        role=role,
        claim_data=claim_data if claim_data is not None else dict(_CLAIM_DATA),
        **kwargs,
    )


def _run_ok(role: ReaderRole, **kwargs) -> object:
    """Exécute le pipeline complet et vérifie que le résultat n'est pas FAIL."""
    inp = _make_input(role, **kwargs)
    result = run(inp, _gate_allow())
    assert result.status != VerificationStatus.FAIL, (
        f"Échec inattendu pour le rôle {role.value} : {result.errors}"
    )
    return result


# ── TestRoleIsolation ─────────────────────────────────────────────────────────


class TestRoleIsolation:
    """Chaque rôle reçoit uniquement la vue qui lui est destinée."""

    def test_gestionnaire_recoit_vue_administrative(self):
        result = _run_ok(ReaderRole.ADMINISTRATIVE_MANAGER)
        view = result.view
        assert view is not None
        assert "claim_id" in view
        assert "patient_pseudonym" in view
        assert "dossier_status" in view
        assert "total_billed" in view

    def test_medecin_recoit_vue_medicale(self):
        result = _run_ok(ReaderRole.MEDICAL_REVIEWER)
        view = result.view
        assert view is not None
        assert "patient_pseudonym" in view
        assert "procedures" in view
        assert "diagnosis_codes" in view

    def test_analyste_recoit_vue_antifraude(self):
        result = _run_ok(ReaderRole.FRAUD_ANALYST)
        view = result.view
        assert view is not None
        assert "patient_pseudonym" in view
        assert "document_hashes" in view
        assert "total_billed" in view

    def test_auditeur_recoit_vue_minimisee(self):
        result = _run_ok(ReaderRole.AUDITOR)
        view = result.view
        assert view is not None
        assert "claim_id" in view
        assert "actor" in view
        assert "outcome" in view

    def test_diagnostics_absents_vue_administrative(self):
        view = _run_ok(ReaderRole.ADMINISTRATIVE_MANAGER).view
        assert "diagnosis_codes" not in view
        assert "procedures" not in view
        assert "encounter_class" not in view

    def test_coordonnees_financieres_absentes_vue_medicale(self):
        view = _run_ok(ReaderRole.MEDICAL_REVIEWER).view
        assert "total_billed" not in view
        assert "payer_name" not in view
        assert "coverage_rate" not in view
        assert "amount_requested" not in view

    def test_donnees_medicales_absentes_vue_antifraude(self):
        view = _run_ok(ReaderRole.FRAUD_ANALYST).view
        assert "diagnosis_codes" not in view
        assert "procedures" not in view
        assert "encounter_class" not in view

    def test_donnees_personnelles_absentes_vue_audit(self):
        view = _run_ok(ReaderRole.AUDITOR).view
        assert "patient_pseudonym" not in view
        assert "total_billed" not in view
        assert "diagnosis_codes" not in view
        assert "procedures" not in view

    def test_texte_ocr_brut_absent_vue_audit(self):
        """Un champ raw_ocr_text dans claim_data ne doit pas fuir dans la vue audit."""
        data = dict(_CLAIM_DATA)
        data["raw_ocr_text"] = "Patient Jean Dupont — DOB 1985-03-22 — INV-2026-0042"
        result = _run_ok(ReaderRole.AUDITOR, claim_data=data)
        view_json = json.dumps(result.view, default=str)
        assert "raw_ocr_text" not in view_json
        assert _PATIENT_NAME not in view_json

    def test_texte_ocr_brut_absent_vue_administrative(self):
        data = dict(_CLAIM_DATA)
        data["raw_ocr_text"] = "Texte OCR brut confidentiel — INV-2026-0042"
        result = _run_ok(ReaderRole.ADMINISTRATIVE_MANAGER, claim_data=data)
        view_json = json.dumps(result.view, default=str)
        assert "raw_ocr_text" not in view_json

    def test_view_role_correspond_au_role_demande(self):
        for role in ReaderRole:
            result = _run_ok(role)
            assert result.view_role == role.value

    def test_vue_construite_pour_chaque_role(self):
        for role in ReaderRole:
            result = _run_ok(role)
            assert result.view is not None, f"Vue None pour le rôle {role.value}"


# ── TestPseudonymConsistency ──────────────────────────────────────────────────


class TestPseudonymConsistency:
    """Même patient → même pseudonyme ; patients distincts → pseudonymes distincts."""

    def test_meme_patient_meme_pseudonyme_gestionnaire(self):
        r1 = _run_ok(ReaderRole.ADMINISTRATIVE_MANAGER)
        r2 = _run_ok(ReaderRole.ADMINISTRATIVE_MANAGER)
        assert r1.view["patient_pseudonym"] == r2.view["patient_pseudonym"]

    def test_meme_patient_meme_pseudonyme_medecin(self):
        r1 = _run_ok(ReaderRole.MEDICAL_REVIEWER)
        r2 = _run_ok(ReaderRole.MEDICAL_REVIEWER)
        assert r1.view["patient_pseudonym"] == r2.view["patient_pseudonym"]

    def test_pseudonyme_coherent_entre_roles_admin_et_medical(self):
        r_admin = _run_ok(ReaderRole.ADMINISTRATIVE_MANAGER)
        r_med = _run_ok(ReaderRole.MEDICAL_REVIEWER)
        assert r_admin.view["patient_pseudonym"] == r_med.view["patient_pseudonym"]

    def test_pseudonyme_coherent_entre_roles_admin_et_fraude(self):
        r_admin = _run_ok(ReaderRole.ADMINISTRATIVE_MANAGER)
        r_fraud = _run_ok(ReaderRole.FRAUD_ANALYST)
        assert r_admin.view["patient_pseudonym"] == r_fraud.view["patient_pseudonym"]

    def test_deux_patients_differents_pseudonymes_differents(self):
        data2 = dict(_CLAIM_DATA)
        data2["patient_id"] = _PATIENT_ID_2
        r1 = _run_ok(ReaderRole.ADMINISTRATIVE_MANAGER)
        r2 = _run_ok(ReaderRole.ADMINISTRATIVE_MANAGER, claim_data=data2)
        assert r1.view["patient_pseudonym"] != r2.view["patient_pseudonym"]

    def test_pseudonyme_patient_format_pat_vue_admin(self):
        result = _run_ok(ReaderRole.ADMINISTRATIVE_MANAGER)
        assert result.view["patient_pseudonym"].startswith("PAT-")

    def test_pseudonyme_patient_format_pat_vue_medicale(self):
        result = _run_ok(ReaderRole.MEDICAL_REVIEWER)
        assert result.view["patient_pseudonym"].startswith("PAT-")

    def test_pseudonyme_patient_format_pat_vue_fraude(self):
        result = _run_ok(ReaderRole.FRAUD_ANALYST)
        assert result.view["patient_pseudonym"].startswith("PAT-")

    def test_pseudonyme_prestataire_format_prv_vue_medicale(self):
        result = _run_ok(ReaderRole.MEDICAL_REVIEWER)
        prv = result.view.get("provider_pseudonym")
        if prv is not None:
            assert prv.startswith("PRV-")

    def test_pseudonyme_prestataire_format_prv_vue_fraude(self):
        result = _run_ok(ReaderRole.FRAUD_ANALYST)
        prv = result.view.get("provider_reference")
        if prv is not None:
            assert prv.startswith("PRV-")

    def test_pseudonyme_format_regex_pat(self):
        import re
        result = _run_ok(ReaderRole.ADMINISTRATIVE_MANAGER)
        pat = result.view["patient_pseudonym"]
        assert re.match(r"^PAT-[0-9A-F]{12}$", pat), f"Format invalide : {pat}"

    def test_patient_sans_id_produit_pat_inconnu(self):
        data = dict(_CLAIM_DATA)
        data.pop("patient_id", None)
        result = _run_ok(ReaderRole.ADMINISTRATIVE_MANAGER, claim_data=data)
        assert result.view["patient_pseudonym"] == "PAT-INCONNU"


# ── TestDataLeakagePrevention ─────────────────────────────────────────────────


class TestDataLeakagePrevention:
    """Données sensibles absentes des vues non autorisées."""

    def _view_json(self, role: ReaderRole) -> str:
        return json.dumps(_run_ok(role).view, default=str)

    def test_nom_complet_absent_vue_administrative(self):
        assert _PATIENT_NAME not in self._view_json(ReaderRole.ADMINISTRATIVE_MANAGER)

    def test_nom_complet_absent_vue_medicale(self):
        assert _PATIENT_NAME not in self._view_json(ReaderRole.MEDICAL_REVIEWER)

    def test_nom_complet_absent_vue_antifraude(self):
        assert _PATIENT_NAME not in self._view_json(ReaderRole.FRAUD_ANALYST)

    def test_nom_complet_absent_vue_audit(self):
        assert _PATIENT_NAME not in self._view_json(ReaderRole.AUDITOR)

    def test_patient_id_brut_absent_vue_administrative(self):
        assert _PATIENT_ID not in self._view_json(ReaderRole.ADMINISTRATIVE_MANAGER)

    def test_patient_id_brut_absent_vue_medicale(self):
        assert _PATIENT_ID not in self._view_json(ReaderRole.MEDICAL_REVIEWER)

    def test_patient_id_brut_absent_comme_cle_de_vue_admin(self):
        view = _run_ok(ReaderRole.ADMINISTRATIVE_MANAGER).view
        assert "patient_name" not in view
        assert "patient_id" not in view

    def test_patient_id_brut_absent_comme_cle_de_vue_medicale(self):
        view = _run_ok(ReaderRole.MEDICAL_REVIEWER).view
        assert "patient_name" not in view
        assert "patient_id" not in view

    def test_donnees_supplementaires_non_exposees(self):
        """Des clés supplémentaires dans claim_data ne doivent pas fuir dans la vue."""
        data = dict(_CLAIM_DATA)
        data["email"] = "jean.dupont@test.fr"
        data["phone"] = "+216 22 123 456"
        data["home_address"] = "1 Rue de la Paix, 75001 Paris"
        result = _run_ok(ReaderRole.ADMINISTRATIVE_MANAGER, claim_data=data)
        view_json = json.dumps(result.view, default=str)
        assert "jean.dupont@test.fr" not in view_json
        assert "123 456" not in view_json
        assert "Rue de la Paix" not in view_json

    def test_fuite_identifiant_brut_detectee_par_verify_view_privacy(self):
        """Un identifiant brut dans une vue doit être détecté par verify_view_privacy."""
        contaminated_view = {
            "patient_name": _PATIENT_NAME,
            "claim_id": _CASE_ID,
        }
        violations = verify_view_privacy(contaminated_view)
        assert any(v.reason_code == "RAW_IDENTITY_IN_VIEW" for v in violations)
        assert any(v.field == "patient_name" for v in violations)

    def test_champ_secret_detecte_par_verify_view_privacy(self):
        """Un champ secret dans une vue doit être détecté par verify_view_privacy."""
        contaminated_view = {
            "patient_pseudonym": "PAT-AABBCCDDEEFF",
            "api_key": "sk-secret-12345",
        }
        violations = verify_view_privacy(contaminated_view)
        assert any(v.reason_code == "SECRET_FIELD_IN_VIEW" for v in violations)
        assert any(v.field == "api_key" for v in violations)

    def test_pseudonyme_sans_prefixe_detecte(self):
        """Un patient_pseudonym sans préfixe PAT- doit être détecté."""
        contaminated_view = {
            "patient_pseudonym": "IDENTIFIANT-BRUT-001",
        }
        violations = verify_view_privacy(contaminated_view)
        assert any(v.reason_code == "INVALID_PSEUDONYM_FORMAT" for v in violations)

    def test_vue_conforme_sans_violations(self):
        """La vue produite par l'agent ne doit produire aucune violation."""
        for role in ReaderRole:
            result = _run_ok(role)
            violations = verify_view_privacy(result.view)
            assert violations == [], (
                f"Violations inattendues pour {role.value} : "
                f"{[(v.field, v.reason_code) for v in violations]}"
            )


# ── TestMaskingFunctions ──────────────────────────────────────────────────────


class TestMaskingFunctions:
    """Adresse, téléphone et e-mail correctement masqués par les utilitaires."""

    def test_masquage_email_basique(self):
        result = mask_email("patient@example.com")
        assert "@example.com" in result
        assert "patient" not in result
        assert result.startswith("p")

    def test_masquage_email_ne_revele_pas_partie_locale(self):
        result = mask_email("jean.dupont@hopital.fr")
        assert "jean.dupont" not in result
        assert "@hopital.fr" in result

    def test_masquage_telephone_conserve_4_derniers_chiffres(self):
        result = mask_phone("+216 22 123 456")
        assert result.endswith("3456")
        assert "****" in result

    def test_masquage_telephone_masque_prefixe(self):
        result = mask_phone("06 12 34 56 78")
        assert result.endswith("5678")
        assert "06" not in result

    def test_masquage_adresse_masque_la_rue(self):
        result = mask_postal_address("1 Rue de la Paix, 75001 Paris")
        assert result.startswith("***")
        assert "Rue de la Paix" not in result
        assert "75001 Paris" in result

    def test_masquage_nom_ne_revele_pas_le_nom_complet(self):
        result = mask_name(_PATIENT_NAME)
        assert _PATIENT_NAME not in result
        assert result.startswith("J")

    def test_donnees_contact_non_exposees_dans_vue(self):
        """Des coordonnées dans claim_data ne se retrouvent pas dans la vue minimisée."""
        data = dict(_CLAIM_DATA)
        data["email"] = "jean.dupont@test.fr"
        data["phone"] = "+33 6 12 34 56 78"
        result = _run_ok(ReaderRole.ADMINISTRATIVE_MANAGER, claim_data=data)
        view_json = json.dumps(result.view, default=str)
        assert "jean.dupont@test.fr" not in view_json
        assert "06 12 34 56 78" not in view_json


# ── TestErrorHandling ─────────────────────────────────────────────────────────


class TestErrorHandling:
    """Gestion des cas d'erreur — rôle inconnu, politique manquante, clé absente, fuite."""

    def test_role_inconnu_bloque(self):
        state = {
            "case_id": _CASE_ID,
            "security_result": _gate_allow(),
            "privacy_input": {"role": "SUPER_ADMIN"},
        }
        updates = node(state)
        result = updates["privacy_result"]
        assert result.status == VerificationStatus.FAIL
        assert result.decision == PrivacyDecision.BLOCK
        assert PrivacyCode.UNKNOWN_ROLE in result.reason_codes

    def test_role_absent_bloque(self):
        state = {
            "case_id": _CASE_ID,
            "security_result": _gate_allow(),
            "privacy_input": {},
        }
        updates = node(state)
        result = updates["privacy_result"]
        assert result.status == VerificationStatus.FAIL
        assert result.decision == PrivacyDecision.BLOCK
        assert PrivacyCode.MISSING_ROLE in result.reason_codes

    def test_politique_inconnue_bloquee(self, monkeypatch):
        from agents.privacy_agent import agent as agent_module

        def _raise_key_error(*a, **kw):
            raise KeyError("role_sans_politique")

        monkeypatch.setattr(agent_module, "compute_masked_fields", _raise_key_error)
        result = run(_make_input(ReaderRole.ADMINISTRATIVE_MANAGER), _gate_allow())
        assert result.status == VerificationStatus.FAIL
        assert result.decision == PrivacyDecision.BLOCK
        assert PrivacyCode.UNKNOWN_POLICY in result.reason_codes

    def test_cle_absente_erreur_controlee(self, monkeypatch):
        from agents.privacy_agent import agent as agent_module
        monkeypatch.setattr(agent_module, "pseudonymization_key_is_available", lambda: False)
        result = run(_make_input(ReaderRole.MEDICAL_REVIEWER), _gate_allow())
        assert result.status == VerificationStatus.FAIL
        assert result.decision == PrivacyDecision.BLOCK
        assert PrivacyCode.MISSING_PSEUDONYMIZATION_KEY in result.reason_codes
        assert result.audit_entry is not None
        assert result.errors

    def test_fuite_identifiant_bloquee_par_agent(self, monkeypatch):
        """Un builder de vue qui retourne un identifiant brut doit être bloqué."""
        from agents.privacy_agent import agent as agent_module

        def _bad_view(*a, **kw):
            return {"patient_name": _PATIENT_NAME, "claim_id": _CASE_ID}

        monkeypatch.setattr(agent_module, "build_view", _bad_view)
        result = run(_make_input(ReaderRole.ADMINISTRATIVE_MANAGER), _gate_allow())
        assert result.status == VerificationStatus.FAIL
        assert result.decision == PrivacyDecision.BLOCK
        assert any(
            c in result.reason_codes
            for c in [PrivacyCode.FORBIDDEN_FIELD_EXPOSED, PrivacyCode.UNMASKED_IDENTIFIER]
        )

    def test_security_gate_absent_bloque(self):
        inp = _make_input(ReaderRole.ADMINISTRATIVE_MANAGER)
        result = run(inp, security_result=None)
        assert result.status == VerificationStatus.FAIL
        assert result.decision == PrivacyDecision.BLOCK

    def test_security_gate_block_bloque(self):
        inp = _make_input(ReaderRole.ADMINISTRATIVE_MANAGER)
        result = run(inp, _gate_block())
        assert result.status == VerificationStatus.FAIL
        assert result.decision == PrivacyDecision.BLOCK

    def test_erreur_pseudonymisation_bloquee(self, monkeypatch):
        from agents.privacy_agent import agent as agent_module

        def _raise_error(*a, **kw):
            raise RuntimeError("erreur injectée de pseudonymisation")

        monkeypatch.setattr(agent_module, "pseudonymize_fields", _raise_error)
        inp = _make_input(ReaderRole.ADMINISTRATIVE_MANAGER, patient_id="PAT-TEST-001")
        result = run(inp, _gate_allow())
        assert result.status == VerificationStatus.FAIL
        assert PrivacyCode.PSEUDONYMIZATION_ERROR in result.reason_codes


# ── TestSourceImmutability ────────────────────────────────────────────────────


class TestSourceImmutability:
    """L'objet source reste strictement inchangé après run()."""

    def test_claim_data_inchange_vue_administrative(self):
        inp = _make_input(ReaderRole.ADMINISTRATIVE_MANAGER)
        snapshot = copy.deepcopy(inp.claim_data)
        run(inp, _gate_allow())
        assert inp.claim_data == snapshot

    def test_claim_data_inchange_vue_medicale(self):
        inp = _make_input(ReaderRole.MEDICAL_REVIEWER)
        snapshot = copy.deepcopy(inp.claim_data)
        run(inp, _gate_allow())
        assert inp.claim_data == snapshot

    def test_claim_data_inchange_vue_antifraude(self):
        inp = _make_input(ReaderRole.FRAUD_ANALYST)
        snapshot = copy.deepcopy(inp.claim_data)
        run(inp, _gate_allow())
        assert inp.claim_data == snapshot

    def test_claim_data_cle_patient_id_non_mutee(self):
        data = dict(_CLAIM_DATA)
        inp = _make_input(ReaderRole.ADMINISTRATIVE_MANAGER, claim_data=data)
        patient_id_avant = inp.claim_data.get("patient_id")
        run(inp, _gate_allow())
        assert inp.claim_data.get("patient_id") == patient_id_avant

    def test_claim_data_cle_patient_name_non_supprimee(self):
        data = dict(_CLAIM_DATA)
        inp = _make_input(ReaderRole.AUDITOR, claim_data=data)
        run(inp, _gate_allow())
        assert inp.claim_data.get("patient_name") == _PATIENT_NAME

    def test_privacy_input_role_inchange(self):
        inp = _make_input(ReaderRole.MEDICAL_REVIEWER)
        original_role = inp.role
        run(inp, _gate_allow())
        assert inp.role == original_role

    def test_privacy_input_case_id_inchange(self):
        inp = _make_input(ReaderRole.FRAUD_ANALYST)
        original_case_id = inp.case_id
        run(inp, _gate_allow())
        assert inp.case_id == original_case_id

    def test_claim_data_hashes_non_mutes(self):
        data = dict(_CLAIM_DATA)
        data["document_hashes"] = {"facture": _SHA_FACTURE}
        inp = _make_input(ReaderRole.FRAUD_ANALYST, claim_data=data)
        hashes_avant = dict(inp.claim_data["document_hashes"])
        run(inp, _gate_allow())
        assert inp.claim_data["document_hashes"] == hashes_avant


# ── TestJSONSerializability ───────────────────────────────────────────────────


class TestJSONSerializability:
    """Toutes les sorties sont sérialisables en JSON."""

    def _serialize(self, result) -> dict:
        dumped = result.model_dump()
        text = json.dumps(dumped, default=str)
        return json.loads(text)

    def test_result_admin_serialisable_json(self):
        result = _run_ok(ReaderRole.ADMINISTRATIVE_MANAGER)
        parsed = self._serialize(result)
        assert "status" in parsed
        assert "decision" in parsed
        assert "view" in parsed

    def test_result_medical_serialisable_json(self):
        result = _run_ok(ReaderRole.MEDICAL_REVIEWER)
        parsed = self._serialize(result)
        assert "view" in parsed

    def test_result_fraude_serialisable_json(self):
        result = _run_ok(ReaderRole.FRAUD_ANALYST)
        parsed = self._serialize(result)
        assert "view" in parsed

    def test_result_audit_serialisable_json(self):
        result = _run_ok(ReaderRole.AUDITOR)
        parsed = self._serialize(result)
        assert "view" in parsed

    def test_result_fail_serialisable_json(self):
        result = run(_make_input(ReaderRole.ADMINISTRATIVE_MANAGER), security_result=None)
        parsed = self._serialize(result)
        assert parsed["status"] == "FAIL"
        assert parsed["decision"] == "BLOCK"

    def test_audit_entry_serialisable_json(self):
        result = _run_ok(ReaderRole.ADMINISTRATIVE_MANAGER)
        assert result.audit_entry is not None
        text = json.dumps(result.audit_entry.model_dump(), default=str)
        parsed = json.loads(text)
        assert "role" in parsed
        assert "outcome" in parsed

    def test_decision_incluse_dans_serialisation(self):
        result = _run_ok(ReaderRole.ADMINISTRATIVE_MANAGER)
        parsed = self._serialize(result)
        assert parsed["decision"] == "ALLOW"

    def test_reason_codes_serialisables_chemin_nominal(self):
        result = _run_ok(ReaderRole.ADMINISTRATIVE_MANAGER)
        parsed = self._serialize(result)
        assert "reason_codes" in parsed
        assert parsed["reason_codes"] == []

    def test_reason_codes_serialisables_chemin_fail(self, monkeypatch):
        from agents.privacy_agent import agent as agent_module
        monkeypatch.setattr(agent_module, "pseudonymization_key_is_available", lambda: False)
        result = run(_make_input(ReaderRole.ADMINISTRATIVE_MANAGER), _gate_allow())
        parsed = self._serialize(result)
        assert "reason_codes" in parsed
        assert "MISSING_PSEUDONYMIZATION_KEY" in parsed["reason_codes"]

    def test_vue_incluse_dans_serialisation_pour_tous_les_roles(self):
        for role in ReaderRole:
            result = _run_ok(role)
            parsed = self._serialize(result)
            assert parsed["view"] is not None, f"Vue None sérialisée pour {role.value}"


# ── TestNoSecretInOutputs ─────────────────────────────────────────────────────


class TestNoSecretInOutputs:
    """Aucun secret, chemin absolu ou donnée personnelle dans les traces d'audit."""

    def test_audit_entry_sans_nom_de_patient(self):
        result = _run_ok(ReaderRole.ADMINISTRATIVE_MANAGER)
        ae = result.audit_entry
        assert ae is not None
        audit_json = json.dumps(ae.model_dump(), default=str)
        assert _PATIENT_NAME not in audit_json

    def test_audit_entry_sans_identifiant_brut(self):
        result = _run_ok(ReaderRole.MEDICAL_REVIEWER)
        ae = result.audit_entry
        assert ae is not None
        audit_json = json.dumps(ae.model_dump(), default=str)
        assert _PATIENT_ID not in audit_json

    def test_result_errors_sans_chemin_absolu(self):
        result = run(_make_input(ReaderRole.ADMINISTRATIVE_MANAGER), security_result=None)
        for err in result.errors:
            assert not err.startswith("/"), f"Chemin absolu dans errors : {err}"
            assert "\\" not in err[:3], f"Chemin Windows dans errors : {err}"

    def test_result_reasons_sans_marqueurs_de_secrets(self):
        result = _run_ok(ReaderRole.AUDITOR)
        combined = " ".join(result.reasons).lower()
        assert "api_key" not in combined
        assert "password" not in combined
        assert "bearer " not in combined

    def test_audit_entry_claim_id_valide(self):
        result = _run_ok(ReaderRole.AUDITOR)
        assert result.audit_entry is not None
        assert result.audit_entry.claim_id == _CASE_ID

    def test_audit_entry_role_sans_donnee_personnelle(self):
        result = _run_ok(ReaderRole.FRAUD_ANALYST)
        ae = result.audit_entry
        assert ae is not None
        assert ae.role == ReaderRole.FRAUD_ANALYST.value
        assert _PATIENT_NAME not in ae.role
        assert _PATIENT_ID not in ae.role

    def test_aucun_diagnostic_dans_audit_entry(self):
        result = _run_ok(ReaderRole.ADMINISTRATIVE_MANAGER)
        ae = result.audit_entry
        assert ae is not None
        audit_json = json.dumps(ae.model_dump(), default=str)
        for code in _CLAIM_DATA.get("diagnosis_codes", []):
            assert code not in audit_json

    def test_audit_entry_reason_codes_sont_des_codes_stables(self):
        # Sur le chemin nominal, reason_codes est vide
        result = _run_ok(ReaderRole.ADMINISTRATIVE_MANAGER)
        ae = result.audit_entry
        assert ae is not None
        assert ae.reason_codes == []
