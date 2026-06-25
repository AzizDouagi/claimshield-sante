"""Tests des fonctions de masquage et pseudonymisation (tools/pseudonymize.py)."""
from __future__ import annotations


from tools.pseudonymize import (
    mask_contract_number,
    mask_email,
    mask_field_value,
    mask_name,
    mask_phone,
    mask_postal_address,
    pseudonymize_fields,
    pseudonymize_id,
    pseudonymize_patient_id,
    pseudonymize_provider_id,
    sanitize_recursive,
)


# ── mask_name ─────────────────────────────────────────────────────────────────


class TestMaskName:
    def test_deux_mots(self):
        assert mask_name("Jean Dupont") == "J*** D*****"

    def test_mot_unique(self):
        assert mask_name("Marie") == "M****"

    def test_chaine_vide(self):
        assert mask_name("") == ""

    def test_un_caractere(self):
        assert mask_name("A") == "A"

    def test_trois_mots(self):
        result = mask_name("Jean Paul Martin")
        parts = result.split()
        assert len(parts) == 3
        assert all(p[0].isalpha() and "*" in p for p in parts if len(p) > 1)


# ── mask_email ────────────────────────────────────────────────────────────────


class TestMaskEmail:
    def test_exemple_standard(self):
        assert mask_email("patient@example.com") == "p*****@example.com"

    def test_premiere_lettre_conservee(self):
        result = mask_email("contact@domain.fr")
        assert result.startswith("c")

    def test_domaine_conserve(self):
        result = mask_email("user@mail.org")
        assert result.endswith("@mail.org")

    def test_sans_arobase(self):
        assert mask_email("invalide") == "[EMAIL MASQUÉ]"

    def test_max_5_etoiles(self):
        # local = "averylongname" (13 chars) → 5 étoiles (plafond)
        assert mask_email("averylongname@example.com") == "a*****@example.com"

    def test_local_deux_chars(self):
        # local = "ab" → 1 étoile (min 1)
        assert mask_email("ab@test.fr") == "a*@test.fr"

    def test_arobase_en_debut(self):
        result = mask_email("@domain.com")
        assert result == "@domain.com"


# ── mask_phone ────────────────────────────────────────────────────────────────


class TestMaskPhone:
    def test_exemple_standard(self):
        assert mask_phone("+216 22 123 456") == "********3456"

    def test_format_francais(self):
        assert mask_phone("06 12 34 56 78") == "********5678"

    def test_prefixe_toujours_8_etoiles(self):
        result = mask_phone("+33 1 23 45 67 89")
        assert result.startswith("*" * 8)

    def test_4_derniers_chiffres(self):
        result = mask_phone("+216 22 123 456")
        assert result.endswith("3456")

    def test_sans_separateurs(self):
        assert mask_phone("0612345678") == "********5678"


# ── mask_contract_number ──────────────────────────────────────────────────────


class TestMaskContractNumber:
    def test_exemple_standard(self):
        assert mask_contract_number("POL-2026-874512") == "********4512"

    def test_slash_separateurs(self):
        assert mask_contract_number("CTR/2025/98765") == "********8765"

    def test_prefixe_toujours_8_etoiles(self):
        result = mask_contract_number("ABC-12345")
        assert result.startswith("*" * 8)

    def test_4_derniers_alphanum(self):
        result = mask_contract_number("POL-2026-874512")
        assert result.endswith("4512")

    def test_court(self):
        # Moins de 4 chars alphanumériques → tout ce qui est disponible
        result = mask_contract_number("AB")
        assert result == "********AB"


# ── mask_postal_address ───────────────────────────────────────────────────────


class TestMaskPostalAddress:
    def test_avec_virgule(self):
        result = mask_postal_address("1 Rue de la Paix, 75001 Paris")
        assert "75001 Paris" in result
        assert "Rue de la Paix" not in result
        assert result.startswith("***")

    def test_plusieurs_virgules(self):
        result = mask_postal_address("12 Av. Bourguiba, 1000 Tunis, Tunisie")
        assert result.endswith("Tunisie")

    def test_sans_virgule(self):
        assert mask_postal_address("Paris") == "***"

    def test_chaine_vide(self):
        assert mask_postal_address("") == ""

    def test_masquage_rue(self):
        result = mask_postal_address("5 Impasse des Lilas, 69000 Lyon")
        assert result.startswith("***")


# ── pseudonymize_patient_id ───────────────────────────────────────────────────


class TestPseudonymizePatientId:
    def test_prefixe_pat(self):
        assert pseudonymize_patient_id("PATIENT-0007").startswith("PAT-")

    def test_deterministe(self):
        v1 = pseudonymize_patient_id("PATIENT-0007")
        v2 = pseudonymize_patient_id("PATIENT-0007")
        assert v1 == v2

    def test_differents_ids(self):
        assert pseudonymize_patient_id("P-001") != pseudonymize_patient_id("P-002")

    def test_pas_pse_prefix(self):
        assert not pseudonymize_patient_id("P-001").startswith("PSE-")

    def test_longueur_16(self):
        # "PAT-" (4) + 12 hex = 16 chars
        result = pseudonymize_patient_id("PATIENT-0007")
        assert len(result) == 16

    def test_exemple_documentation(self):
        # L'exemple "PATIENT-0007 → PAT-A8F3C921" est illustratif (préfixe PAT-)
        result = pseudonymize_patient_id("PATIENT-0007")
        assert result.startswith("PAT-")
        assert len(result) == 16


# ── pseudonymize_provider_id ──────────────────────────────────────────────────


class TestPseudonymizeProviderId:
    def test_prefixe_prv(self):
        assert pseudonymize_provider_id("PROV-0001").startswith("PRV-")

    def test_deterministe(self):
        v1 = pseudonymize_provider_id("PROV-0001")
        v2 = pseudonymize_provider_id("PROV-0001")
        assert v1 == v2

    def test_differents_ids(self):
        assert pseudonymize_provider_id("P-001") != pseudonymize_provider_id("P-002")

    def test_domaine_separe_du_patient(self):
        # Même valeur → pseudonyme différent selon le contexte
        assert pseudonymize_patient_id("P-001") != pseudonymize_provider_id("P-001")

    def test_longueur_16(self):
        result = pseudonymize_provider_id("PROV-0001")
        assert len(result) == 16


# ── pseudonymize_id (générique PSE-) ──────────────────────────────────────────


class TestPseudonymizeId:
    def test_prefixe_pse(self):
        assert pseudonymize_id("uuid-1234").startswith("PSE-")

    def test_deterministe(self):
        assert pseudonymize_id("uuid-1234") == pseudonymize_id("uuid-1234")

    def test_differents(self):
        assert pseudonymize_id("uuid-0001") != pseudonymize_id("uuid-0002")

    def test_domaine_distinct_de_patient_et_provider(self):
        assert pseudonymize_id("X") != pseudonymize_patient_id("X")
        assert pseudonymize_id("X") != pseudonymize_provider_id("X")


# ── mask_field_value dispatch ─────────────────────────────────────────────────


class TestMaskFieldValue:
    def test_patient_id(self):
        assert mask_field_value("patient_id", "P-0007").startswith("PAT-")

    def test_provider_id(self):
        assert mask_field_value("provider_id", "PRV-001").startswith("PRV-")

    def test_organization_id(self):
        assert mask_field_value("organization_id", "ORG-99").startswith("PRV-")

    def test_generic_id(self):
        assert mask_field_value("claim_id", "CLM-001").startswith("PSE-")

    def test_name(self):
        result = mask_field_value("patient_name", "Jean Dupont")
        assert result == "J*** D*****"

    def test_nom(self):
        result = mask_field_value("nom_assure", "Marie")
        assert result == "M****"

    def test_email(self):
        result = mask_field_value("email_contact", "user@test.com")
        assert "@test.com" in result
        assert result.startswith("u")

    def test_phone(self):
        result = mask_field_value("phone_number", "+216 22 123 456")
        assert result == "********3456"

    def test_tel(self):
        result = mask_field_value("tel_portable", "0612345678")
        assert result.startswith("*" * 8)

    def test_address(self):
        result = mask_field_value("address", "1 Rue Test, Paris")
        assert "1 Rue Test" not in result

    def test_adresse(self):
        result = mask_field_value("adresse_postale", "5 Av. X, Tunis")
        assert "5 Av. X" not in result

    def test_contract(self):
        result = mask_field_value("contract_number", "POL-2026-874512")
        assert result == "********4512"

    def test_invoice_number(self):
        result = mask_field_value("invoice_number", "INV-2024-001")
        assert result.startswith("***")

    def test_prescription(self):
        result = mask_field_value("prescription_number", "ORD-2024-9876")
        assert result.startswith("***")

    def test_champ_inconnu(self):
        assert mask_field_value("coverage_rate", "0.80") == "[MASQUÉ]"


# ── sanitize_recursive ────────────────────────────────────────────────────────


class TestSanitizeRecursive:
    def test_supprime_cle_sensible_niveau_1(self):
        data = {"patient_name": "Jean", "claim_id": "CLM-001"}
        result = sanitize_recursive(data, {"patient_name"})
        assert "patient_name" not in result
        assert result["claim_id"] == "CLM-001"

    def test_recursif_dict_imbrique(self):
        data = {"outer": {"patient_name": "Jean", "safe": "ok"}}
        result = sanitize_recursive(data, {"patient_name"})
        assert "patient_name" not in result["outer"]
        assert result["outer"]["safe"] == "ok"

    def test_recursif_liste(self):
        data = [{"patient_name": "Jean"}, {"claim_id": "CLM-001"}]
        result = sanitize_recursive(data, {"patient_name"})
        assert isinstance(result, list)
        assert "patient_name" not in result[0]
        assert result[1]["claim_id"] == "CLM-001"

    def test_scalaire_inchange(self):
        assert sanitize_recursive("test", {"patient_name"}) == "test"
        assert sanitize_recursive(42, {"any"}) == 42
        assert sanitize_recursive(None, {"any"}) is None

    def test_cle_absente_inchangee(self):
        data = {"claim_id": "CLM-001"}
        result = sanitize_recursive(data, {"patient_name"})
        assert result == {"claim_id": "CLM-001"}

    def test_imbrication_profonde(self):
        data = {"a": {"b": {"patient_name": "Jean", "c": "ok"}}}
        result = sanitize_recursive(data, {"patient_name"})
        assert "patient_name" not in result["a"]["b"]
        assert result["a"]["b"]["c"] == "ok"

    def test_plusieurs_cles_sensibles(self):
        data = {"patient_name": "Jean", "patient_id": "P-001", "claim_id": "CLM-001"}
        result = sanitize_recursive(data, {"patient_name", "patient_id"})
        assert "patient_name" not in result
        assert "patient_id" not in result
        assert result["claim_id"] == "CLM-001"

    def test_liste_dans_dict_dans_liste(self):
        data = [{"items": [{"patient_name": "Jean", "code": "A01"}]}]
        result = sanitize_recursive(data, {"patient_name"})
        assert "patient_name" not in result[0]["items"][0]
        assert result[0]["items"][0]["code"] == "A01"

    def test_dict_vide(self):
        assert sanitize_recursive({}, {"patient_name"}) == {}

    def test_liste_vide(self):
        assert sanitize_recursive([], {"patient_name"}) == []

    def test_ne_modifie_pas_l_original(self):
        data = {"patient_name": "Jean", "safe": "ok"}
        _ = sanitize_recursive(data, {"patient_name"})
        assert "patient_name" in data  # l'original est intact


# ── pseudonymize_fields ───────────────────────────────────────────────────────


class TestPseudonymizeFields:
    def test_applique_masquage(self):
        fields = {"patient_name": "Jean Dupont", "invoice_number": "INV-001"}
        result = pseudonymize_fields(fields, ["patient_name", "invoice_number"])
        assert result["patient_name"] == "J*** D*****"
        assert result["invoice_number"].startswith("***")

    def test_none_inchange(self):
        result = pseudonymize_fields({"patient_name": None}, ["patient_name"])
        assert result["patient_name"] is None

    def test_non_masques_inchanges(self):
        result = pseudonymize_fields({"payer_name": "Mutuelle XYZ"}, ["patient_name"])
        assert result["payer_name"] == "Mutuelle XYZ"
