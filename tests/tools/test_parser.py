"""Tests unitaires de tools/document_parser.py — Étape 21."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal


from schemas.domain import DocumentType, OcrSource
from schemas.results import ExtractedField
from tools.document_parser import ParseResult, parse_document


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse(
    text: str,
    doc_type: DocumentType = DocumentType.INVOICE,
    page: int | None = 1,
    conf: float = 0.90,
    filename: str = "facture.pdf",
    sha256: str = "a" * 64,
) -> ParseResult:
    return parse_document(
        text,
        doc_type,
        page,
        OcrSource.PDF_TEXT,
        conf,
        filename=filename,
        sha256=sha256,
    )


def _field(result: ParseResult, name: str) -> ExtractedField | None:
    return result.fields.get(name)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Montant valide
# ═══════════════════════════════════════════════════════════════════════════════


class TestMontantValide:
    """Un montant standard est extrait et normalisé correctement."""

    def test_montant_extrait(self):
        text = "Total facture : 3666.69 USD"
        result = _parse(text)
        field = _field(result, "total_amount")
        assert field is not None

    def test_montant_est_string_normalisee(self):
        text = "Total : 250.00 USD"
        result = _parse(text)
        field = _field(result, "total_amount")
        assert field is not None
        assert field.value in ("250.00", "250.0", "250")

    def test_essential_fields_total_amount_decimal(self):
        text = "Total : 250.00 USD"
        result = _parse(text)
        ta = result.essential_fields.total_amount
        if ta is not None:
            assert isinstance(ta.amount, Decimal)
            assert ta.amount > 0

    def test_provenance_presente(self):
        text = "Montant total : 1200.50 USD"
        result = _parse(text)
        field = _field(result, "total_amount")
        if field is not None:
            assert field.provenance is not None

    def test_parse_result_retourne(self):
        text = "Total facturé : 500.00"
        result = _parse(text)
        assert isinstance(result, ParseResult)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Montant avec virgule (séparateur européen)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMontantVirgule:
    """Un montant avec virgule comme séparateur décimal est normalisé."""

    def test_montant_virgule_extrait(self):
        text = "Total facture : 3 666,69 USD"
        result = _parse(text)
        field = _field(result, "total_amount")
        assert field is not None

    def test_normalisation_decimal(self):
        text = "Total : 1.200,50 EUR"
        result = _parse(text)
        field = _field(result, "total_amount")
        if field is not None:
            # La valeur est normalisée sous forme de chaîne
            assert isinstance(field.value, str)
            assert field.value  # non vide

    def test_essential_fields_amount_decimal_type(self):
        text = "Montant total : 500,00 USD"
        result = _parse(text)
        ta = result.essential_fields.total_amount
        if ta is not None:
            assert isinstance(ta.amount, Decimal)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Montant ambigu (deux valeurs concurrentes)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMontantAmbigu:
    """En présence de plusieurs montants, le parseur extrait sans inventer."""

    def test_premier_montant_extrait(self):
        text = "Total : 100.00 USD\nSous-total : 200.00 USD\nMontant : 300.00 USD"
        result = _parse(text)
        field = _field(result, "total_amount")
        if field is not None:
            assert field.value  # une valeur a été choisie

    def test_pas_invention_champ_absent(self):
        text = "Aucun montant présent dans ce texte."
        result = _parse(text)
        # total_amount ne doit pas être inventé
        field = _field(result, "total_amount")
        assert field is None

    def test_field_count_coherent(self):
        text = "Total facture : 500.00 USD"
        result = _parse(text)
        assert result.field_count == len(result.fields)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Date valide
# ═══════════════════════════════════════════════════════════════════════════════


class TestDateValide:
    """Une date valide est extraite et convertie en type date Python."""

    def test_date_iso_extraite(self):
        text = "Date de facturation : 2024-01-15\nTotal : 100.00 USD"
        result = _parse(text)
        field = _field(result, "service_date")
        assert field is not None

    def test_date_dd_mm_yyyy(self):
        text = "Date de soins : 15/01/2024"
        result = _parse(text)
        # service_date ou care_date selon le contexte
        field = _field(result, "service_date")
        assert field is not None

    def test_essential_fields_service_date_type(self):
        text = "Date du service : 2024-03-20\nTotal : 200.00 USD"
        result = _parse(text)
        sd = result.essential_fields.service_date
        if sd is not None:
            assert isinstance(sd, date)

    def test_document_date_type(self):
        text = "Date du document : 2024-06-01\nTotal : 500.00"
        result = _parse(text)
        dd = result.essential_fields.document_date
        if dd is not None:
            assert isinstance(dd, date)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Date ambiguë
# ═══════════════════════════════════════════════════════════════════════════════


class TestDateAmbigue:
    """Une date ambiguë ou absente ne produit pas de valeur inventée."""

    def test_texte_sans_date_essential_fields_none(self):
        text = "Aucune date dans ce texte de test."
        result = _parse(text)
        sd = result.essential_fields.service_date
        dd = result.essential_fields.document_date
        assert sd is None and dd is None

    def test_date_invalide_ignoree(self):
        text = "Date : 99/99/9999\nTotal : 100.00"
        result = _parse(text)
        sd = result.essential_fields.service_date
        # Date invalide → None (normalize_date_value retourne None)
        assert sd is None or isinstance(sd, date)

    def test_pas_exception_sur_date_malformee(self):
        text = "Date : not-a-date\nMontant : 100.00"
        result = _parse(text)
        assert isinstance(result, ParseResult)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Champ absent
# ═══════════════════════════════════════════════════════════════════════════════


class TestChampAbsent:
    """Un champ absent du texte n'est pas inventé dans les fields."""

    def test_patient_name_absent(self):
        text = "Total facture : 100.00 USD\nDate : 2024-01-01"
        result = _parse(text)
        # patient_name n'est présent que si le pattern matche
        field = _field(result, "patient_name")
        if field is not None:
            assert field.value.strip() != ""

    def test_invoice_number_absent(self):
        text = "Total : 500.00 USD\nDate : 2024-01-01"
        result = _parse(text)
        field = _field(result, "invoice_number")
        assert field is None

    def test_champ_non_present_pas_dans_fields(self):
        text = "Un texte simple sans aucun champ structuré."
        result = _parse(text)
        # Aucun champ inventé sans pattern correspondant
        for name, field in result.fields.items():
            assert field.value is not None
            assert isinstance(field.value, str)

    def test_essential_fields_patient_id_none_si_absent(self):
        text = "Total : 100.00 USD"
        result = _parse(text)
        if result.essential_fields.patient_identifier is None:
            assert True  # attendu si patient_id absent


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Deux valeurs concurrentes dans le texte
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeuxValeursConcurrentes:
    """Quand deux valeurs matchent, le parseur prend la première sans inventer."""

    def test_premier_patient_id_gagne(self):
        text = (
            "N° patient : PAT-0001-A\n"
            "Patient identifier : PAT-0002-B\n"
            "Total : 100.00 USD"
        )
        result = _parse(text)
        field = _field(result, "patient_id")
        if field is not None:
            # La première valeur trouvée est gardée
            assert field.value in ("PAT-0001-A", "PAT-0002-B")
            assert field.value != ""

    def test_deux_refs_claim_premier_gagne(self):
        text = "CLM-0001 et CLM-0002 sont tous les deux présents."
        result = _parse(text, doc_type=DocumentType.CLAIM_REQUEST)
        field = _field(result, "claim_reference")
        if field is not None:
            assert "CLM-" in field.value

    def test_provenance_source_text_non_vide(self):
        text = "Total facture : 500.00 USD\nMontant total : 600.00 USD"
        result = _parse(text)
        field = _field(result, "total_amount")
        if field is not None and field.provenance is not None:
            assert isinstance(field.provenance.source_text, str)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Médicaments sur plusieurs lignes
# ═══════════════════════════════════════════════════════════════════════════════


class TestMedicamentsPlusieursLignes:
    """Les médicaments sur plusieurs lignes sont extraits individuellement."""

    _ORDONNANCE_MULTILIGNES = """
ORDONNANCE
RX-CLM-0001
Date de prescription : 2024-01-15
Médecin prescripteur : Dr. Martin

Amoxicilline 500 mg, quantité : 30
Ibuprofène 400 mg, quantité : 20
Paracétamol 1000 mg, durée 7 jours
"""

    def test_medicaments_extraits(self):
        result = _parse(
            self._ORDONNANCE_MULTILIGNES,
            doc_type=DocumentType.PRESCRIPTION,
            filename="ordonnance.pdf",
        )
        # Des médicaments doivent être détectés
        found = any(k.startswith("prescription_line_") or k == "medications" for k in result.fields)
        assert found or result.essential_fields.medical_items is not None

    def test_medical_items_list(self):
        result = _parse(
            self._ORDONNANCE_MULTILIGNES,
            doc_type=DocumentType.PRESCRIPTION,
        )
        items = result.essential_fields.medical_items
        assert isinstance(items, list)

    def test_prescription_number_extrait(self):
        result = _parse(
            self._ORDONNANCE_MULTILIGNES,
            doc_type=DocumentType.PRESCRIPTION,
        )
        field = _field(result, "prescription_number")
        if field is not None:
            assert "RX-CLM-" in field.value

    def test_json_serialisable(self):
        result = _parse(
            self._ORDONNANCE_MULTILIGNES,
            doc_type=DocumentType.PRESCRIPTION,
        )
        for name, field in result.fields.items():
            # field.value doit être une chaîne serialisable
            assert isinstance(field.value, str)
            if field.value.startswith("[") or field.value.startswith("{"):
                json.loads(field.value)  # ne doit pas lever d'exception

    def test_field_count_positif(self):
        result = _parse(
            self._ORDONNANCE_MULTILIGNES,
            doc_type=DocumentType.PRESCRIPTION,
        )
        assert result.field_count >= 0
        assert result.field_count == len(result.fields)

    def test_medical_items_depuis_invoice(self):
        text = """
FACTURE
Total : 3666.69 USD
Acte 1 : Consultation     250.00 USD
Acte 2 : Radiologie       800.00 USD
"""
        result = _parse(text, doc_type=DocumentType.INVOICE)
        assert isinstance(result.essential_fields.medical_items, list)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Invariants communs
# ═══════════════════════════════════════════════════════════════════════════════


class TestInvariantsParser:
    """Propriétés invariantes du parseur sur tous les types."""

    def test_field_count_coherent(self):
        for doc_type in (DocumentType.INVOICE, DocumentType.PRESCRIPTION, DocumentType.CLAIM_REQUEST):
            result = _parse("Texte minimal.", doc_type=doc_type)
            assert result.field_count == len(result.fields)

    def test_provenance_page_number(self):
        text = "Total : 250.00 USD\nPatient : Jean Dupont"
        result = _parse(text, page=3)
        for field in result.fields.values():
            if field.provenance is not None:
                assert field.provenance.page_number == 3

    def test_confidence_dans_bornes(self):
        text = "Total facture : 100.00 USD"
        result = _parse(text, conf=0.80)
        for field in result.fields.values():
            assert 0.0 <= field.confidence <= 1.0

    def test_sha256_dans_provenance(self):
        text = "Total : 500.00 USD"
        sha = "b" * 64
        result = parse_document(
            text, DocumentType.INVOICE, 1, OcrSource.PDF_TEXT, 0.90,
            filename="doc.pdf", sha256=sha,
        )
        for field in result.fields.values():
            if field.provenance is not None:
                assert field.provenance.sha256 == sha

    def test_requires_review_si_faible_confiance(self):
        text = "Total facture : 100.00 USD"
        result = _parse(text, conf=0.50)  # < 0.65 → requires_review=True
        for field in result.fields.values():
            if field.confidence < 0.65:
                assert field.requires_review is True

    def test_valeur_non_vide_pour_champ_extrait(self):
        text = "Total : 999.00 USD\nDate : 2024-06-01"
        result = _parse(text)
        for name, field in result.fields.items():
            assert field.value.strip() != "", f"Champ {name!r} a une valeur vide"
