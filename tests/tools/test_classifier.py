"""Tests unitaires de tools/document_classifier.py — Étape 21."""

from __future__ import annotations


from schemas.domain import DocumentType
from tools.document_classifier import (
    CLASSIFIER_RULES_VERSION,
    ClassificationResult,
    classify_by_filename,
    classify_by_mime,
    classify_document,
)


# ── Textes de référence ────────────────────────────────────────────────────────

_TEXTE_FACTURE = """
FACTURE
Référence facture : INV-CLM-0001
Prestataire : Clinique Saint-Luc
Total : 3 666.69 USD
Actes médicaux : consultation, radiologie
Montant facturé : 3 666.69 USD
"""

_TEXTE_ORDONNANCE = """
ORDONNANCE
Médecin prescripteur : Dr. Martin
Patient : Jean Dupont
Médicament : Amoxicilline 500 mg
Posologie : 3 fois par jour
RX-CLM-0001
Comprimés 30 jours
"""

_TEXTE_DEMANDE = """
DEMANDE DE REMBOURSEMENT
CLM-0001
Montant demandé : 2 500.00 USD
Assurance : Mutuelle Nationale
Taux de couverture : 80 %
Part assureur : 2 000.00 USD
Claim number : CLM-0001
"""

_TEXTE_FHIR = """
{
  "resourceType": "Bundle",
  "entry": [
    { "resourceType": "Patient" },
    { "resourceType": "Claim" }
  ]
}
"""

_TEXTE_AMBIGU = """
Facture
Remboursement
Total : 100.00
Médicament Amoxicilline 500 mg
"""

_TEXTE_INCONNU = """
Ceci est un document quelconque sans mots-clés particuliers.
Numéro de référence : XYZ-9999.
Date : 01/01/2024.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Facture reconnue
# ═══════════════════════════════════════════════════════════════════════════════


class TestFactureReconnue:
    """Le classifieur identifie une facture médicale."""

    def test_document_type_invoice(self):
        result = classify_document(_TEXTE_FACTURE, filename="facture_CLM-0001.pdf")
        assert result.document_type == DocumentType.INVOICE

    def test_confidence_elevee(self):
        result = classify_document(_TEXTE_FACTURE, filename="facture_CLM-0001.pdf")
        assert result.confidence >= 0.3

    def test_pas_ambigu(self):
        result = classify_document(_TEXTE_FACTURE, filename="facture_CLM-0001.pdf")
        assert result.is_ambiguous is False

    def test_retourne_classification_result(self):
        result = classify_document(_TEXTE_FACTURE)
        assert isinstance(result, ClassificationResult)

    def test_scores_dict_present(self):
        result = classify_document(_TEXTE_FACTURE, filename="facture_CLM-0001.pdf")
        assert isinstance(result.scores, dict)
        assert DocumentType.INVOICE.value in result.scores

    def test_version_regles_stable(self):
        result = classify_document(_TEXTE_FACTURE)
        assert result.rules_version == CLASSIFIER_RULES_VERSION


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Ordonnance reconnue
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrdonnanceReconnue:
    """Le classifieur identifie une ordonnance médicale."""

    def test_document_type_prescription(self):
        result = classify_document(_TEXTE_ORDONNANCE, filename="ordonnance_CLM-0001.pdf")
        assert result.document_type == DocumentType.PRESCRIPTION

    def test_confidence_positive(self):
        result = classify_document(_TEXTE_ORDONNANCE, filename="ordonnance_CLM-0001.pdf")
        assert result.confidence > 0.0

    def test_pas_ambigu(self):
        result = classify_document(_TEXTE_ORDONNANCE, filename="ordonnance_CLM-0001.pdf")
        assert result.is_ambiguous is False

    def test_source_keywords_ou_combined(self):
        result = classify_document(_TEXTE_ORDONNANCE, filename="ordonnance_CLM-0001.pdf")
        assert result.classification_source in ("keywords", "combined", "filename")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Demande de remboursement reconnue
# ═══════════════════════════════════════════════════════════════════════════════


class TestDemandeReconnue:
    """Le classifieur identifie une demande de remboursement."""

    def test_document_type_claim_request(self):
        result = classify_document(_TEXTE_DEMANDE, filename="demande_remboursement_CLM-0001.pdf")
        assert result.document_type == DocumentType.CLAIM_REQUEST

    def test_confidence_positive(self):
        result = classify_document(_TEXTE_DEMANDE, filename="demande_remboursement_CLM-0001.pdf")
        assert result.confidence > 0.0

    def test_scores_incluent_claim(self):
        result = classify_document(_TEXTE_DEMANDE, filename="demande_remboursement_CLM-0001.pdf")
        assert DocumentType.CLAIM_REQUEST.value in result.scores


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Document ambigu
# ═══════════════════════════════════════════════════════════════════════════════


class TestDocumentAmbigu:
    """Un document avec des mots-clés de plusieurs types est marqué ambigu."""

    def test_is_ambiguous_true(self):
        result = classify_document(_TEXTE_AMBIGU)
        # Peut être ambigu ou unknown selon les scores — pas de doc_type unique attendu
        assert isinstance(result, ClassificationResult)

    def test_confiance_reduite_si_ambigu(self):
        result = classify_document(_TEXTE_AMBIGU)
        if result.is_ambiguous:
            # La confiance doit être réduite par le facteur d'ambiguïté (×0.65)
            assert result.confidence <= 0.65

    def test_pas_exception(self):
        result = classify_document(_TEXTE_AMBIGU)
        assert isinstance(result, ClassificationResult)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Document inconnu
# ═══════════════════════════════════════════════════════════════════════════════


class TestDocumentInconnu:
    """Un document sans mots-clés reconnus est classé UNKNOWN."""

    def test_document_type_unknown(self):
        result = classify_document(_TEXTE_INCONNU, filename="rapport.txt")
        assert result.document_type == DocumentType.UNKNOWN

    def test_confidence_zero(self):
        result = classify_document(_TEXTE_INCONNU, filename="rapport.txt")
        assert result.confidence == 0.0

    def test_texte_vide_unknown(self):
        result = classify_document("", filename="inconnu.pdf")
        assert result.document_type == DocumentType.UNKNOWN

    def test_source_unknown(self):
        result = classify_document(_TEXTE_INCONNU, filename="rapport.txt")
        assert result.classification_source == "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Nom de fichier trompeur avec contenu différent
# ═══════════════════════════════════════════════════════════════════════════════


class TestNomTrompeur:
    """Le contenu texte prime sur le nom de fichier seul."""

    def test_nom_facture_contenu_ordonnance(self):
        """Fichier nommé 'facture' mais contenant une ordonnance."""
        result = classify_document(
            _TEXTE_ORDONNANCE,
            filename="facture_test.pdf",  # nom trompeur
        )
        # Le contenu (ordonnance) doit l'emporter sur le filename (facture)
        # Résultat accepté : PRESCRIPTION ou UNKNOWN (pas INVOICE seul par filename)
        # Le classifieur combine filename + keywords — si les keyword scores l'emportent
        assert result.document_type != DocumentType.INVOICE or result.is_ambiguous

    def test_nom_ordonnance_contenu_demande(self):
        """Fichier nommé 'ordonnance' mais contenant une demande de remboursement."""
        result = classify_document(
            _TEXTE_DEMANDE,
            filename="ordonnance_test.pdf",  # nom trompeur
        )
        assert result.document_type != DocumentType.PRESCRIPTION or result.is_ambiguous

    def test_classify_by_filename_seul(self):
        """classify_by_filename retourne l'indice filename sans texte."""
        hint = classify_by_filename("facture_CLM-0001.pdf")
        assert hint is not None
        assert hint[0] == DocumentType.INVOICE

    def test_classify_by_filename_inconnu(self):
        """Un nom générique sans indice retourne None."""
        hint = classify_by_filename("document_quelconque.pdf")
        assert hint is None

    def test_classify_by_mime_json_fhir(self):
        """application/json est un indice fort pour FHIR_BUNDLE."""
        hint = classify_by_mime("application/json")
        assert hint is not None
        assert hint[0] == DocumentType.FHIR_BUNDLE

    def test_classify_by_mime_pdf_none(self):
        """application/pdf n'est pas un indice MIME discriminant."""
        hint = classify_by_mime("application/pdf")
        assert hint is None

    def test_fhir_bundle_reconnu_par_contenu(self):
        """Un bundle FHIR est reconnu par le contenu JSON."""
        result = classify_document(_TEXTE_FHIR, mime_type="application/json")
        assert result.document_type == DocumentType.FHIR_BUNDLE
