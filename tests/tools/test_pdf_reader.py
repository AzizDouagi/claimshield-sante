"""Tests unitaires de tools/pdf_reader.py — Étape 21."""

from __future__ import annotations

import io

from pypdf import PdfWriter

from tools.pdf_reader import PdfReadResult, pdf_to_full_text, read_pdf


# ── Helpers ───────────────────────────────────────────────────────────────────


def _text_pdf(lines: list[str]) -> bytes:
    """Crée un PDF avec texte natif via reportlab."""
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    y = 800
    for line in lines:
        c.drawString(72, y, line)
        y -= 20
    c.save()
    return buf.getvalue()


def _multipage_pdf(pages: list[list[str]]) -> bytes:
    """Crée un PDF multi-pages via reportlab."""
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for lines in pages:
        y = 800
        for line in lines:
            c.drawString(72, y, line)
            y -= 20
        c.showPage()
    c.save()
    return buf.getvalue()


def _blank_pdf(n_pages: int = 1) -> bytes:
    """Crée un PDF avec n pages vides (aucun texte natif)."""
    writer = PdfWriter()
    for _ in range(n_pages):
        writer.add_blank_page(width=595, height=842)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PDF texte valide
# ═══════════════════════════════════════════════════════════════════════════════


class TestPdfTexteValide:
    """Un PDF avec couche texte native est extrait correctement."""

    def test_extraction_sans_erreur(self, tmp_path):
        path = tmp_path / "facture.pdf"
        path.write_bytes(_text_pdf(["Total facture : 250.00 USD", "CLM-0001"]))
        result = read_pdf(path)
        assert result.error is None

    def test_is_text_based_true(self, tmp_path):
        path = tmp_path / "facture.pdf"
        path.write_bytes(_text_pdf(["Total facture : 250.00 USD"]))
        result = read_pdf(path)
        assert result.is_text_based is True

    def test_needs_ocr_false(self, tmp_path):
        path = tmp_path / "facture.pdf"
        path.write_bytes(_text_pdf(["Total facture : 250.00 USD"]))
        result = read_pdf(path)
        assert result.needs_ocr is False

    def test_page_count_un(self, tmp_path):
        path = tmp_path / "facture.pdf"
        path.write_bytes(_text_pdf(["Ligne de texte."]))
        result = read_pdf(path)
        assert result.page_count == 1

    def test_total_chars_positif(self, tmp_path):
        path = tmp_path / "facture.pdf"
        path.write_bytes(_text_pdf(["Contenu lisible du document."]))
        result = read_pdf(path)
        assert result.total_chars > 0

    def test_page_texts_produits(self, tmp_path):
        path = tmp_path / "facture.pdf"
        path.write_bytes(_text_pdf(["Texte page 1."]))
        result = read_pdf(path)
        assert len(result.page_texts) == 1
        assert result.page_texts[0].page_number == 1
        assert result.page_texts[0].confidence == 1.0

    def test_pdf_to_full_text_consolide(self, tmp_path):
        path = tmp_path / "facture.pdf"
        path.write_bytes(_text_pdf(["Montant total 123.00"]))
        result = read_pdf(path)
        full = pdf_to_full_text(result)
        assert isinstance(full, str)
        assert len(full) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PDF de plusieurs pages
# ═══════════════════════════════════════════════════════════════════════════════


class TestPdfPlusieursPages:
    """Un PDF de plusieurs pages est extrait page par page."""

    def test_page_count_correct(self, tmp_path):
        path = tmp_path / "multi.pdf"
        path.write_bytes(_multipage_pdf([
            ["Page un — facture."],
            ["Page deux — ordonnance."],
            ["Page trois — demande."],
        ]))
        result = read_pdf(path)
        assert result.page_count == 3

    def test_pages_numerotees_depuis_1(self, tmp_path):
        path = tmp_path / "multi.pdf"
        path.write_bytes(_multipage_pdf([["Page A."], ["Page B."]]))
        result = read_pdf(path)
        nums = [p.page_number for p in result.pages]
        assert nums == [1, 2]

    def test_total_chars_cumule(self, tmp_path):
        path = tmp_path / "multi.pdf"
        path.write_bytes(_multipage_pdf([
            ["Texte suffisamment long pour la page un."],
            ["Texte suffisamment long pour la page deux."],
        ]))
        result = read_pdf(path)
        assert result.total_chars >= sum(p.char_count for p in result.pages)

    def test_full_text_consolide(self, tmp_path):
        path = tmp_path / "multi.pdf"
        path.write_bytes(_multipage_pdf([["Premier contenu."], ["Second contenu."]]))
        result = read_pdf(path)
        full = pdf_to_full_text(result)
        assert "Premier" in full or "contenu" in full.lower() or len(full) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PDF sans couche texte (PDF scanné)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPdfSansCoucheTexte:
    """Un PDF scanné (aucun texte natif) déclenche needs_ocr=True."""

    def test_needs_ocr_true(self, tmp_path):
        path = tmp_path / "scanne.pdf"
        path.write_bytes(_blank_pdf(1))
        result = read_pdf(path)
        assert result.needs_ocr is True

    def test_is_text_based_false(self, tmp_path):
        path = tmp_path / "scanne.pdf"
        path.write_bytes(_blank_pdf(1))
        result = read_pdf(path)
        assert result.is_text_based is False

    def test_pas_erreur_retournee(self, tmp_path):
        path = tmp_path / "scanne.pdf"
        path.write_bytes(_blank_pdf(1))
        result = read_pdf(path)
        assert result.error is None

    def test_page_count_non_nul(self, tmp_path):
        path = tmp_path / "scanne.pdf"
        path.write_bytes(_blank_pdf(2))
        result = read_pdf(path)
        assert result.page_count == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PDF vide (aucune page)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPdfVide:
    """Un PDF sans aucune page est traité sans exception."""

    def test_page_count_zero(self, tmp_path):
        path = tmp_path / "vide.pdf"
        writer = PdfWriter()
        buf = io.BytesIO()
        writer.write(buf)
        path.write_bytes(buf.getvalue())
        result = read_pdf(path)
        assert result.page_count == 0

    def test_total_chars_zero(self, tmp_path):
        path = tmp_path / "vide.pdf"
        writer = PdfWriter()
        buf = io.BytesIO()
        writer.write(buf)
        path.write_bytes(buf.getvalue())
        result = read_pdf(path)
        assert result.total_chars == 0

    def test_retourne_pdf_read_result(self, tmp_path):
        path = tmp_path / "vide.pdf"
        writer = PdfWriter()
        buf = io.BytesIO()
        writer.write(buf)
        path.write_bytes(buf.getvalue())
        result = read_pdf(path)
        assert isinstance(result, PdfReadResult)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PDF corrompu
# ═══════════════════════════════════════════════════════════════════════════════


class TestPdfCorrompu:
    """Un fichier non-PDF ou corrompu ne propage pas d'exception."""

    def test_garbage_bytes_retourne_result(self, tmp_path):
        path = tmp_path / "garbage.pdf"
        path.write_bytes(b"\x00\x01\x02\x03 GARBAGE DATA NOT A PDF")
        result = read_pdf(path)
        assert isinstance(result, PdfReadResult)

    def test_fichier_txt_renomme_pdf(self, tmp_path):
        path = tmp_path / "texte.pdf"
        path.write_bytes(b"Ceci est un fichier texte renomme en PDF.")
        result = read_pdf(path)
        assert isinstance(result, PdfReadResult)

    def test_pas_exception_propagee(self, tmp_path):
        path = tmp_path / "corrompu.pdf"
        path.write_bytes(b"%PDF-1.4\n%%EOF\n" + b"\xff" * 100)
        result = read_pdf(path)
        assert isinstance(result, PdfReadResult)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PDF chiffré
# ═══════════════════════════════════════════════════════════════════════════════


class TestPdfChiffre:
    """Un PDF chiffré est refusé avec un message d'erreur explicite."""

    def _make_encrypted_pdf(self) -> bytes:
        writer = PdfWriter()
        writer.add_blank_page(width=595, height=842)
        writer.encrypt("secret123")
        buf = io.BytesIO()
        writer.write(buf)
        return buf.getvalue()

    def test_erreur_retournee(self, tmp_path):
        path = tmp_path / "chiffre.pdf"
        path.write_bytes(self._make_encrypted_pdf())
        result = read_pdf(path)
        assert result.error is not None

    def test_message_erreur_chiffre(self, tmp_path):
        path = tmp_path / "chiffre.pdf"
        path.write_bytes(self._make_encrypted_pdf())
        result = read_pdf(path)
        assert result.error is not None
        assert "chiffr" in result.error.lower() or "encrypt" in result.error.lower()

    def test_zero_pages_extraites(self, tmp_path):
        path = tmp_path / "chiffre.pdf"
        path.write_bytes(self._make_encrypted_pdf())
        result = read_pdf(path)
        assert result.page_count == 0

    def test_pas_exception_propagee(self, tmp_path):
        path = tmp_path / "chiffre.pdf"
        path.write_bytes(self._make_encrypted_pdf())
        result = read_pdf(path)
        assert isinstance(result, PdfReadResult)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PDF dépassant la limite de pages
# ═══════════════════════════════════════════════════════════════════════════════


class TestPdfLimitePages:
    """La limite max_pages tronque l'extraction sans erreur."""

    def test_pages_tronquees(self, tmp_path):
        path = tmp_path / "long.pdf"
        path.write_bytes(_multipage_pdf([
            [f"Contenu de la page {i}."] for i in range(1, 6)
        ]))
        result = read_pdf(path, max_pages=2)
        assert result.page_count == 2

    def test_pages_truncated_true(self, tmp_path):
        path = tmp_path / "long.pdf"
        path.write_bytes(_multipage_pdf([
            [f"Contenu de la page {i}."] for i in range(1, 6)
        ]))
        result = read_pdf(path, max_pages=3)
        assert result.pages_truncated is True

    def test_pas_erreur(self, tmp_path):
        path = tmp_path / "long.pdf"
        path.write_bytes(_multipage_pdf([
            [f"Contenu de la page {i}."] for i in range(1, 4)
        ]))
        result = read_pdf(path, max_pages=2)
        assert result.error is None

    def test_limite_superieure_pas_tronquee(self, tmp_path):
        path = tmp_path / "court.pdf"
        path.write_bytes(_multipage_pdf([["Page A."], ["Page B."]]))
        result = read_pdf(path, max_pages=10)
        assert result.pages_truncated is False
        assert result.page_count == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Chemin hors de la zone autorisée
# ═══════════════════════════════════════════════════════════════════════════════


class TestPdfHorsZone:
    """Un chemin hors de allowed_root est refusé immédiatement."""

    def test_erreur_retournee(self, tmp_path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        path = outside / "doc.pdf"
        path.write_bytes(_text_pdf(["Contenu secret."]))
        result = read_pdf(path, allowed_root=allowed)
        assert result.error is not None

    def test_zero_pages(self, tmp_path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        path = outside / "doc.pdf"
        path.write_bytes(_text_pdf(["Contenu secret."]))
        result = read_pdf(path, allowed_root=allowed)
        assert result.page_count == 0

    def test_fichier_dans_zone_accepte(self, tmp_path):
        allowed = tmp_path / "incoming"
        allowed.mkdir()
        path = allowed / "doc.pdf"
        path.write_bytes(_text_pdf(["Contenu autorisé."])  )
        result = read_pdf(path, allowed_root=allowed)
        assert result.error is None
