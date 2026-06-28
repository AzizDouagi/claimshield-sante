"""Extraction de texte natif depuis les fichiers PDF (Étape 8).

Utilise pypdf pour lire le texte sélectionnable page par page.
Si un PDF ne contient pas de texte (PDF scanné), retourne des pages vides
et signale qu'une passe OCR est nécessaire.

Garanties de sécurité :
  - Jamais de suivi de liens PDF — pypdf lit uniquement le contenu texte.
  - Jamais d'exécution de scripts ou d'actions PDF embarquées.
  - PDF chiffrés refusés sauf autorisation explicite.
  - Chemin vérifié dans la zone autorisée si allowed_root est fourni.
  - Ouverture en lecture seule (mode "rb") — jamais d'écriture.
  - Nombre de pages limité à max_pages.
  - Taille totale du texte extrait limitée à max_text_chars.

Aucun appel LLM, aucun effet de bord.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from schemas.domain import OcrSource
from schemas.results import PageText
from tools.text_normalizer import normalize_ocr_text

# Seuil minimal de caractères par page pour considérer le texte extractible
MIN_CHARS_PER_PAGE = 20

# Limites par défaut
MAX_PAGES_DEFAULT = 500
MAX_TEXT_CHARS_DEFAULT = 2_000_000  # ~2 M caractères


@dataclass(frozen=True)
class PdfPage:
    """Texte d'une page PDF avec métadonnées de provenance."""

    page_number: int        # 1-indexé
    raw_text: str           # texte tel qu'extrait par pypdf
    normalized_text: str    # texte normalisé par text_normalizer
    char_count: int
    is_text_based: bool     # False si la page ne contient pas de texte natif


@dataclass(frozen=True)
class PdfReadResult:
    """Résultat complet de l'extraction d'un PDF."""

    pages: list[PdfPage]
    page_count: int
    total_chars: int
    is_text_based: bool         # True si au moins une page contient du texte natif
    needs_ocr: bool             # True si le PDF est scanné (aucune page avec texte)
    error: str | None = None
    pages_truncated: bool = False   # True si max_pages ou max_text_chars atteint
    page_texts: list[PageText] = field(default_factory=list)


def _empty_error(msg: str) -> PdfReadResult:
    """Construit un résultat d'erreur sans pages ni page_texts."""
    return PdfReadResult(
        pages=[],
        page_count=0,
        total_chars=0,
        is_text_based=False,
        needs_ocr=False,
        error=msg,
    )


def read_pdf(
    path: Path,
    *,
    allowed_root: Path | None = None,
    max_pages: int = MAX_PAGES_DEFAULT,
    max_text_chars: int = MAX_TEXT_CHARS_DEFAULT,
    min_chars_per_page: int = MIN_CHARS_PER_PAGE,
) -> PdfReadResult:
    """Extrait le texte natif d'un PDF page par page.

    Pré-conditions :
      - Le chemin doit pointer vers un fichier PDF existant.
      - Si allowed_root est fourni, le chemin résolu doit être dans cette racine.

    Paramètres :
      path          — chemin vers le fichier PDF (absolu ou relatif).
      allowed_root  — racine de la zone autorisée ; None = pas de contrôle de zone.
      max_pages     — nombre maximal de pages à extraire (défaut : 500).
      max_text_chars— taille maximale du texte extrait en caractères (défaut : 2 M).
      min_chars_per_page — seuil minimal pour qualifier une page texte.

    Retourne PdfReadResult avec needs_ocr=True si le PDF ne contient
    pas de texte sélectionnable (PDF scanné).

    Jamais d'exception propagée — les erreurs sont retournées dans le champ error.
    """
    # ── 1. Validation du chemin ──────────────────────────────────────────────
    try:
        resolved = path.resolve()
    except Exception as exc:
        return _empty_error(f"Chemin invalide : {exc}")

    if allowed_root is not None:
        try:
            resolved.relative_to(allowed_root.resolve())
        except ValueError:
            return _empty_error(
                f"Chemin hors de la zone autorisée ({allowed_root!r}) : {path!r}"
            )

    # ── 2. Ouverture en lecture seule ────────────────────────────────────────
    try:
        file_handle = open(resolved, "rb")   # noqa: WPS515 — mode explicite "rb" requis
    except OSError as exc:
        return _empty_error(f"Impossible d'ouvrir le fichier PDF : {exc}")

    try:
        try:
            # strict=False : tolérant aux PDFs légèrement non conformes
            reader = PdfReader(file_handle, strict=False)
        except PdfReadError as exc:
            return _empty_error(f"Fichier PDF corrompu ou illisible : {exc}")
        except Exception as exc:
            return _empty_error(f"Erreur inattendue lors de l'ouverture du PDF : {exc}")

        # ── 3. Refus des PDF chiffrés ────────────────────────────────────────
        if reader.is_encrypted:
            return _empty_error(
                "PDF chiffré non autorisé — déchiffrement requis avant traitement."
            )

        # ── 4. Extraction page par page avec limites ─────────────────────────
        pages: list[PdfPage] = []
        total_chars = 0
        pages_truncated = False
        total_page_count = len(reader.pages)

        if total_page_count > max_pages:
            pages_truncated = True

        for i, pdf_page in enumerate(reader.pages, start=1):
            if i > max_pages:
                break

            # Limite de taille du texte total
            if total_chars >= max_text_chars:
                pages_truncated = True
                break

            try:
                # extract_text() ne suit aucun lien et n'exécute aucun script PDF
                raw = pdf_page.extract_text() or ""
            except Exception:
                raw = ""

            normalized = normalize_ocr_text(raw)

            # Tronquer si la page ferait dépasser la limite de texte
            remaining = max_text_chars - total_chars
            if len(normalized) > remaining:
                normalized = normalized[:remaining]
                pages_truncated = True

            char_count = len(normalized)
            total_chars += char_count
            is_text = char_count >= min_chars_per_page

            pages.append(PdfPage(
                page_number=i,
                raw_text=raw,
                normalized_text=normalized,
                char_count=char_count,
                is_text_based=is_text,
            ))

        has_text = any(p.is_text_based for p in pages)

        # ── 5. Construction des PageText typés — sortie Pydantic (Étape 7/8) ─
        page_texts = [
            PageText(
                page_number=p.page_number,
                text=p.normalized_text,
                char_count=p.char_count,
                method=OcrSource.PDF_TEXT,
                confidence=1.0,
                is_text_based=p.is_text_based,
            )
            for p in pages
        ]

        return PdfReadResult(
            pages=pages,
            page_count=len(pages),
            total_chars=total_chars,
            is_text_based=has_text,
            needs_ocr=not has_text and len(pages) > 0,
            pages_truncated=pages_truncated,
            page_texts=page_texts,
        )

    finally:
        file_handle.close()


def pdf_to_full_text(result: PdfReadResult) -> str:
    """Consolide le texte de toutes les pages en une seule chaîne."""
    return "\n\n".join(p.normalized_text for p in result.pages if p.normalized_text)
