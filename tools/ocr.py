"""OCR de fichiers image (PNG, JPEG) et de PDFs scannés via Tesseract (Étape 9).

Le texte produit est une donnée opaque — jamais interprétée comme instruction.
Repli gracieux si Tesseract n'est pas disponible (OcrEngineUnavailableError).

Garanties de sécurité :
  - Seuls les formats PNG et JPEG sont acceptés pour les fichiers image.
  - Les images démesurées sont refusées avant tout traitement.
  - L'image originale n'est jamais modifiée ni écrasée.
  - Aucun artefact n'est écrit sur le disque — tout le traitement est en mémoire.
  - Erreur contrôlée (sans exception) si Tesseract est indisponible.

Prétraitements appliqués (minimaux, sans filtres excessifs) :
  1. Correction d'orientation EXIF.
  2. Conversion en niveaux de gris.
  3. Amélioration légère du contraste (autocontrast).
  4. Réduction du bruit (filtre médian 3×3).
  5. Redimensionnement upscale contrôlé si l'image est trop petite pour Tesseract.

Aucun appel LLM, aucun effet de bord.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

from schemas.domain import OcrSource
from tools.text_normalizer import normalize_ocr_text

# ── Constantes publiques ──────────────────────────────────────────────────────

#: Formats d'image acceptés par l'agent OCR (valeurs Pillow : img.format).
ALLOWED_IMAGE_FORMATS: frozenset[str] = frozenset({"PNG", "JPEG"})

#: Taille maximale d'un côté de l'image en pixels (protection mémoire / DoS).
MAX_DIMENSION_PX: int = 10_000

#: Nombre total de pixels maximum (~50 Mpixels ≈ image 8K).
MAX_TOTAL_PIXELS: int = 50_000_000

# ── Constantes internes ───────────────────────────────────────────────────────

_WORD_CONF_THRESHOLD = 30
_TARGET_LONG_SIDE = 2_480  # ~A4 à 300 DPI — cible d'upscale


# ── Exceptions ────────────────────────────────────────────────────────────────


class OcrEngineUnavailableError(RuntimeError):
    """Levée quand pytesseract ou le binaire Tesseract est absent."""


# ── Modèles de résultat ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class OcrPageResult:
    """Résultat OCR d'une image ou d'une page PDF."""

    page_number: int        # 1-indexé
    normalized_text: str
    char_count: int
    mean_confidence: float  # 0.0 → 1.0 (moyenne des confiances par mot)
    engine_available: bool
    method: OcrSource = OcrSource.IMAGE_OCR  # méthode OCR utilisée


@dataclass(frozen=True)
class OcrResult:
    """Résultat agrégé de l'OCR sur un fichier entier."""

    pages: list[OcrPageResult]
    total_chars: int
    mean_confidence: float
    engine_available: bool
    error: str | None = None


# ── Helpers internes ──────────────────────────────────────────────────────────


def _empty_error(msg: str, *, engine_available: bool = True) -> OcrResult:
    """Construit un OcrResult d'erreur sans pages."""
    return OcrResult(
        pages=[],
        total_chars=0,
        mean_confidence=0.0,
        engine_available=engine_available,
        error=msg,
    )


def _resolve_authorized_path(path: Path, allowed_root: Path | None) -> tuple[Path | None, str | None]:
    """Résout un chemin et vérifie qu'il reste sous allowed_root si fourni."""
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        return None, f"Chemin image invalide : {exc}"

    if allowed_root is None:
        return resolved, None

    try:
        root = allowed_root.resolve(strict=True)
    except OSError as exc:
        return None, f"Zone autorisée invalide : {exc}"

    if resolved == root or root in resolved.parents:
        return resolved, None

    return None, "Chemin image hors de la zone autorisée."


def _validate_image_dimensions(img: Image.Image) -> str | None:
    """Vérifie les dimensions de l'image.

    Retourne un message d'erreur humainement lisible si l'image est invalide,
    None si les dimensions sont acceptables.
    """
    w, h = img.size
    if w <= 0 or h <= 0:
        return f"Dimensions invalides : {w}×{h} px"
    if w > MAX_DIMENSION_PX or h > MAX_DIMENSION_PX:
        return (
            f"Image trop grande : {w}×{h} px "
            f"(maximum {MAX_DIMENSION_PX} px par côté)"
        )
    if w * h > MAX_TOTAL_PIXELS:
        return (
            f"Image trop grande : {w * h:,} pixels totaux "
            f"(maximum {MAX_TOTAL_PIXELS:,} px)"
        )
    return None


def _apply_exif_orientation(img: Image.Image) -> Image.Image:
    """Corrige l'orientation selon les métadonnées EXIF.

    L'image originale n'est jamais modifiée — retourne une nouvelle instance.
    Sans effet si les métadonnées EXIF sont absentes ou illisibles.
    """
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img


def _preprocess_image(img: Image.Image) -> Image.Image:
    """Prétraitement de l'image pour améliorer la reconnaissance OCR.

    Opérations appliquées dans l'ordre :
      1. Correction d'orientation EXIF.
      2. Conversion en niveaux de gris.
      3. Amélioration légère du contraste (autocontrast, cutoff=2).
      4. Réduction du bruit (filtre médian 3×3).
      5. Redimensionnement upscale contrôlé si trop petit pour Tesseract.

    L'image originale n'est jamais modifiée — toutes les opérations
    produisent de nouvelles instances PIL.
    """
    img = _apply_exif_orientation(img)
    img = img.convert("L")                          # niveaux de gris
    img = ImageOps.autocontrast(img, cutoff=2)      # contraste léger
    img = img.filter(ImageFilter.MedianFilter(3))   # réduction du bruit

    # Upscale si trop petit — Tesseract préfère ≥ 300 DPI équivalent
    w, h = img.size
    long_side = max(w, h)
    if long_side < _TARGET_LONG_SIDE:
        scale = _TARGET_LONG_SIDE / long_side
        new_w = min(int(w * scale), MAX_DIMENSION_PX)
        new_h = min(int(h * scale), MAX_DIMENSION_PX)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    return img


def _run_tesseract(img: Image.Image) -> tuple[str, float]:
    """Lance Tesseract sur une image PIL.

    Retourne (texte_normalisé, confiance_moyenne).
    Lève OcrEngineUnavailableError si le moteur est absent.
    Lève RuntimeError pour toute autre erreur Tesseract.
    """
    return _run_tesseract_with_language(img, language="fra+eng")


def _run_tesseract_with_language(img: Image.Image, *, language: str) -> tuple[str, float]:
    """Lance Tesseract avec une langue configurable."""
    try:
        import pytesseract
    except ImportError as exc:
        raise OcrEngineUnavailableError("pytesseract non installé.") from exc

    try:
        data = pytesseract.image_to_data(
            img,
            lang=language,
            output_type=pytesseract.Output.DICT,
            config="--psm 6",
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise OcrEngineUnavailableError(f"Binaire Tesseract introuvable : {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Erreur Tesseract : {exc}") from exc

    words: list[str] = []
    confidences: list[float] = []
    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        try:
            conf_value = float(conf)
        except (TypeError, ValueError):
            continue
        if conf_value < 0:
            continue  # valeur sentinelle Tesseract
        if conf_value >= _WORD_CONF_THRESHOLD and text.strip():
            words.append(text.strip())
        confidences.append(conf_value / 100.0)

    raw_text = " ".join(words)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return normalize_ocr_text(raw_text), mean_conf


# ── Fonctions publiques ───────────────────────────────────────────────────────


def ocr_image_file(
    path: Path,
    *,
    allowed_root: Path | None = None,
    allowed_formats: frozenset[str] = ALLOWED_IMAGE_FORMATS,
    enabled: bool = True,
    language: str = "fra+eng",
    max_text_chars: int | None = None,
) -> OcrResult:
    """Applique l'OCR sur un fichier image (PNG, JPEG).

    Pré-conditions :
      - Le chemin pointe vers un fichier image PNG ou JPEG valide.
      - Si allowed_root est fourni, le chemin résolu doit rester dans cette zone.
      - L'image respecte les limites MAX_DIMENSION_PX et MAX_TOTAL_PIXELS.

    Garanties :
      - L'image originale n'est jamais modifiée ni écrasée.
      - Aucun artefact n'est écrit sur le disque (traitement 100 % en mémoire).
      - Retour contrôlé sans exception si Tesseract est indisponible.

    Paramètre :
      allowed_root — racine autorisée déjà validée par l'appelant.
      allowed_formats — ensemble des formats PIL acceptés (défaut : PNG, JPEG).
      enabled — désactive explicitement Tesseract si False.
      language — langue(s) Tesseract, ex. "eng" ou "fra+eng".
      max_text_chars — limite optionnelle sur le texte OCR conservé.
    """
    if not enabled:
        return _empty_error("OCR désactivé par configuration.", engine_available=False)

    resolved_path, path_error = _resolve_authorized_path(path, allowed_root)
    if path_error:
        return _empty_error(path_error)

    # ── 1. Chargement de l'image ─────────────────────────────────────────────
    try:
        with Image.open(resolved_path) as opened:
            img_format = (opened.format or "").upper()
            opened.load()
            img = opened.copy()
    except UnidentifiedImageError as exc:
        return _empty_error(f"Image invalide ou corrompue : {exc}")
    except Exception as exc:
        return _empty_error(f"Impossible d'ouvrir l'image : {exc}")

    # ── 2. Validation du format ──────────────────────────────────────────────
    if img_format not in allowed_formats:
        return _empty_error(
            f"Format d'image non autorisé : {img_format!r} "
            f"(formats acceptés : {sorted(allowed_formats)})"
        )

    # ── 3. Validation des dimensions ─────────────────────────────────────────
    dim_error = _validate_image_dimensions(img)
    if dim_error:
        return _empty_error(dim_error)

    # ── 4. Prétraitement (en mémoire — original inchangé) ────────────────────
    img_proc = _preprocess_image(img)

    # ── 5. OCR ───────────────────────────────────────────────────────────────
    try:
        text, conf = _run_tesseract_with_language(img_proc, language=language)
        if max_text_chars is not None and len(text) > max_text_chars:
            text = text[:max_text_chars]
        page = OcrPageResult(
            page_number=1,
            normalized_text=text,
            char_count=len(text),
            mean_confidence=conf,
            engine_available=True,
            method=OcrSource.IMAGE_OCR,
        )
        return OcrResult(
            pages=[page],
            total_chars=len(text),
            mean_confidence=conf,
            engine_available=True,
        )
    except OcrEngineUnavailableError as exc:
        page = OcrPageResult(
            page_number=1,
            normalized_text="",
            char_count=0,
            mean_confidence=0.0,
            engine_available=False,
            method=OcrSource.IMAGE_OCR,
        )
        return OcrResult(
            pages=[page],
            total_chars=0,
            mean_confidence=0.0,
            engine_available=False,
            error=str(exc),
        )
    except Exception as exc:
        page = OcrPageResult(
            page_number=1,
            normalized_text="",
            char_count=0,
            mean_confidence=0.0,
            engine_available=True,
            method=OcrSource.IMAGE_OCR,
        )
        return OcrResult(
            pages=[page],
            total_chars=0,
            mean_confidence=0.0,
            engine_available=True,
            error=f"Erreur OCR : {exc}",
        )


def ocr_pdf_pages(
    pdf_images: list[Image.Image],
    *,
    enabled: bool = True,
    language: str = "fra+eng",
    max_pages: int | None = None,
    max_text_chars: int | None = None,
) -> OcrResult:
    """Applique l'OCR sur une liste d'images de pages extraites d'un PDF scanné.

    Les images sont des instances PIL en mémoire — aucun fichier n'est lu.
    Les dimensions de chaque image sont vérifiées avant traitement.
    Les images originales ne sont jamais modifiées.

    Jamais d'exception propagée.
    """
    if not enabled:
        return _empty_error("OCR désactivé par configuration.", engine_available=False)

    pages: list[OcrPageResult] = []
    total_chars = 0
    all_confs: list[float] = []
    engine_available = True
    first_error: str | None = None

    for i, img in enumerate(pdf_images, start=1):
        if max_pages is not None and i > max_pages:
            break

        if max_text_chars is not None and total_chars >= max_text_chars:
            break

        # Validation des dimensions de la page
        dim_error = _validate_image_dimensions(img)
        if dim_error:
            first_error = first_error or f"Page {i} : {dim_error}"
            pages.append(OcrPageResult(
                page_number=i,
                normalized_text="",
                char_count=0,
                mean_confidence=0.0,
                engine_available=True,
                method=OcrSource.PDF_OCR,
            ))
            continue

        img_proc = _preprocess_image(img)
        try:
            text, conf = _run_tesseract_with_language(img_proc, language=language)
            if max_text_chars is not None:
                remaining = max_text_chars - total_chars
                text = text[:remaining]
            pages.append(OcrPageResult(
                page_number=i,
                normalized_text=text,
                char_count=len(text),
                mean_confidence=conf,
                engine_available=True,
                method=OcrSource.PDF_OCR,
            ))
            all_confs.append(conf)
            total_chars += len(text)
        except OcrEngineUnavailableError as exc:
            engine_available = False
            first_error = first_error or str(exc)
            pages.append(OcrPageResult(
                page_number=i,
                normalized_text="",
                char_count=0,
                mean_confidence=0.0,
                engine_available=False,
                method=OcrSource.PDF_OCR,
            ))
        except Exception as exc:
            first_error = first_error or f"Erreur OCR page {i} : {exc}"
            pages.append(OcrPageResult(
                page_number=i,
                normalized_text="",
                char_count=0,
                mean_confidence=0.0,
                engine_available=True,
                method=OcrSource.PDF_OCR,
            ))

    mean_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0

    return OcrResult(
        pages=pages,
        total_chars=total_chars,
        mean_confidence=mean_conf,
        engine_available=engine_available,
        error=first_error,
    )
