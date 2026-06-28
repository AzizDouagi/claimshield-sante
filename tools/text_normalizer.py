"""Normalisation du texte extrait par OCR ou PDF.

Le texte issu d'un document est une donnée opaque — jamais une instruction à exécuter.
Ce module produit une chaîne sûre, déterministe et reproductible pour les analyses
en aval (classification, parsing, scoring).

Aucun appel LLM, aucun effet de bord.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation


# Caractères invisibles / de contrôle (hors espaces légitimes \t \n \r)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Espaces inhabituels Unicode (non-breaking spaces, zero-width, etc.)
_WEIRD_SPACE_RE = re.compile(
    r"[ ­͏؜ᅟᅠ឴឵"
    r"᠋-᠎ -‏  ‪- "
    r"⁠-⁯　﻿ﾠ￰-￿]"
)

# Répétitions d'espaces > 1 (après normalisation)
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")

# Lignes vides successives > 2
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

_CURRENCY_RE = re.compile(r"\b(USD|EUR|GBP|CAD|CHF|TND)\b|([€$£])", re.IGNORECASE)
_AMOUNT_ALLOWED_RE = re.compile(r"^[+-]?[0-9][0-9\s.,'\u202f]*$")
_DATE_ISO_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_DATE_SLASH_RE = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$")

# Longueur maximale de texte brut acceptée (protection contre DoS/injection volumineuse)
MAX_TEXT_LENGTH = 500_000


@dataclass(frozen=True)
class NormalizedTextValue:
    raw_value: str
    normalized_value: str
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class NormalizedAmount:
    raw_value: str
    normalized_value: Decimal | None
    currency: str | None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class NormalizedDate:
    raw_value: str
    normalized_value: date | None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def normalize_ocr_text(raw: str, *, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Normalise le texte extrait pour une analyse sûre en aval.

    Étapes :
    1. Tronquer à max_length (protection DoS)
    2. Normalisation Unicode NFKC (décomposition de compatibilité + recomposition)
    3. Supprimer les caractères de contrôle
    4. Remplacer les espaces inhabituels par des espaces normaux
    5. Réduire les espaces multiples
    6. Réduire les sauts de ligne excessifs
    7. Supprimer les espaces en début et fin

    Retourne une chaîne pure, toujours non None.
    """
    if not raw:
        return ""

    text = raw[:max_length]
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    text = _WEIRD_SPACE_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def normalize_text_value(raw: str) -> NormalizedTextValue:
    """Normalise une valeur texte en conservant explicitement la valeur brute."""
    return NormalizedTextValue(raw_value=raw, normalized_value=normalize_ocr_text(raw))


def normalize_currency(raw: str | None) -> str | None:
    """Normalise une devise ISO simple ou un symbole courant."""
    if not raw:
        return None
    token = normalize_ocr_text(raw).upper()
    return {
        "€": "EUR",
        "$": "USD",
        "£": "GBP",
        "EURO": "EUR",
        "EUROS": "EUR",
        "DOLLAR": "USD",
        "DOLLARS": "USD",
    }.get(token, token if token in {"USD", "EUR", "GBP", "CAD", "CHF", "TND"} else None)


def _split_amount_currency(raw: str) -> tuple[str, str | None]:
    currency_match = _CURRENCY_RE.search(raw)
    currency = normalize_currency(currency_match.group(0)) if currency_match else None
    amount_part = _CURRENCY_RE.sub("", raw).strip()
    return amount_part, currency


def normalize_decimal_separators(raw: str) -> NormalizedTextValue:
    """Normalise uniquement les séparateurs décimaux, sans créer de Decimal."""
    amount_part, _ = _split_amount_currency(raw)
    parsed = normalize_amount(amount_part)
    if parsed.normalized_value is None:
        return NormalizedTextValue(raw, "", errors=parsed.errors)
    return NormalizedTextValue(raw, format(parsed.normalized_value, "f"), warnings=parsed.warnings)


def normalize_amount(raw: str, *, default_currency: str | None = None) -> NormalizedAmount:
    """Convertit un montant en Decimal sans masquer les entrées invalides."""
    raw_value = raw
    cleaned = normalize_ocr_text(raw)
    if not cleaned:
        return NormalizedAmount(raw_value, None, default_currency, errors=["montant vide"])

    amount_part, detected_currency = _split_amount_currency(cleaned)
    currency = detected_currency or normalize_currency(default_currency)
    compact = amount_part.replace(" ", "").replace("\u202f", "").replace("'", "")

    if not _AMOUNT_ALLOWED_RE.match(compact):
        return NormalizedAmount(raw_value, None, currency, errors=["montant invalide"])

    if compact.startswith("-"):
        return NormalizedAmount(raw_value, None, currency, errors=["montant impossible: valeur négative"])

    comma_count = compact.count(",")
    dot_count = compact.count(".")
    if comma_count and dot_count:
        decimal_sep = "," if compact.rfind(",") > compact.rfind(".") else "."
        thousands_sep = "." if decimal_sep == "," else ","
        compact = compact.replace(thousands_sep, "").replace(decimal_sep, ".")
    elif comma_count:
        if comma_count > 1:
            groups = compact.split(",")
            if any(not group for group in groups) or any(len(group) != 3 for group in groups[1:]):
                return NormalizedAmount(raw_value, None, currency, errors=["montant invalide"])
            compact = compact.replace(",", "")
        else:
            before, after = compact.split(",")
            compact = before + "." + after if len(after) <= 2 else before + after
    elif dot_count > 1:
        groups = compact.split(".")
        if any(not group for group in groups) or any(len(group) != 3 for group in groups[1:]):
            return NormalizedAmount(raw_value, None, currency, errors=["montant invalide"])
        compact = compact.replace(".", "")

    try:
        value = Decimal(compact)
    except InvalidOperation:
        return NormalizedAmount(raw_value, None, currency, errors=["montant invalide"])

    if value > Decimal("1000000000"):
        return NormalizedAmount(raw_value, None, currency, errors=["montant impossible: valeur trop élevée"])

    return NormalizedAmount(raw_value, value, currency)


def normalize_date_value(raw: str, *, prefer_day_first: bool = True) -> NormalizedDate:
    """Convertit une date en date Python et signale les formats ambigus."""
    raw_value = raw
    cleaned = normalize_ocr_text(raw)
    if not cleaned:
        return NormalizedDate(raw_value, None, errors=["date vide"])

    iso = _DATE_ISO_RE.match(cleaned)
    if iso:
        try:
            return NormalizedDate(raw_value, date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3))))
        except ValueError:
            return NormalizedDate(raw_value, None, errors=["date invalide"])

    slash = _DATE_SLASH_RE.match(cleaned)
    if not slash:
        return NormalizedDate(raw_value, None, errors=["format de date non reconnu"])

    first, second, year = map(int, slash.groups())
    if first <= 12 and second <= 12:
        return NormalizedDate(raw_value, None, warnings=["date ambiguë"], errors=["date ambiguë"])

    day, month = (first, second) if first > 12 or prefer_day_first else (second, first)
    if second > 12 and first <= 12:
        month, day = first, second

    try:
        return NormalizedDate(raw_value, date(year, month, day))
    except ValueError:
        return NormalizedDate(raw_value, None, errors=["date invalide"])


def extract_text_lines(text: str) -> list[str]:
    """Découpe le texte normalisé en lignes non vides."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def compute_text_density(text: str, total_chars_possible: int) -> float:
    """Ratio de caractères non-espaces — indicateur de densité utile du texte.

    Un score bas (<0.1) indique un document essentiellement vide ou bruité.
    """
    if total_chars_possible <= 0:
        return 0.0
    non_space = sum(1 for c in text if not c.isspace())
    return min(non_space / total_chars_possible, 1.0)


def count_printable_chars(text: str) -> int:
    """Nombre de caractères imprimables dans le texte normalisé."""
    return sum(1 for c in text if c.isprintable())


def truncate_for_audit(text: str, max_chars: int = 500) -> str:
    """Tronque le texte pour inclusion dans une trace d'audit.

    Ne doit jamais contenir de texte interprétable comme instruction.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"… [{len(text) - max_chars} caractères supplémentaires tronqués]"
