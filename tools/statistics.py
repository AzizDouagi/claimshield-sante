"""Statistiques déterministes de similarité — tools/statistics.py.

Fonctions pures, sans E/S, sans appel LLM, sans décision métier : chacune
retourne un score numérique dans ``[0.0, 1.0]`` (1.0 = identique, 0.0 =
totalement différent), jamais un verdict. Utilisées par
``services/duplicate_index.py`` pour évaluer la proximité entre deux
dossiers déjà indexés — la décision de qualifier un rapprochement de
doublon exact ou de quasi-doublon reste de la responsabilité de l'appelant
(``DuplicateIndex``, via une politique versionnée), jamais de ce module.

``text_similarity`` s'appuie sur ``rapidfuzz`` (déjà une dépendance du
projet, section « Fraude ») plutôt que de réimplémenter un algorithme de
distance de chaînes.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal

from rapidfuzz import fuzz


def amount_similarity(a: Decimal, b: Decimal) -> float:
    """Similarité relative entre deux montants — 1.0 si identiques.

    Décroît linéairement avec l'écart relatif au plus grand des deux
    montants (en valeur absolue) : ``1.0 - |a - b| / max(|a|, |b|)``,
    plafonné à ``0.0``. Deux montants nuls sont considérés identiques.
    """
    abs_a, abs_b = abs(a), abs(b)
    if abs_a == 0 and abs_b == 0:
        return 1.0
    largest = max(abs_a, abs_b)
    ratio = abs(a - b) / largest
    return max(0.0, 1.0 - float(ratio))


def text_similarity(a: str, b: str) -> float:
    """Similarité textuelle entre deux chaînes courtes déjà minimisées —
    jamais un texte OCR complet ni un document brut. ``rapidfuzz.fuzz.ratio``
    (0–100) normalisé en ``[0.0, 1.0]``. Deux chaînes vides sont considérées
    identiques ; une seule vide donne une similarité nulle.
    """
    a_stripped, b_stripped = a.strip(), b.strip()
    if not a_stripped and not b_stripped:
        return 1.0
    if not a_stripped or not b_stripped:
        return 0.0
    return fuzz.ratio(a_stripped, b_stripped) / 100.0


def date_proximity(a: date | None, b: date | None, *, window_days: int) -> float:
    """Proximité temporelle entre deux dates — 1.0 si identiques, décroît
    linéairement jusqu'à ``0.0`` à ``window_days`` jours d'écart ou plus.

    ``window_days`` doit être strictement positif. Une des deux dates
    absente (``None``) rend la comparaison non fiable : retourne ``0.0``,
    jamais une proximité inventée.
    """
    if a is None or b is None:
        return 0.0
    if window_days <= 0:
        return 1.0 if a == b else 0.0
    gap_days = abs((a - b).days)
    return max(0.0, 1.0 - gap_days / window_days)


def weighted_composite_score(
    *,
    amount_score: float,
    text_score: float,
    date_score: float,
    weight_amount: float,
    weight_text: float,
    weight_date: float,
) -> float:
    """Moyenne pondérée de trois scores de similarité déjà calculés —
    aucune signification métier propre, uniquement une combinaison
    mécanique. Les poids sont normalisés (divisés par leur somme) pour
    tolérer des poids qui ne totalisent pas exactement 1.0.
    """
    total_weight = weight_amount + weight_text + weight_date
    if total_weight <= 0:
        return 0.0
    weighted_sum = (
        amount_score * weight_amount + text_score * weight_text + date_score * weight_date
    )
    return max(0.0, min(1.0, weighted_sum / total_weight))
