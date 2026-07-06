"""Tests unitaires des statistiques de similarité — tools/statistics.py.

Fonctions pures : vérifie les bornes [0.0, 1.0], les cas limites (montants
nuls, chaînes vides, dates absentes) et la cohérence de la pondération
composite. Aucune décision métier testée ici — uniquement des scores.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal

from tools.statistics import (
    amount_similarity,
    date_proximity,
    text_similarity,
    weighted_composite_score,
)


# ── amount_similarity ────────────────────────────────────────────────────────


class TestAmountSimilarity:
    def test_identical_amounts_is_one(self):
        assert amount_similarity(Decimal("100.00"), Decimal("100.00")) == 1.0

    def test_both_zero_is_one(self):
        assert amount_similarity(Decimal("0"), Decimal("0")) == 1.0

    def test_small_relative_gap_close_to_one(self):
        score = amount_similarity(Decimal("100.00"), Decimal("101.00"))
        assert 0.98 < score < 1.0

    def test_large_relative_gap_close_to_zero(self):
        score = amount_similarity(Decimal("10.00"), Decimal("1000.00"))
        assert score < 0.05

    def test_result_always_within_bounds(self):
        score = amount_similarity(Decimal("-50"), Decimal("500"))
        assert 0.0 <= score <= 1.0


# ── text_similarity ───────────────────────────────────────────────────────────


class TestTextSimilarity:
    def test_identical_strings_is_one(self):
        assert text_similarity("consultation generale", "consultation generale") == 1.0

    def test_both_empty_is_one(self):
        assert text_similarity("", "") == 1.0

    def test_one_empty_is_zero(self):
        assert text_similarity("consultation", "") == 0.0
        assert text_similarity("", "consultation") == 0.0

    def test_similar_strings_high_score(self):
        score = text_similarity("consultation generale", "consultation generale ")
        assert score > 0.9

    def test_unrelated_strings_low_score(self):
        score = text_similarity("consultation generale", "xyz completely unrelated qwerty")
        assert score < 0.5

    def test_result_always_within_bounds(self):
        assert 0.0 <= text_similarity("a", "b") <= 1.0


# ── date_proximity ────────────────────────────────────────────────────────────


class TestDateProximity:
    def test_identical_dates_is_one(self):
        d = date(2024, 1, 15)
        assert date_proximity(d, d, window_days=3) == 1.0

    def test_one_day_gap_within_window(self):
        score = date_proximity(date(2024, 1, 15), date(2024, 1, 16), window_days=3)
        assert 0.6 < score < 1.0

    def test_gap_beyond_window_is_zero(self):
        score = date_proximity(date(2024, 1, 1), date(2024, 2, 1), window_days=3)
        assert score == 0.0

    def test_missing_date_is_zero(self):
        assert date_proximity(None, date(2024, 1, 15), window_days=3) == 0.0
        assert date_proximity(date(2024, 1, 15), None, window_days=3) == 0.0
        assert date_proximity(None, None, window_days=3) == 0.0

    def test_zero_window_requires_exact_match(self):
        d = date(2024, 1, 15)
        assert date_proximity(d, d, window_days=0) == 1.0
        assert date_proximity(d, date(2024, 1, 16), window_days=0) == 0.0


# ── weighted_composite_score ─────────────────────────────────────────────────


class TestWeightedCompositeScore:
    def test_all_scores_one_gives_one(self):
        score = weighted_composite_score(
            amount_score=1.0,
            text_score=1.0,
            date_score=1.0,
            weight_amount=0.4,
            weight_text=0.4,
            weight_date=0.2,
        )
        assert score == 1.0

    def test_all_scores_zero_gives_zero(self):
        score = weighted_composite_score(
            amount_score=0.0,
            text_score=0.0,
            date_score=0.0,
            weight_amount=0.4,
            weight_text=0.4,
            weight_date=0.2,
        )
        assert score == 0.0

    def test_weights_are_normalized(self):
        """Des poids qui ne totalisent pas 1.0 sont tolérés et normalisés."""
        score = weighted_composite_score(
            amount_score=1.0,
            text_score=1.0,
            date_score=1.0,
            weight_amount=4,
            weight_text=4,
            weight_date=2,
        )
        assert score == 1.0

    def test_zero_total_weight_returns_zero(self):
        score = weighted_composite_score(
            amount_score=1.0,
            text_score=1.0,
            date_score=1.0,
            weight_amount=0.0,
            weight_text=0.0,
            weight_date=0.0,
        )
        assert score == 0.0

    def test_result_always_within_bounds(self):
        score = weighted_composite_score(
            amount_score=0.3,
            text_score=0.9,
            date_score=0.1,
            weight_amount=0.4,
            weight_text=0.4,
            weight_date=0.2,
        )
        assert 0.0 <= score <= 1.0
