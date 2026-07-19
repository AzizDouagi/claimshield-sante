"""Tests unitaires des contrôles de chronologie — tools/date_checks.py.

Couvre les trois familles de cas demandées : cas valide (aucun signal), date
impossible (``IMPOSSIBLE_DATE``), preuve d'acte manquante
(``MISSING_PROCEDURE_EVIDENCE``) — ainsi que la chronologie ordonnance/soin
(``PRESCRIPTION_BEFORE_CARE``/``PRESCRIPTION_TOO_FAR_AFTER_CARE``). Aucune
décision métier vérifiée ici : uniquement la présence, l'attribution
(``evidence_id``/``source``/``field``) et la sévérité des signaux renvoyés.
"""
from __future__ import annotations

from datetime import date, timedelta

from schemas.domain import SeverityLevel
from schemas.results import ClinicalEvidenceSource
from tools.date_checks import (
    MAX_PRESCRIPTION_AFTER_CARE_DAYS,
    check_missing_procedure_evidence,
    check_prescription_before_care,
    check_prescription_too_far_after_care,
    parse_checked_date,
    run_date_checks,
)


# ── parse_checked_date ───────────────────────────────────────────────────────


class TestParseCheckedDate:
    def test_valid_iso_date_parses_without_signal(self):
        parsed, signal = parse_checked_date("2024-01-15", field_name="care_date")
        assert parsed == date(2024, 1, 15)
        assert signal is None

    def test_missing_value_produces_no_signal(self):
        parsed, signal = parse_checked_date(None, field_name="care_date")
        assert parsed is None
        assert signal is None

    def test_empty_value_produces_no_signal(self):
        parsed, signal = parse_checked_date("   ", field_name="care_date")
        assert parsed is None
        assert signal is None

    def test_impossible_date_produces_signal(self):
        """Une date qui n'existe pas (30 février) est structurellement
        impossible — signalée, jamais silencieusement ignorée."""
        parsed, signal = parse_checked_date("2024-02-30", field_name="care_date")
        assert parsed is None
        assert signal is not None
        assert signal.signal_type == "IMPOSSIBLE_DATE"
        assert signal.severity is SeverityLevel.CRITICAL
        assert signal.fields_compared == ["care_date"]

    def test_unrecognized_format_produces_signal(self):
        parsed, signal = parse_checked_date("pas une date", field_name="prescription_date")
        assert parsed is None
        assert signal is not None
        assert signal.signal_type == "IMPOSSIBLE_DATE"

    def test_signal_carries_attributed_evidence(self):
        _, signal = parse_checked_date("2024-02-30", field_name="care_date", document_reference="INVOICE")
        assert signal is not None
        assert len(signal.evidence) == 1
        evidence = signal.evidence[0]
        assert evidence.evidence_id
        assert evidence.source is ClinicalEvidenceSource.OCR_EXTRACTION
        assert evidence.field == "care_date"
        assert evidence.document_reference == "INVOICE"

    def test_both_components_under_12_resolved_never_signaled(self):
        """Correctif Phase 10 (mesure V2) : un format `JJ/MM/AAAA` où jour et
        mois sont tous deux ≤ 12 n'est plus jamais rejeté comme ambigu —
        résolu par convention jour-mois (`prefer_day_first`, défaut du
        projet), jamais un signal `IMPOSSIBLE_DATE` sur une date valide."""
        parsed, signal = parse_checked_date("03/04/2024", field_name="care_date")
        assert parsed == date(2024, 4, 3)
        assert signal is None


# ── check_prescription_before_care ──────────────────────────────────────────


class TestPrescriptionBeforeCare:
    def test_prescription_after_care_no_signal(self):
        signal = check_prescription_before_care(date(2024, 1, 16), date(2024, 1, 15))
        assert signal is None

    def test_prescription_same_day_no_signal(self):
        signal = check_prescription_before_care(date(2024, 1, 15), date(2024, 1, 15))
        assert signal is None

    def test_prescription_before_care_is_signaled(self):
        signal = check_prescription_before_care(date(2024, 1, 10), date(2024, 1, 15))
        assert signal is not None
        assert signal.signal_type == "PRESCRIPTION_BEFORE_CARE"
        assert signal.severity is SeverityLevel.CRITICAL
        assert len(signal.evidence) == 2

    def test_missing_prescription_date_no_signal(self):
        assert check_prescription_before_care(None, date(2024, 1, 15)) is None

    def test_missing_care_date_no_signal(self):
        assert check_prescription_before_care(date(2024, 1, 15), None) is None


# ── check_prescription_too_far_after_care ───────────────────────────────────


class TestPrescriptionTooFarAfterCare:
    def test_within_tolerance_no_signal(self):
        signal = check_prescription_too_far_after_care(
            date(2024, 1, 15) , date(2024, 1, 1)
        )
        assert signal is None

    def test_beyond_tolerance_is_signaled(self):
        signal = check_prescription_too_far_after_care(
            date(2024, 3, 1), date(2024, 1, 1)
        )
        assert signal is not None
        assert signal.signal_type == "PRESCRIPTION_TOO_FAR_AFTER_CARE"
        assert signal.severity is SeverityLevel.MEDIUM

    def test_custom_tolerance_respected(self):
        assert (
            check_prescription_too_far_after_care(
                date(2024, 1, 10), date(2024, 1, 1), max_days=5
            )
            is not None
        )
        assert (
            check_prescription_too_far_after_care(
                date(2024, 1, 5), date(2024, 1, 1), max_days=5
            )
            is None
        )

    def test_default_tolerance_matches_constant(self):
        boundary = date(2024, 1, 1)
        just_over = boundary + timedelta(days=MAX_PRESCRIPTION_AFTER_CARE_DAYS + 1)
        assert check_prescription_too_far_after_care(just_over, boundary) is not None

    def test_missing_dates_no_signal(self):
        assert check_prescription_too_far_after_care(None, date(2024, 1, 1)) is None
        assert check_prescription_too_far_after_care(date(2024, 1, 1), None) is None


# ── check_missing_procedure_evidence ────────────────────────────────────────


class TestMissingProcedureEvidence:
    def test_coded_procedures_present_no_signal(self):
        signal = check_missing_procedure_evidence(care_date=date(2024, 1, 15), coded_count=2)
        assert signal is None

    def test_zero_coded_procedures_is_signaled(self):
        """« Acte absent » : un soin daté sans aucun acte codifié."""
        signal = check_missing_procedure_evidence(care_date=date(2024, 1, 15), coded_count=0)
        assert signal is not None
        assert signal.signal_type == "MISSING_PROCEDURE_EVIDENCE"
        assert signal.severity is SeverityLevel.MEDIUM
        sources = {e.source for e in signal.evidence}
        assert ClinicalEvidenceSource.MEDICAL_CODING in sources
        assert ClinicalEvidenceSource.OCR_EXTRACTION in sources

    def test_missing_care_date_no_signal(self):
        """Rien à dater : absence de date, pas une preuve d'acte manquante."""
        assert check_missing_procedure_evidence(care_date=None, coded_count=0) is None

    def test_coding_not_yet_available_no_signal(self):
        """coded_count=None : codification pas encore disponible, pas une
        preuve d'absence — jamais un signal sur une hypothèse."""
        assert check_missing_procedure_evidence(care_date=date(2024, 1, 15), coded_count=None) is None


# ── run_date_checks — cas composés ───────────────────────────────────────────


class TestRunDateChecks:
    def test_fully_valid_case_has_no_signals(self):
        signals = run_date_checks(
            prescription_date_raw="2024-01-15",
            care_date_raw="2024-01-15",
            coded_count=2,
        )
        assert signals == ()

    def test_impossible_date_is_the_only_signal_reported(self):
        signals = run_date_checks(
            prescription_date_raw="2024-02-30",
            care_date_raw="2024-01-15",
            coded_count=2,
        )
        assert len(signals) == 1
        assert signals[0].signal_type == "IMPOSSIBLE_DATE"

    def test_missing_procedure_evidence_is_reported(self):
        signals = run_date_checks(
            prescription_date_raw=None,
            care_date_raw="2024-01-15",
            coded_count=0,
        )
        assert len(signals) == 1
        assert signals[0].signal_type == "MISSING_PROCEDURE_EVIDENCE"

    def test_prescription_before_care_is_reported(self):
        signals = run_date_checks(
            prescription_date_raw="2024-01-10",
            care_date_raw="2024-01-15",
            coded_count=1,
        )
        assert any(s.signal_type == "PRESCRIPTION_BEFORE_CARE" for s in signals)
        assert not any(s.signal_type == "PRESCRIPTION_TOO_FAR_AFTER_CARE" for s in signals)

    def test_multiple_independent_signals_all_reported(self):
        """Une date impossible et une preuve d'acte manquante peuvent
        coexister — chacune signalée indépendamment, jamais masquée."""
        signals = run_date_checks(
            prescription_date_raw="2024-02-30",
            care_date_raw="2024-01-15",
            coded_count=0,
        )
        signal_types = {s.signal_type for s in signals}
        assert "IMPOSSIBLE_DATE" in signal_types
        assert "MISSING_PROCEDURE_EVIDENCE" in signal_types

    def test_no_business_status_is_ever_produced(self):
        """Garantie centrale : ce module ne retourne que des ClinicalSignal,
        jamais un statut PASS/NEEDS_REVIEW/FAIL ni une recommandation."""
        signals = run_date_checks(
            prescription_date_raw="2024-01-10",
            care_date_raw="2024-01-15",
            coded_count=0,
        )
        for signal in signals:
            assert not hasattr(signal, "status")
            assert not hasattr(signal, "recommendation")
