"""Tests unitaires de l'index de doublons — services/duplicate_index.py.

Couvre les trois familles de cas demandées : hash identique (doublon
exact), quasi-doublon (similarité/montant proches sans hash identique) et
historique absent (index vide — jamais un rapprochement inventé). Vérifie
aussi que la politique de seuils est versionnée/configurable et qu'aucun
champ de décision de fraude n'existe dans les schémas retournés.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from services.duplicate_index import (
    DEFAULT_DUPLICATE_POLICY,
    ClaimFingerprint,
    DuplicateCheckResult,
    DuplicateDetectionPolicy,
    DuplicateIndex,
    DuplicateMatch,
    DuplicateMatchType,
)


def _fingerprint(**overrides) -> ClaimFingerprint:
    payload = {
        "case_id": "CLM-0001",
        "document_hash": "a" * 64,
        "patient_pseudonym": "PAT-AAAAAAAAAAAA",
        "provider_pseudonym": "PRV-BBBBBBBBBBBB",
        "amount": Decimal("100.00"),
        "service_date": date(2024, 1, 15),
        "description": "consultation generale",
    }
    payload.update(overrides)
    return ClaimFingerprint(**payload)


# ── ClaimFingerprint — validation ────────────────────────────────────────────


class TestClaimFingerprintValidation:
    def test_accepts_minimal_valid_payload(self):
        fp = _fingerprint()
        assert fp.case_id == "CLM-0001"

    def test_forbids_unknown_fields(self):
        with pytest.raises(ValidationError):
            _fingerprint(unexpected="oops")

    def test_rejects_invalid_hash_format(self):
        with pytest.raises(ValidationError):
            _fingerprint(document_hash="not-a-hash")

    def test_rejects_short_hash(self):
        with pytest.raises(ValidationError):
            _fingerprint(document_hash="a" * 63)

    def test_rejects_invalid_patient_pseudonym_format(self):
        with pytest.raises(ValidationError):
            _fingerprint(patient_pseudonym="John Doe")

    def test_rejects_raw_patient_id_as_pseudonym(self):
        with pytest.raises(ValidationError):
            _fingerprint(patient_pseudonym="PATIENT-0007")

    def test_provider_pseudonym_is_optional(self):
        fp = _fingerprint(provider_pseudonym=None)
        assert fp.provider_pseudonym is None


# ── DuplicateDetectionPolicy — versionnée et configurable ────────────────────


class TestDuplicateDetectionPolicy:
    def test_default_policy_has_a_version(self):
        assert DEFAULT_DUPLICATE_POLICY.version

    def test_policy_is_configurable(self):
        policy = DuplicateDetectionPolicy(
            version="2.0.0",
            amount_tolerance_ratio=0.05,
            date_window_days=7,
            near_duplicate_score_threshold=0.7,
        )
        assert policy.version == "2.0.0"
        assert policy.amount_tolerance_ratio == 0.05

    def test_rejects_empty_version(self):
        with pytest.raises(ValueError):
            DuplicateDetectionPolicy(version="")

    def test_rejects_out_of_bounds_tolerance(self):
        with pytest.raises(ValueError):
            DuplicateDetectionPolicy(amount_tolerance_ratio=1.5)

    def test_rejects_non_positive_date_window(self):
        with pytest.raises(ValueError):
            DuplicateDetectionPolicy(date_window_days=0)

    def test_result_carries_policy_version(self):
        index = DuplicateIndex(policy=DuplicateDetectionPolicy(version="9.9.9"))
        result = index.check(_fingerprint())
        assert result.policy_version == "9.9.9"


# ── Historique absent ─────────────────────────────────────────────────────────


class TestNoHistory:
    def test_empty_index_has_no_matches(self):
        index = DuplicateIndex()
        result = index.check(_fingerprint())
        assert result.matches == []
        assert result.has_exact_duplicate is False
        assert result.has_near_duplicate is False

    def test_empty_index_never_raises(self):
        index = DuplicateIndex()
        assert isinstance(index.check(_fingerprint()), DuplicateCheckResult)

    def test_len_reflects_registered_count(self):
        index = DuplicateIndex()
        assert len(index) == 0
        index.register(_fingerprint())
        assert len(index) == 1


# ── Doublon exact (hash identique) ───────────────────────────────────────────


class TestExactDuplicate:
    def test_identical_hash_is_flagged_exact(self):
        index = DuplicateIndex()
        index.register(_fingerprint(case_id="CLM-0001"))

        result = index.check(_fingerprint(case_id="CLM-0002", document_hash="a" * 64))

        assert result.has_exact_duplicate is True
        match = result.matches[0]
        assert match.match_type is DuplicateMatchType.EXACT
        assert match.matched_case_id == "CLM-0001"
        assert match.similarity_score == 1.0

    def test_same_case_id_is_never_a_duplicate_of_itself(self):
        index = DuplicateIndex()
        fp = _fingerprint(case_id="CLM-0001")
        index.register(fp)
        result = index.check(fp)
        assert result.matches == []

    def test_different_hash_is_not_exact(self):
        index = DuplicateIndex()
        index.register(_fingerprint(case_id="CLM-0001", document_hash="a" * 64))
        result = index.check(_fingerprint(case_id="CLM-0002", document_hash="b" * 64))
        assert result.has_exact_duplicate is False

    def test_exact_duplicate_across_different_patients_still_flagged(self):
        """Un fichier strictement identique reste structurellement suspect
        même si les pseudonymes patients diffèrent — contrairement au
        quasi-doublon, l'exact ne filtre pas par patient."""
        index = DuplicateIndex()
        index.register(_fingerprint(case_id="CLM-0001", patient_pseudonym="PAT-AAAAAAAAAAAA"))
        result = index.check(
            _fingerprint(
                case_id="CLM-0002", document_hash="a" * 64, patient_pseudonym="PAT-CCCCCCCCCCCC"
            )
        )
        assert result.has_exact_duplicate is True


# ── Quasi-doublon (similarité/montant) ───────────────────────────────────────


class TestNearDuplicate:
    def test_close_amount_and_text_same_patient_is_near_duplicate(self):
        index = DuplicateIndex()
        index.register(
            _fingerprint(
                case_id="CLM-0001",
                document_hash="a" * 64,
                amount=Decimal("100.00"),
                service_date=date(2024, 1, 15),
                description="consultation generale",
            )
        )

        result = index.check(
            _fingerprint(
                case_id="CLM-0002",
                document_hash="b" * 64,
                amount=Decimal("100.50"),
                service_date=date(2024, 1, 16),
                description="consultation generale",
            )
        )

        assert result.has_near_duplicate is True
        match = result.matches[0]
        assert match.match_type is DuplicateMatchType.NEAR
        assert match.matched_case_id == "CLM-0001"
        assert 0.0 < match.similarity_score <= 1.0

    def test_different_patient_is_never_a_near_duplicate(self):
        """Une similarité de montant entre deux patients distincts n'est
        jamais une preuve de doublon — filtré avant tout calcul de score."""
        index = DuplicateIndex()
        index.register(
            _fingerprint(
                case_id="CLM-0001",
                document_hash="a" * 64,
                patient_pseudonym="PAT-AAAAAAAAAAAA",
                amount=Decimal("100.00"),
                description="consultation generale",
            )
        )
        result = index.check(
            _fingerprint(
                case_id="CLM-0002",
                document_hash="b" * 64,
                patient_pseudonym="PAT-CCCCCCCCCCCC",
                amount=Decimal("100.00"),
                description="consultation generale",
            )
        )
        assert result.has_near_duplicate is False

    def test_amount_outside_tolerance_is_not_a_near_duplicate(self):
        index = DuplicateIndex()
        index.register(_fingerprint(case_id="CLM-0001", document_hash="a" * 64, amount=Decimal("100.00")))
        result = index.check(
            _fingerprint(case_id="CLM-0002", document_hash="b" * 64, amount=Decimal("500.00"))
        )
        assert result.has_near_duplicate is False

    def test_unrelated_description_is_not_a_near_duplicate(self):
        index = DuplicateIndex()
        index.register(
            _fingerprint(
                case_id="CLM-0001",
                document_hash="a" * 64,
                amount=Decimal("100.00"),
                description="consultation generale",
            )
        )
        result = index.check(
            _fingerprint(
                case_id="CLM-0002",
                document_hash="b" * 64,
                amount=Decimal("100.00"),
                description="chirurgie orthopedique complexe genou",
            )
        )
        assert result.has_near_duplicate is False

    def test_date_far_outside_window_reduces_below_threshold(self):
        index = DuplicateIndex()
        index.register(
            _fingerprint(
                case_id="CLM-0001",
                document_hash="a" * 64,
                amount=Decimal("100.00"),
                service_date=date(2024, 1, 1),
                description="consultation generale",
            )
        )
        result = index.check(
            _fingerprint(
                case_id="CLM-0002",
                document_hash="b" * 64,
                amount=Decimal("100.00"),
                service_date=date(2024, 6, 1),
                description="consultation generale",
            )
        )
        assert result.has_near_duplicate is False

    def test_custom_policy_changes_sensitivity(self):
        """Une politique plus permissive (fenêtre large, seuil bas) détecte
        un rapprochement que la politique par défaut ignore."""
        strict_index = DuplicateIndex()
        loose_index = DuplicateIndex(
            policy=DuplicateDetectionPolicy(
                version="loose-1.0.0",
                date_window_days=30,
                near_duplicate_score_threshold=0.5,
            )
        )
        for index in (strict_index, loose_index):
            index.register(
                _fingerprint(
                    case_id="CLM-0001",
                    document_hash="a" * 64,
                    amount=Decimal("100.00"),
                    service_date=date(2024, 1, 1),
                    description="consultation generale",
                )
            )

        candidate = _fingerprint(
            case_id="CLM-0002",
            document_hash="b" * 64,
            amount=Decimal("100.00"),
            service_date=date(2024, 1, 10),
            description="consultation",
        )
        assert strict_index.check(candidate).has_near_duplicate is False
        assert loose_index.check(candidate).has_near_duplicate is True


# ── Aucune accusation de fraude ───────────────────────────────────────────────


class TestNoFraudAccusation:
    def test_duplicate_match_has_no_fraud_or_decision_field(self):
        forbidden_fields = {
            "is_fraud",
            "fraud_risk",
            "fraud_score",
            "recommendation",
            "decision",
            "status",
            "verdict",
        }
        assert forbidden_fields.isdisjoint(DuplicateMatch.model_fields.keys())

    def test_duplicate_check_result_has_no_fraud_or_decision_field(self):
        forbidden_fields = {
            "is_fraud",
            "fraud_risk",
            "fraud_score",
            "recommendation",
            "decision",
            "status",
            "verdict",
        }
        assert forbidden_fields.isdisjoint(DuplicateCheckResult.model_fields.keys())

    def test_forbids_unknown_fields_on_match(self):
        with pytest.raises(ValidationError):
            DuplicateMatch(
                match_type=DuplicateMatchType.EXACT,
                matched_case_id="CLM-0001",
                similarity_score=1.0,
                amount_similarity=1.0,
                text_similarity=1.0,
                date_proximity=1.0,
                policy_version="1.0.0",
                is_fraud=True,
            )

    def test_result_is_json_serializable(self):
        index = DuplicateIndex()
        index.register(_fingerprint(case_id="CLM-0001"))
        result = index.check(_fingerprint(case_id="CLM-0002", document_hash="a" * 64))
        assert result.model_dump_json()
