"""Tests unitaires de la détection générique de désaccords — tools/consistency.py.

Les résultats d'agents sont simulés via ``types.SimpleNamespace`` (seul le
champ ``status`` est lu) — aucun agent réel, aucune logique clinique ou
anti-fraude n'est exercée : ce module ne compare qu'un champ structurel
générique commun à plusieurs schémas de résultat.
"""
from __future__ import annotations

from types import SimpleNamespace

from schemas.results import DisagreementPoint
from tools.consistency import (
    GENERIC_STATUS_FIELDS,
    classify_disagreement_severity,
    detect_result_disagreements,
    has_critical_disagreement,
)


def _state(**statuses: str) -> dict:
    """Construit un state minimal : un ``SimpleNamespace(status=...)`` par
    champ ``ClaimState`` nommé (ex. ``fhir_result="PASS"``)."""
    return {field: SimpleNamespace(status=status) for field, status in statuses.items()}


# ── 1. Accord — aucun désaccord signalé ──────────────────────────────────────


class TestAgreement:
    def test_all_pass_no_disagreement(self):
        state = _state(privacy_result="PASS", fhir_result="PASS", coding_result="PASS")
        disagreements = detect_result_disagreements(state)
        assert disagreements == ()
        assert has_critical_disagreement(disagreements) is False

    def test_all_fail_no_disagreement(self):
        """Un échec partagé par tous les résultats n'est pas un désaccord —
        ils sont d'accord (sur un échec)."""
        state = _state(fhir_result="FAIL", coding_result="FAIL")
        disagreements = detect_result_disagreements(state)
        assert disagreements == ()

    def test_single_result_no_disagreement_possible(self):
        """Un seul résultat générique disponible : rien à comparer."""
        state = _state(fhir_result="PASS")
        assert detect_result_disagreements(state) == ()

    def test_no_result_at_all_no_disagreement(self):
        assert detect_result_disagreements({}) == ()

    def test_pending_and_not_evaluated_ignored_not_a_disagreement(self):
        """Un agent qui n'a pas encore tranché (PENDING/NOT_EVALUATED) n'est
        pas en désaccord avec les autres — il n'a simplement pas d'avis."""
        state = _state(
            fhir_result="PASS",
            coding_result="NOT_EVALUATED",
            clinical_result="PENDING",
        )
        assert detect_result_disagreements(state) == ()

    def test_unrecognized_status_value_ignored(self):
        """Un statut hors énumération VerificationStatus est ignoré (rien à
        comparer), jamais interprété comme un désaccord silencieux."""
        state = _state(fhir_result="PASS", coding_result="GARBAGE")
        assert detect_result_disagreements(state) == ()


# ── 2. Désaccord mineur — signalé mais non bloquant ──────────────────────────


class TestMinorDisagreement:
    def test_pass_vs_needs_review_is_minor(self):
        state = _state(fhir_result="PASS", coding_result="NEEDS_REVIEW")
        disagreements = detect_result_disagreements(state)
        assert len(disagreements) == 1
        point = disagreements[0]
        assert isinstance(point, DisagreementPoint)
        assert point.field == "status"
        assert point.expected == "PASS"
        assert point.observed == "NEEDS_REVIEW"
        assert classify_disagreement_severity(point) == "minor"
        assert has_critical_disagreement(disagreements) is False

    def test_needs_review_vs_fail_is_minor(self):
        state = _state(ocr_result="NEEDS_REVIEW", fhir_result="FAIL")
        disagreements = detect_result_disagreements(state)
        assert len(disagreements) == 1
        assert classify_disagreement_severity(disagreements[0]) == "minor"
        assert has_critical_disagreement(disagreements) is False

    def test_disagreement_references_the_dissenting_agent(self):
        """Le désaccord référence l'agent qui s'écarte de la référence,
        jamais un arbitrage sur celui qui a raison."""
        state = _state(privacy_result="PASS", coding_result="NEEDS_REVIEW")
        disagreements = detect_result_disagreements(state)
        assert disagreements[0].agent == "medical_coding"


# ── 3. Désaccord critique — signalé, référencé, jamais résolu ───────────────


class TestCriticalDisagreement:
    def test_pass_vs_fail_is_critical(self):
        state = _state(fhir_result="PASS", coding_result="FAIL")
        disagreements = detect_result_disagreements(state)
        assert len(disagreements) == 1
        point = disagreements[0]
        assert point.expected == "PASS"
        assert point.observed == "FAIL"
        assert classify_disagreement_severity(point) == "critical"
        assert has_critical_disagreement(disagreements) is True

    def test_critical_disagreement_carries_full_references(self):
        """Les références (agent, champ, valeurs) doivent être exploitables
        telles quelles par une revue humaine — jamais une simple alerte
        sans contexte."""
        state = _state(privacy_result="PASS", fraud_result="FAIL")
        disagreements = detect_result_disagreements(state)
        assert len(disagreements) == 1
        point = disagreements[0]
        assert point.agent == "fraud_detection"
        assert point.field == "status"
        assert point.expected == "PASS"
        assert point.observed == "FAIL"

    def test_mixed_minor_and_critical_reports_all_points(self):
        """Plusieurs désaccords simultanés (un mineur, un critique) sont
        tous signalés — aucun n'est masqué par un autre."""
        state = _state(
            privacy_result="PASS",
            coding_result="NEEDS_REVIEW",
            fhir_result="FAIL",
        )
        disagreements = detect_result_disagreements(state)
        assert len(disagreements) == 2
        severities = {classify_disagreement_severity(p) for p in disagreements}
        assert severities == {"minor", "critical"}
        assert has_critical_disagreement(disagreements) is True

    def test_unrecognized_status_value_classified_critical_by_default(self):
        """Un désaccord dont l'une des valeurs échappe à l'énumération
        connue est traité par prudence comme critique, jamais ignoré."""
        point = DisagreementPoint(
            agent="fhir_validator", field="status", expected="PASS", observed="GARBAGE"
        )
        assert classify_disagreement_severity(point) == "critical"


# ── 4. Jamais d'arbitrage automatique ────────────────────────────────────────


class TestNeverPicksAWinner:
    def test_detection_never_mutates_source_results(self):
        """La détection ne modifie jamais les objets résultat source."""
        fhir_result = SimpleNamespace(status="PASS")
        coding_result = SimpleNamespace(status="FAIL")
        state = {"fhir_result": fhir_result, "coding_result": coding_result}

        detect_result_disagreements(state)

        assert fhir_result.status == "PASS"
        assert coding_result.status == "FAIL"

    def test_disagreement_point_never_contains_a_verdict_field(self):
        """Le schéma DisagreementPoint n'expose que agent/field/expected/
        observed — aucun champ de type 'correct'/'winner' qui laisserait
        penser qu'un arbitrage a eu lieu."""
        assert set(DisagreementPoint.model_fields) == {"agent", "field", "expected", "observed"}

    def test_generic_status_fields_exclude_domain_specific_schemas(self):
        """Les schémas dont l'enum de statut est spécifique à un domaine
        (IntakeStatus, SecurityDecision, Recommendation) ou dont le statut
        est imbriqué (identity_coverage_result) sont hors périmètre —
        jamais une règle d'équivalence métier inventée entre ces enums."""
        assert "intake_result" not in GENERIC_STATUS_FIELDS
        assert "security_result" not in GENERIC_STATUS_FIELDS
        assert "identity_coverage_result" not in GENERIC_STATUS_FIELDS
        assert "review_result" not in GENERIC_STATUS_FIELDS
