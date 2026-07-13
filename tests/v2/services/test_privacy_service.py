"""Tests de services/privacy_service.py (V2) — Phase V2-0.

Différentiel contre agents/privacy_agent/views.py (oracle V1, non modifié) :
le service V2 délègue intentionnellement aux builders déjà purs de V1 (voir
docstring de services/privacy_service.py) — l'égalité prouve que le service
ne réintroduit aucune divergence de comportement par rapport à V1.
"""
from __future__ import annotations

import pytest

from agents.privacy_agent.views import build_view as v1_build_view
from schemas.domain import PrivacyCode, ReaderRole, VerificationStatus
from services.privacy_service import PrivacyService

_VALID_SHA256 = "a" * 64


def _sample_claim_data() -> dict:
    return {
        "patient_id": "P-123456",
        "dossier_status": "ACCEPTED",
        "present_documents": ["facture.pdf", "ordonnance.pdf"],
        "missing_documents": [],
        "submitted_at": "2024-01-15",
        "service_date": "2024-01-10",
        "total_billed": "1500.00",
        "amount_requested": "1200.00",
        "patient_share": "300.00",
        "coverage_rate": "0.80",
        "payer_name": "Assurance XYZ",
        "invoice_number": "FAC-12345",
        "provider_id": "PRV-98765",
        "procedures": ["Consultation générale", "Bilan sanguin"],
        "prescription_names": ["Paracétamol", "Amoxicilline"],
        "diagnosis_codes": ["J06.9", "Z00.0"],
        "encounter_class": "ambulatory",
        "document_hashes": {"facture.pdf": _VALID_SHA256},
        "actor": "intake_safety_agent",
        "actor_role": "ADMINISTRATIVE_MANAGER",
        "action": "privacy_evaluation",
        "timestamp": "2024-01-15T10:00:00Z",
        "policy_version": "1.1.0",
        "outcome": "PASS",
        "reason_codes": [],
    }


class TestParityWithV1Views:
    """Le service V2 délègue à agents.privacy_agent.views — égalité attendue."""

    @pytest.mark.parametrize(
        "role",
        [
            ReaderRole.ADMINISTRATIVE_MANAGER,
            ReaderRole.MEDICAL_REVIEWER,
            ReaderRole.FRAUD_ANALYST,
            ReaderRole.AUDITOR,
        ],
    )
    def test_view_matches_v1_oracle(self, role):
        data = _sample_claim_data()
        result = PrivacyService().build_view(case_id="CLM-0001", role=role, claim_data=data)
        oracle_view = v1_build_view(role, "CLM-0001", data)

        assert result.status == VerificationStatus.PASS
        assert result.view == oracle_view


class TestDenyByDefault:
    def test_administrative_view_excludes_medical_fields(self):
        data = _sample_claim_data()
        result = PrivacyService().build_view(
            case_id="CLM-0001", role=ReaderRole.ADMINISTRATIVE_MANAGER, claim_data=data
        )
        assert result.view is not None
        assert "procedures" not in result.view
        assert "diagnosis_codes" not in result.view

    def test_medical_view_excludes_financial_fields(self):
        data = _sample_claim_data()
        result = PrivacyService().build_view(
            case_id="CLM-0001", role=ReaderRole.MEDICAL_REVIEWER, claim_data=data
        )
        assert result.view is not None
        assert "total_billed" not in result.view
        assert "invoice_reference" not in result.view

    def test_auditor_view_is_minimal(self):
        data = _sample_claim_data()
        result = PrivacyService().build_view(
            case_id="CLM-0001", role=ReaderRole.AUDITOR, claim_data=data
        )
        assert result.view is not None
        assert "patient_pseudonym" not in result.view

    def test_redacted_fields_never_empty_for_restricted_role(self):
        result = PrivacyService().build_view(
            case_id="CLM-0001", role=ReaderRole.AUDITOR, claim_data=_sample_claim_data()
        )
        assert len(result.redacted_fields) > 0


class TestPseudonymization:
    def test_patient_pseudonym_has_pat_prefix(self):
        data = _sample_claim_data()
        result = PrivacyService().build_view(
            case_id="CLM-0001", role=ReaderRole.MEDICAL_REVIEWER, claim_data=data
        )
        assert result.view["patient_pseudonym"].startswith("PAT-")

    def test_same_patient_id_produces_stable_pseudonym(self):
        data = _sample_claim_data()
        r1 = PrivacyService().build_view(
            case_id="CLM-0001", role=ReaderRole.FRAUD_ANALYST, claim_data=data
        )
        r2 = PrivacyService().build_view(
            case_id="CLM-0002", role=ReaderRole.FRAUD_ANALYST, claim_data=data
        )
        assert r1.view["patient_pseudonym"] == r2.view["patient_pseudonym"]

    def test_no_raw_patient_id_in_any_view(self):
        data = _sample_claim_data()
        for role in ReaderRole:
            result = PrivacyService().build_view(case_id="CLM-0001", role=role, claim_data=data)
            assert result.view is not None
            assert "P-123456" not in str(result.view)


class TestMissingPseudonymizationKey:
    def test_missing_key_blocks_with_fail(self, monkeypatch):
        monkeypatch.setattr(
            "services.privacy_service.pseudonymization_key_is_available", lambda: False
        )
        result = PrivacyService().build_view(
            case_id="CLM-0001",
            role=ReaderRole.ADMINISTRATIVE_MANAGER,
            claim_data=_sample_claim_data(),
        )
        assert result.status == VerificationStatus.FAIL
        assert PrivacyCode.MISSING_PSEUDONYMIZATION_KEY in result.reason_codes
        assert result.view is None


class TestNeedsReviewOnRealPersonalData:
    def test_real_personal_data_needs_review_but_view_still_built(self):
        result = PrivacyService().build_view(
            case_id="CLM-0001",
            role=ReaderRole.ADMINISTRATIVE_MANAGER,
            claim_data=_sample_claim_data(),
            contains_real_personal_data=True,
        )
        assert result.status == VerificationStatus.NEEDS_REVIEW
        assert result.view is not None


class TestPostViewLeakDetection:
    def test_secret_field_in_view_blocks(self, monkeypatch):
        def _fake_build_view(role, case_id, data):  # noqa: ARG001
            return {"patient_pseudonym": "PAT-ABCDEF123456", "api_key": "should-never-appear"}

        monkeypatch.setattr("services.privacy_service.build_view", _fake_build_view)
        result = PrivacyService().build_view(
            case_id="CLM-0001",
            role=ReaderRole.MEDICAL_REVIEWER,
            claim_data=_sample_claim_data(),
        )
        assert result.status == VerificationStatus.FAIL
        assert PrivacyCode.FORBIDDEN_FIELD_EXPOSED in result.reason_codes
        assert result.view is None


class TestNoLlmDependency:
    """Garantie structurelle : ce service n'importe jamais de module LLM."""

    def test_module_source_has_no_llm_or_langgraph_reference(self):
        import services.privacy_service as mod

        with open(mod.__file__, encoding="utf-8") as f:
            content = f.read()
        assert "import llm" not in content
        assert "langgraph" not in content
        assert "ChatOllama" not in content
