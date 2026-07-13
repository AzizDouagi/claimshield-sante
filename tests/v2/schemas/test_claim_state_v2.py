"""Tests de state/claim_state_v2.py (V2) — Phase V2-1."""
from __future__ import annotations

import pytest

from schemas.domain import ClaimDecisionV2, IntakeSafetyStatus
from schemas.results import LlmMetadata
from schemas.v2_results import IntakeSafetyResult
from state.claim_state_v2 import ClaimStateV2, validate_claim_state_v2, validate_state_update_v2


def _base_state(**overrides) -> dict:
    state: dict = {
        "case_id": "CLM-0001",
        "schema_version": "2.0.0",
        "current_step": "initial",
        "completed_steps": [],
    }
    state.update(overrides)
    return state


class TestClaimStateV2Fields:
    def test_no_human_decision_field(self):
        """La V2 ne bloque jamais — aucun champ human_decision, contrairement à ClaimState (V1)."""
        assert "human_decision" not in ClaimStateV2.__annotations__

    def test_only_five_result_fields(self):
        result_fields = {k for k in ClaimStateV2.__annotations__ if k.endswith("_result")}
        assert result_fields == {
            "intake_safety_result",
            "document_understanding_result",
            "eligibility_result",
            "medical_risk_result",
            "decision_result",
        }

    def test_only_one_consumed_input_field(self):
        input_fields = {k for k in ClaimStateV2.__annotations__ if k.endswith("_input")}
        assert input_fields == {"intake_input"}


class TestValidateStateUpdateV2:
    def test_valid_update_passes(self):
        validate_state_update_v2({"current_step": "document_understanding", "completed_steps": ["intake_safety"]})

    def test_absolute_path_rejected(self):
        with pytest.raises(ValueError, match="chemin absolu"):
            validate_state_update_v2({"current_step": "/etc/passwd"})

    def test_secret_hint_rejected(self):
        with pytest.raises(ValueError, match="secret"):
            validate_state_update_v2({"alerts": ["api_key: sk-should-never-appear-here"]})

    def test_raw_document_key_rejected(self):
        with pytest.raises(ValueError, match="document brut"):
            validate_state_update_v2({"extraction": {"full_text": "contenu OCR complet"}})

    def test_bytes_rejected(self):
        with pytest.raises(ValueError, match="contenu binaire"):
            validate_state_update_v2({"blob": b"raw-bytes"})

    def test_consumed_input_none_is_ignored(self):
        validate_state_update_v2({"intake_input": None})

    def test_pydantic_model_content_is_scanned(self):
        result = IntakeSafetyResult(
            case_id="CLM-0001",
            status=IntakeSafetyStatus.ACCEPTED,
            reasons=["motif propre"],
            llm_trace=LlmMetadata(model_name="gemma4:latest", prompt_version="1.0.0"),
        )
        validate_state_update_v2({"intake_safety_result": result})


class TestValidateClaimStateV2:
    def test_valid_full_state(self):
        validate_claim_state_v2(_base_state(errors=[], alerts=[], audit_trail=[]))

    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="champs inconnus"):
            validate_claim_state_v2(_base_state(human_decision={"action": "APPROVE"}))

    def test_missing_required_key_rejected(self):
        with pytest.raises(ValueError, match="obligatoires manquants"):
            validate_claim_state_v2({"case_id": "CLM-0001"})

    def test_invalid_final_decision_rejected(self):
        with pytest.raises(ValueError, match="final_decision"):
            validate_claim_state_v2(_base_state(final_decision="NOT_A_REAL_DECISION"))

    def test_valid_final_decision_accepted(self):
        validate_claim_state_v2(_base_state(final_decision=ClaimDecisionV2.APPROVE.value))

    def test_invalid_result_model_rejected(self):
        with pytest.raises(ValueError, match="intake_safety_result"):
            validate_claim_state_v2(_base_state(intake_safety_result={"not": "a valid result"}))

    def test_valid_result_model_as_dict_accepted(self):
        result = IntakeSafetyResult(
            case_id="CLM-0001",
            status=IntakeSafetyStatus.ACCEPTED,
            reasons=["motif propre"],
            llm_trace=LlmMetadata(model_name="gemma4:latest", prompt_version="1.0.0"),
        )
        validate_claim_state_v2(_base_state(intake_safety_result=result.model_dump(mode="json")))
