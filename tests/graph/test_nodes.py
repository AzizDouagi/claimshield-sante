"""Tests unitaires des adaptateurs de nœuds — graph/nodes.py.

Stratégie : les agents sont mockés via ``unittest.mock.patch`` sur leur
module réel (``agents.<pkg>.agent.node``).  Aucun agent n'est exécuté ;
les tests vérifient uniquement le comportement de la couche adaptateur :
délégation, isolation d'exceptions et validation du type de résultat.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from graph.nodes import (
    NODE_REGISTRY,
    _AgentConfig,
    _exception_fallback,
    _validate_result_type,
    node_claim_intake,
    node_document_ocr,
    node_fhir_validator,
    node_identity_coverage,
    node_medical_coding,
    node_privacy,
    node_security_gate,
)
from schemas.results import (
    ClaimIntakeResult,
    DocumentOcrResult,
    FhirValidatorResult,
    IdentityCoverageResult,
    MedicalCodingResult,
    PrivacyResult,
    SecurityGateResult,
)


# ── Fixtures et helpers ────────────────────────────────────────────────────────


def _state(case_id: str = "CLM-0001") -> dict:
    return {"case_id": case_id}


def _valid_update(result_key: str, model_class: type) -> dict:
    """Dict de mise à jour simulé par un agent qui réussit."""
    return {
        result_key: MagicMock(spec=model_class),
        "completed_steps": ["step"],
        "current_step": "step",
    }


# ── 1. Helpers internes ────────────────────────────────────────────────────────


class TestValidateResultType:
    def test_none_is_accepted(self):
        _validate_result_type({}, "key", ClaimIntakeResult)  # pas d'exception

    def test_correct_instance_passes(self):
        mock = MagicMock(spec=ClaimIntakeResult)
        _validate_result_type({"key": mock}, "key", ClaimIntakeResult)

    def test_wrong_type_raises_type_error(self):
        with pytest.raises(TypeError, match="invalide"):
            _validate_result_type({"key": "bad_value"}, "key", ClaimIntakeResult)

    def test_wrong_type_name_appears_in_message(self):
        with pytest.raises(TypeError) as exc_info:
            _validate_result_type({"key": 42}, "key", SecurityGateResult)
        assert "int" in str(exc_info.value)
        assert "SecurityGateResult" in str(exc_info.value)

    def test_key_absent_is_accepted(self):
        _validate_result_type({"other_key": "value"}, "key", ClaimIntakeResult)

    def test_invalid_dict_raises_validation_error(self):
        with pytest.raises(ValidationError):
            _validate_result_type(
                {"key": {"bad": "structure"}},
                "key",
                SecurityGateResult,
            )


class TestExceptionFallback:
    CFG = _AgentConfig(
        agent_name="test_agent",
        step_name="test_step",
        result_key="test_result",
        result_model=ClaimIntakeResult,
        input_key="test_input",
    )

    def test_returns_dict(self):
        result = _exception_fallback(_state(), self.CFG, RuntimeError("boom"))
        assert isinstance(result, dict)

    def test_errors_list_populated(self):
        result = _exception_fallback(_state(), self.CFG, RuntimeError("boom"))
        assert len(result["errors"]) == 1

    def test_error_contains_agent_name(self):
        result = _exception_fallback(_state(), self.CFG, RuntimeError("boom"))
        assert "test_agent" in result["errors"][0]

    def test_error_contains_exception_type(self):
        result = _exception_fallback(_state(), self.CFG, ValueError("bad"))
        assert "ValueError" in result["errors"][0]

    def test_error_contains_exception_message(self):
        result = _exception_fallback(_state(), self.CFG, RuntimeError("detail error"))
        assert "detail error" in result["errors"][0]

    def test_input_key_set_to_none(self):
        result = _exception_fallback(_state(), self.CFG, RuntimeError("x"))
        assert result["test_input"] is None

    def test_completed_steps_set(self):
        result = _exception_fallback(_state(), self.CFG, RuntimeError("x"))
        assert "test_step" in result["completed_steps"]

    def test_current_step_set(self):
        result = _exception_fallback(_state(), self.CFG, RuntimeError("x"))
        assert result["current_step"] == "test_step"

    def test_no_input_key_config(self):
        cfg = _AgentConfig(
            agent_name="no_input",
            step_name="step",
            result_key="result",
            result_model=ClaimIntakeResult,
            input_key=None,
        )
        result = _exception_fallback(_state(), cfg, RuntimeError("x"))
        assert "test_input" not in result
        assert "errors" in result


# ── 2. node_claim_intake ──────────────────────────────────────────────────────


class TestNodeClaimIntake:
    PATCH = "agents.claim_intake_agent.agent.node"

    def test_delegates_to_agent(self):
        expected = _valid_update("intake_result", ClaimIntakeResult)
        with patch(self.PATCH, return_value=expected) as mock_node:
            result = node_claim_intake(_state())
        mock_node.assert_called_once()
        assert result is expected

    def test_passes_state_to_agent(self):
        state = _state("CLM-0099")
        expected = _valid_update("intake_result", ClaimIntakeResult)
        with patch(self.PATCH, return_value=expected):
            node_claim_intake(state)

    def test_agent_exception_returns_structured_error(self):
        with patch(self.PATCH, side_effect=RuntimeError("storage down")):
            result = node_claim_intake(_state())
        assert result["errors"]
        assert "claim_intake" in result["errors"][0]
        assert "storage down" in result["errors"][0]
        assert result["intake_input"] is None

    def test_wrong_result_type_returns_structured_error(self):
        bad_update = {"intake_result": "not_a_model", "intake_input": None}
        with patch(self.PATCH, return_value=bad_update):
            result = node_claim_intake(_state())
        assert result["errors"]
        assert "claim_intake" in result["errors"][0]

    def test_completed_steps_in_fallback(self):
        with patch(self.PATCH, side_effect=Exception("crash")):
            result = node_claim_intake(_state())
        assert "claim_intake" in result["completed_steps"]


# ── 3. node_security_gate ─────────────────────────────────────────────────────


class TestNodeSecurityGate:
    PATCH = "agents.security_gate_agent.agent.node"

    def test_delegates_to_agent(self):
        expected = _valid_update("security_result", SecurityGateResult)
        with patch(self.PATCH, return_value=expected) as mock_node:
            result = node_security_gate(_state())
        mock_node.assert_called_once()
        assert result is expected

    def test_agent_exception_returns_structured_error(self):
        with patch(self.PATCH, side_effect=RuntimeError("scan failure")):
            result = node_security_gate(_state())
        assert result["errors"]
        assert "security_gate" in result["errors"][0]
        assert result["security_input"] is None

    def test_wrong_result_type_returns_structured_error(self):
        with patch(self.PATCH, return_value={"security_result": 123}):
            result = node_security_gate(_state())
        assert result["errors"]


# ── 4. node_privacy ───────────────────────────────────────────────────────────


class TestNodePrivacy:
    PATCH = "agents.privacy_agent.agent.node"

    def test_delegates_to_agent(self):
        expected = _valid_update("privacy_result", PrivacyResult)
        with patch(self.PATCH, return_value=expected) as mock_node:
            result = node_privacy(_state())
        mock_node.assert_called_once()
        assert result is expected

    def test_agent_exception_returns_structured_error(self):
        with patch(self.PATCH, side_effect=KeyError("missing_key")):
            result = node_privacy(_state())
        assert result["errors"]
        assert "privacy" in result["errors"][0]
        assert result["privacy_input"] is None

    def test_wrong_result_type_returns_structured_error(self):
        with patch(self.PATCH, return_value={"privacy_result": []}):
            result = node_privacy(_state())
        assert result["errors"]


# ── 5. node_fhir_validator ────────────────────────────────────────────────────


class TestNodeFhirValidator:
    PATCH = "agents.fhir_validator_agent.agent.node"

    def test_delegates_to_agent(self):
        expected = _valid_update("fhir_result", FhirValidatorResult)
        with patch(self.PATCH, return_value=expected) as mock_node:
            result = node_fhir_validator(_state())
        mock_node.assert_called_once()
        assert result is expected

    def test_agent_exception_returns_structured_error(self):
        with patch(self.PATCH, side_effect=OSError("file not found")):
            result = node_fhir_validator(_state())
        assert result["errors"]
        assert "fhir_validator" in result["errors"][0]
        assert result["fhir_input"] is None

    def test_wrong_result_type_returns_structured_error(self):
        with patch(self.PATCH, return_value={"fhir_result": object()}):
            result = node_fhir_validator(_state())
        assert result["errors"]

    def test_completed_steps_in_fallback(self):
        with patch(self.PATCH, side_effect=Exception("x")):
            result = node_fhir_validator(_state())
        assert "fhir_validation" in result["completed_steps"]


# ── 6. node_medical_coding ────────────────────────────────────────────────────


class TestNodeMedicalCoding:
    PATCH = "agents.medical_coding_agent.agent.node"

    def test_delegates_to_agent(self):
        expected = _valid_update("coding_result", MedicalCodingResult)
        with patch(self.PATCH, return_value=expected) as mock_node:
            result = node_medical_coding(_state())
        mock_node.assert_called_once()
        assert result is expected

    def test_agent_exception_returns_structured_error(self):
        with patch(self.PATCH, side_effect=ValueError("bad codes")):
            result = node_medical_coding(_state())
        assert result["errors"]
        assert "medical_coding" in result["errors"][0]
        assert result["coding_input"] is None

    def test_wrong_result_type_returns_structured_error(self):
        with patch(self.PATCH, return_value={"coding_result": False}):
            result = node_medical_coding(_state())
        assert result["errors"]


# ── 7. node_document_ocr ─────────────────────────────────────────────────────


class TestNodeDocumentOcr:
    PATCH = "agents.document_ocr_agent.agent.node"

    def test_delegates_to_agent(self):
        expected = _valid_update("ocr_result", DocumentOcrResult)
        with patch(self.PATCH, return_value=expected) as mock_node:
            result = node_document_ocr(_state())
        mock_node.assert_called_once()
        assert result is expected

    def test_agent_exception_returns_structured_error(self):
        with patch(self.PATCH, side_effect=MemoryError("OOM")):
            result = node_document_ocr(_state())
        assert result["errors"]
        assert "document_ocr" in result["errors"][0]
        assert result["ocr_input"] is None

    def test_wrong_result_type_returns_structured_error(self):
        with patch(self.PATCH, return_value={"ocr_result": "text_string"}):
            result = node_document_ocr(_state())
        assert result["errors"]

    def test_exception_message_preserved(self):
        with patch(self.PATCH, side_effect=RuntimeError("tesseract unavailable")):
            result = node_document_ocr(_state())
        assert "tesseract unavailable" in result["errors"][0]


# ── 8. node_identity_coverage ─────────────────────────────────────────────────


class TestNodeIdentityCoverage:
    PATCH = "agents.identity_coverage_agent.agent.node"

    def test_delegates_to_agent(self):
        expected = _valid_update("identity_coverage_result", IdentityCoverageResult)
        with patch(self.PATCH, return_value=expected) as mock_node:
            result = node_identity_coverage(_state())
        mock_node.assert_called_once()
        assert result is expected

    def test_agent_exception_returns_structured_error(self):
        with patch(self.PATCH, side_effect=TimeoutError("FHIR timeout")):
            result = node_identity_coverage(_state())
        assert result["errors"]
        assert "identity_coverage" in result["errors"][0]
        assert result["identity_coverage_input"] is None

    def test_wrong_result_type_returns_structured_error(self):
        with patch(self.PATCH, return_value={"identity_coverage_result": {}}):
            result = node_identity_coverage(_state())
        assert result["errors"]


# ── 9. Registre NODE_REGISTRY ─────────────────────────────────────────────────


class TestNodeRegistry:
    EXPECTED_KEYS = {
        "claim_intake",
        "security_gate",
        "privacy",
        "fhir_validator",
        "medical_coding",
        "document_ocr",
        "identity_coverage",
        # Agents avec interface injectable (stubs)
        "clinical_consistency",
        "fraud_detection",
        "case_reviewer",
        "audit",
    }

    def test_registry_contains_all_expected_keys(self):
        assert set(NODE_REGISTRY.keys()) == self.EXPECTED_KEYS

    def test_all_entries_are_callable(self):
        for name, fn in NODE_REGISTRY.items():
            assert callable(fn), f"{name} n'est pas callable"

    @pytest.mark.parametrize("key", list(EXPECTED_KEYS))
    def test_each_node_callable_on_empty_state(self, key):
        fn = NODE_REGISTRY[key]
        patch_target = {
            "claim_intake": "agents.claim_intake_agent.agent.node",
            "security_gate": "agents.security_gate_agent.agent.node",
            "privacy": "agents.privacy_agent.agent.node",
            "fhir_validator": "agents.fhir_validator_agent.agent.node",
            "medical_coding": "agents.medical_coding_agent.agent.node",
            "document_ocr": "agents.document_ocr_agent.agent.node",
            "identity_coverage": "agents.identity_coverage_agent.agent.node",
            "clinical_consistency": "agents.clinical_consistency_agent.agent.node",
            "fraud_detection": "agents.fraud_detection_agent.agent.node",
            "case_reviewer": "agents.case_reviewer_agent.agent.node",
            "audit": "agents.audit_agent.agent.node",
        }[key]
        with patch(patch_target, side_effect=Exception("forced")):
            result = fn({})
        assert "errors" in result
        assert isinstance(result["errors"], list)

    def test_node_names_match_registry_keys(self):
        for key, fn in NODE_REGISTRY.items():
            assert fn.__name__ == f"node_{key}", (
                f"{key}: __name__ attendu 'node_{key}', obtenu '{fn.__name__}'"
            )


# ── 10. Invariants transversaux ───────────────────────────────────────────────


class TestNodeWrapperInvariants:
    """Propriétés communes à tous les nœuds quel que soit l'agent."""

    NODE_FNS = [
        ("claim_intake", node_claim_intake, "agents.claim_intake_agent.agent.node"),
        ("security_gate", node_security_gate, "agents.security_gate_agent.agent.node"),
        ("privacy", node_privacy, "agents.privacy_agent.agent.node"),
        ("fhir_validator", node_fhir_validator, "agents.fhir_validator_agent.agent.node"),
        ("medical_coding", node_medical_coding, "agents.medical_coding_agent.agent.node"),
        ("document_ocr", node_document_ocr, "agents.document_ocr_agent.agent.node"),
        ("identity_coverage", node_identity_coverage, "agents.identity_coverage_agent.agent.node"),
    ]

    @pytest.mark.parametrize("name,fn,patch_path", NODE_FNS)
    def test_exception_never_propagates(self, name, fn, patch_path):
        with patch(patch_path, side_effect=RuntimeError("crash")):
            result = fn(_state())
        assert isinstance(result, dict), f"{name}: doit retourner un dict"

    @pytest.mark.parametrize("name,fn,patch_path", NODE_FNS)
    def test_errors_is_a_list_when_exception(self, name, fn, patch_path):
        with patch(patch_path, side_effect=Exception("any")):
            result = fn(_state())
        assert isinstance(result.get("errors"), list)
        assert len(result["errors"]) >= 1

    @pytest.mark.parametrize("name,fn,patch_path", NODE_FNS)
    def test_completed_steps_is_list_when_exception(self, name, fn, patch_path):
        with patch(patch_path, side_effect=Exception("any")):
            result = fn(_state())
        assert isinstance(result.get("completed_steps"), list)
        assert len(result["completed_steps"]) >= 1

    @pytest.mark.parametrize("name,fn,patch_path", NODE_FNS)
    def test_happy_path_returns_agent_dict_unchanged(self, name, fn, patch_path):
        result_model = {
            "claim_intake": ClaimIntakeResult,
            "security_gate": SecurityGateResult,
            "privacy": PrivacyResult,
            "fhir_validator": FhirValidatorResult,
            "medical_coding": MedicalCodingResult,
            "document_ocr": DocumentOcrResult,
            "identity_coverage": IdentityCoverageResult,
        }[name]
        result_key = {
            "claim_intake": "intake_result",
            "security_gate": "security_result",
            "privacy": "privacy_result",
            "fhir_validator": "fhir_result",
            "medical_coding": "coding_result",
            "document_ocr": "ocr_result",
            "identity_coverage": "identity_coverage_result",
        }[name]
        expected = _valid_update(result_key, result_model)
        with patch(patch_path, return_value=expected):
            result = fn(_state())
        assert result is expected
