"""Tests des contrats d'appel d'un agent — orchestrator/orchestrator.py.

Couvre : une requête valide, une sortie valide (succès et échec), et
plusieurs scénarios invalides par contrainte (case_id, modèle demandé
incohérent, contexte autorisé hors ClaimState, invariant succès/erreur,
payload incohérent avec le modèle de l'agent).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchestrator.orchestrator import (
    AGENT_RESULT_MODELS,
    AgentCallOutcome,
    AgentCallRequest,
    AgentName,
    AgentResultValidationError,
    validate_agent_result,
)
from schemas.results import SecurityGateResult, StructuredError

_VALID_SECURITY_RESULT: dict = {
    "claim_id": "CLM-0001",
    "decision": "ALLOW",
    "reasons": ["contrôle nominal"],
}


def _valid_request(**overrides) -> AgentCallRequest:
    fields = {
        "agent_name": AgentName.SECURITY_GATE,
        "case_id": "CLM-0001",
        "current_step": "security_gate",
        "requested_model": "SecurityGateResult",
        "authorized_context": frozenset({"security_input"}),
        "attempt": 1,
    }
    fields.update(overrides)
    return AgentCallRequest(**fields)


# ── 1. AgentCallRequest — cas valide ──────────────────────────────────────────


class TestAgentCallRequestValid:
    def test_valid_request_is_constructed(self):
        request = _valid_request()
        assert request.agent_name is AgentName.SECURITY_GATE
        assert request.case_id == "CLM-0001"
        assert request.current_step == "security_gate"
        assert request.requested_model == "SecurityGateResult"
        assert request.attempt == 1

    def test_authorized_context_default_is_empty(self):
        request = _valid_request(authorized_context=frozenset())
        assert request.authorized_context == frozenset()

    def test_attempt_defaults_to_one(self):
        fields = {
            "agent_name": AgentName.SECURITY_GATE,
            "case_id": "CLM-0001",
            "current_step": "security_gate",
            "requested_model": "SecurityGateResult",
        }
        request = AgentCallRequest(**fields)
        assert request.attempt == 1

    def test_agent_name_accepts_string_value(self):
        request = _valid_request(agent_name="security_gate")
        assert request.agent_name is AgentName.SECURITY_GATE


# ── 2. AgentCallRequest — cas invalides ───────────────────────────────────────


class TestAgentCallRequestInvalid:
    def test_case_id_bad_pattern_rejected(self):
        with pytest.raises(ValidationError):
            _valid_request(case_id="not-a-case-id")

    def test_case_id_empty_rejected(self):
        with pytest.raises(ValidationError):
            _valid_request(case_id="")

    def test_unknown_agent_name_rejected(self):
        with pytest.raises(ValidationError):
            _valid_request(agent_name="not_a_real_agent")

    def test_requested_model_mismatched_with_agent_rejected(self):
        with pytest.raises(ValidationError, match="incohérent"):
            _valid_request(requested_model="ClaimIntakeResult")

    def test_requested_model_empty_rejected(self):
        with pytest.raises(ValidationError):
            _valid_request(requested_model="")

    def test_authorized_context_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="inconnus de"):
            _valid_request(authorized_context=frozenset({"not_a_claim_state_field"}))

    def test_current_step_empty_rejected(self):
        with pytest.raises(ValidationError):
            _valid_request(current_step="")

    def test_attempt_zero_rejected(self):
        with pytest.raises(ValidationError):
            _valid_request(attempt=0)

    def test_attempt_negative_rejected(self):
        with pytest.raises(ValidationError):
            _valid_request(attempt=-1)

    def test_unknown_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            _valid_request(unexpected_field="boom")


# ── 3. AgentCallOutcome — cas valides ─────────────────────────────────────────


class TestAgentCallOutcomeValid:
    def test_success_outcome_is_constructed(self):
        outcome = AgentCallOutcome(
            agent_name=AgentName.SECURITY_GATE,
            case_id="CLM-0001",
            current_step="security_gate",
            attempt=1,
            success=True,
            result_payload=_VALID_SECURITY_RESULT,
        )
        assert outcome.success is True
        assert outcome.error is None
        assert outcome.result_payload == _VALID_SECURITY_RESULT

    def test_failure_outcome_is_constructed(self):
        outcome = AgentCallOutcome(
            agent_name=AgentName.SECURITY_GATE,
            case_id="CLM-0001",
            current_step="security_gate",
            attempt=2,
            success=False,
            error=StructuredError(code="TRANSIENT_NETWORK_ERROR", message="connexion refusée"),
        )
        assert outcome.success is False
        assert outcome.result_payload is None
        assert outcome.error.code == "TRANSIENT_NETWORK_ERROR"

    def test_metadata_defaults_to_empty_dict(self):
        outcome = AgentCallOutcome(
            agent_name=AgentName.SECURITY_GATE,
            case_id="CLM-0001",
            current_step="security_gate",
            attempt=1,
            success=True,
            result_payload=_VALID_SECURITY_RESULT,
        )
        assert outcome.metadata == {}

    def test_metadata_is_preserved(self):
        outcome = AgentCallOutcome(
            agent_name=AgentName.SECURITY_GATE,
            case_id="CLM-0001",
            current_step="security_gate",
            attempt=1,
            success=True,
            result_payload=_VALID_SECURITY_RESULT,
            metadata={"llm_model": "gemma4:latest"},
        )
        assert outcome.metadata == {"llm_model": "gemma4:latest"}

    def test_audit_events_defaults_to_empty_tuple(self):
        outcome = AgentCallOutcome(
            agent_name=AgentName.SECURITY_GATE,
            case_id="CLM-0001",
            current_step="security_gate",
            attempt=1,
            success=True,
            result_payload=_VALID_SECURITY_RESULT,
        )
        assert outcome.audit_events == ()

    def test_audit_events_is_preserved(self):
        from schemas.results import AuditEvent

        event = AuditEvent(
            event_id="evt-1",
            case_id="CLM-0001",
            actor="security_gate",
            action="authorization",
            outcome="ALLOW",
            details={"policy": "MODEL_AUTHORIZED"},
        )
        outcome = AgentCallOutcome(
            agent_name=AgentName.SECURITY_GATE,
            case_id="CLM-0001",
            current_step="security_gate",
            attempt=1,
            success=True,
            result_payload=_VALID_SECURITY_RESULT,
            audit_events=(event,),
        )
        assert outcome.audit_events == (event,)

    def test_from_request_audit_events_default_empty(self):
        request = _valid_request()
        outcome = AgentCallOutcome.from_request(
            request, success=True, result_payload=_VALID_SECURITY_RESULT
        )
        assert outcome.audit_events == ()

    def test_from_request_audit_events_propagated(self):
        from schemas.results import AuditEvent

        event = AuditEvent(
            event_id="evt-1",
            case_id="CLM-0001",
            actor="security_gate",
            action="result",
            outcome="SUCCESS",
        )
        request = _valid_request()
        outcome = AgentCallOutcome.from_request(
            request, success=True, result_payload=_VALID_SECURITY_RESULT, audit_events=[event]
        )
        assert outcome.audit_events == (event,)

    def test_from_request_propagates_identity_fields(self):
        request = _valid_request(attempt=3)
        outcome = AgentCallOutcome.from_request(
            request, success=True, result_payload=_VALID_SECURITY_RESULT
        )
        assert outcome.agent_name == request.agent_name
        assert outcome.case_id == request.case_id
        assert outcome.current_step == request.current_step
        assert outcome.attempt == request.attempt

    def test_from_request_failure_carries_error(self):
        request = _valid_request()
        error = StructuredError(code="BLOCKED", message="menace détectée")
        outcome = AgentCallOutcome.from_request(request, success=False, error=error)
        assert outcome.success is False
        assert outcome.error is error


# ── 4. AgentCallOutcome — cas invalides ───────────────────────────────────────


class TestAgentCallOutcomeInvalid:
    def test_success_true_with_error_rejected(self):
        with pytest.raises(ValidationError, match="error doit être None"):
            AgentCallOutcome(
                agent_name=AgentName.SECURITY_GATE,
                case_id="CLM-0001",
                current_step="security_gate",
                attempt=1,
                success=True,
                result_payload=_VALID_SECURITY_RESULT,
                error=StructuredError(code="X", message="y"),
            )

    def test_success_true_without_result_payload_rejected(self):
        with pytest.raises(ValidationError, match="result_payload obligatoire"):
            AgentCallOutcome(
                agent_name=AgentName.SECURITY_GATE,
                case_id="CLM-0001",
                current_step="security_gate",
                attempt=1,
                success=True,
            )

    def test_success_false_without_error_rejected(self):
        with pytest.raises(ValidationError, match="error obligatoire"):
            AgentCallOutcome(
                agent_name=AgentName.SECURITY_GATE,
                case_id="CLM-0001",
                current_step="security_gate",
                attempt=1,
                success=False,
            )

    def test_success_false_with_result_payload_rejected(self):
        with pytest.raises(ValidationError, match="result_payload doit être None"):
            AgentCallOutcome(
                agent_name=AgentName.SECURITY_GATE,
                case_id="CLM-0001",
                current_step="security_gate",
                attempt=1,
                success=False,
                error=StructuredError(code="X", message="y"),
                result_payload=_VALID_SECURITY_RESULT,
            )

    def test_result_payload_wrong_shape_for_agent_rejected(self):
        with pytest.raises(ValidationError, match="invalide pour"):
            AgentCallOutcome(
                agent_name=AgentName.SECURITY_GATE,
                case_id="CLM-0001",
                current_step="security_gate",
                attempt=1,
                success=True,
                result_payload={"totally": "wrong shape"},
            )

    def test_result_payload_missing_required_field_rejected(self):
        with pytest.raises(ValidationError):
            AgentCallOutcome(
                agent_name=AgentName.SECURITY_GATE,
                case_id="CLM-0001",
                current_step="security_gate",
                attempt=1,
                success=True,
                result_payload={"claim_id": "CLM-0001", "decision": "ALLOW"},  # reasons manquant
            )

    def test_case_id_bad_pattern_rejected(self):
        with pytest.raises(ValidationError):
            AgentCallOutcome(
                agent_name=AgentName.SECURITY_GATE,
                case_id="bad-id",
                current_step="security_gate",
                attempt=1,
                success=True,
                result_payload=_VALID_SECURITY_RESULT,
            )

    def test_attempt_zero_rejected(self):
        with pytest.raises(ValidationError):
            AgentCallOutcome(
                agent_name=AgentName.SECURITY_GATE,
                case_id="CLM-0001",
                current_step="security_gate",
                attempt=0,
                success=True,
                result_payload=_VALID_SECURITY_RESULT,
            )

    def test_unknown_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            AgentCallOutcome(
                agent_name=AgentName.SECURITY_GATE,
                case_id="CLM-0001",
                current_step="security_gate",
                attempt=1,
                success=True,
                result_payload=_VALID_SECURITY_RESULT,
                unexpected_field="boom",
            )


# ── 5. AGENT_RESULT_MODELS — complétude ───────────────────────────────────────


class TestAgentResultModelsCompleteness:
    def test_one_entry_per_agent_name(self):
        assert set(AGENT_RESULT_MODELS.keys()) == set(AgentName)

    def test_eleven_agents(self):
        assert len(AgentName) == 11
        assert len(AGENT_RESULT_MODELS) == 11

    @pytest.mark.parametrize("agent_name", list(AgentName))
    def test_each_model_is_a_class(self, agent_name):
        assert isinstance(AGENT_RESULT_MODELS[agent_name], type)


# ── 6. validate_agent_result — sortie valide, champ manquant, mauvais type, ──
#      contenu non structuré ────────────────────────────────────────────────


class TestValidateAgentResultValid:
    def test_valid_model_instance_is_accepted(self):
        instance = SecurityGateResult(claim_id="CLM-0001", decision="ALLOW", reasons=["ok"])
        payload = validate_agent_result(AgentName.SECURITY_GATE, instance)
        assert payload["decision"] == "ALLOW"
        assert payload["claim_id"] == "CLM-0001"

    def test_valid_dict_is_accepted_and_revalidated(self):
        """Un dict valide est accepté, mais toujours repassé par
        model_validate() — jamais retourné tel quel sans passage par le
        modèle (il doit porter les valeurs par défaut du modèle)."""
        payload = validate_agent_result(AgentName.SECURITY_GATE, dict(_VALID_SECURITY_RESULT))
        # Champ à valeur par défaut, absent du dict fourni : preuve que le
        # dict a bien traversé le modèle Pydantic et non recopié tel quel.
        assert payload["policy_version"] == "1.0.0"
        assert payload["decision"] == "ALLOW"

    def test_returned_payload_is_a_plain_dict(self):
        payload = validate_agent_result(AgentName.SECURITY_GATE, dict(_VALID_SECURITY_RESULT))
        assert isinstance(payload, dict)
        assert not isinstance(payload, SecurityGateResult)


class TestValidateAgentResultMissingField:
    def test_missing_required_field_is_rejected(self):
        incomplete = {"claim_id": "CLM-0001", "decision": "ALLOW"}  # reasons manquant
        with pytest.raises(AgentResultValidationError) as exc_info:
            validate_agent_result(AgentName.SECURITY_GATE, incomplete)
        assert exc_info.value.structured.code == "AGENT_RESULT_INVALID"
        assert "reasons" in exc_info.value.structured.message

    def test_none_result_is_missing_not_invalid(self):
        """Absence totale de résultat : code dédié AGENT_RESULT_MISSING,
        distinct d'un dict incomplet (AGENT_RESULT_INVALID)."""
        with pytest.raises(AgentResultValidationError) as exc_info:
            validate_agent_result(AgentName.SECURITY_GATE, None)
        assert exc_info.value.structured.code == "AGENT_RESULT_MISSING"

    def test_error_attributes_the_right_agent(self):
        with pytest.raises(AgentResultValidationError) as exc_info:
            validate_agent_result(AgentName.SECURITY_GATE, {})
        assert "security_gate" in exc_info.value.structured.message


class TestValidateAgentResultWrongType:
    def test_wrong_field_type_is_rejected(self):
        bad = {"claim_id": "CLM-0001", "decision": "ALLOW", "reasons": 12345}
        with pytest.raises(AgentResultValidationError) as exc_info:
            validate_agent_result(AgentName.SECURITY_GATE, bad)
        assert exc_info.value.structured.code == "AGENT_RESULT_INVALID"
        assert "reasons" in exc_info.value.structured.message

    def test_invalid_enum_value_is_rejected(self):
        bad = {"claim_id": "CLM-0001", "decision": "NOT_A_REAL_DECISION", "reasons": ["x"]}
        with pytest.raises(AgentResultValidationError) as exc_info:
            validate_agent_result(AgentName.SECURITY_GATE, bad)
        assert exc_info.value.structured.code == "AGENT_RESULT_INVALID"

    def test_instance_of_a_different_agents_model_is_rejected(self):
        """Un résultat structurellement valide, mais pour le mauvais agent,
        n'est jamais accepté comme résultat final de security_gate."""
        from schemas.results import ClaimIntakeResult

        wrong_model_instance = ClaimIntakeResult.model_construct(
            claim_id="CLM-0001", status="accepted", accepted_count=0, missing_documents=[]
        )
        with pytest.raises(AgentResultValidationError):
            validate_agent_result(AgentName.SECURITY_GATE, wrong_model_instance)


class TestValidateAgentResultUnstructured:
    def test_free_text_string_is_rejected(self):
        with pytest.raises(AgentResultValidationError) as exc_info:
            validate_agent_result(AgentName.SECURITY_GATE, "ALLOW, tout va bien.")
        assert exc_info.value.structured.code == "AGENT_RESULT_UNSTRUCTURED"

    def test_list_is_rejected(self):
        with pytest.raises(AgentResultValidationError) as exc_info:
            validate_agent_result(AgentName.SECURITY_GATE, ["ALLOW", "ok"])
        assert exc_info.value.structured.code == "AGENT_RESULT_UNSTRUCTURED"

    def test_integer_is_rejected(self):
        with pytest.raises(AgentResultValidationError) as exc_info:
            validate_agent_result(AgentName.SECURITY_GATE, 42)
        assert exc_info.value.structured.code == "AGENT_RESULT_UNSTRUCTURED"

    def test_unstructured_content_never_attempts_model_validation(self):
        """Un texte libre ne doit même pas être tenté en model_validate —
        rejeté catégoriquement, pas juste 'invalide'."""
        with pytest.raises(AgentResultValidationError) as exc_info:
            validate_agent_result(AgentName.SECURITY_GATE, "texte libre quelconque")
        assert exc_info.value.structured.code == "AGENT_RESULT_UNSTRUCTURED"
        assert exc_info.value.structured.code != "AGENT_RESULT_INVALID"


class TestValidateAgentResultNeverLeaksSensitiveContent:
    """Ne journalise jamais la réponse brute lorsqu'elle contient des
    données sensibles — vérifié sur les trois chemins de refus."""

    _SENSITIVE_MARKER = "ssn:123-45-6789;api_key=sk-live-abcdef"

    def test_unstructured_sensitive_text_never_appears_in_error(self):
        with pytest.raises(AgentResultValidationError) as exc_info:
            validate_agent_result(AgentName.SECURITY_GATE, self._SENSITIVE_MARKER)
        assert self._SENSITIVE_MARKER not in exc_info.value.structured.message
        assert self._SENSITIVE_MARKER not in str(exc_info.value)

    def test_invalid_dict_with_sensitive_value_never_leaks_it(self):
        bad = {
            "claim_id": "CLM-0001",
            "decision": "ALLOW",
            "reasons": ["ok"],
            "evidence_summary": self._SENSITIVE_MARKER,
            "next_allowed_action": "x" * 500,  # dépasse max_length -> erreur
        }
        with pytest.raises(AgentResultValidationError) as exc_info:
            validate_agent_result(AgentName.SECURITY_GATE, bad)
        assert self._SENSITIVE_MARKER not in exc_info.value.structured.message

    def test_agent_call_outcome_direct_construction_message_never_leaks(self):
        """Même garde-fou côté AgentCallOutcome construit directement (pas
        seulement via validate_agent_result) : le message que *nous*
        construisons (``err['msg']``) ne contient jamais la valeur brute.

        Note : ``str(ValidationError)`` de Pydantic peut, lui, inclure un
        aperçu tronqué de la valeur d'entrée complète (``err['input']``) —
        un comportement du framework, hors du message que ce validateur
        construit. En pratique, ``execute_agent`` ne construit jamais
        d'``AgentCallOutcome`` avec un ``result_payload`` invalide : il
        passe toujours par ``validate_agent_result`` au préalable, qui
        garantit un payload déjà propre avant construction.
        """
        with pytest.raises(ValidationError) as exc_info:
            AgentCallOutcome(
                agent_name=AgentName.SECURITY_GATE,
                case_id="CLM-0001",
                current_step="security_gate",
                attempt=1,
                success=True,
                result_payload={
                    "claim_id": "CLM-0001",
                    "decision": "ALLOW",
                    # reasons manquant + valeur sensible ailleurs
                    "evidence_summary": self._SENSITIVE_MARKER,
                },
            )
        messages = [err["msg"] for err in exc_info.value.errors()]
        assert all(self._SENSITIVE_MARKER not in msg for msg in messages)
