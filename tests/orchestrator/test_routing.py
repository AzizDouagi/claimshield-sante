"""Tests de validation des préconditions d'appel — orchestrator/routing.py.

Couvre : préconditions satisfaites (premier agent, progression nominale,
nouvelle tentative), préconditions absentes (résultat requis manquant) et
préconditions incohérentes (dossier, étape déclarée, position pipeline).
"""
from __future__ import annotations

import pytest

from orchestrator.orchestrator import AGENT_RESULT_MODELS, AgentCallRequest, AgentName
from orchestrator.policies import PolicyEffect
from orchestrator.routing import (
    AGENT_PIPELINE_ORDER,
    AGENT_RESULT_FIELD,
    PIPELINE_START,
    evaluate_call_preconditions,
)


def _request(agent_name: AgentName, current_step: str, requested_model: str, **overrides) -> AgentCallRequest:
    fields = {
        "agent_name": agent_name,
        "case_id": "CLM-0001",
        "current_step": current_step,
        "requested_model": requested_model,
        "attempt": 1,
    }
    fields.update(overrides)
    return AgentCallRequest(**fields)


# ── 1. Structure de AGENT_PIPELINE_ORDER / AGENT_RESULT_FIELD ────────────────


class TestPipelineStructure:
    def test_pipeline_order_has_all_eleven_agents_once(self):
        assert len(AGENT_PIPELINE_ORDER) == 11
        assert set(AGENT_PIPELINE_ORDER) == set(AgentName)

    def test_pipeline_order_starts_with_claim_intake(self):
        assert AGENT_PIPELINE_ORDER[0] is AgentName.CLAIM_INTAKE

    def test_pipeline_order_ends_with_audit(self):
        assert AGENT_PIPELINE_ORDER[-1] is AgentName.AUDIT

    def test_result_field_has_one_entry_per_agent(self):
        assert set(AGENT_RESULT_FIELD.keys()) == set(AgentName)

    def test_result_field_reuses_relaunch_result_fields_for_seven_agents(self):
        from graph.edges import RELAUNCH_RESULT_FIELDS

        for name, field in RELAUNCH_RESULT_FIELDS.items():
            assert AGENT_RESULT_FIELD[AgentName(name)] == field


# ── 2. Préconditions satisfaites ──────────────────────────────────────────────


class TestPreconditionsSatisfied:
    def test_first_agent_allowed_from_pipeline_start(self):
        state = {"case_id": "CLM-0001", "current_step": PIPELINE_START}
        request = _request(AgentName.CLAIM_INTAKE, PIPELINE_START, "ClaimIntakeResult")
        decision = evaluate_call_preconditions(state, request)
        assert decision.effect is PolicyEffect.ALLOW
        assert decision.reason.code == "PRECONDITIONS_SATISFIED"

    def test_nominal_progression_allowed(self):
        state = {
            "case_id": "CLM-0001",
            "current_step": "claim_intake",
            "intake_result": object(),
        }
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        decision = evaluate_call_preconditions(state, request)
        assert decision.effect is PolicyEffect.ALLOW

    def test_retry_of_the_same_agent_allowed(self):
        """Rejouer l'agent qui vient d'échouer (état current_step déjà sur
        cet agent) est une précondition valide — cf. retry technique étape 11."""
        state = {
            "case_id": "CLM-0001",
            "current_step": "security_gate",
            "intake_result": object(),
        }
        request = _request(
            AgentName.SECURITY_GATE, "security_gate", "SecurityGateResult", attempt=2
        )
        decision = evaluate_call_preconditions(state, request)
        assert decision.effect is PolicyEffect.ALLOW

    @pytest.mark.parametrize(
        "index",
        list(range(1, len(AGENT_PIPELINE_ORDER))),
    )
    def test_every_nominal_step_of_the_pipeline_is_satisfiable(self, index):
        """Pour chaque agent (sauf le premier), l'état produit par son
        prédécesseur satisfait ses préconditions."""
        agent = AGENT_PIPELINE_ORDER[index]
        predecessor = AGENT_PIPELINE_ORDER[index - 1]
        state = {
            "case_id": "CLM-0001",
            "current_step": predecessor.value,
            AGENT_RESULT_FIELD[predecessor]: object(),
        }
        request = _request(agent, predecessor.value, AGENT_RESULT_MODELS[agent].__name__)
        decision = evaluate_call_preconditions(state, request)
        assert decision.effect is PolicyEffect.ALLOW, decision.reason.message


# ── 3. Préconditions absentes ─────────────────────────────────────────────────


class TestPreconditionsMissing:
    def test_missing_predecessor_result_denied(self):
        state = {
            "case_id": "CLM-0001",
            "current_step": "claim_intake",
            "intake_result": None,
        }
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        decision = evaluate_call_preconditions(state, request)
        assert decision.effect is PolicyEffect.DENY
        assert decision.reason.code == "PRECONDITION_RESULT_MISSING"

    def test_missing_predecessor_result_field_absent_entirely(self):
        state = {"case_id": "CLM-0001", "current_step": "claim_intake"}
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        decision = evaluate_call_preconditions(state, request)
        assert decision.effect is PolicyEffect.DENY
        assert decision.reason.code == "PRECONDITION_RESULT_MISSING"

    def test_missing_result_reason_names_the_field_and_predecessor(self):
        state = {"case_id": "CLM-0001", "current_step": "privacy", "privacy_result": None}
        request = _request(AgentName.DOCUMENT_OCR, "privacy", "DocumentOcrResult")
        decision = evaluate_call_preconditions(state, request)
        assert "privacy_result" in decision.reason.message
        assert "privacy" in decision.reason.message

    def test_claim_intake_never_requires_a_missing_result(self):
        """Premier agent : aucune précondition de résultat — seule l'étape
        de départ est vérifiée."""
        state = {"case_id": "CLM-0001", "current_step": PIPELINE_START}
        request = _request(AgentName.CLAIM_INTAKE, PIPELINE_START, "ClaimIntakeResult")
        decision = evaluate_call_preconditions(state, request)
        assert decision.effect is PolicyEffect.ALLOW


# ── 4. Préconditions incohérentes ─────────────────────────────────────────────


class TestPreconditionsInconsistent:
    def test_case_id_mismatch_denied(self):
        state = {
            "case_id": "CLM-9999",
            "current_step": "claim_intake",
            "intake_result": object(),
        }
        request = _request(AgentName.SECURITY_GATE, "claim_intake", "SecurityGateResult")
        decision = evaluate_call_preconditions(state, request)
        assert decision.effect is PolicyEffect.DENY
        assert decision.reason.code == "CASE_ID_MISMATCH"

    def test_declared_current_step_inconsistent_with_state(self):
        """La requête prétend être à une étape que l'état réel dément —
        vue périmée ou forgée."""
        state = {
            "case_id": "CLM-0001",
            "current_step": "claim_intake",
            "intake_result": object(),
        }
        request = _request(AgentName.SECURITY_GATE, "privacy", "SecurityGateResult")
        decision = evaluate_call_preconditions(state, request)
        assert decision.effect is PolicyEffect.DENY
        assert decision.reason.code == "STEP_DECLARATION_MISMATCH"

    def test_agent_does_not_match_current_pipeline_position_skip_ahead(self):
        state = {
            "case_id": "CLM-0001",
            "current_step": "claim_intake",
            "intake_result": object(),
        }
        request = _request(AgentName.MEDICAL_CODING, "claim_intake", "MedicalCodingResult")
        decision = evaluate_call_preconditions(state, request)
        assert decision.effect is PolicyEffect.DENY
        assert decision.reason.code == "STEP_MISMATCH"

    def test_agent_does_not_match_current_pipeline_position_out_of_order(self):
        """security_gate n'est pas rappelable juste après privacy (ordre
        inversé) sans passer par la route de relance HITL."""
        state = {
            "case_id": "CLM-0001",
            "current_step": "privacy",
            "privacy_result": object(),
        }
        request = _request(AgentName.SECURITY_GATE, "privacy", "SecurityGateResult")
        decision = evaluate_call_preconditions(state, request)
        assert decision.effect is PolicyEffect.DENY
        assert decision.reason.code == "STEP_MISMATCH"

    def test_first_agent_called_after_pipeline_already_advanced_denied(self):
        state = {
            "case_id": "CLM-0001",
            "current_step": "security_gate",
            "intake_result": object(),
        }
        request = _request(AgentName.CLAIM_INTAKE, "security_gate", "ClaimIntakeResult")
        decision = evaluate_call_preconditions(state, request)
        assert decision.effect is PolicyEffect.DENY
        assert decision.reason.code == "STEP_MISMATCH"

    def test_step_mismatch_reason_lists_expected_steps(self):
        state = {
            "case_id": "CLM-0001",
            "current_step": "claim_intake",
            "intake_result": object(),
        }
        request = _request(AgentName.AUDIT, "claim_intake", "AuditResult")
        decision = evaluate_call_preconditions(state, request)
        assert "case_reviewer" in decision.reason.message
