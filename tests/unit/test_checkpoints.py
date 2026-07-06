"""Tests de la configuration centralisée des checkpoints LangGraph."""
from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
import pytest

from graph.checkpoints import (
    assert_same_thread_id,
    get_checkpointer,
    make_thread_config,
    validate_checkpoint_state,
)
from schemas.domain import IntakeStatus, SecurityDecision
from schemas.results import (
    AuditEvent,
    ClaimIntakeResult,
    ClaimManifest,
    LlmMetadata,
    SecurityGateResult,
)


def _checkpointable_state() -> dict:
    intake_result = ClaimIntakeResult(
        claim_id="CLM-0001",
        status=IntakeStatus.ACCEPTED,
        manifest=ClaimManifest(
            claim_id="CLM-0001",
            file_count=1,
            total_size_bytes=1234,
            status=IntakeStatus.ACCEPTED,
        ),
        accepted_count=1,
        quarantined_count=0,
        llm_metadata=LlmMetadata(model_name="test-llm", prompt_version="test"),
    )
    security_result = SecurityGateResult(
        claim_id="CLM-0001",
        decision=SecurityDecision.ALLOW,
        reasons=["Aucune menace détectée."],
    )
    audit = AuditEvent(
        event_id="evt-1",
        case_id="CLM-0001",
        actor="test",
        action="checkpoint_validation",
        outcome="PASS",
    )
    return {
        "case_id": "CLM-0001",
        "schema_version": "1.0.0",
        "intake_status": IntakeStatus.ACCEPTED,
        "current_step": "security_gate",
        "completed_steps": ["claim_intake", "security_gate"],
        "intake_result": intake_result,
        "security_result": security_result,
        "errors": [],
        "alerts": ["Document optionnel absent"],
        "audit_trail": [audit],
        "human_decision": None,
        "final_recommendation": None,
        "final_justification": [],
    }


def test_get_checkpointer_memory_for_tests():
    checkpointer = get_checkpointer(backend="memory")
    assert isinstance(checkpointer, InMemorySaver)


def test_state_complet_serialisable_par_checkpointer_memoire():
    state = _checkpointable_state()
    validate_checkpoint_state(state)

    checkpointer = get_checkpointer(backend="memory")
    typed_payload = checkpointer.serde.dumps_typed(state)
    assert typed_payload[0]
    assert typed_payload[1]


def test_checkpoint_refuse_documents_originaux_secrets_et_prompts():
    forbidden_states = [
        {"case_id": "CLM-0001", "document_bytes": b"%PDF-1.4"},
        {"case_id": "CLM-0001", "metadata": {"api_key": "sk-secret"}},
        {"case_id": "CLM-0001", "llm": {"raw_response": "{\"status\":\"PASS\"}"}},
        {"case_id": "CLM-0001", "source_path": "/tmp/facture.pdf"},
    ]

    for state in forbidden_states:
        with pytest.raises(ValueError):
            validate_checkpoint_state(state)


def test_thread_id_obligatoire_et_stable_pour_reprise():
    initial = make_thread_config("CLM-0001")
    resume = make_thread_config("CLM-0001")
    assert initial["configurable"]["thread_id"] == "CLM-0001"
    assert_same_thread_id(initial, resume)

    with pytest.raises(ValueError, match="thread_id différent"):
        assert_same_thread_id(initial, make_thread_config("CLM-0002"))

    with pytest.raises(ValueError, match="thread_id obligatoire"):
        make_thread_config("")
