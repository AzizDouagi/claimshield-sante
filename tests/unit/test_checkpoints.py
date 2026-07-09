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


class TestSqliteBackend:
    """Backend `sqlite` — jamais exercé contre un vrai graphe compilé avant
    l'item E (packaging Docker) : `SqliteSaver.from_conn_string` est un
    context manager qui ferme la connexion à la sortie du `with` ;
    `get_checkpointer` construit désormais `SqliteSaver` directement à
    partir d'une connexion `sqlite3` ouverte pour la durée du process."""

    def test_get_checkpointer_sqlite_returns_usable_saver(self, tmp_path):
        from graph.checkpoints import get_checkpointer
        from langgraph.checkpoint.sqlite import SqliteSaver
        from config.settings import Settings

        settings = Settings(LANGGRAPH_CHECKPOINT_DB=tmp_path / "checkpoints.db")
        checkpointer = get_checkpointer(backend="sqlite", settings=settings)

        assert isinstance(checkpointer, SqliteSaver)
        assert (tmp_path / "checkpoints.db").exists()

    def test_state_persists_across_fresh_checkpointer_same_file(self, tmp_path):
        """Simule un redémarrage de conteneur : un nouveau `get_checkpointer()`
        pointé sur le même fichier retrouve l'état déjà persisté."""
        from graph.checkpoints import get_checkpointer
        from config.settings import Settings

        db_path = tmp_path / "checkpoints.db"
        settings = Settings(LANGGRAPH_CHECKPOINT_DB=db_path)

        from langgraph.checkpoint.base import empty_checkpoint

        checkpointer_1 = get_checkpointer(backend="sqlite", settings=settings)
        config = make_thread_config("CLM-9999")
        checkpointer_1.put(config, empty_checkpoint(), {"source": "test", "step": 0, "parents": {}}, {})

        checkpointer_2 = get_checkpointer(backend="sqlite", settings=settings)
        tuple_read = checkpointer_2.get_tuple(config)

        assert tuple_read is not None
