"""Tests — CheckpointerFactory et CheckpointSession — ClaimShield Santé.

Ces tests couvrent les deux nouvelles abstractions de graph/checkpoints.py :

  - ``CheckpointerFactory`` — injection du checkpointer à la compilation.
  - ``CheckpointSession``   — thread_id stable, reprise interdite avec un
    id différent, sauvegarde/restauration d'un ClaimState minimal.

Les fonctions de bas niveau (``get_checkpointer``, ``make_thread_config``,
``assert_same_thread_id``, ``validate_checkpoint_state``) sont déjà couvertes
dans ``tests/unit/test_checkpoints.py`` ; elles ne sont pas dupliquées ici.
"""
from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from graph.checkpoints import (
    CheckpointerFactory,
    CheckpointSession,
    make_thread_config,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _minimal_state() -> dict:
    """ClaimState minimal passant validate_checkpoint_state."""
    return {
        "case_id": "CLM-SESS-001",
        "schema_version": "1.0.0",
        "current_step": "claim_intake",
        "completed_steps": ["claim_intake"],
        "errors": [],
        "alerts": [],
        "final_justification": [],
    }


# ── TestCheckpointerFactory ───────────────────────────────────────────────────


class TestCheckpointerFactory:
    def test_for_tests_returns_in_memory_saver(self):
        factory = CheckpointerFactory.for_tests()
        checkpointer = factory.build()
        assert isinstance(checkpointer, InMemorySaver)

    def test_for_tests_each_call_fresh_instance(self):
        # Deux appels à for_tests() produisent deux InMemorySaver indépendants.
        a = CheckpointerFactory.for_tests().build()
        b = CheckpointerFactory.for_tests().build()
        assert a is not b

    def test_injected_instance_returned_directly(self):
        saver = InMemorySaver()
        factory = CheckpointerFactory(saver)
        assert factory.build() is saver

    def test_injected_instance_not_recreated_on_repeated_build(self):
        saver = InMemorySaver()
        factory = CheckpointerFactory(saver)
        assert factory.build() is factory.build() is saver

    def test_build_with_backend_memory(self):
        factory = CheckpointerFactory(backend="memory")
        checkpointer = factory.build()
        assert isinstance(checkpointer, InMemorySaver)

    def test_from_settings_returns_factory(self):
        factory = CheckpointerFactory.from_settings()
        assert isinstance(factory, CheckpointerFactory)

    def test_from_settings_build_uses_default_backend(self):
        # Le backend par défaut issu de .env est "memory" en environnement de test.
        factory = CheckpointerFactory.from_settings()
        checkpointer = factory.build()
        assert checkpointer is not None

    def test_backend_overrides_settings(self):
        factory = CheckpointerFactory(backend="memory")
        assert isinstance(factory.build(), InMemorySaver)

    def test_injected_takes_priority_over_backend(self):
        saver = InMemorySaver()
        factory = CheckpointerFactory(saver, backend="memory")
        # L'instance injectée est toujours retournée.
        assert factory.build() is saver


# ── TestCheckpointSession ─────────────────────────────────────────────────────


class TestCheckpointSession:
    def test_case_id_stored(self):
        session = CheckpointSession("CLM-0001")
        assert session.case_id == "CLM-0001"

    def test_whitespace_stripped_from_case_id(self):
        session = CheckpointSession("  CLM-0002  ")
        assert session.case_id == "CLM-0002"

    def test_thread_id_equals_case_id(self):
        session = CheckpointSession("CLM-0003")
        assert session.thread_id == "CLM-0003"

    def test_config_has_correct_thread_id(self):
        session = CheckpointSession("CLM-0004")
        assert session.config["configurable"]["thread_id"] == "CLM-0004"

    def test_config_has_checkpoint_ns(self):
        session = CheckpointSession("CLM-0005")
        assert "checkpoint_ns" in session.config["configurable"]

    def test_empty_case_id_raises(self):
        with pytest.raises(ValueError, match="case_id"):
            CheckpointSession("")

    def test_whitespace_only_case_id_raises(self):
        with pytest.raises(ValueError, match="case_id"):
            CheckpointSession("   ")

    def test_assert_resume_same_thread_id_passes(self):
        session = CheckpointSession("CLM-0006")
        same_config = make_thread_config("CLM-0006")
        session.assert_resume(same_config)  # pas d'exception

    def test_assert_resume_different_thread_id_raises(self):
        session = CheckpointSession("CLM-0007")
        other_config = make_thread_config("CLM-0008")
        with pytest.raises(ValueError, match="thread_id différent"):
            session.assert_resume(other_config)

    def test_assert_resume_missing_configurable_raises(self):
        session = CheckpointSession("CLM-0009")
        with pytest.raises(ValueError):
            session.assert_resume({})

    def test_config_is_stable_across_calls(self):
        session = CheckpointSession("CLM-0010")
        assert session.config == session.config


# ── TestSaveAndRestore ────────────────────────────────────────────────────────


class TestSaveAndRestore:
    def test_load_returns_none_before_any_save(self):
        saver = InMemorySaver()
        session = CheckpointSession("CLM-SR-001")
        assert session.load(saver) is None

    def test_save_returns_config_with_checkpoint_id(self):
        saver = InMemorySaver()
        session = CheckpointSession("CLM-SR-002")
        saved_config = session.save(saver, _minimal_state(), step=1)
        assert "checkpoint_id" in saved_config["configurable"]

    def test_load_restores_case_id(self):
        saver = InMemorySaver()
        session = CheckpointSession("CLM-SR-003")
        state = _minimal_state() | {"case_id": "CLM-SR-003"}
        session.save(saver, state, step=1)
        restored = session.load(saver)
        assert restored["case_id"] == "CLM-SR-003"

    def test_load_restores_current_step(self):
        saver = InMemorySaver()
        session = CheckpointSession("CLM-SR-004")
        state = _minimal_state() | {"current_step": "privacy"}
        session.save(saver, state, step=2)
        restored = session.load(saver)
        assert restored["current_step"] == "privacy"

    def test_load_restores_completed_steps(self):
        saver = InMemorySaver()
        session = CheckpointSession("CLM-SR-005")
        state = _minimal_state() | {"completed_steps": ["claim_intake", "security_gate"]}
        session.save(saver, state, step=2)
        restored = session.load(saver)
        assert restored["completed_steps"] == ["claim_intake", "security_gate"]

    def test_load_restores_all_minimal_fields(self):
        saver = InMemorySaver()
        session = CheckpointSession("CLM-SR-006")
        state = _minimal_state()
        session.save(saver, state, step=1)
        restored = session.load(saver)
        assert restored == state

    def test_load_isolated_from_other_session(self):
        saver = InMemorySaver()
        session_a = CheckpointSession("CLM-SR-007A")
        session_b = CheckpointSession("CLM-SR-007B")
        session_a.save(saver, _minimal_state() | {"case_id": "CLM-SR-007A"}, step=1)
        # session_b n'a rien sauvegardé — son load doit retourner None.
        assert session_b.load(saver) is None

    def test_load_does_not_cross_sessions(self):
        saver = InMemorySaver()
        session_a = CheckpointSession("CLM-SR-008A")
        session_b = CheckpointSession("CLM-SR-008B")
        state_a = _minimal_state() | {"case_id": "CLM-SR-008A", "current_step": "intake_a"}
        state_b = _minimal_state() | {"case_id": "CLM-SR-008B", "current_step": "intake_b"}
        session_a.save(saver, state_a, step=1)
        session_b.save(saver, state_b, step=1)
        assert session_a.load(saver)["current_step"] == "intake_a"
        assert session_b.load(saver)["current_step"] == "intake_b"

    def test_save_rejects_bytes_content(self):
        saver = InMemorySaver()
        session = CheckpointSession("CLM-SR-009")
        bad_state = _minimal_state() | {"document_bytes": b"%PDF"}
        with pytest.raises(ValueError):
            session.save(saver, bad_state, step=1)

    def test_save_rejects_absolute_path(self):
        saver = InMemorySaver()
        session = CheckpointSession("CLM-SR-010")
        bad_state = _minimal_state() | {"some_field": "/tmp/secret.pdf"}
        with pytest.raises(ValueError):
            session.save(saver, bad_state, step=1)

    def test_save_rejects_missing_required_keys(self):
        saver = InMemorySaver()
        session = CheckpointSession("CLM-SR-011")
        incomplete = {"case_id": "CLM-SR-011"}  # manque schema_version, current_step, completed_steps
        with pytest.raises(ValueError):
            session.save(saver, incomplete, step=1)

    def test_saved_config_checkpoint_id_is_string(self):
        saver = InMemorySaver()
        session = CheckpointSession("CLM-SR-012")
        saved_config = session.save(saver, _minimal_state(), step=1)
        chk_id = saved_config["configurable"]["checkpoint_id"]
        assert isinstance(chk_id, str) and len(chk_id) > 0

    def test_each_save_retrievable_by_checkpoint_id(self):
        # InMemorySaver ne garantit pas un ordre "latest" sur put() directs.
        # En revanche, chaque config retournée par save() permet de relire
        # exactement le checkpoint correspondant via checkpointer.get(cfg).
        saver = InMemorySaver()
        session = CheckpointSession("CLM-SR-013")
        state_v1 = _minimal_state() | {"current_step": "step_one"}
        state_v2 = _minimal_state() | {"current_step": "step_two"}
        cfg_v1 = session.save(saver, state_v1, step=1)
        cfg_v2 = session.save(saver, state_v2, step=2)
        assert saver.get(cfg_v1)["channel_values"]["current_step"] == "step_one"
        assert saver.get(cfg_v2)["channel_values"]["current_step"] == "step_two"

    def test_assert_resume_with_saved_config(self):
        saver = InMemorySaver()
        session = CheckpointSession("CLM-SR-014")
        saved_config = session.save(saver, _minimal_state(), step=1)
        # La config retournée par save (avec checkpoint_id) est valide pour assert_resume.
        session.assert_resume(saved_config)  # pas d'exception

    def test_assert_resume_rejects_different_case(self):
        saver = InMemorySaver()
        session = CheckpointSession("CLM-SR-015")
        session.save(saver, _minimal_state(), step=1)
        wrong_config = make_thread_config("CLM-AUTRE")
        with pytest.raises(ValueError, match="thread_id différent"):
            session.assert_resume(wrong_config)
