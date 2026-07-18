"""Tests de chat/conversation_store.py — plan de remédiation « autonomie
décisionnelle V2 », Phase 8."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from chat.conversation_store import ConversationAccessError, ConversationStore
from chat.memory_schemas import ConversationSemanticState, ConversationTurn


def _turn(turn_id: str = "t1", case_id: str | None = "CLM-8001") -> ConversationTurn:
    return ConversationTurn(
        turn_id=turn_id,
        message_digest="a" * 64,
        reply_digest="b" * 64,
        case_id=case_id,
        created_at=datetime.now(UTC),
    )


class TestIsolation:
    def test_get_on_unknown_thread_returns_none(self):
        store = ConversationStore()
        assert store.get(user_id="alice", thread_id="thread-1") is None

    def test_different_users_never_share_a_thread(self):
        store = ConversationStore()
        store.append_turn(user_id="alice", thread_id="thread-1", turn=_turn())
        with pytest.raises(ConversationAccessError):
            store.get(user_id="mallory", thread_id="thread-1")
        with pytest.raises(ConversationAccessError):
            store.append_turn(user_id="mallory", thread_id="thread-1", turn=_turn("t2"))

    def test_same_user_can_reuse_own_thread(self):
        store = ConversationStore()
        store.append_turn(user_id="alice", thread_id="thread-1", turn=_turn())
        context = store.append_turn(user_id="alice", thread_id="thread-1", turn=_turn("t2"))
        assert len(context.turns) == 2

    def test_two_users_have_fully_independent_contexts_on_different_threads(self):
        store = ConversationStore()
        store.append_turn(user_id="alice", thread_id="thread-1", turn=_turn("t1", "CLM-8001"))
        store.append_turn(user_id="bob", thread_id="thread-2", turn=_turn("t2", "CLM-9001"))
        alice_context = store.get(user_id="alice", thread_id="thread-1")
        bob_context = store.get(user_id="bob", thread_id="thread-2")
        assert alice_context.turns[0].case_id == "CLM-8001"
        assert bob_context.turns[0].case_id == "CLM-9001"


class TestBoundedWindow:
    def test_window_truncates_oldest_turns_first(self):
        store = ConversationStore(max_recent_turns=3)
        for i in range(5):
            store.append_turn(user_id="alice", thread_id="thread-1", turn=_turn(f"t{i}"))
        context = store.get(user_id="alice", thread_id="thread-1")
        assert len(context.turns) == 3
        assert [t.turn_id for t in context.turns] == ["t2", "t3", "t4"]

    def test_invalid_max_recent_turns_rejected(self):
        with pytest.raises(ValueError):
            ConversationStore(max_recent_turns=0)


class TestSemanticStateUpdate:
    def test_update_semantic_state_requires_existing_conversation(self):
        store = ConversationStore()
        state = ConversationSemanticState(conversation_summary="résumé", updated_at=datetime.now(UTC))
        with pytest.raises(ConversationAccessError):
            store.update_semantic_state(user_id="alice", thread_id="thread-1", semantic_state=state)

    def test_update_semantic_state_never_touches_turns(self):
        store = ConversationStore()
        store.append_turn(user_id="alice", thread_id="thread-1", turn=_turn())
        state = ConversationSemanticState(conversation_summary="résumé", updated_at=datetime.now(UTC))
        updated = store.update_semantic_state(user_id="alice", thread_id="thread-1", semantic_state=state)
        assert len(updated.turns) == 1
        assert updated.semantic_state == state


class TestExpiration:
    def test_expire_older_than_removes_whole_conversation(self):
        store = ConversationStore()
        store.append_turn(user_id="alice", thread_id="thread-1", turn=_turn())
        removed = store.expire_older_than(ttl_seconds=-1)
        assert removed == 1
        assert store.get(user_id="alice", thread_id="thread-1") is None

    def test_expire_older_than_keeps_fresh_conversations(self):
        store = ConversationStore()
        store.append_turn(user_id="alice", thread_id="thread-1", turn=_turn())
        removed = store.expire_older_than(ttl_seconds=3600)
        assert removed == 0
        assert store.get(user_id="alice", thread_id="thread-1") is not None

    def test_thread_id_freed_after_expiration_can_be_reclaimed_by_another_user(self):
        """Après expiration complète, un thread_id n'appartient plus à
        personne — un autre utilisateur peut légitimement le réutiliser."""
        store = ConversationStore()
        store.append_turn(user_id="alice", thread_id="thread-1", turn=_turn())
        store.expire_older_than(ttl_seconds=-1)
        # N'importe qui peut désormais réclamer ce thread_id — jamais une
        # erreur pour un thread_id réellement libéré.
        context = store.append_turn(user_id="bob", thread_id="thread-1", turn=_turn())
        assert context.user_id == "bob"
