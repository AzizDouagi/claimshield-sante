"""Stockage de la mémoire conversationnelle — chat/conversation_store.py.

Plan de remédiation « autonomie décisionnelle V2 », Phase 8. Isolation
stricte par clé composite `(user_id, thread_id)` — jamais par `thread_id`
seul : un `thread_id` ne suffit jamais, à lui seul, à retrouver la
conversation d'un autre utilisateur.

**Limite honnête, documentée plutôt que masquée** : ce projet ne dispose
d'aucune authentification utilisateur réelle. `user_id` est une simple
déclaration de l'appelant (ex. `cl.user_session["actor"]` côté UI
Chainlit), jamais un jeton vérifié cryptographiquement. `ConversationAccessError`
détecte la réutilisation frauduleuse la plus simple à vérifier sans
authentification réelle : un `thread_id` déjà connu sous un `user_id`
différent — ce n'est pas une garantie de sécurité complète, seulement un
garde-fou déterministe contre la collision la plus évidente.

En mémoire uniquement (durée du processus), injectable — même patron que
`DuplicateIndex`/`AuditStore`/`ModelRegistry` : aucune instance globale
cachée. Fenêtre bornée de tours récents (`max_recent_turns`, tronque les
plus anciens, jamais ce schéma). TTL configurable
(`Settings.claimshield_chat_memory_ttl_seconds`) — `expire_older_than()`
retire des conversations **entières** (jamais un tour isolé) au-delà du TTL.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from chat.memory_schemas import ConversationContext, ConversationSemanticState, ConversationTurn, SimulationContext
from chat.schemas import SimulationPatch

__all__ = ["DEFAULT_MAX_RECENT_TURNS", "ConversationAccessError", "ConversationStore"]

DEFAULT_MAX_RECENT_TURNS = 20
"""Fenêtre bornée par défaut — nombre maximal de tours conservés par
conversation (les plus anciens sont retirés en premier, jamais le résumé
sémantique qui les remplace)."""


class ConversationAccessError(Exception):
    """Levée quand un `thread_id` est réclamé sous un `user_id` différent de
    celui qui l'a créé — jamais une conversation silencieusement partagée
    entre deux utilisateurs déclarés différents."""


class ConversationStore:
    """Store injectable, en mémoire — aucune instance globale cachée."""

    def __init__(self, *, max_recent_turns: int = DEFAULT_MAX_RECENT_TURNS) -> None:
        if max_recent_turns < 1:
            raise ValueError("max_recent_turns doit être >= 1")
        self._max_recent_turns = max_recent_turns
        self._contexts: dict[tuple[str, str], ConversationContext] = {}
        self._owner_by_thread: dict[str, str] = {}

    def _check_ownership(self, *, user_id: str, thread_id: str) -> None:
        owner = self._owner_by_thread.get(thread_id)
        if owner is not None and owner != user_id:
            raise ConversationAccessError(
                f"thread_id {thread_id!r} appartient déjà à un autre utilisateur."
            )

    def get(self, *, user_id: str, thread_id: str) -> ConversationContext | None:
        """`None` si la conversation n'existe pas encore (jamais une
        exception pour ce cas normal — seule une réutilisation frauduleuse
        d'un `thread_id` déjà attribué à un autre `user_id` lève)."""
        self._check_ownership(user_id=user_id, thread_id=thread_id)
        return self._contexts.get((user_id, thread_id))

    def append_turn(
        self,
        *,
        user_id: str,
        thread_id: str,
        turn: ConversationTurn,
        simulation: SimulationContext | None = None,
    ) -> ConversationContext:
        """Ajoute un tour — tronque la fenêtre à `max_recent_turns` (les plus
        anciens tours disparaissent, jamais le résumé sémantique, qui les
        remplace précisément pour cette raison). Le résumé sémantique
        existant est préservé tel quel — seul `chat.semantic_summarizer`,
        via `update_semantic_state`, le fait évoluer."""
        self._check_ownership(user_id=user_id, thread_id=thread_id)
        self._owner_by_thread[thread_id] = user_id

        existing = self._contexts.get((user_id, thread_id))
        now = datetime.now(UTC)
        turns = [*(existing.turns if existing else []), turn][-self._max_recent_turns :]
        simulations = list(existing.simulations) if existing else []
        if simulation is not None:
            simulations.append(simulation)

        context = ConversationContext(
            thread_id=thread_id,
            user_id=user_id,
            turns=turns,
            simulations=simulations,
            semantic_state=existing.semantic_state if existing else None,
            updated_at=now,
        )
        self._contexts[(user_id, thread_id)] = context
        return context

    def update_semantic_state(
        self, *, user_id: str, thread_id: str, semantic_state: ConversationSemanticState
    ) -> ConversationContext:
        """Met à jour uniquement le résumé sémantique d'une conversation
        déjà existante — ne modifie jamais `turns`/`simulations`."""
        self._check_ownership(user_id=user_id, thread_id=thread_id)
        existing = self._contexts.get((user_id, thread_id))
        if existing is None:
            raise ConversationAccessError(
                f"Aucune conversation existante pour thread_id {thread_id!r} — "
                "un tour doit d'abord être enregistré via append_turn()."
            )
        updated = existing.model_copy(
            update={"semantic_state": semantic_state, "updated_at": datetime.now(UTC)}
        )
        self._contexts[(user_id, thread_id)] = updated
        return updated

    def update_active_simulation(
        self, *, user_id: str, thread_id: str, patches: list[SimulationPatch]
    ) -> ConversationContext:
        """Remplace `active_simulation_patches` par `patches` (déjà fusionné
        par l'appelant — `chat.agent._merge_simulation_patches` — jamais une
        fusion recalculée ici). Ne modifie jamais `turns`/`simulations`/
        `semantic_state`."""
        self._check_ownership(user_id=user_id, thread_id=thread_id)
        existing = self._contexts.get((user_id, thread_id))
        if existing is None:
            raise ConversationAccessError(
                f"Aucune conversation existante pour thread_id {thread_id!r} — "
                "un tour doit d'abord être enregistré via append_turn()."
            )
        updated = existing.model_copy(
            update={"active_simulation_patches": patches, "updated_at": datetime.now(UTC)}
        )
        self._contexts[(user_id, thread_id)] = updated
        return updated

    def expire_older_than(self, *, ttl_seconds: int) -> int:
        """Retire toute conversation entière (jamais un tour isolé) dont
        `updated_at` dépasse le TTL — retourne le nombre de conversations
        retirées."""
        cutoff = datetime.now(UTC) - timedelta(seconds=ttl_seconds)
        expired_keys = [key for key, ctx in self._contexts.items() if ctx.updated_at < cutoff]
        for key in expired_keys:
            del self._contexts[key]
            self._owner_by_thread.pop(key[1], None)
        return len(expired_keys)
