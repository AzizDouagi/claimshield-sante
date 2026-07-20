"""Tests d'intégration de chat/agent.py — chat.nlu.extract_intent (LLM) et
chat.response_composer._invoke_llm_compose (LLM) mockés, chat.tools.* mocké
directement (le câblage HTTP réel est déjà couvert par
`tests/v2/chat/test_tools.py`/`test_simulation_engine.py`). Corpus de 10+
formulations libres couvrant EXPLAIN/CORRECT/ANALYZE — critère d'acceptation
explicite de la Phase V2-11a (plan de refonte V2 §4). SIMULATE (V2-11b) est
couvert séparément dans `TestSimulate` ci-dessous."""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest
from chat.agent import handle_message, handle_message_streaming
from chat.conversation_store import ConversationAccessError, ConversationStore
from chat.memory_schemas import ConversationSemanticState, ConversationTurn, DiscussedScenario
from chat.schemas import (
    ChatIntent,
    ChatStepStatus,
    ExplanationFacts,
    LlmIntentDecision,
    SimulationChangeRequest,
    SimulationPatch,
    SimulationResult,
)

pytestmark = pytest.mark.asyncio


def _mock_intent(
    monkeypatch,
    intents: list[ChatIntent],
    case_id: str | None = None,
    simulation_changes: SimulationChangeRequest | None = None,
) -> None:
    monkeypatch.setattr(
        "chat.nlu._invoke_llm_intent",
        Mock(
            return_value=LlmIntentDecision(
                intents=intents, case_id=case_id, simulation_changes=simulation_changes
            )
        ),
    )


def _mock_compose_passthrough(monkeypatch) -> None:
    """Compose de façon déterministe sans dépendre du garde-fou
    anti-hallucination (déjà testé séparément) — évite un vrai appel LLM."""
    monkeypatch.setattr(
        "chat.response_composer._invoke_llm_compose",
        Mock(return_value="Réponse groundée de test."),
    )


class TestFreeFormPhrasingsExplain:
    @pytest.mark.parametrize(
        "message",
        [
            "Pourquoi ce dossier est rejeté ?",
            "Explique-moi la décision sur ce dossier.",
            "Quel est le motif du blocage ?",
            "Peux-tu m'expliquer pourquoi le dossier CLM-2001 n'est pas approuvé ?",
        ],
    )
    async def test_explain_phrasings_produce_grounded_reply(self, monkeypatch, message):
        _mock_intent(monkeypatch, [ChatIntent.EXPLAIN], case_id="CLM-2001")
        _mock_compose_passthrough(monkeypatch)
        monkeypatch.setattr(
            "chat.agent.explain_claim",
            AsyncMock(
                return_value=ExplanationFacts(
                    case_id="CLM-2001", final_decision="REJECT", decision_summary=["motif"]
                )
            ),
        )
        reply = await handle_message(message, case_id="CLM-2001")
        assert reply == "Réponse groundée de test."


class TestFreeFormPhrasingsCorrect:
    @pytest.mark.parametrize(
        "message",
        [
            "Qu'est-ce que je dois corriger en premier ?",
            "Que faut-il compléter dans ce dossier ?",
            "Comment débloquer ce dossier ?",
        ],
    )
    async def test_correct_phrasings_produce_grounded_reply(self, monkeypatch, message):
        from chat.schemas import CorrectionRecommendation

        _mock_intent(monkeypatch, [ChatIntent.CORRECT], case_id="CLM-2002")
        _mock_compose_passthrough(monkeypatch)
        monkeypatch.setattr(
            "chat.agent.recommend_corrections",
            AsyncMock(
                return_value=[
                    CorrectionRecommendation(trigger="motif", action="Fournir le document manquant.")
                ]
            ),
        )
        reply = await handle_message(message, case_id="CLM-2002")
        assert reply == "Réponse groundée de test."


class TestFreeFormPhrasingsAnalyze:
    @pytest.mark.parametrize(
        "message",
        [
            "Fais-moi une synthèse de ce dossier.",
            "Où en est ce dossier ?",
            "Analyse ce dossier pour moi.",
        ],
    )
    async def test_analyze_phrasings_produce_grounded_reply(self, monkeypatch, message):
        _mock_intent(monkeypatch, [ChatIntent.ANALYZE], case_id="CLM-2003")
        _mock_compose_passthrough(monkeypatch)
        monkeypatch.setattr(
            "chat.agent.get_claim_context",
            AsyncMock(return_value={"case_id": "CLM-2003", "final_decision": "QUARANTINE"}),
        )
        reply = await handle_message(message, case_id="CLM-2003")
        assert reply == "Réponse groundée de test."


class TestClarifyNeeded:
    async def test_ambiguous_message_asks_for_clarification(self, monkeypatch):
        _mock_intent(monkeypatch, [ChatIntent.CLARIFY_NEEDED])
        reply = await handle_message("bonjour")
        assert "identifiant" in reply.lower()

    async def test_explain_without_case_id_asks_for_clarification(self, monkeypatch):
        _mock_intent(monkeypatch, [ChatIntent.EXPLAIN], case_id=None)
        reply = await handle_message("Pourquoi cette décision ?")
        assert "identifiant" in reply.lower()

    async def test_llm_unavailable_asks_for_clarification(self, monkeypatch):
        monkeypatch.setattr("chat.nlu._invoke_llm_intent", Mock(return_value=None))
        reply = await handle_message("N'importe quoi de confus")
        assert "identifiant" in reply.lower()


class TestSimulate:
    async def test_simulate_without_changes_asks_to_specify(self, monkeypatch):
        _mock_intent(monkeypatch, [ChatIntent.SIMULATE], case_id="CLM-2004")
        reply = await handle_message("Et si le montant était différent ?", case_id="CLM-2004")
        assert "préciser" in reply.lower()

    async def test_simulate_with_remove_document_change_produces_grounded_reply(self, monkeypatch):
        _mock_intent(
            monkeypatch,
            [ChatIntent.SIMULATE],
            case_id="CLM-2008",
            simulation_changes=SimulationChangeRequest(remove_document="ordonnance"),
        )
        _mock_compose_passthrough(monkeypatch)
        spy = AsyncMock(
            return_value=SimulationResult(
                case_id="CLM-2008",
                applied=True,
                original_decision="APPROVE",
                simulated_decision="REQUEST_MORE_INFO",
                decision_changed=True,
            )
        )
        monkeypatch.setattr("chat.agent.simulate_changes", spy)
        reply = await handle_message("Et si on retirait l'ordonnance ?", case_id="CLM-2008")
        assert reply == "Réponse groundée de test."
        spy.assert_called_once_with(
            "CLM-2008", SimulationChangeRequest(remove_document="ordonnance")
        )

    async def test_simulate_failure_still_returns_graceful_reply(self, monkeypatch):
        _mock_intent(
            monkeypatch,
            [ChatIntent.SIMULATE],
            case_id="CLM-2009",
            simulation_changes=SimulationChangeRequest(remove_document="facture"),
        )
        monkeypatch.setattr("chat.response_composer._invoke_llm_compose", Mock(return_value=None))
        monkeypatch.setattr(
            "chat.agent.simulate_changes",
            AsyncMock(
                return_value=SimulationResult(
                    case_id="CLM-2009", applied=False, error="Dossier introuvable."
                )
            ),
        )
        reply = await handle_message("Et si on retirait la facture ?", case_id="CLM-2009")
        assert "impossible" in reply.lower()


class TestAuditAndDraftMessage:
    """AUDIT et DRAFT_MESSAGE (Phase V2-11c) — exécutés réellement, plus
    « bientôt disponible » (voir `tests/v2/chat/test_planner.py::TestPlanNotYetAvailable`,
    qui documente pourquoi ce chemin devient structurellement inatteignable)."""

    async def test_audit_intent_produces_grounded_reply(self, monkeypatch):
        from chat.schemas import AuditSummary

        _mock_intent(monkeypatch, [ChatIntent.AUDIT], case_id="CLM-2005")
        _mock_compose_passthrough(monkeypatch)
        spy = AsyncMock(
            return_value=AuditSummary(
                case_id="CLM-2005",
                event_count=3,
                chain_intact=True,
                event_type_counts={"agent_called": 3},
                actors=["system:intake_safety"],
                issues_count=0,
            )
        )
        monkeypatch.setattr("chat.agent.get_audit_summary", spy)
        reply = await handle_message("Montre-moi l'historique complet.", case_id="CLM-2005")
        assert reply == "Réponse groundée de test."
        spy.assert_called_once_with("CLM-2005")

    async def test_draft_message_intent_produces_grounded_reply(self, monkeypatch):
        _mock_intent(monkeypatch, [ChatIntent.DRAFT_MESSAGE], case_id="CLM-2006")
        monkeypatch.setattr(
            "chat.response_composer._invoke_llm_patient_message",
            Mock(return_value="Réponse patient groundée de test."),
        )
        spy = AsyncMock(return_value={"case_id": "CLM-2006", "final_decision": "APPROVE"})
        monkeypatch.setattr("chat.agent.generate_patient_message", spy)
        reply = await handle_message("Rédige un message pour le patient.", case_id="CLM-2006")
        assert reply == "Réponse patient groundée de test."
        spy.assert_called_once_with("CLM-2006")

    async def test_draft_message_uses_patient_prompt_not_gestionnaire_prompt(self, monkeypatch):
        """Vérifie la bifurcation réelle de `chat/response_composer.py::compose`
        — `_invoke_llm_patient_message` est appelé, jamais `_invoke_llm_compose`,
        quand DRAFT_MESSAGE est l'intention."""
        _mock_intent(monkeypatch, [ChatIntent.DRAFT_MESSAGE], case_id="CLM-2010")
        monkeypatch.setattr(
            "chat.agent.generate_patient_message",
            AsyncMock(return_value={"case_id": "CLM-2010", "final_decision": "APPROVE"}),
        )
        gestionnaire_spy = Mock(return_value="ne devrait jamais être utilisé")
        patient_spy = Mock(return_value="Bonjour, votre dossier a été approuvé.")
        monkeypatch.setattr("chat.response_composer._invoke_llm_compose", gestionnaire_spy)
        monkeypatch.setattr("chat.response_composer._invoke_llm_patient_message", patient_spy)
        reply = await handle_message("Rédige un message pour le patient.", case_id="CLM-2010")
        assert reply == "Bonjour, votre dossier a été approuvé."
        patient_spy.assert_called_once()
        gestionnaire_spy.assert_not_called()


class TestUnknownCase:
    async def test_case_not_found_returns_graceful_message(self, monkeypatch):
        _mock_intent(monkeypatch, [ChatIntent.EXPLAIN], case_id="CLM-2099")
        monkeypatch.setattr("chat.agent.explain_claim", AsyncMock(return_value=None))
        reply = await handle_message("Pourquoi ?", case_id="CLM-2099")
        assert "introuvable" in reply.lower()


class TestConversationMemory:
    """Phase 8 (plan de remédiation « autonomie décisionnelle V2 », §6) —
    mémoire conversationnelle entièrement opt-in : absence de
    `thread_id`/`user_id`/`conversation_store` reproduit exactement le
    comportement d'avant cette phase (couvert par toutes les classes
    ci-dessus, jamais modifiées)."""

    async def test_memory_disabled_when_params_absent_no_store_interaction(self, monkeypatch):
        _mock_intent(monkeypatch, [ChatIntent.EXPLAIN], case_id="CLM-6001")
        _mock_compose_passthrough(monkeypatch)
        monkeypatch.setattr(
            "chat.agent.explain_claim",
            AsyncMock(return_value=ExplanationFacts(case_id="CLM-6001", final_decision="APPROVE")),
        )
        store = ConversationStore()
        reply = await handle_message("Pourquoi ?", case_id="CLM-6001")
        assert reply == "Réponse groundée de test."
        # Store jamais touché puisqu'il n'a même pas été transmis.
        assert store.get(user_id="anyone", thread_id="anyone") is None

    async def test_memory_disabled_when_only_thread_id_given(self, monkeypatch):
        """Les trois paramètres mémoire doivent être fournis ensemble —
        jamais un état partiel silencieusement actif."""
        _mock_intent(monkeypatch, [ChatIntent.EXPLAIN], case_id="CLM-6002")
        _mock_compose_passthrough(monkeypatch)
        monkeypatch.setattr(
            "chat.agent.explain_claim",
            AsyncMock(return_value=ExplanationFacts(case_id="CLM-6002", final_decision="APPROVE")),
        )
        store = ConversationStore()
        reply = await handle_message("Pourquoi ?", case_id="CLM-6002", thread_id="thread-1")
        assert reply == "Réponse groundée de test."
        assert store.get(user_id="alice", thread_id="thread-1") is None

    async def test_turn_recorded_with_digest_never_raw_message(self, monkeypatch):
        _mock_intent(monkeypatch, [ChatIntent.EXPLAIN], case_id="CLM-6003")
        _mock_compose_passthrough(monkeypatch)
        monkeypatch.setattr(
            "chat.agent.explain_claim",
            AsyncMock(return_value=ExplanationFacts(case_id="CLM-6003", final_decision="APPROVE")),
        )
        store = ConversationStore()
        secret_message = "Pourquoi le dossier CLM-6003 est-il refusé, ma sécu est 123-45-6789 ?"
        await handle_message(
            secret_message, case_id="CLM-6003", thread_id="thread-1", user_id="alice", conversation_store=store
        )
        context = store.get(user_id="alice", thread_id="thread-1")
        assert context is not None
        assert len(context.turns) == 1
        turn = context.turns[0]
        assert secret_message not in turn.model_dump_json()
        assert turn.message_digest != secret_message
        assert ChatIntent.EXPLAIN in turn.intents
        assert turn.case_id == "CLM-6003"

    async def test_conversation_access_error_propagates(self, monkeypatch):
        _mock_intent(monkeypatch, [ChatIntent.EXPLAIN], case_id="CLM-6004")
        _mock_compose_passthrough(monkeypatch)
        monkeypatch.setattr(
            "chat.agent.explain_claim",
            AsyncMock(return_value=ExplanationFacts(case_id="CLM-6004", final_decision="APPROVE")),
        )
        store = ConversationStore()
        await handle_message(
            "Pourquoi ?", case_id="CLM-6004", thread_id="thread-1", user_id="alice", conversation_store=store
        )
        with pytest.raises(ConversationAccessError):
            await handle_message(
                "Pourquoi ?", case_id="CLM-6004", thread_id="thread-1", user_id="mallory", conversation_store=store
            )

    async def test_semantic_state_updated_after_turn(self, monkeypatch):
        _mock_intent(monkeypatch, [ChatIntent.EXPLAIN], case_id="CLM-6005")
        _mock_compose_passthrough(monkeypatch)
        monkeypatch.setattr(
            "chat.agent.explain_claim",
            AsyncMock(return_value=ExplanationFacts(case_id="CLM-6005", final_decision="APPROVE")),
        )
        new_state = ConversationSemanticState(
            conversation_summary="Le dossier CLM-6005 a été approuvé.",
            updated_at=datetime.now(UTC),
        )
        summarizer_spy = Mock(return_value=new_state)
        monkeypatch.setattr("chat.agent.update_semantic_state", summarizer_spy)
        store = ConversationStore()
        await handle_message(
            "Pourquoi ?", case_id="CLM-6005", thread_id="thread-1", user_id="alice", conversation_store=store
        )
        summarizer_spy.assert_called_once()
        context = store.get(user_id="alice", thread_id="thread-1")
        assert context.semantic_state == new_state

    async def test_memory_failure_never_breaks_the_reply(self, monkeypatch):
        """Une panne de la couche mémoire (bug/LLM de résumé) ne doit
        jamais faire échouer une réponse déjà composée."""
        _mock_intent(monkeypatch, [ChatIntent.EXPLAIN], case_id="CLM-6006")
        _mock_compose_passthrough(monkeypatch)
        monkeypatch.setattr(
            "chat.agent.explain_claim",
            AsyncMock(return_value=ExplanationFacts(case_id="CLM-6006", final_decision="APPROVE")),
        )
        monkeypatch.setattr(
            "chat.agent.update_semantic_state", Mock(side_effect=RuntimeError("panne inattendue"))
        )
        store = ConversationStore()
        reply = await handle_message(
            "Pourquoi ?", case_id="CLM-6006", thread_id="thread-1", user_id="alice", conversation_store=store
        )
        assert reply == "Réponse groundée de test."

    async def test_resolved_scenario_included_in_composed_context(self, monkeypatch):
        _mock_intent(monkeypatch, [ChatIntent.EXPLAIN], case_id="CLM-6007")
        monkeypatch.setattr(
            "chat.agent.explain_claim",
            AsyncMock(return_value=ExplanationFacts(case_id="CLM-6007", final_decision="APPROVE")),
        )
        compose_spy = Mock(return_value="Réponse groundée de test.")
        monkeypatch.setattr("chat.response_composer._invoke_llm_compose", compose_spy)

        store = ConversationStore()
        scenario = DiscussedScenario(
            scenario_id="SCENARIO-1", description="Premier scénario discuté.", kind="REAL_DECISION"
        )
        store.append_turn(
            user_id="alice",
            thread_id="thread-1",
            turn=ConversationTurn(
                turn_id="t0", message_digest="a" * 64, reply_digest="b" * 64, created_at=datetime.now(UTC)
            ),
        )
        store.update_semantic_state(
            user_id="alice",
            thread_id="thread-1",
            semantic_state=ConversationSemanticState(
                conversation_summary="résumé",
                discussed_scenarios=[scenario],
                resolved_references={"le premier scénario": "SCENARIO-1"},
                updated_at=datetime.now(UTC),
            ),
        )

        monkeypatch.setattr("chat.nlu._resolve_scenario_reference", Mock(return_value="SCENARIO-1"))
        await handle_message(
            "Compare avec le premier scénario",
            case_id="CLM-6007",
            thread_id="thread-1",
            user_id="alice",
            conversation_store=store,
        )
        composed_data = compose_spy.call_args[0][0]
        assert composed_data.get("resolved_scenario") is not None
        assert composed_data["resolved_scenario"]["scenario_id"] == "SCENARIO-1"


class TestActiveSimulationAccumulation:
    """Phase 9 (plan de remédiation « autonomie décisionnelle V2 », §7) —
    les patches d'une simulation ciblée s'accumulent pour un même thread
    (« et si on changeait aussi... ») plutôt que de repartir à chaque
    message d'un dossier réel non modifié."""

    def _patch(self, field: str, value: str) -> SimulationPatch:
        return SimulationPatch(field=field, value=value)

    async def test_first_targeted_simulation_has_no_previous_patches_to_merge(self, monkeypatch):
        _mock_intent(monkeypatch, [ChatIntent.SIMULATE], case_id="CLM-9001")
        _mock_compose_passthrough(monkeypatch)
        sim_spy = AsyncMock(
            return_value=SimulationResult(
                case_id="CLM-9001", applied=True, original_decision="REJECT", simulated_decision="APPROVE",
                decision_changed=True,
            )
        )
        monkeypatch.setattr("chat.agent.simulate_changes", sim_spy)
        store = ConversationStore()

        changes = SimulationChangeRequest(field_patches=[self._patch("COVERAGE_STATUS", "PASS")])
        _mock_intent(monkeypatch, [ChatIntent.SIMULATE], case_id="CLM-9001", simulation_changes=changes)
        await handle_message(
            "Et si la couverture était confirmée ?",
            case_id="CLM-9001",
            thread_id="thread-1",
            user_id="alice",
            conversation_store=store,
        )

        sent_changes = sim_spy.call_args[0][1]
        assert len(sent_changes.field_patches) == 1
        assert sent_changes.field_patches[0].field.value == "COVERAGE_STATUS"

        context = store.get(user_id="alice", thread_id="thread-1")
        assert len(context.active_simulation_patches) == 1

    async def test_second_targeted_simulation_merges_with_active_patches(self, monkeypatch):
        _mock_compose_passthrough(monkeypatch)
        sim_spy = AsyncMock(
            return_value=SimulationResult(
                case_id="CLM-9002", applied=True, original_decision="REJECT", simulated_decision="APPROVE",
                decision_changed=True,
            )
        )
        monkeypatch.setattr("chat.agent.simulate_changes", sim_spy)
        store = ConversationStore()

        first_changes = SimulationChangeRequest(field_patches=[self._patch("COVERAGE_STATUS", "PASS")])
        _mock_intent(monkeypatch, [ChatIntent.SIMULATE], case_id="CLM-9002", simulation_changes=first_changes)
        await handle_message(
            "Et si la couverture était confirmée ?",
            case_id="CLM-9002",
            thread_id="thread-1",
            user_id="alice",
            conversation_store=store,
        )

        second_changes = SimulationChangeRequest(field_patches=[self._patch("IDENTITY_STATUS", "PASS")])
        _mock_intent(monkeypatch, [ChatIntent.SIMULATE], case_id="CLM-9002", simulation_changes=second_changes)
        await handle_message(
            "Et si en plus l'identité était confirmée ?",
            case_id="CLM-9002",
            thread_id="thread-1",
            user_id="alice",
            conversation_store=store,
        )

        sent_changes = sim_spy.call_args[0][1]
        sent_fields = {p.field.value for p in sent_changes.field_patches}
        assert sent_fields == {"COVERAGE_STATUS", "IDENTITY_STATUS"}

        context = store.get(user_id="alice", thread_id="thread-1")
        assert len(context.active_simulation_patches) == 2

    async def test_new_patch_on_same_field_replaces_the_old_one(self, monkeypatch):
        _mock_compose_passthrough(monkeypatch)
        sim_spy = AsyncMock(
            return_value=SimulationResult(
                case_id="CLM-9003", applied=True, original_decision="REJECT", simulated_decision="REJECT",
                decision_changed=False,
            )
        )
        monkeypatch.setattr("chat.agent.simulate_changes", sim_spy)
        store = ConversationStore()

        first_changes = SimulationChangeRequest(field_patches=[self._patch("COVERAGE_STATUS", "PASS")])
        _mock_intent(monkeypatch, [ChatIntent.SIMULATE], case_id="CLM-9003", simulation_changes=first_changes)
        await handle_message(
            "Et si la couverture était confirmée ?",
            case_id="CLM-9003", thread_id="thread-1", user_id="alice", conversation_store=store,
        )

        second_changes = SimulationChangeRequest(field_patches=[self._patch("COVERAGE_STATUS", "FAIL")])
        _mock_intent(monkeypatch, [ChatIntent.SIMULATE], case_id="CLM-9003", simulation_changes=second_changes)
        await handle_message(
            "En fait, et si la couverture était refusée ?",
            case_id="CLM-9003", thread_id="thread-1", user_id="alice", conversation_store=store,
        )

        sent_changes = sim_spy.call_args[0][1]
        assert len(sent_changes.field_patches) == 1
        assert sent_changes.field_patches[0].value == "FAIL"

    async def test_no_memory_never_accumulates(self, monkeypatch):
        """Sans mémoire, chaque simulation ciblée est indépendante — jamais
        une accumulation implicite."""
        _mock_compose_passthrough(monkeypatch)
        sim_spy = AsyncMock(
            return_value=SimulationResult(
                case_id="CLM-9004", applied=True, original_decision="REJECT", simulated_decision="REJECT",
                decision_changed=False,
            )
        )
        monkeypatch.setattr("chat.agent.simulate_changes", sim_spy)

        changes = SimulationChangeRequest(field_patches=[self._patch("COVERAGE_STATUS", "PASS")])
        _mock_intent(monkeypatch, [ChatIntent.SIMULATE], case_id="CLM-9004", simulation_changes=changes)
        await handle_message("Et si la couverture était confirmée ?", case_id="CLM-9004")

        sent_changes = sim_spy.call_args[0][1]
        assert len(sent_changes.field_patches) == 1


class TestCallerCaseIdWinsOverMessage:
    async def test_caller_case_id_used_even_if_message_mentions_another(self, monkeypatch):
        """Le `case_id` transmis explicitement par l'appelant (contexte
        UI) est toujours utilisé, jamais un identifiant halluciné/detecté
        différent dans le texte libre."""
        _mock_intent(monkeypatch, [ChatIntent.EXPLAIN], case_id="CLM-2007")
        _mock_compose_passthrough(monkeypatch)
        spy = AsyncMock(
            return_value=ExplanationFacts(case_id="CLM-2007", final_decision="APPROVE")
        )
        monkeypatch.setattr("chat.agent.explain_claim", spy)
        await handle_message("Explique-moi le dossier CLM-9999", case_id="CLM-2007")
        spy.assert_called_once_with("CLM-2007")


class TestStreamingStepEvents:
    """Visibilité temps réel des étapes/tokens (demandée par AZIZ, « comme
    Claude Code ») — `handle_message_streaming`/`_run_turn(on_step=...)`.
    `handle_message` (sans callback) reste inchangée, déjà couverte par
    toutes les classes ci-dessus sans aucune modification."""

    async def test_comprehension_step_started_then_completed(self, monkeypatch):
        _mock_intent(monkeypatch, [ChatIntent.EXPLAIN], case_id="CLM-3001")
        _mock_compose_passthrough(monkeypatch)
        monkeypatch.setattr(
            "chat.agent.explain_claim",
            AsyncMock(return_value=ExplanationFacts(case_id="CLM-3001", final_decision="APPROVE")),
        )
        events = []

        async def on_step(event):
            events.append(event)

        await handle_message_streaming("Pourquoi ce dossier ?", case_id="CLM-3001", on_step=on_step)

        comprehension_events = [e for e in events if e.step_name == "comprehension"]
        assert [e.status for e in comprehension_events] == [ChatStepStatus.STARTED, ChatStepStatus.COMPLETED]

    async def test_tool_and_composition_steps_emitted_in_order(self, monkeypatch):
        _mock_intent(monkeypatch, [ChatIntent.EXPLAIN], case_id="CLM-3002")
        _mock_compose_passthrough(monkeypatch)
        monkeypatch.setattr(
            "chat.agent.explain_claim",
            AsyncMock(return_value=ExplanationFacts(case_id="CLM-3002", final_decision="APPROVE")),
        )
        events = []

        async def on_step(event):
            events.append(event)

        await handle_message_streaming("Explique.", case_id="CLM-3002", on_step=on_step)

        step_names_in_order = [e.step_name for e in events]
        assert step_names_in_order == [
            "comprehension",
            "comprehension",
            "outil_explain",
            "outil_explain",
            "composition",
            "composition",
        ]

    async def test_failed_comprehension_step_when_llm_unavailable(self, monkeypatch):
        monkeypatch.setattr("chat.nlu._invoke_llm_intent", Mock(return_value=None))
        events = []

        async def on_step(event):
            events.append(event)

        reply = await handle_message_streaming("n'importe quoi", on_step=on_step)

        assert reply  # message de clarification, comportement inchangé
        assert len(events) == 2
        assert events[0].status == ChatStepStatus.STARTED
        assert events[1].status == ChatStepStatus.FAILED

    async def test_composition_step_never_emitted_when_case_not_found(self, monkeypatch):
        _mock_intent(monkeypatch, [ChatIntent.EXPLAIN], case_id="CLM-3003")
        monkeypatch.setattr("chat.agent.explain_claim", AsyncMock(return_value=None))
        events = []

        async def on_step(event):
            events.append(event)

        await handle_message_streaming("Explique.", case_id="CLM-3003", on_step=on_step)

        assert "composition" not in [e.step_name for e in events]

    async def test_no_callback_reproduces_handle_message_exactly(self, monkeypatch):
        """`on_step=None` (donc `handle_message`) ne doit strictement rien
        changer au comportement — même résultat qu'avant cette
        fonctionnalité, sans jamais construire le moindre `ChatStepEvent`."""
        _mock_intent(monkeypatch, [ChatIntent.EXPLAIN], case_id="CLM-3004")
        _mock_compose_passthrough(monkeypatch)
        monkeypatch.setattr(
            "chat.agent.explain_claim",
            AsyncMock(return_value=ExplanationFacts(case_id="CLM-3004", final_decision="APPROVE")),
        )
        reply = await handle_message("Explique.", case_id="CLM-3004")
        assert reply == "Réponse groundée de test."

    async def test_simulate_step_labelled_targeted_vs_full(self, monkeypatch):
        _mock_intent(
            monkeypatch,
            [ChatIntent.SIMULATE],
            case_id="CLM-3005",
            simulation_changes=SimulationChangeRequest(remove_document="ordonnance"),
        )
        _mock_compose_passthrough(monkeypatch)
        monkeypatch.setattr(
            "chat.agent.simulate_changes",
            AsyncMock(
                return_value=SimulationResult(
                    case_id="CLM-3005", applied=True, original_decision="APPROVE", simulated_decision="APPROVE"
                )
            ),
        )
        events = []

        async def on_step(event):
            events.append(event)

        await handle_message_streaming(
            "Et si on retirait l'ordonnance ?", case_id="CLM-3005", on_step=on_step
        )

        simulate_events = [e for e in events if e.step_name == "outil_simulate"]
        assert len(simulate_events) == 2
        assert "complète" in simulate_events[0].label.lower()
        assert simulate_events[0].input_tokens is None
        assert simulate_events[0].output_tokens is None
