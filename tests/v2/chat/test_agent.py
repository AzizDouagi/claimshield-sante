"""Tests d'intégration de chat/agent.py — chat.nlu.extract_intent (LLM) et
chat.response_composer._invoke_llm_compose (LLM) mockés, chat.tools.* mocké
directement (le câblage HTTP réel est déjà couvert par
`tests/v2/chat/test_tools.py`/`test_simulation_engine.py`). Corpus de 10+
formulations libres couvrant EXPLAIN/CORRECT/ANALYZE — critère d'acceptation
explicite de la Phase V2-11a (plan de refonte V2 §4). SIMULATE (V2-11b) est
couvert séparément dans `TestSimulate` ci-dessous."""
from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from chat.agent import handle_message
from chat.schemas import (
    ChatIntent,
    ExplanationFacts,
    LlmIntentDecision,
    SimulationChangeRequest,
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
