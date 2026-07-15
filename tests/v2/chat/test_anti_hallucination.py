"""Tests du garde-fou anti-hallucination — chat/response_composer.py (plan
V2 §6, Phase V2-11a, jamais reporté à une sous-phase ultérieure).

Toute réponse LLM citant un montant ou une date absent des données
groundées transmises aux outils est rejetée et remplacée par une
composition déterministe de repli (`_fallback_compose`) — jamais un fait
inventé transmis au gestionnaire."""
from __future__ import annotations

from unittest.mock import Mock

from chat.response_composer import compose
from chat.schemas import ChatIntent


class TestAmountHallucination:
    def test_amount_absent_from_context_rejects_llm_reply(self, monkeypatch):
        monkeypatch.setattr(
            "chat.response_composer._invoke_llm_compose",
            Mock(return_value="Le montant remboursé est de 999.99 USD."),
        )
        tool_results = {"context": {"case_id": "CLM-1005", "final_decision": "APPROVE"}}
        result = compose(case_id="CLM-1005", intents=[ChatIntent.EXPLAIN], tool_results=tool_results)
        assert "999.99" not in result
        assert "CLM-1005" in result  # repli déterministe toujours groundé

    def test_amount_present_in_context_is_allowed(self, monkeypatch):
        monkeypatch.setattr(
            "chat.response_composer._invoke_llm_compose",
            Mock(return_value="Le montant demandé est de 123.45 USD."),
        )
        tool_results = {
            "context": {
                "case_id": "CLM-1006",
                "final_decision": "APPROVE",
                "decision_summary": ["Montant demandé : 123.45 USD."],
            }
        }
        result = compose(case_id="CLM-1006", intents=[ChatIntent.EXPLAIN], tool_results=tool_results)
        assert result == "Le montant demandé est de 123.45 USD."

    def test_multiple_unknown_amounts_all_rejected(self, monkeypatch):
        monkeypatch.setattr(
            "chat.response_composer._invoke_llm_compose",
            Mock(return_value="Montants de 111.11 USD et 222.22 USD non couverts."),
        )
        tool_results = {"context": {"case_id": "CLM-1010"}}
        result = compose(case_id="CLM-1010", intents=[ChatIntent.ANALYZE], tool_results=tool_results)
        assert "111.11" not in result
        assert "222.22" not in result

    def test_one_unknown_amount_among_known_ones_still_rejects_entirely(self, monkeypatch):
        """Un seul montant halluciné parmi plusieurs montants réels rejette
        toute la réponse — jamais une correction partielle qui laisserait
        passer le reste d'une réponse déjà compromise."""
        monkeypatch.setattr(
            "chat.response_composer._invoke_llm_compose",
            Mock(return_value="Montant demandé 123.45 USD, remboursé 999.99 USD."),
        )
        tool_results = {
            "context": {
                "case_id": "CLM-1012",
                "decision_summary": ["Montant demandé : 123.45 USD."],
            }
        }
        result = compose(case_id="CLM-1012", intents=[ChatIntent.EXPLAIN], tool_results=tool_results)
        assert "999.99" not in result
        assert "123.45" not in result  # réponse LLM entièrement écartée, pas éditée


class TestDateHallucination:
    def test_date_iso_absent_from_context_rejects_llm_reply(self, monkeypatch):
        monkeypatch.setattr(
            "chat.response_composer._invoke_llm_compose",
            Mock(return_value="Les soins ont eu lieu le 2026-01-15."),
        )
        tool_results = {"context": {"case_id": "CLM-1007", "final_decision": "REJECT"}}
        result = compose(case_id="CLM-1007", intents=[ChatIntent.EXPLAIN], tool_results=tool_results)
        assert "2026-01-15" not in result

    def test_date_slash_format_absent_from_context_rejects_llm_reply(self, monkeypatch):
        monkeypatch.setattr(
            "chat.response_composer._invoke_llm_compose",
            Mock(return_value="Document daté du 15/01/2026."),
        )
        tool_results = {"context": {"case_id": "CLM-1008", "final_decision": "REJECT"}}
        result = compose(case_id="CLM-1008", intents=[ChatIntent.EXPLAIN], tool_results=tool_results)
        assert "15/01/2026" not in result

    def test_date_present_in_context_is_allowed(self, monkeypatch):
        monkeypatch.setattr(
            "chat.response_composer._invoke_llm_compose",
            Mock(return_value="Le dossier date du 2026-01-15."),
        )
        tool_results = {
            "context": {
                "case_id": "CLM-1009",
                "final_decision": "APPROVE",
                "decision_summary": ["Date de soins : 2026-01-15"],
            }
        }
        result = compose(case_id="CLM-1009", intents=[ChatIntent.EXPLAIN], tool_results=tool_results)
        assert result == "Le dossier date du 2026-01-15."


class TestDraftMessageHallucination:
    """`generate_patient_message` (Phase V2-11c) utilise un prompt distinct
    (`_invoke_llm_patient_message`) mais partage exactement le même
    post-check anti-hallucination — critère d'acceptation explicite de
    V2-11c : aucun message patient généré ne cite un montant/date absent
    du dossier."""

    def test_amount_absent_from_context_rejects_patient_message(self, monkeypatch):
        monkeypatch.setattr(
            "chat.response_composer._invoke_llm_patient_message",
            Mock(return_value="Vous serez remboursé de 999.99 USD prochainement."),
        )
        tool_results = {
            "patient_message_context": {"case_id": "CLM-1020", "final_decision": "APPROVE"}
        }
        result = compose(
            case_id="CLM-1020", intents=[ChatIntent.DRAFT_MESSAGE], tool_results=tool_results
        )
        assert "999.99" not in result
        assert "CLM-1020" in result  # repli déterministe toujours groundé

    def test_date_absent_from_context_rejects_patient_message(self, monkeypatch):
        monkeypatch.setattr(
            "chat.response_composer._invoke_llm_patient_message",
            Mock(return_value="Votre remboursement sera versé le 2026-02-01."),
        )
        tool_results = {
            "patient_message_context": {"case_id": "CLM-1021", "final_decision": "APPROVE"}
        }
        result = compose(
            case_id="CLM-1021", intents=[ChatIntent.DRAFT_MESSAGE], tool_results=tool_results
        )
        assert "2026-02-01" not in result

    def test_amount_present_in_context_is_allowed_in_patient_message(self, monkeypatch):
        monkeypatch.setattr(
            "chat.response_composer._invoke_llm_patient_message",
            Mock(return_value="Le montant demandé était de 123.45 USD."),
        )
        tool_results = {
            "patient_message_context": {
                "case_id": "CLM-1022",
                "final_decision": "APPROVE",
                "decision_summary": ["Montant demandé : 123.45 USD."],
            }
        }
        result = compose(
            case_id="CLM-1022", intents=[ChatIntent.DRAFT_MESSAGE], tool_results=tool_results
        )
        assert result == "Le montant demandé était de 123.45 USD."

    def test_gestionnaire_prompt_never_invoked_for_draft_message(self, monkeypatch):
        """Bifurcation réelle vers le prompt patient — le prompt
        gestionnaire n'est jamais sollicité pour DRAFT_MESSAGE, même en cas
        de rejet par le post-check."""
        gestionnaire_spy = Mock(return_value="ne devrait jamais être appelé")
        monkeypatch.setattr("chat.response_composer._invoke_llm_compose", gestionnaire_spy)
        monkeypatch.setattr(
            "chat.response_composer._invoke_llm_patient_message",
            Mock(return_value="Montant halluciné : 555.55 USD."),
        )
        tool_results = {"patient_message_context": {"case_id": "CLM-1023"}}
        compose(case_id="CLM-1023", intents=[ChatIntent.DRAFT_MESSAGE], tool_results=tool_results)
        gestionnaire_spy.assert_not_called()

    def test_patient_message_llm_unavailable_falls_back_gracefully(self, monkeypatch):
        monkeypatch.setattr(
            "chat.response_composer._invoke_llm_patient_message", Mock(return_value=None)
        )
        tool_results = {
            "patient_message_context": {"case_id": "CLM-1024", "final_decision": "REJECT"}
        }
        result = compose(
            case_id="CLM-1024", intents=[ChatIntent.DRAFT_MESSAGE], tool_results=tool_results
        )
        assert "CLM-1024" in result


class TestNonBusinessTokensNeverFlagged:
    def test_identifiers_never_treated_as_hallucination(self, monkeypatch):
        """Les identifiants (CLM-XXXX, EVID-...) ne sont jamais des cibles
        du post-check — ce ne sont pas des montants/dates métier."""
        monkeypatch.setattr(
            "chat.response_composer._invoke_llm_compose",
            Mock(return_value="Voir la preuve EVID-abc123 du dossier CLM-1011."),
        )
        tool_results = {"context": {"case_id": "CLM-1011"}}
        result = compose(case_id="CLM-1011", intents=[ChatIntent.ANALYZE], tool_results=tool_results)
        assert result == "Voir la preuve EVID-abc123 du dossier CLM-1011."

    def test_reply_with_no_number_like_token_never_rejected(self, monkeypatch):
        monkeypatch.setattr(
            "chat.response_composer._invoke_llm_compose",
            Mock(return_value="Le dossier a été rejeté faute de preuve de couverture."),
        )
        tool_results = {"context": {"case_id": "CLM-1013"}}
        result = compose(case_id="CLM-1013", intents=[ChatIntent.EXPLAIN], tool_results=tool_results)
        assert result == "Le dossier a été rejeté faute de preuve de couverture."
