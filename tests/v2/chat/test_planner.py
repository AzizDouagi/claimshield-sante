"""Tests de chat/planner.py (Phase V2-11a/V2-11b/V2-11c) — fonction pure, aucun mock requis."""
from __future__ import annotations

from chat.planner import _SUPPORTED_INTENTS, plan
from chat.schemas import ChatIntent, ChatPlanAction, NluResult, SimulationChangeRequest


class TestPlanExecute:
    def test_explain_with_case_id_executes(self):
        result = plan(NluResult(intents=[ChatIntent.EXPLAIN], case_id="CLM-1001"))
        assert result.action is ChatPlanAction.EXECUTE
        assert result.case_id == "CLM-1001"
        assert result.intents == [ChatIntent.EXPLAIN]

    def test_analyze_with_case_id_executes(self):
        result = plan(NluResult(intents=[ChatIntent.ANALYZE], case_id="CLM-1002"))
        assert result.action is ChatPlanAction.EXECUTE

    def test_correct_with_case_id_executes(self):
        result = plan(NluResult(intents=[ChatIntent.CORRECT], case_id="CLM-1003"))
        assert result.action is ChatPlanAction.EXECUTE

    def test_simulate_with_case_id_executes(self):
        result = plan(
            NluResult(
                intents=[ChatIntent.SIMULATE],
                case_id="CLM-1010",
                simulation_changes=SimulationChangeRequest(remove_document="ordonnance"),
            )
        )
        assert result.action is ChatPlanAction.EXECUTE
        assert result.simulation_changes == SimulationChangeRequest(remove_document="ordonnance")

    def test_simulate_without_simulation_changes_still_executes(self):
        """`simulation_changes is None` est un cas géré par `chat/agent.py`
        (message demandant de préciser le changement) — le planner ne
        bloque pas cette combinaison, seul `case_id` manquant le fait."""
        result = plan(NluResult(intents=[ChatIntent.SIMULATE], case_id="CLM-1011"))
        assert result.action is ChatPlanAction.EXECUTE
        assert result.simulation_changes is None

    def test_multiple_supported_intents_all_kept(self):
        result = plan(
            NluResult(intents=[ChatIntent.ANALYZE, ChatIntent.EXPLAIN], case_id="CLM-1004")
        )
        assert result.action is ChatPlanAction.EXECUTE
        assert set(result.intents) == {ChatIntent.ANALYZE, ChatIntent.EXPLAIN}

    def test_audit_with_case_id_executes(self):
        """AUDIT livré en V2-11c — plus « bientôt disponible »."""
        result = plan(NluResult(intents=[ChatIntent.AUDIT], case_id="CLM-1006"))
        assert result.action is ChatPlanAction.EXECUTE
        assert result.intents == [ChatIntent.AUDIT]

    def test_draft_message_with_case_id_executes(self):
        """DRAFT_MESSAGE livré en V2-11c — plus « bientôt disponible »."""
        result = plan(NluResult(intents=[ChatIntent.DRAFT_MESSAGE], case_id="CLM-1007"))
        assert result.action is ChatPlanAction.EXECUTE
        assert result.intents == [ChatIntent.DRAFT_MESSAGE]

    def test_mixed_audit_and_explain_both_execute(self):
        """Un message mêlant EXPLAIN et AUDIT exécute désormais les deux —
        les deux sont livrés depuis V2-11c."""
        result = plan(
            NluResult(intents=[ChatIntent.EXPLAIN, ChatIntent.AUDIT], case_id="CLM-1008")
        )
        assert result.action is ChatPlanAction.EXECUTE
        assert set(result.intents) == {ChatIntent.EXPLAIN, ChatIntent.AUDIT}
        assert result.unsupported_intents == []


class TestPlanClarifyNeeded:
    def test_nlu_none_yields_clarify_needed(self):
        result = plan(None)
        assert result.action is ChatPlanAction.CLARIFY_NEEDED
        assert result.case_id is None

    def test_explicit_clarify_needed_intent(self):
        result = plan(NluResult(intents=[ChatIntent.CLARIFY_NEEDED]))
        assert result.action is ChatPlanAction.CLARIFY_NEEDED

    def test_supported_intent_without_case_id_yields_clarify_needed(self):
        result = plan(NluResult(intents=[ChatIntent.EXPLAIN], case_id=None))
        assert result.action is ChatPlanAction.CLARIFY_NEEDED

    def test_simulate_without_case_id_yields_clarify_needed(self):
        result = plan(NluResult(intents=[ChatIntent.SIMULATE], case_id=None))
        assert result.action is ChatPlanAction.CLARIFY_NEEDED

    def test_never_invents_a_case_id(self):
        """Aucune combinaison d'entrées ne peut produire un `case_id` non
        None sur une action différente de EXECUTE."""
        for intents in (
            [ChatIntent.EXPLAIN],
            [ChatIntent.ANALYZE],
            [ChatIntent.CORRECT],
            [ChatIntent.SIMULATE],
        ):
            result = plan(NluResult(intents=intents, case_id=None))
            if result.action is not ChatPlanAction.EXECUTE:
                assert result.case_id is None


class TestPlanNotYetAvailable:
    """Depuis V2-11c, les 6 intentions non-CLARIFY_NEEDED de `ChatIntent`
    sont toutes livrées (`_SUPPORTED_INTENTS`) — `NOT_YET_AVAILABLE` n'est
    donc plus jamais atteignable via une valeur réelle de l'énumération.
    Le chemin de code reste présent dans `chat/planner.py::plan()` par
    prudence défensive (jamais supprimé), mais ce n'est plus un
    comportement fonctionnel testable avec les valeurs actuelles de
    `ChatIntent` — seule la structure est vérifiée ici."""

    def test_all_non_clarify_intents_are_supported(self):
        all_intents = {i for i in ChatIntent if i is not ChatIntent.CLARIFY_NEEDED}
        assert _SUPPORTED_INTENTS == all_intents

    def test_not_yet_available_path_unreachable_with_real_enum_values(self):
        """Aucune combinaison de vraies valeurs `ChatIntent` (hors
        CLARIFY_NEEDED, intercepté avant) ne peut plus produire
        `NOT_YET_AVAILABLE` — contre-preuve directe sur les 6 valeurs
        livrées, individuellement et combinées."""
        for intent in ChatIntent:
            if intent is ChatIntent.CLARIFY_NEEDED:
                continue
            result = plan(NluResult(intents=[intent], case_id="CLM-2000"))
            assert result.action is not ChatPlanAction.NOT_YET_AVAILABLE

        result = plan(
            NluResult(
                intents=[i for i in ChatIntent if i is not ChatIntent.CLARIFY_NEEDED],
                case_id="CLM-2001",
            )
        )
        assert result.action is ChatPlanAction.EXECUTE
        assert result.unsupported_intents == []
