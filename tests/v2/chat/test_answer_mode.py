"""Tests de chat/answer_mode.py — plan de remédiation « autonomie
décisionnelle V2 », Phase 7. Fonction pure, aucun mock requis."""
from __future__ import annotations

from chat.answer_mode import AnswerMode, detect_answer_modes
from chat.schemas import ChatIntent, ExplanationFacts, SimulationResult
from schemas.v2_results import DecisionAssumption, MissingInformation, MissingInformationDimension, MissingInformationImportance


def _explanation(**overrides) -> ExplanationFacts:
    defaults = dict(case_id="CLM-7201", final_decision="APPROVE")
    defaults.update(overrides)
    return ExplanationFacts(**defaults)


def _missing_information() -> MissingInformation:
    return MissingInformation(
        code="UNRESOLVED_CODING",
        description="Codification non résolue.",
        importance=MissingInformationImportance.IMPORTANT,
        affected_dimension=MissingInformationDimension.CODING,
        source_agent="medical_risk_agent",
        impact_on_decision="Confiance réduite.",
    )


def _assumption() -> DecisionAssumption:
    return DecisionAssumption(code="X", description="Hypothèse retenue.")


class TestFactMode:
    def test_context_alone_yields_fact(self):
        modes = detect_answer_modes(intents=[ChatIntent.ANALYZE], tool_results={"context": {"case_id": "CLM-7201"}})
        assert modes == [AnswerMode.FACT]

    def test_explanation_without_gaps_yields_fact_only(self):
        modes = detect_answer_modes(
            intents=[ChatIntent.EXPLAIN], tool_results={"explanation": _explanation()}
        )
        assert modes == [AnswerMode.FACT]

    def test_empty_tool_results_falls_back_to_fact(self):
        modes = detect_answer_modes(intents=[ChatIntent.ANALYZE], tool_results={})
        assert modes == [AnswerMode.FACT]


class TestAssumptionMode:
    def test_missing_information_triggers_assumption(self):
        modes = detect_answer_modes(
            intents=[ChatIntent.EXPLAIN],
            tool_results={"explanation": _explanation(missing_information=[_missing_information()])},
        )
        assert AnswerMode.ASSUMPTION in modes
        assert AnswerMode.FACT in modes

    def test_assumptions_field_triggers_assumption(self):
        modes = detect_answer_modes(
            intents=[ChatIntent.EXPLAIN],
            tool_results={"explanation": _explanation(assumptions=[_assumption()])},
        )
        assert AnswerMode.ASSUMPTION in modes


class TestSimulationMode:
    def test_applied_simulation_triggers_simulation_mode(self):
        simulation = SimulationResult(case_id="CLM-7201", applied=True, original_decision="APPROVE", simulated_decision="REJECT", decision_changed=True)
        modes = detect_answer_modes(intents=[ChatIntent.SIMULATE], tool_results={"simulation": simulation})
        assert AnswerMode.SIMULATION in modes

    def test_unapplied_simulation_never_triggers_simulation_mode(self):
        simulation = SimulationResult(case_id="CLM-7201", applied=False, error="dossier introuvable")
        modes = detect_answer_modes(intents=[ChatIntent.SIMULATE], tool_results={"simulation": simulation})
        assert AnswerMode.SIMULATION not in modes

    def test_simulation_never_presented_as_fact_only(self):
        """Une simulation appliquée reste toujours étiquetée SIMULATION,
        jamais uniquement FACT — même si son résultat contient des données
        déjà connues (décision réelle actuelle, par exemple)."""
        simulation = SimulationResult(case_id="CLM-7201", applied=True, original_decision="APPROVE", simulated_decision="APPROVE", decision_changed=False)
        modes = detect_answer_modes(intents=[ChatIntent.SIMULATE], tool_results={"simulation": simulation})
        assert AnswerMode.SIMULATION in modes


class TestMultipleModes:
    def test_simulation_and_assumption_can_coexist(self):
        simulation = SimulationResult(case_id="CLM-7201", applied=True, original_decision="APPROVE", simulated_decision="REJECT", decision_changed=True)
        explanation = _explanation(missing_information=[_missing_information()])
        modes = detect_answer_modes(
            intents=[ChatIntent.EXPLAIN, ChatIntent.SIMULATE],
            tool_results={"simulation": simulation, "explanation": explanation},
        )
        assert set(modes) == {AnswerMode.SIMULATION, AnswerMode.ASSUMPTION, AnswerMode.FACT}

    def test_never_invents_a_mode_absent_from_data(self):
        """Aucune combinaison de données ne produit jamais un mode non
        justifié par le contenu réel des résultats d'outils."""
        modes = detect_answer_modes(intents=[ChatIntent.CORRECT], tool_results={"corrections": []})
        assert modes == [AnswerMode.FACT]
