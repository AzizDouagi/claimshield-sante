"""Tests de chat/explanation_engine.py et chat/correction_engine.py — fonctions
pures, aucun mock requis."""
from __future__ import annotations

from chat.correction_engine import build_corrections
from chat.explanation_engine import build_explanation_facts
from chat.schemas import ExplanationFacts


class TestBuildExplanationFacts:
    def test_all_fields_reused_from_context(self):
        context = {
            "case_id": "CLM-1001",
            "final_decision": "REJECT",
            "decision_summary": ["motif 1", "motif 2"],
            "bounded_by": ["garde-fou 1"],
            "errors": ["erreur 1"],
            "alerts": ["alerte 1"],
        }
        facts = build_explanation_facts(context)
        assert facts == ExplanationFacts(
            case_id="CLM-1001",
            final_decision="REJECT",
            decision_summary=["motif 1", "motif 2"],
            bounded_by=["garde-fou 1"],
            errors=["erreur 1"],
            alerts=["alerte 1"],
        )

    def test_missing_optional_fields_default_empty(self):
        facts = build_explanation_facts({"case_id": "CLM-1002"})
        assert facts.final_decision is None
        assert facts.decision_summary == []
        assert facts.bounded_by == []
        assert facts.errors == []
        assert facts.alerts == []

    def test_never_invents_a_fact_not_in_context(self):
        context = {"case_id": "CLM-1003", "final_decision": "APPROVE"}
        facts = build_explanation_facts(context)
        assert facts.errors == []
        assert facts.bounded_by == []

    def test_explainability_fields_reused_from_context(self):
        """Phase 7 (plan de remédiation « autonomie décisionnelle V2 ») —
        `missing_information`/`assumptions`/`decisive_factors`/
        `counterfactuals`/`recommended_action`/`evidence_completeness`
        arrivent comme des dicts JSON déjà sérialisés (réponse API) et sont
        revalidés par Pydantic à la construction, jamais un nouveau calcul."""
        context = {
            "case_id": "CLM-1004",
            "missing_information": [
                {
                    "code": "UNRESOLVED_CODING",
                    "description": "Codification non résolue.",
                    "importance": "IMPORTANT",
                    "affected_dimension": "CODING",
                    "source_agent": "medical_risk_agent",
                    "impact_on_decision": "Confiance réduite.",
                }
            ],
            "assumptions": [{"code": "X", "description": "Hypothèse retenue."}],
            "decisive_factors": [
                {"code": "IDENTITY_CONFIRMED", "description": "Identité confirmée.", "source_agent": "eligibility_agent"}
            ],
            "counterfactuals": [
                {
                    "condition": "Code résolu",
                    "current_value": "Non résolu",
                    "required_value": "Résolu",
                    "resulting_decision": "APPROVE",
                    "explanation": "Un code résolu permettrait une approbation complète.",
                }
            ],
            "recommended_action": "Vérifier la codification.",
            "evidence_completeness": "PARTIAL",
        }
        facts = build_explanation_facts(context)
        assert len(facts.missing_information) == 1
        assert facts.missing_information[0].code == "UNRESOLVED_CODING"
        assert len(facts.assumptions) == 1
        assert len(facts.decisive_factors) == 1
        assert len(facts.counterfactuals) == 1
        assert facts.recommended_action == "Vérifier la codification."
        assert facts.evidence_completeness.value == "PARTIAL"

    def test_explainability_fields_default_empty(self):
        facts = build_explanation_facts({"case_id": "CLM-1005"})
        assert facts.missing_information == []
        assert facts.assumptions == []
        assert facts.decisive_factors == []
        assert facts.counterfactuals == []
        assert facts.recommended_action == ""
        assert facts.evidence_completeness is None


class TestBuildCorrections:
    def test_payer_name_absent_triggers_recommendation(self):
        context = {"errors": ["[eligibility] Assureur (payer_name) absent ou vide"], "alerts": []}
        recs = build_corrections(context)
        assert len(recs) == 1
        assert "assureur" in recs[0].action.lower()

    def test_patient_name_absent_triggers_recommendation(self):
        context = {"errors": ["[eligibility] Nom patient absent ou vide"], "alerts": []}
        recs = build_corrections(context)
        assert any("nom du patient" in r.action.lower() for r in recs)

    def test_no_matching_keyword_yields_empty_list(self):
        context = {"errors": ["[unknown] Motif totalement générique sans mot-clé connu"], "alerts": []}
        recs = build_corrections(context)
        assert recs == []

    def test_empty_context_yields_empty_list(self):
        assert build_corrections({}) == []
        assert build_corrections({"errors": [], "alerts": []}) == []

    def test_same_action_never_duplicated(self):
        context = {
            "errors": [
                "[eligibility] Assureur (payer_name) absent ou vide",
                "[eligibility] Assureur (payer_name) absent ou vide — couverture non vérifiable",
            ],
            "alerts": [],
        }
        recs = build_corrections(context)
        actions = {r.action for r in recs}
        assert len(actions) == len(recs)

    def test_multiple_distinct_triggers_each_produce_a_recommendation(self):
        context = {
            "errors": [
                "[eligibility] Assureur (payer_name) absent ou vide",
                "[eligibility] Nom patient absent ou vide",
            ],
            "alerts": ["[medical_risk] Codification non résolue"],
        }
        recs = build_corrections(context)
        assert len(recs) >= 2

    def test_recommendations_only_come_from_the_fixed_table(self):
        """Toute action recommandée doit provenir de la table déterministe —
        jamais une phrase absente de `_CORRECTION_TABLE`."""
        from chat.correction_engine import _CORRECTION_TABLE

        known_actions = {action for _, action in _CORRECTION_TABLE}
        context = {
            "errors": [
                "[eligibility] Assureur (payer_name) absent ou vide",
                "[medical_risk] Codification non résolue",
                "[document_understanding] date impossible détectée",
            ],
            "alerts": [],
        }
        recs = build_corrections(context)
        assert all(r.action in known_actions for r in recs)

    def test_trigger_field_preserves_original_text(self):
        original = "[eligibility] Nom patient absent ou vide"
        context = {"errors": [original], "alerts": []}
        recs = build_corrections(context)
        assert recs[0].trigger == original
