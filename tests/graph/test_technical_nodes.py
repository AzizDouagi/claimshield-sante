"""Tests unitaires des nœuds techniques — ClaimShield Santé.

Chaque classe couvre un nœud : quarantine, needs_review, failure, finalize.
Les tests vérifient exclusivement les mises à jour partielles produites,
sans aucun mock LLM ni appel d'agent.

Propriétés vérifiées par nœud
------------------------------
- current_step — valeur correcte
- completed_steps — liste contenant exactement le nom de l'étape
- alerts / errors — présence, absence, contenu minimal
- final_recommendation — présence (failure) ou absence (autres)
- Aucune clé ``*_result`` dans la mise à jour
- ``validate_state_update`` passe sans lever d'exception
- case_id par défaut ("INCONNU") si absent de l'état
"""
from __future__ import annotations

import pytest

from graph.nodes import NODE_REGISTRY
from graph.technical_nodes import (
    TECHNICAL_NODE_REGISTRY,
    _TechnicalNodeConfig,
    _make_technical_node,
    node_failure,
    node_finalize,
    node_needs_review,
    node_quarantine,
)
from schemas.domain import Recommendation
from state.claim_state import validate_state_update

# ── Helpers ───────────────────────────────────────────────────────────────────

_RESULT_KEYS = {
    "intake_result", "security_result", "privacy_result", "fhir_result",
    "ocr_result", "coding_result", "clinical_result", "fraud_result",
    "review_result", "audit_result", "identity_coverage_result",
}


def _state(case_id: str = "CLM-TEST") -> dict:
    return {"case_id": case_id}


def _empty_state() -> dict:
    return {}


# ── TestNodeQuarantine ────────────────────────────────────────────────────────


class TestNodeQuarantine:
    def test_current_step(self):
        result = node_quarantine(_state())
        assert result["current_step"] == "quarantine"

    def test_completed_steps_contains_step(self):
        result = node_quarantine(_state())
        assert result["completed_steps"] == ["quarantine"]

    def test_alert_is_present(self):
        result = node_quarantine(_state())
        assert "alerts" in result
        assert len(result["alerts"]) == 1

    def test_alert_contains_case_id(self):
        result = node_quarantine(_state("CLM-0042"))
        assert "CLM-0042" in result["alerts"][0]

    def test_alert_fallback_when_no_case_id(self):
        result = node_quarantine(_empty_state())
        assert "INCONNU" in result["alerts"][0]

    def test_no_error(self):
        result = node_quarantine(_state())
        assert result.get("errors", []) == []

    def test_no_final_recommendation(self):
        assert "final_recommendation" not in node_quarantine(_state())

    def test_no_result_key(self):
        result = node_quarantine(_state())
        assert _RESULT_KEYS.isdisjoint(result.keys())

    def test_validate_state_update_passes(self):
        result = node_quarantine(_state())
        validate_state_update(result)  # ne lève pas

    def test_node_name(self):
        assert node_quarantine.__name__ == "node_quarantine"


# ── TestNodeNeedsReview ───────────────────────────────────────────────────────


class TestNodeNeedsReview:
    def test_current_step(self):
        result = node_needs_review(_state())
        assert result["current_step"] == "needs_review"

    def test_completed_steps_contains_step(self):
        result = node_needs_review(_state())
        assert result["completed_steps"] == ["needs_review"]

    def test_alert_is_present(self):
        result = node_needs_review(_state())
        assert "alerts" in result
        assert len(result["alerts"]) == 1

    def test_alert_contains_case_id(self):
        result = node_needs_review(_state("CLM-0007"))
        assert "CLM-0007" in result["alerts"][0]

    def test_alert_fallback_when_no_case_id(self):
        result = node_needs_review(_empty_state())
        assert "INCONNU" in result["alerts"][0]

    def test_no_error(self):
        result = node_needs_review(_state())
        assert result.get("errors", []) == []

    def test_no_final_recommendation(self):
        assert "final_recommendation" not in node_needs_review(_state())

    def test_no_result_key(self):
        result = node_needs_review(_state())
        assert _RESULT_KEYS.isdisjoint(result.keys())

    def test_validate_state_update_passes(self):
        validate_state_update(node_needs_review(_state()))

    def test_node_name(self):
        assert node_needs_review.__name__ == "node_needs_review"


# ── TestNodeFailure ───────────────────────────────────────────────────────────


class TestNodeFailure:
    def test_current_step(self):
        result = node_failure(_state())
        assert result["current_step"] == "failure"

    def test_completed_steps_contains_step(self):
        result = node_failure(_state())
        assert result["completed_steps"] == ["failure"]

    def test_error_is_present(self):
        result = node_failure(_state())
        assert "errors" in result
        assert len(result["errors"]) == 1

    def test_error_contains_case_id(self):
        result = node_failure(_state("CLM-0099"))
        assert "CLM-0099" in result["errors"][0]

    def test_error_fallback_when_no_case_id(self):
        result = node_failure(_empty_state())
        assert "INCONNU" in result["errors"][0]

    def test_no_alert(self):
        result = node_failure(_state())
        assert result.get("alerts", []) == []

    def test_final_recommendation_is_reject(self):
        result = node_failure(_state())
        assert result["final_recommendation"] is Recommendation.REJECT

    def test_no_result_key(self):
        result = node_failure(_state())
        assert _RESULT_KEYS.isdisjoint(result.keys())

    def test_validate_state_update_passes(self):
        validate_state_update(node_failure(_state()))

    def test_node_name(self):
        assert node_failure.__name__ == "node_failure"


# ── TestNodeFinalize ──────────────────────────────────────────────────────────


class TestNodeFinalize:
    def test_current_step(self):
        result = node_finalize(_state())
        assert result["current_step"] == "finalize"

    def test_completed_steps_contains_step(self):
        result = node_finalize(_state())
        assert result["completed_steps"] == ["finalize"]

    def test_no_alert(self):
        result = node_finalize(_state())
        assert result.get("alerts", []) == []

    def test_no_error(self):
        result = node_finalize(_state())
        assert result.get("errors", []) == []

    def test_no_final_recommendation(self):
        # case_reviewer a déjà fixé la recommandation — finalize ne l'écrase pas
        assert "final_recommendation" not in node_finalize(_state())

    def test_no_result_key(self):
        result = node_finalize(_state())
        assert _RESULT_KEYS.isdisjoint(result.keys())

    def test_validate_state_update_passes(self):
        validate_state_update(node_finalize(_state()))

    def test_node_name(self):
        assert node_finalize.__name__ == "node_finalize"

    def test_does_not_depend_on_case_id(self):
        # finalize n'insère aucun message — case_id irrelevant
        result_with = node_finalize(_state("CLM-XYZ"))
        result_without = node_finalize(_empty_state())
        assert result_with == result_without


# ── TestTechnicalNodeRegistry ─────────────────────────────────────────────────


class TestTechnicalNodeRegistry:
    EXPECTED_KEYS = {"quarantine", "needs_review", "failure", "finalize"}

    def test_registry_contains_all_expected_keys(self):
        assert set(TECHNICAL_NODE_REGISTRY.keys()) == self.EXPECTED_KEYS

    def test_registry_all_callable(self):
        for name, fn in TECHNICAL_NODE_REGISTRY.items():
            assert callable(fn), f"TECHNICAL_NODE_REGISTRY[{name!r}] n'est pas callable"

    def test_technical_nodes_absent_from_agent_registry(self):
        agent_keys = set(NODE_REGISTRY.keys())
        overlap = self.EXPECTED_KEYS & agent_keys
        assert not overlap, f"Nœuds techniques trouvés dans NODE_REGISTRY : {overlap}"

    def test_registry_nodes_match_module_attributes(self):
        import graph.technical_nodes as mod
        for name, fn in TECHNICAL_NODE_REGISTRY.items():
            assert fn is getattr(mod, f"node_{name}")


# ── TestTechnicalNodeConfig ───────────────────────────────────────────────────


class TestTechnicalNodeConfig:
    def test_frozen_immutable(self):
        cfg = _TechnicalNodeConfig(step_name="test")
        with pytest.raises((AttributeError, TypeError)):
            cfg.step_name = "autre"  # type: ignore[misc]

    def test_defaults(self):
        cfg = _TechnicalNodeConfig(step_name="x")
        assert cfg.alert is None
        assert cfg.error is None
        assert cfg.final_recommendation is None

    def test_make_technical_node_respects_config(self):
        cfg = _TechnicalNodeConfig(
            step_name="custom",
            alert="alerte {case_id}",
            error="erreur {case_id}",
            final_recommendation=Recommendation.REJECT,
        )
        fn = _make_technical_node(cfg)
        result = fn({"case_id": "CLM-X"})
        assert result["current_step"] == "custom"
        assert result["alerts"] == ["alerte CLM-X"]
        assert result["errors"] == ["erreur CLM-X"]
        assert result["final_recommendation"] is Recommendation.REJECT

    def test_make_technical_node_no_imports_from_agents(self):
        import inspect
        import graph.technical_nodes as mod
        src = inspect.getsource(mod)
        assert "from agents." not in src
        assert "import agents." not in src
