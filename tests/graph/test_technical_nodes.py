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

from graph.nodes import build_node_registry, build_orchestrator
from graph.technical_nodes import (
    ALLOWED_HUMAN_ACTIONS,
    TECHNICAL_NODE_REGISTRY,
    _build_human_review_payload,
    _collect_minimized_evidence,
    _collect_motifs,
    _TechnicalNodeConfig,
    _make_technical_node,
    node_await_human_review,
    node_failure,
    node_finalize,
    node_needs_review,
    node_quarantine,
)
from human_review.service import HumanDecisionValidationError
from schemas.domain import Recommendation
from state.claim_state import ClaimState, validate_state_update

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


# ── TestNodeAwaitHumanReview ──────────────────────────────────────────────────


class TestHumanReviewHelpers:
    def test_build_payload_contains_required_keys(self):
        payload = _build_human_review_payload(_state("CLM-0055"))
        assert payload["case_id"] == "CLM-0055"
        assert isinstance(payload["motifs"], list) and payload["motifs"]
        assert isinstance(payload["preuves_minimisees"], dict)
        assert payload["actions_autorisees"] == list(ALLOWED_HUMAN_ACTIONS)

    def test_build_payload_fallback_case_id(self):
        payload = _build_human_review_payload(_empty_state())
        assert payload["case_id"] == "INCONNU"

    def test_collect_motifs_uses_alerts_then_errors(self):
        state = {"alerts": ["confiance OCR limite"], "errors": ["rôle inconnu"]}
        motifs = _collect_motifs(state)
        assert motifs == ["confiance OCR limite", "rôle inconnu"]

    def test_collect_motifs_deduplicates(self):
        state = {"alerts": ["x", "x"], "errors": []}
        assert _collect_motifs(state) == ["x"]

    def test_collect_motifs_default_when_empty(self):
        assert _collect_motifs(_empty_state()) == [
            "Revue humaine requise — aucun motif spécifique enregistré."
        ]

    def test_collect_minimized_evidence_extracts_status(self):
        from schemas.domain import VerificationStatus
        from types import SimpleNamespace

        state = {"ocr_result": SimpleNamespace(status=VerificationStatus.NEEDS_REVIEW)}
        evidence = _collect_minimized_evidence(state)
        assert evidence == {"ocr_result": "NEEDS_REVIEW"}

    def test_collect_minimized_evidence_ignores_missing_results(self):
        assert _collect_minimized_evidence(_empty_state()) == {}

    # La validation de la décision humaine (justification obligatoire,
    # actions/target_node autorisés) n'est plus une fonction maison de ce
    # module : ``node_await_human_review`` délègue entièrement à
    # ``human_review.service.validate_and_audit_human_decision`` — voir
    # ``tests/human_review/test_service.py``/``test_interrupt_resume.py``
    # pour la couverture exhaustive de cette validation (Pydantic stricte).
    # Les tests ci-dessous (``TestNodeAwaitHumanReview``) vérifient
    # uniquement l'intégration bout-en-bout dans le nœud LangGraph réel.


class TestNodeAwaitHumanReview:
    """Le nœud appelle interrupt() — testé via un graphe LangGraph minimal.

    Appeler node_await_human_review() directement, hors contexte d'exécution
    LangGraph, lève RuntimeError (interrupt() a besoin du runtime du graphe) :
    ce comportement est vérifié explicitement plutôt que contourné.
    """

    def test_direct_call_outside_graph_raises(self):
        with pytest.raises(RuntimeError):
            node_await_human_review(_state("CLM-DIRECT"))

    def test_node_name(self):
        assert node_await_human_review.__name__ == "node_await_human_review"

    def _build_app(self):
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(ClaimState)
        graph.add_node("await_human_review", node_await_human_review)
        graph.add_edge(START, "await_human_review")
        graph.add_edge("await_human_review", END)
        return graph.compile(checkpointer=InMemorySaver())

    def test_interrupts_with_expected_payload(self):
        app = self._build_app()
        config = {"configurable": {"thread_id": "CLM-INT-1", "checkpoint_ns": ""}}
        result = app.invoke(
            {"case_id": "CLM-INT-1", "alerts": ["confiance limite"], "errors": []},
            config=config,
        )
        assert "__interrupt__" in result
        payload = result["__interrupt__"][0].value
        assert payload["case_id"] == "CLM-INT-1"
        assert payload["motifs"] == ["confiance limite"]
        assert payload["actions_autorisees"] == list(ALLOWED_HUMAN_ACTIONS)

    def test_simple_interrupt_stores_case_id_thread_id_review_result_and_actions(self):
        """Test d'interruption simple : le payload minimal transmis à
        ``interrupt()`` porte bien case_id, thread_id, review_result (même
        absent) et les actions autorisées — rien de plus n'est requis pour
        qu'un humain sache quel dossier revoir, sur quel thread reprendre et
        avec quelles actions."""
        app = self._build_app()
        config = {"configurable": {"thread_id": "CLM-SIMPLE-1", "checkpoint_ns": ""}}
        result = app.invoke(
            {"case_id": "CLM-SIMPLE-1", "alerts": [], "errors": []},
            config=config,
        )

        assert "__interrupt__" in result
        payload = result["__interrupt__"][0].value
        assert payload["case_id"] == "CLM-SIMPLE-1"
        assert payload["thread_id"] == "CLM-SIMPLE-1"
        assert payload["review_result"] is None
        assert payload["actions_autorisees"] == list(ALLOWED_HUMAN_ACTIONS)

    def test_interrupt_payload_includes_review_result_summary_when_available(self):
        from schemas.results import CaseReviewerResult, CaseReviewerResultPayload, LlmMetadata

        app = self._build_app()
        config = {"configurable": {"thread_id": "CLM-SIMPLE-2", "checkpoint_ns": ""}}
        review_result = CaseReviewerResult(
            case_id="CLM-SIMPLE-2",
            llm_trace=LlmMetadata(model_name="test-llm", prompt_version="test"),
            result_payload=CaseReviewerResultPayload(
                recommendation=Recommendation.PENDING,
                justification=["Motif de synthèse."],
                risks=["Score de risque de fraude élevé (0.75)."],
                human_review_reasons=["Validation humaine obligatoire avant toute décision finale."],
            ),
        )
        result = app.invoke(
            {
                "case_id": "CLM-SIMPLE-2",
                "alerts": [],
                "errors": [],
                "review_result": review_result,
            },
            config=config,
        )

        payload = result["__interrupt__"][0].value
        assert payload["review_result"] == {
            "recommendation": "PENDING",
            "justification": ["Motif de synthèse."],
            "risks": ["Score de risque de fraude élevé (0.75)."],
        }

    def test_resume_with_same_thread_id_completes(self):
        from langgraph.types import Command

        app = self._build_app()
        config = {"configurable": {"thread_id": "CLM-9002", "checkpoint_ns": ""}}
        app.invoke({"case_id": "CLM-9002", "alerts": [], "errors": []}, config=config)

        result = app.invoke(
            Command(
                resume={
                    "actor": "reviewer@example.com",
                    "action": "APPROVE",
                    "justification": "Dossier conforme.",
                }
            ),
            config=config,
        )
        assert "__interrupt__" not in result
        assert result["human_decision"]["action"] == "APPROVE"
        assert result["completed_steps"] == ["await_human_review"]

    def test_resume_produces_an_audit_event(self):
        """La décision humaine est désormais auditée — voir
        ``human_review.service.build_human_decision_audit_event``."""
        from langgraph.types import Command

        app = self._build_app()
        config = {"configurable": {"thread_id": "CLM-9010", "checkpoint_ns": ""}}
        app.invoke({"case_id": "CLM-9010", "alerts": [], "errors": []}, config=config)

        result = app.invoke(
            Command(
                resume={
                    "actor": "reviewer@example.com",
                    "action": "APPROVE",
                    "justification": "Dossier conforme.",
                }
            ),
            config=config,
        )
        assert len(result["audit_trail"]) == 1
        event = result["audit_trail"][0]
        assert event.outcome == "APPROVE"
        assert event.details["justification"] == "Dossier conforme."

    def test_resume_needs_more_info_increments_correction_attempts(self):
        from langgraph.types import Command

        app = self._build_app()
        config = {"configurable": {"thread_id": "CLM-9003", "checkpoint_ns": ""}}
        app.invoke({"case_id": "CLM-9003", "alerts": [], "errors": []}, config=config)

        result = app.invoke(
            Command(
                resume={
                    "actor": "reviewer@example.com",
                    "action": "RETRY",
                    "justification": "Pièce manquante.",
                    "target_node": "document_ocr",
                }
            ),
            config=config,
        )
        assert result["correction_attempts"] == 1
        assert result["human_decision"]["target_node"] == "document_ocr"

    def test_resume_approve_does_not_touch_correction_attempts(self):
        from langgraph.types import Command

        app = self._build_app()
        config = {"configurable": {"thread_id": "CLM-9004", "checkpoint_ns": ""}}
        app.invoke(
            {"case_id": "CLM-9004", "alerts": [], "errors": [], "correction_attempts": 2},
            config=config,
        )

        result = app.invoke(
            Command(
                resume={
                    "actor": "reviewer@example.com",
                    "action": "APPROVE",
                    "justification": "Dossier conforme.",
                }
            ),
            config=config,
        )
        assert result["correction_attempts"] == 2  # inchangé — APPROVE ne consomme pas d'essai

    def test_correction_attempts_accumulates_across_relaunches(self):
        """Compteur minimal : chaque RETRY incrémente à partir de
        la valeur déjà présente dans le state (pas de remise à zéro)."""
        from langgraph.types import Command

        app = self._build_app()
        config = {"configurable": {"thread_id": "CLM-9005", "checkpoint_ns": ""}}
        app.invoke(
            {"case_id": "CLM-9005", "alerts": [], "errors": [], "correction_attempts": 2},
            config=config,
        )

        result = app.invoke(
            Command(
                resume={
                    "actor": "reviewer@example.com",
                    "action": "RETRY",
                    "justification": "Pièce manquante.",
                    "target_node": "document_ocr",
                }
            ),
            config=config,
        )
        assert result["correction_attempts"] == 3

    def test_resume_with_different_thread_id_does_not_resume(self):
        """Reprendre avec un thread_id différent ne retrouve pas l'interruption
        en attente : LangGraph ne trouve aucun checkpoint pour ce thread et
        redémarre une exécution indépendante depuis START, qui réinterrompt
        aussitôt sur un state vide — la décision fournie n'est jamais
        appliquée au dossier interrompu. Ceci matérialise l'exigence de
        stabilité du thread_id entre l'interruption et la reprise.
        """
        from langgraph.types import Command

        app = self._build_app()
        config = {"configurable": {"thread_id": "CLM-INT-3", "checkpoint_ns": ""}}
        app.invoke({"case_id": "CLM-INT-3", "alerts": [], "errors": []}, config=config)

        other_config = {"configurable": {"thread_id": "CLM-INT-OTHER", "checkpoint_ns": ""}}
        result = app.invoke(
            Command(
                resume={
                    "actor": "reviewer@example.com",
                    "action": "APPROVE",
                    "justification": "Dossier conforme.",
                }
            ),
            config=other_config,
        )
        assert "__interrupt__" in result
        assert result["__interrupt__"][0].value["case_id"] == "INCONNU"
        assert "human_decision" not in result

    def test_resume_with_invalid_decision_raises(self):
        from langgraph.types import Command

        app = self._build_app()
        config = {"configurable": {"thread_id": "CLM-INT-4", "checkpoint_ns": ""}}
        app.invoke({"case_id": "CLM-INT-4", "alerts": [], "errors": []}, config=config)

        with pytest.raises(HumanDecisionValidationError):
            app.invoke(
                Command(resume={"actor": "a", "action": "MAYBE", "justification": "Motif."}),
                config=config,
            )

    def test_resume_without_justification_raises(self):
        """Justification obligatoire : une décision par ailleurs valide mais
        sans justification est refusée, jamais acceptée par défaut."""
        from langgraph.types import Command

        app = self._build_app()
        config = {"configurable": {"thread_id": "CLM-INT-NOJUST", "checkpoint_ns": ""}}
        app.invoke({"case_id": "CLM-INT-NOJUST", "alerts": [], "errors": []}, config=config)

        with pytest.raises(HumanDecisionValidationError):
            app.invoke(Command(resume={"actor": "a", "action": "APPROVE"}), config=config)

    def test_never_reaches_end_without_valid_human_decision(self):
        """Garantie centrale : une décision humaine invalide ne fait jamais
        progresser le graphe — ni ``human_decision`` ni ``current_step`` ne
        sont mis à jour, et le nœud reste en échec (jamais un ``END``
        silencieux ni une valeur par défaut acceptée à sa place)."""
        from langgraph.types import Command

        app = self._build_app()
        config = {"configurable": {"thread_id": "CLM-INT-5", "checkpoint_ns": ""}}
        app.invoke({"case_id": "CLM-INT-5", "alerts": [], "errors": []}, config=config)

        with pytest.raises(HumanDecisionValidationError):
            app.invoke(
                Command(resume={"actor": "a", "action": "MAYBE", "justification": "Motif."}),
                config=config,
            )

        state_after_invalid = app.get_state(config)
        assert state_after_invalid.values.get("human_decision") is None
        assert "current_step" not in state_after_invalid.values
        assert "__end__" not in state_after_invalid.values
        assert len(state_after_invalid.tasks) == 1
        assert state_after_invalid.tasks[0].error is not None

    def test_valid_decision_on_a_fresh_thread_reaches_end(self):
        """Contrepreuve : sur un dossier distinct, une décision valide dès la
        reprise fait bien progresser le graphe jusqu'à ``END`` — la garantie
        ci-dessus ne bloque que les décisions invalides, jamais les valides."""
        from langgraph.types import Command

        app = self._build_app()
        config = {"configurable": {"thread_id": "CLM-9006", "checkpoint_ns": ""}}
        app.invoke({"case_id": "CLM-9006", "alerts": [], "errors": []}, config=config)

        result = app.invoke(
            Command(
                resume={"actor": "a", "action": "APPROVE", "justification": "Dossier conforme."}
            ),
            config=config,
        )
        assert "__interrupt__" not in result
        assert result["human_decision"]["action"] == "APPROVE"
        assert result["current_step"] == "await_human_review"


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
    EXPECTED_KEYS = {
        "quarantine", "needs_review", "await_human_review", "failure", "finalize",
        "verification_fan_in", "consistency_fan_in",
    }

    def test_registry_contains_all_expected_keys(self):
        assert set(TECHNICAL_NODE_REGISTRY.keys()) == self.EXPECTED_KEYS

    def test_registry_all_callable(self):
        for name, fn in TECHNICAL_NODE_REGISTRY.items():
            assert callable(fn), f"TECHNICAL_NODE_REGISTRY[{name!r}] n'est pas callable"

    def test_technical_nodes_absent_from_agent_registry(self):
        agent_keys = set(build_node_registry(build_orchestrator()).keys())
        overlap = self.EXPECTED_KEYS & agent_keys
        assert not overlap, f"Nœuds techniques trouvés dans le registre d'agents : {overlap}"

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
