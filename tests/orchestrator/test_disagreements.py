"""Tests d'intégration — désaccords entre résultats d'agents orchestrés.

Combine deux mécanismes déjà testés isolément ailleurs :
  - ``Orchestrator.execute_agent()`` (``orchestrator/executor.py``) — produit
    les résultats d'agents à partir d'exécuteurs synthétiques, sans aucun
    appel LLM réel ;
  - ``tools.consistency``/``graph.edges.route_result_consistency`` — détecte
    les désaccords génériques entre ces résultats et route vers
    ``needs_review`` en cas de désaccord critique.

Deux scénarios : résultats compatibles (accord) puis résultats explicitement
contradictoires (désaccord critique). Vérifie que le désaccord critique
produit bien ``needs_review`` avec une preuve minimale (référence
agent/champ/valeurs, jamais de contenu métier), et qu'aucune décision
médicale ou financière finale (recommandation, montant de couverture, statut
de fraude...) n'est jamais générée par ce mécanisme — il ne fait que
signaler, jamais arbitrer.

Aucun appel LLM : les exécuteurs d'agents enregistrés sont de simples
fonctions Python retournant des résultats Pydantic construits à la main.
"""
from __future__ import annotations

from orchestrator.executor import Orchestrator
from orchestrator.model_registry import ModelRegistry
from orchestrator.orchestrator import AgentCallRequest, AgentName
from orchestrator.policies import PolicyDecision, PolicyEffect
from schemas.domain import VerificationStatus
from schemas.results import DisagreementPoint, FhirValidatorResult, FraudDetectionResult, StructuredError
from tools.consistency import detect_result_disagreements, has_critical_disagreement
from graph.edges import CONTINUE, NEEDS_REVIEW, route_result_consistency


# ── Helpers ────────────────────────────────────────────────────────────────────


def _allow(code: str = "OK") -> PolicyDecision:
    return PolicyDecision(effect=PolicyEffect.ALLOW, reason=StructuredError(code=code, message="ok"))


def _request(agent_name: AgentName, current_step: str, requested_model: str) -> AgentCallRequest:
    return AgentCallRequest(
        agent_name=agent_name,
        case_id="CLM-0001",
        current_step=current_step,
        requested_model=requested_model,
        attempt=1,
    )


def _permissive_orchestrator(agent_registry: dict) -> Orchestrator:
    """Orchestrateur dont préconditions/modèle/outils autorisent toujours —
    isole le comportement à tester (production de résultats synthétiques,
    jamais un appel LLM) des contrôles de permission déjà testés ailleurs."""
    return Orchestrator(
        model_registry=ModelRegistry(),
        agent_registry=agent_registry,
        preconditions_check=lambda state, request: _allow(),
        model_check=lambda registry, agent_name, model_id: _allow(),
        tools_check=lambda agent_name: (),
    )


def _run_synthetic_agent(orchestrator: Orchestrator, agent_name: AgentName, current_step: str, requested_model: str):
    """Exécute un agent synthétique via l'orchestrateur et retourne son
    ``AgentCallOutcome`` — jamais d'appel réseau ni de modèle LLM réel."""
    request = _request(agent_name, current_step, requested_model)
    state = {"case_id": "CLM-0001", "current_step": current_step}
    return orchestrator.execute_agent(request, state, model_id="synthetic-model")


def _fhir_runner(status: VerificationStatus):
    def _runner(state: dict) -> dict:
        return {
            "fhir_result": FhirValidatorResult(
                case_id=state["case_id"], status=status, bundle_expected=True
            )
        }

    return _runner


def _fraud_runner(status: VerificationStatus):
    def _runner(state: dict) -> dict:
        return {"fraud_result": FraudDetectionResult(case_id=state["case_id"], status=status)}

    return _runner


def _build_state_from_synthetic_results(fhir_status: VerificationStatus, fraud_status: VerificationStatus) -> dict:
    """Fait tourner deux agents synthétiques (fhir_validator, fraud_detection)
    via l'orchestrateur puis assemble un ``ClaimState`` minimal à partir de
    leurs résultats — exactement la forme que ``graph/nodes.py`` produirait,
    sans jamais invoquer de LLM."""
    fhir_orchestrator = _permissive_orchestrator(
        {AgentName.FHIR_VALIDATOR: _fhir_runner(fhir_status)}
    )
    fraud_orchestrator = _permissive_orchestrator(
        {AgentName.FRAUD_DETECTION: _fraud_runner(fraud_status)}
    )

    fhir_outcome = _run_synthetic_agent(fhir_orchestrator, AgentName.FHIR_VALIDATOR, "privacy", "FhirValidatorResult")
    fraud_outcome = _run_synthetic_agent(
        fraud_orchestrator, AgentName.FRAUD_DETECTION, "clinical_consistency", "FraudDetectionResult"
    )

    assert fhir_outcome.success is True
    assert fraud_outcome.success is True

    return {
        "case_id": "CLM-0001",
        "current_step": "fraud_detection",
        "completed_steps": ["fhir_validator", "fraud_detection"],
        **fhir_outcome.state_updates,
        **fraud_outcome.state_updates,
    }


# ── 1. Résultats compatibles — accord ────────────────────────────────────────


class TestCompatibleResults:
    def test_two_pass_results_produce_no_disagreement(self):
        state = _build_state_from_synthetic_results(VerificationStatus.PASS, VerificationStatus.PASS)

        disagreements = detect_result_disagreements(state)

        assert disagreements == ()
        assert has_critical_disagreement(disagreements) is False

    def test_two_pass_results_route_to_continue(self):
        state = _build_state_from_synthetic_results(VerificationStatus.PASS, VerificationStatus.PASS)

        assert route_result_consistency(state) == CONTINUE

    def test_two_fail_results_agree_and_route_to_continue(self):
        """Un échec partagé par les deux agents n'est pas un désaccord — ils
        sont d'accord (sur un échec) ; la décision d'y donner suite reste
        celle des routes métier existantes (route_fhir/route_coding),
        jamais celle de la détection générique de désaccords."""
        state = _build_state_from_synthetic_results(VerificationStatus.FAIL, VerificationStatus.FAIL)

        disagreements = detect_result_disagreements(state)

        assert disagreements == ()
        assert route_result_consistency(state) == CONTINUE


# ── 2. Résultats explicitement contradictoires — désaccord critique ────────


class TestContradictoryResults:
    def test_pass_vs_fail_is_flagged_as_critical_disagreement(self):
        state = _build_state_from_synthetic_results(VerificationStatus.PASS, VerificationStatus.FAIL)

        disagreements = detect_result_disagreements(state)

        assert len(disagreements) == 1
        assert has_critical_disagreement(disagreements) is True

    def test_critical_disagreement_routes_to_needs_review(self):
        state = _build_state_from_synthetic_results(VerificationStatus.PASS, VerificationStatus.FAIL)

        assert route_result_consistency(state) == NEEDS_REVIEW

    def test_critical_disagreement_carries_minimal_evidence(self):
        """La preuve du désaccord se limite à agent/champ/valeurs attendues
        et observées — jamais de contenu métier (aucun document, aucune
        donnée clinique ou financière), directement exploitable par la
        revue humaine visée par needs_review."""
        state = _build_state_from_synthetic_results(VerificationStatus.PASS, VerificationStatus.FAIL)

        disagreements = detect_result_disagreements(state)

        assert len(disagreements) == 1
        point = disagreements[0]
        assert isinstance(point, DisagreementPoint)
        assert set(point.model_dump().keys()) == {"agent", "field", "expected", "observed"}
        assert point.agent == "fraud_detection"
        assert point.field == "status"
        assert point.expected == "PASS"
        assert point.observed == "FAIL"

    def test_symmetric_contradiction_is_also_flagged(self):
        """L'ordre des agents ne change rien : fhir=FAIL / fraud=PASS est
        tout autant un désaccord critique que l'inverse."""
        state = _build_state_from_synthetic_results(VerificationStatus.FAIL, VerificationStatus.PASS)

        disagreements = detect_result_disagreements(state)

        assert has_critical_disagreement(disagreements) is True
        assert route_result_consistency(state) == NEEDS_REVIEW


# ── 3. Aucune décision médicale ou financière finale ─────────────────────────


class TestNoFinalBusinessDecisionIsEverProduced:
    """La détection de désaccords ne fait que signaler et router — elle ne
    doit jamais produire, modifier ou anticiper une décision finale (celle
    de ``case_reviewer_agent``, seul habilité à recommander APPROVE/REJECT,
    ou toute donnée financière de couverture)."""

    def test_no_final_recommendation_field_is_ever_set(self):
        for fhir_status, fraud_status in (
            (VerificationStatus.PASS, VerificationStatus.PASS),
            (VerificationStatus.PASS, VerificationStatus.FAIL),
        ):
            state = _build_state_from_synthetic_results(fhir_status, fraud_status)
            detect_result_disagreements(state)
            route_result_consistency(state)
            assert "final_recommendation" not in state
            assert "review_result" not in state

    def test_detection_never_calls_the_case_reviewer_or_any_decision_agent(self):
        """Aucun agent de décision (case_reviewer, clinical_consistency,
        fraud_detection) n'est jamais invoqué par la détection de
        désaccords — elle n'opère que sur les résultats déjà présents dans
        le state, sans jamais appeler d'agent elle-même."""
        calls: list[str] = []

        def _case_reviewer_spy(state: dict) -> dict:
            calls.append("case_reviewer")
            raise AssertionError("la détection de désaccords ne doit jamais appeler d'agent")

        state = _build_state_from_synthetic_results(VerificationStatus.PASS, VerificationStatus.FAIL)
        orchestrator = _permissive_orchestrator({AgentName.CASE_REVIEWER: _case_reviewer_spy})

        disagreements = detect_result_disagreements(state)
        route = route_result_consistency(state)

        assert calls == []
        assert route == NEEDS_REVIEW
        assert has_critical_disagreement(disagreements) is True
        # L'orchestrateur reste disponible mais n'est jamais sollicité par
        # la détection elle-même — seul un appel explicite le déclencherait.
        assert orchestrator.agent_registry[AgentName.CASE_REVIEWER] is _case_reviewer_spy

    def test_disagreement_result_never_contains_a_verdict_or_amount_field(self):
        """Le schéma DisagreementPoint n'expose que agent/field/expected/
        observed — jamais de champ de type recommandation, montant ou
        décision, qui laisserait penser qu'un jugement métier a eu lieu."""
        state = _build_state_from_synthetic_results(VerificationStatus.PASS, VerificationStatus.FAIL)
        disagreements = detect_result_disagreements(state)
        for point in disagreements:
            fields = set(point.model_dump().keys())
            assert fields == {"agent", "field", "expected", "observed"}
            assert "amount" not in fields
            assert "recommendation" not in fields
            assert "decision" not in fields


# ── 4. Aucun appel LLM réel ────────────────────────────────────────────────────


class TestNoRealLlmInvolved:
    def test_orchestrators_used_have_no_model_registered(self):
        """Les orchestrateurs synthétiques n'ont aucun modèle enregistré —
        aucune fabrique de client LLM n'existe donc nulle part dans ces
        scénarios."""
        orchestrator = _permissive_orchestrator({AgentName.FHIR_VALIDATOR: _fhir_runner(VerificationStatus.PASS)})
        assert orchestrator.model_registry.list_models() == ()

    def test_synthetic_agent_runners_are_plain_python_no_llm_factory(self):
        """Les exécuteurs enregistrés sont de simples fonctions construisant
        des instances Pydantic à la main — jamais ``llm.factory.get_llm``
        ni aucun appel réseau."""
        import inspect

        runner = _fhir_runner(VerificationStatus.PASS)
        source = inspect.getsource(runner)
        assert "get_llm" not in source
        assert "ChatOllama" not in source
