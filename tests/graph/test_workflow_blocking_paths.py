"""Chemins de blocage — graph/workflow.py — ClaimShield Santé.

Huit scénarios paramétrés, un par branche de blocage prévue (QUARANTINE ou
NEEDS_REVIEW) du pipeline nominal :

  1. claim_intake   → QUARANTINED        → quarantine
  2. security_gate  → QUARANTINE         → quarantine
  3. document_ocr   → NEEDS_REVIEW       → needs_review
  4. fhir_validator → NEEDS_REVIEW       → needs_review
  5. identity_coverage → NEEDS_REVIEW    → needs_review
  6. medical_coding → NEEDS_REVIEW       → needs_review
  7. case_reviewer  → PENDING            → needs_review
  8. case_reviewer  → APPROVE + human_review_required=True → needs_review

Pour chaque scénario, tous les nœuds amont du nœud bloquant restent nominaux
(ACCEPTED/ALLOW/PASS) ; aucun appel LLM réel (les 7 agents réels et
case_reviewer sont des faux agents/implémentation injectée déterministes).

Chaque test vérifie :
  - la destination atteinte (quarantine ou needs_review) ;
  - qu'aucun nœud strictement postérieur au blocage ne s'exécute
    (``completed_steps`` + compteurs d'appels) ;
  - le code motif, la preuve (résultat d'agent) et le statut final
    (``current_step``) enregistrés dans le state ;
  - qu'aucune exception n'est levée ni avalée silencieusement
    (``app.invoke()`` ne lève pas, ``errors`` reste vide, une alerte est
    bien enregistrée).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pytest

import graph.workflow as wf
from graph.workflow import compile_workflow
from schemas.domain import (
    IntakeStatus,
    PrivacyDecision,
    Recommendation,
    SecurityDecision,
    VerificationStatus,
)
from schemas.results import CaseReviewerResult

# ── Ordre métier nominal (feuille de route) ──────────────────────────────────

EXPECTED_NOMINAL_ORDER: list[str] = [
    "claim_intake",
    "security_gate",
    "privacy",
    "document_ocr",
    "fhir_validator",
    "identity_coverage",
    "medical_coding",
    "clinical_consistency",
    "fraud_detection",
    "case_reviewer",
    "audit",
    "finalize",
]

_MOCKED_AGENT_NODES: tuple[str, ...] = (
    "claim_intake",
    "security_gate",
    "privacy",
    "document_ocr",
    "fhir_validator",
    "identity_coverage",
    "medical_coding",
)


@dataclass
class _StubResult:
    decision: Any = None
    status: Any = None


@dataclass
class _StubSubStatus:
    status: Any = None


@dataclass
class _StubIdentityCoverageResult:
    identity: _StubSubStatus
    coverage: _StubSubStatus


class _CaseReviewerStub:
    """Faux agent injecté (``CaseReviewerRunnable``) — recommandation contrôlée
    par le scénario, jamais de LLM."""

    def __init__(
        self,
        call_counts: dict[str, int],
        *,
        recommendation: Recommendation,
        human_review_required: bool,
        reasons: list[str],
    ) -> None:
        self._call_counts = call_counts
        self._recommendation = recommendation
        self._human_review_required = human_review_required
        self._reasons = reasons

    def run(self, state: dict) -> CaseReviewerResult:
        self._call_counts["case_reviewer"] += 1
        return CaseReviewerResult(
            case_id=str(state.get("case_id", "UNKNOWN")),
            recommendation=self._recommendation,
            justification=list(self._reasons),
            human_review_required=self._human_review_required,
            human_review_reasons=list(self._reasons),
        )


def _initial_state() -> dict:
    return {
        "case_id": "CLM-0001",
        "schema_version": "1.0.0",
        "current_step": "initial",
        "completed_steps": [],
        "errors": [],
        "alerts": [],
        "final_justification": [],
    }


# ── Faux agents nominaux (ACCEPTED/ALLOW/PASS) et bloquants ─────────────────


def _nominal_claim_intake(call_counts: dict[str, int]) -> Callable:
    def _node(state: dict) -> dict:
        call_counts["claim_intake"] += 1
        return {
            "intake_status": IntakeStatus.ACCEPTED,
            "intake_input": None,
            "current_step": "claim_intake",
            "completed_steps": ["claim_intake"],
        }
    return _node


def _blocking_claim_intake(call_counts: dict[str, int]) -> Callable:
    def _node(state: dict) -> dict:
        call_counts["claim_intake"] += 1
        return {
            "intake_status": IntakeStatus.QUARANTINED,
            "intake_input": None,
            "current_step": "claim_intake",
            "completed_steps": ["claim_intake"],
            "alerts": ["[claim_intake] fichier suspect détecté — quarantaine."],
        }
    return _node


def _nominal_security_gate(call_counts: dict[str, int]) -> Callable:
    def _node(state: dict) -> dict:
        call_counts["security_gate"] += 1
        return {
            "security_result": _StubResult(decision=SecurityDecision.ALLOW),
            "security_input": None,
            "current_step": "security_gate",
            "completed_steps": ["security_gate"],
        }
    return _node


def _blocking_security_gate(call_counts: dict[str, int]) -> Callable:
    def _node(state: dict) -> dict:
        call_counts["security_gate"] += 1
        return {
            "security_result": _StubResult(decision=SecurityDecision.QUARANTINE),
            "security_input": None,
            "current_step": "security_gate",
            "completed_steps": ["security_gate"],
            "alerts": ["[security_gate] contenu suspect — quarantaine."],
        }
    return _node


def _nominal_privacy(call_counts: dict[str, int]) -> Callable:
    def _node(state: dict) -> dict:
        call_counts["privacy"] += 1
        return {
            "privacy_result": _StubResult(decision=PrivacyDecision.ALLOW),
            "privacy_input": None,
            "current_step": "privacy",
            "completed_steps": ["privacy"],
        }
    return _node


def _nominal_document_ocr(call_counts: dict[str, int]) -> Callable:
    def _node(state: dict) -> dict:
        call_counts["document_ocr"] += 1
        return {
            "ocr_result": _StubResult(status=VerificationStatus.PASS),
            "ocr_input": None,
            "current_step": "document_ocr",
            "completed_steps": ["document_ocr"],
        }
    return _node


def _blocking_document_ocr(call_counts: dict[str, int]) -> Callable:
    def _node(state: dict) -> dict:
        call_counts["document_ocr"] += 1
        return {
            "ocr_result": _StubResult(status=VerificationStatus.NEEDS_REVIEW),
            "ocr_input": None,
            "current_step": "document_ocr",
            "completed_steps": ["document_ocr"],
            "alerts": ["[document_ocr] confiance limite — revue requise."],
        }
    return _node


def _nominal_fhir_validator(call_counts: dict[str, int]) -> Callable:
    def _node(state: dict) -> dict:
        call_counts["fhir_validator"] += 1
        return {
            "fhir_result": _StubResult(status=VerificationStatus.PASS),
            "fhir_input": None,
            "current_step": "fhir_validator",
            "completed_steps": ["fhir_validator"],
        }
    return _node


def _blocking_fhir_validator(call_counts: dict[str, int]) -> Callable:
    def _node(state: dict) -> dict:
        call_counts["fhir_validator"] += 1
        return {
            "fhir_result": _StubResult(status=VerificationStatus.NEEDS_REVIEW),
            "fhir_input": None,
            "current_step": "fhir_validator",
            "completed_steps": ["fhir_validator"],
            "alerts": ["[fhir_validator] référence non résolue — revue requise."],
        }
    return _node


def _nominal_identity_coverage(call_counts: dict[str, int]) -> Callable:
    def _node(state: dict) -> dict:
        call_counts["identity_coverage"] += 1
        return {
            "identity_coverage_result": _StubIdentityCoverageResult(
                identity=_StubSubStatus(status=VerificationStatus.PASS),
                coverage=_StubSubStatus(status=VerificationStatus.PASS),
            ),
            "identity_coverage_input": None,
            "current_step": "identity_coverage",
            "completed_steps": ["identity_coverage"],
        }
    return _node


def _blocking_identity_coverage(call_counts: dict[str, int]) -> Callable:
    def _node(state: dict) -> dict:
        call_counts["identity_coverage"] += 1
        return {
            "identity_coverage_result": _StubIdentityCoverageResult(
                identity=_StubSubStatus(status=VerificationStatus.NEEDS_REVIEW),
                coverage=_StubSubStatus(status=VerificationStatus.PASS),
            ),
            "identity_coverage_input": None,
            "current_step": "identity_coverage",
            "completed_steps": ["identity_coverage"],
            "alerts": ["[identity_coverage] identité ambiguë — revue requise."],
        }
    return _node


def _nominal_medical_coding(call_counts: dict[str, int]) -> Callable:
    def _node(state: dict) -> dict:
        call_counts["medical_coding"] += 1
        return {
            "coding_result": _StubResult(status=VerificationStatus.PASS),
            "coding_input": None,
            "current_step": "medical_coding",
            "completed_steps": ["medical_coding"],
        }
    return _node


def _blocking_medical_coding(call_counts: dict[str, int]) -> Callable:
    def _node(state: dict) -> dict:
        call_counts["medical_coding"] += 1
        return {
            "coding_result": _StubResult(status=VerificationStatus.NEEDS_REVIEW),
            "coding_input": None,
            "current_step": "medical_coding",
            "completed_steps": ["medical_coding"],
            "alerts": ["[medical_coding] code ambigu — revue requise."],
        }
    return _node


_NOMINAL_FACTORIES: dict[str, Callable] = {
    "claim_intake": _nominal_claim_intake,
    "security_gate": _nominal_security_gate,
    "privacy": _nominal_privacy,
    "document_ocr": _nominal_document_ocr,
    "fhir_validator": _nominal_fhir_validator,
    "identity_coverage": _nominal_identity_coverage,
    "medical_coding": _nominal_medical_coding,
}

_BLOCKING_FACTORIES: dict[str, Callable] = {
    "claim_intake": _blocking_claim_intake,
    "security_gate": _blocking_security_gate,
    "document_ocr": _blocking_document_ocr,
    "fhir_validator": _blocking_fhir_validator,
    "identity_coverage": _blocking_identity_coverage,
    "medical_coding": _blocking_medical_coding,
}


# ── Scénarios ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _BlockingScenario:
    scenario_id: str
    blocking_node: str
    destination: str  # "quarantine" | "needs_review"
    reason_code: Any
    result_field: str  # champ du state portant la preuve


SCENARIOS: list[_BlockingScenario] = [
    _BlockingScenario(
        "claim_intake_quarantined", "claim_intake", "quarantine",
        IntakeStatus.QUARANTINED, "intake_status",
    ),
    _BlockingScenario(
        "security_gate_quarantine", "security_gate", "quarantine",
        SecurityDecision.QUARANTINE, "security_result",
    ),
    _BlockingScenario(
        "document_ocr_needs_review", "document_ocr", "needs_review",
        VerificationStatus.NEEDS_REVIEW, "ocr_result",
    ),
    _BlockingScenario(
        "fhir_validator_needs_review", "fhir_validator", "needs_review",
        VerificationStatus.NEEDS_REVIEW, "fhir_result",
    ),
    _BlockingScenario(
        "identity_coverage_needs_review", "identity_coverage", "needs_review",
        VerificationStatus.NEEDS_REVIEW, "identity_coverage_result",
    ),
    _BlockingScenario(
        "medical_coding_needs_review", "medical_coding", "needs_review",
        VerificationStatus.NEEDS_REVIEW, "coding_result",
    ),
    _BlockingScenario(
        "case_reviewer_pending", "case_reviewer", "needs_review",
        Recommendation.PENDING, "review_result",
    ),
    _BlockingScenario(
        "case_reviewer_human_review_required", "case_reviewer", "needs_review",
        Recommendation.APPROVE, "review_result",
    ),
]

assert len(SCENARIOS) == 8, "Huit branches de blocage attendues"


def _build_app_for_scenario(monkeypatch, scenario: _BlockingScenario):
    """Construit un workflow compilé où seul ``scenario.blocking_node`` bloque.

    Tous les nœuds amont restent nominaux (ACCEPTED/ALLOW/PASS) ; aucun n'est
    un appel LLM réel — ce sont des faux agents déterministes.
    """
    call_counts: dict[str, int] = {name: 0 for name in _MOCKED_AGENT_NODES}
    call_counts["case_reviewer"] = 0

    for name, nominal_factory in _NOMINAL_FACTORIES.items():
        if name == scenario.blocking_node:
            node_fn = _BLOCKING_FACTORIES[name](call_counts)
        else:
            node_fn = nominal_factory(call_counts)
        monkeypatch.setattr(wf, f"node_{name}", node_fn)

    case_reviewer_impl = None
    if scenario.blocking_node == "case_reviewer":
        case_reviewer_impl = _CaseReviewerStub(
            call_counts,
            recommendation=scenario.reason_code,
            human_review_required=True,
            reasons=[f"Scénario de test : {scenario.scenario_id}."],
        )

    app = compile_workflow(interrupt_before=[], case_reviewer_impl=case_reviewer_impl)
    return app, call_counts


def _extract_recorded_code(result: dict, scenario: _BlockingScenario) -> Any:
    """Extrait le code motif depuis la preuve (résultat d'agent) enregistrée."""
    if scenario.blocking_node == "claim_intake":
        return result.get("intake_status")

    evidence = result.get(scenario.result_field)
    assert evidence is not None, f"Preuve absente dans le state : {scenario.result_field}"

    if scenario.blocking_node == "security_gate":
        return evidence.decision
    if scenario.blocking_node in ("document_ocr", "fhir_validator", "medical_coding"):
        return evidence.status
    if scenario.blocking_node == "identity_coverage":
        return evidence.identity.status
    if scenario.blocking_node == "case_reviewer":
        return evidence.recommendation
    raise AssertionError(f"Nœud bloquant inattendu : {scenario.blocking_node}")


_SCENARIO_IDS = [s.scenario_id for s in SCENARIOS]


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestBlockingPaths:
    def test_exactly_eight_scenarios_defined(self):
        assert len(SCENARIOS) == 8

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=_SCENARIO_IDS)
    def test_reaches_expected_destination(self, monkeypatch, scenario: _BlockingScenario):
        app, _ = _build_app_for_scenario(monkeypatch, scenario)
        result = app.invoke(_initial_state())

        assert scenario.destination in result.get("completed_steps", [])
        assert result.get("current_step") == scenario.destination

        if scenario.destination == "quarantine":
            assert "__interrupt__" not in result
        else:
            assert "__interrupt__" in result

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=_SCENARIO_IDS)
    def test_downstream_agents_never_execute(self, monkeypatch, scenario: _BlockingScenario):
        app, call_counts = _build_app_for_scenario(monkeypatch, scenario)
        result = app.invoke(_initial_state())

        blocking_index = EXPECTED_NOMINAL_ORDER.index(scenario.blocking_node)
        downstream = set(EXPECTED_NOMINAL_ORDER[blocking_index + 1:])

        completed = set(result.get("completed_steps", []))
        overlap = downstream & completed
        assert overlap == set(), (
            f"Nœud(s) exécuté(s) après le blocage ({scenario.scenario_id}) : {overlap}"
        )

        for node in downstream & set(call_counts):
            assert call_counts[node] == 0, (
                f"{node} appelé après le blocage ({scenario.scenario_id})"
            )

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=_SCENARIO_IDS)
    def test_blocking_node_runs_exactly_once(self, monkeypatch, scenario: _BlockingScenario):
        app, call_counts = _build_app_for_scenario(monkeypatch, scenario)
        app.invoke(_initial_state())

        assert call_counts[scenario.blocking_node] == 1

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=_SCENARIO_IDS)
    def test_reason_code_evidence_and_final_status_recorded(
        self, monkeypatch, scenario: _BlockingScenario
    ):
        app, _ = _build_app_for_scenario(monkeypatch, scenario)
        result = app.invoke(_initial_state())

        recorded_code = _extract_recorded_code(result, scenario)
        assert recorded_code == scenario.reason_code
        assert result.get("current_step") == scenario.destination

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=_SCENARIO_IDS)
    def test_blocking_does_not_raise_or_swallow_exception(
        self, monkeypatch, scenario: _BlockingScenario
    ):
        app, _ = _build_app_for_scenario(monkeypatch, scenario)

        result = app.invoke(_initial_state())  # ne doit jamais lever

        assert result.get("errors", []) == []
        assert len(result.get("alerts", [])) >= 1
