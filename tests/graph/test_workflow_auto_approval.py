"""P1-4 — auto-approbation bornée de case_reviewer sur le graphe compilé réel.

Complète ``test_workflow_paths.py`` (chemin nominal → interruption HITL) en
prouvant le chemin alternatif introduit par P1-4 : un dossier dont
``case_reviewer_agent`` pose ``result_payload.auto_decision ==
"AUTO_APPROVED_LOW_RISK"`` traverse ``audit``/``finalize`` sans jamais passer
par ``needs_review``/``await_human_review`` — mais traverse quand même
``audit`` avant ``finalize`` (garantie « aucun chemin terminal ne contourne
l'audit », inchangée par P1-4). Un second scénario (même dossier, sans
``auto_decision``) prouve que le comportement par défaut reste inchangé —
interruption HITL, exactement comme ``test_workflow_paths.py``.

Même patron que ``test_workflow_paths.py`` : 7 agents réels remplacés par de
faux agents déterministes (aucun appel LLM), ``case_reviewer_impl`` injecté
via le mécanisme officiel. L'agent ``audit`` s'exécute réellement ici (le
dossier ne s'arrête plus avant) — sa normalisation LLM reste mockée par
l'autouse ``tests.conftest.deterministic_agent_llm``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

import graph.workflow as wf
from graph.workflow import compile_workflow
from schemas.domain import IntakeStatus, PrivacyDecision, Recommendation, SecurityDecision, VerificationStatus
from schemas.results import CaseReviewerResult, CaseReviewerResultPayload, LlmMetadata


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


class _CaseReviewer:
    """Faux agent injecté — recommandation APPROVE, ``auto_decision``
    configurable pour exercer les deux chemins (avec/sans court-circuit)."""

    def __init__(self, *, auto_decision: str | None) -> None:
        self._auto_decision = auto_decision

    def run(self, state: dict) -> CaseReviewerResult:
        return CaseReviewerResult(
            case_id=str(state.get("case_id", "UNKNOWN")),
            llm_trace=LlmMetadata(model_name="test-llm", prompt_version="test"),
            result_payload=CaseReviewerResultPayload(
                recommendation=Recommendation.APPROVE,
                justification=["Toutes les vérifications ont réussi."],
                human_review_reasons=["Validation humaine obligatoire avant toute décision finale."],
                auto_decision=self._auto_decision,
                auto_decision_criteria=(
                    ["Critères P1-4 réunis (scénario de test)."] if self._auto_decision else []
                ),
            ),
        )


@pytest.fixture
def deterministic_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remplace les 7 agents réels par de faux agents déterministes — même
    contenu que ``test_workflow_paths.py::deterministic_agents``, sans le
    compteur d'appels (non nécessaire ici)."""

    def mock_claim_intake(state: dict) -> dict:
        return {
            "intake_status": IntakeStatus.ACCEPTED,
            "intake_input": None,
            "current_step": "claim_intake",
            "completed_steps": ["claim_intake"],
        }

    def mock_security_gate(state: dict) -> dict:
        return {
            "security_result": _StubResult(decision=SecurityDecision.ALLOW),
            "security_input": None,
            "current_step": "security_gate",
            "completed_steps": ["security_gate"],
        }

    def mock_privacy(state: dict) -> dict:
        return {
            "privacy_result": _StubResult(decision=PrivacyDecision.ALLOW),
            "privacy_input": None,
            "current_step": "privacy",
            "completed_steps": ["privacy"],
        }

    def mock_document_ocr(state: dict) -> dict:
        return {
            "ocr_result": _StubResult(status=VerificationStatus.PASS),
            "ocr_input": None,
            "current_step": "document_ocr",
            "completed_steps": ["document_ocr"],
        }

    def mock_fhir_validator(state: dict) -> dict:
        return {
            "fhir_result": _StubResult(status=VerificationStatus.PASS),
            "fhir_input": None,
            "current_step": "fhir_validator",
            "completed_steps": ["fhir_validator"],
        }

    def mock_identity_coverage(state: dict) -> dict:
        return {
            "identity_coverage_result": _StubIdentityCoverageResult(
                identity=_StubSubStatus(status=VerificationStatus.PASS),
                coverage=_StubSubStatus(status=VerificationStatus.PASS),
            ),
            "identity_coverage_input": None,
            "current_step": "identity_coverage",
            "completed_steps": ["identity_coverage"],
        }

    def mock_medical_coding(state: dict) -> dict:
        return {
            "coding_result": _StubResult(status=VerificationStatus.PASS),
            "coding_input": None,
            "current_step": "medical_coding",
            "completed_steps": ["medical_coding"],
        }

    monkeypatch.setattr(wf, "node_claim_intake", mock_claim_intake)
    monkeypatch.setattr(wf, "node_security_gate", mock_security_gate)
    monkeypatch.setattr(wf, "node_privacy", mock_privacy)
    monkeypatch.setattr(wf, "node_document_ocr", mock_document_ocr)
    monkeypatch.setattr(wf, "node_fhir_validator", mock_fhir_validator)
    monkeypatch.setattr(wf, "node_identity_coverage", mock_identity_coverage)
    monkeypatch.setattr(wf, "node_medical_coding", mock_medical_coding)


@pytest.fixture
def valid_claim_state() -> dict:
    return {
        "case_id": "CLM-0001",
        "schema_version": "1.0.0",
        "current_step": "initial",
        "completed_steps": [],
        "errors": [],
        "alerts": [],
        "final_justification": [],
    }


class TestAutoApprovedCaseSkipsHumanReview:
    """auto_decision="AUTO_APPROVED_LOW_RISK" : traverse audit puis finalize,
    jamais needs_review/await_human_review."""

    def test_reaches_finalize_without_interruption(self, deterministic_agents, valid_claim_state):
        app = compile_workflow(interrupt_before=[], case_reviewer_impl=_CaseReviewer(auto_decision="AUTO_APPROVED_LOW_RISK"))

        result = app.invoke(valid_claim_state)

        assert "__interrupt__" not in result
        assert result.get("current_step") == "finalize"

    def test_audit_is_never_bypassed(self, deterministic_agents, valid_claim_state):
        """Garantie déjà établie (« aucun chemin terminal ne contourne
        l'audit ») — vérifiée explicitement inchangée par P1-4."""
        app = compile_workflow(interrupt_before=[], case_reviewer_impl=_CaseReviewer(auto_decision="AUTO_APPROVED_LOW_RISK"))

        result = app.invoke(valid_claim_state)

        assert "audit" in result["completed_steps"]
        assert "finalize" in result["completed_steps"]
        assert result["completed_steps"].index("audit") < result["completed_steps"].index("finalize")

    def test_needs_review_never_reached(self, deterministic_agents, valid_claim_state):
        app = compile_workflow(interrupt_before=[], case_reviewer_impl=_CaseReviewer(auto_decision="AUTO_APPROVED_LOW_RISK"))

        result = app.invoke(valid_claim_state)

        assert "needs_review" not in result["completed_steps"]
        assert "await_human_review" not in result["completed_steps"]

    def test_final_recommendation_is_approve(self, deterministic_agents, valid_claim_state):
        app = compile_workflow(interrupt_before=[], case_reviewer_impl=_CaseReviewer(auto_decision="AUTO_APPROVED_LOW_RISK"))

        result = app.invoke(valid_claim_state)

        assert result.get("final_recommendation") == Recommendation.APPROVE
        assert result.get("errors", []) == []


class TestWithoutAutoDecisionBehaviorUnchanged:
    """Même dossier, sans auto_decision : comportement par défaut inchangé —
    interruption HITL exactement comme avant P1-4 (test_workflow_paths.py)."""

    def test_reaches_human_review(self, deterministic_agents, valid_claim_state):
        app = compile_workflow(interrupt_before=[], case_reviewer_impl=_CaseReviewer(auto_decision=None))

        result = app.invoke(valid_claim_state)

        assert "__interrupt__" in result
        assert result.get("current_step") == "needs_review"
        assert "audit" not in result["completed_steps"]
        assert "finalize" not in result["completed_steps"]
