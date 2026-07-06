"""Chemin nominal jusqu'à revue humaine — graph/workflow.py — ClaimShield Santé.

Un seul scénario, plusieurs angles de vérification : toutes les vérifications
réussissent (ACCEPTED/ALLOW/PASS/APPROVE), puis le Case Reviewer produit une
pré-recommandation non finale. La revue humaine est obligatoire : le pipeline
s'arrête donc à ``needs_review`` avec interruption HITL, sans audit/finalize
avant validation humaine.

Aucun appel LLM réel : les 7 agents réels (claim_intake, security_gate,
privacy, document_ocr, fhir_validator, identity_coverage, medical_coding)
sont remplacés par des faux agents déterministes patchés dans l'espace de
noms ``graph.workflow`` **avant** l'appel à ``compile_workflow()`` — LangGraph
capture les fonctions au moment du ``add_node()``. case_reviewer utilise le
mécanisme d'injection réel (``case_reviewer_impl=``) plutôt qu'un
monkeypatch, pour exercer le chemin d'injection officiel. clinical_consistency,
fraud_detection reste sur son implémentation déterministe de test ; audit ne
s'exécute pas avant la revue humaine.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

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
from schemas.results import CaseReviewerResult, CaseReviewerResultPayload, LlmMetadata

# ── Ordre métier attendu (feuille de route) ──────────────────────────────────

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
    "needs_review",
]

# Agents réels remplacés par de faux agents déterministes (jamais de LLM).
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
    """Résultat d'agent mocké minimal — porte uniquement le champ lu par le
    routage (``decision`` ou ``status``), jamais de contenu métier réel."""

    decision: Any = None
    status: Any = None


@dataclass
class _StubSubStatus:
    status: Any = None


@dataclass
class _StubIdentityCoverageResult:
    identity: _StubSubStatus
    coverage: _StubSubStatus


class _ApprovingCaseReviewer:
    """Faux agent injecté (``CaseReviewerRunnable``) — approbation nominale.

    Utilise le mécanisme d'injection officiel de ``compile_workflow()``
    (``case_reviewer_impl=``) plutôt qu'un monkeypatch de nœud : aucun appel
    LLM, aucune logique métier fictive au-delà d'une approbation déterministe.
    """

    def __init__(self, call_counts: dict[str, int]) -> None:
        self._call_counts = call_counts

    def run(self, state: dict) -> CaseReviewerResult:
        self._call_counts["case_reviewer"] += 1
        return CaseReviewerResult(
            case_id=str(state.get("case_id", "UNKNOWN")),
            llm_trace=LlmMetadata(model_name="test-llm", prompt_version="test"),
            result_payload=CaseReviewerResultPayload(
                recommendation=Recommendation.APPROVE,
                justification=["Toutes les vérifications ont réussi — approbation nominale."],
                human_review_reasons=["Validation humaine obligatoire avant toute décision finale."],
            ),
        )


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def call_counts() -> dict[str, int]:
    """Compteur d'appels par nœud — une entrée par agent remplacé."""
    counts = {name: 0 for name in _MOCKED_AGENT_NODES}
    counts["case_reviewer"] = 0
    return counts


@pytest.fixture
def deterministic_agents(monkeypatch: pytest.MonkeyPatch, call_counts: dict[str, int]) -> dict[str, int]:
    """Remplace les 7 agents réels par de faux agents déterministes.

    Chaque faux agent incrémente son propre compteur d'appels et renvoie une
    mise à jour minimale et cohérente (ACCEPTED/ALLOW/PASS) qui satisfait le
    routage de ``graph/edges.py`` sans jamais invoquer Ollama ni aucun outil
    de fichier/OCR/FHIR réel.
    """

    def mock_claim_intake(state: dict) -> dict:
        call_counts["claim_intake"] += 1
        return {
            "intake_status": IntakeStatus.ACCEPTED,
            "intake_input": None,
            "current_step": "claim_intake",
            "completed_steps": ["claim_intake"],
        }

    def mock_security_gate(state: dict) -> dict:
        call_counts["security_gate"] += 1
        return {
            "security_result": _StubResult(decision=SecurityDecision.ALLOW),
            "security_input": None,
            "current_step": "security_gate",
            "completed_steps": ["security_gate"],
        }

    def mock_privacy(state: dict) -> dict:
        call_counts["privacy"] += 1
        return {
            "privacy_result": _StubResult(decision=PrivacyDecision.ALLOW),
            "privacy_input": None,
            "current_step": "privacy",
            "completed_steps": ["privacy"],
        }

    def mock_document_ocr(state: dict) -> dict:
        call_counts["document_ocr"] += 1
        return {
            "ocr_result": _StubResult(status=VerificationStatus.PASS),
            "ocr_input": None,
            "current_step": "document_ocr",
            "completed_steps": ["document_ocr"],
        }

    def mock_fhir_validator(state: dict) -> dict:
        call_counts["fhir_validator"] += 1
        return {
            "fhir_result": _StubResult(status=VerificationStatus.PASS),
            "fhir_input": None,
            "current_step": "fhir_validator",
            "completed_steps": ["fhir_validator"],
        }

    def mock_identity_coverage(state: dict) -> dict:
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

    def mock_medical_coding(state: dict) -> dict:
        call_counts["medical_coding"] += 1
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

    return call_counts


@pytest.fixture
def nominal_app(deterministic_agents: dict[str, int], call_counts: dict[str, int]):
    """Workflow compilé pour le chemin nominal complet.

    ``interrupt_before=[]`` : aucune interruption HITL statique n'est
    pertinente ici ; l'interruption dynamique vient de ``await_human_review``.
    ``case_reviewer_impl`` : injection réelle d'une approbation déterministe.
    """
    reviewer = _ApprovingCaseReviewer(call_counts)
    return compile_workflow(interrupt_before=[], case_reviewer_impl=reviewer)


@pytest.fixture
def valid_claim_state() -> dict:
    """État initial valide — dossier complet, prêt à traverser tout le pipeline
    sans qu'aucune vérification n'échoue."""
    return {
        "case_id": "CLM-0001",
        "schema_version": "1.0.0",
        "current_step": "initial",
        "completed_steps": [],
        "errors": [],
        "alerts": [],
        "final_justification": [],
    }


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestNominalPath:
    """Chemin nominal — toutes les vérifications réussissent avant HITL."""

    def test_reaches_human_review_with_approve_and_no_errors(self, nominal_app, valid_claim_state):
        result = nominal_app.invoke(valid_claim_state)

        assert "__interrupt__" in result
        assert result.get("final_recommendation") == Recommendation.APPROVE
        assert result.get("current_step") == "needs_review"
        assert result.get("errors", []) == []

    def test_completed_steps_follows_business_order(self, nominal_app, valid_claim_state):
        result = nominal_app.invoke(valid_claim_state)

        assert result["completed_steps"] == EXPECTED_NOMINAL_ORDER

    def test_each_expected_node_runs_exactly_once(
        self, nominal_app, valid_claim_state, call_counts
    ):
        result = nominal_app.invoke(valid_claim_state)

        step_counts = Counter(result["completed_steps"])
        for node in EXPECTED_NOMINAL_ORDER:
            assert step_counts[node] == 1, (
                f"{node} exécuté {step_counts[node]} fois dans completed_steps (attendu 1)"
            )

        # Double vérification indépendante de completed_steps : compteurs
        # d'appels réels des faux agents et de l'implémentation injectée.
        for name, count in call_counts.items():
            assert count == 1, f"{name} appelé {count} fois (attendu 1)"

    def test_no_node_outside_expected_order_completes(self, nominal_app, valid_claim_state):
        """Aucune étape terminale ne s'exécute avant validation humaine."""
        result = nominal_app.invoke(valid_claim_state)

        unexpected = set(result["completed_steps"]) - set(EXPECTED_NOMINAL_ORDER)
        assert unexpected == set()
        assert "audit" not in result["completed_steps"]
        assert "finalize" not in result["completed_steps"]
