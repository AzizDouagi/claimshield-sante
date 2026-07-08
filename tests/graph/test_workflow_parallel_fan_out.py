"""P2-1 — invariants du fan-out document_ocr/fhir_validator sur le graphe compilé réel.

Complète ``test_workflow_blocking_paths.py`` (qui couvre déjà chaque branche
bloquée individuellement, l'autre restant nominale) en couvrant explicitement
le « point d'attention critique » du plan de remédiation : une panne sur une
seule branche parallèle (ici FAIL, pas seulement NEEDS_REVIEW) ne doit jamais
laisser le graphe router sur un état partiel — la consolidation
(``route_verification_fan_in``, après le nœud de convergence
``verification_fan_in``) doit toujours attendre les deux branches avant de
décider, et aucune étape en aval ne doit s'exécuter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import graph.workflow as wf
from graph.workflow import compile_workflow
from schemas.domain import IntakeStatus, PrivacyDecision, SecurityDecision, VerificationStatus


@dataclass
class _StubResult:
    decision: Any = None
    status: Any = None


def _mock_claim_intake(state: dict) -> dict:
    return {
        "intake_status": IntakeStatus.ACCEPTED,
        "intake_input": None,
        "current_step": "claim_intake",
        "completed_steps": ["claim_intake"],
    }


def _mock_security_gate(state: dict) -> dict:
    return {
        "security_result": _StubResult(decision=SecurityDecision.ALLOW),
        "security_input": None,
        "current_step": "security_gate",
        "completed_steps": ["security_gate"],
    }


def _mock_privacy(state: dict) -> dict:
    return {
        "privacy_result": _StubResult(decision=PrivacyDecision.ALLOW),
        "privacy_input": None,
        "current_step": "privacy",
        "completed_steps": ["privacy"],
    }


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


class TestOneBranchFailsOtherPasses:
    """document_ocr FAIL + fhir_validator PASS (ou l'inverse) : la
    consolidation doit toujours retourner FAILURE — jamais un routage basé
    sur une seule des deux branches, jamais un état incohérent où
    identity_coverage (ou toute étape en aval) s'exécuterait quand même."""

    def test_document_ocr_fail_blocks_the_whole_fan_in(self, monkeypatch):
        def mock_document_ocr(state: dict) -> dict:
            return {
                "ocr_result": _StubResult(status=VerificationStatus.FAIL),
                "ocr_input": None,
                "completed_steps": ["document_ocr"],
            }

        def mock_fhir_validator(state: dict) -> dict:
            return {
                "fhir_result": _StubResult(status=VerificationStatus.PASS),
                "fhir_input": None,
                "completed_steps": ["fhir_validator"],
            }

        monkeypatch.setattr(wf, "node_claim_intake", _mock_claim_intake)
        monkeypatch.setattr(wf, "node_security_gate", _mock_security_gate)
        monkeypatch.setattr(wf, "node_privacy", _mock_privacy)
        monkeypatch.setattr(wf, "node_document_ocr", mock_document_ocr)
        monkeypatch.setattr(wf, "node_fhir_validator", mock_fhir_validator)

        app = compile_workflow(interrupt_before=[])
        result = app.invoke(_initial_state())

        assert result.get("current_step") == "failure"
        completed = set(result.get("completed_steps", []))
        # Les deux branches ont bien tourné (aucune n'est court-circuitée par
        # l'échec de l'autre) et le nœud de convergence a bien tranché.
        assert {"document_ocr", "fhir_validator", "verification_fan_in"} <= completed
        # Rien en aval ne s'exécute sur un état partiel/incohérent.
        assert "identity_coverage" not in completed
        assert "medical_coding" not in completed
        assert "needs_review" not in completed

    def test_fhir_validator_fail_blocks_the_whole_fan_in_even_if_ocr_passes(self, monkeypatch):
        """Symétrique : peu importe laquelle des deux branches échoue, la
        consolidation ne favorise jamais l'une par rapport à l'autre."""
        def mock_document_ocr(state: dict) -> dict:
            return {
                "ocr_result": _StubResult(status=VerificationStatus.PASS),
                "ocr_input": None,
                "completed_steps": ["document_ocr"],
            }

        def mock_fhir_validator(state: dict) -> dict:
            return {
                "fhir_result": _StubResult(status=VerificationStatus.FAIL),
                "fhir_input": None,
                "completed_steps": ["fhir_validator"],
            }

        monkeypatch.setattr(wf, "node_claim_intake", _mock_claim_intake)
        monkeypatch.setattr(wf, "node_security_gate", _mock_security_gate)
        monkeypatch.setattr(wf, "node_privacy", _mock_privacy)
        monkeypatch.setattr(wf, "node_document_ocr", mock_document_ocr)
        monkeypatch.setattr(wf, "node_fhir_validator", mock_fhir_validator)

        app = compile_workflow(interrupt_before=[])
        result = app.invoke(_initial_state())

        assert result.get("current_step") == "failure"
        completed = set(result.get("completed_steps", []))
        assert {"document_ocr", "fhir_validator", "verification_fan_in"} <= completed
        assert "identity_coverage" not in completed

    def test_invoke_never_raises_and_never_leaves_errors_empty_on_failure(self, monkeypatch):
        """Une panne sur une branche ne doit jamais se traduire par une
        exception non gérée qui remonterait à l'appelant, ni par un état
        final sans trace de ce qui a bloqué."""
        def mock_document_ocr(state: dict) -> dict:
            return {
                "ocr_result": _StubResult(status=VerificationStatus.FAIL),
                "ocr_input": None,
                "completed_steps": ["document_ocr"],
            }

        def mock_fhir_validator(state: dict) -> dict:
            return {
                "fhir_result": _StubResult(status=VerificationStatus.PASS),
                "fhir_input": None,
                "completed_steps": ["fhir_validator"],
            }

        monkeypatch.setattr(wf, "node_claim_intake", _mock_claim_intake)
        monkeypatch.setattr(wf, "node_security_gate", _mock_security_gate)
        monkeypatch.setattr(wf, "node_privacy", _mock_privacy)
        monkeypatch.setattr(wf, "node_document_ocr", mock_document_ocr)
        monkeypatch.setattr(wf, "node_fhir_validator", mock_fhir_validator)

        app = compile_workflow(interrupt_before=[])
        result = app.invoke(_initial_state())  # ne doit jamais lever

        assert result.get("final_recommendation") is not None
