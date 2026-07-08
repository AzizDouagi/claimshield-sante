"""Tests de l'API minimale — ``api/main.py``.

Même stratégie de mock que les tests du graphe (``tests/graph/test_workflow*.py``) :
les 7 agents réels sont remplacés par de faux agents déterministes patchés
dans l'espace de noms ``graph.workflow`` **avant** l'appel à ``create_app()``
(qui compile le graphe en interne) — jamais d'appel LLM réel, jamais de
dépendance à Ollama. ``document_ocr`` renvoie NEEDS_REVIEW (fhir_validator
PASS) pour atteindre ``needs_review``/``await_human_review`` sans jamais
traverser ``case_reviewer`` — évite d'avoir à mocker aussi les agents avals
(clinical_consistency/fraud_detection/case_reviewer restent sur leur
implémentation réelle, mais ne sont jamais exécutés dans ce scénario).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

import graph.workflow as wf
from api.main import create_app
from schemas.domain import IntakeStatus, PrivacyDecision, SecurityDecision, VerificationStatus

_API_KEY = "claimshield-dev-api-key-change-in-production"


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


def _mock_document_ocr(state: dict) -> dict:
    return {
        "ocr_result": _StubResult(status=VerificationStatus.NEEDS_REVIEW),
        "ocr_input": None,
        "completed_steps": ["document_ocr"],
        "alerts": ["[document_ocr] confiance limite — revue requise."],
    }


def _mock_fhir_validator(state: dict) -> dict:
    return {
        "fhir_result": _StubResult(status=VerificationStatus.PASS),
        "fhir_input": None,
        "completed_steps": ["fhir_validator"],
    }


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(wf, "node_claim_intake", _mock_claim_intake)
    monkeypatch.setattr(wf, "node_security_gate", _mock_security_gate)
    monkeypatch.setattr(wf, "node_privacy", _mock_privacy)
    monkeypatch.setattr(wf, "node_document_ocr", _mock_document_ocr)
    monkeypatch.setattr(wf, "node_fhir_validator", _mock_fhir_validator)

    app = create_app(checkpointer=InMemorySaver())
    return TestClient(app)


def _submit(client: TestClient, case_id: str) -> Any:
    return client.post(
        "/claims",
        json={"case_id": case_id, "source_path": "/tmp/does-not-need-to-exist"},
        headers={"X-API-Key": _API_KEY},
    )


class TestHealthz:
    def test_healthz_no_auth_required(self, client: TestClient):
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestAuthentication:
    def test_submit_claim_without_api_key_returns_401(self, client: TestClient):
        response = client.post("/claims", json={"case_id": "CLM-0001", "source_path": "/tmp/x"})
        assert response.status_code == 401

    def test_submit_claim_with_wrong_api_key_returns_401(self, client: TestClient):
        response = client.post(
            "/claims",
            json={"case_id": "CLM-0001", "source_path": "/tmp/x"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401

    def test_human_decision_without_api_key_returns_401(self, client: TestClient):
        response = client.post(
            "/claims/CLM-0001/human-decision",
            json={"actor": "a", "action": "APPROVE", "justification": "motif"},
        )
        assert response.status_code == 401

    def test_get_claim_status_does_not_require_api_key(self, client: TestClient):
        """Lecture seule — pas d'action métier, aucune authentification requise."""
        response = client.get("/claims/CLM-0099")
        assert response.status_code == 404  # pas 401 : la requête a bien été traitée


class TestSubmitClaim:
    def test_submit_claim_reaches_needs_review_with_pending_review(self, client: TestClient):
        response = _submit(client, "CLM-1001")

        assert response.status_code == 201
        body = response.json()
        assert body["case_id"] == "CLM-1001"
        assert body["interrupted"] is True
        assert body["current_step"] == "needs_review"
        assert body["pending_review"] is not None
        assert body["pending_review"]["case_id"] == "CLM-1001"
        assert "APPROVE" in body["pending_review"]["options"]

    def test_submit_claim_never_leaks_raw_content(self, client: TestClient):
        response = _submit(client, "CLM-1002")
        raw = response.text
        assert "full_text" not in raw
        assert "system_prompt" not in raw

    def test_submit_claim_invalid_case_id_rejected(self, client: TestClient):
        response = client.post(
            "/claims",
            json={"case_id": "not-a-valid-id", "source_path": "/tmp/x"},
            headers={"X-API-Key": _API_KEY},
        )
        assert response.status_code == 422


class TestGetClaimStatus:
    def test_get_claim_status_after_submission(self, client: TestClient):
        _submit(client, "CLM-1003")

        response = client.get("/claims/CLM-1003")

        assert response.status_code == 200
        body = response.json()
        assert body["case_id"] == "CLM-1003"
        assert body["interrupted"] is True

    def test_get_unknown_claim_returns_404(self, client: TestClient):
        response = client.get("/claims/CLM-9999")
        assert response.status_code == 404


class TestHumanDecision:
    def test_approve_resumes_pipeline_to_completion(self, client: TestClient):
        """Ce scénario interrompt avant case_reviewer (document_ocr NEEDS_REVIEW
        court-circuite directement vers needs_review) — final_recommendation
        n'est donc jamais fixé par case_reviewer ici (seul node_failure fixe
        REJECT inconditionnellement ; node_finalize ne fixe rien), même
        comportement que tests/graph/test_workflow_interrupt_resume.py. Ce
        test vérifie la progression réelle (audit → finalize, jamais
        failure), pas une valeur de recommandation absente dans ce scénario."""
        _submit(client, "CLM-1004")

        response = client.post(
            "/claims/CLM-1004/human-decision",
            json={"actor": "reviewer@example.com", "action": "APPROVE", "justification": "Dossier conforme."},
            headers={"X-API-Key": _API_KEY},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["interrupted"] is False
        assert "audit" in body["completed_steps"]
        assert "finalize" in body["completed_steps"]
        assert "failure" not in body["completed_steps"]

    def test_reject_routes_to_failure(self, client: TestClient):
        _submit(client, "CLM-1005")

        response = client.post(
            "/claims/CLM-1005/human-decision",
            json={"actor": "reviewer@example.com", "action": "REJECT", "justification": "Preuves insuffisantes."},
            headers={"X-API-Key": _API_KEY},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["final_recommendation"] == "REJECT"
        assert "failure" in body["completed_steps"]

    def test_missing_justification_returns_422(self, client: TestClient):
        _submit(client, "CLM-1006")

        response = client.post(
            "/claims/CLM-1006/human-decision",
            json={"actor": "reviewer@example.com", "action": "APPROVE", "justification": ""},
            headers={"X-API-Key": _API_KEY},
        )

        assert response.status_code == 422

    def test_decision_on_unknown_claim_returns_404(self, client: TestClient):
        response = client.post(
            "/claims/CLM-9998/human-decision",
            json={"actor": "reviewer@example.com", "action": "APPROVE", "justification": "motif"},
            headers={"X-API-Key": _API_KEY},
        )
        assert response.status_code == 404

    def test_decision_on_non_interrupted_claim_returns_409(self, client: TestClient):
        _submit(client, "CLM-1007")
        client.post(
            "/claims/CLM-1007/human-decision",
            json={"actor": "reviewer@example.com", "action": "APPROVE", "justification": "Dossier conforme."},
            headers={"X-API-Key": _API_KEY},
        )

        # Le dossier est déjà finalisé — une seconde décision est refusée.
        response = client.post(
            "/claims/CLM-1007/human-decision",
            json={"actor": "reviewer@example.com", "action": "APPROVE", "justification": "Nouvelle tentative."},
            headers={"X-API-Key": _API_KEY},
        )
        assert response.status_code == 409

    def test_retry_without_target_node_returns_422(self, client: TestClient):
        _submit(client, "CLM-1008")

        response = client.post(
            "/claims/CLM-1008/human-decision",
            json={"actor": "reviewer@example.com", "action": "RETRY", "justification": "Pièce manquante."},
            headers={"X-API-Key": _API_KEY},
        )
        assert response.status_code == 422


class TestNoDirectAgentAccess:
    """Vérification statique légère : l'API n'importe aucun module agent
    directement — le seul point d'entrée métier est le graphe compilé (même
    esprit que ``tests/graph/test_architecture.py``)."""

    def test_main_module_does_not_import_agents_package(self):
        import api.main as main_module

        source = main_module.__file__
        with open(source, encoding="utf-8") as f:
            content = f.read()
        assert "from agents" not in content
        assert "import agents" not in content
