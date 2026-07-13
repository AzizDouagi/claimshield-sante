"""Tests de l'API v2 — ``api/v2/{claims.py, chat.py}`` (Phase V2-9).

Même stratégie de mock que ``tests/v2/graph/test_workflow_v2.py`` : les 5
agents V2 sont exécutés réellement (Phase A déterministe + `create_react_agent`/
`with_structured_output` réels), seules leurs fonctions `_invoke_llm_*` sont
monkeypatchées — jamais d'appel réel à Ollama. Le graphe V2 compilé est monté
sur une application FastAPI de test dédiée via
``api.v2.claims.build_v2_router(compiled_graph=..., override_store=...)`` —
jamais via le routeur par défaut ``api.v2.v2_router`` (qui compilerait un
graphe réel contre le backend de checkpoint de l'environnement).
"""
from __future__ import annotations

from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from api.v2.chat import build_chat_router
from api.v2.claims import build_v2_router
from graph.workflow_v2 import compile_workflow_v2
from schemas.domain import ClaimDecisionV2
from services.override_store import OverrideStore

_API_KEY = "claimshield-dev-api-key-change-in-production"


def _make_pdf(path, lines: list[str]) -> None:
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path))
    y = 800
    for line in lines:
        c.drawString(72, y, line)
        y -= 20
    c.save()


def _stage_deposit_folder(tmp_path) -> str:
    deposit_dir = tmp_path / "deposit"
    deposit_dir.mkdir(parents=True, exist_ok=True)
    _make_pdf(
        deposit_dir / "facture_test.pdf",
        [
            "FACTURE MEDICALE",
            "Numero : INV-9999",
            "patient_id : PAT-9999-DEMO",
            "Prestataire : Bernard Leclerc",
            "Date du document : 2026-03-01",
            "Date de soins : 2026-02-25",
            "Devise : USD",
            "Montant total facture : 500.00 USD",
        ],
    )
    return str(deposit_dir)


@pytest.fixture()
def deterministic_v2_llm(monkeypatch):
    """Mocke les 5 fonctions `_invoke_llm_*` des agents V2 — aucun appel réel
    à Ollama pendant les tests API (même fixture que
    ``tests/v2/graph/test_workflow_v2.py``, dupliquée volontairement — pas
    d'import cross-module de test)."""
    from agents.autonomous_decision_agent.schemas import LlmAutonomousDecision
    from agents.document_understanding_agent.schemas import LlmDocumentUnderstandingDecision
    from agents.identity_coverage_agent.schemas import LlmIdentityCoverageDecision
    from agents.intake_safety_agent.schemas import LlmIntakeSafetyDecision
    from agents.medical_risk_agent.schemas import LlmMedicalRiskDecision

    monkeypatch.setattr(
        "agents.intake_safety_agent.agent._invoke_llm_intake_safety",
        Mock(
            return_value=LlmIntakeSafetyDecision(
                status="ACCEPTED", reasons=["Dossier conforme."], explanation=""
            )
        ),
    )
    monkeypatch.setattr(
        "agents.document_understanding_agent.agent._invoke_llm_document_understanding",
        Mock(return_value=LlmDocumentUnderstandingDecision(fhir_recommended_status="PASS")),
    )
    monkeypatch.setattr(
        "agents.identity_coverage_agent.agent._invoke_llm_identity_coverage",
        Mock(return_value=LlmIdentityCoverageDecision()),
    )
    monkeypatch.setattr(
        "agents.medical_risk_agent.agent._invoke_llm_medical_risk",
        Mock(return_value=LlmMedicalRiskDecision()),
    )
    monkeypatch.setattr(
        "agents.autonomous_decision_agent.agent._invoke_llm_autonomous_decision",
        Mock(
            return_value=LlmAutonomousDecision(
                decision="APPROVE", summary="Dossier conforme.", confidence=0.95
            )
        ),
    )
    yield


@pytest.fixture()
def v2_client(deterministic_v2_llm) -> TestClient:
    compiled_graph = compile_workflow_v2(InMemorySaver())
    override_store = OverrideStore()
    app = FastAPI()
    app.include_router(build_v2_router(compiled_graph=compiled_graph, override_store=override_store))
    app.include_router(build_chat_router())
    return TestClient(app)


def _submit(client: TestClient, case_id: str, source_path: str):
    return client.post(
        "/claims",
        json={"case_id": case_id, "source_path": source_path},
        headers={"X-API-Key": _API_KEY},
    )


class TestAuthentication:
    def test_submit_claim_without_api_key_returns_401(self, v2_client: TestClient, tmp_path):
        response = v2_client.post(
            "/claims", json={"case_id": "CLM-8001", "source_path": _stage_deposit_folder(tmp_path)}
        )
        assert response.status_code == 401

    def test_submit_claim_with_wrong_api_key_returns_401(self, v2_client: TestClient, tmp_path):
        response = v2_client.post(
            "/claims",
            json={"case_id": "CLM-8001", "source_path": _stage_deposit_folder(tmp_path)},
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401

    def test_get_claim_status_does_not_require_api_key(self, v2_client: TestClient):
        response = v2_client.get("/claims/CLM-8099")
        assert response.status_code == 404  # traité, pas 401

    def test_override_without_api_key_returns_401(self, v2_client: TestClient):
        response = v2_client.post(
            "/claims/CLM-8001/override",
            json={"actor": "a", "action": "CONFIRM", "justification": "motif"},
        )
        assert response.status_code == 401


class TestSubmitClaim:
    def test_submit_reaches_terminal_decision_never_interrupted(
        self, v2_client: TestClient, tmp_path
    ):
        response = _submit(v2_client, "CLM-8001", _stage_deposit_folder(tmp_path))
        assert response.status_code == 201
        body = response.json()
        assert body["case_id"] == "CLM-8001"
        assert body["final_decision"] in {d.value for d in ClaimDecisionV2}
        assert body["current_step"] == "finalize"

    def test_response_never_exposes_pending_review_or_interrupted_fields(
        self, v2_client: TestClient, tmp_path
    ):
        """Contrairement à V1 (``ClaimStatusResponse.pending_review``/``interrupted``),
        le schéma V2 ne porte structurellement aucun champ de ce type — le
        graphe ne s'interrompt jamais (§0 du plan)."""
        response = _submit(v2_client, "CLM-8002", _stage_deposit_folder(tmp_path))
        body = response.json()
        assert "pending_review" not in body
        assert "interrupted" not in body

    def test_empty_folder_short_circuits_to_reject(self, v2_client: TestClient, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        response = _submit(v2_client, "CLM-8003", str(empty_dir))
        assert response.status_code == 201
        body = response.json()
        assert body["final_decision"] == ClaimDecisionV2.REJECT.value
        assert "document_understanding" not in body["completed_steps"]


class TestGetClaimStatus:
    def test_unknown_case_returns_404(self, v2_client: TestClient):
        response = v2_client.get("/claims/CLM-8098")
        assert response.status_code == 404

    def test_known_case_returns_same_final_decision_as_submission(
        self, v2_client: TestClient, tmp_path
    ):
        submit_response = _submit(v2_client, "CLM-8004", _stage_deposit_folder(tmp_path))
        status_response = v2_client.get("/claims/CLM-8004")
        assert status_response.status_code == 200
        assert status_response.json()["final_decision"] == submit_response.json()["final_decision"]


class TestOverride:
    def test_override_on_unknown_case_returns_404(self, v2_client: TestClient):
        response = v2_client.post(
            "/claims/CLM-8097/override",
            json={"actor": "gestionnaire.dupont", "action": "CONFIRM", "justification": "motif valide"},
            headers={"X-API-Key": _API_KEY},
        )
        assert response.status_code == 404

    def test_override_recorded_after_submission(self, v2_client: TestClient, tmp_path):
        _submit(v2_client, "CLM-8005", _stage_deposit_folder(tmp_path))
        response = v2_client.post(
            "/claims/CLM-8005/override",
            json={
                "actor": "gestionnaire.dupont",
                "action": "OVERRIDE_REJECT",
                "justification": "Document complémentaire reçu hors pipeline, dossier réexaminé.",
            },
            headers={"X-API-Key": _API_KEY},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["case_id"] == "CLM-8005"
        assert body["action"] == "OVERRIDE_REJECT"
        assert body["original_decision"] is not None

    def test_override_without_justification_returns_422(self, v2_client: TestClient, tmp_path):
        _submit(v2_client, "CLM-8006", _stage_deposit_folder(tmp_path))
        response = v2_client.post(
            "/claims/CLM-8006/override",
            json={"actor": "a", "action": "CONFIRM", "justification": ""},
            headers={"X-API-Key": _API_KEY},
        )
        assert response.status_code == 422


class TestChatStub:
    def test_chat_returns_501_not_implemented(self, v2_client: TestClient):
        response = v2_client.post(
            "/chat",
            json={"message": "pourquoi ce dossier est rejeté ?"},
            headers={"X-API-Key": _API_KEY},
        )
        assert response.status_code == 501

    def test_chat_requires_api_key(self, v2_client: TestClient):
        response = v2_client.post("/chat", json={"message": "test"})
        assert response.status_code == 401
