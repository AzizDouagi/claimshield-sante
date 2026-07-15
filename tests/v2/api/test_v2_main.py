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

import httpx
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
def v2_client(deterministic_v2_llm, monkeypatch) -> TestClient:
    compiled_graph = compile_workflow_v2(InMemorySaver())
    override_store = OverrideStore()
    app = FastAPI()
    app.include_router(build_v2_router(compiled_graph=compiled_graph, override_store=override_store))
    app.include_router(build_chat_router())

    def _test_chat_client() -> httpx.AsyncClient:
        # `chat/tools.py` appelle `/v2/*` en HTTP — jamais un vrai serveur en
        # test, cible la même application FastAPI en mémoire (ASGITransport).
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",  # routeur monté sans préfixe /v2 dans cette app de test
            headers={"X-API-Key": _API_KEY},
            timeout=30.0,
        )

    monkeypatch.setattr("chat.tools._build_client", _test_chat_client)
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


class TestChatEndpoint:
    """Tests d'intégration bout en bout (`chat.tools` → HTTP réel via
    ASGITransport → `/v2/claims/*`) — la couverture exhaustive du NLU/
    planner/response_composer/anti-hallucination vit dans
    `tests/v2/chat/`, ce fichier vérifie uniquement le câblage endpoint."""

    def test_chat_requires_api_key(self, v2_client: TestClient):
        response = v2_client.post("/chat", json={"message": "test"})
        assert response.status_code == 401

    def test_chat_explain_returns_grounded_reply_for_known_case(
        self, v2_client: TestClient, tmp_path, monkeypatch
    ):
        from chat.schemas import ChatIntent, LlmIntentDecision

        case_id = "CLM-8010"
        _submit(v2_client, case_id, _stage_deposit_folder(tmp_path))

        monkeypatch.setattr(
            "chat.nlu._invoke_llm_intent",
            Mock(return_value=LlmIntentDecision(intents=[ChatIntent.EXPLAIN], case_id=case_id)),
        )
        compose_spy = Mock(return_value=f"Le dossier {case_id} a été traité, voici les motifs.")
        monkeypatch.setattr("chat.response_composer._invoke_llm_compose", compose_spy)

        response = v2_client.post(
            "/chat",
            json={"message": "Pourquoi cette décision ?", "case_id": case_id},
            headers={"X-API-Key": _API_KEY},
        )
        assert response.status_code == 200
        assert response.json()["reply"]
        # Preuve que `chat.tools.explain_claim` a bien récupéré le contexte
        # réel du dossier via HTTP (pas un 404 masqué) : le composer a été
        # appelé avec un `case_id` non vide dans ses données groundées —
        # régression directe sur un bug réel trouvé pendant l'écriture de
        # ces tests (préfixe /v2 absent de l'app de test, `get_claim_context`
        # retournait silencieusement `None`).
        compose_spy.assert_called_once()
        composed_data = compose_spy.call_args[0][0]
        assert composed_data["case_id"] == case_id
        assert composed_data.get("explanation") is not None
        assert composed_data["explanation"]["case_id"] == case_id

    def test_chat_unknown_case_returns_graceful_message(self, v2_client: TestClient, monkeypatch):
        from chat.schemas import ChatIntent, LlmIntentDecision

        monkeypatch.setattr(
            "chat.nlu._invoke_llm_intent",
            Mock(
                return_value=LlmIntentDecision(intents=[ChatIntent.EXPLAIN], case_id="CLM-8099")
            ),
        )
        response = v2_client.post(
            "/chat",
            json={"message": "Pourquoi ?", "case_id": "CLM-8099"},
            headers={"X-API-Key": _API_KEY},
        )
        assert response.status_code == 200
        assert "introuvable" in response.json()["reply"].lower()

    def test_chat_audit_intent_returns_grounded_reply(self, v2_client: TestClient, monkeypatch):
        """AUDIT est livré depuis V2-11c (plus de « bientôt disponible » —
        voir `tests/v2/chat/test_planner.py::TestPlanNotYetAvailable`, qui
        documente pourquoi ce chemin défensif est désormais inatteignable
        avec de vraies valeurs `ChatIntent`)."""
        from chat.schemas import ChatIntent, LlmIntentDecision

        monkeypatch.setattr(
            "chat.nlu._invoke_llm_intent",
            Mock(return_value=LlmIntentDecision(intents=[ChatIntent.AUDIT], case_id="CLM-8011")),
        )
        compose_spy = Mock(return_value="Aucun événement d'audit trouvé pour ce dossier.")
        monkeypatch.setattr("chat.response_composer._invoke_llm_compose", compose_spy)
        response = v2_client.post(
            "/chat",
            json={"message": "Montre-moi l'historique complet.", "case_id": "CLM-8011"},
            headers={"X-API-Key": _API_KEY},
        )
        assert response.status_code == 200
        assert response.json()["reply"] == "Aucun événement d'audit trouvé pour ce dossier."
        compose_spy.assert_called_once()

    def test_chat_llm_unavailable_returns_clarify_message(self, v2_client: TestClient, monkeypatch):
        monkeypatch.setattr("chat.nlu._invoke_llm_intent", Mock(return_value=None))
        response = v2_client.post(
            "/chat",
            json={"message": "Bonjour"},
            headers={"X-API-Key": _API_KEY},
        )
        assert response.status_code == 200
        assert response.json()["reply"]
