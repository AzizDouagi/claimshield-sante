"""Tests de chat/tools.py (Phase V2-11a) — wrappers HTTP vers `/v2/*`.

Cible une application FastAPI de test en mémoire (`httpx.ASGITransport`),
jamais un vrai serveur réseau. Vérifie aussi statiquement (`test_tools.py::
TestNoDirectAgentAccess`) que ce module n'importe jamais `graph.*`/
`agents.*` — même garantie que `ui/api_client_v2.py`.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI
from langgraph.checkpoint.memory import InMemorySaver

import chat.tools as chat_tools
from api.v2.claims import build_v2_router
from chat.schemas import CorrectionRecommendation, ExplanationFacts
from graph.workflow_v2 import compile_workflow_v2
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
            "Date du document : 2026-03-01",
            "Montant total facture : 500.00 USD",
        ],
    )
    return str(deposit_dir)


@pytest.fixture()
def wired_app(monkeypatch):
    """Monte le vrai routeur `/v2/claims` et branche `chat.tools._build_client`
    dessus — aucun appel réseau réel."""
    from agents.autonomous_decision_agent.schemas import LlmAutonomousDecision
    from agents.document_understanding_agent.schemas import LlmDocumentUnderstandingDecision
    from agents.identity_coverage_agent.schemas import LlmIdentityCoverageDecision
    from agents.intake_safety_agent.schemas import LlmIntakeSafetyDecision
    from agents.medical_risk_agent.schemas import LlmMedicalRiskDecision

    monkeypatch.setattr(
        "agents.intake_safety_agent.agent._invoke_llm_intake_safety",
        Mock(return_value=LlmIntakeSafetyDecision(status="ACCEPTED", reasons=["ok"], explanation="")),
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
                recommended_decision="REJECT", reasoning_summary="Motif de test."
            )
        ),
    )

    compiled_graph = compile_workflow_v2(InMemorySaver())
    app = FastAPI()
    app.include_router(build_v2_router(compiled_graph=compiled_graph, override_store=OverrideStore()))

    def _test_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",  # routeur monté sans préfixe /v2 dans cette app de test
            headers={"X-API-Key": _API_KEY},
            timeout=30.0,
        )

    monkeypatch.setattr("chat.tools._build_client", _test_client)
    return app


def _submit_via_app(app: FastAPI, case_id: str, source_path: str) -> None:
    from fastapi.testclient import TestClient

    TestClient(app).post(
        "/claims",
        json={"case_id": case_id, "source_path": source_path},
        headers={"X-API-Key": _API_KEY},
    )


class TestGetClaimContext:
    @pytest.mark.asyncio
    async def test_known_case_returns_dict(self, wired_app, tmp_path):
        case_id = "CLM-4001"
        _submit_via_app(wired_app, case_id, _stage_deposit_folder(tmp_path))
        context = await chat_tools.get_claim_context(case_id)
        assert context is not None
        assert context["case_id"] == case_id

    @pytest.mark.asyncio
    async def test_unknown_case_returns_none(self, wired_app):
        context = await chat_tools.get_claim_context("CLM-4099")
        assert context is None


class TestRunClaimAnalysis:
    @pytest.mark.asyncio
    async def test_same_data_as_get_claim_context(self, wired_app, tmp_path):
        case_id = "CLM-4002"
        _submit_via_app(wired_app, case_id, _stage_deposit_folder(tmp_path))
        analysis = await chat_tools.run_claim_analysis(case_id)
        context = await chat_tools.get_claim_context(case_id)
        assert analysis == context


class TestExplainClaim:
    @pytest.mark.asyncio
    async def test_known_case_returns_explanation_facts(self, wired_app, tmp_path):
        case_id = "CLM-4003"
        _submit_via_app(wired_app, case_id, _stage_deposit_folder(tmp_path))
        result = await chat_tools.explain_claim(case_id)
        assert isinstance(result, ExplanationFacts)
        assert result.case_id == case_id

    @pytest.mark.asyncio
    async def test_unknown_case_returns_none(self, wired_app):
        result = await chat_tools.explain_claim("CLM-4098")
        assert result is None


class TestRecommendCorrections:
    @pytest.mark.asyncio
    async def test_unknown_case_returns_empty_list(self, wired_app):
        result = await chat_tools.recommend_corrections("CLM-4097")
        assert result == []

    @pytest.mark.asyncio
    async def test_known_case_returns_list_of_recommendations(self, wired_app, tmp_path):
        case_id = "CLM-4004"
        _submit_via_app(wired_app, case_id, _stage_deposit_folder(tmp_path))
        result = await chat_tools.recommend_corrections(case_id)
        assert isinstance(result, list)
        assert all(isinstance(item, CorrectionRecommendation) for item in result)


class TestNoDirectAgentAccess:
    def test_tools_module_never_imports_agents_or_graph_directly(self):
        source = Path(chat_tools.__file__).read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("import agents.")
            assert not stripped.startswith("from agents.")
            assert not stripped.startswith("import graph.")
            assert not stripped.startswith("from graph.")

    def test_tools_module_only_talks_http(self):
        source = Path(chat_tools.__file__).read_text(encoding="utf-8")
        assert "httpx" in source
        assert "/claims/" in source
