"""Auto-approbation P1-4 (``AUTO_APPROVED_LOW_RISK``) exercée au niveau API —
complète ``tests/graph/test_workflow_auto_approval.py`` (qui prouve le
chemin uniquement au niveau du graphe, via ``app.invoke()`` direct) en
prouvant que ``POST /claims`` lui-même renvoie une réponse ``finalize``
sans jamais passer par ``needs_review``.

Même stratégie de mock que ``tests/api/test_main.py`` (7 agents amont
mockés, aucun appel LLM réel) — ``case_reviewer_impl`` est injecté via le
mécanisme officiel ``compile_workflow(case_reviewer_impl=...)``, rendu
accessible côté API par ``api.main.create_app(compiled_graph=...)``.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from api.main import create_app
from graph.workflow import compile_workflow
from tests.graph.test_workflow_auto_approval import deterministic_agents  # noqa: F401
from tests.support.api_client import submit_claim
from tests.support.stubs import CaseReviewerApproveStub

pytestmark = pytest.mark.e2e


@pytest.fixture
def auto_approval_client(deterministic_agents: None) -> TestClient:  # noqa: F811
    graph = compile_workflow(
        InMemorySaver(),
        interrupt_before=[],
        case_reviewer_impl=CaseReviewerApproveStub(auto_decision="AUTO_APPROVED_LOW_RISK"),
    )
    app = create_app(compiled_graph=graph)
    return TestClient(app)


class TestApiAutoApproval:
    def test_submission_reaches_finalize_without_interruption(self, auto_approval_client: TestClient):
        response = submit_claim(auto_approval_client, "CLM-2001", "/tmp/does-not-need-to-exist")

        assert response.status_code == 201
        body = response.json()
        assert body["interrupted"] is False
        assert body["current_step"] == "finalize"
        assert body["final_recommendation"] == "APPROVE"
        assert "needs_review" not in body["completed_steps"]
        assert "await_human_review" not in body["completed_steps"]
        assert "audit" in body["completed_steps"]
        assert "finalize" in body["completed_steps"]
