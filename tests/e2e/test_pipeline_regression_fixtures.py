"""Non-régression rapide (nœuds ``document_ocr``/``fhir_validator`` mockés,
pas de dépendance à ``tesseract``) sur la partie du pipeline exercée avec de
vrais fichiers sur plusieurs dossiers de fixtures : ``claim_intake``,
``security_gate`` et ``privacy`` tournent en Phase A réelle (I/O fichiers,
scan sécurité, RBAC/pseudonymisation) — seul le LLM est stubbé (autouse
``deterministic_agent_llm``).

**Complémentaire à `tests/e2e/test_full_pipeline_real_agents.py`**, qui
exerce lui aussi ces mêmes fixtures mais avec les 11 agents réels (y compris
``document_ocr``/``fhir_validator``, câblés via ``graph/input_builders.py``
depuis la résolution du gap documenté dans `CLAUDE.md`) — plus lent
(tesseract requis) mais couvre le chemin réel de bout en bout. Cette suite-ci
reste utile comme filet rapide sur la partie ingestion/sécurité/privacy sans
dépendre de l'OCR réel.
"""
from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

import graph.workflow as wf
from api.main import create_app
from schemas.domain import VerificationStatus
from tests.e2e.conftest import fixture_input_dir
from tests.support.api_client import submit_claim
from tests.support.stubs import StubResult

pytestmark = pytest.mark.e2e

_SAMPLE_CASE_IDS = ["CLM-0001", "CLM-0010", "CLM-0020", "CLM-0030"]


def _mock_document_ocr(state: dict) -> dict:
    return {
        "ocr_result": StubResult(status=VerificationStatus.NEEDS_REVIEW),
        "ocr_input": None,
        "completed_steps": ["document_ocr"],
        "alerts": ["[document_ocr] confiance limite — revue requise."],
    }


def _mock_fhir_validator(state: dict) -> dict:
    return {
        "fhir_result": StubResult(status=VerificationStatus.PASS),
        "fhir_input": None,
        "completed_steps": ["fhir_validator"],
    }


@pytest.fixture
def partial_real_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """``claim_intake``/``security_gate``/``privacy`` réels ; ``document_ocr``/
    ``fhir_validator`` mockés (câblage manifest→input pas encore construit —
    voir docstring du module)."""
    monkeypatch.setattr(wf, "node_document_ocr", _mock_document_ocr)
    monkeypatch.setattr(wf, "node_fhir_validator", _mock_fhir_validator)

    app = create_app(checkpointer=InMemorySaver())
    return TestClient(app)


class TestRealIngestionSecurityPrivacyAcrossFixtures:
    @pytest.mark.parametrize("case_id", _SAMPLE_CASE_IDS)
    def test_submission_completes_without_exception(
        self, partial_real_client: TestClient, case_id: str
    ) -> None:
        response = submit_claim(partial_real_client, case_id, fixture_input_dir(case_id))

        assert response.status_code == 201
        body: dict[str, Any] = response.json()
        assert body["case_id"] == case_id
        assert body["errors"] == []
        assert body["interrupted"] is True
        assert body["current_step"] == "needs_review"
        assert {"claim_intake", "security_gate", "privacy"}.issubset(set(body["completed_steps"]))

    def test_submission_never_leaks_raw_content(self, partial_real_client: TestClient) -> None:
        response = submit_claim(partial_real_client, "CLM-0001", fixture_input_dir("CLM-0001"))
        raw = response.text
        assert "full_text" not in raw
        assert "system_prompt" not in raw
