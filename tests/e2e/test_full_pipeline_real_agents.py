"""Pipeline complet avec les 11 agents réels (Phase A déterministe réelle,
LLM stubbé via l'autouse ``deterministic_agent_llm``), piloté via l'API
réelle (``TestClient``, ``real_app_client`` — aucun nœud mocké) sur de
vraies fixtures de démo.

**Résout le gap documenté dans `CLAUDE.md`** (« rien en production ne
construit ocr_input/fhir_input/coding_input/identity_coverage_input ») via
``graph/input_builders.py`` — voir aussi le plan de câblage minimal.
**Limite MVP assumée, non résolue ici** (décision AZIZ) : un seul document
par dossier passe par l'OCR (`ClaimState.ocr_result` reste singulier),
`coding_input` reste toujours `procedures=[]`/`medications=[]` — aucune
extraction acte/médicament fiable n'existe encore.

Marqué ``e2e`` : nécessite le binaire système ``tesseract`` (OCR réel via
``pytesseract``) — exécuter avec ``pytest -m "not e2e"`` si absent.

Sur les 4 dossiers échantillons, la Phase A réelle (confiance OCR/warnings
de schéma FHIR sur les données Synthea) aboutit systématiquement à
``NEEDS_REVIEW`` sur ``ocr_result``/``fhir_result`` — le pipeline s'arrête
donc à ``verification_fan_in`` avant ``identity_coverage``, court-circuit
déjà existant et volontaire (voir `graph/edges.py::route_verification_fan_in`,
Phase 2 du plan de remédiation). Les assertions ci-dessous n'imposent donc
pas un chemin précis (early-exit à `verification_fan_in` OU poursuite
jusqu'à `case_reviewer`) — dans les deux cas le pipeline doit atteindre
``needs_review`` proprement, sans jamais l'erreur "entrée absente" qui
caractérisait le bug résolu ici.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.e2e.conftest import fixture_input_dir
from tests.support.api_client import submit_claim

pytestmark = pytest.mark.e2e

_SAMPLE_CASE_IDS = ["CLM-0001", "CLM-0010", "CLM-0020", "CLM-0030"]

_MISSING_INPUT_MARKERS = (
    "ocr_input absent",
    "fhir_input absent",
    "entrée absente",
)


class TestFullPipelineRealAgentsReachesHumanReview:
    @pytest.mark.parametrize("case_id", _SAMPLE_CASE_IDS)
    def test_submission_never_fails_on_missing_input(
        self, real_app_client: TestClient, case_id: str
    ) -> None:
        """Non-régression directe sur le bug résolu : plus jamais de FAIL
        par absence d'``ocr_input``/``fhir_input`` sur une vraie soumission."""
        response = submit_claim(real_app_client, case_id, fixture_input_dir(case_id))

        assert response.status_code == 201
        body = response.json()
        for marker in _MISSING_INPUT_MARKERS:
            assert not any(marker in err for err in body["errors"]), body["errors"]

    def test_submission_runs_document_ocr_and_fhir_validator_for_real(
        self, real_app_client: TestClient
    ) -> None:
        response = submit_claim(real_app_client, "CLM-0001", fixture_input_dir("CLM-0001"))

        assert response.status_code == 201
        body = response.json()
        assert {"document_ocr_agent", "fhir_validation", "verification_fan_in"}.issubset(
            set(body["completed_steps"])
        )

    @pytest.mark.parametrize("case_id", _SAMPLE_CASE_IDS)
    def test_submission_reaches_human_review_not_failure(
        self, real_app_client: TestClient, case_id: str
    ) -> None:
        """Que le pipeline s'arrête tôt (``verification_fan_in``, OCR/FHIR
        NEEDS_REVIEW) ou aille jusqu'à ``case_reviewer`` (jamais
        auto-approuvé sous LLM stubbé — confiance par défaut 0.0), le
        dossier doit toujours atteindre la revue humaine, jamais ``failure``."""
        response = submit_claim(real_app_client, case_id, fixture_input_dir(case_id))

        assert response.status_code == 201
        body = response.json()
        assert body["current_step"] == "needs_review"
        assert body["interrupted"] is True
        assert body["pending_review"] is not None
        assert "failure" not in body["completed_steps"]

    def test_submission_never_leaks_raw_content(self, real_app_client: TestClient) -> None:
        response = submit_claim(real_app_client, "CLM-0001", fixture_input_dir("CLM-0001"))
        raw = response.text
        assert "full_text" not in raw
        assert "system_prompt" not in raw
