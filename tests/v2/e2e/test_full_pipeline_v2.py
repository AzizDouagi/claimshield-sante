"""Pipeline V2 complet, sans aucun nœud mocké, sur de vraies fixtures de
démo — plan de refonte V2, Phase V2-10 (jalon de stabilité, obligatoire
avant toute ligne de `chat/`).

Même convention que ``tests/e2e/test_full_pipeline_real_agents.py`` (V1,
non modifié) : la Phase A déterministe des 5 agents V2 s'exécute réellement
(I/O fichiers, OCR réel via ``pytesseract``, validation FHIR structurelle,
vue privacy minimisée) sur de vrais documents — seul le LLM (Phase B) reste
stubbé via les 5 ``_invoke_llm_*`` (même patron que
``tests/v2/graph/test_workflow_v2.py::deterministic_v2_llm``, dupliqué
volontairement, pas d'import cross-module de test) : la suite pytest
n'appelle jamais un vrai Ollama, ici comme partout ailleurs dans le projet.
La preuve avec le **vrai** LLM sur les 37 fixtures est le rôle de
``scripts/evaluate_recommendations_v2.py`` (hors CI, exécuté manuellement).

Marqué ``e2e`` : nécessite le binaire système ``tesseract`` — exclure avec
``pytest -m "not e2e"`` si absent.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from graph.checkpoints import make_thread_config
from graph.workflow_v2 import compile_workflow_v2
from schemas.domain import ClaimDecisionV2

pytestmark = pytest.mark.e2e

FIXTURES_ROOT = Path(__file__).resolve().parents[3] / "datasets" / "fixtures" / "valid"
_SAMPLE_CASE_IDS = ["CLM-0001", "CLM-0010", "CLM-0020", "CLM-0030"]


def _fixture_input_dir(case_id: str) -> str:
    input_dir = FIXTURES_ROOT / case_id / "input"
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Fixture introuvable : {input_dir}")
    return str(input_dir)


@pytest.fixture()
def deterministic_v2_llm(monkeypatch):
    """Mocke les 5 fonctions `_invoke_llm_*` des agents V2 — aucun appel
    réel à Ollama, même dans ce test « e2e » (voir docstring du module).
    Réponses neutres/optimistes : la Phase B ne doit jamais elle-même
    bloquer un dossier dont la Phase A réelle est propre."""
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
                recommended_decision="APPROVE", reasoning_summary="Dossier conforme."
            )
        ),
    )
    yield


def _run_pipeline(case_id: str) -> dict:
    app = compile_workflow_v2(InMemorySaver())
    config = make_thread_config(case_id)
    return app.invoke(
        {
            "case_id": case_id,
            "schema_version": "2.0.0",
            "current_step": "initial",
            "completed_steps": [],
            "errors": [],
            "alerts": [],
            "intake_input": {"source_path": _fixture_input_dir(case_id), "required_documents": []},
            "reader_role": "ADMINISTRATIVE_MANAGER",
        },
        config=config,
    )


class TestFullPipelineRealDocumentsNeverFails:
    @pytest.mark.parametrize("case_id", _SAMPLE_CASE_IDS)
    def test_pipeline_completes_without_exception(self, deterministic_v2_llm, case_id: str) -> None:
        """Non-régression directe sur le jalon de stabilité V2-10 : un vrai
        dossier de démo, avec de vrais documents (OCR/FHIR réels), traverse
        le graphe complet jusqu'à `finalize` sans jamais lever."""
        state = _run_pipeline(case_id)

        assert state["current_step"] == "finalize"
        assert state["final_decision"] in set(ClaimDecisionV2)

    @pytest.mark.parametrize("case_id", _SAMPLE_CASE_IDS)
    def test_pipeline_never_interrupts(self, deterministic_v2_llm, case_id: str) -> None:
        app = compile_workflow_v2(InMemorySaver())
        config = make_thread_config(case_id)
        app.invoke(
            {
                "case_id": case_id,
                "schema_version": "2.0.0",
                "current_step": "initial",
                "completed_steps": [],
                "errors": [],
                "alerts": [],
                "intake_input": {"source_path": _fixture_input_dir(case_id), "required_documents": []},
                "reader_role": "ADMINISTRATIVE_MANAGER",
            },
            config=config,
        )
        snapshot = app.get_state(config)
        assert snapshot.next == ()

    def test_pipeline_runs_document_ocr_and_fhir_validation_for_real(
        self, deterministic_v2_llm
    ) -> None:
        """Preuve que la Phase A réelle (pas seulement la topologie) s'est
        exécutée : un vrai document OCR est présent dans le résultat, avec
        un statut FHIR structurel réellement calculé."""
        state = _run_pipeline("CLM-0001")

        assert "document_understanding" in state["completed_steps"]
        result = state["document_understanding_result"]
        assert result.extraction is not None
        assert result.extraction.confidence_score is not None
        assert result.fhir_summary is not None
        assert result.fhir_summary["validation_scope"] in {"STRUCTURAL_ONLY", "NOT_PROVIDED"}

    @pytest.mark.parametrize("case_id", _SAMPLE_CASE_IDS)
    def test_pipeline_traverses_all_seven_nodes_when_accepted(
        self, deterministic_v2_llm, case_id: str
    ) -> None:
        """Sur un dossier accepté à l'admission (Phase A réelle propre pour
        les 4 fixtures échantillons), aucun raccourci — les 7 nœuds
        s'exécutent tous, jamais un court-circuit silencieux."""
        state = _run_pipeline(case_id)

        if state["intake_safety_result"].status.value != "ACCEPTED":
            pytest.skip(f"{case_id} : admission non ACCEPTED en Phase A réelle, court-circuit attendu.")

        for step in (
            "intake_safety",
            "document_understanding",
            "eligibility",
            "medical_risk",
            "autonomous_decision",
            "audit_service",
            "finalize",
        ):
            assert step in state["completed_steps"], f"{step} manquant pour {case_id}"

    def test_pipeline_never_leaks_raw_content(self, deterministic_v2_llm) -> None:
        state = _run_pipeline("CLM-0001")
        serialized = str(state)
        assert "system_prompt" not in serialized
        extraction = state["document_understanding_result"].extraction
        if extraction is not None:
            assert extraction.full_text == ""

    def test_pipeline_always_produces_audit_trail(self, deterministic_v2_llm) -> None:
        state = _run_pipeline("CLM-0001")
        assert len(state["audit_trail"]) >= 1
