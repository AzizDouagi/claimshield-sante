"""Tests d'intégration du graphe V2 compilé — Phase V2-7.

Chaque agent est monkeypatché sur sa propre fonction `_invoke_llm_*` (même
patron que `tests/conftest.py::deterministic_agent_llm` côté V1) — jamais
d'appel réel à Ollama. Vérifie la topologie réelle : court-circuit
BLOCKED/QUARANTINED, chemin nominal jusqu'à `finalize`, `audit_service`
toujours traversé, et l'absence de toute interruption (`__interrupt__`),
garantie centrale du plan V2 (§0 — « le graphe V2 ne bloque jamais »).
"""
from __future__ import annotations

from unittest.mock import Mock

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from graph.checkpoints import make_thread_config
from graph.workflow_v2 import build_workflow_v2, compile_workflow_v2
from schemas.domain import ClaimDecisionV2


def _make_pdf(path, lines: list[str]) -> None:
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path))
    y = 800
    for line in lines:
        c.drawString(72, y, line)
        y -= 20
    c.save()


def _stage_deposit_folder(tmp_path) -> str:
    """Crée un répertoire de dépôt (jamais `storage/incoming/` directement —
    c'est le rôle d'`intake_safety_agent` d'y déplacer le fichier) contenant
    une facture PDF valide, texte natif (pas d'OCR/tesseract nécessaire)."""
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
    à Ollama pendant les tests d'intégration du graphe."""
    from agents.autonomous_decision_agent.schemas import LlmAutonomousDecision
    from agents.document_understanding_agent.schemas import LlmDocumentUnderstandingDecision
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
    # eligibility_agent délègue à identity_coverage_agent (V1) — mocké comme
    # le fait déjà tests/conftest.py::deterministic_agent_llm.
    from agents.identity_coverage_agent.schemas import LlmIdentityCoverageDecision

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


class TestNominalPath:
    def test_full_pipeline_reaches_finalize_through_all_eight_nodes(
        self, deterministic_v2_llm, tmp_path
    ):
        """Un dossier accepté à l'admission traverse réellement les 8 nœuds
        du graphe (aucun raccourci) et atteint `finalize` avec une décision
        parmi les 6 valides — sans présumer laquelle : les données
        synthétiques de ce test (aucun contrat, identité non appariable)
        déclenchent légitimement des signaux de risque réels côté
        `medical_risk`/`eligibility`, ce qui plafonne à son tour la
        décision autonome via `_allowed_decisions` (garde-fou
        `TestBoundedAuthority`, déjà testé exhaustivement dans
        `tests/v2/agents/test_autonomous_decision_agent.py`) — jamais un
        APPROVE artificiellement forcé ici."""
        case_id = "CLM-7001"
        source_path = _stage_deposit_folder(tmp_path)
        app = compile_workflow_v2(InMemorySaver())
        config = make_thread_config(case_id)
        state = app.invoke(
            {
                "case_id": case_id,
                "schema_version": "2.0.0",
                "current_step": "initial",
                "completed_steps": [],
                "intake_input": {"source_path": source_path},
                "reader_role": "ADMINISTRATIVE_MANAGER",
            },
            config=config,
        )
        assert state["current_step"] == "finalize"
        assert state["intake_safety_result"].status.value == "ACCEPTED"
        for step in (
            "intake_safety",
            "document_understanding",
            "eligibility",
            "medical_risk",
            "recovery",
            "autonomous_decision",
            "audit_service",
            "finalize",
        ):
            assert step in state["completed_steps"], f"{step} manquant dans completed_steps"
        assert state["final_decision"] in set(ClaimDecisionV2)
        assert len(state["audit_trail"]) >= 1

    def test_never_interrupts(self, deterministic_v2_llm, tmp_path):
        case_id = "CLM-7002"
        source_path = _stage_deposit_folder(tmp_path)
        app = compile_workflow_v2(InMemorySaver())
        config = make_thread_config(case_id)
        app.invoke(
            {
                "case_id": case_id,
                "schema_version": "2.0.0",
                "current_step": "initial",
                "completed_steps": [],
                "intake_input": {"source_path": source_path},
                "reader_role": "ADMINISTRATIVE_MANAGER",
            },
            config=config,
        )
        snapshot = app.get_state(config)
        assert snapshot.next == ()


class TestBlockedShortCircuit:
    def test_empty_folder_short_circuits_to_reject(self, deterministic_v2_llm, tmp_path):
        case_id = "CLM-7010"
        empty_dir = tmp_path / "empty_input"
        empty_dir.mkdir()
        app = compile_workflow_v2(InMemorySaver())
        config = make_thread_config(case_id)
        state = app.invoke(
            {
                "case_id": case_id,
                "schema_version": "2.0.0",
                "current_step": "initial",
                "completed_steps": [],
                "intake_input": {"source_path": str(empty_dir)},
            },
            config=config,
        )
        assert state["final_decision"] is ClaimDecisionV2.REJECT
        assert "document_understanding" not in state["completed_steps"]
        assert "eligibility" not in state["completed_steps"]
        assert "medical_risk" not in state["completed_steps"]
        assert "recovery" not in state["completed_steps"]
        assert "autonomous_decision" not in state["completed_steps"]
        assert "audit_service" in state["completed_steps"]
        assert "finalize" in state["completed_steps"]

    def test_audit_trail_always_populated(self, deterministic_v2_llm, tmp_path):
        case_id = "CLM-7011"
        empty_dir = tmp_path / "empty_input"
        empty_dir.mkdir()
        app = compile_workflow_v2(InMemorySaver())
        config = make_thread_config(case_id)
        state = app.invoke(
            {
                "case_id": case_id,
                "schema_version": "2.0.0",
                "current_step": "initial",
                "completed_steps": [],
                "intake_input": {"source_path": str(empty_dir)},
            },
            config=config,
        )
        assert len(state["audit_trail"]) >= 1


class TestTopology:
    def test_build_workflow_v2_has_eight_nodes(self):
        graph = build_workflow_v2()
        assert set(graph.nodes) == {
            "intake_safety",
            "document_understanding",
            "eligibility",
            "medical_risk",
            "recovery",
            "autonomous_decision",
            "audit_service",
            "finalize",
        }
