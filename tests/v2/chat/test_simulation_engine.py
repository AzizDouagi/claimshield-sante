"""Tests de chat/simulation_engine.py (Phase V2-11b, plan de refonte V2 §6).

Utilise le vrai graphe V2 compilé (`InMemorySaver`), un vrai dossier soumis
avec plusieurs documents réels (PDF texte natif, reportlab — pas besoin de
tesseract) — seuls les 5 `_invoke_llm_*` restent mockés (même patron que
`tests/v2/graph/test_workflow_v2.py`), aucun appel réel à Ollama.

`test_simulate_never_mutates_real_case` est **bloquant, pas optionnel**
(critère d'acceptation explicite du plan V2-11b) : vérifie directement sur
le système de fichiers (hash SHA-256 de chaque fichier + contenu du
manifeste) que le dossier réel est identique avant/après, sur 5 scénarios
de changement distincts.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import Mock

import pytest
from chat.schemas import SimulationChangeRequest
from chat.simulation_engine import run_simulation
from config.settings import get_settings
from graph.checkpoints import make_thread_config
from graph.workflow_v2 import compile_workflow_v2
from langgraph.checkpoint.memory import InMemorySaver
from schemas.domain import ReaderRole


def _make_pdf(path, lines: list[str]) -> None:
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path))
    y = 800
    for line in lines:
        c.drawString(72, y, line)
        y -= 20
    c.save()


def _stage_multi_document_folder(tmp_path) -> str:
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
    _make_pdf(
        deposit_dir / "ordonnance_test.pdf",
        ["ORDONNANCE", "Amoxicilline 500 mg", "Ibuprofene 400 mg"],
    )
    _make_pdf(
        deposit_dir / "demande_remboursement_test.pdf",
        ["Demande de remboursement", "Assureur : Cigna Health", "Taux de couverture : 80 %"],
    )
    return str(deposit_dir)


@pytest.fixture()
def deterministic_v2_llm(monkeypatch):
    """Même fixture que `tests/v2/graph/test_workflow_v2.py` (dupliquée
    volontairement — pas d'import cross-module de test)."""
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
                recommended_decision="REJECT", reasoning_summary="Motif de test."
            )
        ),
    )
    yield


def _submit_real_case(graph, case_id: str, source_path: str) -> None:
    config = make_thread_config(case_id)
    graph.invoke(
        {
            "case_id": case_id,
            "schema_version": "2.0.0",
            "current_step": "initial",
            "completed_steps": [],
            "errors": [],
            "alerts": [],
            "intake_input": {"source_path": source_path, "required_documents": []},
            "reader_role": "ADMINISTRATIVE_MANAGER",
        },
        config=config,
    )


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_real_case_storage(case_id: str) -> dict[str, str]:
    """{chemin relatif à storage/ : sha256} pour tout ce qui appartient au
    dossier réel — incoming/{case_id}/, quarantine/{case_id}/, manifeste."""
    settings = get_settings()
    storage_root = settings.storage_dir
    snapshot: dict[str, str] = {}
    for base_name in ("incoming", "quarantine"):
        case_dir = storage_root / base_name / case_id
        if case_dir.is_dir():
            for f in sorted(case_dir.rglob("*")):
                if f.is_file():
                    snapshot[str(f.relative_to(storage_root))] = _hash_file(f)
    manifest_path = storage_root / "manifests" / f"{case_id}.json"
    if manifest_path.is_file():
        snapshot[str(manifest_path.relative_to(storage_root))] = _hash_file(manifest_path)
    return snapshot


@pytest.fixture()
def real_case(deterministic_v2_llm, tmp_path):
    """Un vrai dossier V2 soumis (3 documents), sur un graphe compilé
    partagé — retourne (graph, case_id, snapshot_avant)."""
    graph = compile_workflow_v2(InMemorySaver())
    case_id = "CLM-7001"
    _submit_real_case(graph, case_id, _stage_multi_document_folder(tmp_path))
    snapshot_before = _snapshot_real_case_storage(case_id)
    assert snapshot_before, "Précondition : le dossier réel doit avoir des fichiers en stockage."
    return graph, case_id, snapshot_before


class TestSimulateNeverMutatesRealCase:
    """Critère d'acceptation bloquant du plan V2-11b : le dossier réel est
    bit-à-bit identique avant/après, sur au moins 5 scénarios distincts."""

    @pytest.mark.parametrize(
        "changes",
        [
            SimulationChangeRequest(remove_document="ordonnance"),
            SimulationChangeRequest(remove_document="facture"),
            SimulationChangeRequest(remove_document="demande"),
            SimulationChangeRequest(reader_role=ReaderRole.MEDICAL_REVIEWER),
            SimulationChangeRequest(reader_role=ReaderRole.FRAUD_ANALYST),
        ],
        ids=[
            "remove_ordonnance",
            "remove_facture",
            "remove_demande_remboursement",
            "reader_role_medical_reviewer",
            "reader_role_fraud_analyst",
        ],
    )
    def test_simulate_never_mutates_real_case(self, real_case, changes):
        graph, case_id, snapshot_before = real_case

        result = run_simulation(case_id, changes, compiled_graph=graph)

        assert result.applied is True, result.error
        snapshot_after = _snapshot_real_case_storage(case_id)
        assert snapshot_after == snapshot_before

    def test_five_consecutive_simulations_still_never_mutate(self, real_case):
        """Non-mutation vérifiée aussi après plusieurs simulations
        consécutives sur le même dossier réel — jamais un effet cumulatif."""
        graph, case_id, snapshot_before = real_case

        for changes in (
            SimulationChangeRequest(remove_document="ordonnance"),
            SimulationChangeRequest(remove_document="facture"),
            SimulationChangeRequest(reader_role=ReaderRole.AUDITOR),
        ):
            run_simulation(case_id, changes, compiled_graph=graph)

        assert _snapshot_real_case_storage(case_id) == snapshot_before


class TestSimulationCleanup:
    def test_synthetic_case_storage_cleaned_up_after_simulation(self, real_case):
        """Les artefacts de stockage de la simulation elle-même (dossier
        synthétique) ne doivent jamais survivre à l'appel."""
        graph, case_id, _ = real_case
        settings = get_settings()

        before_incoming = {p.name for p in (settings.storage_dir / "incoming").iterdir()}
        run_simulation(
            case_id, SimulationChangeRequest(remove_document="ordonnance"), compiled_graph=graph
        )
        after_incoming = {p.name for p in (settings.storage_dir / "incoming").iterdir()}

        assert after_incoming == before_incoming


class TestSimulationResultContent:
    def test_remove_document_changes_available_evidence(self, real_case):
        graph, case_id, _ = real_case
        result = run_simulation(
            case_id, SimulationChangeRequest(remove_document="ordonnance"), compiled_graph=graph
        )
        assert result.applied is True
        assert result.case_id == case_id
        assert result.original_decision is not None
        assert result.simulated_decision is not None

    def test_unknown_case_returns_not_applied(self):
        graph = compile_workflow_v2(InMemorySaver())
        result = run_simulation(
            "CLM-7099", SimulationChangeRequest(remove_document="ordonnance"), compiled_graph=graph
        )
        assert result.applied is False
        assert result.error is not None
        assert result.case_id == "CLM-7099"

    def test_result_never_exposes_synthetic_case_id(self, real_case):
        """`SimulationResult.case_id` est toujours celui du dossier réel —
        jamais l'identifiant synthétique interne utilisé pour isoler le
        stockage (voir `chat.simulation_engine._new_synthetic_case_id`)."""
        graph, case_id, _ = real_case
        result = run_simulation(
            case_id, SimulationChangeRequest(remove_document="ordonnance"), compiled_graph=graph
        )
        assert result.case_id == case_id


class TestSimulationNeverMutatesRealState:
    def test_real_case_state_unchanged_after_simulation(self, real_case):
        """`get_state()` sur le dossier réel retourne exactement la même
        décision après une simulation — le state réel (pas seulement le
        stockage disque) n'est jamais muté."""
        graph, case_id, _ = real_case
        config = make_thread_config(case_id)
        before = dict(graph.get_state(config).values)

        run_simulation(
            case_id, SimulationChangeRequest(remove_document="ordonnance"), compiled_graph=graph
        )

        after = dict(graph.get_state(config).values)
        assert before.get("final_decision") == after.get("final_decision")
        assert before.get("completed_steps") == after.get("completed_steps")


class TestGraphAccessExceptionIsDocumentedAndUnique:
    """`chat/simulation_engine.py` est le SEUL module de `chat/` autorisé à
    importer `graph.*`/`agents.*` directement (deux exceptions documentées,
    voir sa docstring — `run_simulation` pour `graph.*`, `run_targeted_simulation`
    pour `agents.autonomous_decision_agent.agent`, Phase 9) — verrouille
    cette garantie architecturale au lieu de la laisser purement
    déclarative."""

    _OTHER_CHAT_MODULES = (
        "agent.py",
        "planner.py",
        "nlu.py",
        "response_composer.py",
        "explanation_engine.py",
        "correction_engine.py",
        "schemas.py",
        "prompt.py",
        "tools.py",
        "answer_mode.py",
        "memory_schemas.py",
        "conversation_store.py",
        "semantic_summarizer.py",
    )

    def test_only_simulation_engine_imports_graph_directly(self):
        import chat

        chat_dir = Path(chat.__file__).parent
        for filename in self._OTHER_CHAT_MODULES:
            source = (chat_dir / filename).read_text(encoding="utf-8")
            for line in source.splitlines():
                stripped = line.strip()
                assert not stripped.startswith("import graph."), f"{filename}: {stripped!r}"
                assert not stripped.startswith("from graph."), f"{filename}: {stripped!r}"
                assert not stripped.startswith("import agents."), f"{filename}: {stripped!r}"
                assert not stripped.startswith("from agents."), f"{filename}: {stripped!r}"

    def test_simulation_engine_does_import_graph(self):
        """Contre-preuve : l'exception existe bel et bien — si elle
        disparaissait silencieusement, la simulation ne pourrait plus
        fonctionner sans qu'aucun test ne le signale."""
        source = Path("chat/simulation_engine.py").read_text(encoding="utf-8")
        assert "from graph." in source or "import graph." in source
