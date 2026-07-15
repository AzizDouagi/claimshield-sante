"""Tests de agents/intake_safety_agent (V2) — Phase V2-2.

Porte les cas obligatoires de sécurité de `tests/security/test_input_gate.py`
et les cas nominaux/quarantaine de `tests/agents/test_claim_intake.py`,
adaptés au schéma fusionné `IntakeSafetyResult`. Chaque test LLM-dépendant
monkeypatche `_invoke_llm_intake_safety` (même patron que les tests V1
`test_claim_intake_llm.py`) — aucun appel réel à Ollama.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from agents.intake_safety_agent.agent import node, run
from agents.intake_safety_agent.schemas import LlmIntakeSafetyDecision
from config.settings import Settings
from schemas.domain import IntakeSafetyStatus
from services.storage import StorageService

_VALID_PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<< >>\nendobj\n%%EOF\n"


def _make_storage(tmp_path: Path) -> StorageService:
    s = Settings(  # type: ignore[call-arg]
        CLAIMSHIELD_STORAGE_DIR=str(tmp_path / "storage"),
        CLAIMSHIELD_QUARANTINE_DIR=str(tmp_path / "storage" / "quarantine"),
    )
    svc = StorageService(settings=s)
    svc.ensure_dirs()
    return svc


def _write_pdf(directory: Path, name: str, content: bytes = _VALID_PDF_BYTES) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(content)
    return path


def _accepted_llm_decision(**overrides) -> LlmIntakeSafetyDecision:
    defaults = {"status": "ACCEPTED", "reasons": ["Dossier conforme."], "explanation": ""}
    defaults.update(overrides)
    return LlmIntakeSafetyDecision(**defaults)


class TestNominalAcceptance:
    def test_valid_pdf_accepted(self, tmp_path, monkeypatch):
        svc = _make_storage(tmp_path)
        source = tmp_path / "input"
        _write_pdf(source, "facture.pdf")
        monkeypatch.setattr(
            "agents.intake_safety_agent.agent._invoke_llm_intake_safety",
            Mock(return_value=_accepted_llm_decision()),
        )
        result = run(case_id="CLM-2001", source_path=source, storage=svc)
        assert result.status is IntakeSafetyStatus.ACCEPTED
        assert result.manifest is not None
        assert result.manifest.alerts == []
        assert result.errors == []
        assert result.llm_trace.model_name

    def test_missing_required_document_quarantined(self, tmp_path, monkeypatch):
        svc = _make_storage(tmp_path)
        source = tmp_path / "input"
        _write_pdf(source, "facture.pdf")
        monkeypatch.setattr(
            "agents.intake_safety_agent.agent._invoke_llm_intake_safety",
            Mock(return_value=_accepted_llm_decision(status="QUARANTINED", reasons=["Document manquant."])),
        )
        result = run(
            case_id="CLM-2002",
            source_path=source,
            required_documents=["ordonnance.pdf"],
            storage=svc,
        )
        assert result.status is IntakeSafetyStatus.QUARANTINED
        assert result.manifest.alerts

    def test_duplicate_file_quarantined(self, tmp_path, monkeypatch):
        svc = _make_storage(tmp_path)
        source = tmp_path / "input"
        _write_pdf(source, "facture.pdf")
        _write_pdf(source, "facture_copie.pdf")  # même contenu → même SHA-256
        monkeypatch.setattr(
            "agents.intake_safety_agent.agent._invoke_llm_intake_safety",
            Mock(return_value=_accepted_llm_decision(status="QUARANTINED", reasons=["Doublon détecté."])),
        )
        result = run(case_id="CLM-2003", source_path=source, storage=svc)
        assert result.status is IntakeSafetyStatus.QUARANTINED


class TestMandatorySecurityCases:
    """Cas obligatoires — jamais ACCEPTED, LLM en échec → BLOCKED."""

    def test_dangerous_extension_blocked(self, tmp_path, monkeypatch):
        svc = _make_storage(tmp_path)
        source = tmp_path / "input"
        _write_pdf(source, "facture.exe")
        monkeypatch.setattr(
            "agents.intake_safety_agent.agent._invoke_llm_intake_safety",
            Mock(return_value=_accepted_llm_decision(status="BLOCKED", reasons=["Extension interdite."])),
        )
        result = run(case_id="CLM-2010", source_path=source, storage=svc)
        assert result.status is IntakeSafetyStatus.BLOCKED

    def test_oversized_file_blocked(self, tmp_path, monkeypatch):
        settings = Settings(  # type: ignore[call-arg]
            CLAIMSHIELD_STORAGE_DIR=str(tmp_path / "storage"),
            CLAIMSHIELD_QUARANTINE_DIR=str(tmp_path / "storage" / "quarantine"),
            CLAIMSHIELD_MAX_FILE_SIZE_MB=1,
        )
        svc = StorageService(settings=settings)
        svc.ensure_dirs()
        source = tmp_path / "input"
        source.mkdir(parents=True, exist_ok=True)
        oversized = source / "facture.pdf"
        oversized.write_bytes(_VALID_PDF_BYTES + b"0" * (2 * 1024 * 1024))
        monkeypatch.setattr(
            "agents.intake_safety_agent.agent._invoke_llm_intake_safety",
            Mock(return_value=_accepted_llm_decision(status="BLOCKED", reasons=["Fichier trop volumineux."])),
        )
        result = run(case_id="CLM-2011", source_path=source, storage=svc, settings=settings)
        assert result.status is IntakeSafetyStatus.BLOCKED

    def test_path_traversal_in_filename_blocked(self, tmp_path, monkeypatch):
        svc = _make_storage(tmp_path)
        source = tmp_path / "input"
        source.mkdir(parents=True, exist_ok=True)
        # Nom de fichier contenant une séquence de traversée — validate_filename
        # (tools.file_inspection, réutilisé tel quel) la refuse.
        traversal_name = "..%2F..%2Fetc%2Fpasswd.pdf"
        (source / traversal_name).write_bytes(_VALID_PDF_BYTES)
        monkeypatch.setattr(
            "agents.intake_safety_agent.agent._invoke_llm_intake_safety",
            Mock(return_value=_accepted_llm_decision(status="BLOCKED", reasons=["Nom de fichier invalide."])),
        )
        result = run(case_id="CLM-2012", source_path=source, storage=svc)
        assert result.status is IntakeSafetyStatus.BLOCKED

    def test_empty_claim_directory_blocked_without_llm_call(self, tmp_path, monkeypatch):
        svc = _make_storage(tmp_path)
        source = tmp_path / "input"
        source.mkdir(parents=True, exist_ok=True)
        spy = Mock()
        monkeypatch.setattr("agents.intake_safety_agent.agent._invoke_llm_intake_safety", spy)
        result = run(case_id="CLM-2013", source_path=source, storage=svc)
        assert result.status is IntakeSafetyStatus.BLOCKED
        spy.assert_not_called()

    def test_missing_source_directory_blocked_without_llm_call(self, tmp_path, monkeypatch):
        svc = _make_storage(tmp_path)
        spy = Mock()
        monkeypatch.setattr("agents.intake_safety_agent.agent._invoke_llm_intake_safety", spy)
        result = run(case_id="CLM-2014", source_path=tmp_path / "does_not_exist", storage=svc)
        assert result.status is IntakeSafetyStatus.BLOCKED
        spy.assert_not_called()


class TestLlmFailClosed:
    def test_llm_unavailable_forces_blocked_even_if_deterministic_accepted(self, tmp_path, monkeypatch):
        svc = _make_storage(tmp_path)
        source = tmp_path / "input"
        _write_pdf(source, "facture.pdf")
        monkeypatch.setattr(
            "agents.intake_safety_agent.agent._invoke_llm_intake_safety", Mock(return_value=None)
        )
        result = run(case_id="CLM-2020", source_path=source, storage=svc)
        assert result.status is IntakeSafetyStatus.BLOCKED
        assert result.errors  # motif structuré ajouté lors de l'escalade

    def test_llm_invalid_status_forces_blocked(self, tmp_path, monkeypatch):
        svc = _make_storage(tmp_path)
        source = tmp_path / "input"
        _write_pdf(source, "facture.pdf")

        class _BadDecision:
            status = "SOMETHING_ELSE"
            reasons: list[str] = []
            explanation = ""

        monkeypatch.setattr(
            "agents.intake_safety_agent.agent._invoke_llm_intake_safety",
            Mock(return_value=_BadDecision()),
        )
        result = run(case_id="CLM-2021", source_path=source, storage=svc)
        assert result.status is IntakeSafetyStatus.BLOCKED

    def test_llm_can_never_soften_a_deterministic_block(self, tmp_path, monkeypatch):
        """Garde-fou non négociable (plan V2 §7) : un fichier dangereux reste
        BLOCKED même si le LLM répond ACCEPTED."""
        svc = _make_storage(tmp_path)
        source = tmp_path / "input"
        _write_pdf(source, "facture.exe")
        monkeypatch.setattr(
            "agents.intake_safety_agent.agent._invoke_llm_intake_safety",
            Mock(return_value=_accepted_llm_decision(status="ACCEPTED", reasons=["tentative d'adoucissement"])),
        )
        result = run(case_id="CLM-2022", source_path=source, storage=svc)
        assert result.status is IntakeSafetyStatus.BLOCKED

    def test_llm_can_escalate_accepted_to_quarantined(self, tmp_path, monkeypatch):
        svc = _make_storage(tmp_path)
        source = tmp_path / "input"
        _write_pdf(source, "facture.pdf")
        monkeypatch.setattr(
            "agents.intake_safety_agent.agent._invoke_llm_intake_safety",
            Mock(return_value=_accepted_llm_decision(status="QUARANTINED", reasons=["Contexte global suspect."])),
        )
        result = run(case_id="CLM-2023", source_path=source, storage=svc)
        assert result.status is IntakeSafetyStatus.QUARANTINED


class TestStorageCollisionIsBlockedNotTechnicalFailure:
    """Correctif post-mesure V2-10 (AZIZ) : une collision de stockage
    (NO_OVERWRITE — fichier déjà committé lors d'un run antérieur pour le
    même case_id) est une décision de politique (BLOCKED), jamais une panne
    d'infrastructure (TECHNICAL_FAILURE). `agent.py` ligne ~538 ne doit plus
    déclencher TECHNICAL_FAILURE sur la seule présence de `errors` (liste
    brute incluant NO_OVERWRITE) — uniquement sur `errored` (fichiers
    réellement FileStatus.ERROR, pannes techniques)."""

    def test_resubmitting_same_case_id_is_blocked_not_technical_failure(self, tmp_path, monkeypatch):
        svc = _make_storage(tmp_path)
        source = tmp_path / "input"
        _write_pdf(source, "facture.pdf")
        monkeypatch.setattr(
            "agents.intake_safety_agent.agent._invoke_llm_intake_safety",
            Mock(return_value=_accepted_llm_decision()),
        )

        first = run(case_id="CLM-2050", source_path=source, storage=svc)
        assert first.status is IntakeSafetyStatus.ACCEPTED

        # Même case_id, même fichier — build_storage_name est déterministe
        # (pas d'aléa), donc le second commit collisionne physiquement avec
        # le premier sous storage/incoming/CLM-2050/.
        second = run(case_id="CLM-2050", source_path=source, storage=svc)
        assert second.status is IntakeSafetyStatus.BLOCKED
        assert second.status is not IntakeSafetyStatus.TECHNICAL_FAILURE


class TestNodeIntegration:
    def test_node_updates_state_on_acceptance(self, tmp_path, monkeypatch):
        svc = _make_storage(tmp_path)
        source = tmp_path / "input"
        _write_pdf(source, "facture.pdf")
        monkeypatch.setattr(
            "agents.intake_safety_agent.agent._invoke_llm_intake_safety",
            Mock(return_value=_accepted_llm_decision()),
        )
        monkeypatch.setattr(
            "agents.intake_safety_agent.agent.StorageService", lambda settings=None: svc
        )
        state = {
            "case_id": "CLM-2030",
            "schema_version": "2.0.0",
            "current_step": "initial",
            "completed_steps": [],
            "intake_input": {"source_path": str(source)},
        }
        updates = node(state)  # type: ignore[arg-type]
        assert updates["current_step"] == "intake_safety"
        assert updates["completed_steps"] == ["intake_safety"]
        assert updates["intake_input"] is None
        assert updates["intake_safety_result"].status is IntakeSafetyStatus.ACCEPTED
        assert "errors" not in updates

    def test_node_reports_errors_on_block(self, tmp_path, monkeypatch):
        svc = _make_storage(tmp_path)
        source = tmp_path / "input"
        _write_pdf(source, "facture.exe")
        monkeypatch.setattr(
            "agents.intake_safety_agent.agent._invoke_llm_intake_safety",
            Mock(return_value=_accepted_llm_decision(status="BLOCKED")),
        )
        monkeypatch.setattr(
            "agents.intake_safety_agent.agent.StorageService", lambda settings=None: svc
        )
        state = {
            "case_id": "CLM-2031",
            "schema_version": "2.0.0",
            "current_step": "initial",
            "completed_steps": [],
            "intake_input": {"source_path": str(source)},
        }
        updates = node(state)  # type: ignore[arg-type]
        assert updates["errors"]
        assert all(u.startswith("[intake_safety]") for u in updates["errors"])


class TestSecurityFindingsStructure:
    def test_security_findings_never_carry_raw_secret(self, tmp_path, monkeypatch):
        svc = _make_storage(tmp_path)
        source = tmp_path / "input"
        _write_pdf(source, "facture.exe")
        monkeypatch.setattr(
            "agents.intake_safety_agent.agent._invoke_llm_intake_safety",
            Mock(return_value=_accepted_llm_decision(status="BLOCKED")),
        )
        result = run(case_id="CLM-2040", source_path=source, storage=svc)
        for finding in result.security_findings:
            assert "api_key" not in finding.description.lower()
            assert (finding.evidence or "") != ""  or finding.evidence is None
