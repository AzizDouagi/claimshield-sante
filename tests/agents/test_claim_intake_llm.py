from pathlib import Path

from agents.claim_intake_agent.agent import run
from agents.claim_intake_agent.schemas import LlmIntakeDecision
from config.settings import Settings
from schemas.domain import IntakeReasonCode, IntakeStatus
from services.storage import StorageService


def _storage(tmp_path: Path) -> StorageService:
    settings = Settings(  # type: ignore[call-arg]
        CLAIMSHIELD_STORAGE_DIR=str(tmp_path / "storage"),
        CLAIMSHIELD_QUARANTINE_DIR=str(tmp_path / "storage" / "quarantine"),
    )
    service = StorageService(settings=settings)
    service.ensure_dirs()
    return service


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "claim.json").write_text('{"resourceType":"Bundle"}', encoding="utf-8")
    return source


def test_claim_intake_llm_nominal(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agents.claim_intake_agent.agent._invoke_llm_intake",
        lambda **_: LlmIntakeDecision(status="ACCEPTED", reasons=["Dossier accepté par LLM."]),
    )

    result = run("CLM-9002", _source(tmp_path), storage=_storage(tmp_path))

    assert result.status == IntakeStatus.ACCEPTED
    assert result.manifest.status == IntakeStatus.ACCEPTED
    assert result.reasons == ["Dossier accepté par LLM."]
    assert result.llm_metadata is not None


def test_claim_intake_llm_indisponible_error(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.claim_intake_agent.agent._invoke_llm_intake", lambda **_: None)

    result = run("CLM-9003", _source(tmp_path), storage=_storage(tmp_path))

    assert result.status == IntakeStatus.ERROR
    assert result.manifest.status == IntakeStatus.ERROR
    assert result.llm_metadata is not None
    assert any("LLM indisponible" in reason for reason in result.reasons)
    assert any(error.code == IntakeReasonCode.LLM_OUTPUT_INVALID for error in result.errors)


def test_claim_intake_llm_indisponible_sur_dossier_vide_fail_closed(tmp_path, monkeypatch):
    calls = []

    def unavailable_llm(**kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr("agents.claim_intake_agent.agent._invoke_llm_intake", unavailable_llm)
    empty_source = tmp_path / "empty-source"
    empty_source.mkdir()

    result = run("CLM-9005", empty_source, storage=_storage(tmp_path))

    assert len(calls) == 1
    assert calls[0]["file_count"] == 0
    assert result.status == IntakeStatus.ERROR
    assert result.manifest.status == IntakeStatus.ERROR
    assert result.llm_metadata is not None
    assert any(error.code == IntakeReasonCode.EMPTY_CLAIM for error in result.errors)
    assert any(error.code == IntakeReasonCode.LLM_OUTPUT_INVALID for error in result.errors)


def test_claim_intake_llm_json_invalide_error(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.claim_intake_agent.agent._invoke_llm_intake", lambda **_: None)

    result = run("CLM-9004", _source(tmp_path), storage=_storage(tmp_path))

    assert result.status == IntakeStatus.ERROR
    assert any(error.code == IntakeReasonCode.LLM_OUTPUT_INVALID for error in result.errors)
