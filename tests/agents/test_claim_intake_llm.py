from pathlib import Path

from agents.claim_intake_agent.agent import run
from agents.claim_intake_agent.schemas import LlmIntakeDecision
from config.settings import Settings
from schemas.domain import IntakeStatus
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


def test_claim_intake_llm_indisponible_error(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.claim_intake_agent.agent._invoke_llm_intake", lambda **_: None)

    result = run("CLM-9003", _source(tmp_path), storage=_storage(tmp_path))

    assert result.status == IntakeStatus.ERROR
    assert result.manifest.status == IntakeStatus.ERROR
    assert any("LLM indisponible" in reason for reason in result.reasons)


def test_claim_intake_llm_json_invalide_error(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.claim_intake_agent.agent._invoke_llm_intake", lambda **_: None)

    result = run("CLM-9004", _source(tmp_path), storage=_storage(tmp_path))

    assert result.status == IntakeStatus.ERROR
