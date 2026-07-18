"""Tests de services/storage.py::commit_file — plan de remédiation « rejouabilité
des dossiers » (phase 1).

`services/storage.py` est partagé V1/V2 (utilisé par `claim_intake_agent` V1 et
`intake_safety_agent` V2) — `expected_sha256` est un paramètre optionnel additif :
son absence (défaut `None`) doit reproduire exactement le comportement historique
(non-régression V1 explicite ci-dessous).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import Settings
from schemas.domain import FileStatus
from services.storage import StorageError, StorageService


def _make_storage(tmp_path: Path) -> StorageService:
    settings = Settings(  # type: ignore[call-arg]
        CLAIMSHIELD_STORAGE_DIR=str(tmp_path / "storage"),
        CLAIMSHIELD_QUARANTINE_DIR=str(tmp_path / "storage" / "quarantine"),
    )
    svc = StorageService(settings=settings)
    svc.ensure_dirs()
    return svc


def _stage(svc: StorageService, case_id: str, name: str, content: bytes) -> Path:
    temp_path, _ = svc.stage_file(case_id=case_id, original_name=name, source=content)
    return temp_path


class TestExpectedSha256AbsentIsUnchangedBehaviorV1:
    """Non-régression V1 : sans `expected_sha256` (défaut), une collision de
    destination lève toujours `NO_OVERWRITE`, quel que soit le contenu —
    comportement historique, jamais modifié pour les appelants qui ne
    passent pas ce paramètre (V1 `claim_intake_agent`)."""

    def test_no_expected_sha256_raises_on_any_existing_destination(self, tmp_path):
        svc = _make_storage(tmp_path)
        temp_path = _stage(svc, "CLM-9001", "facture.pdf", b"contenu-a")
        svc.commit_file(temp_path=temp_path, case_id="CLM-9001", physical_name="doc.pdf", status=FileStatus.ACCEPTED)

        temp_path2 = _stage(svc, "CLM-9001", "facture.pdf", b"contenu-a")  # contenu identique
        with pytest.raises(StorageError) as exc_info:
            svc.commit_file(
                temp_path=temp_path2, case_id="CLM-9001", physical_name="doc.pdf", status=FileStatus.ACCEPTED
            )
        assert exc_info.value.structured.code == "NO_OVERWRITE"


class TestIdempotentReplayWithExpectedSha256:
    def test_identical_content_is_a_silent_no_op(self, tmp_path):
        svc = _make_storage(tmp_path)
        temp_path = _stage(svc, "CLM-9002", "facture.pdf", b"contenu-identique")
        dest1 = svc.commit_file(
            temp_path=temp_path,
            case_id="CLM-9002",
            physical_name="doc.pdf",
            status=FileStatus.ACCEPTED,
            expected_sha256=None,
        )

        from tools.file_inspection import compute_sha256

        expected_hash = compute_sha256(dest1)

        temp_path2 = _stage(svc, "CLM-9002", "facture.pdf", b"contenu-identique")
        dest2 = svc.commit_file(
            temp_path=temp_path2,
            case_id="CLM-9002",
            physical_name="doc.pdf",
            status=FileStatus.ACCEPTED,
            expected_sha256=expected_hash,
        )
        assert dest2 == dest1
        assert not temp_path2.exists()  # fichier temporaire superflu supprimé
        assert compute_sha256(dest1) == expected_hash  # contenu original jamais altéré

    def test_different_content_still_raises_no_overwrite(self, tmp_path):
        svc = _make_storage(tmp_path)
        temp_path = _stage(svc, "CLM-9003", "facture.pdf", b"contenu-original")
        svc.commit_file(
            temp_path=temp_path,
            case_id="CLM-9003",
            physical_name="doc.pdf",
            status=FileStatus.ACCEPTED,
            expected_sha256=None,
        )

        temp_path2 = _stage(svc, "CLM-9003", "facture.pdf", b"contenu-different")
        with pytest.raises(StorageError) as exc_info:
            svc.commit_file(
                temp_path=temp_path2,
                case_id="CLM-9003",
                physical_name="doc.pdf",
                status=FileStatus.ACCEPTED,
                expected_sha256="0" * 64,  # hash attendu ne correspond ni à l'ancien ni au nouveau contenu
            )
        assert exc_info.value.structured.code == "NO_OVERWRITE"
        assert not temp_path2.exists()
