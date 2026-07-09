"""Tests unitaires purs sur ``ui/uploads.py`` — objets factices imitant la
forme de ``chainlit.types.AskFileResponse`` (``.name``/``.path``), jamais un
import ``chainlit`` réel."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from config.settings import get_settings
from ui.uploads import stage_uploaded_files


@dataclass
class _FakeUploadedFile:
    name: str
    path: str | None = None
    content: bytes | str | None = None


@pytest.fixture(autouse=True)
def _isolated_upload_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "claimshield_ui_upload_dir", tmp_path)
    yield


class TestStageUploadedFilesFromPath:
    def test_copies_file_by_path(self, tmp_path):
        source = tmp_path / "source.pdf"
        source.write_bytes(b"%PDF-1.4 fake content")
        f = _FakeUploadedFile(name="document.pdf", path=str(source))

        result_dir = stage_uploaded_files("CLM-0001", [f])

        staged = list(Path(result_dir).iterdir())
        assert len(staged) == 1
        assert staged[0].name == "document.pdf"
        assert staged[0].read_bytes() == b"%PDF-1.4 fake content"

    def test_result_dir_ends_with_case_id_input(self, tmp_path):
        source = tmp_path / "s.json"
        source.write_text("{}")
        f = _FakeUploadedFile(name="bundle.json", path=str(source))

        result_dir = stage_uploaded_files("CLM-0002", [f])

        assert Path(result_dir) == tmp_path / "CLM-0002" / "input"


class TestStageUploadedFilesFromContent:
    def test_writes_bytes_content_when_no_path(self):
        f = _FakeUploadedFile(name="raw.pdf", path=None, content=b"raw bytes")

        result_dir = stage_uploaded_files("CLM-0003", [f])

        dest = Path(result_dir) / "raw.pdf"
        assert dest.read_bytes() == b"raw bytes"

    def test_no_path_no_content_raises(self):
        f = _FakeUploadedFile(name="empty.pdf", path=None, content=None)

        with pytest.raises(ValueError, match="empty.pdf"):
            stage_uploaded_files("CLM-0004", [f])
