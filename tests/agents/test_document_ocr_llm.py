from pathlib import Path

from agents.document_ocr_agent import agent as ocr_agent
from agents.document_ocr_agent.agent import run
from agents.document_ocr_agent.schemas import DocumentOcrInput, LlmOcrDecision
from schemas.domain import OcrSource, SecurityDecision, VerificationStatus
from schemas.results import DocumentPageContent, SecurityGateResult


def _input() -> DocumentOcrInput:
    return DocumentOcrInput(
        claim_id="CLM-9012",
        document_id="doc-0",
        filename="facture.pdf",
        mime_type="application/pdf",
        sha256="a" * 64,
        sanitized_path="incoming/CLM-9012/facture.pdf",
        security_decision=SecurityDecision.ALLOW,
    )


def _gate() -> SecurityGateResult:
    return SecurityGateResult(
        claim_id="CLM-9012",
        decision=SecurityDecision.ALLOW,
        reasons=["ok"],
    )


def _patch_phase_a(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "facture.pdf"
    path.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(
        ocr_agent,
        "verify_file_integrity",
        lambda *_args, **_kwargs: ocr_agent.FileVerification(abs_path=path, sha256_ok=True),
    )
    monkeypatch.setattr(
        ocr_agent,
        "extract_pages",
        lambda **_: ocr_agent.ExtractedPages(
            pages_content=[
                DocumentPageContent(
                    page_number=1,
                    text=(
                        "Facture INV-CLM-9012 total 42.00 EUR Patient ID PAT-001 "
                        "Référence complémentaire REF-42"
                    ),
                    char_count=95,
                    ocr_source=OcrSource.PDF_TEXT,
                    confidence=1.0,
                )
            ],
            full_text=(
                "Facture INV-CLM-9012 total 42.00 EUR Patient ID PAT-001 "
                "Référence complémentaire REF-42"
            ),
            ocr_source=OcrSource.PDF_TEXT,
            ocr_raw_confidence=1.0,
            extraction_error=None,
            reason_codes=[],
        ),
    )


def test_document_ocr_llm_nominal(monkeypatch, tmp_path):
    _patch_phase_a(monkeypatch, tmp_path)
    monkeypatch.setattr(
        ocr_agent,
        "_invoke_llm_ocr",
        lambda _: LlmOcrDecision(
            document_type="INVOICE",
            extracted_fields={"llm_reference": "REF-42"},
            confidence_assessment="Confiance LLM correcte.",
            reasons=["Champ complémentaire proposé."],
        ),
    )

    result = run(_input(), _gate(), storage_root=tmp_path)

    assert result.status in (VerificationStatus.PASS, VerificationStatus.NEEDS_REVIEW)
    assert result.document_type.value == "INVOICE"
    assert "llm_reference" in result.extracted_fields
    assert result.extracted_fields["llm_reference"].provenance is not None
    assert "REF-42" in result.extracted_fields["llm_reference"].provenance.source_text
    assert "Confiance LLM correcte." in result.reasons


def test_document_ocr_llm_ne_pas_inventer_champ_absent(monkeypatch, tmp_path):
    _patch_phase_a(monkeypatch, tmp_path)
    monkeypatch.setattr(
        ocr_agent,
        "_invoke_llm_ocr",
        lambda _: LlmOcrDecision(
            document_type="INVOICE",
            extracted_fields={"llm_reference": "REF-INVENTEE"},
            confidence_assessment="Suggestion non confirmée.",
        ),
    )

    result = run(_input(), _gate(), storage_root=tmp_path)

    assert "llm_reference" not in result.extracted_fields


def test_document_ocr_llm_ignore_champs_si_confiance_ocr_faible(monkeypatch, tmp_path):
    path = tmp_path / "facture.pdf"
    path.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(
        ocr_agent,
        "verify_file_integrity",
        lambda *_args, **_kwargs: ocr_agent.FileVerification(abs_path=path, sha256_ok=True),
    )
    monkeypatch.setattr(
        ocr_agent,
        "extract_pages",
        lambda **_: ocr_agent.ExtractedPages(
            pages_content=[
                DocumentPageContent(
                    page_number=1,
                    text="Facture REF-LOW",
                    char_count=15,
                    ocr_source=OcrSource.IMAGE_OCR,
                    confidence=0.30,
                )
            ],
            full_text="Facture REF-LOW",
            ocr_source=OcrSource.IMAGE_OCR,
            ocr_raw_confidence=0.30,
            extraction_error="Confiance OCR insuffisante.",
            reason_codes=[],
        ),
    )
    monkeypatch.setattr(
        ocr_agent,
        "_invoke_llm_ocr",
        lambda _: LlmOcrDecision(
            document_type="INVOICE",
            extracted_fields={"llm_reference": "REF-LOW"},
            confidence_assessment="Suggestion faible confiance.",
        ),
    )

    result = run(_input(), _gate(), storage_root=tmp_path)

    assert result.status in (VerificationStatus.NEEDS_REVIEW, VerificationStatus.FAIL)
    assert "llm_reference" not in result.extracted_fields


def test_document_ocr_llm_indisponible_fallback_deterministe(monkeypatch, tmp_path):
    _patch_phase_a(monkeypatch, tmp_path)
    monkeypatch.setattr(ocr_agent, "_invoke_llm_ocr", lambda _: None)

    result = run(_input(), _gate(), storage_root=tmp_path)

    assert result.document_type.value == "INVOICE"
    assert any("LLM indisponible" in reason for reason in result.reasons)


def test_document_ocr_llm_json_invalide_fallback_deterministe(monkeypatch, tmp_path):
    _patch_phase_a(monkeypatch, tmp_path)
    monkeypatch.setattr(ocr_agent, "_invoke_llm_ocr", lambda _: None)

    result = run(_input(), _gate(), storage_root=tmp_path)

    assert result.status in (VerificationStatus.PASS, VerificationStatus.NEEDS_REVIEW)
