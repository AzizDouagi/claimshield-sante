"""Outils @tool du document_ocr_agent — classification/extraction read-only."""
from __future__ import annotations

from langchain_core.tools import tool

from schemas.domain import DocumentType, OcrSource
from security.policies import DEFAULT_POLICY
from security.scanners import scan_text_security
from tools.document_classifier import classify_document as _classify_document
from tools.document_parser import parse_document


@tool
def classifier_document(texte: str, nom_fichier: str, mime: str) -> dict:
    """Classifie le type de document (INVOICE, PRESCRIPTION, CLAIM_REQUEST, etc.)."""
    result = _classify_document(texte, filename=nom_fichier, mime_type=mime)
    return {
        "document_type": result.document_type.value,
        "confidence": result.confidence,
        "is_ambiguous": result.is_ambiguous,
        "scores": result.scores,
        "classification_source": result.classification_source,
        "rules_version": result.rules_version,
    }


@tool
def extraire_champs(texte: str, type_doc: str) -> dict:
    """Extrait les champs essentiels du texte OCR selon le type documentaire."""
    try:
        document_type = DocumentType(type_doc)
    except ValueError:
        document_type = DocumentType.UNKNOWN
    parsed = parse_document(
        text=texte,
        document_type=document_type,
        page_number=None,
        ocr_source=OcrSource.PDF_TEXT,
        base_confidence=0.80,
        filename="llm_context",
        sha256="",
    )
    return {
        name: field.normalized_value or field.value
        for name, field in parsed.fields.items()
    }


@tool
def scanner_injection(texte: str) -> dict:
    """Détecte les tentatives d'injection de prompt dans le texte extrait."""
    result = scan_text_security(texte[: DEFAULT_POLICY.max_text_length], DEFAULT_POLICY)
    return {
        "detected": result.detected,
        "severity": result.severity,
        "triggers": result.triggers,
        "categories": [finding.category for finding in result.findings],
    }
