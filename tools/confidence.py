"""Calculs de confiance pour les champs et les documents extraits.

Toutes les fonctions sont pures, déterministes et bornées dans [0.0, 1.0].
La méthode de calcul est versionnée pour l'audit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Mapping

from schemas.domain import DocumentType, OcrSource
from schemas.results import ExtractedField

CONFIDENCE_METHOD_VERSION = "confidence-v2-field-document"

# Seuils de décision demandés pour l'étape 14.
CONFIDENCE_PASS = 0.80
CONFIDENCE_NEEDS_REVIEW = 0.50
FIELD_RELIABLE_THRESHOLD = 0.80
FIELD_REVIEW_THRESHOLD = 0.50

MIN_USEFUL_CHARS = 50


@dataclass(frozen=True)
class FieldConfidence:
    """Score de confiance d'un champ individuel."""

    field_name: str
    score: float
    method: OcrSource
    ocr_confidence: float
    format_valid: bool
    has_value: bool
    competing_values: int = 0
    calculation_version: str = CONFIDENCE_METHOD_VERSION
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConfidenceBreakdown:
    """Détail du calcul documentaire pour la traçabilité."""

    ocr_raw: float
    text_density: float
    classification: float
    field_coverage: float
    format_validation: float
    extraction_method: float
    competing_values: float
    field_scores: dict[str, float]
    required_below_threshold: list[str]
    final_score: float
    calculation_version: str = CONFIDENCE_METHOD_VERSION


_EXPECTED_FIELDS: dict[DocumentType, int] = {
    DocumentType.INVOICE: 8,
    DocumentType.PRESCRIPTION: 7,
    DocumentType.CLAIM_REQUEST: 8,
    DocumentType.FHIR_BUNDLE: 3,
    DocumentType.UNKNOWN: 1,
    DocumentType.UNSUPPORTED: 0,
}

_REQUIRED_FIELDS: dict[DocumentType, tuple[str, ...]] = {
    DocumentType.INVOICE: (
        "invoice_number",
        "patient_id",
        "provider",
        "invoice_date",
        "care_date",
        "total_amount",
        "currency",
    ),
    DocumentType.PRESCRIPTION: (
        "patient_id",
        "prescription_date",
        "prescriber",
        "medications",
        "dosages",
    ),
    DocumentType.CLAIM_REQUEST: (
        "claim_number",
        "patient_id",
        "contract_number",
        "care_date",
        "requested_amount",
        "currency",
        "invoice_reference",
        "declared_provider",
    ),
}

_DOCUMENT_WEIGHTS = {
    "fields": 0.35,
    "ocr": 0.20,
    "classification": 0.20,
    "coverage": 0.15,
    "text_density": 0.10,
}


def _clamp(value: float) -> float:
    return round(min(max(value, 0.0), 1.0), 4)


def _method_factor(method: OcrSource) -> float:
    if method == OcrSource.PDF_TEXT:
        return 1.0
    if method in (OcrSource.IMAGE_OCR, OcrSource.PDF_OCR):
        return 0.75
    return 0.0


def _has_competing_values(value: str) -> int:
    """Compte les valeurs concurrentes dans une valeur JSON ou séparée."""
    if not value:
        return 0
    try:
        parsed = json.loads(value)
    except Exception:
        return 1 if " ou " in value.lower() or "|" in value else 0
    if isinstance(parsed, list):
        return max(len(parsed) - 1, 0)
    return 0


def compute_field_confidence(
    *,
    field_name: str,
    value: str | None,
    method: OcrSource,
    ocr_confidence: float,
    format_valid: bool,
    competing_values: int = 0,
) -> FieldConfidence:
    """Calcule le score d'un champ selon les règles simples de l'étape 14."""
    reasons: list[str] = []
    has_value = bool(value and str(value).strip())
    if not has_value:
        return FieldConfidence(
            field_name=field_name,
            score=0.0,
            method=method,
            ocr_confidence=_clamp(ocr_confidence),
            format_valid=format_valid,
            has_value=False,
            competing_values=competing_values,
            reasons=["valeur absente"],
        )
    if not format_valid:
        return FieldConfidence(
            field_name=field_name,
            score=0.0,
            method=method,
            ocr_confidence=_clamp(ocr_confidence),
            format_valid=False,
            has_value=True,
            competing_values=competing_values,
            reasons=["format invalide"],
        )
    if competing_values > 0:
        return FieldConfidence(
            field_name=field_name,
            score=0.50,
            method=method,
            ocr_confidence=_clamp(ocr_confidence),
            format_valid=True,
            has_value=True,
            competing_values=competing_values,
            reasons=["plusieurs valeurs concurrentes"],
        )

    if method == OcrSource.PDF_TEXT:
        score = 1.00
        reasons.append("valeur extraite directement et format validé")
    elif method in (OcrSource.IMAGE_OCR, OcrSource.PDF_OCR):
        score = min(max(ocr_confidence, 0.0), 0.75)
        reasons.append("valeur issue OCR")
    else:
        score = 0.0
        reasons.append("méthode d'extraction non supportée")

    if method != OcrSource.PDF_TEXT and ocr_confidence >= 0.90:
        score = max(score, 0.90)
        reasons.append("règle déterministe fiable avec OCR clair")

    return FieldConfidence(
        field_name=field_name,
        score=_clamp(score),
        method=method,
        ocr_confidence=_clamp(ocr_confidence),
        format_valid=True,
        has_value=True,
        competing_values=competing_values,
        reasons=reasons,
    )


def score_extracted_fields(fields: Mapping[str, ExtractedField]) -> dict[str, FieldConfidence]:
    """Calcule un score pour chaque ExtractedField."""
    scores: dict[str, FieldConfidence] = {}
    for name, field_obj in fields.items():
        method = field_obj.provenance.method if field_obj.provenance else OcrSource.UNSUPPORTED
        ocr_conf = field_obj.provenance.confidence if field_obj.provenance else field_obj.confidence
        format_valid = not field_obj.requires_review and not field_obj.warnings
        competing = _has_competing_values(field_obj.value)
        scores[name] = compute_field_confidence(
            field_name=name,
            value=field_obj.value,
            method=method,
            ocr_confidence=ocr_conf,
            format_valid=format_valid,
            competing_values=competing,
        )
    return scores


def required_fields_for(document_type: DocumentType) -> tuple[str, ...]:
    return _REQUIRED_FIELDS.get(document_type, ())


def compute_confidence(
    ocr_raw_confidence: float,
    total_chars: int,
    classification_confidence: float,
    document_type: DocumentType,
    field_count: int,
    ocr_source: OcrSource,
    *,
    field_scores: Mapping[str, FieldConfidence] | None = None,
    required_fields: tuple[str, ...] | None = None,
) -> ConfidenceBreakdown:
    """Calcule le score global d'un document."""
    if ocr_source == OcrSource.PDF_TEXT:
        ocr_raw_confidence = 1.0
    elif ocr_source == OcrSource.UNSUPPORTED:
        ocr_raw_confidence = 0.0

    ocr_raw = _clamp(ocr_raw_confidence)
    text_density = _clamp(min(total_chars / max(MIN_USEFUL_CHARS * 10, 1), 1.0))
    expected = _EXPECTED_FIELDS.get(document_type, 2)
    field_coverage = _clamp(min(field_count / expected, 1.0) if expected > 0 else 0.0)
    extraction_method = _method_factor(ocr_source)

    field_score_map = dict(field_scores or {})
    if field_score_map:
        field_average = _clamp(sum(fc.score for fc in field_score_map.values()) / len(field_score_map))
        format_validation = _clamp(
            sum(1 for fc in field_score_map.values() if fc.format_valid and fc.has_value)
            / len(field_score_map)
        )
        competing_values = _clamp(
            1.0 - min(sum(fc.competing_values for fc in field_score_map.values()) / len(field_score_map), 1.0)
        )
    else:
        field_average = field_coverage
        format_validation = 1.0 if field_count > 0 else 0.0
        competing_values = 1.0

    required = required_fields if required_fields is not None else required_fields_for(document_type)
    required_below = [
        name for name in required
        if field_score_map and (name not in field_score_map or field_score_map[name].score < FIELD_RELIABLE_THRESHOLD)
    ]

    raw_score = (
        _DOCUMENT_WEIGHTS["fields"] * field_average
        + _DOCUMENT_WEIGHTS["ocr"] * ocr_raw
        + _DOCUMENT_WEIGHTS["classification"] * _clamp(classification_confidence)
        + _DOCUMENT_WEIGHTS["coverage"] * field_coverage
        + _DOCUMENT_WEIGHTS["text_density"] * text_density
    )
    raw_score *= format_validation
    raw_score *= competing_values

    if total_chars < MIN_USEFUL_CHARS:
        raw_score *= 0.3
    if required_below:
        raw_score = min(raw_score, CONFIDENCE_PASS - 0.01)

    return ConfidenceBreakdown(
        ocr_raw=ocr_raw,
        text_density=text_density,
        classification=_clamp(classification_confidence),
        field_coverage=field_coverage,
        format_validation=format_validation,
        extraction_method=_clamp(extraction_method),
        competing_values=competing_values,
        field_scores={name: fc.score for name, fc in field_score_map.items()},
        required_below_threshold=required_below,
        final_score=_clamp(raw_score),
    )


def is_readable(score: float) -> bool:
    return score >= CONFIDENCE_NEEDS_REVIEW


def requires_human_review(score: float) -> bool:
    return score < CONFIDENCE_PASS


def human_review_reasons(
    score: float,
    breakdown: ConfidenceBreakdown,
    document_type: DocumentType,
) -> list[str]:
    reasons: list[str] = []
    if score < CONFIDENCE_NEEDS_REVIEW:
        reasons.append(
            f"Document illisible — score de confiance {score:.2f} inférieur au seuil {CONFIDENCE_NEEDS_REVIEW}."
        )
    if score < CONFIDENCE_PASS:
        if breakdown.ocr_raw < 0.50:
            reasons.append(f"Confiance OCR faible ({breakdown.ocr_raw:.2f}) — qualité de numérisation insuffisante.")
        if breakdown.text_density < 0.20:
            reasons.append("Densité de texte très faible — le document est peut-être vide ou entièrement imagé.")
        if breakdown.classification < 0.40:
            reasons.append(f"Type de document incertain ({document_type.value}) — classification ambiguë.")
        if breakdown.field_coverage < 0.30:
            reasons.append("Peu de champs structurés extraits — vérification manuelle recommandée.")
        if breakdown.format_validation < 1.0:
            reasons.append("Au moins un champ extrait présente un format non validé.")
        if breakdown.required_below_threshold:
            missing = ", ".join(breakdown.required_below_threshold)
            reasons.append(f"Champ obligatoire sous le seuil de confiance : {missing}.")
        if breakdown.competing_values < 1.0:
            reasons.append("Plusieurs valeurs concurrentes détectées pour au moins un champ.")
    return reasons
