"""Classification déterministe du type de document médical.

Le nom du fichier est un indice pondéré, jamais une preuve unique. Le texte
extrait reste le signal principal pour les types documentaires métier.

Aucun appel LLM. Le texte est traité comme une donnée opaque.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from schemas.domain import DocumentType

CLASSIFIER_RULES_VERSION = "document-classifier-rules-v1"

CLASSIFICATION_SOURCE_FILENAME = "filename"
CLASSIFICATION_SOURCE_KEYWORDS = "keywords"
CLASSIFICATION_SOURCE_MIME = "mime"
CLASSIFICATION_SOURCE_COMBINED = "combined"
CLASSIFICATION_SOURCE_UNKNOWN = "unknown"

# Un nom de fichier seul ne doit pas franchir ce seuil.
_MIN_SCORE_THRESHOLD = 3.0
_AMBIGUITY_RATIO = 1.35
_AMBIGUITY_MARGIN = 1.0
_FILENAME_HINT_WEIGHT = 1.6
_MIME_HINT_WEIGHT = 1.0
_FHIR_MIME_HINT_WEIGHT = 2.0


@dataclass(frozen=True)
class RuleMatch:
    rule_id: str
    weight: float
    count: int
    source: str


@dataclass(frozen=True)
class ClassificationResult:
    document_type: DocumentType
    confidence: float
    scores: dict[str, float]
    is_ambiguous: bool
    classification_source: str
    rules_version: str = CLASSIFIER_RULES_VERSION
    matched_rules: dict[str, list[str]] = field(default_factory=dict)


_FILENAME_RULES: dict[DocumentType, list[tuple[str, re.Pattern[str], float]]] = {
    DocumentType.INVOICE: [
        ("filename.invoice.fr", re.compile(r"(?<![a-z0-9])facture(?![a-z0-9])", re.IGNORECASE), _FILENAME_HINT_WEIGHT),
        ("filename.invoice.en", re.compile(r"(?<![a-z0-9])invoice(?![a-z0-9])", re.IGNORECASE), _FILENAME_HINT_WEIGHT),
        (
            "filename.invoice.medical",
            re.compile(r"(?<![a-z0-9])medical.{0,5}invoice(?![a-z0-9])", re.IGNORECASE),
            _FILENAME_HINT_WEIGHT,
        ),
    ],
    DocumentType.PRESCRIPTION: [
        (
            "filename.prescription.fr",
            re.compile(r"(?<![a-z0-9])ordonnance(?![a-z0-9])", re.IGNORECASE),
            _FILENAME_HINT_WEIGHT,
        ),
        (
            "filename.prescription.en",
            re.compile(r"(?<![a-z0-9])prescription(?![a-z0-9])", re.IGNORECASE),
            _FILENAME_HINT_WEIGHT,
        ),
    ],
    DocumentType.CLAIM_REQUEST: [
        (
            "filename.claim.fr",
            re.compile(r"(?<![a-z0-9])demande.{0,5}remboursement(?![a-z0-9])", re.IGNORECASE),
            _FILENAME_HINT_WEIGHT,
        ),
        (
            "filename.claim.en",
            re.compile(r"(?<![a-z0-9])claim.{0,5}request(?![a-z0-9])", re.IGNORECASE),
            _FILENAME_HINT_WEIGHT,
        ),
        (
            "filename.claim.reimbursement",
            re.compile(r"(?<![a-z0-9])reimbursement(?![a-z0-9])", re.IGNORECASE),
            _FILENAME_HINT_WEIGHT,
        ),
    ],
    DocumentType.FHIR_BUNDLE: [
        ("filename.fhir.bundle", re.compile(r"(?<![a-z0-9])fhir.{0,5}bundle(?![a-z0-9])", re.IGNORECASE), 1.5),
        ("filename.fhir", re.compile(r"(?<![a-z0-9])fhir(?![a-z0-9])", re.IGNORECASE), 1.0),
    ],
}

_MIME_HINTS: dict[str, tuple[DocumentType, float, str]] = {
    "application/json": (DocumentType.FHIR_BUNDLE, _FHIR_MIME_HINT_WEIGHT, "mime.fhir.json"),
    "text/json": (DocumentType.FHIR_BUNDLE, _MIME_HINT_WEIGHT, "mime.fhir.text_json"),
}

_KEYWORD_RULES: dict[DocumentType, list[tuple[str, str, float]]] = {
    DocumentType.INVOICE: [
        ("invoice.fr.facture", r"\bfacture\b", 2.5),
        ("invoice.en.invoice", r"\binvoice\b", 2.5),
        ("invoice.total", r"\btotal\b", 1.2),
        ("invoice.amount", r"\bamount\b|\bmontant\b", 1.2),
        ("invoice.provider", r"\bprovider\b|\bprestataire\b|\bfournisseur\b", 1.2),
        ("invoice.number", r"\binvoice\s+number\b|n[°o.]\s*(?:de\s+)?facture", 2.0),
        ("invoice.amount.total", r"montant\s+(?:total|factur[eé])|total\s+(?:factur[eé]|ttc|ht)", 1.8),
        ("invoice.identifier", r"\binv[-_ ][a-z0-9-]+\b", 1.5),
        ("invoice.services", r"actes?\s+m[eé]dicaux?|honoraires|prestation", 1.0),
    ],
    DocumentType.PRESCRIPTION: [
        ("prescription.fr.ordonnance", r"\bordonnance\b", 2.5),
        ("prescription.en.prescription", r"\bprescription\b", 2.5),
        ("prescription.medication", r"\bmedication\b|m[eé]dicament", 1.8),
        ("prescription.dosage", r"\bdosage\b|\bposologie\b|\bmg\b|\bml\b", 1.8),
        ("prescription.prescriber", r"\bprescriber\b|m[eé]decin\s+prescripteur", 1.8),
        ("prescription.rx", r"\brx[-_ ]?[a-z0-9-]*\b|n[°o.]\s*ordonnance", 1.5),
        ("prescription.forms", r"comprim[eé]s?|g[eé]lules?|ampoules?|solution|sirop", 1.0),
    ],
    DocumentType.CLAIM_REQUEST: [
        ("claim.claim", r"\bclaim\b", 1.8),
        ("claim.reimbursement", r"\breimbursement\b|\bremboursement\b", 2.0),
        ("claim.requested_amount", r"\brequested\s+amount\b|montant\s+demand[eé]", 2.0),
        ("claim.policy", r"\bpolicy\b|\bpolice\b|assurance", 1.4),
        ("claim.number", r"\bclaim\s+number\b|\bclm[-_ ]?\d+\b", 1.6),
        ("claim.fr.request", r"demande\s+de\s+remboursement", 2.8),
        ("claim.coverage", r"prise\s+en\s+charge|taux\s+de\s+couverture", 1.8),
        ("claim.parts", r"part\s+(?:assureur|patient|mutuelle)", 1.5),
    ],
    DocumentType.FHIR_BUNDLE: [
        ("fhir.bundle", r'"resourceType"\s*:\s*"Bundle"', 5.0),
        ("fhir.patient", r'"resourceType"\s*:\s*"Patient"', 2.0),
        ("fhir.claim", r'"resourceType"\s*:\s*"Claim"', 2.0),
        ("fhir.entry", r'"entry"\s*:', 1.5),
        ("fhir.keyword", r"\bfhir\b", 1.5),
    ],
}

_COMPILED_KEYWORDS: dict[DocumentType, list[tuple[str, re.Pattern[str], float]]] = {
    doc_type: [(rule_id, re.compile(pattern, re.IGNORECASE), weight) for rule_id, pattern, weight in rules]
    for doc_type, rules in _KEYWORD_RULES.items()
}


def _empty_scores() -> dict[DocumentType, float]:
    return {doc_type: 0.0 for doc_type in DocumentType}


def _format_scores(scores: dict[DocumentType, float]) -> dict[str, float]:
    return {doc_type.value: round(score, 3) for doc_type, score in scores.items()}


def _record_match(matches: dict[DocumentType, list[RuleMatch]], doc_type: DocumentType, match: RuleMatch) -> None:
    matches.setdefault(doc_type, []).append(match)


def _source_from_matches(matches: list[RuleMatch]) -> str:
    sources = {match.source for match in matches}
    if not sources:
        return CLASSIFICATION_SOURCE_UNKNOWN
    if len(sources) > 1:
        return CLASSIFICATION_SOURCE_COMBINED
    return next(iter(sources))


def classify_by_filename(filename: str) -> tuple[DocumentType, float] | None:
    """Retourne seulement l'indice filename le plus fort.

    Cette fonction reste disponible pour les tests bas niveau, mais le pipeline
    principal ne l'utilise jamais comme preuve unique.
    """
    stem = Path(filename).stem
    best: tuple[DocumentType, float] | None = None
    for doc_type, rules in _FILENAME_RULES.items():
        for _, pattern, weight in rules:
            if pattern.search(stem) and (best is None or weight > best[1]):
                best = (doc_type, weight)
    return best


def classify_by_mime(mime_type: str | None) -> tuple[DocumentType, float] | None:
    """Retourne un indice MIME, ou None si non concluant."""
    if not mime_type:
        return None
    hint = _MIME_HINTS.get(mime_type.lower())
    if not hint:
        return None
    doc_type, weight, _ = hint
    return doc_type, weight


def _score_filename(filename: str | None, scores: dict[DocumentType, float], matches: dict[DocumentType, list[RuleMatch]]) -> None:
    if not filename:
        return
    stem = Path(filename).stem
    for doc_type, rules in _FILENAME_RULES.items():
        for rule_id, pattern, weight in rules:
            if pattern.search(stem):
                scores[doc_type] += weight
                _record_match(matches, doc_type, RuleMatch(rule_id, weight, 1, CLASSIFICATION_SOURCE_FILENAME))


def _score_mime(mime_type: str | None, scores: dict[DocumentType, float], matches: dict[DocumentType, list[RuleMatch]]) -> None:
    if not mime_type:
        return
    hint = _MIME_HINTS.get(mime_type.lower())
    if not hint:
        return
    doc_type, weight, rule_id = hint
    scores[doc_type] += weight
    _record_match(matches, doc_type, RuleMatch(rule_id, weight, 1, CLASSIFICATION_SOURCE_MIME))


def _score_keywords(text: str, scores: dict[DocumentType, float], matches: dict[DocumentType, list[RuleMatch]]) -> None:
    if not text or not text.strip():
        return
    for doc_type, rules in _COMPILED_KEYWORDS.items():
        for rule_id, pattern, weight in rules:
            count = len(pattern.findall(text))
            if count:
                scores[doc_type] += count * weight
                _record_match(
                    matches,
                    doc_type,
                    RuleMatch(rule_id, weight, count, CLASSIFICATION_SOURCE_KEYWORDS),
                )


def _unknown(scores: dict[DocumentType, float], matches: dict[DocumentType, list[RuleMatch]]) -> ClassificationResult:
    return ClassificationResult(
        document_type=DocumentType.UNKNOWN,
        confidence=0.0,
        scores=_format_scores(scores),
        is_ambiguous=False,
        classification_source=CLASSIFICATION_SOURCE_UNKNOWN,
        matched_rules={dt.value: [m.rule_id for m in ms] for dt, ms in matches.items()},
    )


def classify_document(
    text: str,
    *,
    filename: str | None = None,
    mime_type: str | None = None,
) -> ClassificationResult:
    """Classifie un document médical par règles déterministes versionnées."""
    scores = _empty_scores()
    matches: dict[DocumentType, list[RuleMatch]] = {}

    _score_filename(filename, scores, matches)
    _score_mime(mime_type, scores, matches)
    _score_keywords(text, scores, matches)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_type, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    if best_score < _MIN_SCORE_THRESHOLD:
        return _unknown(scores, matches)

    if best_type != DocumentType.FHIR_BUNDLE:
        best_sources = {match.source for match in matches.get(best_type, [])}
        if CLASSIFICATION_SOURCE_KEYWORDS not in best_sources:
            return _unknown(scores, matches)

    is_ambiguous = second_score > 0 and (
        (best_score / second_score) < _AMBIGUITY_RATIO
        or (best_score - second_score) < _AMBIGUITY_MARGIN
    )

    total_score = sum(scores.values())
    confidence = best_score / total_score if total_score else 0.0
    if is_ambiguous:
        confidence *= 0.65

    best_matches = matches.get(best_type, [])
    return ClassificationResult(
        document_type=best_type,
        confidence=round(min(confidence, 1.0), 3),
        scores=_format_scores(scores),
        is_ambiguous=is_ambiguous,
        classification_source=_source_from_matches(best_matches),
        matched_rules={dt.value: [m.rule_id for m in ms] for dt, ms in matches.items()},
    )
