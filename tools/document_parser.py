"""Extraction de champs structurés depuis le texte normalisé d'un document médical.

Approche déterministe par expressions régulières — aucun appel LLM.
Chaque champ extrait est accompagné de sa provenance complète (FieldProvenance).
Le texte est une donnée opaque : jamais exécuté, jamais évalué.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime

from schemas.domain import DocumentType, OcrSource
from schemas.results import EssentialFields, ExtractedField, FieldProvenance, MedicalItem, MonetaryAmount
from tools.text_normalizer import normalize_amount, normalize_date_value

# Version stable du parseur — à incrémenter si la logique d'extraction change
_PARSER_VERSION = "document-parser-v1"

# ── Patterns d'extraction par champ ──────────────────────────────────────────

_CLAIM_REF_RE = re.compile(r"\bCLM-(\d{4})\b")
_INVOICE_NUM_RE = re.compile(r"\bINV-CLM-(\d{4})\b")
_RX_NUM_RE = re.compile(r"\bRX-CLM-(\d{4})\b")
_CONTRACT_NUM_RE = re.compile(
    r"(?:contrat|contract|policy)\s*(?:n[°o.]|number|#)?\s*[:\-–]?\s*([A-Z0-9][A-Z0-9\-_/]{3,40})",
    re.IGNORECASE,
)
_DATE_RE = re.compile(r"\b(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})\b")
_AMOUNT_RE = re.compile(
    r"(?:total[^\n]{0,40}?|montant[^\n]{0,40}?|facturé[^\n]{0,40}?)"
    r"(\d{1,6}[,. ]\d{2,3})\s*(?:USD|EUR|€|\$)?",
    re.IGNORECASE,
)
_REQUESTED_AMOUNT_RE = re.compile(
    r"(?:montant\s+demand[eé]|requested\s+amount|amount\s+requested)[^\n]{0,40}?"
    r"(\d{1,6}(?:[,. ]\d{2,3})?)\s*(?:USD|EUR|€|\$)?",
    re.IGNORECASE,
)
_COVERAGE_RE = re.compile(r"\b(\d{2,3})\s*%", re.IGNORECASE)
_PATIENT_NAME_RE = re.compile(
    r"(?:patient|assuré[e]?)\s*[:\-–]\s*([A-ZÀ-Ÿa-zà-ÿ][A-Za-zÀ-Ÿà-ÿ''\- ]{3,60})",
    re.IGNORECASE,
)
_PATIENT_ID_RE = re.compile(
    r"(?:n[°o.]\s*patient|patient[_ ]?id|identifiant(?:\s+patient)?|patient\s+identifier)\s*[:\-–]?\s*"
    r"([A-Z0-9][A-Z0-9\-_/]{3,60}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)
_PAYER_RE = re.compile(
    r"(?:assureur|mutuelle|assurance|payer|cigna|blue\s+cross|axa|harmonie|malakoff)",
    re.IGNORECASE,
)
_MEDICATION_RE = re.compile(
    r"\b([A-ZÀ-Ÿ][a-zà-ÿA-ZÀ-Ÿ]{3,30})\s+(\d+\s*mg|\d+\s*ml|\d+\s*µg)\b",
)
_DOCUMENT_DATE_LABELED_RE = re.compile(
    r"(?:date\s*(?:du\s*document|d[e']?\s*(?:facturation|prescription|émission))\s*[:\-–]?\s*)"
    r"(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)
_SERVICE_DATE_LABELED_RE = re.compile(
    r"(?:date\s*(?:de\s*(?:soins|consultation|service|prestation)|du\s*service)\s*[:\-–]?\s*)"
    r"(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)
_PROVIDER_RE = re.compile(
    r"(?:prescripteur|médecin|docteur|dr\.?\s*|prestataire|provider|praticien|"
    r"clinique|hôpital|établissement)\s*[:\-–]\s*"
    r"([A-ZÀ-Ÿa-zà-ÿ][A-Za-zÀ-Ÿà-ÿ''\- ]{2,60})",
    re.IGNORECASE,
)
_CURRENCY_IN_TEXT_RE = re.compile(r"\b(USD|EUR|GBP|CAD|CHF)\b|([€$£])")
_INVOICE_REF_RE = re.compile(
    r"(?:r[eé]f[eé]rence\s+facture|invoice\s+reference|invoice\s+ref|facture)\s*[:\-–]?\s*"
    r"((?:INV-)?CLM-\d{4}|INV-[A-Z0-9\-_/]{3,40})",
    re.IGNORECASE,
)
_DECLARED_PROVIDER_RE = re.compile(
    r"(?:fournisseur\s+d[eé]clar[eé]|declared\s+provider|provider\s+declared)\s*[:\-–]\s*"
    r"([A-ZÀ-Ÿa-zà-ÿ][A-Za-zÀ-Ÿà-ÿ''\- ]{2,80})",
    re.IGNORECASE,
)
_INVOICE_LINE_RE = re.compile(
    r"(?im)^\s*(?:acte|service|prestation|procedure|ligne)\s*[:#\-–]?\s*"
    r"(.{3,120}?)(?:\s{2,}|\s+-\s+|\s+)(\d{1,6}(?:[,.]\d{2})?)\s*(USD|EUR|€|\$)?\s*$"
)
_MEDICATION_LINE_RE = re.compile(
    r"(?im)^\s*(?:m[eé]dicament|medication|rx|traitement)?\s*[:#\-–]?\s*"
    r"([A-ZÀ-Ÿ][A-Za-zÀ-Ÿà-ÿ'\-]{2,40})"
    r"(?:\s+(?P<dosage>\d+(?:[,.]\d+)?\s*(?:mg|ml|µg|mcg|g)))?"
    r"(?:[,\s;-]+(?:quantit[eé]|qty|quantity)\s*[:\-]?\s*(?P<quantity>\d+))?"
    r"(?:[,\s;-]+(?:dur[eé]e|duration)\s*[:\-]?\s*(?P<duration>\d+\s*(?:jours?|days?|semaines?|weeks?|mois|months?)))?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParseResult:
    fields: dict[str, ExtractedField]
    field_count: int
    essential_fields: EssentialFields


def _make_field(
    name: str,
    raw_value: str,
    page: int | None,
    source: OcrSource,
    confidence: float,
    filename: str = "",
    sha256: str = "",
    source_text: str = "",
    position: dict[str, int] | None = None,
) -> ExtractedField:
    """Construit un ExtractedField avec FieldProvenance complète."""
    now = datetime.now(UTC)
    provenance = FieldProvenance(
        filename=filename or "unknown",
        sha256=sha256,
        page_number=page,
        method=source,
        source_text=source_text[:200],
        position=position,
        confidence=confidence,
        parser_version=_PARSER_VERSION,
        extracted_at=now,
    )
    value = raw_value.strip()
    return ExtractedField(
        field_name=name,
        value=value,
        normalized_value=value,
        confidence=round(confidence, 3),
        provenance=provenance,
        warnings=[],
        requires_review=confidence < 0.65,
    )


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def _match_source(pattern: re.Pattern[str], text: str) -> tuple[str | None, str]:
    m = pattern.search(text)
    if not m:
        return None, ""
    return m.group(1).strip(), m.group(0)[:200]


def _first_match_full(pattern: re.Pattern[str], text: str) -> str | None:
    m = pattern.search(text)
    return m.group(0).strip() if m else None


def _source_snippet(pattern: re.Pattern[str], text: str) -> str:
    """Retourne la portion de texte matchée (≤ 200 car.) pour la provenance."""
    m = pattern.search(text)
    return m.group(0)[:200] if m else ""


def _parse_date(raw: str | None) -> date | None:
    """Convertit DD/MM/YYYY ou YYYY-MM-DD en date Python ; None si invalide."""
    if not raw:
        return None
    return normalize_date_value(raw).normalized_value


def _detect_currency(text: str) -> str:
    """Détecte la devise dans le texte (USD par défaut)."""
    m = _CURRENCY_IN_TEXT_RE.search(text)
    if not m:
        return "USD"
    code = m.group(1)
    if code:
        return code.upper()
    return {"€": "EUR", "$": "USD", "£": "GBP"}.get(m.group(2), "USD")


def _parse_monetary(raw: str | None, currency: str) -> MonetaryAmount | None:
    """Convertit une chaîne de montant en MonetaryAmount ; None si invalide."""
    normalized = normalize_amount(raw or "", default_currency=currency)
    if normalized.normalized_value is None or not normalized.currency:
        return None
    return MonetaryAmount(amount=normalized.normalized_value, currency=normalized.currency)


def _normalise_amount(raw: str | None) -> str | None:
    if not raw:
        return None
    return raw.strip().replace(" ", "").replace(",", ".")


def _json_value(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _extract_medical_items(text: str) -> list[MedicalItem]:
    """Extrait les médicaments détectés dans le texte sous forme de MedicalItem."""
    items: list[MedicalItem] = []
    seen: set[str] = set()
    for m in _MEDICATION_RE.finditer(text):
        desc = f"{m.group(1).strip()} {m.group(2).strip()}"
        if desc not in seen:
            seen.add(desc)
            items.append(MedicalItem(description=desc, quantity=1))
    return items


def _extract_invoice_lines(text: str, currency: str) -> list[dict[str, str]]:
    lines: list[dict[str, str]] = []
    seen: set[str] = set()
    for m in _INVOICE_LINE_RE.finditer(text):
        description = m.group(1).strip(" -:\t")
        amount = _normalise_amount(m.group(2)) or ""
        line_currency = (m.group(3) or currency).replace("€", "EUR").replace("$", "USD")
        key = f"{description}|{amount}|{line_currency}"
        if key in seen:
            continue
        seen.add(key)
        lines.append({
            "description": description,
            "amount": amount,
            "currency": line_currency,
            "source_text": m.group(0)[:200],
        })
    return lines


def _extract_prescription_lines(text: str) -> list[dict[str, str]]:
    lines: list[dict[str, str]] = []
    seen: set[str] = set()
    for m in _MEDICATION_LINE_RE.finditer(text):
        medication = m.group(1).strip()
        # Avoid generic labels captured as medication names.
        if medication.lower() in {"ordonnance", "prescription", "medicament", "médicament"}:
            continue
        dosage = (m.group("dosage") or "").strip()
        quantity = (m.group("quantity") or "").strip()
        duration = (m.group("duration") or "").strip()
        if not (dosage or quantity or duration):
            continue
        key = f"{medication}|{dosage}|{quantity}|{duration}"
        if key in seen:
            continue
        seen.add(key)
        lines.append({
            "medication": medication,
            "dosage": dosage,
            "quantity": quantity,
            "duration": duration,
            "source_text": m.group(0)[:200],
        })
    return lines


def parse_document(
    text: str,
    document_type: DocumentType,
    page_number: int | None,
    ocr_source: OcrSource,
    base_confidence: float,
    *,
    filename: str = "",
    sha256: str = "",
) -> ParseResult:
    """Extrait les champs structurés d'un texte de document médical.

    La confiance de chaque champ est dérivée de la confiance de base OCR
    et du contexte dans lequel le champ a été trouvé.
    Chaque champ embarque un FieldProvenance complet.
    """
    fields: dict[str, ExtractedField] = {}

    def add(name: str, value: str | None, conf_bonus: float = 0.0, *, source_text: str = "") -> None:
        if value:
            conf = min(base_confidence + conf_bonus, 1.0)
            start = text.find(source_text) if source_text else -1
            position = (
                {"start": start, "end": start + len(source_text)}
                if start >= 0 and source_text
                else None
            )
            fields[name] = _make_field(name, value, page_number, ocr_source, conf,
                                       filename=filename, sha256=sha256,
                                       source_text=source_text, position=position)

    # ── Champs communs à tous les types ───────────────────────────────────────
    claim_ref = _first_match(_CLAIM_REF_RE, text)
    add("claim_reference",
        f"CLM-{claim_ref}" if claim_ref else None,
        source_text=_source_snippet(_CLAIM_REF_RE, text))

    date_val = _first_match(_DATE_RE, text)
    add("service_date", date_val, conf_bonus=0.1,
        source_text=_source_snippet(_DATE_RE, text))

    name_val = _first_match(_PATIENT_NAME_RE, text)
    add("patient_name", name_val,
        source_text=_source_snippet(_PATIENT_NAME_RE, text))

    pid_val = _first_match(_PATIENT_ID_RE, text)
    add("patient_id", pid_val, 0.15,
        source_text=_source_snippet(_PATIENT_ID_RE, text))

    # ── Champs spécifiques par type ───────────────────────────────────────────
    if document_type == DocumentType.INVOICE:
        inv_num = _first_match(_INVOICE_NUM_RE, text)
        add("invoice_number",
            f"INV-CLM-{inv_num}" if inv_num else None,
            source_text=_source_snippet(_INVOICE_NUM_RE, text))
        provider, provider_source = _match_source(_PROVIDER_RE, text)
        add("provider", provider, 0.05, source_text=provider_source)

        invoice_date, invoice_date_source = _match_source(_DOCUMENT_DATE_LABELED_RE, text)
        add("invoice_date", invoice_date, 0.1, source_text=invoice_date_source)

        care_date, care_date_source = _match_source(_SERVICE_DATE_LABELED_RE, text)
        add("care_date", care_date, 0.1, source_text=care_date_source)

        currency = _detect_currency(text)
        currency_source = _source_snippet(_CURRENCY_IN_TEXT_RE, text)
        add("currency", currency if currency_source else None, 0.05, source_text=currency_source)

        amount = _first_match(_AMOUNT_RE, text)
        if amount:
            normalized_amount = _normalise_amount(amount)
            add("total_amount", normalized_amount,
                source_text=_source_snippet(_AMOUNT_RE, text))
            add("total_billed", normalized_amount,
                source_text=_source_snippet(_AMOUNT_RE, text))
        for idx, line in enumerate(_extract_invoice_lines(text, currency), start=1):
            add(f"invoice_line_{idx}", _json_value({
                "description": line["description"],
                "amount": line["amount"],
                "currency": line["currency"],
            }), source_text=line["source_text"])
        invoice_lines = _extract_invoice_lines(text, currency)
        if invoice_lines:
            add("invoice_lines", _json_value([
                {k: v for k, v in line.items() if k != "source_text"}
                for line in invoice_lines
            ]), source_text="\n".join(line["source_text"] for line in invoice_lines)[:200])
        coverage = _first_match(_COVERAGE_RE, text)
        if coverage:
            add("coverage_rate",
                f"0.{coverage.zfill(2)}" if len(coverage) <= 2 else f"0.{coverage}",
                source_text=_source_snippet(_COVERAGE_RE, text))
        payer_m = _PAYER_RE.search(text)
        if payer_m:
            add("payer_name", payer_m.group(0), 0.05,
                source_text=payer_m.group(0)[:200])

    elif document_type == DocumentType.PRESCRIPTION:
        rx_num = _first_match(_RX_NUM_RE, text)
        add("prescription_number",
            f"RX-CLM-{rx_num}" if rx_num else None,
            source_text=_source_snippet(_RX_NUM_RE, text))
        prescription_date, prescription_date_source = _match_source(_DOCUMENT_DATE_LABELED_RE, text)
        add("prescription_date", prescription_date, 0.1, source_text=prescription_date_source)

        prescriber, prescriber_source = _match_source(_PROVIDER_RE, text)
        add("prescriber", prescriber, 0.05, source_text=prescriber_source)

        med_m = _MEDICATION_RE.search(text)
        if med_m:
            add("medication_name", med_m.group(1), source_text=med_m.group(0)[:200])
            add("medication_dosage", med_m.group(2), source_text=med_m.group(0)[:200])
        prescription_lines = _extract_prescription_lines(text)
        medications: list[str] = []
        dosages: list[str] = []
        quantities: list[str] = []
        durations: list[str] = []
        for idx, line in enumerate(prescription_lines, start=1):
            medications.append(line["medication"])
            if line["dosage"]:
                dosages.append(line["dosage"])
            if line["quantity"]:
                quantities.append(line["quantity"])
            if line["duration"]:
                durations.append(line["duration"])
            add(f"prescription_line_{idx}", _json_value({
                "medication": line["medication"],
                "dosage": line["dosage"],
                "quantity": line["quantity"],
                "duration": line["duration"],
            }), source_text=line["source_text"])
        if medications:
            add("medications", _json_value(medications), source_text="\n".join(
                line["source_text"] for line in prescription_lines
            )[:200])
        if dosages:
            add("dosages", _json_value(dosages), source_text="\n".join(
                line["source_text"] for line in prescription_lines if line["dosage"]
            )[:200])
        if quantities:
            add("quantities", _json_value(quantities), source_text="\n".join(
                line["source_text"] for line in prescription_lines if line["quantity"]
            )[:200])
        if durations:
            add("durations", _json_value(durations), source_text="\n".join(
                line["source_text"] for line in prescription_lines if line["duration"]
            )[:200])

    elif document_type == DocumentType.CLAIM_REQUEST:
        cr_ref = _first_match(_CLAIM_REF_RE, text)
        add("claim_number",
            f"CLM-{cr_ref}" if cr_ref else None,
            source_text=_source_snippet(_CLAIM_REF_RE, text))
        add("claim_reference",
            f"CLM-{cr_ref}" if cr_ref else None,
            source_text=_source_snippet(_CLAIM_REF_RE, text))

        contract_number, contract_source = _match_source(_CONTRACT_NUM_RE, text)
        add("contract_number", contract_number, 0.05, source_text=contract_source)

        care_date, care_date_source = _match_source(_SERVICE_DATE_LABELED_RE, text)
        add("care_date", care_date, 0.1, source_text=care_date_source)

        currency = _detect_currency(text)
        currency_source = _source_snippet(_CURRENCY_IN_TEXT_RE, text)
        add("currency", currency if currency_source else None, 0.05, source_text=currency_source)

        amount = _first_match(_REQUESTED_AMOUNT_RE, text) or _first_match(_AMOUNT_RE, text)
        if amount:
            amount_source = (
                _source_snippet(_REQUESTED_AMOUNT_RE, text)
                or _source_snippet(_AMOUNT_RE, text)
            )
            add("requested_amount", _normalise_amount(amount), source_text=amount_source)
            add("amount_requested", _normalise_amount(amount), source_text=amount_source)

        invoice_ref, invoice_ref_source = _match_source(_INVOICE_REF_RE, text)
        add("invoice_reference", invoice_ref, 0.05, source_text=invoice_ref_source)

        declared_provider, declared_provider_source = _match_source(_DECLARED_PROVIDER_RE, text)
        if not declared_provider:
            declared_provider, declared_provider_source = _match_source(_PROVIDER_RE, text)
        add("declared_provider", declared_provider, 0.05, source_text=declared_provider_source)

        coverage = _first_match(_COVERAGE_RE, text)
        if coverage:
            add("coverage_rate", f"0.{coverage.zfill(2)}",
                source_text=_source_snippet(_COVERAGE_RE, text))

    # Correction ref claim (évite la double extraction avec valeur None)
    if "claim_reference" in fields and fields["claim_reference"].value.startswith("CLM-None"):
        del fields["claim_reference"]

    # ── Champs essentiels (Étape 7) ───────────────────────────────────────────
    currency = _detect_currency(text)

    patient_id_val: str | None = None
    if "patient_id" in fields:
        patient_id_val = fields["patient_id"].value or None

    doc_ref: str | None = None
    for ref_key in ("invoice_number", "prescription_number", "claim_reference"):
        if ref_key in fields:
            doc_ref = fields[ref_key].value or None
            break

    document_date = _parse_date(_first_match(_DOCUMENT_DATE_LABELED_RE, text))
    service_date = _parse_date(
        _first_match(_SERVICE_DATE_LABELED_RE, text) or _first_match(_DATE_RE, text)
    )

    provider_raw = _first_match(_PROVIDER_RE, text)
    provider_identifier_or_name = provider_raw.strip() if provider_raw else None

    total_amount = _parse_monetary(
        fields["total_billed"].value if "total_billed" in fields else None,
        currency,
    )
    requested_amount = _parse_monetary(
        fields["amount_requested"].value if "amount_requested" in fields else None,
        currency,
    )

    essential = EssentialFields(
        patient_identifier=patient_id_val,
        document_reference=doc_ref,
        document_date=document_date,
        service_date=service_date,
        provider_identifier_or_name=provider_identifier_or_name,
        total_amount=total_amount,
        requested_amount=requested_amount,
        medical_items=_extract_medical_items(text),
    )

    return ParseResult(fields=fields, field_count=len(fields), essential_fields=essential)
