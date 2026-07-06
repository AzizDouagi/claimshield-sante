"""Vérifications déterministes de chronologie de dates — tools/date_checks.py.

Compare des dates déjà extraites/normalisées (OCR) et des compteurs d'actes
déjà codifiés (``coding_result``) pour détecter des anomalies de
chronologie : ordonnance datée avant ou trop longtemps après le soin
correspondant, acte absent pour un soin daté, date impossible ou ambiguë.

Ne prend jamais de décision métier : ce module ne retourne que des
``ClinicalSignal`` structurés (``schemas.results``, même schéma que
``clinical_consistency_agent`` — jamais un nouveau schéma dupliqué), chacun
attribué à au moins une ``ClinicalEvidence`` (``evidence_id``/``source``/
``field``). Ne calcule et n'expose aucun statut PASS/NEEDS_REVIEW/FAIL —
cette interprétation reste la responsabilité de l'agent appelant (voir
``agents/clinical_consistency_agent/agent.py``), qui décide seul de la
sévérité globale du dossier à partir des signaux renvoyés ici.

Réutilise ``tools.text_normalizer.normalize_date_value`` (source unique de
vérité pour le parsing de dates OCR, déjà utilisée par
``tools/document_parser.py``) plutôt que de réimplémenter un parseur de
dates. Aucun état mutable, aucune E/S, aucun appel LLM.
"""
from __future__ import annotations

from datetime import date

from schemas.domain import SeverityLevel
from schemas.results import ClinicalEvidence, ClinicalEvidenceSource, ClinicalSignal
from tools.text_normalizer import normalize_date_value

MAX_PRESCRIPTION_AFTER_CARE_DAYS = 30
"""Tolérance (en jours) au-delà de laquelle une ordonnance datée après le
soin correspondant devient un signal — au-delà, le lien entre l'ordonnance
et le soin facturé n'est plus mécaniquement plausible."""


def parse_checked_date(
    raw: str | None,
    *,
    field_name: str,
    document_reference: str | None = None,
) -> tuple[date | None, ClinicalSignal | None]:
    """Analyse une date brute et signale toute date impossible ou ambiguë.

    Retourne ``(date_analysée, signal)`` : ``date_analysée`` est ``None`` si
    ``raw`` est absent, vide, invalide ou ambigu — dans ce dernier cas,
    ``signal`` porte un ``IMPOSSIBLE_DATE`` attribué (``evidence`` référence
    la valeur brute fautive, jamais un contenu de document complet).
    ``raw`` absent (``None``) ne produit aucun signal : l'absence d'une date
    est une autre anomalie (voir ``check_missing_procedure_evidence``),
    jamais une date « impossible ».
    """
    if raw is None or not raw.strip():
        return None, None

    normalized = normalize_date_value(raw)
    if normalized.normalized_value is not None:
        return normalized.normalized_value, None

    reason = "; ".join(normalized.errors) or "date invalide"
    signal = ClinicalSignal(
        signal_type="IMPOSSIBLE_DATE",
        description=f"Date illisible ou impossible dans le champ {field_name!r} : {reason}.",
        fields_compared=[field_name],
        evidence=[
            ClinicalEvidence(
                source=ClinicalEvidenceSource.OCR_EXTRACTION,
                field=field_name,
                document_reference=document_reference,
                value=raw[:500],
            )
        ],
        severity=SeverityLevel.CRITICAL,
    )
    return None, signal


def check_prescription_before_care(
    prescription_date: date | None,
    care_date: date | None,
    *,
    prescription_field: str = "prescription_date",
    care_field: str = "care_date",
    document_reference: str | None = None,
) -> ClinicalSignal | None:
    """Signale une ordonnance datée strictement avant le soin correspondant.

    Cliniquement invraisemblable : une ordonnance est émise à l'issue d'une
    consultation, jamais avant qu'elle n'ait eu lieu. ``None`` si une des
    deux dates est absente (rien à comparer) ou si l'ordre est cohérent.
    """
    if prescription_date is None or care_date is None:
        return None
    if prescription_date >= care_date:
        return None

    return ClinicalSignal(
        signal_type="PRESCRIPTION_BEFORE_CARE",
        description=(
            f"Ordonnance datée du {prescription_date.isoformat()}, avant le soin "
            f"du {care_date.isoformat()} — ordre chronologique invraisemblable."
        ),
        fields_compared=[prescription_field, care_field],
        evidence=[
            ClinicalEvidence(
                source=ClinicalEvidenceSource.OCR_EXTRACTION,
                field=prescription_field,
                document_reference=document_reference,
                value=prescription_date.isoformat(),
            ),
            ClinicalEvidence(
                source=ClinicalEvidenceSource.OCR_EXTRACTION,
                field=care_field,
                document_reference=document_reference,
                value=care_date.isoformat(),
            ),
        ],
        severity=SeverityLevel.CRITICAL,
    )


def check_prescription_too_far_after_care(
    prescription_date: date | None,
    care_date: date | None,
    *,
    prescription_field: str = "prescription_date",
    care_field: str = "care_date",
    document_reference: str | None = None,
    max_days: int = MAX_PRESCRIPTION_AFTER_CARE_DAYS,
) -> ClinicalSignal | None:
    """Signale une ordonnance datée trop longtemps après le soin correspondant.

    ``None`` si une des deux dates est absente, si l'ordonnance précède ou
    coïncide avec le soin (couvert par ``check_prescription_before_care``),
    ou si l'écart reste dans la tolérance ``max_days``.
    """
    if prescription_date is None or care_date is None:
        return None
    gap_days = (prescription_date - care_date).days
    if gap_days <= max_days:
        return None

    return ClinicalSignal(
        signal_type="PRESCRIPTION_TOO_FAR_AFTER_CARE",
        description=(
            f"Ordonnance datée {gap_days} jour(s) après le soin correspondant "
            f"(tolérance {max_days} jours) — lien avec le soin facturé incertain."
        ),
        fields_compared=[prescription_field, care_field],
        evidence=[
            ClinicalEvidence(
                source=ClinicalEvidenceSource.OCR_EXTRACTION,
                field=prescription_field,
                document_reference=document_reference,
                value=prescription_date.isoformat(),
            ),
            ClinicalEvidence(
                source=ClinicalEvidenceSource.OCR_EXTRACTION,
                field=care_field,
                document_reference=document_reference,
                value=care_date.isoformat(),
            ),
        ],
        severity=SeverityLevel.MEDIUM,
    )


def check_missing_procedure_evidence(
    *,
    care_date: date | None,
    coded_count: int | None,
    care_field: str = "care_date",
    document_reference: str | None = None,
) -> ClinicalSignal | None:
    """Signale un soin daté sans aucun acte codifié associé (« acte absent »).

    ``None`` si ``care_date`` est absente (rien à dater), ou si
    ``coded_count`` est ``None`` (codification non encore disponible — pas
    une preuve d'absence) ou strictement positif (au moins un acte résolu).
    """
    if care_date is None or coded_count is None or coded_count > 0:
        return None

    return ClinicalSignal(
        signal_type="MISSING_PROCEDURE_EVIDENCE",
        description=(
            f"Soin daté du {care_date.isoformat()} sans aucun acte codifié "
            "associé — preuve d'acte manquante."
        ),
        fields_compared=[care_field, "coding_result.codings"],
        evidence=[
            ClinicalEvidence(
                source=ClinicalEvidenceSource.OCR_EXTRACTION,
                field=care_field,
                document_reference=document_reference,
                value=care_date.isoformat(),
            ),
            ClinicalEvidence(
                source=ClinicalEvidenceSource.MEDICAL_CODING,
                field="codings",
                document_reference="coding_result",
                value=str(coded_count),
            ),
        ],
        severity=SeverityLevel.MEDIUM,
    )


def run_date_checks(
    *,
    prescription_date_raw: str | None,
    care_date_raw: str | None,
    coded_count: int | None = None,
    prescription_field: str = "prescription_date",
    care_field: str = "care_date",
    document_reference: str | None = None,
    max_prescription_after_care_days: int = MAX_PRESCRIPTION_AFTER_CARE_DAYS,
) -> tuple[ClinicalSignal, ...]:
    """Point d'entrée composant les contrôles de chronologie disponibles.

    Analyse d'abord les deux dates brutes (``IMPOSSIBLE_DATE`` par champ
    fautif) puis, pour les dates effectivement analysables, la chronologie
    ordonnance/soin et l'absence d'acte codifié. Une date impossible retire
    mécaniquement les contrôles qui en dépendent (rien à comparer de
    fiable) — jamais un signal construit sur une valeur invalide.
    """
    signals: list[ClinicalSignal] = []

    prescription_date, prescription_error = parse_checked_date(
        prescription_date_raw, field_name=prescription_field, document_reference=document_reference
    )
    if prescription_error is not None:
        signals.append(prescription_error)

    care_date, care_error = parse_checked_date(
        care_date_raw, field_name=care_field, document_reference=document_reference
    )
    if care_error is not None:
        signals.append(care_error)

    before = check_prescription_before_care(
        prescription_date,
        care_date,
        prescription_field=prescription_field,
        care_field=care_field,
        document_reference=document_reference,
    )
    if before is not None:
        signals.append(before)
    else:
        after = check_prescription_too_far_after_care(
            prescription_date,
            care_date,
            prescription_field=prescription_field,
            care_field=care_field,
            document_reference=document_reference,
            max_days=max_prescription_after_care_days,
        )
        if after is not None:
            signals.append(after)

    missing_procedure = check_missing_procedure_evidence(
        care_date=care_date,
        coded_count=coded_count,
        care_field=care_field,
        document_reference=document_reference,
    )
    if missing_procedure is not None:
        signals.append(missing_procedure)

    return tuple(signals)
