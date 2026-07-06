"""Outils @tool du clinical_consistency_agent — wrappers read-only sur tools/date_checks.py.

Même patron que ``agents/medical_coding_agent/tools.py`` : un wrapper fin,
sans logique propre, autour d'un module ``tools/`` pur et déjà testé.
N'effectue aucune E/S, ne mute jamais d'état, ne prend aucune décision
métier — retourne uniquement des signaux structurés (voir
``schemas.results.ClinicalSignal``) que l'agent ReAct peut consulter pour
étayer son contexte explicatif, jamais pour changer le statut déterministe.
"""
from __future__ import annotations

from langchain_core.tools import tool

from tools.date_checks import run_date_checks


@tool
def verifier_chronologie(
    prescription_date: str | None,
    care_date: str | None,
    coded_count: int | None,
) -> list[dict]:
    """Vérifie la chronologie ordonnance/soin et l'absence d'acte codifié.

    Args:
        prescription_date: Date brute de l'ordonnance (ISO 8601 ou DD/MM/YYYY),
            ``None`` si absente des documents.
        care_date: Date brute du soin/consultation facturé, ``None`` si absente.
        coded_count: Nombre d'actes déjà résolus par la codification médicale,
            ``None`` si la codification n'est pas encore disponible.

    Returns:
        Liste de signaux structurés (``model_dump(mode="json")`` de
        ``ClinicalSignal``) : ``IMPOSSIBLE_DATE`` (date illisible/ambiguë),
        ``PRESCRIPTION_BEFORE_CARE``/``PRESCRIPTION_TOO_FAR_AFTER_CARE``
        (chronologie), ``MISSING_PROCEDURE_EVIDENCE`` (acte absent). Liste
        vide si aucune anomalie détectée.
    """
    signals = run_date_checks(
        prescription_date_raw=prescription_date,
        care_date_raw=care_date,
        coded_count=coded_count,
    )
    return [signal.model_dump(mode="json") for signal in signals]
