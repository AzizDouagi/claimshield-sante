"""Outils @tool du fraud_detection_agent — wrappers read-only sur services/duplicate_index.py.

Même patron que ``agents/clinical_consistency_agent/tools.py`` : un wrapper
fin, sans logique propre, autour d'un module déjà testé. Ne prend aucune
décision métier — retourne uniquement des scores structurés (voir
``services.duplicate_index.DuplicateCheckResult``), jamais un verdict de
fraude, que l'agent ReAct peut consulter pour étayer sa justification,
jamais pour changer le score de risque.
"""
from __future__ import annotations

from datetime import date as _date
from decimal import Decimal, InvalidOperation

from langchain_core.tools import tool

from services.duplicate_index import ClaimFingerprint, DuplicateIndex

_DEFAULT_DUPLICATE_INDEX = DuplicateIndex()
"""Index de doublons partagé par défaut — visible et injectable, jamais un
singleton caché (même patron documenté que ``graph.workflow._default_orchestrator``).
``agent.py`` (Phase A) et ce module (outil ReAct Phase B) partagent cette
même instance par défaut afin de voir le même historique en production ;
``agent.run(duplicate_index=...)`` permet d'injecter une instance différente
(tests) — locale à cet appel, sans effet sur cet outil."""


@tool
def verifier_doublon(
    case_id: str,
    document_hash: str,
    patient_pseudonym: str,
    amount: str,
    service_date: str | None,
    description: str,
) -> dict:
    """Vérifie si un dossier correspond à un doublon exact ou un
    quasi-doublon déjà indexé — jamais un verdict de fraude, uniquement des
    scores de similarité structurels.

    Args:
        case_id: identifiant du dossier à vérifier.
        document_hash: SHA-256 hexadécimal du document facturé (64 caractères).
        patient_pseudonym: pseudonyme patient (PAT-…) — jamais l'identifiant réel.
        amount: montant demandé, chaîne décimale (ex. "120.50").
        service_date: date de soins ISO 8601 (YYYY-MM-DD), ou None si absente.
        description: référence courte déjà minimisée (ex. référence facture masquée).

    Returns:
        Dict JSON-sérialisable (``DuplicateCheckResult.model_dump(mode="json")``) :
        ``matches`` (liste de rapprochements, potentiellement vide),
        ``policy_version``.
    """
    try:
        parsed_amount = Decimal(amount)
    except (InvalidOperation, TypeError, ValueError):
        parsed_amount = Decimal("0")
    parsed_date = _date.fromisoformat(service_date) if service_date else None
    try:
        fingerprint = ClaimFingerprint(
            case_id=case_id,
            document_hash=document_hash,
            patient_pseudonym=patient_pseudonym,
            amount=parsed_amount,
            service_date=parsed_date,
            description=description,
        )
    except Exception:
        return {"case_id": case_id, "policy_version": _DEFAULT_DUPLICATE_INDEX.policy.version, "matches": []}
    result = _DEFAULT_DUPLICATE_INDEX.check(fingerprint)
    return result.model_dump(mode="json")
