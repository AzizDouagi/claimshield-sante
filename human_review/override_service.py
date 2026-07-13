"""Service de validation + persistance des overrides humains (V2) —
human_review/override_service.py.

Combine validation (`OverrideRequest`) et persistance
(`services.override_store.OverrideStore`) en un seul point d'entrée — même
patron que `human_review.service.validate_and_audit_human_decision` (V1,
non modifié, §0 du plan), mais la destination est `OverrideStore`, jamais
`ClaimStateV2.audit_trail` : un override n'entre jamais dans le state du
graphe (décision AZIZ, plan V2 §0).

`REOPEN` ne déclenche jamais lui-même une reprise du graphe — ce service ne
fait que valider et journaliser l'intention humaine. C'est à l'appelant
(API v2, Phase V2-9) de lancer une **nouvelle invocation complète** du
graphe V2 depuis START avec les documents mis à jour si `action == REOPEN`
— jamais une reprise partielle d'un thread déjà terminé (le graphe V2 ne
dispose d'aucun mécanisme de reprise, cohérent avec l'absence
d'`interrupt()`).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from human_review.override_models import OverrideRequest
from schemas.results import StructuredError
from services.override_store import OverrideRecord, OverrideStore

__all__ = ["OverrideValidationError", "validate_and_record_override", "validate_override_request"]


class OverrideValidationError(ValueError):
    """Erreur de validation d'une requête d'override — porte une liste de
    `StructuredError`, jamais la valeur brute fautive (même patron que
    `orchestrator.orchestrator.AgentResultValidationError`)."""

    def __init__(self, errors: list[StructuredError]) -> None:
        super().__init__("; ".join(e.message for e in errors) or "Requête d'override invalide")
        self.errors = errors


def _sanitized_validation_error_fields(exc: ValidationError) -> list[StructuredError]:
    """Ne journalise jamais `err['input']` ni `str(exc)` — une justification
    peut contenir un commentaire libre potentiellement sensible, seuls les
    chemins de champs Pydantic (`err['loc']`) apparaissent dans le message."""
    fields: list[StructuredError] = []
    for err in exc.errors():
        field = ".".join(str(part) for part in err["loc"])
        fields.append(
            StructuredError(
                code="OVERRIDE_REQUEST_INVALID",
                message=f"Champ invalide : {field or '(racine)'}",
                field=field or None,
            )
        )
    return fields


def validate_override_request(raw: Any) -> OverrideRequest:
    """Valide une requête brute d'override — jamais un contenu non
    structuré accepté tel quel."""
    if not isinstance(raw, Mapping):
        raise OverrideValidationError(
            [
                StructuredError(
                    code="OVERRIDE_REQUEST_UNSTRUCTURED",
                    message="La requête d'override doit être un objet structuré.",
                    field=None,
                )
            ]
        )
    try:
        return OverrideRequest.model_validate(dict(raw))
    except ValidationError as exc:
        raise OverrideValidationError(_sanitized_validation_error_fields(exc)) from None


def validate_and_record_override(
    raw: Mapping[str, Any],
    *,
    store: OverrideStore,
    original_decision: str | None = None,
) -> OverrideRecord:
    """Valide puis persiste un override — point d'entrée unique recommandé.

    `original_decision` (optionnel) est purement informatif — la valeur de
    `ClaimDecisionV2` observée au moment de l'override, jamais une autorité
    sur la décision d'origine (`decision_result` reste immuable, voir
    `services.override_store.OverrideRecord`).
    """
    request = validate_override_request(raw)
    record = OverrideRecord(
        case_id=request.case_id,
        actor=request.actor,
        action=request.action,
        justification=request.justification,
        original_decision=original_decision,
    )
    store.record_override(record)
    return record
