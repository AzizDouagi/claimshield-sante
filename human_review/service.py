"""Service HITL — prépare le payload humain et valide les décisions humaines.

Distinct de ``graph/technical_nodes.py::node_await_human_review``
(interruption LangGraph, payload ad hoc pour ``interrupt()``) : ce module ne
dépend pas de LangGraph — il expose un service réutilisable (ex. par une
future API) construit sur le contrat Pydantic de ``human_review/models.py``
(``HumanDecision``/``ReviewAction``).

Trois responsabilités strictement séparées :
  1. ``build_human_review_payload`` — prépare un payload minimal pour
     l'humain (résumé, preuves, options) à partir du ``ClaimState`` — jamais
     de document brut, de texte OCR complet, de secret ou de prompt complet.
  2. ``validate_human_decision`` — valide une décision humaine brute contre
     ``HumanDecision`` et lève des erreurs structurées (``StructuredError``)
     si elle est invalide, jamais l'exception Pydantic brute (qui peut
     contenir la valeur fautive).
  3. ``build_human_decision_audit_event``/``validate_and_audit_human_decision``
     — construisent un événement d'audit minimal (``AuditEvent``, réutilise
     l'interface append-only existante de ``schemas.results``) pour toute
     décision humaine validée, quelle que soit l'action (APPROVE/MODIFY/
     REJECT/RETRY). Prépare l'intégration complète avec l'étape 14 (audit
     persistant) sans l'implémenter : ces fonctions construisent
     l'événement, elles ne l'ajoutent jamais elles-mêmes à
     ``state["audit_trail"]`` (à l'appelant de le faire, comme pour
     ``Orchestrator.execute_agent()``).
"""
from __future__ import annotations

import uuid
from typing import Any, Mapping, Sequence

from pydantic import Field, ValidationError

from human_review.models import HumanDecision, ReviewAction
from schemas.domain import StrictModel
from schemas.results import AuditEvent, StructuredError
from state.claim_state import ClaimState

# ── Préparation du payload humain ────────────────────────────────────────────

_EVIDENCE_RESULT_KEYS: tuple[str, ...] = (
    "intake_result",
    "security_result",
    "privacy_result",
    "ocr_result",
    "fhir_result",
    "identity_coverage_result",
    "coding_result",
    "clinical_result",
    "fraud_result",
    "review_result",
)
"""Résultats agents inspectés pour construire les preuves minimisées — mêmes
clés que ``graph.technical_nodes._EVIDENCE_RESULT_KEYS``."""


def _extract_status_marker(value: Any) -> str | None:
    """Extrait un statut/décision sous forme de chaîne courte.

    Ne lit jamais que les champs ``status``/``decision`` déjà exposés par les
    schémas Pydantic des résultats d'agents — jamais un champ métier brut,
    un texte OCR, un document ou un prompt.
    """
    for attr in ("status", "decision"):
        marker = getattr(value, attr, None)
        if marker is not None:
            return str(getattr(marker, "value", marker))
    return None


def _collect_evidence(state: ClaimState) -> dict[str, str]:
    """Construit les preuves minimisées : statut/décision par résultat d'agent."""
    evidence: dict[str, str] = {}
    for key in _EVIDENCE_RESULT_KEYS:
        marker = _extract_status_marker(state.get(key))
        if marker is not None:
            evidence[key] = marker
    return evidence


def _collect_summary(state: ClaimState) -> list[str]:
    """Retourne le résumé (motifs de revue) — alertes puis erreurs, sans doublon."""
    combined: list[str] = [*(state.get("alerts") or []), *(state.get("errors") or [])]
    summary = list(dict.fromkeys(combined))
    return summary or ["Revue humaine requise — aucun motif spécifique enregistré."]


class HumanReviewPayload(StrictModel):
    """Payload minimal transmis à l'humain — jamais de contenu brut.

    ``summary``/``evidence`` ne portent que des motifs déjà publics
    (alertes/erreurs déjà agrégées dans ``ClaimState``) et des statuts/
    décisions déjà exposés par les schémas Pydantic des résultats d'agents —
    jamais un document, un texte OCR complet, un secret ou un prompt.
    """

    case_id: str
    summary: list[str] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)
    options: list[ReviewAction] = Field(default_factory=list)


def build_human_review_payload(state: ClaimState) -> HumanReviewPayload:
    """Prépare le payload minimal transmis à l'humain : résumé, preuves, options.

    Ne renvoie jamais de document brut, de texte OCR complet, de secret ou de
    prompt complet — uniquement des motifs déjà agrégés (``alerts``/``errors``)
    et des statuts/décisions déjà publics des résultats d'agents.
    """
    return HumanReviewPayload(
        case_id=str(state.get("case_id", "INCONNU")),
        summary=_collect_summary(state),
        evidence=_collect_evidence(state),
        options=list(ReviewAction),
    )


# ── Validation de la décision humaine ────────────────────────────────────────


class HumanDecisionValidationError(ValueError):
    """Levée quand une décision humaine brute ne correspond pas à
    ``HumanDecision`` — toujours structurée (``StructuredError``, même patron
    que ``AgentResultValidationError``/``ModelRegistryError``), jamais
    accompagnée de la valeur brute fautive (potentiellement sensible)."""

    def __init__(self, errors: list[StructuredError]) -> None:
        self.errors = errors
        super().__init__("; ".join(err.message for err in errors))


def _sanitized_validation_error_fields(exc: ValidationError) -> list[str]:
    """Chemins de champs en erreur (``err['loc']``) uniquement — jamais
    ``err['input']`` (la valeur fautive) ni ``str(exc)`` (qui l'inclut par
    défaut) : une décision humaine peut contenir un commentaire libre, une
    erreur de validation ne doit jamais le faire fuiter dans un message
    structuré potentiellement journalisé."""
    return sorted(
        {".".join(str(part) for part in err["loc"]) or "<racine>" for err in exc.errors()}
    )


def validate_human_decision(raw: Mapping[str, Any]) -> HumanDecision:
    """Valide une décision humaine brute contre ``HumanDecision``.

    Rejette explicitement, sans jamais accepter une décision partielle :
      - toute valeur qui n'est pas un mapping — texte libre, liste, etc.
        (code ``HUMAN_DECISION_UNSTRUCTURED``), jamais tentée en validation
        Pydantic, catégoriquement rejetée ;
      - un mapping qui ne valide pas contre ``HumanDecision`` — action
        inconnue, justification absente, ``target_node`` manquant pour
        ``RETRY`` ou fourni hors ``RETRY``... (code ``HUMAN_DECISION_INVALID``).

    Lève ``HumanDecisionValidationError`` dans les deux cas — jamais
    silencieux, jamais la valeur brute (potentiellement sensible) dans le
    message : seuls les chemins de champs en erreur y figurent.
    """
    if not isinstance(raw, Mapping):
        raise HumanDecisionValidationError(
            [
                StructuredError(
                    code="HUMAN_DECISION_UNSTRUCTURED",
                    message=(
                        f"Décision humaine rejetée : type {type(raw).__name__!r} "
                        "inattendu — un mapping structuré est toujours requis."
                    ),
                    field="decision",
                )
            ]
        )

    try:
        return HumanDecision.model_validate(raw)
    except ValidationError as exc:
        raise HumanDecisionValidationError(
            [
                StructuredError(
                    code="HUMAN_DECISION_INVALID",
                    message=(
                        "Décision humaine invalide — champs en erreur : "
                        f"{_sanitized_validation_error_fields(exc)}."
                    ),
                    field="decision",
                )
            ]
        ) from exc


# ── Audit de la décision humaine ──────────────────────────────────────────────

_MAX_AUDITED_JUSTIFICATION_LENGTH = 500
"""Borne de sécurité supplémentaire, appliquée à la construction de
l'événement d'audit (en plus de ``HumanDecision.justification.max_length``,
1000) — un événement d'audit reste un enregistrement minimal, jamais un
second emplacement de stockage pour un texte long."""


def build_human_decision_audit_event(
    decision: HumanDecision, *, evidence_ids: Sequence[str] = ()
) -> AuditEvent:
    """Construit un événement d'audit minimal pour une décision humaine.

    Trace exactement cinq éléments, tous déjà validés ou déjà calculés en
    amont — jamais un nouveau contenu inventé ou recalculé ici :
      - ``action`` (``decision.action``, une des 4 valeurs de ``ReviewAction``) ;
      - ``justification`` (commentaire humain, tronqué à
        ``_MAX_AUDITED_JUSTIFICATION_LENGTH`` caractères) ;
      - ``actor`` (identifiant de l'intervenant — simulé dans les tests, réel
        en production) ;
      - horodatage (``decision.decided_at`` — l'instant de la décision
        elle-même, pas celui de la construction de cet événement) ;
      - ``evidence_ids`` (preuves déjà calculées par les agents amont, ex.
        ``CaseReviewerResult.evidence_ids`` — jamais un identifiant inventé
        ici, cette fonction ne fait qu'agréger ce qui lui est fourni).

    Ne contient jamais de document brut, de prompt complet ou de texte OCR
    complet : ces champs ne sont structurellement jamais lus par cette
    fonction, qui ne connaît que ``HumanDecision`` et une liste d'identifiants
    de preuve déjà minimisés.
    """
    return AuditEvent(
        event_id=str(uuid.uuid4()),
        case_id=decision.case_id,
        actor=decision.actor,
        action="human_review_decision",
        outcome=decision.action.value,
        timestamp=decision.decided_at,
        details={
            "justification": decision.justification[:_MAX_AUDITED_JUSTIFICATION_LENGTH],
            "target_node": decision.target_node or "",
            "evidence_ids": ",".join(evidence_ids),
        },
    )


def validate_and_audit_human_decision(
    raw: Mapping[str, Any], *, evidence_ids: Sequence[str] = ()
) -> tuple[HumanDecision, AuditEvent]:
    """Valide une décision humaine brute et construit son événement d'audit.

    Combine ``validate_human_decision``/``build_human_decision_audit_event``
    en un seul point d'entrée — préparé pour une future intégration complète
    (étape 14) où l'appelant ajoutera (append) l'événement retourné à
    ``state["audit_trail"]``, jamais construit ni ajouté silencieusement ici.
    Lève ``HumanDecisionValidationError`` (jamais d'audit construit) si la
    décision brute est invalide — un audit ne peut jamais porter sur une
    décision qui n'a pas été acceptée.
    """
    decision = validate_human_decision(raw)
    event = build_human_decision_audit_event(decision, evidence_ids=evidence_ids)
    return decision, event


__all__ = [
    "HumanDecisionValidationError",
    "HumanReviewPayload",
    "build_human_decision_audit_event",
    "build_human_review_payload",
    "validate_and_audit_human_decision",
    "validate_human_decision",
]
