"""Service d'audit 100% déterministe — services/audit_service.py (V2).

Remplace `agents/audit_agent` (V1, étape 14) pour la V2 : **aucun appel
LLM**. `agents/audit_agent/agent.py` (V1) soumettait chaque événement à un
LLM de normalisation (`LlmAuditNormalizedEvent`) avant persistance — c'était
la source de latence principale identifiée dans le plan de remédiation
(jusqu'à ~110-140s par nœud avant le batching de l'option C). Ce service
supprime cette étape entièrement : la rédaction déterministe
(`tools.audit_redaction.redact_audit_payload`, déjà 100% pure côté V1) est
conservée telle quelle, mais aucune paraphrase LLM ne s'intercale plus
avant `services.audit_store.AuditStore.record_event` — lui-même déjà
entièrement déterministe côté V1 (chaînage SHA-256, aucune dépendance LLM).

Autrement dit : `services.audit_store.AuditStore` et
`tools.audit_redaction.redact_audit_payload` sont réutilisés tels quels
(aucune duplication) ; ce qui disparaît en V2 est uniquement la couche
`agents/audit_agent` qui les enveloppait d'un appel LLM.

Point d'entrée unique : `AuditService.record(...)`.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from schemas.audit import AuditEvent, AuditEventType, RedactionStatus
from services.audit_store import AuditorExport, AuditStore, ClaimIntegrityReport
from tools.audit_redaction import redact_audit_payload

__all__ = ["AuditService"]

_MAX_OUTCOME_LENGTH = 2000
"""Miroir de schemas.audit.AuditEvent.outcome (max_length=2000) — garde-fou
appliqué ici pour ne jamais tenter de construire un AuditEvent invalide."""

_KEY_SECRET_HINT_RE = re.compile(r"(?:api[_-]?key|secret|password|token|bearer)", re.IGNORECASE)
"""tools.audit_redaction.redact_audit_payload ne filtre un secret que sur le
CONTENU d'une valeur (motif "api_key: xyz"), jamais sur le seul NOM d'une
clé (ex. {"api_key": "sk-..."} sans séparateur ":"/"=" dans la valeur
elle-même) — comportement V1 volontairement non modifié ici (fichier hors
liste blanche §0). Ce filtre complémentaire, purement local à ce service,
retire aussi les clés dont le NOM évoque un secret, en défense en
profondeur avant que schemas.audit.AuditEvent ne rejette de toute façon
un `outcome` contenant un tel motif (double contrôle, jamais un contournement)."""


def _drop_secret_looking_keys(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if not _KEY_SECRET_HINT_RE.search(key)}


def _deterministic_outcome(outcome: str, details: Mapping[str, Any] | None) -> tuple[str, RedactionStatus]:
    """Combine un résumé texte court avec un détail structuré déjà rédigé.

    Fonction pure, aucun appel LLM : le détail est simplement rédigé
    (`redact_audit_payload`) puis sérialisé en JSON compact et concaténé au
    résumé — jamais paraphrasé. Le résultat est borné à
    `_MAX_OUTCOME_LENGTH`, tronqué de façon déterministe si nécessaire.
    """
    if not details:
        return outcome, RedactionStatus.NOT_REDACTED

    redacted = redact_audit_payload(dict(details))
    status_value = redacted.pop("redaction_status", RedactionStatus.NOT_REDACTED.value)
    status = RedactionStatus(status_value)
    redacted = _drop_secret_looking_keys(redacted)

    suffix = json.dumps(redacted, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    combined = f"{outcome} | details={suffix}"
    if len(combined) > _MAX_OUTCOME_LENGTH:
        combined = combined[: _MAX_OUTCOME_LENGTH - 1] + "…"
    return combined, status


class AuditService:
    """Service d'audit déterministe, injectable — enveloppe un `AuditStore`.

    Aucune instance globale cachée — à instancier explicitement (ou à
    injecter un `AuditStore` déjà existant, ex. partagé avec l'orchestrateur
    V2), même convention que les autres services du projet.
    """

    def __init__(self, store: AuditStore | None = None) -> None:
        self._store = store if store is not None else AuditStore()

    @property
    def store(self) -> AuditStore:
        return self._store

    def record(
        self,
        *,
        case_id: str,
        event_type: AuditEventType,
        actor: str,
        outcome: str,
        agent_name: str | None = None,
        tool_calls: Sequence[str] = (),
        evidence_ids: Sequence[str] = (),
        details: Mapping[str, Any] | None = None,
    ) -> AuditEvent:
        """Construit et persiste un `AuditEvent` — sans jamais appeler de LLM.

        `details` (optionnel) est un payload structuré brut (jamais transmis
        tel quel) : rédigé déterministiquement puis condensé dans `outcome`.
        `model_name`/`prompt_version` de `AuditEvent` restent `None` — ce
        service ne normalise jamais via un modèle, contrairement à
        `agents/audit_agent` (V1).
        """
        final_outcome, redaction_status = _deterministic_outcome(outcome, details)
        return self._store.record_event(
            case_id=case_id,
            event_type=event_type,
            actor=actor,
            outcome=final_outcome,
            redaction_status=redaction_status,
            agent_name=agent_name,
            model_name=None,
            prompt_version=None,
            tool_calls=tool_calls,
            evidence_ids=evidence_ids,
        )

    def verify_claim_integrity(self, case_id: str) -> ClaimIntegrityReport:
        return self._store.verify_claim_integrity(case_id)

    def export_for_auditor(self, case_id: str | None = None) -> AuditorExport:
        return self._store.export_for_auditor(case_id=case_id)

    def export_to_json(self, case_id: str | None = None, *, indent: int | None = 2) -> str:
        return self._store.export_to_json(case_id=case_id, indent=indent)

    def export_to_jsonl(self, case_id: str | None = None) -> str:
        return self._store.export_to_jsonl(case_id=case_id)
