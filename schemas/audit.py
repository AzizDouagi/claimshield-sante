"""Schéma enrichi de l'événement d'audit — journal chaîné (Audit Agent, étape 14).

Distinct de ``schemas.results.AuditEvent`` (événement léger — case_id, actor,
action, outcome, agent_version, timestamp, details — déjà consommé par une
dizaine de modules : ``privacy_agent``, ``orchestrator/executor.py``,
``human_review/service.py``, tous les nœuds agents via ``ClaimState.audit_trail``).
Celui-ci n'est ni remplacé ni migré ici — seule une migration explicite,
répercutée sur tous ses consommateurs, pourrait un jour l'unifier avec ce
schéma. ``schemas.audit.AuditEvent`` porte le format enrichi et chaîné que
l'Audit Agent produira pour son propre journal structuré : traçabilité LLM
(``model_name``/``prompt_version``/``tool_calls``), lien vers les preuves
amont (``evidence_ids``), état de minimisation (``redaction_status``) et
chaînage cryptographique (``previous_hash``/``event_hash``) garantissant
qu'aucun événement déjà journalisé ne peut être modifié ou retiré sans
casser la chaîne.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from schemas.domain import StrictModel

# Dupliqué volontairement (jamais importé) : schemas/ est une couche basse,
# orchestrator/ en dépend et non l'inverse — même motif que
# human_review/models.py::CASE_ID_PATTERN, miroir de
# orchestrator/orchestrator.py::CASE_ID_PATTERN.
_CASE_ID_PATTERN = r"^CLM-\d{4,}$"

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

_ABSOLUTE_PATH_RE = re.compile(r"^(?:/|[A-Za-z]:[/\\]|\\\\)")
_SECRET_HINT_RE = re.compile(
    r"(?:api[_-]?key|secret\s*[:=]|password\s*[:=]|token\s*[:=]|bearer\s+[a-z0-9._~+/=-]+)",
    re.IGNORECASE,
)


def _reject_security_leak(value: str, field_name: str) -> str:
    """Interdit chemin absolu, traversée de répertoire ou marqueur de secret."""
    if _ABSOLUTE_PATH_RE.match(value) or ".." in value:
        raise ValueError(f"{field_name} : chemin absolu ou traversée de répertoire interdit")
    if _SECRET_HINT_RE.search(value):
        raise ValueError(f"{field_name} : marqueur de secret détecté, valeur refusée")
    return value


class AuditEventType(str, Enum):
    """Catégories stables d'événement d'audit —10 valeurs, jamais une chaîne libre.

    Chaque valeur couvre un moment distinct et non ambigu du cycle de vie
    d'un dossier, documenté dans ``AUDIT_EVENT_TYPE_DESCRIPTIONS`` (même
    patron que ``PRIVACY_CODE_DESCRIPTIONS``/``SECURITY_CODE_DESCRIPTIONS``
    dans ``schemas/domain.py``). Un ``event_type`` hors de cette énumération
    est toujours rejeté par Pydantic — jamais une valeur inventée à la
    volée par un producteur d'événement.
    """

    CLAIM_STARTED = "claim_started"
    AGENT_CALLED = "agent_called"
    TOOL_CALLED = "tool_called"
    ERROR = "error"
    HUMAN_DECISION = "human_decision"
    SECURITY_DECISION = "security_decision"
    RETRY = "retry"
    FAILURE = "failure"
    FINAL_REPORT = "final_report"
    ANOMALY = "anomaly"


AUDIT_EVENT_TYPE_DESCRIPTIONS: dict[AuditEventType, str] = {
    AuditEventType.CLAIM_STARTED: (
        "Ouverture du traitement d'un dossier — première trace du pipeline "
        "pour ce case_id, sert de genèse à la chaîne (previous_hash=None)."
    ),
    AuditEventType.AGENT_CALLED: (
        "Un agent métier a été invoqué via Orchestrator.execute_agent() — "
        "que l'appel ait réussi, échoué ou été refusé par une politique."
    ),
    AuditEventType.TOOL_CALLED: (
        "Un outil explicitement autorisé (allowlist par agent) a été "
        "effectivement invoqué pendant l'exécution d'un agent."
    ),
    AuditEventType.ERROR: (
        "Erreur technique ou de validation (sortie d'agent invalide, "
        "exception non catégorisée...) — jamais une décision métier."
    ),
    AuditEventType.HUMAN_DECISION: (
        "Décision humaine (HITL) validée — APPROVE/MODIFY/REJECT/RETRY, "
        "voir human_review.service.validate_and_audit_human_decision."
    ),
    AuditEventType.SECURITY_DECISION: (
        "Décision du Security Gate ou d'un scanner de sécurité — "
        "ALLOW/BLOCK/QUARANTINE ou détection de contenu suspect."
    ),
    AuditEventType.RETRY: (
        "Nouvelle tentative programmée après une panne transitoire "
        "(technique, jamais une décision humaine de type RETRY)."
    ),
    AuditEventType.FAILURE: (
        "Le dossier a atteint un état d'échec terminal — rejet contrôlé "
        "ou blocage, route graph.edges vers le nœud failure."
    ),
    AuditEventType.FINAL_REPORT: (
        "Synthèse finale du dossier à la clôture du pipeline (nœud "
        "finalize) — dernier événement attendu d'une chaîne complète."
    ),
    AuditEventType.ANOMALY: (
        "Anomalie ou désaccord détecté entre résultats déjà validés "
        "(tools.consistency.detect_result_disagreements ou équivalent)."
    ),
}


class RedactionStatus(str, Enum):
    """État de minimisation du contenu porté par l'événement.

    Champ obligatoire (aucun défaut) : force chaque producteur d'événement
    à se prononcer explicitement plutôt que de supposer un état sûr par
    défaut — cohérent avec la règle DENY-by-default déjà appliquée ailleurs
    dans le projet (``security/access_policies.py``).
    """

    NOT_REDACTED = "not_redacted"
    PARTIALLY_REDACTED = "partially_redacted"
    FULLY_REDACTED = "fully_redacted"


class AuditEvent(StrictModel):
    """Événement d'audit chaîné et attribué — journal structuré de l'Audit Agent.

    ``extra="forbid"`` hérité de ``StrictModel`` : aucun champ hors de ceux
    listés ci-dessous n'est accepté, aucune donnée libre non structurée ne
    peut s'y glisser.
    """

    # ── Identité de l'événement ──────────────────────────────────────────
    case_id: str = Field(..., pattern=_CASE_ID_PATTERN)
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: AuditEventType
    actor: str = Field(
        ..., min_length=1, max_length=255, description="Agent ou identifiant utilisateur"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # ── Traçabilité agent / LLM ───────────────────────────────────────────
    agent_name: str | None = Field(default=None, max_length=255)
    model_name: str | None = Field(default=None, max_length=255)
    prompt_version: str | None = Field(default=None, max_length=32)
    tool_calls: list[str] = Field(default_factory=list)
    outcome: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Résultat normalisé — jamais un document ou un contenu excessif : "
        "borné pour qu'un événement d'audit ne puisse jamais servir à faire transiter "
        "une donnée volumineuse.",
    )

    # ── Chaînage et preuves ───────────────────────────────────────────────
    previous_hash: str | None = Field(
        default=None,
        description="event_hash de l'événement précédent dans la chaîne — "
        "None uniquement pour le premier événement (genèse).",
    )
    event_hash: str = Field(..., description="Empreinte SHA-256 de cet événement.")
    evidence_ids: list[str] = Field(default_factory=list)
    redaction_status: RedactionStatus

    @field_validator("actor", "outcome", "agent_name", "model_name", "prompt_version")
    @classmethod
    def _validate_text_fields(cls, v: str | None, info) -> str | None:
        if v is None:
            return v
        return _reject_security_leak(v, info.field_name)

    @field_validator("tool_calls", "evidence_ids")
    @classmethod
    def _validate_string_lists(cls, v: list[str], info) -> list[str]:
        for item in v:
            if not item:
                raise ValueError(f"{info.field_name} : élément vide interdit")
            _reject_security_leak(item, info.field_name)
        return v

    @field_validator("previous_hash", "event_hash")
    @classmethod
    def _validate_hash_format(cls, v: str | None, info) -> str | None:
        if v is None:
            return v
        if not _HEX64_RE.match(v.lower()):
            raise ValueError(
                f"{info.field_name} : doit être une empreinte SHA-256 hexadécimale (64 caractères)"
            )
        return v.lower()

    @model_validator(mode="after")
    def _chain_is_not_degenerate(self) -> "AuditEvent":
        if self.previous_hash is not None and self.previous_hash == self.event_hash:
            raise ValueError(
                "event_hash ne peut pas être identique à previous_hash — chaîne dégénérée"
            )
        return self


# ── Calcul du hash canonique ──────────────────────────────────────────────────

_PLACEHOLDER_HASH = "0" * 64
"""Valeur factice pour la construction provisoire dans ``build_audit_event`` —
jamais un SHA-256 réel de contenu : ``compute_event_hash`` exclut
structurellement ``event_hash`` du contenu canonique, donc cette valeur
n'influence jamais le hash final calculé."""


def canonical_event_content(event: AuditEvent) -> dict:
    """Contenu canonique d'un événement — tous les champs sauf ``event_hash``
    lui-même (qui en est dérivé : le hash ne peut jamais dépendre de
    lui-même). ``previous_hash`` est inclus : modifier la position d'un
    événement dans la chaîne change donc aussi son empreinte."""
    return event.model_dump(mode="json", exclude={"event_hash"})


def compute_event_hash(event: AuditEvent) -> str:
    """SHA-256 hexadécimal du contenu canonique de l'événement.

    Sérialisation JSON triée par clé (``sort_keys=True``) et sans espace :
    un résultat déterministe, indépendant de l'ordre de construction des
    champs Python — deux événements avec un contenu identique produisent
    toujours le même hash, et le moindre changement de contenu (y compris
    ``previous_hash``) en produit un différent.
    """
    payload = canonical_event_content(event)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_audit_event(
    *,
    case_id: str,
    event_type: AuditEventType,
    actor: str,
    outcome: str,
    previous_hash: str | None,
    redaction_status: RedactionStatus,
    event_id: str | None = None,
    timestamp: datetime | None = None,
    agent_name: str | None = None,
    model_name: str | None = None,
    prompt_version: str | None = None,
    tool_calls: Sequence[str] = (),
    evidence_ids: Sequence[str] = (),
) -> AuditEvent:
    """Construit un ``AuditEvent`` dont ``event_hash`` est réellement calculé
    à partir de son contenu canonique — jamais une valeur inventée ou
    fournie par l'appelant.

    ``previous_hash`` reste à la charge de l'appelant (typiquement
    ``services.audit_store.AuditStore``, seul à savoir quel est le dernier
    ``event_hash`` enregistré pour ce ``case_id``) : cette fonction ne
    connaît qu'un seul événement à la fois, jamais la chaîne complète d'un
    dossier.

    Construction en deux temps, invisible depuis l'extérieur : un
    événement provisoire porte ``_PLACEHOLDER_HASH`` le temps de disposer
    d'un objet dont dériver le contenu canonique, puis un second événement,
    identique en tout point sauf ``event_hash``, est retourné avec
    l'empreinte réellement calculée.
    """
    provisional = AuditEvent(
        case_id=case_id,
        event_id=event_id or str(uuid4()),
        event_type=event_type,
        actor=actor,
        timestamp=timestamp or datetime.now(UTC),
        agent_name=agent_name,
        model_name=model_name,
        prompt_version=prompt_version,
        tool_calls=list(tool_calls),
        outcome=outcome,
        previous_hash=previous_hash,
        event_hash=_PLACEHOLDER_HASH,
        evidence_ids=list(evidence_ids),
        redaction_status=redaction_status,
    )
    real_hash = compute_event_hash(provisional)
    return AuditEvent(**{**provisional.model_dump(mode="json"), "event_hash": real_hash})


__all__ = [
    "AUDIT_EVENT_TYPE_DESCRIPTIONS",
    "AuditEvent",
    "AuditEventType",
    "RedactionStatus",
    "build_audit_event",
    "canonical_event_content",
    "compute_event_hash",
]
