"""Journal d'audit append-only, injectable — services/audit_store.py.

Stocke des ``schemas.audit.AuditEvent`` déjà validés (chaîne
``previous_hash``/``event_hash`` déjà vérifiée au niveau du schéma). Une
seule méthode de mutation existe : ``append_event``. Il n'y a
volontairement aucune méthode ``update_event``/``delete_event`` — ce n'est
pas une omission mais une garantie structurelle : rien dans cette classe
ne permet d'altérer ou de retirer un événement déjà accepté.

``append_event`` refuse lui-même deux types de tentative de modification
déguisée en ajout :
  - réutiliser un ``event_id`` déjà présent dans le journal ;
  - rompre la chaîne d'un dossier (``previous_hash`` qui ne prolonge pas
    exactement le dernier ``event_hash`` enregistré pour ce ``case_id``).

Chaque événement est copié en profondeur à l'entrée (``append_event``) et à
la sortie (``read_by_case_id``/``export_for_auditor``) : ni la mutation de
l'objet passé par l'appelant après l'ajout, ni la mutation d'un objet lu,
ne peuvent jamais altérer ce qui est réellement stocké.

Aucune instance globale cachée — même patron que
``services.duplicate_index.DuplicateIndex`` /
``orchestrator.model_registry.ModelRegistry`` : à instancier et injecter
explicitement. Persistance en mémoire uniquement pour la durée de vie de
l'objet ; l'écriture sur disque (``logs/audit/``, voir
``config.settings.Settings.claimshield_audit_dir``) est hors périmètre de
ce module.

``record_event`` est le point d'entrée recommandé pour ajouter un
événement : il calcule lui-même ``previous_hash`` (dernier ``event_hash``
connu pour ce ``case_id``, ``None`` pour le premier) et ``event_hash``
(``schemas.audit.compute_event_hash``, empreinte du contenu canonique) —
l'appelant ne fournit jamais lui-même de hash. ``append_event`` reste
disponible pour un événement déjà entièrement construit (ex. rechargement
depuis une persistance externe), avec les mêmes garanties de refus.

``verify_claim_integrity`` effectue une vérification d'intégrité complète
d'un dossier : recalcul du hash de chaque événement à partir de son
contenu (détecte un contenu modifié sans que l'empreinte n'ait été mise à
jour), vérification du chaînage ``previous_hash``/``event_hash`` (détecte
un trou ou une réécriture de chaîne) et de l'ordre chronologique des
horodatages (détecte une réorganisation de l'ordre de stockage). Ne
suppose jamais que les événements en mémoire sont restés honnêtes entre
deux appels — utile en défense en profondeur si un bug ou une future
persistance venait un jour contourner ``append_event``.

``export_for_auditor``/``export_to_json``/``export_to_jsonl`` produisent un
export en lecture seule, filtrable par ``case_id`` (``None`` = tous les
dossiers), incluant les événements (hash-chain ``previous_hash``/
``event_hash`` et versions ``model_name``/``prompt_version`` compris) et
les anomalies d'intégrité déjà détectées par ``verify_claim_integrity`` —
jamais un simple booléen masquant le détail. Ne contiennent que des champs
déjà validés par ``schemas.audit.AuditEvent`` (``extra="forbid"``, secrets/
chemins rejetés, champs texte bornés en longueur) : aucune donnée brute ou
excessive n'est jamais ajoutée par l'export lui-même.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import Enum

from pydantic import Field

from schemas.audit import AuditEvent, AuditEventType, RedactionStatus, build_audit_event, compute_event_hash
from schemas.domain import StrictModel
from schemas.results import StructuredError


class AuditStoreError(Exception):
    """Erreur du journal d'audit — porte toujours un StructuredError."""

    def __init__(self, structured: StructuredError) -> None:
        super().__init__(structured.message)
        self.structured = structured


class IntegrityIssueType(str, Enum):
    """Nature d'une anomalie détectée par ``verify_claim_integrity`` —
    trois dimensions indépendantes, jamais confondues entre elles."""

    CHAIN_BROKEN = "chain_broken"
    """previous_hash ne prolonge pas le event_hash de l'événement
    précédent (ou n'est pas None pour le premier)."""
    CONTENT_TAMPERED = "content_tampered"
    """event_hash stocké ne correspond plus au contenu canonique
    recalculé de l'événement — le contenu a été modifié après coup."""
    INVALID_ORDER = "invalid_order"
    """timestamp d'un événement antérieur à celui de l'événement qui le
    précède dans le journal — ordre chronologique incohérent."""


class IntegrityIssue(StrictModel):
    """Une anomalie ponctuelle, attribuée à un event_id précis."""

    event_id: str
    issue_type: IntegrityIssueType
    detail: str = Field(..., min_length=1)


EXPORT_FORMAT_VERSION = "1.0.0"
"""Version du format d'export auditeur (JSON/JSONL) — indépendante de
``schemas.audit.AuditEventType``/``RedactionStatus`` ou de la version d'un
prompt LLM. Change uniquement si la forme de l'export elle-même évolue
(nouveau champ, renommage...), jamais pour un simple ajout de données."""


class AuditorExport(StrictModel):
    """Export en lecture seule destiné à un auditeur externe.

    Ne porte aucune méthode ni aucun champ de modification — un simple
    instantané. ``chain_intact``/``broken_at_event_id``/``issues`` sont
    recalculés à chaque export à partir des événements réellement stockés
    (``AuditStore.verify_claim_integrity``), jamais mis en cache : un export
    ne peut jamais affirmer une intégrité qu'il n'a pas lui-même vérifiée au
    moment de l'appel.

    Ne contient que des champs déjà validés par ``schemas.audit.AuditEvent``
    (``extra="forbid"``, secrets/chemins/traversées rejetés à la
    construction, ``outcome``/``actor``/``agent_name``/``model_name``/
    ``prompt_version`` bornés en longueur) — jamais un champ brut recalculé
    ou une donnée non structurée ajoutée par l'export lui-même.
    """

    export_format_version: str = EXPORT_FORMAT_VERSION
    case_id: str | None = Field(
        default=None, description="None = export global, tous les dossiers confondus."
    )
    exported_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_count: int = Field(ge=0)
    chain_intact: bool
    broken_at_event_id: str | None = None
    issues: tuple[IntegrityIssue, ...] = Field(
        default_factory=tuple,
        description="Anomalies détectées (chaîne rompue, contenu altéré, ordre incohérent) — "
        "vide si le journal exporté est intact.",
    )
    events: tuple[AuditEvent, ...] = Field(default_factory=tuple)


# ── Vérification d'intégrité complète d'un dossier ───────────────────────────


class ClaimIntegrityReport(StrictModel):
    """Résultat d'une vérification d'intégrité complète d'un dossier.

    ``intact`` est calculé (jamais fourni) : vrai si et seulement si
    ``issues`` est vide. Un dossier inconnu (aucun événement) est
    vacuously intact — absence d'historique n'est jamais une anomalie."""

    case_id: str
    event_count: int = Field(ge=0)
    intact: bool
    issues: tuple[IntegrityIssue, ...] = Field(default_factory=tuple)
    verified_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AuditStore:
    """Journal d'audit append-only, en mémoire, injectable.

    Voir le docstring du module pour les garanties de non-modification.
    """

    def __init__(self) -> None:
        self._events_by_case: dict[str, list[AuditEvent]] = {}
        self._event_ids: set[str] = set()

    def __len__(self) -> int:
        return len(self._event_ids)

    # ── Seule méthode de mutation ─────────────────────────────────────────

    def append_event(self, event: AuditEvent) -> None:
        """Ajoute un événement déjà validé au journal.

        Lève ``AuditStoreError`` (jamais un ajout partiel ni un
        écrasement silencieux) si :
          - ``event.event_id`` est déjà présent dans le journal —
            réutiliser un identifiant existant serait une modification
            déguisée en ajout, toujours refusée ;
          - ``event.previous_hash`` ne prolonge pas exactement le dernier
            ``event_hash`` déjà enregistré pour ce ``case_id`` (ou n'est
            pas ``None`` pour le tout premier événement du dossier) —
            un trou ou une réécriture de chaîne est détecté avant
            stockage, jamais après coup.
        """
        if event.event_id in self._event_ids:
            raise AuditStoreError(
                StructuredError(
                    code="AUDIT_EVENT_ALREADY_EXISTS",
                    message=(
                        f"event_id '{event.event_id}' déjà présent dans le journal — "
                        "une modification d'événement existant est refusée."
                    ),
                    field="event_id",
                )
            )

        chain = self._events_by_case.setdefault(event.case_id, [])
        expected_previous = chain[-1].event_hash if chain else None
        if event.previous_hash != expected_previous:
            raise AuditStoreError(
                StructuredError(
                    code="AUDIT_CHAIN_BROKEN",
                    message=(
                        f"previous_hash incohérent pour case_id='{event.case_id}' : "
                        f"attendu {expected_previous!r}, reçu {event.previous_hash!r} — "
                        "insertion refusée pour ne jamais rompre la chaîne."
                    ),
                    field="previous_hash",
                )
            )

        # Copie défensive : la mutation ultérieure de l'objet de l'appelant
        # (validate_assignment=True autorise l'assignation côté StrictModel)
        # ne doit jamais atteindre ce qui est réellement stocké.
        chain.append(event.model_copy(deep=True))
        self._event_ids.add(event.event_id)

    def record_event(
        self,
        *,
        case_id: str,
        event_type: AuditEventType,
        actor: str,
        outcome: str,
        redaction_status: RedactionStatus,
        agent_name: str | None = None,
        model_name: str | None = None,
        prompt_version: str | None = None,
        tool_calls: Sequence[str] = (),
        evidence_ids: Sequence[str] = (),
    ) -> AuditEvent:
        """Construit et ajoute un nouvel événement en calculant lui-même
        ``previous_hash`` et ``event_hash`` — l'appelant ne fournit jamais
        lui-même de valeur de hash.

        ``previous_hash`` est le ``event_hash`` du dernier événement déjà
        enregistré pour ce ``case_id`` (``None`` s'il s'agit du premier).
        Délègue le calcul de ``event_hash`` à
        ``schemas.audit.build_audit_event`` (empreinte du contenu
        canonique), puis passe par ``append_event`` — qui revérifie la
        chaîne indépendamment, en défense en profondeur.
        """
        chain = self._events_by_case.get(case_id, [])
        previous_hash = chain[-1].event_hash if chain else None

        event = build_audit_event(
            case_id=case_id,
            event_type=event_type,
            actor=actor,
            outcome=outcome,
            previous_hash=previous_hash,
            redaction_status=redaction_status,
            agent_name=agent_name,
            model_name=model_name,
            prompt_version=prompt_version,
            tool_calls=tool_calls,
            evidence_ids=evidence_ids,
        )
        self.append_event(event)
        return event

    # ── Lecture ────────────────────────────────────────────────────────────

    def read_by_case_id(self, case_id: str) -> tuple[AuditEvent, ...]:
        """Retourne les événements d'un dossier, dans l'ordre d'ajout.

        Dossier inconnu → tuple vide, jamais une erreur (absence
        d'historique n'est pas une anomalie). Toujours des copies —
        muter la valeur retournée n'affecte jamais le journal."""
        return tuple(event.model_copy(deep=True) for event in self._events_by_case.get(case_id, ()))

    # ── Export auditeur ──────────────────────────────────────────────────

    def export_for_auditor(self, case_id: str | None = None) -> AuditorExport:
        """Export en lecture seule pour un auditeur externe, filtrable par
        ``case_id``.

        ``case_id=None`` exporte tous les dossiers connus (ordre
        d'insertion des dossiers, puis ordre d'ajout au sein de chacun).
        La vérification d'intégrité (``verify_claim_integrity`` — chaîne
        rompue, contenu altéré, ordre incohérent) porte sur chaque dossier
        indépendamment — les chaînes ne sont jamais comparées entre
        dossiers distincts. ``issues`` agrège les anomalies de tous les
        dossiers exportés ; ``chain_intact``/``broken_at_event_id`` restent
        des résumés pratiques dérivés de ``issues`` (jamais recalculés
        séparément), pour compatibilité avec les consommateurs existants.
        """
        if case_id is not None:
            case_ids = [case_id] if case_id in self._events_by_case else []
        else:
            case_ids = list(self._events_by_case.keys())

        all_events: list[AuditEvent] = []
        all_issues: list[IntegrityIssue] = []
        for cid in case_ids:
            report = self.verify_claim_integrity(cid)
            all_events.extend(self.read_by_case_id(cid))
            all_issues.extend(report.issues)

        chain_intact = not any(
            issue.issue_type is IntegrityIssueType.CHAIN_BROKEN for issue in all_issues
        )
        broken_at = next(
            (
                issue.event_id
                for issue in all_issues
                if issue.issue_type is IntegrityIssueType.CHAIN_BROKEN
            ),
            None,
        )

        return AuditorExport(
            case_id=case_id,
            event_count=len(all_events),
            chain_intact=chain_intact,
            broken_at_event_id=broken_at,
            issues=tuple(all_issues),
            events=tuple(all_events),
        )

    def export_to_json(self, case_id: str | None = None, *, indent: int | None = 2) -> str:
        """Retourne l'export auditeur (``export_for_auditor``) sérialisé en
        JSON — un seul objet, filtrable par ``case_id``.

        Ne sérialise que des champs déjà validés (``AuditorExport``/
        ``AuditEvent``/``IntegrityIssue``, tous ``extra="forbid"``) —
        aucune donnée brute ou non structurée n'est jamais ajoutée par
        cette méthode."""
        export = self.export_for_auditor(case_id=case_id)
        return export.model_dump_json(indent=indent)

    def export_to_jsonl(self, case_id: str | None = None) -> str:
        """Retourne l'export auditeur en JSON Lines (une ligne d'en-tête
        résumant l'intégrité, puis une ligne par événement), filtrable par
        ``case_id``.

        Format adapté à un pipeline d'ingestion externe (SIEM, entrepôt de
        logs) qui traite un événement à la fois plutôt qu'un unique objet
        JSON volumineux. La ligne d'en-tête porte ``export_format_version``,
        le compte d'événements, l'état de la chaîne et les anomalies
        détectées (``issues``) — jamais de contenu d'événement dupliqué.
        Chaque ligne événement reprend exactement les champs de
        ``schemas.audit.AuditEvent`` (hash-chain ``previous_hash``/
        ``event_hash`` et versions ``model_name``/``prompt_version``
        compris), sans aucun champ supplémentaire.
        """
        export = self.export_for_auditor(case_id=case_id)
        header = {
            "type": "export_summary",
            "export_format_version": export.export_format_version,
            "case_id": export.case_id,
            "exported_at": export.exported_at.isoformat(),
            "event_count": export.event_count,
            "chain_intact": export.chain_intact,
            "broken_at_event_id": export.broken_at_event_id,
            "anomalies": [issue.model_dump(mode="json") for issue in export.issues],
        }
        lines = [json.dumps(header, ensure_ascii=False, sort_keys=True)]
        for event in export.events:
            event_line = {"type": "event", **event.model_dump(mode="json")}
            lines.append(json.dumps(event_line, ensure_ascii=False, sort_keys=True))
        return "\n".join(lines) + "\n"

    # ── Vérification d'intégrité complète ────────────────────────────────

    def verify_claim_integrity(self, case_id: str) -> ClaimIntegrityReport:
        """Vérifie l'intégrité complète du journal d'un dossier.

        Pour chaque événement, dans l'ordre de stockage, contrôle trois
        dimensions indépendantes (une anomalie sur l'une n'empêche jamais
        de détecter les autres) :
          - ``CHAIN_BROKEN`` : ``previous_hash`` ne correspond pas au
            ``event_hash`` de l'événement précédent (ou n'est pas ``None``
            pour le premier) ;
          - ``CONTENT_TAMPERED`` : le contenu canonique recalculé
            (``schemas.audit.compute_event_hash``) ne correspond plus à
            l'``event_hash`` stocké — le contenu a changé après coup ;
          - ``INVALID_ORDER`` : l'horodatage de l'événement est antérieur
            à celui de l'événement qui le précède dans le journal.

        Dossier inconnu → ``event_count=0``, ``intact=True`` (absence
        d'historique n'est jamais une anomalie).
        """
        events = self.read_by_case_id(case_id)
        issues: list[IntegrityIssue] = []
        expected_previous: str | None = None
        previous_timestamp: datetime | None = None

        for event in events:
            if event.previous_hash != expected_previous:
                issues.append(
                    IntegrityIssue(
                        event_id=event.event_id,
                        issue_type=IntegrityIssueType.CHAIN_BROKEN,
                        detail=(
                            f"previous_hash={event.previous_hash!r} ne correspond pas "
                            f"au hash attendu {expected_previous!r}."
                        ),
                    )
                )

            recomputed = compute_event_hash(event)
            if recomputed != event.event_hash:
                issues.append(
                    IntegrityIssue(
                        event_id=event.event_id,
                        issue_type=IntegrityIssueType.CONTENT_TAMPERED,
                        detail=(
                            "event_hash stocké ne correspond pas au contenu recalculé "
                            "de l'événement — contenu modifié après coup."
                        ),
                    )
                )

            if previous_timestamp is not None and event.timestamp < previous_timestamp:
                issues.append(
                    IntegrityIssue(
                        event_id=event.event_id,
                        issue_type=IntegrityIssueType.INVALID_ORDER,
                        detail=(
                            f"timestamp {event.timestamp.isoformat()} antérieur à "
                            "l'événement précédent dans le journal — ordre "
                            "chronologique incohérent."
                        ),
                    )
                )

            expected_previous = event.event_hash
            previous_timestamp = event.timestamp

        return ClaimIntegrityReport(
            case_id=case_id,
            event_count=len(events),
            intact=not issues,
            issues=tuple(issues),
        )


__all__ = [
    "AuditStore",
    "AuditStoreError",
    "AuditorExport",
    "ClaimIntegrityReport",
    "EXPORT_FORMAT_VERSION",
    "IntegrityIssue",
    "IntegrityIssueType",
]
