"""Modèle SQLAlchemy de persistance du journal d'audit — database/audit_models.py.

Persiste ``schemas.audit.AuditEvent`` (chaîne ``previous_hash``/``event_hash``
déjà validée au niveau Pydantic — voir ``schemas/audit.py`` et
``services/audit_store.py``, qui restent la source de vérité en mémoire
pour la durée de vie du processus). Ce module ajoute une couche de
persistance optionnelle, jamais un remplacement de l'un ou l'autre.

Compatible SQLite (développement local — ``DATABASE_URL`` par défaut de
``config.settings``, pilote ``aiosqlite``) et PostgreSQL (déploiement
futur, pilote ``asyncpg``) sans changement de schéma : aucun type
spécifique à un backend n'est utilisé (pas de JSONB natif Postgres, pas
d'ENUM natif SQL) — uniquement des types génériques SQLAlchemy (String,
DateTime(timezone=True), JSON) portables sur les deux moteurs. Les deux
pilotes installés (``aiosqlite``, ``asyncpg``, voir ``requirements.txt``)
sont asynchrones — ``init_db`` utilise donc ``AsyncEngine``, jamais un
moteur synchrone (aucun pilote sync n'est une dépendance du projet).

Append-only par convention d'usage (voir ``AuditEventRow``) — SQLAlchemy
ne peut pas l'interdire aussi strictement que
``services.audit_store.AuditStore`` (rien n'empêche un appelant d'émettre
un ``UPDATE``/``DELETE`` explicite sur la session), mais ce module ne
fournit lui-même aucune fonction de mise à jour ou de suppression, et
``audit_event_to_row``/``row_to_audit_event`` ne sont prévus que pour un
usage en insertion pure.
"""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Index, String
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from schemas.audit import AuditEvent, AuditEventType, RedactionStatus


class Base(DeclarativeBase):
    """Base déclarative dédiée à ce module.

    Pas de metadata partagée avec un éventuel futur module
    ``database/*_models.py`` : chaque domaine gère ses propres tables et sa
    propre initialisation, jamais une metadata globale implicite qui
    coupleraient des tables sans rapport entre elles.
    """


class AuditEventRow(Base):
    """Ligne persistée d'un ``schemas.audit.AuditEvent``.

    ``event_id`` est la clé primaire (déjà un identifiant globalement
    unique, généré par ``schemas.audit.build_audit_event`` — une clé
    primaire est automatiquement indexée par SQLite comme par PostgreSQL,
    inutile de dupliquer un ``Index`` explicite dessus). ``case_id``,
    ``event_type`` et ``timestamp`` portent chacun un index explicite,
    comme demandé — ce sont les trois axes de lecture attendus : tous les
    événements d'un dossier, tous les événements d'une catégorie, une
    fenêtre temporelle.

    ``event_type``/``redaction_status`` sont stockés comme de simples
    chaînes (la valeur de l'enum Pydantic), pas comme un ENUM SQL natif :
    un ENUM PostgreSQL exige une migration ``ALTER TYPE`` à chaque nouvelle
    valeur ajoutée à ``AuditEventType``/``RedactionStatus``, ce qui
    contredirait l'objectif de rester un modèle minimal et portable — la
    validation réelle des valeurs reste de toute façon assurée par
    Pydantic (``row_to_audit_event`` revalide systématiquement).
    """

    __tablename__ = "audit_events"
    __table_args__ = (
        Index("ix_audit_events_case_id", "case_id"),
        Index("ix_audit_events_event_type", "event_type"),
        Index("ix_audit_events_timestamp", "timestamp"),
    )

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    case_id: Mapped[str] = mapped_column(String(32), nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    actor: Mapped[str] = mapped_column(String(255), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    agent_name: Mapped[str | None] = mapped_column(String(255), nullable=True, default=None)
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True, default=None)
    prompt_version: Mapped[str | None] = mapped_column(String(32), nullable=True, default=None)
    tool_calls: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    outcome: Mapped[str] = mapped_column(String(2000), nullable=False)

    previous_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, default=None)
    event_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    redaction_status: Mapped[str] = mapped_column(String(32), nullable=False)


# ── Conversion Pydantic <-> ORM ───────────────────────────────────────────────


def audit_event_to_row(event: AuditEvent) -> AuditEventRow:
    """Convertit un ``AuditEvent`` déjà validé en ligne insérable.

    Ne fait aucune validation propre : ``event`` est supposé déjà validé
    par Pydantic (construit via ``schemas.audit.build_audit_event`` ou
    déjà accepté par ``services.audit_store.AuditStore``). Le timestamp
    est normalisé en UTC avant stockage — SQLite (contrairement à
    PostgreSQL) ne conserve pas le fuseau horaire d'un ``DateTime`` : sans
    cette normalisation à l'écriture, la reconstruction sur SQLite (voir
    ``row_to_audit_event``) devrait deviner le fuseau d'origine."""
    return AuditEventRow(
        event_id=event.event_id,
        case_id=event.case_id,
        event_type=event.event_type.value,
        actor=event.actor,
        timestamp=event.timestamp.astimezone(UTC),
        agent_name=event.agent_name,
        model_name=event.model_name,
        prompt_version=event.prompt_version,
        tool_calls=list(event.tool_calls),
        outcome=event.outcome,
        previous_hash=event.previous_hash,
        event_hash=event.event_hash,
        evidence_ids=list(event.evidence_ids),
        redaction_status=event.redaction_status.value,
    )


def row_to_audit_event(row: AuditEventRow) -> AuditEvent:
    """Reconstruit un ``AuditEvent`` Pydantic à partir d'une ligne persistée.

    Revalidé intégralement (``extra='forbid'``, format des hash, chaîne
    non dégénérée, ``event_type``/``redaction_status`` dans l'énumération
    attendue...) — une ligne corrompue ou modifiée hors de ce module est
    détectée ici, jamais acceptée silencieusement.

    ``timestamp`` : si la valeur relue est naïve (sans fuseau — cas de
    SQLite, qui ne conserve pas le fuseau d'un ``DateTime(timezone=True)``),
    on lui rattache ``UTC`` — cohérent avec ``audit_event_to_row`` qui
    normalise systématiquement en UTC avant écriture. Sur PostgreSQL
    (``TIMESTAMPTZ``), la valeur relue est déjà correctement rattachée à un
    fuseau et cette étape est un no-op."""
    timestamp = row.timestamp
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return AuditEvent(
        event_id=row.event_id,
        case_id=row.case_id,
        event_type=AuditEventType(row.event_type),
        actor=row.actor,
        timestamp=timestamp,
        agent_name=row.agent_name,
        model_name=row.model_name,
        prompt_version=row.prompt_version,
        tool_calls=list(row.tool_calls or []),
        outcome=row.outcome,
        previous_hash=row.previous_hash,
        event_hash=row.event_hash,
        evidence_ids=list(row.evidence_ids or []),
        redaction_status=RedactionStatus(row.redaction_status),
    )


# ── Initialisation minimale (équivalent d'une migration) ─────────────────────


async def init_db(database_url: str | None = None) -> AsyncEngine:
    """Crée la table ``audit_events`` si elle n'existe pas encore.

    Équivalent minimal d'une migration pour un projet qui n'a pas encore
    de scaffolding Alembic (``database/`` restait un stub avant ce module
    — voir ``CLAUDE.md``) : ``create_all`` n'agit que sur les tables
    absentes, jamais ``DROP`` ni ``ALTER`` sur une table déjà présente.

    ``database_url`` par défaut : ``config.settings.get_settings().database_url``
    (ex. ``sqlite+aiosqlite:///./storage/claimshield.db``). Fournir une URL
    explicite permet de pointer vers un PostgreSQL de test sans toucher à
    la configuration globale.
    """
    from config.settings import get_settings

    url = database_url or get_settings().database_url
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine


__all__ = [
    "AuditEventRow",
    "Base",
    "audit_event_to_row",
    "init_db",
    "row_to_audit_event",
]


if __name__ == "__main__":
    import asyncio

    asyncio.run(init_db())
