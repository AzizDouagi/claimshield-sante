"""Tests de la persistance SQLAlchemy du journal d'audit — database/audit_models.py.

Utilise SQLite en mémoire (``sqlite+aiosqlite:///:memory:``) — même pilote
async que ``DATABASE_URL`` par défaut du projet (``config/settings.py``),
sans toucher au fichier ``storage/claimshield.db`` réel. Couvre :
création de la table et des index attendus, aller-retour Pydantic <-> ORM
(y compris le fuseau horaire, perdu par SQLite sur un ``DateTime``),
chaînage de plusieurs événements, filtrage par ``case_id``/``event_type``,
et rejet d'une ligne dont l'``event_type`` ne correspond à aucune valeur
connue.
"""
from __future__ import annotations

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from database.audit_models import (
    AuditEventRow,
    audit_event_to_row,
    init_db,
    row_to_audit_event,
)
from schemas.audit import AuditEventType, RedactionStatus, build_audit_event

pytestmark = pytest.mark.asyncio


async def _fresh_engine() -> AsyncEngine:
    return await init_db("sqlite+aiosqlite:///:memory:")


def _event(**overrides):
    payload = {
        "case_id": "CLM-0001",
        "event_type": AuditEventType.CLAIM_STARTED,
        "actor": "audit_agent",
        "outcome": "started",
        "previous_hash": None,
        "redaction_status": RedactionStatus.NOT_REDACTED,
    }
    payload.update(overrides)
    return build_audit_event(**payload)


# ── init_db : table et index ─────────────────────────────────────────────────


async def test_init_db_cree_la_table_et_les_index_demandes():
    engine = await _fresh_engine()

    def _inspect(sync_conn):
        insp = inspect(sync_conn)
        assert "audit_events" in insp.get_table_names()
        index_names = {ix["name"] for ix in insp.get_indexes("audit_events")}
        assert index_names == {
            "ix_audit_events_case_id",
            "ix_audit_events_event_type",
            "ix_audit_events_timestamp",
        }
        pk = insp.get_pk_constraint("audit_events")
        assert pk["constrained_columns"] == ["event_id"]

    async with engine.connect() as conn:
        await conn.run_sync(_inspect)
    await engine.dispose()


async def test_init_db_est_idempotent_ne_leve_pas_si_rappele():
    engine = await _fresh_engine()
    # Rappeler init_db sur le même moteur ne doit jamais lever (create_all
    # n'agit que sur les tables absentes).
    async with engine.begin() as conn:
        from database.audit_models import Base

        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()


# ── Aller-retour Pydantic <-> ORM ─────────────────────────────────────────────


async def test_round_trip_preserve_le_contenu_et_le_hash():
    engine = await _fresh_engine()
    event = _event(tool_calls=["verifier_chronologie"], evidence_ids=["ev-1", "ev-2"])
    row = audit_event_to_row(event)

    session_factory = async_sessionmaker(engine)
    async with session_factory() as session:
        session.add(row)
        await session.commit()

    async with session_factory() as session:
        result = await session.execute(
            select(AuditEventRow).where(AuditEventRow.event_id == event.event_id)
        )
        fetched = result.scalar_one()
        rebuilt = row_to_audit_event(fetched)

    assert rebuilt == event
    assert rebuilt.event_hash == event.event_hash
    assert rebuilt.timestamp == event.timestamp  # fuseau horaire correctement rétabli
    await engine.dispose()


async def test_round_trip_preserve_une_chaine_de_deux_evenements():
    engine = await _fresh_engine()
    e1 = _event(outcome="started")
    e2 = _event(event_type=AuditEventType.AGENT_CALLED, outcome="called", previous_hash=e1.event_hash)

    session_factory = async_sessionmaker(engine)
    async with session_factory() as session:
        session.add_all([audit_event_to_row(e1), audit_event_to_row(e2)])
        await session.commit()

    async with session_factory() as session:
        result = await session.execute(
            select(AuditEventRow)
            .where(AuditEventRow.case_id == "CLM-0001")
            .order_by(AuditEventRow.timestamp)
        )
        rows = result.scalars().all()

    rebuilt = [row_to_audit_event(row) for row in rows]
    assert [e.event_hash for e in rebuilt] == [e1.event_hash, e2.event_hash]
    assert rebuilt[1].previous_hash == rebuilt[0].event_hash
    await engine.dispose()


# ── Filtrage par case_id / event_type ────────────────────────────────────────


async def test_filtre_par_case_id_isole_bien_chaque_dossier():
    engine = await _fresh_engine()
    e1 = _event(case_id="CLM-0001")
    e2 = _event(case_id="CLM-0002")

    session_factory = async_sessionmaker(engine)
    async with session_factory() as session:
        session.add_all([audit_event_to_row(e1), audit_event_to_row(e2)])
        await session.commit()

    async with session_factory() as session:
        result = await session.execute(
            select(AuditEventRow).where(AuditEventRow.case_id == "CLM-0001")
        )
        rows = result.scalars().all()

    assert len(rows) == 1
    assert rows[0].event_id == e1.event_id
    await engine.dispose()


async def test_filtre_par_event_type():
    engine = await _fresh_engine()
    e1 = _event(event_type=AuditEventType.CLAIM_STARTED)
    e2 = _event(event_type=AuditEventType.ANOMALY, previous_hash=e1.event_hash)

    session_factory = async_sessionmaker(engine)
    async with session_factory() as session:
        session.add_all([audit_event_to_row(e1), audit_event_to_row(e2)])
        await session.commit()

    async with session_factory() as session:
        result = await session.execute(
            select(AuditEventRow).where(AuditEventRow.event_type == AuditEventType.ANOMALY.value)
        )
        rows = result.scalars().all()

    assert len(rows) == 1
    assert rows[0].event_id == e2.event_id
    await engine.dispose()


# ── Ligne corrompue : rejet à la reconstruction ──────────────────────────────


async def test_row_to_audit_event_rejette_un_event_type_inconnu():
    engine = await _fresh_engine()
    event = _event()
    row = audit_event_to_row(event)
    row.event_type = "not_a_real_event_type"  # simule une ligne corrompue en base

    session_factory = async_sessionmaker(engine)
    async with session_factory() as session:
        session.add(row)
        await session.commit()

    async with session_factory() as session:
        result = await session.execute(
            select(AuditEventRow).where(AuditEventRow.event_id == event.event_id)
        )
        fetched = result.scalar_one()

    with pytest.raises(ValueError):
        row_to_audit_event(fetched)

    await engine.dispose()
