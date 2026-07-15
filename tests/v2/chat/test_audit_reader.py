"""Tests de chat/audit_reader.py (Phase V2-11c) — `AuditService` injecté
explicitement (jamais l'instance par défaut construite par
`build_audit_summary`, voir la limite opérationnelle documentée dans le
module : elle serait déconnectée de celle réellement utilisée par le graphe
compilé)."""
from __future__ import annotations

from pathlib import Path

from chat.audit_reader import build_audit_summary
from chat.schemas import AuditSummary
from schemas.audit import AuditEventType
from services.audit_service import AuditService


def _service() -> AuditService:
    return AuditService()


class TestBuildAuditSummaryUnknownCase:
    def test_unknown_case_returns_empty_intact_summary(self):
        summary = build_audit_summary("CLM-9000", audit_service=_service())
        assert summary == AuditSummary(
            case_id="CLM-9000",
            event_count=0,
            chain_intact=True,
            event_type_counts={},
            actors=[],
            issues_count=0,
        )


class TestBuildAuditSummaryPopulatedCase:
    def test_event_count_and_chain_intact(self):
        service = _service()
        service.record(
            case_id="CLM-9001",
            event_type=AuditEventType.CLAIM_STARTED,
            actor="system:intake_safety",
            outcome="Dossier reçu.",
        )
        service.record(
            case_id="CLM-9001",
            event_type=AuditEventType.AGENT_CALLED,
            actor="system:medical_risk",
            outcome="Analyse effectuée.",
        )
        summary = build_audit_summary("CLM-9001", audit_service=service)
        assert summary.case_id == "CLM-9001"
        assert summary.event_count == 2
        assert summary.chain_intact is True
        assert summary.issues_count == 0

    def test_event_type_counts_aggregated_by_type(self):
        service = _service()
        for _ in range(3):
            service.record(
                case_id="CLM-9002",
                event_type=AuditEventType.AGENT_CALLED,
                actor="system:medical_risk",
                outcome="Analyse effectuée.",
            )
        service.record(
            case_id="CLM-9002",
            event_type=AuditEventType.ERROR,
            actor="system:document_understanding",
            outcome="Échec technique.",
        )
        summary = build_audit_summary("CLM-9002", audit_service=service)
        assert summary.event_type_counts == {"agent_called": 3, "error": 1}

    def test_actors_deduplicated_and_sorted(self):
        service = _service()
        service.record(
            case_id="CLM-9003",
            event_type=AuditEventType.AGENT_CALLED,
            actor="system:medical_risk",
            outcome="A",
        )
        service.record(
            case_id="CLM-9003",
            event_type=AuditEventType.AGENT_CALLED,
            actor="system:eligibility",
            outcome="B",
        )
        service.record(
            case_id="CLM-9003",
            event_type=AuditEventType.AGENT_CALLED,
            actor="system:medical_risk",
            outcome="C",
        )
        summary = build_audit_summary("CLM-9003", audit_service=service)
        assert summary.actors == ["system:eligibility", "system:medical_risk"]

    def test_events_from_other_cases_never_leak_into_summary(self):
        service = _service()
        service.record(
            case_id="CLM-9004",
            event_type=AuditEventType.AGENT_CALLED,
            actor="system:medical_risk",
            outcome="Dossier A.",
        )
        service.record(
            case_id="CLM-9005",
            event_type=AuditEventType.AGENT_CALLED,
            actor="system:medical_risk",
            outcome="Dossier B.",
        )
        summary = build_audit_summary("CLM-9004", audit_service=service)
        assert summary.event_count == 1


class TestBuildAuditSummaryNeverExposesRawOutcome:
    """Critère d'acceptation explicite de V2-11c : `get_audit_summary`
    n'expose jamais le contenu brut d'un `outcome`."""

    def test_raw_outcome_text_absent_from_summary(self):
        service = _service()
        secret_outcome = "Détail interne très spécifique : réf-XYZ-CONFIDENTIEL-42."
        service.record(
            case_id="CLM-9006",
            event_type=AuditEventType.AGENT_CALLED,
            actor="system:medical_risk",
            outcome=secret_outcome,
        )
        summary = build_audit_summary("CLM-9006", audit_service=service)
        dumped = summary.model_dump_json()
        assert secret_outcome not in dumped
        assert "CONFIDENTIEL" not in dumped

    def test_summary_schema_has_no_outcome_field(self):
        """Garantie structurelle, pas seulement comportementale : le schéma
        `AuditSummary` lui-même ne porte aucun champ capable de contenir un
        `outcome` brut."""
        assert "outcome" not in AuditSummary.model_fields


class TestAuditReaderArchitectureException:
    """`chat/audit_reader.py` est, avec `chat/simulation_engine.py`, l'un
    des deux seuls modules de `chat/` autorisés à accéder directement à
    `services.*`/`graph.*` métier — verrouille cette garantie plutôt que de
    la laisser purement déclarative (même patron que
    `tests/v2/chat/test_simulation_engine.py::
    TestGraphAccessExceptionIsDocumentedAndUnique`)."""

    _OTHER_CHAT_MODULES = (
        "agent.py",
        "planner.py",
        "nlu.py",
        "response_composer.py",
        "explanation_engine.py",
        "correction_engine.py",
        "schemas.py",
        "prompt.py",
    )

    def test_only_audit_reader_and_simulation_engine_import_services_directly(self):
        import chat

        chat_dir = Path(chat.__file__).parent
        for filename in self._OTHER_CHAT_MODULES:
            source = (chat_dir / filename).read_text(encoding="utf-8")
            for line in source.splitlines():
                stripped = line.strip()
                assert not stripped.startswith("import services."), f"{filename}: {stripped!r}"
                assert not stripped.startswith("from services."), f"{filename}: {stripped!r}"

    def test_audit_reader_does_import_audit_service(self):
        """Contre-preuve : l'exception existe bel et bien."""
        source = Path("chat/audit_reader.py").read_text(encoding="utf-8")
        assert "from services.audit_service import AuditService" in source
