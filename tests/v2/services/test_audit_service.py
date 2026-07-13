"""Tests de services/audit_service.py (V2) — Phase V2-0.

Garantit : aucun appel LLM, rédaction déterministe des détails structurés,
chaîne SHA-256 vérifiable via AuditStore (réutilisé tel quel, non modifié).
"""
from __future__ import annotations

import json

from schemas.audit import AuditEventType, RedactionStatus
from services.audit_service import AuditService
from services.audit_store import AuditStore


class TestRecordBasics:
    def test_record_persists_event_in_store(self):
        service = AuditService()
        event = service.record(
            case_id="CLM-0001",
            event_type=AuditEventType.CLAIM_STARTED,
            actor="intake_safety_agent",
            outcome="Dossier accepté",
        )
        assert event.case_id == "CLM-0001"
        assert event.model_name is None
        assert event.prompt_version is None
        stored = service.store.read_by_case_id("CLM-0001")
        assert stored == (event,)

    def test_shared_store_can_be_injected(self):
        store = AuditStore()
        service = AuditService(store=store)
        service.record(
            case_id="CLM-0001",
            event_type=AuditEventType.CLAIM_STARTED,
            actor="intake_safety_agent",
            outcome="Dossier accepté",
        )
        assert len(store) == 1


class TestChainIntegrity:
    def test_chain_is_verifiable_across_multiple_events(self):
        service = AuditService()
        service.record(
            case_id="CLM-0001",
            event_type=AuditEventType.CLAIM_STARTED,
            actor="intake_safety_agent",
            outcome="Dossier accepté",
        )
        service.record(
            case_id="CLM-0001",
            event_type=AuditEventType.AGENT_CALLED,
            actor="document_understanding_agent",
            outcome="Extraction terminée",
        )
        report = service.verify_claim_integrity("CLM-0001")
        assert report.intact
        assert report.event_count == 2

    def test_export_to_json_is_readable_and_intact(self):
        service = AuditService()
        service.record(
            case_id="CLM-0001",
            event_type=AuditEventType.FINAL_REPORT,
            actor="autonomous_decision_agent",
            outcome="APPROVE",
        )
        export = service.export_for_auditor("CLM-0001")
        assert export.chain_intact
        assert export.event_count == 1

        payload = service.export_to_json("CLM-0001")
        parsed = json.loads(payload)
        assert parsed["chain_intact"] is True

    def test_export_to_jsonl_has_one_line_per_event_plus_header(self):
        service = AuditService()
        service.record(
            case_id="CLM-0001",
            event_type=AuditEventType.CLAIM_STARTED,
            actor="intake_safety_agent",
            outcome="Dossier accepté",
        )
        service.record(
            case_id="CLM-0001",
            event_type=AuditEventType.FINAL_REPORT,
            actor="autonomous_decision_agent",
            outcome="APPROVE",
        )
        jsonl = service.export_to_jsonl("CLM-0001")
        lines = [line for line in jsonl.strip().split("\n") if line]
        assert len(lines) == 3  # 1 en-tête + 2 événements


class TestDeterministicRedaction:
    def test_details_are_redacted_before_being_stored(self):
        service = AuditService()
        event = service.record(
            case_id="CLM-0001",
            event_type=AuditEventType.ERROR,
            actor="document_understanding_agent",
            outcome="Erreur d'extraction",
            details={"full_text": "Contenu OCR complet ne devant jamais apparaître ici" * 5},
        )
        assert "Contenu OCR complet" not in event.outcome
        assert event.redaction_status in (
            RedactionStatus.PARTIALLY_REDACTED,
            RedactionStatus.FULLY_REDACTED,
        )

    def test_short_clean_details_are_not_redacted(self):
        service = AuditService()
        event = service.record(
            case_id="CLM-0001",
            event_type=AuditEventType.AGENT_CALLED,
            actor="medical_risk_agent",
            outcome="Codification terminée",
            details={"risk_level": "LOW", "signal_count": 0},
        )
        assert event.redaction_status == RedactionStatus.NOT_REDACTED
        assert "risk_level" in event.outcome

    def test_secret_in_details_is_dropped(self):
        service = AuditService()
        event = service.record(
            case_id="CLM-0001",
            event_type=AuditEventType.ERROR,
            actor="eligibility_agent",
            outcome="Erreur technique",
            details={"api_key": "sk-should-never-leak", "case_id": "CLM-0001"},
        )
        assert "sk-should-never-leak" not in event.outcome

    def test_outcome_never_exceeds_schema_max_length(self):
        service = AuditService()
        huge_list = {"items": [f"item-{i}" for i in range(1000)]}
        event = service.record(
            case_id="CLM-0001",
            event_type=AuditEventType.AGENT_CALLED,
            actor="medical_risk_agent",
            outcome="Traitement",
            details=huge_list,
        )
        assert len(event.outcome) <= 2000


class TestNoLlmDependency:
    """Garantie structurelle : ce service n'importe jamais de module LLM."""

    def test_module_source_has_no_llm_reference(self):
        import services.audit_service as mod

        with open(mod.__file__, encoding="utf-8") as f:
            content = f.read()
        assert "import llm" not in content
        assert "langgraph" not in content
        assert "ChatOllama" not in content
        assert "with_structured_output" not in content
        assert "create_react_agent" not in content
