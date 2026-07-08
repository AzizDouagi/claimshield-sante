"""Tests unitaires du journal d'audit append-only — services/audit_store.py.

Couvre : ajout normal, refus de toute tentative de modification (event_id
réutilisé, chaîne rompue), absence de méthode update/delete, isolation des
copies (mutation de l'original ou de la lecture sans effet sur le
journal), lecture par case_id et export auditeur (mono-dossier et global).

Complété par : calcul automatique de previous_hash/event_hash
(``record_event``, chaîne réellement dérivée du contenu canonique) et
vérification d'intégrité complète d'un dossier (``verify_claim_integrity``
— rupture de chaîne, contenu falsifié, ordre chronologique invalide). Ces
scénarios de corruption ne peuvent être simulés qu'en manipulant
directement l'attribut privé ``_events_by_case`` : c'est volontaire —
``append_event``/``record_event`` refusent déjà toute entrée invalide, la
vérification d'intégrité existe précisément en défense en profondeur pour
le cas où quelque chose contournerait un jour ce garde-fou (bug, future
persistance disque).
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from schemas.audit import AuditEvent, AuditEventType, RedactionStatus, compute_event_hash
from services.audit_store import (
    EXPORT_FORMAT_VERSION,
    AuditStore,
    AuditStoreError,
    AuditorExport,
    ClaimIntegrityReport,
    IntegrityIssueType,
)


def _hash(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def _event(**overrides) -> AuditEvent:
    payload = {
        "case_id": "CLM-0001",
        "event_id": str(uuid4()),
        "event_type": AuditEventType.CLAIM_STARTED,
        "actor": "audit_agent",
        "outcome": "success",
        "previous_hash": None,
        "event_hash": _hash(str(uuid4())),
        "redaction_status": RedactionStatus.NOT_REDACTED,
    }
    payload.update(overrides)
    return AuditEvent(**payload)


# ── Ajout normal et chaîne ────────────────────────────────────────────────────


def test_append_event_puis_lecture():
    store = AuditStore()
    event = _event()
    store.append_event(event)

    read = store.read_by_case_id("CLM-0001")
    assert len(read) == 1
    assert read[0].event_id == event.event_id
    assert len(store) == 1


def test_append_event_accepte_la_prolongation_valide_de_la_chaine():
    store = AuditStore()
    h1 = _hash("e1")
    h2 = _hash("e2")
    e1 = _event(event_hash=h1, previous_hash=None)
    e2 = _event(event_hash=h2, previous_hash=h1)

    store.append_event(e1)
    store.append_event(e2)

    read = store.read_by_case_id("CLM-0001")
    assert [e.event_hash for e in read] == [h1, h2]


def test_deux_dossiers_ont_des_chaines_independantes():
    store = AuditStore()
    e1 = _event(case_id="CLM-0001", previous_hash=None)
    e2 = _event(case_id="CLM-0002", previous_hash=None)

    store.append_event(e1)
    store.append_event(e2)  # previous_hash=None deux fois : OK, dossiers distincts

    assert len(store.read_by_case_id("CLM-0001")) == 1
    assert len(store.read_by_case_id("CLM-0002")) == 1


# ── Refus de modification ────────────────────────────────────────────────────


def test_append_event_refuse_un_event_id_deja_present():
    store = AuditStore()
    event = _event()
    store.append_event(event)

    with pytest.raises(AuditStoreError) as exc_info:
        store.append_event(event)

    assert exc_info.value.structured.code == "AUDIT_EVENT_ALREADY_EXISTS"
    assert len(store) == 1  # aucun doublon accepté


def test_append_event_refuse_un_event_id_deja_present_meme_avec_contenu_different():
    store = AuditStore()
    event_id = str(uuid4())
    original = _event(event_id=event_id, outcome="success")
    store.append_event(original)

    tampered = _event(event_id=event_id, outcome="tampered", previous_hash=original.event_hash)
    with pytest.raises(AuditStoreError):
        store.append_event(tampered)

    stored = store.read_by_case_id("CLM-0001")
    assert stored[0].outcome == "success"  # jamais écrasé


def test_append_event_refuse_une_chaine_rompue_previous_hash_incorrect():
    store = AuditStore()
    h1 = _hash("e1")
    store.append_event(_event(event_hash=h1, previous_hash=None))

    wrong = _event(event_hash=_hash("e2"), previous_hash=_hash("not-h1"))
    with pytest.raises(AuditStoreError) as exc_info:
        store.append_event(wrong)

    assert exc_info.value.structured.code == "AUDIT_CHAIN_BROKEN"
    assert len(store.read_by_case_id("CLM-0001")) == 1  # l'événement invalide n'est jamais stocké


def test_append_event_refuse_previous_hash_non_none_sur_le_premier_evenement():
    store = AuditStore()
    first = _event(previous_hash=_hash("ghost-parent"))
    with pytest.raises(AuditStoreError):
        store.append_event(first)


def test_aucune_methode_update_ou_delete_n_existe():
    store = AuditStore()
    for forbidden in ("update_event", "delete_event", "remove_event", "clear", "edit_event"):
        assert not hasattr(store, forbidden), f"{forbidden} ne devrait pas exister sur AuditStore"


# ── Isolation des copies (aucune mutation possible du journal) ───────────────


def test_muter_l_evenement_original_apres_ajout_est_sans_effet():
    store = AuditStore()
    event = _event()
    store.append_event(event)

    event.outcome = "tampered"  # mutation de l'objet appelant, pas du journal

    stored = store.read_by_case_id("CLM-0001")
    assert stored[0].outcome == "success"


def test_muter_un_evenement_lu_est_sans_effet_sur_le_journal():
    store = AuditStore()
    store.append_event(_event())

    read_once = store.read_by_case_id("CLM-0001")
    read_once[0].outcome = "tampered"

    read_again = store.read_by_case_id("CLM-0001")
    assert read_again[0].outcome == "success"


# ── Lecture par case_id ───────────────────────────────────────────────────────


def test_read_by_case_id_inconnu_retourne_un_tuple_vide():
    store = AuditStore()
    assert store.read_by_case_id("CLM-9999") == ()


def test_read_by_case_id_respecte_l_ordre_d_ajout():
    store = AuditStore()
    h1, h2, h3 = _hash("a"), _hash("b"), _hash("c")
    store.append_event(_event(event_hash=h1, previous_hash=None))
    store.append_event(_event(event_hash=h2, previous_hash=h1))
    store.append_event(_event(event_hash=h3, previous_hash=h2))

    read = store.read_by_case_id("CLM-0001")
    assert [e.event_hash for e in read] == [h1, h2, h3]


# ── Export auditeur ───────────────────────────────────────────────────────────


def test_export_for_auditor_un_seul_dossier():
    store = AuditStore()
    event = _event()
    store.append_event(event)

    export = store.export_for_auditor(case_id="CLM-0001")

    assert isinstance(export, AuditorExport)
    assert export.case_id == "CLM-0001"
    assert export.event_count == 1
    assert export.chain_intact is True
    assert export.broken_at_event_id is None
    assert export.events[0].event_id == event.event_id


def test_export_for_auditor_dossier_inconnu_est_vide_et_intact():
    store = AuditStore()
    export = store.export_for_auditor(case_id="CLM-9999")

    assert export.event_count == 0
    assert export.chain_intact is True
    assert export.events == ()


def test_export_for_auditor_global_couvre_tous_les_dossiers():
    store = AuditStore()
    store.append_event(_event(case_id="CLM-0001", previous_hash=None))
    store.append_event(_event(case_id="CLM-0002", previous_hash=None))

    export = store.export_for_auditor()

    assert export.case_id is None
    assert export.event_count == 2
    assert export.chain_intact is True


def test_export_for_auditor_est_recalcule_a_chaque_appel():
    store = AuditStore()
    h1 = _hash("first")
    store.append_event(_event(event_hash=h1, previous_hash=None))

    export1 = store.export_for_auditor(case_id="CLM-0001")
    assert export1.event_count == 1

    store.append_event(_event(event_hash=_hash("second"), previous_hash=h1))
    export2 = store.export_for_auditor(case_id="CLM-0001")

    assert export2.event_count == 2
    assert export1.event_count == 1  # le premier export reste un instantané figé


def test_export_for_auditor_est_lui_meme_immuable_en_extra():
    store = AuditStore()
    store.append_event(_event())
    export = store.export_for_auditor(case_id="CLM-0001")

    with pytest.raises(Exception):
        AuditorExport(**{**export.model_dump(), "extra_field": "not_allowed"})


# ── record_event : calcul automatique de previous_hash / event_hash ─────────


def test_record_event_calcule_un_previous_hash_none_pour_le_premier_evenement():
    store = AuditStore()
    event = store.record_event(
        case_id="CLM-0001",
        event_type=AuditEventType.CLAIM_STARTED,
        actor="audit_agent",
        outcome="started",
        redaction_status=RedactionStatus.NOT_REDACTED,
    )
    assert event.previous_hash is None
    assert len(event.event_hash) == 64


def test_record_event_chaine_previous_hash_sur_le_event_hash_precedent():
    store = AuditStore()
    e1 = store.record_event(
        case_id="CLM-0001",
        event_type=AuditEventType.CLAIM_STARTED,
        actor="audit_agent",
        outcome="started",
        redaction_status=RedactionStatus.NOT_REDACTED,
    )
    e2 = store.record_event(
        case_id="CLM-0001",
        event_type=AuditEventType.AGENT_CALLED,
        actor="audit_agent",
        outcome="called",
        redaction_status=RedactionStatus.NOT_REDACTED,
    )
    assert e2.previous_hash == e1.event_hash
    assert e1.event_hash != e2.event_hash


def test_record_event_hash_couvre_le_contenu_reel_pas_seulement_le_format():
    store = AuditStore()
    e1 = store.record_event(
        case_id="CLM-0001",
        event_type=AuditEventType.ERROR,
        actor="audit_agent",
        outcome="succes",
        redaction_status=RedactionStatus.NOT_REDACTED,
    )
    store2 = AuditStore()
    e2 = store2.record_event(
        case_id="CLM-0001",
        event_type=AuditEventType.ERROR,
        actor="audit_agent",
        outcome="echec",  # seul le contenu change
        redaction_status=RedactionStatus.NOT_REDACTED,
    )
    assert e1.event_hash != e2.event_hash  # un contenu différent -> un hash différent


def test_record_event_deux_dossiers_distincts_chainent_independamment():
    store = AuditStore()
    e1 = store.record_event(
        case_id="CLM-0001",
        event_type=AuditEventType.CLAIM_STARTED,
        actor="a",
        outcome="ok",
        redaction_status=RedactionStatus.NOT_REDACTED,
    )
    e2 = store.record_event(
        case_id="CLM-0002",
        event_type=AuditEventType.CLAIM_STARTED,
        actor="a",
        outcome="ok",
        redaction_status=RedactionStatus.NOT_REDACTED,
    )
    assert e1.previous_hash is None
    assert e2.previous_hash is None  # nouveau dossier, jamais lié au premier


def test_record_event_utilise_bien_verify_claim_integrity_sans_anomalie():
    store = AuditStore()
    for outcome in ("started", "called", "final"):
        store.record_event(
            case_id="CLM-0001",
            event_type=AuditEventType.CLAIM_STARTED,
            actor="audit_agent",
            outcome=outcome,
            redaction_status=RedactionStatus.NOT_REDACTED,
        )
    report = store.verify_claim_integrity("CLM-0001")
    assert isinstance(report, ClaimIntegrityReport)
    assert report.intact is True
    assert report.issues == ()
    assert report.event_count == 3


# ── verify_claim_integrity : dossier sain et dossier inconnu ────────────────


def test_verify_claim_integrity_dossier_inconnu_est_intact_et_vide():
    store = AuditStore()
    report = store.verify_claim_integrity("CLM-9999")
    assert report.intact is True
    assert report.event_count == 0
    assert report.issues == ()


# ── verify_claim_integrity : rupture de chaîne ──────────────────────────────


def test_verify_claim_integrity_detecte_une_rupture_de_chaine():
    store = AuditStore()
    store.record_event(
        case_id="CLM-0001",
        event_type=AuditEventType.CLAIM_STARTED,
        actor="a",
        outcome="ok",
        redaction_status=RedactionStatus.NOT_REDACTED,
    )
    e2 = store.record_event(
        case_id="CLM-0001",
        event_type=AuditEventType.AGENT_CALLED,
        actor="a",
        outcome="ok",
        redaction_status=RedactionStatus.NOT_REDACTED,
    )

    # Corruption directe du journal (impossible via l'API publique, qui
    # refuse déjà toute chaîne rompue à l'ajout) : simule un bug ou une
    # future persistance qui aurait laissé passer un maillon invalide.
    corrupted = e2.model_copy(update={"previous_hash": _hash("not-e1")})
    store._events_by_case["CLM-0001"][1] = corrupted

    report = store.verify_claim_integrity("CLM-0001")

    assert report.intact is False
    assert any(
        issue.issue_type is IntegrityIssueType.CHAIN_BROKEN and issue.event_id == e2.event_id
        for issue in report.issues
    )


# ── verify_claim_integrity : contenu falsifié ───────────────────────────────


def test_verify_claim_integrity_detecte_un_contenu_falsifie():
    store = AuditStore()
    e1 = store.record_event(
        case_id="CLM-0001",
        event_type=AuditEventType.CLAIM_STARTED,
        actor="a",
        outcome="succes",
        redaction_status=RedactionStatus.NOT_REDACTED,
    )
    assert compute_event_hash(e1) == e1.event_hash  # contenu et hash cohérents avant corruption

    # Le contenu change (outcome) mais event_hash reste celui de l'ancien
    # contenu — exactement ce qu'un contenu falsifié après coup produirait.
    tampered = e1.model_copy(update={"outcome": "echec"})
    store._events_by_case["CLM-0001"][0] = tampered

    report = store.verify_claim_integrity("CLM-0001")

    assert report.intact is False
    assert any(
        issue.issue_type is IntegrityIssueType.CONTENT_TAMPERED and issue.event_id == e1.event_id
        for issue in report.issues
    )


# ── verify_claim_integrity : ordre chronologique invalide ───────────────────


def test_verify_claim_integrity_detecte_un_ordre_invalide():
    store = AuditStore()
    e1 = store.record_event(
        case_id="CLM-0001",
        event_type=AuditEventType.CLAIM_STARTED,
        actor="a",
        outcome="ok",
        redaction_status=RedactionStatus.NOT_REDACTED,
    )
    e2 = store.record_event(
        case_id="CLM-0001",
        event_type=AuditEventType.AGENT_CALLED,
        actor="a",
        outcome="ok",
        redaction_status=RedactionStatus.NOT_REDACTED,
    )

    # e2 est repositionné avec un timestamp antérieur à e1, tout en gardant
    # une chaîne previous_hash/event_hash structurellement valide — seule
    # la vérification d'ordre chronologique doit signaler l'anomalie.
    reordered = e2.model_copy(update={"timestamp": e1.timestamp - timedelta(seconds=5)})
    store._events_by_case["CLM-0001"][1] = reordered

    report = store.verify_claim_integrity("CLM-0001")

    assert report.intact is False
    assert any(
        issue.issue_type is IntegrityIssueType.INVALID_ORDER and issue.event_id == e2.event_id
        for issue in report.issues
    )
    # Le chaînage previous_hash/event_hash lui, reste intact sur ce scénario.
    assert not any(issue.issue_type is IntegrityIssueType.CHAIN_BROKEN for issue in report.issues)


def test_verify_claim_integrity_peut_detecter_plusieurs_anomalies_a_la_fois():
    store = AuditStore()
    store.record_event(
        case_id="CLM-0001",
        event_type=AuditEventType.CLAIM_STARTED,
        actor="a",
        outcome="ok",
        redaction_status=RedactionStatus.NOT_REDACTED,
    )
    e2 = store.record_event(
        case_id="CLM-0001",
        event_type=AuditEventType.AGENT_CALLED,
        actor="a",
        outcome="ok",
        redaction_status=RedactionStatus.NOT_REDACTED,
    )

    # Un seul événement corrompu sur les deux axes : chaîne ET contenu.
    corrupted = e2.model_copy(
        update={"previous_hash": _hash("wrong"), "outcome": "modifie"}
    )
    store._events_by_case["CLM-0001"][1] = corrupted

    report = store.verify_claim_integrity("CLM-0001")

    issue_types = {issue.issue_type for issue in report.issues}
    assert IntegrityIssueType.CHAIN_BROKEN in issue_types
    assert IntegrityIssueType.CONTENT_TAMPERED in issue_types


# ── export_for_auditor : anomalies (issues) ─────────────────────────────────


def test_export_for_auditor_inclut_les_anomalies_detectees():
    store = AuditStore()
    e1 = store.record_event(
        case_id="CLM-0001",
        event_type=AuditEventType.CLAIM_STARTED,
        actor="a",
        outcome="succes",
        redaction_status=RedactionStatus.NOT_REDACTED,
    )
    tampered = e1.model_copy(update={"outcome": "falsifie"})
    store._events_by_case["CLM-0001"][0] = tampered

    export = store.export_for_auditor(case_id="CLM-0001")

    # Seul le contenu est falsifié (pas la chaîne) : chain_intact reste vrai,
    # l'anomalie n'en est pas moins bien remontée dans issues.
    assert export.chain_intact is True
    assert any(
        issue.issue_type is IntegrityIssueType.CONTENT_TAMPERED and issue.event_id == e1.event_id
        for issue in export.issues
    )


def test_export_for_auditor_sans_anomalie_a_les_issues_vides():
    store = AuditStore()
    store.record_event(
        case_id="CLM-0001",
        event_type=AuditEventType.CLAIM_STARTED,
        actor="a",
        outcome="ok",
        redaction_status=RedactionStatus.NOT_REDACTED,
    )

    export = store.export_for_auditor(case_id="CLM-0001")

    assert export.issues == ()
    assert export.chain_intact is True
    assert export.broken_at_event_id is None


def test_export_for_auditor_broken_at_event_id_reflete_la_chaine_rompue():
    store = AuditStore()
    store.record_event(
        case_id="CLM-0001",
        event_type=AuditEventType.CLAIM_STARTED,
        actor="a",
        outcome="ok",
        redaction_status=RedactionStatus.NOT_REDACTED,
    )
    e2 = store.record_event(
        case_id="CLM-0001",
        event_type=AuditEventType.AGENT_CALLED,
        actor="a",
        outcome="ok",
        redaction_status=RedactionStatus.NOT_REDACTED,
    )
    corrupted = e2.model_copy(update={"previous_hash": _hash("not-e1")})
    store._events_by_case["CLM-0001"][1] = corrupted

    export = store.export_for_auditor(case_id="CLM-0001")

    assert export.chain_intact is False
    assert export.broken_at_event_id == e2.event_id


# ── export_to_json / export_to_jsonl — export auditeur lisible et vérifiable ─


def _sha256_hex(text: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", text))


def test_export_to_json_est_un_json_valide_et_filtrable_par_case_id():
    store = AuditStore()
    store.record_event(
        case_id="CLM-0001",
        event_type=AuditEventType.CLAIM_STARTED,
        actor="claim_intake_agent",
        outcome="ALLOW",
        redaction_status=RedactionStatus.NOT_REDACTED,
        model_name="gemma4:latest",
        prompt_version="1.0.0",
    )
    store.record_event(
        case_id="CLM-0002",
        event_type=AuditEventType.SECURITY_DECISION,
        actor="security_gate_agent",
        outcome="BLOCK",
        redaction_status=RedactionStatus.PARTIALLY_REDACTED,
    )

    raw = store.export_to_json(case_id="CLM-0001")
    parsed = json.loads(raw)  # lisible : un consommateur externe peut le reparser sans effort

    assert parsed["export_format_version"] == EXPORT_FORMAT_VERSION
    assert parsed["case_id"] == "CLM-0001"
    assert parsed["event_count"] == 1
    assert len(parsed["events"]) == 1
    assert parsed["events"][0]["case_id"] == "CLM-0001"
    assert parsed["events"][0]["outcome"] == "ALLOW"
    # Le dossier CLM-0002 est bien exclu par le filtre case_id.
    assert all(event["case_id"] == "CLM-0001" for event in parsed["events"])


def test_export_to_json_inclut_hash_chain_versions_et_anomalies():
    store = AuditStore()
    e1 = store.record_event(
        case_id="CLM-0003",
        event_type=AuditEventType.CLAIM_STARTED,
        actor="claim_intake_agent",
        outcome="ALLOW",
        redaction_status=RedactionStatus.NOT_REDACTED,
        model_name="gemma4:latest",
        prompt_version="1.2.0",
    )
    store.record_event(
        case_id="CLM-0003",
        event_type=AuditEventType.SECURITY_DECISION,
        actor="security_gate_agent",
        outcome="BLOCK",
        redaction_status=RedactionStatus.PARTIALLY_REDACTED,
        model_name="gemma4:latest",
        prompt_version="1.0.0",
    )

    raw = store.export_to_json(case_id="CLM-0003")
    parsed = json.loads(raw)

    events = parsed["events"]
    assert len(events) == 2
    # Hash-chain : previous_hash du 2e événement == event_hash du 1er, et
    # chaque event_hash est réellement une empreinte SHA-256 hexadécimale.
    assert events[1]["previous_hash"] == events[0]["event_hash"]
    assert _sha256_hex(events[0]["event_hash"])
    assert _sha256_hex(events[1]["event_hash"])
    # Versions : modèle et version de prompt tracés par événement.
    assert events[0]["model_name"] == "gemma4:latest"
    assert events[0]["prompt_version"] == "1.2.0"
    assert events[1]["prompt_version"] == "1.0.0"
    # Anomalies : absentes ici, journal intact.
    assert parsed["issues"] == []
    assert parsed["chain_intact"] is True

    # Vérifiable : recalculer le hash à partir du contenu doit retomber sur
    # l'event_hash exporté (même garantie que verify_claim_integrity).
    reconstructed = AuditEvent(**{k: v for k, v in events[0].items() if k != "type"})
    assert compute_event_hash(reconstructed) == events[0]["event_hash"]
    assert reconstructed == e1


def test_export_to_json_expose_les_anomalies_detectees():
    store = AuditStore()
    e1 = store.record_event(
        case_id="CLM-0004",
        event_type=AuditEventType.CLAIM_STARTED,
        actor="a",
        outcome="succes",
        redaction_status=RedactionStatus.NOT_REDACTED,
    )
    tampered = e1.model_copy(update={"outcome": "falsifie"})
    store._events_by_case["CLM-0004"][0] = tampered

    raw = store.export_to_json(case_id="CLM-0004")
    parsed = json.loads(raw)

    assert parsed["chain_intact"] is True  # seul le contenu est falsifié, pas la chaîne
    assert len(parsed["issues"]) == 1
    assert parsed["issues"][0]["issue_type"] == IntegrityIssueType.CONTENT_TAMPERED.value
    assert parsed["issues"][0]["event_id"] == e1.event_id


def test_export_to_jsonl_a_une_ligne_d_entete_et_une_ligne_par_evenement():
    store = AuditStore()
    store.record_event(
        case_id="CLM-0005",
        event_type=AuditEventType.CLAIM_STARTED,
        actor="claim_intake_agent",
        outcome="ALLOW",
        redaction_status=RedactionStatus.NOT_REDACTED,
    )
    store.record_event(
        case_id="CLM-0005",
        event_type=AuditEventType.SECURITY_DECISION,
        actor="security_gate_agent",
        outcome="BLOCK",
        redaction_status=RedactionStatus.PARTIALLY_REDACTED,
    )

    raw = store.export_to_jsonl(case_id="CLM-0005")
    lines = [line for line in raw.splitlines() if line]

    assert len(lines) == 3  # 1 en-tête + 2 événements

    header = json.loads(lines[0])
    assert header["type"] == "export_summary"
    assert header["export_format_version"] == EXPORT_FORMAT_VERSION
    assert header["case_id"] == "CLM-0005"
    assert header["event_count"] == 2
    assert header["chain_intact"] is True
    assert header["anomalies"] == []

    event_lines = [json.loads(line) for line in lines[1:]]
    assert all(entry["type"] == "event" for entry in event_lines)
    assert {entry["case_id"] for entry in event_lines} == {"CLM-0005"}
    assert event_lines[0]["actor"] == "claim_intake_agent"
    assert event_lines[1]["actor"] == "security_gate_agent"


def test_export_to_jsonl_chaque_ligne_est_un_json_independant_et_verifiable():
    store = AuditStore()
    store.record_event(
        case_id="CLM-0006",
        event_type=AuditEventType.CLAIM_STARTED,
        actor="a",
        outcome="ok",
        redaction_status=RedactionStatus.NOT_REDACTED,
    )

    raw = store.export_to_jsonl(case_id="CLM-0006")
    lines = [line for line in raw.splitlines() if line]

    # Chaque ligne se reparse indépendamment (propriété clé du JSON Lines) —
    # une ligne corrompue n'empêche jamais de lire les autres.
    for line in lines:
        parsed_line = json.loads(line)
        assert isinstance(parsed_line, dict)

    event_entry = json.loads(lines[1])
    reconstructed = AuditEvent(**{k: v for k, v in event_entry.items() if k != "type"})
    assert compute_event_hash(reconstructed) == event_entry["event_hash"]


def test_export_to_jsonl_anomalie_visible_dans_la_ligne_d_entete():
    store = AuditStore()
    e1 = store.record_event(
        case_id="CLM-0007",
        event_type=AuditEventType.CLAIM_STARTED,
        actor="a",
        outcome="succes",
        redaction_status=RedactionStatus.NOT_REDACTED,
    )
    tampered = e1.model_copy(update={"outcome": "falsifie"})
    store._events_by_case["CLM-0007"][0] = tampered

    raw = store.export_to_jsonl(case_id="CLM-0007")
    header = json.loads(raw.splitlines()[0])

    assert len(header["anomalies"]) == 1
    assert header["anomalies"][0]["issue_type"] == IntegrityIssueType.CONTENT_TAMPERED.value


def test_export_global_sans_case_id_couvre_tous_les_dossiers_en_json_et_jsonl():
    store = AuditStore()
    store.record_event(
        case_id="CLM-0008",
        event_type=AuditEventType.CLAIM_STARTED,
        actor="a",
        outcome="ok",
        redaction_status=RedactionStatus.NOT_REDACTED,
    )
    store.record_event(
        case_id="CLM-0009",
        event_type=AuditEventType.CLAIM_STARTED,
        actor="a",
        outcome="ok",
        redaction_status=RedactionStatus.NOT_REDACTED,
    )

    json_parsed = json.loads(store.export_to_json())
    assert json_parsed["case_id"] is None
    assert json_parsed["event_count"] == 2

    jsonl_lines = [line for line in store.export_to_jsonl().splitlines() if line]
    assert len(jsonl_lines) == 3  # 1 en-tête + 2 événements
    header = json.loads(jsonl_lines[0])
    assert header["case_id"] is None
    assert header["event_count"] == 2


# ── Ne contient aucune donnée excessive ──────────────────────────────────────


def test_outcome_excessivement_long_est_rejete_par_le_schema():
    """Aucun événement ne peut jamais porter un contenu volumineux (document
    entier, texte OCR complet...) dans ``outcome`` — l'export ne fait que
    sérialiser des événements déjà bornés à la construction."""
    with pytest.raises(ValidationError):
        AuditEvent(
            case_id="CLM-0010",
            event_id=str(uuid4()),
            event_type=AuditEventType.CLAIM_STARTED,
            actor="a",
            outcome="X" * 2001,
            previous_hash=None,
            event_hash=_hash("excessive"),
            redaction_status=RedactionStatus.NOT_REDACTED,
        )


def test_export_to_json_ne_contient_aucune_cle_hors_schema_valide():
    store = AuditStore()
    store.record_event(
        case_id="CLM-0011",
        event_type=AuditEventType.CLAIM_STARTED,
        actor="claim_intake_agent",
        outcome="ALLOW",
        redaction_status=RedactionStatus.NOT_REDACTED,
    )

    parsed = json.loads(store.export_to_json(case_id="CLM-0011"))
    event = parsed["events"][0]

    # Uniquement les champs de schemas.audit.AuditEvent (extra="forbid" à la
    # construction) — jamais un champ additionnel injecté par l'export.
    assert set(event.keys()) == set(AuditEvent.model_fields.keys())
    assert all(len(str(value)) <= 2000 for value in event.values())
