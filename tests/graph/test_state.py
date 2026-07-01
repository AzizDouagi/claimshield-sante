"""Contrat du ClaimState utilisé par le graphe LangGraph."""
from __future__ import annotations

import json
import operator
from datetime import UTC, datetime
from typing import Annotated, get_args, get_origin, get_type_hints

import pytest

from graph.checkpoints import (
    get_checkpointer,
    make_thread_config,
    serialize_checkpoint_state,
    validate_checkpoint_state,
)
from schemas.domain import (
    DataClassification,
    DocumentType,
    ExtractionStatus,
    IntakeStatus,
    OcrSource,
    Recommendation,
    SecurityDecision,
    VerificationStatus,
)
from schemas.results import (
    AuditEvent,
    AuditResult,
    CaseReviewerResult,
    ClaimIntakeResult,
    ClaimManifest,
    ClinicalConsistencyResult,
    CoverageResult,
    DocumentOcrResult,
    FhirValidatorResult,
    FraudDetectionResult,
    IdentityCoverageResult,
    IdentityResult,
    MedicalCodingResult,
    PrivacyResult,
    SecurityGateResult,
)
from state.claim_state import ClaimState, validate_claim_state


def _reducer_for(field_name: str):
    annotation = get_type_hints(ClaimState, include_extras=True)[field_name]
    if get_origin(annotation) is Annotated:
        return get_args(annotation)[1]
    return None


def _merge_like_langgraph(*updates: dict) -> dict:
    merged: dict = {}
    for update in updates:
        for key, value in update.items():
            reducer = _reducer_for(key) if key in ClaimState.__annotations__ else None
            if reducer is not None and key in merged:
                merged[key] = reducer(merged[key], value)
            else:
                merged[key] = value
    return merged


def _minimal_state() -> dict:
    return {
        "case_id": "CLM-0001",
        "schema_version": "1.0.0",
        "current_step": "claim_intake",
        "completed_steps": [],
    }


def _agent_results() -> dict:
    case_id = "CLM-0001"
    return {
        "intake_result": ClaimIntakeResult(
            claim_id=case_id,
            status=IntakeStatus.ACCEPTED,
            manifest=ClaimManifest(
                claim_id=case_id,
                file_count=1,
                total_size_bytes=1234,
                status=IntakeStatus.ACCEPTED,
            ),
            accepted_count=1,
            quarantined_count=0,
        ),
        "security_result": SecurityGateResult(
            claim_id=case_id,
            decision=SecurityDecision.ALLOW,
            reasons=["Aucune menace détectée."],
        ),
        "privacy_result": PrivacyResult(
            case_id=case_id,
            status=VerificationStatus.PASS,
            data_classification=DataClassification.SYNTHETIC_TEST_DATA,
            contains_real_personal_data=False,
        ),
        "identity_coverage_result": IdentityCoverageResult(
            case_id=case_id,
            identity=IdentityResult(status=VerificationStatus.PASS),
            coverage=CoverageResult(status=VerificationStatus.PASS),
        ),
        "fhir_result": FhirValidatorResult(
            case_id=case_id,
            status=VerificationStatus.PASS,
            bundle_expected=True,
        ),
        "ocr_result": DocumentOcrResult(
            claim_id=case_id,
            file_path="incoming/CLM-0001/facture.pdf",
            sha256="a" * 64,
            mime_type="application/pdf",
            extraction_status=ExtractionStatus.SUCCESS,
            status=VerificationStatus.PASS,
            document_type=DocumentType.INVOICE,
            ocr_source=OcrSource.PDF_TEXT,
            artifact_id="ocr-artifact-1",
            artifact_path="artifacts/document_ocr/CLM-0001/facture.json",
        ),
        "coding_result": MedicalCodingResult(case_id=case_id, status=VerificationStatus.PASS),
        "clinical_result": ClinicalConsistencyResult(case_id=case_id, status=VerificationStatus.PASS),
        "fraud_result": FraudDetectionResult(case_id=case_id, status=VerificationStatus.PASS),
        "review_result": CaseReviewerResult(
            case_id=case_id,
            recommendation=Recommendation.APPROVE,
        ),
        "audit_result": AuditResult(case_id=case_id, status=VerificationStatus.PASS),
    }


def _full_state_with_11_results() -> dict:
    return {
        **_minimal_state(),
        "intake_status": IntakeStatus.ACCEPTED,
        "current_step": "case_review",
        "completed_steps": ["claim_intake", "security_gate"],
        "errors": [],
        "alerts": ["Revue humaine non requise"],
        "audit_trail": [
            AuditEvent(
                event_id="evt-1",
                case_id="CLM-0001",
                actor="claim_intake_agent",
                action="claim_intake",
                outcome="ACCEPTED",
            )
        ],
        "human_decision": None,
        "final_recommendation": Recommendation.APPROVE,
        "final_justification": ["Tous les contrôles minimaux sont passés."],
        **_agent_results(),
    }


def test_state_minimal_valide_est_accepte():
    validate_claim_state(_minimal_state())


def test_state_sans_case_id_est_rejete():
    state = _minimal_state()
    del state["case_id"]
    with pytest.raises(ValueError, match="case_id"):
        validate_claim_state(state)


def test_valeur_inconnue_ou_mal_typee_est_rejetee():
    with pytest.raises(ValueError, match="champs inconnus"):
        validate_claim_state({**_minimal_state(), "champ_inconnu": "x"})

    with pytest.raises(ValueError, match="completed_steps"):
        validate_claim_state({**_minimal_state(), "completed_steps": "claim_intake"})


def test_state_serialise_puis_restaure_en_json():
    state = _full_state_with_11_results()
    validate_claim_state(state)

    canonical = serialize_checkpoint_state(state)
    restored = json.loads(json.dumps(canonical, ensure_ascii=False))

    validate_claim_state(restored)
    assert restored == canonical


def test_deux_resultats_agents_fusionnent_sans_perte():
    intake_update = {
        "completed_steps": ["claim_intake"],
        "intake_result": _agent_results()["intake_result"],
    }
    security_update = {
        "completed_steps": ["security_gate"],
        "security_result": _agent_results()["security_result"],
    }

    merged = _merge_like_langgraph(intake_update, security_update)

    assert merged["completed_steps"] == ["claim_intake", "security_gate"]
    assert isinstance(merged["intake_result"], ClaimIntakeResult)
    assert isinstance(merged["security_result"], SecurityGateResult)


def test_errors_et_alerts_restent_deux_listes_distinctes():
    state = {
        **_minimal_state(),
        "errors": ["[security_gate] blocage"],
        "alerts": ["Document optionnel absent"],
    }
    validate_claim_state(state)

    assert state["errors"] == ["[security_gate] blocage"]
    assert state["alerts"] == ["Document optionnel absent"]
    assert state["errors"] is not state["alerts"]


def test_historiques_append_only_ne_sont_pas_ecrases():
    audit_a = AuditEvent(
        event_id="evt-a",
        case_id="CLM-0001",
        actor="a",
        action="a",
        outcome="PASS",
    )
    audit_b = AuditEvent(
        event_id="evt-b",
        case_id="CLM-0001",
        actor="b",
        action="b",
        outcome="PASS",
    )

    merged = _merge_like_langgraph(
        {
            "completed_steps": ["a"],
            "errors": ["err-a"],
            "alerts": ["alert-a"],
            "audit_trail": [audit_a],
        },
        {
            "completed_steps": ["b"],
            "errors": ["err-b"],
            "alerts": ["alert-b"],
            "audit_trail": [audit_b],
        },
    )

    assert _reducer_for("completed_steps") is operator.add
    assert merged["completed_steps"] == ["a", "b"]
    assert merged["errors"] == ["err-a", "err-b"]
    assert merged["alerts"] == ["alert-a", "alert-b"]
    assert merged["audit_trail"] == [audit_a, audit_b]


def test_mutation_invalide_est_refusee():
    with pytest.raises(ValueError, match="current_step"):
        validate_claim_state({**_minimal_state(), "current_step": 42})

    with pytest.raises(ValueError, match="contenu binaire"):
        validate_claim_state({**_minimal_state(), "document_bytes": b"%PDF-1.4"})


def test_aucun_document_brut_secret_ou_texte_ocr_complet_present():
    state = _full_state_with_11_results()
    validate_claim_state(state)
    dumped = json.dumps(serialize_checkpoint_state(state), ensure_ascii=False).lower()

    assert "%pdf" not in dumped
    assert "api_key" not in dumped
    assert "texte ocr complet" not in dumped
    assert "raw_response" not in dumped

    with pytest.raises(ValueError):
        validate_claim_state({**_minimal_state(), "ocr_result": {"full_text": "texte OCR complet"}})


def test_taille_state_reste_raisonnable_avec_11_agents():
    state = _full_state_with_11_results()
    validate_claim_state(state)
    dumped = json.dumps(serialize_checkpoint_state(state), ensure_ascii=False)

    assert len(dumped.encode("utf-8")) < 50_000


def test_checkpoint_sauvegarde_puis_restaure_le_state():
    state = _full_state_with_11_results()
    validate_checkpoint_state(state)
    canonical = serialize_checkpoint_state(state)

    checkpointer = get_checkpointer(backend="memory")
    checkpoint = {
        "v": 1,
        "ts": datetime.now(UTC).isoformat(),
        "id": "checkpoint-state-test",
        "channel_values": {"state": canonical},
        "channel_versions": {"state": "1"},
        "versions_seen": {},
    }
    config = make_thread_config(state["case_id"])
    saved_config = checkpointer.put(config, checkpoint, {"source": "test"}, {"state": "1"})

    restored_checkpoint = checkpointer.get(saved_config)
    assert restored_checkpoint is not None
    restored_state = restored_checkpoint["channel_values"]["state"]

    validate_claim_state(restored_state)
    assert restored_state == canonical


def test_state_restaure_identique_au_state_valide_initial():
    state = _full_state_with_11_results()
    validate_claim_state(state)
    initial = serialize_checkpoint_state(state)

    checkpointer = get_checkpointer(backend="memory")
    typed = checkpointer.serde.dumps_typed(initial)
    restored = checkpointer.serde.loads_typed(typed)

    validate_claim_state(restored)
    assert restored == initial
