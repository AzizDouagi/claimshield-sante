"""Tests de tools/audit_redaction.py — rédaction déterministe des payloads d'audit."""
from __future__ import annotations

from schemas.audit import RedactionStatus
from tools.audit_redaction import (
    MAX_LIST_ITEMS,
    MAX_SHORT_TEXT_LENGTH,
    redact_audit_payload,
)

_VALID_HASH = "a" * 64


class TestSecretLeak:
    """Aucune clé/secret/mot de passe/token ne doit jamais survivre à la rédaction."""

    def test_api_key_field_dropped(self):
        payload = {
            "case_id": "CLM-9001",
            "outcome": "BLOCK",
            "notes": "attention api_key=sk-ABCDEF1234567890 exposée",
        }
        redacted = redact_audit_payload(payload)
        assert "notes" not in redacted
        assert "sk-ABCDEF1234567890" not in str(redacted)
        assert redacted["case_id"] == "CLM-9001"
        assert redacted["outcome"] == "BLOCK"
        assert redacted["redaction_status"] == RedactionStatus.PARTIALLY_REDACTED.value

    def test_api_key_only_field_leaves_only_identifiers_fully_redacted(self):
        payload = {"case_id": "CLM-9006", "notes": "attention api_key=sk-ABCDEF1234567890 exposée"}
        redacted = redact_audit_payload(payload)
        assert "notes" not in redacted
        assert redacted["redaction_status"] == RedactionStatus.FULLY_REDACTED.value

    def test_password_field_dropped(self):
        payload = {"case_id": "CLM-9002", "comment": "password: hunter2 trouvé dans les logs"}
        redacted = redact_audit_payload(payload)
        assert "comment" not in redacted
        assert "hunter2" not in str(redacted)

    def test_bearer_token_dropped(self):
        payload = {"case_id": "CLM-9003", "header": "Authorization: Bearer abc123.def456.ghi789"}
        redacted = redact_audit_payload(payload)
        assert "header" not in redacted
        assert "abc123.def456.ghi789" not in str(redacted)

    def test_secret_nested_in_dict_dropped(self):
        payload = {
            "case_id": "CLM-9004",
            "tool_output": {"env": "API_KEY=super-secret-value", "code": "OK"},
        }
        redacted = redact_audit_payload(payload)
        assert "env" not in redacted["tool_output"]
        assert "super-secret-value" not in str(redacted)
        assert redacted["tool_output"]["code"] == "OK"

    def test_secret_in_list_item_dropped_others_kept(self):
        payload = {
            "case_id": "CLM-9005",
            "reasons": ["Aucune anomalie", "token=abcdef0123456789", "Dossier conforme"],
        }
        redacted = redact_audit_payload(payload)
        assert "token=abcdef0123456789" not in str(redacted)
        assert "Aucune anomalie" in redacted["reasons"]
        assert "Dossier conforme" in redacted["reasons"]
        assert redacted["redaction_status"] == RedactionStatus.PARTIALLY_REDACTED.value


class TestCompleteOcrRemoved:
    """Le texte OCR complet ne doit jamais être transmis, même partiellement."""

    def test_full_text_key_always_dropped(self):
        long_ocr = "Facture médicale ligne par ligne. " * 40
        assert len(long_ocr) > MAX_SHORT_TEXT_LENGTH
        payload = {"case_id": "CLM-9101", "ocr_result": {"full_text": long_ocr, "confidence": 0.9}}
        redacted = redact_audit_payload(payload)
        assert "full_text" not in redacted["ocr_result"]
        assert long_ocr not in str(redacted)
        assert redacted["ocr_result"]["confidence"] == 0.9

    def test_short_ocr_text_still_dropped_by_key_name(self):
        """Même un OCR court reste dangereux par construction : le nom du
        champ suffit à le retirer, indépendamment de sa longueur."""
        payload = {"case_id": "CLM-9102", "ocr_text": "court"}
        redacted = redact_audit_payload(payload)
        assert "ocr_text" not in redacted

    def test_raw_text_and_extracted_text_dropped(self):
        payload = {
            "case_id": "CLM-9103",
            "raw_text": "X" * 10,
            "extracted_text": "Y" * 10,
        }
        redacted = redact_audit_payload(payload)
        assert "raw_text" not in redacted
        assert "extracted_text" not in redacted

    def test_ocr_result_key_dropped_entirely_when_only_content_was_unsafe(self):
        payload = {"case_id": "CLM-9104", "ocr_result": {"full_text": "Z" * 500}}
        redacted = redact_audit_payload(payload)
        assert "ocr_result" not in redacted
        assert redacted["redaction_status"] == RedactionStatus.FULLY_REDACTED.value


class TestCompletePromptRemoved:
    """Le prompt système complet ne doit jamais être transmis."""

    def test_system_prompt_key_always_dropped(self):
        long_prompt = "Tu es l'Audit Agent de ClaimShield Santé. " * 30
        payload = {"case_id": "CLM-9201", "system_prompt": long_prompt}
        redacted = redact_audit_payload(payload)
        assert "system_prompt" not in redacted
        assert long_prompt not in str(redacted)
        assert redacted["redaction_status"] == RedactionStatus.FULLY_REDACTED.value

    def test_messages_list_dropped(self):
        payload = {
            "case_id": "CLM-9202",
            "messages": [
                {"role": "system", "content": "Tu es l'agent..."},
                {"role": "user", "content": "Voici le dossier..."},
            ],
        }
        redacted = redact_audit_payload(payload)
        assert "messages" not in redacted

    def test_long_unnamed_free_text_dropped_by_length_alone(self):
        """Un prompt/texte libre non nommé explicitement reste couvert par
        le seuil générique de longueur — pas seulement par le nom du champ."""
        long_text = "Contexte clinique détaillé du patient. " * 20
        assert len(long_text) > MAX_SHORT_TEXT_LENGTH
        payload = {"case_id": "CLM-9203", "notes_libres": long_text}
        redacted = redact_audit_payload(payload)
        assert "notes_libres" not in redacted
        assert long_text not in str(redacted)


class TestKeepsIdentifiersEvidenceShortFieldsAndHashes:
    def test_identifiers_kept(self):
        payload = {
            "case_id": "CLM-9301",
            "event_id": "evt-1",
            "agent_name": "security_gate_agent",
            "actor": "security_gate_agent",
            "entry_id": "file-0",
        }
        redacted = redact_audit_payload(payload)
        for key, value in payload.items():
            assert redacted[key] == value
        assert redacted["redaction_status"] == RedactionStatus.NOT_REDACTED.value

    def test_evidence_ids_kept(self):
        payload = {"case_id": "CLM-9302", "evidence_ids": ["PROMPT_INJECTION_DETECTED", "SHELL_ACCESS_ATTEMPT"]}
        redacted = redact_audit_payload(payload)
        assert redacted["evidence_ids"] == ["PROMPT_INJECTION_DETECTED", "SHELL_ACCESS_ATTEMPT"]

    def test_short_generic_field_kept(self):
        payload = {"case_id": "CLM-9303", "input_type": "file", "policy_applied": "default"}
        redacted = redact_audit_payload(payload)
        assert redacted["input_type"] == "file"
        assert redacted["policy_applied"] == "default"
        assert redacted["redaction_status"] == RedactionStatus.NOT_REDACTED.value

    def test_hash_kept_even_though_64_characters(self):
        payload = {"case_id": "CLM-9304", "document_hash": _VALID_HASH}
        redacted = redact_audit_payload(payload)
        assert redacted["document_hash"] == _VALID_HASH

    def test_numbers_and_booleans_kept(self):
        payload = {"case_id": "CLM-9305", "confidence_score": 0.87, "prompt_injection_detected": True, "count": 3}
        redacted = redact_audit_payload(payload)
        assert redacted["confidence_score"] == 0.87
        assert redacted["prompt_injection_detected"] is True
        assert redacted["count"] == 3
        assert redacted["redaction_status"] == RedactionStatus.NOT_REDACTED.value

    def test_none_kept(self):
        payload = {"case_id": "CLM-9306", "target_node": None}
        redacted = redact_audit_payload(payload)
        assert redacted["target_node"] is None


class TestListTruncation:
    def test_list_over_max_items_truncated_and_marked_partially_redacted(self):
        payload = {"case_id": "CLM-9401", "tool_calls": [f"tool_{i}" for i in range(MAX_LIST_ITEMS + 5)]}
        redacted = redact_audit_payload(payload)
        assert len(redacted["tool_calls"]) == MAX_LIST_ITEMS
        assert redacted["redaction_status"] == RedactionStatus.PARTIALLY_REDACTED.value


class TestRedactionStatusComputation:
    def test_not_redacted_when_nothing_removed(self):
        payload = {"case_id": "CLM-9501", "outcome": "ALLOW"}
        redacted = redact_audit_payload(payload)
        assert redacted["redaction_status"] == RedactionStatus.NOT_REDACTED.value

    def test_partially_redacted_when_some_content_survives(self):
        payload = {"case_id": "CLM-9502", "system_prompt": "X" * 400, "reason": "Dossier conforme"}
        redacted = redact_audit_payload(payload)
        assert redacted["reason"] == "Dossier conforme"
        assert "system_prompt" not in redacted
        assert redacted["redaction_status"] == RedactionStatus.PARTIALLY_REDACTED.value

    def test_fully_redacted_when_only_identifiers_survive(self):
        payload = {"case_id": "CLM-9503", "system_prompt": "X" * 400, "full_text": "Y" * 400}
        redacted = redact_audit_payload(payload)
        assert set(redacted.keys()) == {"case_id", "redaction_status"}
        assert redacted["redaction_status"] == RedactionStatus.FULLY_REDACTED.value


class TestDoesNotMutateInput:
    def test_original_payload_untouched(self):
        payload = {"case_id": "CLM-9601", "system_prompt": "secret prompt", "notes": "ok"}
        original_copy = dict(payload)
        redact_audit_payload(payload)
        assert payload == original_copy
