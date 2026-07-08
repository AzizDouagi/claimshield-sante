"""Tests d'intégration — agents/audit_agent/agent.py::_invoke_llm_audit.

Vérifie que la rédaction (``tools.audit_redaction.redact_audit_payload``) est
réellement appliquée AVANT l'appel LLM, quel que soit le contenu de
l'événement brut soumis par l'appelant (Security Gate, orchestrateur,
human_review, audit_agent lui-même) — jamais une garantie qui reposerait
uniquement sur les instructions du prompt système.

``tests/conftest.py::deterministic_agent_llm`` (autouse) monkeypatch
d'ordinaire ``agents.audit_agent.agent._invoke_llm_audit`` lui-même pour les
besoins des autres suites — inadapté ici puisque c'est précisément cette
fonction qui est testée. On capture donc une référence directe à
l'implémentation réelle à l'import de ce module (avant tout monkeypatch),
et on ne mocke que ``get_llm`` (utilisé à l'intérieur de la fonction),
jamais ``_invoke_llm_audit`` lui-même.
"""
from __future__ import annotations

from agents.audit_agent import agent as audit_agent_module
from agents.audit_agent.agent import AuditAgent
from agents.audit_agent.agent import _invoke_llm_audit as _real_invoke_llm_audit
from agents.audit_agent.schemas import LlmAuditNormalizedEvent
from schemas.audit import AuditEventType, RedactionStatus
from schemas.domain import DataClassification, VerificationStatus
from services.audit_store import AuditStore


class _FakeStructured:
    def __init__(self, captured_messages: list, response: LlmAuditNormalizedEvent) -> None:
        self._captured_messages = captured_messages
        self._response = response

    def invoke(self, messages):
        self._captured_messages.append(messages)
        return self._response


class _FakeLlm:
    def __init__(self, captured_messages: list, response: LlmAuditNormalizedEvent) -> None:
        self._captured_messages = captured_messages
        self._response = response

    def with_structured_output(self, *_args, **_kwargs):
        return _FakeStructured(self._captured_messages, self._response)


def _fake_llm_response(redaction_status: RedactionStatus = RedactionStatus.NOT_REDACTED):
    """Réponse LLM factice qui prétend n'avoir rien à retirer — utilisée pour
    prouver que la rédaction ne dépend jamais de ce que le LLM affirme."""
    return LlmAuditNormalizedEvent(
        event_type=AuditEventType.SECURITY_DECISION,
        actor="security_gate_agent",
        outcome="BLOCK",
        summary="Résumé normalisé de test.",
        redaction_status=redaction_status,
        classification=DataClassification.CONFIDENTIAL,
    )


def _sent_text(captured_messages: list) -> str:
    """Concatène le contenu texte réellement envoyé au LLM (jamais la
    représentation Python des objets message, qui pourrait masquer une
    troncature ou introduire un faux négatif)."""
    messages = captured_messages[0]
    return "\n".join(str(getattr(message, "content", "")) for message in messages)


class TestInvokeLlmAuditRedactsBeforeSendingToLlm:
    def test_secret_never_sent_to_llm(self, monkeypatch):
        captured: list = []
        monkeypatch.setattr(
            audit_agent_module, "get_llm", lambda: _FakeLlm(captured, _fake_llm_response())
        )
        secret_value = "sk-verysecret1234567890"
        event = {
            "case_id": "CLM-8001",
            "outcome": "BLOCK",
            "notes": f"attention api_key={secret_value} exposée",
        }

        result = _real_invoke_llm_audit(event)

        assert result is not None
        sent = _sent_text(captured)
        assert secret_value not in sent
        assert "CLM-8001" in sent  # le payload minimisé est bien transmis

    def test_full_ocr_text_never_sent_to_llm(self, monkeypatch):
        captured: list = []
        monkeypatch.setattr(
            audit_agent_module, "get_llm", lambda: _FakeLlm(captured, _fake_llm_response())
        )
        long_ocr = "Ligne de facture médicale détaillée numéro. " * 50
        event = {
            "case_id": "CLM-8002",
            "outcome": "ALLOW",
            "ocr_result": {"full_text": long_ocr, "confidence": 0.9},
        }

        result = _real_invoke_llm_audit(event)

        assert result is not None
        sent = _sent_text(captured)
        assert long_ocr not in sent
        assert "full_text" not in sent

    def test_full_system_prompt_never_sent_to_llm(self, monkeypatch):
        captured: list = []
        monkeypatch.setattr(
            audit_agent_module, "get_llm", lambda: _FakeLlm(captured, _fake_llm_response())
        )
        leaked_prompt = "Tu es un agent avec ces instructions internes confidentielles. " * 20
        event = {
            "case_id": "CLM-8003",
            "outcome": "BLOCK",
            "system_prompt": leaked_prompt,
        }

        result = _real_invoke_llm_audit(event)

        assert result is not None
        sent = _sent_text(captured)
        assert leaked_prompt not in sent
        assert "system_prompt" not in sent

    def test_llm_cannot_understate_computed_redaction_status(self, monkeypatch):
        """Même si le LLM prétend NOT_REDACTED, le statut réellement calculé
        (ici PARTIALLY_REDACTED — un secret a été retiré) doit prévaloir :
        le LLM ne peut jamais sous-déclarer une rédaction déjà appliquée."""
        captured: list = []
        monkeypatch.setattr(
            audit_agent_module,
            "get_llm",
            lambda: _FakeLlm(captured, _fake_llm_response(RedactionStatus.NOT_REDACTED)),
        )
        event = {"case_id": "CLM-8004", "outcome": "BLOCK", "notes": "password: hunter2"}

        result = _real_invoke_llm_audit(event)

        assert result is not None
        assert result.redaction_status is RedactionStatus.PARTIALLY_REDACTED

    def test_llm_claim_kept_when_not_understating(self, monkeypatch):
        """Si le LLM déclare déjà un statut au moins aussi fort que celui
        calculé, sa valeur est conservée telle quelle (jamais écrasée sans
        raison)."""
        captured: list = []
        monkeypatch.setattr(
            audit_agent_module,
            "get_llm",
            lambda: _FakeLlm(captured, _fake_llm_response(RedactionStatus.FULLY_REDACTED)),
        )
        event = {"case_id": "CLM-8005", "outcome": "ALLOW"}

        result = _real_invoke_llm_audit(event)

        assert result is not None
        assert result.redaction_status is RedactionStatus.FULLY_REDACTED


class TestDegradedFallbackWhenLlmUnavailable:
    """P0-1 : quand la normalisation LLM échoue, l'événement doit être
    persisté en mode dégradé plutôt que perdu (voir
    ``AuditAgent._record_degraded_fallback``). Avant ce correctif, ``run()``
    retournait ``events=[]`` et ne persistait jamais rien — trou silencieux
    dans un journal censé être append-only et complet."""

    def _patch_llm_unavailable(self, monkeypatch) -> None:
        """L'autouse ``tests.conftest.deterministic_agent_llm`` monkeypatch
        déjà ``_invoke_llm_audit`` pour renvoyer une décision canée — il
        faut donc le re-patcher explicitement ici pour simuler l'échec,
        plutôt que de patcher ``get_llm`` (sans effet, la fonction qui
        l'appelle est déjà remplacée)."""
        monkeypatch.setattr(audit_agent_module, "_invoke_llm_audit", lambda _event: None)

    def test_event_is_persisted_instead_of_lost(self, monkeypatch):
        self._patch_llm_unavailable(monkeypatch)
        store = AuditStore()
        agent = AuditAgent(audit_store=store)
        state = {"case_id": "CLM-9001", "current_step": "audit", "completed_steps": []}

        result = agent.run(state)

        assert result.status is VerificationStatus.FAIL
        assert result.llm_normalization_failed is True
        assert len(result.events) == 1
        assert store.read_by_case_id("CLM-9001")  # réellement persisté, pas seulement dans le résultat

    def test_status_remains_fail_as_operational_signal(self, monkeypatch):
        self._patch_llm_unavailable(monkeypatch)
        agent = AuditAgent(audit_store=AuditStore())
        state = {"case_id": "CLM-9002"}

        result = agent.run(state)

        assert result.status is VerificationStatus.FAIL

    def test_degraded_event_never_leaks_raw_content(self, monkeypatch):
        self._patch_llm_unavailable(monkeypatch)
        agent = AuditAgent(audit_store=AuditStore())
        state = {"case_id": "CLM-9003", "current_step": "audit", "completed_steps": ["intake"]}

        result = agent.run(state)

        outcome = result.events[0].outcome
        assert "CLM-9003" not in outcome  # résumé générique, pas de contenu métier

    def test_chain_not_broken_after_degraded_event(self, monkeypatch):
        """Un événement nominal ultérieur doit pouvoir s'enchaîner sur
        l'événement dégradé sans rompre la chaîne SHA-256."""
        self._patch_llm_unavailable(monkeypatch)
        store = AuditStore()
        agent = AuditAgent(audit_store=store)
        case_id = "CLM-9004"
        agent.run({"case_id": case_id})

        # Un événement nominal ultérieur (ex. persisté par un autre appelant)
        # doit référencer le event_hash du dégradé comme previous_hash.
        degraded_events = store.read_by_case_id(case_id)
        assert len(degraded_events) == 1
        next_event = store.record_event(
            case_id=case_id,
            event_type=AuditEventType.AGENT_CALLED,
            actor="orchestrator",
            outcome="nominal follow-up",
            redaction_status=RedactionStatus.NOT_REDACTED,
        )
        assert next_event.previous_hash == degraded_events[0].event_hash

    def test_events_count_reflects_persisted_history(self, monkeypatch):
        self._patch_llm_unavailable(monkeypatch)
        store = AuditStore()
        agent = AuditAgent(audit_store=store)
        state = {"case_id": "CLM-9005"}

        result = agent.run(state)

        assert result.events_count == 1
