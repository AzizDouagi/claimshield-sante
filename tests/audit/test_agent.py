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
from agents.audit_agent.agent import _invoke_llm_audit_batch as _real_invoke_llm_audit_batch
from agents.audit_agent.schemas import (
    LlmAuditNormalizedEvent,
    LlmAuditNormalizedEventBatch,
    LlmAuditNormalizedEventItem,
)
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

    def test_degraded_fallback_emits_structured_log(self, monkeypatch, caplog):
        """P3-2 : le point de décision fail-safe critique (perte d'événement
        évitée) est journalisé pour alerte opérationnelle."""
        self._patch_llm_unavailable(monkeypatch)
        agent = AuditAgent(audit_store=AuditStore())
        state = {"case_id": "CLM-9006"}

        with caplog.at_level("WARNING"):
            agent.run(state)

        messages = [r.getMessage() for r in caplog.records]
        assert any(
            "audit_llm_normalization_failed_degraded_fallback" in m and "CLM-9006" in m
            for m in messages
        )

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


# ── Normalisation batchée (option C d'AZIZ) ─────────────────────────────────


def _normalized(
    *,
    event_type: AuditEventType = AuditEventType.AGENT_CALLED,
    actor: str = "orchestrator",
    outcome: str = "OK",
    redaction_status: RedactionStatus = RedactionStatus.NOT_REDACTED,
) -> LlmAuditNormalizedEvent:
    return LlmAuditNormalizedEvent(
        event_type=event_type,
        actor=actor,
        outcome=outcome,
        summary="Résumé de test.",
        redaction_status=redaction_status,
        classification=DataClassification.SYNTHETIC_TEST_DATA,
    )


class _StructuredBatch:
    def __init__(self, router: "_FakeLlmRouter") -> None:
        self._router = router

    def invoke(self, messages):
        self._router.batch_calls += 1
        self._router.captured_batch_messages.append(messages)
        response = self._router.batch_response
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(messages)
        return response


class _StructuredSingle:
    def __init__(self, router: "_FakeLlmRouter") -> None:
        self._router = router

    def invoke(self, messages):
        self._router.single_calls += 1
        text = "\n".join(str(getattr(m, "content", "")) for m in messages)
        for marker, response in self._router.single_response_by_marker.items():
            if marker in text:
                if isinstance(response, Exception):
                    raise response
                return response
        return self._router.single_default_response


class _FakeLlmRouter:
    """Fake LLM dont le comportement dépend du schéma structuré demandé
    (``LlmAuditNormalizedEventBatch`` vs ``LlmAuditNormalizedEvent``) —
    ``_invoke_llm_audit_batch`` et son repli individuel (``_invoke_llm_audit``)
    appellent tous deux ``get_llm()``, donc un seul fake doit savoir
    distinguer les deux appels."""

    def __init__(self) -> None:
        self.batch_calls = 0
        self.single_calls = 0
        self.batch_response: object = None
        self.single_response_by_marker: dict[str, object] = {}
        self.single_default_response: LlmAuditNormalizedEvent | None = None
        self.captured_batch_messages: list = []

    def with_structured_output(self, schema, **_kwargs):
        if schema is LlmAuditNormalizedEventBatch:
            return _StructuredBatch(self)
        return _StructuredSingle(self)


def _batch_sent_text(router: _FakeLlmRouter) -> str:
    messages = router.captured_batch_messages[0]
    return "\n".join(str(getattr(message, "content", "")) for message in messages)


class TestInvokeLlmAuditBatchSingleCallWhenFullyResolved:
    def test_full_batch_response_resolves_in_a_single_llm_call(self, monkeypatch):
        router = _FakeLlmRouter()
        router.batch_response = LlmAuditNormalizedEventBatch(
            events=[
                LlmAuditNormalizedEventItem(index=0, normalized=_normalized(outcome="A")),
                LlmAuditNormalizedEventItem(index=1, normalized=_normalized(outcome="B")),
            ]
        )
        monkeypatch.setattr(audit_agent_module, "get_llm", lambda: router)
        events = [{"case_id": "CLM-9100", "outcome": "x"}, {"case_id": "CLM-9101", "outcome": "y"}]

        output = _real_invoke_llm_audit_batch(events)

        assert router.batch_calls == 1
        assert router.single_calls == 0
        assert output[0] is not None and output[0].outcome == "A"
        assert output[1] is not None and output[1].outcome == "B"

    def test_empty_events_returns_empty_list_without_any_call(self, monkeypatch):
        router = _FakeLlmRouter()
        monkeypatch.setattr(audit_agent_module, "get_llm", lambda: router)

        assert _real_invoke_llm_audit_batch([]) == []
        assert router.batch_calls == 0


class TestInvokeLlmAuditBatchTargetedFallback:
    """``tests.conftest.deterministic_agent_llm`` (autouse) monkeypatch déjà
    ``_invoke_llm_audit`` pour le reste de la suite — comme la classe
    ``TestInvokeLlmAuditBatchTotalFailureFallsBackFully`` ci-dessous, ces
    tests exercent précisément le repli individuel *réel*
    (``_invoke_llm_audit``, qui appelle ``get_llm()``) et doivent donc le
    restaurer explicitement avant chaque test (même patron que
    ``TestInvokeLlmAuditBatchSingleCallWhenFullyResolved`` plus haut dans ce
    fichier pour ``_invoke_llm_audit`` seul)."""

    def _restore_real_individual_fallback(self, monkeypatch) -> None:
        monkeypatch.setattr(audit_agent_module, "_invoke_llm_audit", _real_invoke_llm_audit)

    def test_missing_index_falls_back_individually_only_for_that_event(self, monkeypatch):
        self._restore_real_individual_fallback(monkeypatch)
        router = _FakeLlmRouter()
        router.batch_response = LlmAuditNormalizedEventBatch(
            events=[LlmAuditNormalizedEventItem(index=0, normalized=_normalized(outcome="from-batch"))]
        )
        router.single_response_by_marker = {"CLM-9202": _normalized(outcome="from-fallback")}
        monkeypatch.setattr(audit_agent_module, "get_llm", lambda: router)
        events = [{"case_id": "CLM-9201", "outcome": "x"}, {"case_id": "CLM-9202", "outcome": "y"}]

        output = _real_invoke_llm_audit_batch(events)

        assert router.batch_calls == 1
        assert router.single_calls == 1  # repli ciblé uniquement pour l'index 1
        assert output[0].outcome == "from-batch"
        assert output[1].outcome == "from-fallback"

    def test_duplicate_index_invalidates_both_occurrences_and_falls_back(self, monkeypatch):
        self._restore_real_individual_fallback(monkeypatch)
        router = _FakeLlmRouter()
        router.batch_response = LlmAuditNormalizedEventBatch(
            events=[
                LlmAuditNormalizedEventItem(index=0, normalized=_normalized(outcome="first-claim")),
                LlmAuditNormalizedEventItem(index=0, normalized=_normalized(outcome="second-claim")),
            ]
        )
        router.single_response_by_marker = {"CLM-9301": _normalized(outcome="confirmed-individually")}
        monkeypatch.setattr(audit_agent_module, "get_llm", lambda: router)
        events = [{"case_id": "CLM-9301", "outcome": "x"}]

        output = _real_invoke_llm_audit_batch(events)

        # Ni "first-claim" ni "second-claim" n'est retenu sans confirmation —
        # jamais un "premier gagnant" silencieux (contrainte 2).
        assert router.single_calls == 1
        assert output[0].outcome == "confirmed-individually"

    def test_out_of_bounds_index_is_discarded_and_origin_event_falls_back(self, monkeypatch):
        self._restore_real_individual_fallback(monkeypatch)
        router = _FakeLlmRouter()
        router.batch_response = LlmAuditNormalizedEventBatch(
            events=[LlmAuditNormalizedEventItem(index=99, normalized=_normalized(outcome="bogus"))]
        )
        router.single_response_by_marker = {"CLM-9401": _normalized(outcome="from-fallback")}
        monkeypatch.setattr(audit_agent_module, "get_llm", lambda: router)
        events = [{"case_id": "CLM-9401", "outcome": "x"}]

        output = _real_invoke_llm_audit_batch(events)

        assert len(output) == 1
        assert router.single_calls == 1
        assert output[0].outcome == "from-fallback"

    def test_both_batch_and_individual_fallback_fail_yields_none(self, monkeypatch):
        """Contrat : ``output[i] is None`` seulement si le batch ET le repli
        individuel ont échoué pour cet événement précis — jamais une
        exception, jamais un résultat fabriqué (contrainte 1)."""
        self._restore_real_individual_fallback(monkeypatch)
        router = _FakeLlmRouter()
        router.batch_response = LlmAuditNormalizedEventBatch(
            events=[LlmAuditNormalizedEventItem(index=1, normalized=_normalized(outcome="only-index-1"))]
        )
        router.single_default_response = None  # le repli individuel échoue aussi
        monkeypatch.setattr(audit_agent_module, "get_llm", lambda: router)
        events = [{"case_id": "CLM-9501", "outcome": "x"}, {"case_id": "CLM-9502", "outcome": "y"}]

        output = _real_invoke_llm_audit_batch(events)

        assert len(output) == 2
        assert output[0] is None
        assert output[1] is not None and output[1].outcome == "only-index-1"


class TestInvokeLlmAuditBatchTotalFailureFallsBackFully:
    def _restore_real_individual_fallback(self, monkeypatch) -> None:
        monkeypatch.setattr(audit_agent_module, "_invoke_llm_audit", _real_invoke_llm_audit)

    def test_llm_exception_falls_back_to_one_call_per_event(self, monkeypatch):
        self._restore_real_individual_fallback(monkeypatch)
        router = _FakeLlmRouter()
        router.batch_response = RuntimeError("Ollama indisponible")
        router.single_response_by_marker = {
            "CLM-9601": _normalized(outcome="fallback-1"),
            "CLM-9602": _normalized(outcome="fallback-2"),
        }
        monkeypatch.setattr(audit_agent_module, "get_llm", lambda: router)
        events = [{"case_id": "CLM-9601", "outcome": "x"}, {"case_id": "CLM-9602", "outcome": "y"}]

        output = _real_invoke_llm_audit_batch(events)

        assert router.single_calls == 2
        assert output[0].outcome == "fallback-1"
        assert output[1].outcome == "fallback-2"

    def test_malformed_batch_response_falls_back_to_individual_calls(self, monkeypatch):
        self._restore_real_individual_fallback(monkeypatch)
        router = _FakeLlmRouter()
        router.batch_response = {"unexpected": "shape"}
        router.single_response_by_marker = {"CLM-9701": _normalized(outcome="fallback")}
        monkeypatch.setattr(audit_agent_module, "get_llm", lambda: router)
        events = [{"case_id": "CLM-9701", "outcome": "x"}]

        output = _real_invoke_llm_audit_batch(events)

        assert router.single_calls == 1
        assert output[0].outcome == "fallback"


class TestInvokeLlmAuditBatchSizeCap:
    def test_more_than_max_batch_size_skips_batch_entirely(self, monkeypatch):
        calls: list[dict] = []

        def fake_individual(event):
            calls.append(event)
            return _normalized(outcome="individual")

        def fail_if_batch_attempted():
            raise AssertionError("le mode batch ne doit jamais être tenté au-delà du cap")

        monkeypatch.setattr(audit_agent_module, "_invoke_llm_audit", fake_individual)
        monkeypatch.setattr(audit_agent_module, "get_llm", fail_if_batch_attempted)
        events = [{"case_id": f"CLM-97{i:02d}", "outcome": "x"} for i in range(26)]

        output = _real_invoke_llm_audit_batch(events)

        assert len(calls) == 26
        assert all(item is not None and item.outcome == "individual" for item in output)


class TestInvokeLlmAuditBatchRedactionPerElement:
    def test_secret_in_one_event_never_sent_and_other_events_unaffected(self, monkeypatch):
        router = _FakeLlmRouter()
        router.batch_response = LlmAuditNormalizedEventBatch(
            events=[
                LlmAuditNormalizedEventItem(index=0, normalized=_normalized(outcome="A")),
                LlmAuditNormalizedEventItem(index=1, normalized=_normalized(outcome="B")),
            ]
        )
        monkeypatch.setattr(audit_agent_module, "get_llm", lambda: router)
        secret_value = "sk-verysecret1234567890"
        events = [
            {"case_id": "CLM-9801", "outcome": "x", "notes": f"api_key={secret_value}"},
            {"case_id": "CLM-9802", "outcome": "y"},
        ]

        output = _real_invoke_llm_audit_batch(events)

        sent = _batch_sent_text(router)
        assert secret_value not in sent
        assert "CLM-9801" in sent
        assert "CLM-9802" in sent
        assert output[0] is not None
        assert output[1] is not None

    def test_redaction_floor_applies_independently_per_index(self, monkeypatch):
        """Le plancher de rédaction (le LLM ne peut jamais sous-déclarer un
        statut déjà appliqué) est calculé événement par événement — un
        secret dans un seul événement du lot ne doit jamais faire monter le
        statut déclaré pour l'événement voisin, non concerné."""
        router = _FakeLlmRouter()
        router.batch_response = LlmAuditNormalizedEventBatch(
            events=[
                LlmAuditNormalizedEventItem(
                    index=0, normalized=_normalized(outcome="A", redaction_status=RedactionStatus.NOT_REDACTED)
                ),
                LlmAuditNormalizedEventItem(
                    index=1, normalized=_normalized(outcome="B", redaction_status=RedactionStatus.NOT_REDACTED)
                ),
            ]
        )
        monkeypatch.setattr(audit_agent_module, "get_llm", lambda: router)
        events = [
            {"case_id": "CLM-9901", "outcome": "x", "notes": "password: hunter2"},
            {"case_id": "CLM-9902", "outcome": "y"},
        ]

        output = _real_invoke_llm_audit_batch(events)

        assert output[0].redaction_status is RedactionStatus.PARTIALLY_REDACTED
        assert output[1].redaction_status is RedactionStatus.NOT_REDACTED


class TestInvokeLlmAuditBatchOutOfOrderReassociation:
    def test_out_of_order_response_still_maps_by_explicit_index(self, monkeypatch):
        router = _FakeLlmRouter()
        router.batch_response = LlmAuditNormalizedEventBatch(
            events=[
                LlmAuditNormalizedEventItem(index=2, normalized=_normalized(outcome="third")),
                LlmAuditNormalizedEventItem(index=0, normalized=_normalized(outcome="first")),
                LlmAuditNormalizedEventItem(index=1, normalized=_normalized(outcome="second")),
            ]
        )
        monkeypatch.setattr(audit_agent_module, "get_llm", lambda: router)
        events = [
            {"case_id": "CLM-9001-A", "outcome": "x"},
            {"case_id": "CLM-9001-B", "outcome": "y"},
            {"case_id": "CLM-9001-C", "outcome": "z"},
        ]

        output = _real_invoke_llm_audit_batch(events)

        assert output[0].outcome == "first"
        assert output[1].outcome == "second"
        assert output[2].outcome == "third"


class TestInvokeLlmAuditBatchNeverMixesSameCaseIdOrActor:
    """Contrainte 3 (revue AZIZ) : deux événements partageant le même
    case_id/actor mais des event_type/outcome différents ne doivent jamais
    voir leurs normalisations interverties."""

    def test_two_events_same_case_id_and_actor_never_swapped(self, monkeypatch):
        router = _FakeLlmRouter()
        # Volontairement dans le désordre — même case_id/actor pour les deux
        # événements d'origine, seul l'index doit faire foi.
        router.batch_response = LlmAuditNormalizedEventBatch(
            events=[
                LlmAuditNormalizedEventItem(
                    index=1,
                    normalized=_normalized(
                        event_type=AuditEventType.FINAL_REPORT, actor="orchestrator", outcome="SUCCESS"
                    ),
                ),
                LlmAuditNormalizedEventItem(
                    index=0,
                    normalized=_normalized(
                        event_type=AuditEventType.AGENT_CALLED, actor="orchestrator", outcome="ATTEMPT"
                    ),
                ),
            ]
        )
        monkeypatch.setattr(audit_agent_module, "get_llm", lambda: router)
        events = [
            {"case_id": "CLM-9950", "actor": "orchestrator", "action": "call", "outcome": "ATTEMPT"},
            {"case_id": "CLM-9950", "actor": "orchestrator", "action": "result", "outcome": "SUCCESS"},
        ]

        output = _real_invoke_llm_audit_batch(events)

        assert output[0].event_type is AuditEventType.AGENT_CALLED
        assert output[0].outcome == "ATTEMPT"
        assert output[1].event_type is AuditEventType.FINAL_REPORT
        assert output[1].outcome == "SUCCESS"


class TestNormalizeEventsBatchPublicAlias:
    def test_delegates_to_invoke_llm_audit_batch(self, monkeypatch):
        sentinel = [_normalized(outcome="via-alias")]
        monkeypatch.setattr(audit_agent_module, "_invoke_llm_audit_batch", lambda events: sentinel)

        result = audit_agent_module.normalize_events_batch([{"case_id": "CLM-9999"}])

        assert result is sentinel
