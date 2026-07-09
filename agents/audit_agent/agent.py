"""Audit Agent — normalisation LLM et persistance via AuditStore.

Chaque exécution passe par le LLM : l'agent lui soumet un événement structuré
et minimisé, puis valide la réponse Pydantic. Le LLM ne persiste rien et ne
calcule jamais la chaîne d'audit ; l'écriture append-only reste déléguée à
``services.audit_store.AuditStore.record_event``.
"""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Callable, Protocol, runtime_checkable

from langchain_core.messages import HumanMessage, SystemMessage

from agents.audit_agent.prompt import load_audit_batch_prompt, load_audit_prompt
from agents.audit_agent.schemas import LlmAuditNormalizedEvent, LlmAuditNormalizedEventBatch
from config.logging import get_logger
from config.settings import get_settings
from llm.factory import get_llm
from schemas.audit import AuditEventType, RedactionStatus
from schemas.domain import DataClassification, VerificationStatus
from schemas.results import AuditEvent as StateAuditEvent
from schemas.results import AuditResult, LlmMetadata
from services.audit_store import AuditStore, AuditStoreError
from state.claim_state import ClaimState, validate_state_update
from tools.audit_redaction import redact_audit_payload

_STEP_NAME = "audit"
_AGENT_NAME = "audit_agent"

logger = get_logger(__name__)
_POLICY_VERSION = "1.0.0"

_REDACTION_RANK: dict[RedactionStatus, int] = {
    RedactionStatus.NOT_REDACTED: 0,
    RedactionStatus.PARTIALLY_REDACTED: 1,
    RedactionStatus.FULLY_REDACTED: 2,
}

_MAX_BATCH_SIZE = 25
"""Cap défensif de ``_invoke_llm_audit_batch`` — même valeur que
``LlmAuditNormalizedEventBatch.events`` (``max_length``). Au-delà, pas de
tentative batch : repli direct sur le mode individuel (un nœud produit
typiquement 3 à 9 événements d'audit, jamais plus de quelques dizaines)."""

_RESULT_KEYS: tuple[str, ...] = (
    "intake_result",
    "security_result",
    "privacy_result",
    "identity_coverage_result",
    "fhir_result",
    "ocr_result",
    "coding_result",
    "clinical_result",
    "fraud_result",
    "review_result",
    "audit_result",
)


# ── Interface ────────────────────────────────────────────────────────────────


@runtime_checkable
class AuditAgentRunnable(Protocol):
    """Interface minimale requise par le nœud LangGraph."""

    def run(self, state: ClaimState) -> AuditResult: ...


# ── Appel LLM obligatoire ────────────────────────────────────────────────────


def _compute_redaction(event: dict) -> tuple[dict, RedactionStatus]:
    """Rédaction déterministe d'un événement, indépendante du succès LLM.

    Appelable aussi bien avant une tentative de normalisation LLM (pour lui
    servir de plancher, voir ``_invoke_llm_audit``) que dans le chemin de
    secours dégradé (``AuditAgent.run``) lorsque le LLM est indisponible —
    dans les deux cas, le calcul est identique et ne dépend jamais du LLM.
    """
    redacted_event = redact_audit_payload(event)
    return redacted_event, RedactionStatus(redacted_event["redaction_status"])


def _invoke_llm_audit(event: dict) -> LlmAuditNormalizedEvent | None:
    """Normalise l'événement structuré via LLM et sortie Pydantic stricte.

    ``event`` est d'abord passé par ``redact_audit_payload`` — défense en
    profondeur indépendante de la bonne volonté du LLM : un prompt complet,
    un OCR complet, un secret ou un texte médical long n'atteint jamais le
    message envoyé, quelle que soit la provenance de l'événement (Security
    Gate, orchestrateur, human_review, audit_agent lui-même). Le statut de
    rédaction réellement appliqué (calculé, jamais deviné) sert ensuite de
    plancher : le LLM ne peut jamais déclarer un ``redaction_status`` plus
    faible que ce qui a été effectivement retiré.
    """
    redacted_event, computed_status = _compute_redaction(event)
    try:
        prompt = load_audit_prompt()
        llm = get_llm()
        structured = llm.with_structured_output(
            LlmAuditNormalizedEvent,
            method="json_schema",
        )
        payload = {
            "prompt_version": prompt.version,
            "event": redacted_event,
            "allowed_event_types": [event_type.value for event_type in AuditEventType],
            "allowed_redaction_statuses": [status.value for status in RedactionStatus],
            "allowed_classifications": [
                classification.value for classification in DataClassification
            ],
        }
        result = structured.invoke([
            SystemMessage(content=prompt.system_prompt),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
        ])
        if isinstance(result, dict):
            result = LlmAuditNormalizedEvent(**result)
        if not isinstance(result, LlmAuditNormalizedEvent):
            return None
        if _REDACTION_RANK[computed_status] > _REDACTION_RANK[result.redaction_status]:
            result = result.model_copy(update={"redaction_status": computed_status})
        return result
    except Exception:
        return None


def normalize_event(event: dict) -> LlmAuditNormalizedEvent | None:
    """Point public injectable pour normaliser un événement d'audit."""
    return _invoke_llm_audit(event)


def _invoke_llm_audit_batch(events: list[dict]) -> list[LlmAuditNormalizedEvent | None]:
    """Normalise plusieurs événements en un seul appel LLM (option C d'AZIZ).

    Réduit le nombre d'appels LLM par nœud de N (un par événement d'audit)
    à 1 — but strictement de performance, jamais un affaiblissement de la
    garantie de conformité étape 14 : chaque événement du lot est rédigé
    individuellement (``_compute_redaction``, inchangée) avant l'appel, et
    toute normalisation absente/invalide (index manquant, dupliqué, hors
    bornes, ou échec total du lot) déclenche un repli sur ``_invoke_llm_audit``
    (chemin single-event, inchangé) pour les événements concernés.

    Contrat de sortie : ``len(output) == len(events)`` toujours ;
    ``output[i] is None`` seulement si la normalisation (batch **et** repli
    individuel) a échoué pour l'événement à cet index précis — dans ce cas
    l'appelant (``Orchestrator._persist_audit_events``) ne doit jamais
    persister cet événement.
    """
    if not events:
        return []

    if len(events) > _MAX_BATCH_SIZE:
        return [_invoke_llm_audit(event) for event in events]

    redacted_pairs = [_compute_redaction(event) for event in events]

    try:
        prompt = load_audit_batch_prompt()
        llm = get_llm()
        structured = llm.with_structured_output(
            LlmAuditNormalizedEventBatch,
            method="json_schema",
        )
        payload = {
            "prompt_version": prompt.version,
            "events": [
                {"index": i, "event": redacted_event}
                for i, (redacted_event, _computed_status) in enumerate(redacted_pairs)
            ],
            "allowed_event_types": [event_type.value for event_type in AuditEventType],
            "allowed_redaction_statuses": [status.value for status in RedactionStatus],
            "allowed_classifications": [
                classification.value for classification in DataClassification
            ],
        }
        result = structured.invoke([
            SystemMessage(content=prompt.system_prompt),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
        ])
        if isinstance(result, dict):
            result = LlmAuditNormalizedEventBatch(**result)
        if not isinstance(result, LlmAuditNormalizedEventBatch):
            return [_invoke_llm_audit(event) for event in events]
    except Exception:
        # Repli total — comportement identique à aujourd'hui, jamais un crash.
        return [_invoke_llm_audit(event) for event in events]

    output: list[LlmAuditNormalizedEvent | None] = [None] * len(events)
    seen_indices: set[int] = set()
    duplicate_indices: set[int] = set()
    for item in result.events:
        if item.index < 0 or item.index >= len(events):
            # Index hors bornes : écarté, l'événement d'origine (si son
            # propre index est par ailleurs valide) reste marqué manquant.
            continue
        if item.index in seen_indices:
            duplicate_indices.add(item.index)
            continue
        seen_indices.add(item.index)
        output[item.index] = item.normalized

    for duplicate_index in duplicate_indices:
        # Un index dupliqué invalide les deux occurrences — aucune n'est
        # retenue sans confirmation individuelle (jamais un "premier gagnant").
        output[duplicate_index] = None

    missing_indices = [i for i in range(len(events)) if output[i] is None]
    for i in missing_indices:
        output[i] = _invoke_llm_audit(events[i])

    for i, normalized in enumerate(output):
        if normalized is None:
            continue
        _, computed_status = redacted_pairs[i]
        if _REDACTION_RANK[computed_status] > _REDACTION_RANK[normalized.redaction_status]:
            output[i] = normalized.model_copy(update={"redaction_status": computed_status})

    return output


def normalize_events_batch(events: list[dict]) -> list[LlmAuditNormalizedEvent | None]:
    """Point public injectable pour normaliser un lot d'événements d'audit."""
    return _invoke_llm_audit_batch(events)


# ── Implémentation réelle ────────────────────────────────────────────────────


class AuditAgent:
    """Audit Agent réel, avec store injectable pour les tests et l'orchestration."""

    def __init__(self, audit_store: AuditStore | None = None) -> None:
        self.audit_store = audit_store if audit_store is not None else AuditStore()

    def run(self, state: ClaimState) -> AuditResult:
        case_id = str(state.get("case_id", "UNKNOWN"))
        llm_metadata = _build_llm_metadata()
        structured_event = _build_structured_event(state, case_id)

        normalized = _invoke_llm_audit(structured_event)
        if normalized is None:
            return self._record_degraded_fallback(state, case_id, structured_event, llm_metadata)

        llm_metadata.confidence = normalized.confidence_score

        try:
            persisted = self.audit_store.record_event(
                case_id=case_id,
                event_type=normalized.event_type,
                actor=normalized.actor,
                outcome=_persistent_outcome(normalized),
                redaction_status=normalized.redaction_status,
                agent_name=normalized.agent_name or _AGENT_NAME,
                model_name=llm_metadata.model_name,
                prompt_version=llm_metadata.prompt_version,
                tool_calls=normalized.tool_calls,
                evidence_ids=normalized.evidence_ids,
            )
        except AuditStoreError as exc:
            return AuditResult(
                case_id=case_id,
                status=VerificationStatus.FAIL,
                events_count=len(self.audit_store.read_by_case_id(case_id)),
                events=[],
                policy_version=_POLICY_VERSION,
                reasons=[exc.structured.message],
                llm_metadata=llm_metadata,
            )
        except Exception as exc:
            return AuditResult(
                case_id=case_id,
                status=VerificationStatus.FAIL,
                events_count=len(self.audit_store.read_by_case_id(case_id)),
                events=[],
                policy_version=_POLICY_VERSION,
                reasons=[f"Persistance audit impossible après normalisation LLM : {exc}"],
                llm_metadata=llm_metadata,
            )

        light_event = _to_state_audit_event(persisted, normalized)
        events_count = len(self.audit_store.read_by_case_id(case_id))
        reasons = normalized.reasons or [normalized.summary]

        return AuditResult(
            case_id=case_id,
            status=VerificationStatus.PASS,
            events_count=events_count,
            events=[light_event],
            policy_version=_POLICY_VERSION,
            reasons=reasons,
            llm_metadata=llm_metadata,
        )

    def _record_degraded_fallback(
        self,
        state: ClaimState,
        case_id: str,
        structured_event: dict,
        llm_metadata: LlmMetadata,
    ) -> AuditResult:
        """Persiste un événement dégradé plutôt que de laisser un trou
        silencieux dans le journal d'audit lorsque le LLM est indisponible.

        ``status`` reste ``FAIL`` (signal opérationnel : Ollama indisponible
        doit rester visible) mais, contrairement au comportement précédent,
        ``events`` n'est plus systématiquement vide — l'événement est
        persisté avec un plancher de rédaction calculé déterministiquement
        (jamais deviné) et un acteur/outcome explicitement marqués comme
        dégradés. Seul un échec de la persistance elle-même (chaîne rompue,
        `AuditStoreError`) reste un événement réellement non persisté.
        """
        # P0-1/P3-2 : point de décision fail-safe critique — perte évitée
        # d'un événement d'audit, journalisé pour alerte opérationnelle
        # (Ollama probablement indisponible).
        logger.warning("audit_llm_normalization_failed_degraded_fallback", case_id=case_id)
        _, computed_status = _compute_redaction(structured_event)
        reasons = [
            "AUDIT_LLM_NORMALIZATION_FAILED : événement persisté en mode "
            "dégradé, sans normalisation LLM."
        ]
        try:
            persisted = self.audit_store.record_event(
                case_id=case_id,
                event_type=AuditEventType.ANOMALY,
                actor="system:audit_agent_degraded_fallback",
                outcome=_degraded_outcome(state),
                redaction_status=computed_status,
                agent_name=_AGENT_NAME,
                model_name=None,
                prompt_version=None,
                tool_calls=(),
                evidence_ids=(),
            )
        except AuditStoreError as exc:
            return AuditResult(
                case_id=case_id,
                status=VerificationStatus.FAIL,
                events_count=len(self.audit_store.read_by_case_id(case_id)),
                events=[],
                policy_version=_POLICY_VERSION,
                reasons=[*reasons, exc.structured.message],
                llm_metadata=llm_metadata,
                llm_normalization_failed=True,
            )
        except Exception as exc:
            return AuditResult(
                case_id=case_id,
                status=VerificationStatus.FAIL,
                events_count=len(self.audit_store.read_by_case_id(case_id)),
                events=[],
                policy_version=_POLICY_VERSION,
                reasons=[*reasons, f"Persistance dégradée impossible : {exc}"],
                llm_metadata=llm_metadata,
                llm_normalization_failed=True,
            )

        light_event = StateAuditEvent(
            event_id=str(uuid.uuid4()),
            case_id=persisted.case_id,
            actor=_AGENT_NAME,
            action=persisted.event_type.value,
            outcome=persisted.outcome,
            agent_version=_POLICY_VERSION,
            timestamp=persisted.timestamp,
            details={
                "redaction_status": persisted.redaction_status.value,
                "event_hash": persisted.event_hash,
                "persistent_event_id": persisted.event_id,
                "llm_normalization_failed": "true",
            },
        )
        return AuditResult(
            case_id=case_id,
            status=VerificationStatus.FAIL,
            events_count=len(self.audit_store.read_by_case_id(case_id)),
            events=[light_event],
            policy_version=_POLICY_VERSION,
            reasons=reasons,
            llm_metadata=llm_metadata,
            llm_normalization_failed=True,
        )


# Alias conservé pour les imports historiques ; l'implémentation n'est plus un stub.
_NotImplementedStub = AuditAgent
_DEFAULT_IMPL: AuditAgentRunnable = AuditAgent()


# ── Adaptation d'entrée et de sortie ─────────────────────────────────────────


def _build_llm_metadata() -> LlmMetadata:
    settings = get_settings()
    return LlmMetadata(
        model_name=settings.claimshield_llm_model,
        prompt_version=load_audit_prompt().version,
    )


def _degraded_outcome(state: ClaimState) -> str:
    """Résumé déterministe (sans LLM) pour l'événement d'audit dégradé.

    Volontairement minimal — mêmes garanties de minimisation que
    ``_compact_results``/``_build_structured_event`` : aucun contenu de
    document, texte OCR ou prompt, uniquement des identifiants/compteurs.
    """
    current_step = state.get("current_step") or "inconnu"
    completed = list(state.get("completed_steps") or [])[-5:]
    errors_count = len(state.get("errors") or [])
    return (
        f"Audit dégradé (normalisation LLM indisponible) — "
        f"étape={current_step}, dernières étapes={completed}, "
        f"erreurs={errors_count}."
    )


def _build_structured_event(state: ClaimState, case_id: str) -> dict:
    audit_trail = list(state.get("audit_trail") or [])
    return {
        "case_id": case_id,
        "observed_at": datetime.now(UTC).isoformat(),
        "current_step": state.get("current_step"),
        "completed_steps": list(state.get("completed_steps") or [])[-20:],
        "errors_count": len(state.get("errors") or []),
        "alerts_count": len(state.get("alerts") or []),
        "final_recommendation": _enum_or_value(state.get("final_recommendation")),
        "human_decision": _compact_value(state.get("human_decision")),
        "latest_audit_event": _compact_value(audit_trail[-1]) if audit_trail else None,
        "audit_trail_count": len(audit_trail),
        "candidate_event_type": _candidate_event_type(state).value,
        "upstream_results": _compact_results(state),
    }


def _candidate_event_type(state: ClaimState) -> AuditEventType:
    if state.get("final_recommendation") is not None:
        return AuditEventType.FINAL_REPORT
    if state.get("human_decision") is not None:
        return AuditEventType.HUMAN_DECISION
    if state.get("errors"):
        return AuditEventType.FAILURE
    security_result = state.get("security_result")
    if security_result is not None:
        return AuditEventType.SECURITY_DECISION
    if state.get("current_step"):
        return AuditEventType.AGENT_CALLED
    return AuditEventType.CLAIM_STARTED


def _compact_results(state: ClaimState) -> dict[str, dict]:
    compacted: dict[str, dict] = {}
    for key in _RESULT_KEYS:
        result = state.get(key)  # type: ignore[literal-required]
        if result is None:
            continue
        data = _compact_value(result)
        if not isinstance(data, dict):
            continue
        compacted[key] = {
            "status": _first_present(data, "status", "decision"),
            "recommendation": _first_present(data, "recommendation", "final_recommendation"),
            "reasons": list(data.get("reasons") or [])[:5],
            "errors_count": len(data.get("errors") or []),
            "alerts_count": len(data.get("alerts") or []),
        }
    return compacted


def _first_present(data: dict, *keys: str) -> object | None:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def _compact_value(value: object) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")  # type: ignore[attr-defined]
    if isinstance(value, dict):
        return {str(k): _compact_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_compact_value(item) for item in value[:20]]
    return _enum_or_value(value)


def _enum_or_value(value: object) -> object:
    return getattr(value, "value", value)


def _persistent_outcome(normalized: LlmAuditNormalizedEvent) -> str:
    return f"{normalized.outcome} — {normalized.summary}"


def _to_state_audit_event(
    persisted,
    normalized: LlmAuditNormalizedEvent,
) -> StateAuditEvent:
    return StateAuditEvent(
        event_id=str(uuid.uuid4()),
        case_id=persisted.case_id,
        actor=_AGENT_NAME,
        action=persisted.event_type.value,
        outcome=normalized.outcome,
        agent_version=_POLICY_VERSION,
        timestamp=persisted.timestamp,
        details={
            "summary": normalized.summary,
            "redaction_status": persisted.redaction_status.value,
            "classification": normalized.classification.value,
            "anomalies": "; ".join(normalized.anomalies[:10]),
            "redactions": "; ".join(normalized.redactions[:10]),
            "event_hash": persisted.event_hash,
            "persistent_event_id": persisted.event_id,
        },
    )


# ── Factory et nœud LangGraph ────────────────────────────────────────────────


def make_node(
    impl: AuditAgentRunnable = _DEFAULT_IMPL,
) -> Callable[[ClaimState], dict]:
    """Crée un nœud LangGraph avec l'implémentation injectable fournie."""
    def _node(state: ClaimState) -> dict:
        result = impl.run(state)
        updates: dict = {
            "audit_result": result,
            "current_step": _STEP_NAME,
            "completed_steps": [_STEP_NAME],
        }
        if result.events:
            updates["audit_trail"] = result.events
        if result.status is VerificationStatus.FAIL:
            reason = "; ".join(result.reasons) if result.reasons else "Journalisation échouée."
            updates["errors"] = [f"[{_AGENT_NAME}] {reason}"]
        validate_state_update(updates)
        return updates

    _node.__name__ = f"node_{_STEP_NAME}"
    return _node


# Nœud stable — nom utilisé comme clé dans le StateGraph.
node = make_node()
