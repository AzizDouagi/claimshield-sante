"""Production et validation du résumé sémantique conversationnel —
chat/semantic_summarizer.py.

Plan de remédiation « autonomie décisionnelle V2 », Phase 8, §6.3. Un appel
LLM propose une mise à jour de `ConversationSemanticState` à partir de
l'état sémantique précédent, du nouveau tour et des identifiants/décisions
déjà connus — mais **rien n'est jamais accepté sans validation Python**
(même patron anti-hallucination que partout ailleurs dans le projet) :
toute référence à une preuve/scénario/décision non réellement connue est
silencieusement retirée, jamais acceptée telle quelle.

**Fail-closed** : toute exception (LLM indisponible, réponse invalide)
laisse l'état sémantique précédent inchangé — jamais une perte silencieuse,
jamais un crash de la conversation.

Toutes les chaînes libres (`conversation_summary`, `description` des
scénarios, `open_questions`) sont tronquées à `tools.audit_redaction.MAX_SHORT_TEXT_LENGTH`
puis passées par `tools.audit_redaction.redact_audit_payload` avant d'être
acceptées — défense en profondeur supplémentaire, indépendante de la bonne
volonté du LLM (même mécanisme que l'Audit Agent, étape 14/V1)."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from chat.llm_usage import record_usage
from chat.memory_schemas import ConversationSemanticState, DiscussedScenario, LlmSemanticSummaryProposal
from chat.prompt import load_chat_semantic_summary_prompt
from llm.factory import get_llm
from tools.audit_redaction import MAX_SHORT_TEXT_LENGTH, redact_audit_payload

__all__ = ["update_semantic_state"]

_LAST_USER_GOAL_MAX_LENGTH = 200
"""Borne de `ConversationSemanticState.last_user_goal` — distincte du seuil
générique de rédaction (300), appliquée en plus pour ne jamais violer le
schéma final."""


def _invoke_llm_semantic_summary(
    data: dict[str, Any], usage_sink: dict | None = None
) -> LlmSemanticSummaryProposal | None:
    """`usage_sink` optionnel (`None` par défaut, aucun changement de
    comportement) — voir `chat/llm_usage.py`."""
    try:
        prompt = load_chat_semantic_summary_prompt()
        llm = get_llm()
        structured = llm.with_structured_output(
            LlmSemanticSummaryProposal, method="json_schema", include_raw=True
        )
        raw_result = structured.invoke(
            [
                SystemMessage(content=prompt.system_prompt),
                HumanMessage(content=json.dumps(data, ensure_ascii=False, default=str)),
            ]
        )
        if raw_result.get("parsing_error") is not None:
            return None
        record_usage(raw_result.get("raw"), usage_sink)
        result = raw_result.get("parsed")
        if isinstance(result, LlmSemanticSummaryProposal):
            return result
        if isinstance(result, dict):
            return LlmSemanticSummaryProposal(**result)
        return None
    except Exception:
        return None


def _redact_text(text: str, *, max_length: int = MAX_SHORT_TEXT_LENGTH) -> str:
    """Troncature déterministe à `max_length` **avant** rédaction — garantit
    que `redact_audit_payload` ne retire le champ que sur un motif de secret
    réel, jamais sur la seule longueur (puisqu'après troncature, la longueur
    ne dépasse jamais son propre seuil générique de 300 caractères)."""
    truncated = text[:max_length]
    redacted = redact_audit_payload({"text": truncated})
    value = redacted.get("text")
    return value if isinstance(value, str) else ""


def _validate_discussed_scenarios(
    scenarios: list[DiscussedScenario],
    *,
    known_evidence_ids: set[str],
    decisions_by_kind: dict[str, set[str]],
) -> list[DiscussedScenario]:
    """Retire silencieusement toute preuve inconnue et toute
    `related_decision` incohérente avec `kind` ou non réellement calculée —
    jamais une hallucination de scénario acceptée telle quelle."""
    validated: list[DiscussedScenario] = []
    for scenario in scenarios:
        evidence_ids = [e for e in scenario.evidence_ids if e in known_evidence_ids]
        related_decision = scenario.related_decision
        if related_decision is not None and related_decision not in decisions_by_kind.get(scenario.kind, set()):
            related_decision = None
        validated.append(
            scenario.model_copy(
                update={
                    "description": _redact_text(scenario.description),
                    "evidence_ids": evidence_ids,
                    "related_decision": related_decision,
                }
            )
        )
    return validated


def _validate_resolved_references(references: dict[str, str], *, known_scenario_ids: set[str]) -> dict[str, str]:
    return {phrase: scenario_id for phrase, scenario_id in references.items() if scenario_id in known_scenario_ids}


def update_semantic_state(
    *,
    previous: ConversationSemanticState | None,
    turn_summary: dict[str, Any],
    known_evidence_ids: set[str],
    real_decision: str | None,
    simulation_decisions: set[str],
    counterfactual_decisions: set[str],
    usage_sink: dict | None = None,
) -> ConversationSemanticState:
    """Calcule le nouvel état sémantique conversationnel.

    Args:
        previous: état sémantique précédent, `None` au premier tour.
        turn_summary: résumé déjà minimisé du tour courant (intentions,
            dossier, modes de réponse, identifiants de preuve — jamais le
            texte intégral du message).
        known_evidence_ids: identifiants de preuve réellement connus de ce
            tour — toute citation hors de cet ensemble est retirée.
        real_decision: décision réelle du dossier (`ClaimDecisionV2.value`),
            seule valeur citable pour un scénario `kind="REAL_DECISION"`.
        simulation_decisions/counterfactual_decisions: décisions déjà
            calculées par une simulation/un contrefactuel réel — seules
            valeurs citables pour `kind="SIMULATION"`/`"COUNTERFACTUAL"`.
        usage_sink: optionnel (`None` par défaut, aucun changement de
            comportement) — voir `chat/llm_usage.py`.

    Returns:
        Un `ConversationSemanticState` toujours valide — jamais une
        exception propagée, jamais un contenu inventé. Panne LLM ou réponse
        invalide → `previous` inchangé (ou un état neutre si `previous` est
        `None`), jamais une perte silencieuse ni un crash.
    """
    payload = {
        "previous_state": previous.model_dump(mode="json") if previous is not None else None,
        "turn": turn_summary,
        "known_evidence_ids": sorted(known_evidence_ids),
        "known_real_decision": real_decision,
        "known_simulation_decisions": sorted(simulation_decisions),
        "known_counterfactual_decisions": sorted(counterfactual_decisions),
    }
    proposal = _invoke_llm_semantic_summary(payload, usage_sink)
    now = datetime.now(UTC)

    if proposal is None:
        if previous is not None:
            return previous
        return ConversationSemanticState(conversation_summary="", updated_at=now)

    decisions_by_kind: dict[str, set[str]] = {
        "REAL_DECISION": {real_decision} if real_decision else set(),
        "SIMULATION": simulation_decisions,
        "COUNTERFACTUAL": counterfactual_decisions,
    }
    validated_scenarios = _validate_discussed_scenarios(
        proposal.discussed_scenarios, known_evidence_ids=known_evidence_ids, decisions_by_kind=decisions_by_kind
    )
    known_scenario_ids = {s.scenario_id for s in validated_scenarios}
    validated_references = _validate_resolved_references(
        proposal.resolved_references, known_scenario_ids=known_scenario_ids
    )
    known_decision_values = {d for d in (real_decision,) if d} | simulation_decisions | counterfactual_decisions
    validated_compared_decisions = [d for d in proposal.compared_decisions if d in known_decision_values]

    last_user_goal = (
        _redact_text(proposal.last_user_goal, max_length=_LAST_USER_GOAL_MAX_LENGTH)
        if proposal.last_user_goal
        else None
    )

    return ConversationSemanticState(
        conversation_summary=_redact_text(proposal.conversation_summary),
        last_user_goal=last_user_goal or None,
        discussed_scenarios=validated_scenarios,
        open_questions=[q for q in (_redact_text(q) for q in proposal.open_questions) if q],
        resolved_references=validated_references,
        compared_decisions=validated_compared_decisions,
        updated_at=now,
    )
