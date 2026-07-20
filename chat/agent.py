"""Chat Reasoning Agent — point d'entrée unique, chat/agent.py (plan V2 §6).

`handle_message` délègue systématiquement : compréhension (`chat/nlu.py`)
→ planification (`chat/planner.py`) → outils (`chat/tools.py`) →
composition (`chat/response_composer.py`) — ne répond jamais directement,
ne contient aucune logique métier propre.

Mémoire conversationnelle (Phase 8, plan de remédiation « autonomie
décisionnelle V2 », §6) — entièrement **optionnelle et opt-in** :
`thread_id`/`user_id`/`conversation_store` par défaut à `None` reproduisent
exactement le comportement d'avant cette phase (aucune mémoire, aucun appel
supplémentaire). Les trois doivent être fournis ensemble pour activer la
mémoire — jamais un état partiel silencieusement actif. Le texte intégral du
message/de la réponse n'est **jamais** conservé (`chat.memory_schemas.ConversationTurn.message_digest`/
`reply_digest`, empreintes SHA-256 non réversibles calculées par `_digest`).

Visibilité temps réel des étapes + tokens (demandé par AZIZ, « comme Claude
Code ») — `_run_turn` accepte un callback optionnel `on_step` qui reçoit un
`ChatStepEvent` à chaque frontière d'étape (compréhension, outil dispatché,
composition, mémoire). `handle_message` (inchangée, utilisée par tous les
appelants existants) appelle `_run_turn(..., on_step=None)` — un `on_step`
absent ne modifie strictement rien au comportement ni à la performance.
Seuls les 4 appels LLM propres à ce module (NLU/composition/message
patient/résumé sémantique) rapportent des tokens réels
(`chat/llm_usage.py`) ; l'intention `SIMULATE` peut ré-invoquer jusqu'à 5
agents V2 en interne (`chat/simulation_engine.py`) — leurs tokens ne sont
**pas** comptabilisés ici (limite assumée, chantier séparé), l'étape
correspondante est émise avec `input_tokens`/`output_tokens=None` explicite,
jamais un zéro trompeur."""
from __future__ import annotations

import hashlib
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from uuid import uuid4

from chat.answer_mode import detect_answer_modes
from chat.conversation_store import ConversationAccessError, ConversationStore
from chat.memory_schemas import (
    ConversationSemanticState,
    ConversationTurn,
    DiscussedScenario,
    SimulationContext,
)
from chat.nlu import extract_intent
from chat.planner import plan
from chat.response_composer import compose
from chat.schemas import (
    ChatIntent,
    ChatPlan,
    ChatPlanAction,
    ChatStepEvent,
    ChatStepStatus,
    ExplanationFacts,
    SimulationPatch,
    SimulationResult,
)
from chat.semantic_summarizer import update_semantic_state
from chat.tools import (
    explain_claim,
    generate_patient_message,
    get_audit_summary,
    get_claim_context,
    recommend_corrections,
    simulate_changes,
)
from config.logging import get_logger

__all__ = ["handle_message", "handle_message_streaming"]

OnStepCallback = Callable[[ChatStepEvent], Awaitable[None]]

logger = get_logger(__name__)

_CLARIFY_NEEDED_MESSAGE = (
    "Pour vous répondre, j'ai besoin de l'identifiant du dossier concerné "
    "(ex. CLM-0001) et d'une question plus précise."
)
_NOT_YET_AVAILABLE_MESSAGE = (
    "Cette fonctionnalité n'est pas encore disponible dans cette version du chat."
)
_CASE_NOT_FOUND_MESSAGE = "Dossier introuvable — jamais soumis ou thread expiré."
_SIMULATE_MISSING_CHANGES_MESSAGE = (
    "Merci de préciser le changement à simuler : retirer un document déjà "
    "présent (ex. « et si on retirait l'ordonnance ? ») ou changer le rôle "
    "de lecture (ex. « simule avec le rôle FRAUD_ANALYST »). Un nouveau "
    "document ne peut pas être ajouté via le chat."
)


def _digest(text: str) -> str:
    """Empreinte SHA-256 non réversible — jamais le texte intégral conservé
    en mémoire conversationnelle (plan V2 §6.2)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def _emit(
    on_step: OnStepCallback | None,
    *,
    step_name: str,
    label: str,
    status: ChatStepStatus,
    model_name: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    duration_ms: int | None = None,
    detail: str = "",
) -> None:
    """N'émet rien si `on_step is None` — coût et comportement strictement
    nuls pour `handle_message()` (aucun callback fourni)."""
    if on_step is None:
        return
    await on_step(
        ChatStepEvent(
            step_name=step_name,
            label=label,
            status=status,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            detail=detail,
        )
    )


def _elapsed_ms(started_at: float) -> int:
    return int((time.monotonic() - started_at) * 1000)


def _collect_known_evidence_ids(tool_results: dict) -> set[str]:
    explanation = tool_results.get("explanation")
    if not isinstance(explanation, ExplanationFacts):
        return set()
    ids: set[str] = set()
    for item in explanation.missing_information:
        ids.update(item.evidence_ids)
    for item in explanation.decisive_factors:
        ids.update(item.evidence_ids)
    return ids


def _collect_real_decision(tool_results: dict) -> str | None:
    explanation = tool_results.get("explanation")
    if isinstance(explanation, ExplanationFacts) and explanation.final_decision:
        return explanation.final_decision
    context = tool_results.get("context")
    if isinstance(context, dict) and context.get("final_decision"):
        return context["final_decision"]
    return None


def _collect_simulation_decisions(tool_results: dict) -> set[str]:
    simulation = tool_results.get("simulation")
    if isinstance(simulation, SimulationResult) and simulation.applied and simulation.simulated_decision:
        return {simulation.simulated_decision}
    return set()


def _collect_counterfactual_decisions(tool_results: dict) -> set[str]:
    explanation = tool_results.get("explanation")
    if not isinstance(explanation, ExplanationFacts):
        return set()
    return {c.resulting_decision.value for c in explanation.counterfactuals}


def _find_resolved_scenario(
    scenario_id: str | None, discussed_scenarios: list[DiscussedScenario]
) -> DiscussedScenario | None:
    if scenario_id is None:
        return None
    return next((s for s in discussed_scenarios if s.scenario_id == scenario_id), None)


def _merge_simulation_patches(
    existing: list[SimulationPatch], new: list[SimulationPatch]
) -> list[SimulationPatch]:
    """Fusionne les patches d'une simulation ciblée en cours de discussion
    (Phase 9, §7 du plan) — un nouveau patch sur un champ déjà patché
    **remplace** l'ancien (jamais les deux conservés pour le même champ),
    l'ordre d'insertion des champs déjà connus est préservé."""
    merged: dict = {patch.field: patch for patch in existing}
    for patch in new:
        merged[patch.field] = patch
    return list(merged.values())


_TOOL_STEP_LABELS: dict[ChatIntent, tuple[str, str]] = {
    ChatIntent.ANALYZE: ("outil_analyze", "Récupération du contexte du dossier"),
    ChatIntent.EXPLAIN: ("outil_explain", "Explication de la décision"),
    ChatIntent.CORRECT: ("outil_correct", "Recommandations de correction"),
    ChatIntent.AUDIT: ("outil_audit", "Résumé d'audit"),
    ChatIntent.DRAFT_MESSAGE: ("outil_draft_message", "Rédaction du message patient"),
}
"""Slug/libellé d'étape par intention exécutable — `SIMULATE` en est
volontairement absent (libellé dynamique ciblée/complète, voir `_run_turn`)."""


async def handle_message(
    message: str,
    case_id: str | None = None,
    *,
    thread_id: str | None = None,
    user_id: str | None = None,
    conversation_store: ConversationStore | None = None,
) -> str:
    """Traite un message libre — `case_id` optionnel (contexte déjà connu
    de l'appelant, ex. dossier affiché côté UI) prime toujours sur un
    identifiant que le NLU prétendrait avoir détecté dans le texte.

    `thread_id`/`user_id`/`conversation_store` (Phase 8) activent la mémoire
    conversationnelle uniquement si les trois sont fournis ensemble —
    reproduit sinon exactement le comportement d'avant cette phase.

    Wrapper fin sur `_run_turn(..., on_step=None)` — voir `handle_message_streaming`
    pour la variante avec visibilité temps réel des étapes/tokens."""
    return await _run_turn(
        message,
        case_id,
        thread_id=thread_id,
        user_id=user_id,
        conversation_store=conversation_store,
        on_step=None,
    )


async def handle_message_streaming(
    message: str,
    case_id: str | None = None,
    *,
    thread_id: str | None = None,
    user_id: str | None = None,
    conversation_store: ConversationStore | None = None,
    on_step: OnStepCallback,
) -> str:
    """Identique à `handle_message`, avec émission d'un `ChatStepEvent` à
    chaque frontière d'étape via `on_step` (obligatoire ici — voir
    `api/v2/chat.py::chat_v2_stream` pour l'usage réel via une file
    asyncio consommée par un générateur SSE)."""
    return await _run_turn(
        message,
        case_id,
        thread_id=thread_id,
        user_id=user_id,
        conversation_store=conversation_store,
        on_step=on_step,
    )


async def _run_turn(
    message: str,
    case_id: str | None = None,
    *,
    thread_id: str | None = None,
    user_id: str | None = None,
    conversation_store: ConversationStore | None = None,
    on_step: OnStepCallback | None,
) -> str:
    memory_enabled = thread_id is not None and user_id is not None and conversation_store is not None
    context = conversation_store.get(user_id=user_id, thread_id=thread_id) if memory_enabled else None
    recent_turns = context.turns if context is not None else None
    semantic_state = context.semantic_state if context is not None else None

    started_at = time.monotonic()
    nlu_usage: dict = {}
    await _emit(on_step, step_name="comprehension", label="Compréhension de la demande", status=ChatStepStatus.STARTED)
    nlu_result = extract_intent(
        message, case_id, recent_turns=recent_turns, semantic_state=semantic_state, usage_sink=nlu_usage
    )
    await _emit(
        on_step,
        step_name="comprehension",
        label="Compréhension de la demande",
        status=ChatStepStatus.COMPLETED if nlu_result is not None else ChatStepStatus.FAILED,
        model_name=nlu_usage.get("model_name"),
        input_tokens=nlu_usage.get("input_tokens"),
        output_tokens=nlu_usage.get("output_tokens"),
        duration_ms=_elapsed_ms(started_at),
        detail="" if nlu_result is not None else "LLM indisponible ou réponse invalide",
    )
    chat_plan = plan(nlu_result)

    reply: str
    tool_results: dict = {}
    merged_simulation_patches: list[SimulationPatch] | None = None

    if chat_plan.action is ChatPlanAction.CLARIFY_NEEDED:
        reply = _CLARIFY_NEEDED_MESSAGE
    elif chat_plan.action is ChatPlanAction.NOT_YET_AVAILABLE:
        reply = _NOT_YET_AVAILABLE_MESSAGE
    else:
        resolved_case_id = chat_plan.case_id
        if resolved_case_id is None:
            # Défensif — `chat.planner.plan` garantit déjà un case_id non
            # None pour ACTION.EXECUTE, jamais atteint en pratique.
            reply = _CLARIFY_NEEDED_MESSAGE
        elif ChatIntent.SIMULATE in chat_plan.intents and chat_plan.simulation_changes is None:
            reply = _SIMULATE_MISSING_CHANGES_MESSAGE
        else:
            for intent, (step_name, label) in _TOOL_STEP_LABELS.items():
                if intent not in chat_plan.intents:
                    continue
                tool_started_at = time.monotonic()
                await _emit(on_step, step_name=step_name, label=label, status=ChatStepStatus.STARTED)
                if intent is ChatIntent.ANALYZE:
                    tool_results["context"] = await get_claim_context(resolved_case_id)
                elif intent is ChatIntent.EXPLAIN:
                    tool_results["explanation"] = await explain_claim(resolved_case_id)
                elif intent is ChatIntent.CORRECT:
                    tool_results["corrections"] = await recommend_corrections(resolved_case_id)
                elif intent is ChatIntent.AUDIT:
                    tool_results["audit_summary"] = await get_audit_summary(resolved_case_id)
                elif intent is ChatIntent.DRAFT_MESSAGE:
                    tool_results["patient_message_context"] = await generate_patient_message(resolved_case_id)
                await _emit(
                    on_step,
                    step_name=step_name,
                    label=label,
                    status=ChatStepStatus.COMPLETED,
                    duration_ms=_elapsed_ms(tool_started_at),
                )
            if ChatIntent.SIMULATE in chat_plan.intents and chat_plan.simulation_changes is not None:
                effective_changes = chat_plan.simulation_changes
                # Simulation ciblée (Phase 9) — accumule avec les patches
                # déjà actifs pour ce thread (« et si on changeait aussi... »)
                # plutôt que de repartir du dossier réel à chaque message.
                if memory_enabled and effective_changes.field_patches:
                    previous_patches = context.active_simulation_patches if context is not None else []
                    merged_simulation_patches = _merge_simulation_patches(
                        previous_patches, effective_changes.field_patches
                    )
                    effective_changes = effective_changes.model_copy(
                        update={"field_patches": merged_simulation_patches}
                    )
                simulate_label = (
                    "Simulation ciblée : autonomous_decision"
                    if effective_changes.field_patches
                    else "Simulation complète : pipeline entier"
                )
                # Jusqu'à 5 appels LLM internes aux agents V2 ré-invoqués —
                # tokens non comptabilisés ici (limite assumée, voir
                # docstring de module), `input_tokens`/`output_tokens`
                # explicitement absents plutôt qu'un zéro trompeur.
                simulate_started_at = time.monotonic()
                await _emit(on_step, step_name="outil_simulate", label=simulate_label, status=ChatStepStatus.STARTED)
                tool_results["simulation"] = await simulate_changes(resolved_case_id, effective_changes)
                await _emit(
                    on_step,
                    step_name="outil_simulate",
                    label=simulate_label,
                    status=ChatStepStatus.COMPLETED,
                    duration_ms=_elapsed_ms(simulate_started_at),
                )

            if semantic_state is not None:
                resolved_scenario = _find_resolved_scenario(
                    chat_plan.resolved_scenario_id, semantic_state.discussed_scenarios
                )
                if resolved_scenario is not None:
                    tool_results["resolved_scenario"] = resolved_scenario

            if not tool_results or all(value in (None, [], {}) for value in tool_results.values()):
                reply = _CASE_NOT_FOUND_MESSAGE
            else:
                compose_started_at = time.monotonic()
                compose_usage: dict = {}
                await _emit(
                    on_step, step_name="composition", label="Composition de la réponse", status=ChatStepStatus.STARTED
                )
                reply = compose(
                    case_id=resolved_case_id,
                    intents=chat_plan.intents,
                    tool_results=tool_results,
                    usage_sink=compose_usage,
                )
                await _emit(
                    on_step,
                    step_name="composition",
                    label="Composition de la réponse",
                    status=ChatStepStatus.COMPLETED,
                    model_name=compose_usage.get("model_name"),
                    input_tokens=compose_usage.get("input_tokens"),
                    output_tokens=compose_usage.get("output_tokens"),
                    duration_ms=_elapsed_ms(compose_started_at),
                )

    if memory_enabled:
        assert thread_id is not None and user_id is not None and conversation_store is not None
        memory_started_at = time.monotonic()
        memory_usage: dict = {}
        await _emit(on_step, step_name="memoire", label="Mise à jour de la mémoire conversationnelle", status=ChatStepStatus.STARTED)
        memory_failed = False
        try:
            _record_turn_and_update_memory(
                message=message,
                reply=reply,
                chat_plan=chat_plan,
                tool_results=tool_results,
                thread_id=thread_id,
                user_id=user_id,
                conversation_store=conversation_store,
                previous_semantic_state=semantic_state,
                previous_simulations_count=len(context.simulations) if context is not None else 0,
                merged_simulation_patches=merged_simulation_patches,
                usage_sink=memory_usage,
            )
        except ConversationAccessError:
            # Réutilisation frauduleuse d'un thread_id — doit rester visible
            # à l'appelant, jamais avalée silencieusement.
            memory_failed = True
            await _emit(
                on_step,
                step_name="memoire",
                label="Mise à jour de la mémoire conversationnelle",
                status=ChatStepStatus.FAILED,
                duration_ms=_elapsed_ms(memory_started_at),
                detail="Accès mémoire refusé",
            )
            raise
        except Exception:
            # Une panne de la couche mémoire (bug, LLM de résumé, etc.) ne
            # doit jamais faire échouer une réponse déjà composée et déjà
            # envoyée à l'utilisateur — journalisée, jamais propagée.
            memory_failed = True
            logger.warning("chat_memory_recording_failed", thread_id=thread_id)
            await _emit(
                on_step,
                step_name="memoire",
                label="Mise à jour de la mémoire conversationnelle",
                status=ChatStepStatus.FAILED,
                duration_ms=_elapsed_ms(memory_started_at),
                detail="Panne de la couche mémoire",
            )
        if not memory_failed:
            await _emit(
                on_step,
                step_name="memoire",
                label="Mise à jour de la mémoire conversationnelle",
                status=ChatStepStatus.COMPLETED,
                model_name=memory_usage.get("model_name"),
                input_tokens=memory_usage.get("input_tokens"),
                output_tokens=memory_usage.get("output_tokens"),
                duration_ms=_elapsed_ms(memory_started_at),
            )

    return reply


def _record_turn_and_update_memory(
    *,
    message: str,
    reply: str,
    chat_plan: ChatPlan,
    tool_results: dict,
    thread_id: str,
    user_id: str,
    conversation_store: ConversationStore,
    previous_semantic_state: ConversationSemanticState | None,
    previous_simulations_count: int,
    merged_simulation_patches: list[SimulationPatch] | None,
    usage_sink: dict | None = None,
) -> None:
    """Enregistre le tour courant et met à jour le résumé sémantique — ne
    lève jamais d'exception qui interromprait la réponse déjà composée à
    l'utilisateur (une panne de mémoire ne doit jamais faire échouer la
    conversation elle-même), à l'exception explicite de
    `ConversationAccessError` (réutilisation frauduleuse d'un `thread_id`,
    qui doit rester visible à l'appelant). `usage_sink` optionnel (`None`
    par défaut, aucun changement de comportement) — voir `chat/llm_usage.py`."""
    answer_modes = detect_answer_modes(intents=chat_plan.intents, tool_results=tool_results) if tool_results else []
    known_evidence_ids = _collect_known_evidence_ids(tool_results)

    turn = ConversationTurn(
        turn_id=str(uuid4()),
        message_digest=_digest(message),
        reply_digest=_digest(reply),
        intents=chat_plan.intents,
        case_id=chat_plan.case_id,
        answer_modes=answer_modes,
        evidence_ids=sorted(known_evidence_ids),
        created_at=datetime.now(UTC),
    )

    simulation_result = tool_results.get("simulation")
    simulation_context: SimulationContext | None = None
    if isinstance(simulation_result, SimulationResult) and simulation_result.applied:
        simulation_context = SimulationContext(
            scenario_id=f"SCENARIO-{previous_simulations_count + 1}",
            original_decision=simulation_result.original_decision,
            simulated_decision=simulation_result.simulated_decision,
            decision_changed=simulation_result.decision_changed,
        )

    conversation_store.append_turn(
        user_id=user_id, thread_id=thread_id, turn=turn, simulation=simulation_context
    )

    if merged_simulation_patches is not None:
        conversation_store.update_active_simulation(
            user_id=user_id, thread_id=thread_id, patches=merged_simulation_patches
        )

    turn_summary: dict = {
        "intents": [i.value for i in chat_plan.intents],
        "case_id": chat_plan.case_id,
        "answer_modes": [m.value for m in answer_modes],
        "evidence_ids": sorted(known_evidence_ids),
    }
    if simulation_context is not None:
        # Contexte minimal permettant au LLM de résumé de décrire le
        # scénario simulé (Phase 9) sans jamais recalculer/inventer quoi
        # que ce soit — mêmes valeurs déjà présentes dans `simulation_context`.
        turn_summary["simulation"] = {
            "scenario_id": simulation_context.scenario_id,
            "original_decision": simulation_context.original_decision,
            "simulated_decision": simulation_context.simulated_decision,
            "decision_changed": simulation_context.decision_changed,
        }

    new_semantic_state = update_semantic_state(
        previous=previous_semantic_state,
        turn_summary=turn_summary,
        known_evidence_ids=known_evidence_ids,
        real_decision=_collect_real_decision(tool_results),
        simulation_decisions=_collect_simulation_decisions(tool_results),
        counterfactual_decisions=_collect_counterfactual_decisions(tool_results),
        usage_sink=usage_sink,
    )
    conversation_store.update_semantic_state(
        user_id=user_id, thread_id=thread_id, semantic_state=new_semantic_state
    )
