"""Response Composer — chat/response_composer.py (plan V2 §6, Phase V2-11a/V2-11c).

Point de passage obligé de toute réponse du chat : un seul appel LLM de
composition, strictement limité aux sorties déjà structurées des outils
(`chat/tools.py`) — jamais un document brut, un texte OCR complet ou un
prompt système. Suivi d'un **post-check déterministe anti-hallucination**
(jamais reporté à une sous-phase ultérieure, voir plan V2 §6) : toute
réponse citant un montant ou une date absent des données fournies est
rejetée et remplacée par une composition déterministe de repli — jamais un
fait inventé transmis au gestionnaire.

`DRAFT_MESSAGE` (Phase V2-11c) utilise un **prompt distinct**
(`prompts/chat_patient_message.yaml`, destinataire patient plutôt que
gestionnaire, ton et interdictions différents — jamais de promesse de
remboursement) mais partage exactement le même post-check
anti-hallucination — critère d'acceptation explicite de V2-11c (« contrôle
déterministe post-génération identique à celui de V2-11a, appliqué
systématiquement à `generate_patient_message` »)."""
from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from chat.answer_mode import detect_answer_modes
from chat.llm_usage import record_usage
from chat.memory_schemas import DiscussedScenario
from chat.prompt import load_chat_patient_message_prompt, load_chat_reasoning_prompt
from chat.schemas import (
    AuditSummary,
    ChatIntent,
    CorrectionRecommendation,
    ExplanationFacts,
    SimulationResult,
)
from llm.factory import get_llm

__all__ = ["compose"]

_AMOUNT_RE = re.compile(r"\b\d+[.,]\d{2}\b")
_DATE_ISO_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_DATE_SLASH_RE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")


def _extract_number_like_tokens(text: str) -> set[str]:
    """Montants/dates — jamais les identifiants (`CLM-\\d+`, `EVID-...`),
    volontairement exclus : ce sont des références structurelles, pas des
    faits métier pouvant être inventés."""
    return set(_AMOUNT_RE.findall(text)) | set(_DATE_ISO_RE.findall(text)) | set(
        _DATE_SLASH_RE.findall(text)
    )


def _grounded_tokens(tool_results: dict) -> set[str]:
    blob = json.dumps(tool_results, ensure_ascii=False, default=str)
    return _extract_number_like_tokens(blob)


def _invoke_llm_compose(data: dict, usage_sink: dict | None = None) -> str | None:
    try:
        prompt = load_chat_reasoning_prompt()
        llm = get_llm()
        result = llm.invoke(
            [
                SystemMessage(content=prompt.system_prompt),
                HumanMessage(content=json.dumps(data, ensure_ascii=False, default=str)),
            ]
        )
        record_usage(result, usage_sink)
        content = getattr(result, "content", None)
        return content if isinstance(content, str) and content.strip() else None
    except Exception:
        return None


def _invoke_llm_patient_message(data: dict, usage_sink: dict | None = None) -> str | None:
    try:
        prompt = load_chat_patient_message_prompt()
        llm = get_llm()
        result = llm.invoke(
            [
                SystemMessage(content=prompt.system_prompt),
                HumanMessage(content=json.dumps(data, ensure_ascii=False, default=str)),
            ]
        )
        record_usage(result, usage_sink)
        content = getattr(result, "content", None)
        return content if isinstance(content, str) and content.strip() else None
    except Exception:
        return None


def _fallback_compose(*, case_id: str | None, tool_results: dict) -> str:
    """Composition déterministe, sans LLM — utilisée si le LLM est
    indisponible ou si sa réponse est rejetée par le post-check
    anti-hallucination. Ne cite jamais rien de plus que les données déjà
    structurées reçues."""
    lines: list[str] = []
    if case_id:
        lines.append(f"Dossier {case_id}.")

    context = tool_results.get("context")
    if isinstance(context, dict):
        decision = context.get("final_decision")
        if decision:
            lines.append(f"Décision actuelle : {decision}.")
        step = context.get("current_step")
        if step:
            lines.append(f"Étape courante : {step}.")

    explanation = tool_results.get("explanation")
    if isinstance(explanation, ExplanationFacts):
        if explanation.final_decision and not isinstance(context, dict):
            lines.append(f"[FAIT] Décision actuelle : {explanation.final_decision}.")
        if explanation.decision_summary:
            lines.append("[FAIT] Justification : " + " ; ".join(explanation.decision_summary))
        if explanation.bounded_by:
            lines.append("[FAIT] Garde-fous appliqués : " + " ; ".join(explanation.bounded_by))
        if explanation.missing_information:
            lines.append(
                "[HYPOTHÈSE] Informations manquantes : "
                + " ; ".join(m.description for m in explanation.missing_information)
            )
        if explanation.assumptions:
            lines.append(
                "[HYPOTHÈSE] Hypothèses retenues : "
                + " ; ".join(a.description for a in explanation.assumptions)
            )
        if explanation.counterfactuals:
            lines.append(
                "[HYPOTHÈSE] Ce qui changerait la décision : "
                + " ; ".join(
                    f"{c.condition} → {c.resulting_decision.value}" for c in explanation.counterfactuals
                )
            )
        if explanation.recommended_action:
            lines.append(f"[FAIT] Action recommandée : {explanation.recommended_action}.")

    corrections = tool_results.get("corrections")
    if corrections:
        actions = [c.action for c in corrections if isinstance(c, CorrectionRecommendation)]
        if actions:
            lines.append("Actions recommandées : " + " ; ".join(actions))

    simulation = tool_results.get("simulation")
    if isinstance(simulation, SimulationResult):
        if not simulation.applied:
            lines.append(f"[SIMULATION] Simulation impossible : {simulation.error or 'motif inconnu'}.")
        else:
            lines.append(
                f"[SIMULATION] décision actuelle {simulation.original_decision}, "
                f"décision simulée {simulation.simulated_decision}."
            )
            lines.append(
                "[SIMULATION] La décision changerait."
                if simulation.decision_changed
                else "[SIMULATION] La décision ne changerait pas."
            )

    resolved_scenario = tool_results.get("resolved_scenario")
    if isinstance(resolved_scenario, DiscussedScenario):
        line = f"[FAIT] Scénario référencé : {resolved_scenario.description}"
        if resolved_scenario.related_decision:
            line += f" (décision : {resolved_scenario.related_decision})"
        lines.append(line + ".")

    audit_summary = tool_results.get("audit_summary")
    if isinstance(audit_summary, AuditSummary):
        lines.append(
            f"Audit : {audit_summary.event_count} événement(s), "
            f"chaîne {'intacte' if audit_summary.chain_intact else 'rompue'}."
        )
        if audit_summary.event_type_counts:
            details = ", ".join(f"{k}: {v}" for k, v in sorted(audit_summary.event_type_counts.items()))
            lines.append(f"Types d'événements : {details}.")

    patient_message_context = tool_results.get("patient_message_context")
    if isinstance(patient_message_context, dict):
        decision = patient_message_context.get("final_decision")
        if decision:
            lines.append(f"Votre dossier {patient_message_context.get('case_id', '')} : décision {decision}.")

    if not lines:
        return "Aucune information exploitable n'est disponible pour ce dossier."
    return "\n".join(lines)


def _serialize(value: object) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


def compose(
    *,
    case_id: str | None,
    intents: list[ChatIntent],
    tool_results: dict,
    usage_sink: dict | None = None,
) -> str:
    """Compose la réponse finale — jamais de fait hors des données déjà
    calculées par `chat/tools.py` (`context`/`explanation`/`corrections`/
    `simulation`/`audit_summary`/`patient_message_context`).

    `DRAFT_MESSAGE` bascule vers le prompt patient
    (`_invoke_llm_patient_message`) — priorité sur toute autre intention
    mélangée dans le même message (rédiger un message cohérent pour le
    patient prime, jamais un mélange de tons) ; le post-check
    anti-hallucination qui suit reste identique dans les deux cas.

    `usage_sink` (optionnel, `None` par défaut — aucun changement de
    comportement pour les appelants existants) : dict mutable rempli par
    `_invoke_llm_compose`/`_invoke_llm_patient_message` avec
    `model_name`/`input_tokens`/`output_tokens` si l'appel LLM réussit —
    voir `chat/agent.py` (visibilité temps réel des tokens, AZIZ)."""
    has_data = any(value not in (None, [], {}) for value in tool_results.values())
    if not has_data:
        return "Information non disponible pour ce dossier."

    serialized = {key: _serialize(value) for key, value in tool_results.items()}
    grounded_tokens = _grounded_tokens(serialized)
    answer_modes = [mode.value for mode in detect_answer_modes(intents=intents, tool_results=tool_results)]

    llm_data = {
        "case_id": case_id,
        "intentions": [intent.value for intent in intents],
        "answer_modes": answer_modes,
        **serialized,
    }
    invoke = _invoke_llm_patient_message if ChatIntent.DRAFT_MESSAGE in intents else _invoke_llm_compose
    llm_text = invoke(llm_data, usage_sink)
    if llm_text is None:
        return _fallback_compose(case_id=case_id, tool_results=tool_results)

    response_tokens = _extract_number_like_tokens(llm_text)
    unknown_tokens = response_tokens - grounded_tokens
    if unknown_tokens:
        return _fallback_compose(case_id=case_id, tool_results=tool_results)

    return llm_text
