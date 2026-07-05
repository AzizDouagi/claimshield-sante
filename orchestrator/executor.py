"""Exécution contrôlée d'un agent — orchestrator/executor.py.

``Orchestrator`` est le seul point d'entrée qui appelle réellement un
agent : il enchaîne, dans cet ordre strict, les validations déjà définies
ailleurs dans ``orchestrator/`` — préconditions (``routing.py``), modèle
(``policies.py``), outils (``policies.py``) — puis délègue l'exécution à
l'agent injecté. Aucune étape n'est sautée, aucun contournement n'est
possible : le premier refus empêche définitivement l'appel de l'agent.

``Orchestrator`` ne possède ni prompt ni logique métier : il ne connaît
aucune règle clinique, financière ou de fraude, et n'interprète jamais le
*sens* du résultat produit par un agent (cela reste le rôle de
``graph/edges.py`` et, plus tard, de ``case_reviewer_agent``). Il valide en
revanche systématiquement sa *forme* — chaque sortie d'agent est repassée
par ``orchestrator.orchestrator.validate_agent_result`` avant d'être
acceptée comme résultat final : un dictionnaire brut ou du texte libre
n'est jamais accepté tel quel, toute anomalie devient un
``AgentCallOutcome`` en échec structuré, jamais une exception qui remonte.
Tout ce dont ``Orchestrator`` a besoin est reçu par injection — aucun
import direct d'un module ``agents/*`` ni de ``graph/`` :

- ``model_registry`` : ``ModelRegistry`` (``orchestrator.model_registry``).
- ``agent_registry`` : mapping ``AgentName -> Callable[[ClaimState], dict]``
  — même signature que ``agents.<nom>.agent.node`` — fourni par
  l'appelant (production : agents réels ; tests : doubles déterministes).
- Les trois contrôles (``preconditions_check``, ``model_check``,
  ``tools_check``) sont substituables ; leurs valeurs par défaut sont les
  implémentations réelles de ``orchestrator.routing``/``orchestrator.policies``.

Vit dans un module séparé de ``orchestrator/orchestrator.py`` pour éviter
tout import circulaire : ``routing.py`` et ``policies.py`` importent déjà
``orchestrator.orchestrator`` (contrats) — ce dernier ne peut donc pas
importer ``routing.py``/``policies.py`` en retour, alors qu'``Orchestrator``
a besoin des trois.

Politique de retry (``RetryPolicy``)
-------------------------------------
Configurable (``max_attempts``, codes d'erreur rejouables), injectée dans
``Orchestrator`` au même titre que les registres et contrôles. Ne s'applique
**qu'à l'appel de l'agent** (dernière étape) : les préconditions, le modèle
et les outils ne sont vérifiés qu'une seule fois par ``execute_agent`` —
un refus à l'une de ces trois étapes est déterministe (même état, même
registre, même allowlist) et n'est donc jamais rejoué, quel que soit
``max_attempts``. Seules deux catégories d'échec sont rejouables :
  - une panne transitoire de l'agent (exception dont le type appartient à
    ``RetryPolicy.transient_exceptions`` — même catégorisation que
    ``graph/nodes.py::_TRANSIENT_NODE_EXCEPTIONS``, redéfinie ici pour ne
    pas importer ``graph/nodes.py``, cf. plus haut) ;
  - une sortie d'agent explicitement marquée réparable
    (``RetryPolicy.retryable_error_codes`` — par défaut les codes de
    ``validate_agent_result`` qui traduisent une sortie malformée plutôt
    qu'une absence de résultat : ``AGENT_RESULT_INVALID``,
    ``AGENT_RESULT_UNSTRUCTURED``).
Entre deux tentatives, seul ``AgentCallRequest.attempt`` change (incrémenté
via ``model_copy``) — ``case_id``, ``current_step``, ``agent_name`` et
``authorized_context`` restent strictement identiques, et le même ``state``
est repassé sans modification : même exécution, même contexte.

Distincte du retry technique de ``graph/nodes.py`` (étape 11, interne à un
seul appel de nœud LangGraph) : ``RetryPolicy`` opère au niveau de
l'orchestrateur, sur un nombre de tentatives explicitement configurable par
l'appelant, indépendamment de tout graphe. Distincte aussi de
``langgraph.types.RetryPolicy`` (autre classe, autre module, jamais
importée ici).

Fallback modèle (``_resolve_model``)
-------------------------------------
Si ``model_check`` refuse le modèle demandé, ``Orchestrator`` tente **un
seul** modèle de remplacement — celui retourné par
``ModelRegistry.find_fallback`` (enregistré, activé, compatible avec
l'agent) — puis rejoue le **même** ``model_check`` sur ce candidat. Le
fallback n'est retenu que s'il est lui aussi explicitement autorisé : jamais
de contournement du contrôle de permission. Sans candidat, ou si le
candidat est lui-même refusé, l'erreur d'origine est retournée telle
quelle — jamais masquée par une erreur générique de fallback. Le schéma de
sortie attendu (``AGENT_RESULT_MODELS``) et les outils autorisés
(``tools_check``, qui ne dépend que de l'agent, pas du modèle) restent
strictement inchangés par un fallback. Le modèle initial, le motif du refus
et le modèle de remplacement retenu sont journalisés (``logging``) et
reportés dans ``AgentCallOutcome.metadata``.

Événements d'audit (``_build_audit_event``)
--------------------------------------------
Chaque étape d'``execute_agent`` émet un ``AuditEvent`` — schéma déjà
existant (``schemas.results.AuditEvent``), déjà consommé de façon
append-only par ``agents/privacy_agent`` dans ``ClaimState.audit_trail`` :
aucun nouveau schéma, aucune nouvelle interface. Six natures d'événement
(``action``) : ``authorization`` (contrôle franchi), ``refusal`` (contrôle
refusé — arrête l'exécution), ``call`` (tentative d'appel de l'agent),
``retry`` (une nouvelle tentative va être jouée), ``fallback`` (bascule de
modèle appliquée ou rejetée) et ``result`` (issue finale de
``execute_agent``, succès ou échec). Chaque événement porte ``case_id``
(champ natif) et ``actor`` = nom de l'agent (champ natif), puis dans
``details`` : ``model_id``, ``tools`` (noms joints par une virgule),
``policy`` (code du ``StructuredError`` de la décision concernée),
``attempt`` et ``final_status``. Jamais de secret, de prompt complet, de
document brut ou de texte OCR complet — ``details`` ne contient que des
identifiants, des codes et des compteurs, jamais le contenu produit par un
agent (voir ``AgentCallOutcome.result_payload``, distinct et déjà
minimisé). Les événements sont accumulés dans
``AgentCallOutcome.audit_events`` (ordre chronologique, jamais réordonnés)
— ``Orchestrator`` ne les ajoute jamais lui-même à ``ClaimState`` (il ne
mute jamais le state, voir plus haut) : c'est à l'appelant de les ajouter à
``state["audit_trail"]``, exactement comme le ferait un nœud LangGraph.
N'implémente pas l'Audit Agent (étape 12, toujours stub).
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import httpx
from langchain_core.tools import BaseTool

from orchestrator.model_registry import AGENT_REQUIRED_CAPABILITIES, ModelCapability, ModelRegistry
from orchestrator.orchestrator import (
    AGENT_RESULT_FIELD,
    AgentCallOutcome,
    AgentCallRequest,
    AgentName,
    AgentResultValidationError,
    validate_agent_result,
)
from orchestrator.policies import PolicyDecision, build_authorized_tools, evaluate_model_authorization
from orchestrator.routing import evaluate_call_preconditions
from schemas.results import AuditEvent, StructuredError
from state.claim_state import ClaimState

logger = logging.getLogger(__name__)

EVENT_AUTHORIZATION = "authorization"
EVENT_REFUSAL = "refusal"
EVENT_CALL = "call"
EVENT_RETRY = "retry"
EVENT_FALLBACK = "fallback"
EVENT_RESULT = "result"


def _build_audit_event(
    request: AgentCallRequest,
    *,
    action: str,
    outcome: str,
    model_id: str = "",
    tools: Sequence[str] = (),
    policy: str = "",
    attempt: int | None = None,
    final_status: str | None = None,
) -> AuditEvent:
    """Construit un événement d'audit minimisé pour ``request.agent_name``.

    Reprend l'interface append-only existante (``AuditEvent`` —
    ``schemas.results``) sans en créer une nouvelle. ``details`` ne contient
    que des identifiants/codes/compteurs (``model_id``, ``tools`` joints par
    une virgule, ``policy``, ``attempt``, ``final_status``) — jamais de
    secret, de prompt complet, de document brut ni de texte OCR complet, qui
    n'atteignent d'ailleurs jamais cette couche (``Orchestrator`` ne
    manipule que des décisions et des résultats déjà validés)."""
    return AuditEvent(
        event_id=str(uuid.uuid4()),
        case_id=request.case_id,
        actor=request.agent_name.value,
        action=action,
        outcome=outcome,
        details={
            "model_id": model_id or "",
            "tools": ",".join(tools) if tools else "",
            "policy": policy or "",
            "attempt": str(attempt if attempt is not None else request.attempt),
            "final_status": final_status or "IN_PROGRESS",
        },
    )

AgentRunner = Callable[[ClaimState], dict[str, Any]]
"""Signature d'un exécuteur d'agent injecté — identique à
``agents.<nom>.agent.node`` : lit le state, retourne une mise à jour
partielle contenant (au minimum) le champ ``*_result`` de l'agent."""

_DEFAULT_TRANSIENT_EXCEPTIONS: tuple[type[Exception], ...] = (
    httpx.ConnectError,
    httpx.TimeoutException,
    ConnectionError,
)

DEFAULT_RETRYABLE_ERROR_CODES: frozenset[str] = frozenset({
    "AGENT_RESULT_INVALID",
    "AGENT_RESULT_UNSTRUCTURED",
})
"""Codes de ``StructuredError`` considérés comme des sorties explicitement
réparables par défaut — une sortie malformée ou non structurée peut
raisonnablement différer d'une tentative à l'autre (nouvel appel LLM). Ne
contient volontairement aucun code de refus (préconditions, modèle, outils)
ni ``AGENT_RESULT_MISSING``/``AGENT_NOT_REGISTERED`` (absence de résultat ou
d'exécuteur : plus probablement un défaut de câblage qu'une panne
transitoire — non réparable par un simple nouvel essai)."""


@dataclass(frozen=True)
class RetryPolicy:
    """Politique de retry, configurable et injectable.

    ``max_attempts`` borne le nombre total de tentatives (1 = aucun retry,
    valeur par défaut — comportement inchangé si non fourni). Seules les
    pannes transitoires (``transient_exceptions``) et les sorties
    explicitement réparables (``retryable_error_codes``) sont rejouées —
    jamais un refus de permission ni une précondition non satisfaite, qui
    ne sont de toute façon jamais soumis à cette politique (voir le
    docstring du module).
    """

    max_attempts: int = 1
    retryable_error_codes: frozenset[str] = field(
        default_factory=lambda: DEFAULT_RETRYABLE_ERROR_CODES
    )
    transient_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: _DEFAULT_TRANSIENT_EXCEPTIONS
    )

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts doit être ≥ 1, reçu : {self.max_attempts}")

    def is_retryable(self, *, error_code: str, exc: Exception | None) -> bool:
        """Décide si un échec est rejouable.

        ``AGENT_EXECUTION_FAILED`` (l'agent a levé une exception) n'est
        rejouable que si le type de l'exception appartient à
        ``transient_exceptions`` — une exception non catégorisée (bug,
        valeur invalide) échoue immédiatement, sans retry, exactement comme
        le retry technique de ``graph/nodes.py``. Les autres codes suivent
        ``retryable_error_codes`` tel quel.
        """
        if error_code == "AGENT_EXECUTION_FAILED":
            return exc is not None and isinstance(exc, self.transient_exceptions)
        return error_code in self.retryable_error_codes


@dataclass(frozen=True)
class Orchestrator:
    """Exécute un agent après validation complète — préconditions, modèle,
    outils, dans cet ordre, sans exception possible. L'appel de l'agent lui
    même suit ``retry_policy`` (défaut : une seule tentative)."""

    model_registry: ModelRegistry
    agent_registry: Mapping[AgentName, AgentRunner]
    preconditions_check: Callable[[ClaimState, AgentCallRequest], PolicyDecision] = (
        evaluate_call_preconditions
    )
    model_check: Callable[[ModelRegistry, AgentName, str], PolicyDecision] = (
        evaluate_model_authorization
    )
    tools_check: Callable[[AgentName], tuple[BaseTool, ...]] = build_authorized_tools
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)

    def execute_agent(
        self, request: AgentCallRequest, state: ClaimState, *, model_id: str
    ) -> AgentCallOutcome:
        """Exécute ``request.agent_name`` après trois contrôles successifs.

        Ordre strict, jamais réordonné : préconditions -> modèle -> outils
        -> appel de l'agent. Le premier refus retourné par un contrôle
        empêche tous les suivants — l'agent injecté (``agent_registry``)
        n'est alors jamais appelé. Aucun contournement possible : il n'y a
        pas de chemin de code qui atteigne l'agent sans être passé par les
        trois contrôles dans cet ordre. Ces trois contrôles ne sont évalués
        qu'une seule fois — jamais rejoués, y compris si ``retry_policy``
        autorise plusieurs tentatives d'appel de l'agent.

        Chaque étape émet un ``AuditEvent`` (voir docstring du module) —
        accumulés dans ``AgentCallOutcome.audit_events``, dans l'ordre.
        """
        events: list[AuditEvent] = []

        precondition_decision = self.preconditions_check(state, request)
        events.append(
            _build_audit_event(
                request,
                action=EVENT_AUTHORIZATION if precondition_decision.allowed else EVENT_REFUSAL,
                outcome=precondition_decision.effect.value,
                policy=precondition_decision.reason.code,
                final_status=None if precondition_decision.allowed else precondition_decision.reason.code,
            )
        )
        if not precondition_decision.allowed:
            return AgentCallOutcome.from_request(
                request, success=False, error=precondition_decision.reason, audit_events=events
            )

        model_decision, effective_model_id, fallback_from, model_events = self._resolve_model(
            request, model_id
        )
        events.extend(model_events)
        if not model_decision.allowed:
            return AgentCallOutcome.from_request(
                request, success=False, error=model_decision.reason, audit_events=events
            )

        outcome_metadata = {"model_id": effective_model_id}
        if fallback_from is not None:
            outcome_metadata["model_fallback_from"] = fallback_from

        authorized_tools = self.tools_check(request.agent_name)
        tool_names = tuple(getattr(tool, "name", str(tool)) for tool in authorized_tools)
        required_capabilities = AGENT_REQUIRED_CAPABILITIES.get(request.agent_name, frozenset())
        if ModelCapability.TOOL_CALLING in required_capabilities and not authorized_tools:
            events.append(
                _build_audit_event(
                    request,
                    action=EVENT_REFUSAL,
                    outcome="DENY",
                    model_id=effective_model_id,
                    policy="NO_AUTHORIZED_TOOLS",
                    final_status="NO_AUTHORIZED_TOOLS",
                )
            )
            return AgentCallOutcome.from_request(
                request,
                success=False,
                error=StructuredError(
                    code="NO_AUTHORIZED_TOOLS",
                    message=(
                        f"Aucun outil autorisé pour l'agent {request.agent_name.value!r}, "
                        "requis pour son exécution (capacité TOOL_CALLING)."
                    ),
                    field="agent_name",
                ),
                audit_events=events,
            )

        events.append(
            _build_audit_event(
                request,
                action=EVENT_AUTHORIZATION,
                outcome="ALLOW",
                model_id=effective_model_id,
                tools=tool_names,
                policy="TOOLS_AUTHORIZED",
            )
        )

        agent_runner = self.agent_registry.get(request.agent_name)
        if agent_runner is None:
            events.append(
                _build_audit_event(
                    request,
                    action=EVENT_REFUSAL,
                    outcome="DENY",
                    model_id=effective_model_id,
                    tools=tool_names,
                    policy="AGENT_NOT_REGISTERED",
                    final_status="AGENT_NOT_REGISTERED",
                )
            )
            return AgentCallOutcome.from_request(
                request,
                success=False,
                error=StructuredError(
                    code="AGENT_NOT_REGISTERED",
                    message=f"Aucun exécuteur enregistré pour {request.agent_name.value!r}.",
                    field="agent_name",
                ),
                audit_events=events,
            )

        return self._call_agent_with_retry(
            agent_runner,
            request,
            state,
            metadata=outcome_metadata,
            model_id=effective_model_id,
            tools=tool_names,
            prior_events=events,
        )

    def _resolve_model(
        self, request: AgentCallRequest, model_id: str
    ) -> tuple[PolicyDecision, str, str | None, tuple[AuditEvent, ...]]:
        """Résout le modèle à utiliser pour ``request.agent_name``.

        Tente d'abord ``model_id`` via ``self.model_check``. S'il est
        refusé, tente **un seul** candidat de repli — celui retourné par
        ``ModelRegistry.find_fallback`` (enregistré, activé, compatible avec
        l'agent) — rejoué à travers le **même** ``model_check`` : le
        fallback n'est jamais accepté par construction, seulement parce
        qu'il existe dans le registre. Sans candidat, ou si le candidat est
        lui-même refusé, l'erreur d'origine est retournée inchangée — jamais
        masquée par un code générique.

        Retourne ``(decision, model_id_effectif, model_id_initial_ou_None,
        audit_events)`` — le troisième élément n'est pas ``None`` que si un
        fallback a effectivement été retenu (utile pour journaliser et pour
        ``AgentCallOutcome.metadata``).
        """
        decision = self.model_check(self.model_registry, request.agent_name, model_id)
        if decision.allowed:
            event = _build_audit_event(
                request,
                action=EVENT_AUTHORIZATION,
                outcome="ALLOW",
                model_id=model_id,
                policy=decision.reason.code,
            )
            return decision, model_id, None, (event,)

        fallback_spec = self.model_registry.find_fallback(request.agent_name, exclude_model_id=model_id)
        if fallback_spec is None:
            event = _build_audit_event(
                request,
                action=EVENT_REFUSAL,
                outcome="DENY",
                model_id=model_id,
                policy=decision.reason.code,
                final_status=decision.reason.code,
            )
            return decision, model_id, None, (event,)

        fallback_decision = self.model_check(self.model_registry, request.agent_name, fallback_spec.model_id)
        if not fallback_decision.allowed:
            logger.warning(
                "Fallback modèle refusé pour l'agent %s : modèle initial %r indisponible "
                "(%s), candidat %r également refusé (%s) — panne d'origine conservée.",
                request.agent_name.value,
                model_id,
                decision.reason.code,
                fallback_spec.model_id,
                fallback_decision.reason.code,
            )
            fallback_event = _build_audit_event(
                request,
                action=EVENT_FALLBACK,
                outcome="REJECTED",
                model_id=fallback_spec.model_id,
                policy=fallback_decision.reason.code,
                final_status=decision.reason.code,
            )
            refusal_event = _build_audit_event(
                request,
                action=EVENT_REFUSAL,
                outcome="DENY",
                model_id=model_id,
                policy=decision.reason.code,
                final_status=decision.reason.code,
            )
            return decision, model_id, None, (fallback_event, refusal_event)

        logger.warning(
            "Fallback modèle appliqué pour l'agent %s : modèle initial %r indisponible "
            "(%s) → remplacement par %r.",
            request.agent_name.value,
            model_id,
            decision.reason.code,
            fallback_spec.model_id,
        )
        fallback_event = _build_audit_event(
            request,
            action=EVENT_FALLBACK,
            outcome="APPLIED",
            model_id=fallback_spec.model_id,
            policy=fallback_decision.reason.code,
        )
        return fallback_decision, fallback_spec.model_id, model_id, (fallback_event,)

    def _call_agent_with_retry(
        self,
        agent_runner: AgentRunner,
        request: AgentCallRequest,
        state: ClaimState,
        *,
        metadata: dict[str, str] | None = None,
        model_id: str = "",
        tools: Sequence[str] = (),
        prior_events: Sequence[AuditEvent] = (),
    ) -> AgentCallOutcome:
        """Appelle l'agent jusqu'à ``retry_policy.max_attempts`` fois.

        Même ``state`` et même contenu de requête (hormis ``attempt``, qui
        avance à chaque tentative) d'un essai à l'autre — même exécution,
        même dossier, même contexte. S'arrête dès le premier succès, ou dès
        qu'un échec n'est pas rejouable (``RetryPolicy.is_retryable``),
        sans attendre l'épuisement des tentatives.

        Émet un événement ``call`` par tentative, un événement ``retry``
        avant chaque nouvel essai décidé, et exactement un événement
        ``result`` final (succès ou échec) — accumulés à la suite de
        ``prior_events`` (préconditions/modèle/outils déjà émis par
        ``execute_agent``) dans ``AgentCallOutcome.audit_events``.
        """
        events: list[AuditEvent] = list(prior_events)
        outcome: AgentCallOutcome | None = None
        for attempt_number in range(1, self.retry_policy.max_attempts + 1):
            call_request = (
                request
                if attempt_number == request.attempt
                else request.model_copy(update={"attempt": attempt_number})
            )
            events.append(
                _build_audit_event(
                    call_request,
                    action=EVENT_CALL,
                    outcome="ATTEMPT",
                    model_id=model_id,
                    tools=tools,
                    attempt=attempt_number,
                )
            )
            outcome, exc = self._call_agent_once(agent_runner, call_request, state, metadata=metadata)

            retryable = (not outcome.success) and self.retry_policy.is_retryable(
                error_code=outcome.error.code, exc=exc
            )
            has_budget = attempt_number < self.retry_policy.max_attempts

            if outcome.success or not retryable or not has_budget:
                final_status = "SUCCESS" if outcome.success else outcome.error.code
                events.append(
                    _build_audit_event(
                        call_request,
                        action=EVENT_RESULT,
                        outcome=final_status,
                        model_id=model_id,
                        tools=tools,
                        attempt=attempt_number,
                        final_status=final_status,
                    )
                )
                return outcome.model_copy(update={"audit_events": tuple(events)})

            events.append(
                _build_audit_event(
                    call_request,
                    action=EVENT_RETRY,
                    outcome="SCHEDULED",
                    model_id=model_id,
                    tools=tools,
                    attempt=attempt_number + 1,
                )
            )
        return outcome.model_copy(update={"audit_events": tuple(events)})

    def _call_agent_once(
        self,
        agent_runner: AgentRunner,
        call_request: AgentCallRequest,
        state: ClaimState,
        *,
        metadata: dict[str, str] | None = None,
    ) -> tuple[AgentCallOutcome, Exception | None]:
        """Une tentative unique : appel de l'agent puis validation de sa
        sortie. Retourne l'exception levée par l'agent (``None`` sinon),
        nécessaire à ``RetryPolicy.is_retryable`` pour juger de la
        transience — jamais dérivée après coup du message d'erreur."""
        try:
            updates = agent_runner(state)
        except Exception as exc:  # noqa: BLE001
            outcome = AgentCallOutcome.from_request(
                call_request,
                success=False,
                error=StructuredError(
                    code="AGENT_EXECUTION_FAILED",
                    message=f"{type(exc).__name__} : {exc}",
                    field="agent_name",
                ),
                metadata=metadata,
            )
            return outcome, exc

        result_field = AGENT_RESULT_FIELD[call_request.agent_name]
        raw_result = updates.get(result_field) if isinstance(updates, dict) else None

        try:
            payload = validate_agent_result(call_request.agent_name, raw_result)
        except AgentResultValidationError as exc:
            return (
                AgentCallOutcome.from_request(
                    call_request, success=False, error=exc.structured, metadata=metadata
                ),
                None,
            )

        return (
            AgentCallOutcome.from_request(
                call_request,
                success=True,
                result_payload=payload,
                metadata=metadata,
                state_updates=updates if isinstance(updates, dict) else None,
            ),
            None,
        )
