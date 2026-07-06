"""Adaptateurs de nœuds LangGraph — ClaimShield Santé.

Point d'importation unique pour workflow.py : chaque nœud est construit par
``build_node_registry(orchestrator)`` et **appelle l'agent exclusivement via
``Orchestrator.execute_agent()``** (``orchestrator/executor.py``) — plus
aucun appel direct à ``agent_module.node(state)`` depuis ce module. Le
« qui peut appeler quel agent, avec quel modèle et quels outils » ainsi que
la journalisation d'audit sont désormais entièrement la responsabilité de
l'orchestrateur ; ``graph/nodes.py`` ne fait plus que :

1. **Construire** la requête d'appel (``AgentCallRequest``) à partir du state.
2. **Traduire** l'``AgentCallOutcome`` retourné en mise à jour partielle de
   ``ClaimState`` — même forme que celle définie à l'étape 10
   (``result_key``, ``current_step``, ``completed_steps``, champ d'entrée
   remis à ``None``, ``errors`` en cas d'échec) — jamais de logique métier.
3. **Isoler** toute exception de construction de requête (state invalide) :
   convertie en entrée structurée dans ``state["errors"]``, jamais propagée.

``Orchestrator`` est **injecté** — construit soit par l'appelant, soit par
``build_orchestrator()`` (appelée par défaut depuis
``graph/workflow.py::build_workflow()`` si aucun orchestrateur n'est fourni).
Aucune instance globale cachée dans ce module.

Agents avec implémentation réelle
---------------------------------
claim_intake · security_gate · privacy · fhir_validator
medical_coding · document_ocr · identity_coverage
clinical_consistency · fraud_detection · case_reviewer

Agents avec interface injectable (stub NOT_EVALUATED par défaut)
----------------------------------------------------------------
audit

``clinical_consistency`` et ``fraud_detection`` conservent eux aussi leur
point d'injection (``*_impl``, voir ``agents/clinical_consistency_agent/agent.py``
et ``agents/fraud_detection_agent/agent.py``) pour l'injection de tests, mais
leur implémentation par défaut exécute désormais une évaluation réelle
(Phase A déterministe + Phase B LLM) — ce n'est plus un stub NOT_EVALUATED.
``case_reviewer`` suit la même convention d'injection, avec une implémentation
LLM réelle par défaut qui produit une pré-recommandation non finale et force la
revue humaine.

Pour case_reviewer et audit, l'implémentation alternative se fournit à
``build_orchestrator(*_impl=...)`` (ou ``build_workflow(*_impl=...)``, qui la
lui transmet) — jamais en modifiant ce module.

Retry technique automatique (erreurs transitoires)
---------------------------------------------------
Anciennement porté par ce module (étape 11, ``tenacity.Retrying``), ce
mécanisme est désormais celui d'``Orchestrator.retry_policy``
(``orchestrator/executor.py::RetryPolicy``) : ``build_orchestrator()``
l'injecte avec ``max_attempts = Settings.claimshield_max_node_retry_attempts``
et les mêmes exceptions transitoires qu'avant
(``httpx.ConnectError``, ``httpx.TimeoutException``, ``ConnectionError``).
Comportement inchangé côté ``ClaimState`` : une panne non catégorisée
échoue immédiatement, sans retry.

Préconditions allégées pour l'intégration LangGraph
----------------------------------------------------
``orchestrator.routing.evaluate_call_preconditions`` (contrôle générique de
l'orchestrateur, utile à un futur appelant hors LangGraph) suppose un
``current_step`` aligné sur les valeurs d'``AgentName``
(``orchestrator.routing.AGENT_PIPELINE_ORDER``) — or certains agents réels
écrivent en interne un ``current_step`` différent de leur propre nom de nœud
(``document_ocr`` écrit ``"document_ocr_agent"``, ``fhir_validator`` écrit
``"fhir_validation"`` — voir CLAUDE.md). Rejouer ce contrôle strict ici
bloquerait le pipeline nominal pour une raison purement nominale, alors que
LangGraph garantit déjà structurellement l'ordre d'exécution via ses arêtes
conditionnelles (``graph/edges.py``, câblées dans ``graph/workflow.py``).
``_graph_preconditions_check`` ne revérifie donc que la cohérence
``case_id`` — la seule anomalie qu'une arête de graphe ne peut pas exclure.

Exception unique et vérifiée à l'appel direct d'un agent (``.node``/``.run``)
------------------------------------------------------------------------------
La seule fonction de ce module autorisée à référencer
``agent_module.node``/``agent_module.run`` est celle nommée par
``_ORCHESTRATOR_REGISTRATION_FUNCTION`` (``build_orchestrator``) — et
uniquement pour **enregistrer** l'appelable dans ``agent_registry`` (fermeture
résolue à l'appel, jamais exécutée immédiatement) ; ``Orchestrator`` reste le
seul à l'invoquer réellement, via ``execute_agent()``. Cette exception est
vérifiée statiquement (analyse AST, pas à l'exécution) par
``tests/graph/test_architecture.py`` — importer
``_ORCHESTRATOR_REGISTRATION_FUNCTION`` depuis ce module plutôt que de
recopier le nom en dur y évite un contrôle purement décoratif qui se
désynchroniserait silencieusement d'un renommage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pydantic import ValidationError

from agents.audit_agent import agent as _audit
from agents.audit_agent.agent import AuditAgentRunnable
from agents.case_reviewer_agent import agent as _case_reviewer
from agents.case_reviewer_agent.agent import CaseReviewerRunnable
from agents.claim_intake_agent import agent as _claim_intake
from agents.clinical_consistency_agent import agent as _clinical_consistency
from agents.clinical_consistency_agent.agent import ClinicalConsistencyRunnable
from agents.document_ocr_agent import agent as _document_ocr
from agents.fhir_validator_agent import agent as _fhir_validator
from agents.fraud_detection_agent import agent as _fraud_detection
from agents.fraud_detection_agent.agent import FraudDetectionRunnable
from agents.identity_coverage_agent import agent as _identity_coverage
from agents.medical_coding_agent import agent as _medical_coding
from agents.privacy_agent import agent as _privacy
from agents.security_gate_agent import agent as _security_gate
from config.settings import get_settings
from orchestrator.executor import AgentRunner, Orchestrator, RetryPolicy
from orchestrator.model_registry import ModelRegistry, build_default_registry
from orchestrator.orchestrator import AgentCallOutcome, AgentCallRequest, AgentName, without_computed_fields
from orchestrator.policies import PolicyDecision, PolicyEffect
from schemas.results import (
    AuditResult,
    CaseReviewerResult,
    ClaimIntakeResult,
    ClinicalConsistencyResult,
    DocumentOcrResult,
    FhirValidatorResult,
    FraudDetectionResult,
    IdentityCoverageResult,
    MedicalCodingResult,
    PrivacyResult,
    SecurityGateResult,
    StructuredError,
)
from state.claim_state import ClaimState, validate_state_update

# ── Exception unique, nommée et vérifiée, à l'appel direct d'un agent ────────

_ORCHESTRATOR_REGISTRATION_FUNCTION = "build_orchestrator"
"""Nom de l'unique fonction de ce module autorisée à écrire
``agent_module.node(...)``/``agent_module.run(...)`` — uniquement pour
enregistrer la fermeture dans ``agent_registry``, jamais pour l'exécuter sur
place. Toute autre fonction de ce module (en particulier le nœud construit
par ``_build_node``) ne doit invoquer que
``Orchestrator.execute_agent()``. Référencé par nom depuis
``tests/graph/test_architecture.py`` plutôt que recopié en dur, pour que le
contrôle architectural ne se désynchronise jamais silencieusement d'un
renommage — voir le docstring du module."""

_AGENT_MODULE_ALIASES = frozenset({
    "_claim_intake", "_security_gate", "_privacy", "_fhir_validator",
    "_medical_coding", "_document_ocr", "_identity_coverage",
    "_clinical_consistency", "_fraud_detection", "_case_reviewer", "_audit",
})
"""Noms des alias d'import des 11 modules agents (``from agents.<nom> import
agent as _<alias>``) — seule liste que ``build_orchestrator`` a le droit
d'appeler via ``.node``/``.run``. Réutilisée par le test architectural pour
ne rechercher que de véritables agents, jamais un faux positif accidentel
(ex. une variable locale nommée ``node``)."""


# ── Configuration immuable par agent ─────────────────────────────────────────


@dataclass(frozen=True)
class _AgentConfig:
    """Paramètres stables d'un nœud : identité orchestrateur, noms de champs
    et modèle de résultat."""

    agent_name: str          # libellé court pour les messages d'erreur
    agent_enum: AgentName    # identité orchestrateur (AgentCallRequest.agent_name)
    step_name: str           # valeur ajoutée dans completed_steps / current_step
    result_key: str          # clé du résultat dans ClaimState
    result_model: type
    input_key: str | None = None  # clé d'entrée consommée (remise à None)


# ── Helpers internes ───────────────────────────────────────────────────────────


def _validate_result_type(updates: dict, key: str, model: type) -> None:
    """Vérifie que updates[key] est une instance valide du modèle attendu.

    None est accepté (le nœud peut avoir omis le champ dans un cas d'erreur).
    Un dict est accepté s'il passe model.model_validate().
    Lève TypeError ou ValidationError sinon.
    """
    value = updates.get(key)
    if value is None:
        return
    if isinstance(value, model):
        return
    if isinstance(value, dict):
        model.model_validate(value)
        return
    raise TypeError(
        f"{key} : type {type(value).__name__!r} invalide, attendu {model.__name__!r}"
    )


def _exception_fallback(state: ClaimState, config: _AgentConfig, exc: Exception) -> dict:
    """Construit une mise à jour minimale de state pour une exception inattendue.

    Le type et le message de l'exception sont préservés dans ``errors``.
    Le champ d'entrée est remis à None pour éviter tout retraitement accidentel.
    """
    updates: dict = {
        "errors": [
            f"[{config.agent_name}] Exception inattendue "
            f"({type(exc).__name__}) : {exc}"
        ],
        "completed_steps": [config.step_name],
        "current_step": config.step_name,
    }
    if config.input_key is not None:
        updates[config.input_key] = None
    validate_state_update(updates)
    return updates


# ── Requête d'appel et préconditions allégées (voir docstring du module) ────


def _build_request(state: ClaimState, config: _AgentConfig) -> AgentCallRequest:
    """Construit la requête d'appel de ``config.agent_enum`` à partir du state.

    ``current_step`` retombe sur ``config.step_name`` si le state n'en porte
    pas encore (ex. tout premier appel) — jamais une chaîne vide, qui
    échouerait la validation ``AgentCallRequest`` (``min_length=1``).
    """
    case_id = state.get("case_id")
    current_step = state.get("current_step") or config.step_name
    return AgentCallRequest(
        agent_name=config.agent_enum,
        case_id=str(case_id) if case_id is not None else "",
        current_step=str(current_step),
        requested_model=config.result_model.__name__,
    )


def _graph_preconditions_check(state: ClaimState, request: AgentCallRequest) -> PolicyDecision:
    """Précondition allégée pour l'intégration LangGraph — voir docstring du
    module : l'ordre d'exécution est déjà garanti par la topologie du graphe,
    seule la cohérence ``case_id`` est revérifiée ici."""
    case_id = state.get("case_id")
    if case_id is not None and case_id != request.case_id:
        return PolicyDecision(
            effect=PolicyEffect.DENY,
            reason=StructuredError(
                code="CASE_ID_MISMATCH",
                message=(
                    f"Requête pour {request.case_id!r} appliquée à un état "
                    f"portant case_id={case_id!r}."
                ),
                field="case_id",
            ),
        )
    return PolicyDecision(
        effect=PolicyEffect.ALLOW,
        reason=StructuredError(
            code="GRAPH_TOPOLOGY_GUARANTEES_ORDER",
            message=(
                f"Ordre d'exécution garanti par la topologie du graphe pour "
                f"{request.agent_name.value!r}."
            ),
            field="agent_name",
        ),
    )


# ── Traduction AgentCallOutcome -> mise à jour partielle de ClaimState ───────


def _translate_outcome(config: _AgentConfig, outcome: AgentCallOutcome) -> dict:
    """Traduit un ``AgentCallOutcome`` en mise à jour ``ClaimState`` — même
    forme que celle définie à l'étape 10.

    Succès : ``outcome.state_updates`` (bookkeeping déjà produit par l'agent
    — ``current_step``, ``completed_steps``, ``errors``/``alerts``
    conditionnels, champ d'entrée consommé, propre ``audit_trail`` métier)
    est repris tel quel, seul ``result_key`` est remplacé par l'instance
    validée par l'orchestrateur (jamais le dict brut). Échec : repli
    structuré équivalent à l'ancien ``_exception_fallback``, attribué au bon
    agent via ``outcome.error``. Dans les deux cas, les événements d'audit de
    l'orchestrateur (``outcome.audit_events``) sont ajoutés (append) à
    ``audit_trail`` — jamais substitués à celui déjà produit par l'agent."""
    audit_events = list(outcome.audit_events)

    if outcome.success:
        updates = dict(outcome.state_updates)
        payload = without_computed_fields(config.result_model, outcome.result_payload)
        updates[config.result_key] = config.result_model.model_validate(payload)
        _validate_result_type(updates, config.result_key, config.result_model)
        if audit_events:
            updates["audit_trail"] = list(updates.get("audit_trail") or []) + audit_events
        return updates

    updates = {
        "errors": [f"[{config.agent_name}] {outcome.error.code} : {outcome.error.message}"],
        "completed_steps": [config.step_name],
        "current_step": config.step_name,
    }
    if config.input_key is not None:
        updates[config.input_key] = None
    if audit_events:
        updates["audit_trail"] = audit_events
    validate_state_update(updates)
    return updates


def _build_node(orchestrator: Orchestrator, config: _AgentConfig) -> Callable[[ClaimState], dict]:
    """Génère la fonction nœud LangGraph pour ``config`` — appelle
    exclusivement ``orchestrator.execute_agent()``, jamais l'agent
    directement."""

    def _node(state: ClaimState) -> dict:
        try:
            request = _build_request(state, config)
        except ValidationError as exc:
            return _exception_fallback(state, config, exc)

        model_id = get_settings().claimshield_llm_model
        outcome = orchestrator.execute_agent(request, state, model_id=model_id)
        return _translate_outcome(config, outcome)

    _node.__name__ = f"node_{config.agent_name}"
    _node.__qualname__ = f"node_{config.agent_name}"
    return _node


# ── Configuration des 11 agents ───────────────────────────────────────────────

_AGENT_CONFIGS: dict[str, _AgentConfig] = {
    "claim_intake": _AgentConfig(
        agent_name="claim_intake",
        agent_enum=AgentName.CLAIM_INTAKE,
        step_name="claim_intake",
        result_key="intake_result",
        result_model=ClaimIntakeResult,
        input_key="intake_input",
    ),
    "security_gate": _AgentConfig(
        agent_name="security_gate",
        agent_enum=AgentName.SECURITY_GATE,
        step_name="security_gate",
        result_key="security_result",
        result_model=SecurityGateResult,
        input_key="security_input",
    ),
    "privacy": _AgentConfig(
        agent_name="privacy",
        agent_enum=AgentName.PRIVACY,
        step_name="privacy",
        result_key="privacy_result",
        result_model=PrivacyResult,
        input_key="privacy_input",
    ),
    "fhir_validator": _AgentConfig(
        agent_name="fhir_validator",
        agent_enum=AgentName.FHIR_VALIDATOR,
        step_name="fhir_validation",
        result_key="fhir_result",
        result_model=FhirValidatorResult,
        input_key="fhir_input",
    ),
    "medical_coding": _AgentConfig(
        agent_name="medical_coding",
        agent_enum=AgentName.MEDICAL_CODING,
        step_name="medical_coding",
        result_key="coding_result",
        result_model=MedicalCodingResult,
        input_key="coding_input",
    ),
    "document_ocr": _AgentConfig(
        agent_name="document_ocr",
        agent_enum=AgentName.DOCUMENT_OCR,
        step_name="document_ocr_agent",
        result_key="ocr_result",
        result_model=DocumentOcrResult,
        input_key="ocr_input",
    ),
    "identity_coverage": _AgentConfig(
        agent_name="identity_coverage",
        agent_enum=AgentName.IDENTITY_COVERAGE,
        step_name="identity_coverage",
        result_key="identity_coverage_result",
        result_model=IdentityCoverageResult,
        input_key="identity_coverage_input",
    ),
    "clinical_consistency": _AgentConfig(
        agent_name="clinical_consistency",
        agent_enum=AgentName.CLINICAL_CONSISTENCY,
        step_name="clinical_consistency",
        result_key="clinical_result",
        result_model=ClinicalConsistencyResult,
    ),
    "fraud_detection": _AgentConfig(
        agent_name="fraud_detection",
        agent_enum=AgentName.FRAUD_DETECTION,
        step_name="fraud_detection",
        result_key="fraud_result",
        result_model=FraudDetectionResult,
    ),
    "case_reviewer": _AgentConfig(
        agent_name="case_reviewer",
        agent_enum=AgentName.CASE_REVIEWER,
        step_name="case_reviewer",
        result_key="review_result",
        result_model=CaseReviewerResult,
    ),
    "audit": _AgentConfig(
        agent_name="audit",
        agent_enum=AgentName.AUDIT,
        step_name="audit",
        result_key="audit_result",
        result_model=AuditResult,
    ),
}


# ── Construction de l'orchestrateur (injection) ──────────────────────────────


def build_orchestrator(
    *,
    clinical_consistency_impl: ClinicalConsistencyRunnable | None = None,
    fraud_detection_impl: FraudDetectionRunnable | None = None,
    case_reviewer_impl: CaseReviewerRunnable | None = None,
    audit_impl: AuditAgentRunnable | None = None,
    model_registry: ModelRegistry | None = None,
    retry_policy: RetryPolicy | None = None,
) -> Orchestrator:
    """Construit l'``Orchestrator`` utilisé par les nœuds réels du graphe.

    Aucune instance globale cachée : chaque appel construit un orchestrateur
    frais (ou réutilise ``model_registry``/``retry_policy`` fournis).
    ``graph/workflow.py::build_workflow()`` l'appelle par défaut si aucun
    ``orchestrator`` n'est explicitement injecté — voir docstring du module.

    Les agents réels sont enregistrés via des fermetures qui résolvent
    ``agent_module.node`` **à l'appel** (jamais capturé à la construction) —
    permet à un test de patcher ``agents.<nom>.agent.node`` après coup, comme
    avant l'introduction de l'orchestrateur.

    Les 4 agents à point d'injection (``clinical_consistency``,
    ``fraud_detection``, ``case_reviewer``, ``audit``) utilisent leur
    implémentation injectée (``*_impl``) si fournie ; sinon, ``clinical_consistency``
    et ``fraud_detection`` retombent sur leur évaluation réelle par défaut
    (étape 12) tandis que ``case_reviewer`` et ``audit`` retombent sur leur
    stub NOT_EVALUATED/PENDING — jamais importés en dur ailleurs que dans ce
    module.
    """
    registry = model_registry if model_registry is not None else build_default_registry()
    policy = (
        retry_policy
        if retry_policy is not None
        else RetryPolicy(max_attempts=get_settings().claimshield_max_node_retry_attempts)
    )

    agent_registry: dict[AgentName, AgentRunner] = {
        AgentName.CLAIM_INTAKE: lambda state: _claim_intake.node(state),
        AgentName.SECURITY_GATE: lambda state: _security_gate.node(state),
        AgentName.PRIVACY: lambda state: _privacy.node(state),
        AgentName.FHIR_VALIDATOR: lambda state: _fhir_validator.node(state),
        AgentName.MEDICAL_CODING: lambda state: _medical_coding.node(state),
        AgentName.DOCUMENT_OCR: lambda state: _document_ocr.node(state),
        AgentName.IDENTITY_COVERAGE: lambda state: _identity_coverage.node(state),
        AgentName.CLINICAL_CONSISTENCY: (
            _clinical_consistency.make_node(clinical_consistency_impl)
            if clinical_consistency_impl is not None
            else (lambda state: _clinical_consistency.node(state))
        ),
        AgentName.FRAUD_DETECTION: (
            _fraud_detection.make_node(fraud_detection_impl)
            if fraud_detection_impl is not None
            else (lambda state: _fraud_detection.node(state))
        ),
        AgentName.CASE_REVIEWER: (
            _case_reviewer.make_node(case_reviewer_impl)
            if case_reviewer_impl is not None
            else (lambda state: _case_reviewer.node(state))
        ),
        AgentName.AUDIT: (
            _audit.make_node(audit_impl)
            if audit_impl is not None
            else (lambda state: _audit.node(state))
        ),
    }

    return Orchestrator(
        model_registry=registry,
        agent_registry=agent_registry,
        preconditions_check=_graph_preconditions_check,
        retry_policy=policy,
    )


# ── Registre de nœuds ─────────────────────────────────────────────────────────


def build_node_registry(orchestrator: Orchestrator) -> dict[str, Callable[[ClaimState], dict]]:
    """Construit les 11 fonctions nœud LangGraph, câblées sur ``orchestrator``.

    Remplace l'ancien ``NODE_REGISTRY`` statique : les nœuds ne peuvent plus
    être pré-construits sans orchestrateur (aucune instance implicite créée
    à l'import de ce module — voir ``build_orchestrator``)."""
    return {name: _build_node(orchestrator, config) for name, config in _AGENT_CONFIGS.items()}
