"""Adaptateurs de nœuds LangGraph — ClaimShield Santé.

Point d'importation unique pour workflow.py : chaque fonction publique
(``node_<agent>``) wrape l'appel à ``agent.node(state)`` de l'agent concerné.

Responsabilités
---------------
1. **Déléguer** à ``agent.node(state)`` — aucune logique métier.
2. **Isoler** les exceptions inattendues : elles sont converties en entrées
   structurées dans ``state["errors"]`` sans être propagées à LangGraph.
3. **Valider** que le dict retourné contient bien une instance du modèle
   Pydantic attendu pour ce nœud (ou un dict valide comme tel).

Agents avec implémentation réelle (délégation directe)
------------------------------------------------------
claim_intake · security_gate · privacy · fhir_validator
medical_coding · document_ocr · identity_coverage

Agents avec interface injectable (stub NOT_EVALUATED/PENDING par défaut)
-----------------------------------------------------------------------
clinical_consistency · fraud_detection · case_reviewer · audit

Pour ces derniers, ``make_node_<name>(impl)`` permet d'injecter une
implémentation alternative sans modifier le code de production.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

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
)
from state.claim_state import ClaimState, validate_state_update


# ── Configuration immuable par agent ─────────────────────────────────────────


@dataclass(frozen=True)
class _AgentConfig:
    """Paramètres stables d'un nœud : noms de champs et modèle de résultat."""

    agent_name: str   # libellé court pour les messages d'erreur
    step_name: str    # valeur ajoutée dans completed_steps / current_step
    result_key: str   # clé du résultat dans ClaimState
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


def _make_node(agent_module: Any, config: _AgentConfig):
    """Génère une fonction nœud LangGraph wrappant ``agent_module.node``.

    Le module est résolu à l'appel (pas à la définition) pour permettre
    le monkey-patching dans les tests.
    """
    def _node(state: ClaimState) -> dict:
        try:
            updates = agent_module.node(state)
            _validate_result_type(updates, config.result_key, config.result_model)
            return updates
        except Exception as exc:  # noqa: BLE001
            return _exception_fallback(state, config, exc)

    _node.__name__ = f"node_{config.agent_name}"
    _node.__qualname__ = f"node_{config.agent_name}"
    return _node


# ── Nœuds publics ─────────────────────────────────────────────────────────────

node_claim_intake = _make_node(
    _claim_intake,
    _AgentConfig(
        agent_name="claim_intake",
        step_name="claim_intake",
        result_key="intake_result",
        result_model=ClaimIntakeResult,
        input_key="intake_input",
    ),
)

node_security_gate = _make_node(
    _security_gate,
    _AgentConfig(
        agent_name="security_gate",
        step_name="security_gate",
        result_key="security_result",
        result_model=SecurityGateResult,
        input_key="security_input",
    ),
)

node_privacy = _make_node(
    _privacy,
    _AgentConfig(
        agent_name="privacy",
        step_name="privacy",
        result_key="privacy_result",
        result_model=PrivacyResult,
        input_key="privacy_input",
    ),
)

node_fhir_validator = _make_node(
    _fhir_validator,
    _AgentConfig(
        agent_name="fhir_validator",
        step_name="fhir_validation",
        result_key="fhir_result",
        result_model=FhirValidatorResult,
        input_key="fhir_input",
    ),
)

node_medical_coding = _make_node(
    _medical_coding,
    _AgentConfig(
        agent_name="medical_coding",
        step_name="medical_coding",
        result_key="coding_result",
        result_model=MedicalCodingResult,
        input_key="coding_input",
    ),
)

node_document_ocr = _make_node(
    _document_ocr,
    _AgentConfig(
        agent_name="document_ocr",
        step_name="document_ocr_agent",
        result_key="ocr_result",
        result_model=DocumentOcrResult,
        input_key="ocr_input",
    ),
)

node_identity_coverage = _make_node(
    _identity_coverage,
    _AgentConfig(
        agent_name="identity_coverage",
        step_name="identity_coverage",
        result_key="identity_coverage_result",
        result_model=IdentityCoverageResult,
        input_key="identity_coverage_input",
    ),
)

# ── Nœuds stubs — agents non encore implémentés ──────────────────────────────
#
# Ces nœuds utilisent les stubs NOT_EVALUATED/PENDING par défaut.
# Pour injecter une implémentation : utiliser make_node_<name>(impl=MyImpl()).


def make_node_clinical_consistency(
    impl: ClinicalConsistencyRunnable | None = None,
) -> Callable[[ClaimState], dict]:
    """Crée un nœud clinical_consistency avec isolation d'exceptions.

    Args:
        impl: Implémentation injectable.  None utilise le stub NOT_EVALUATED.
    """
    config = _AgentConfig(
        agent_name="clinical_consistency",
        step_name="clinical_consistency",
        result_key="clinical_result",
        result_model=ClinicalConsistencyResult,
    )
    if impl is not None:
        inner = _clinical_consistency.make_node(impl)

        def _injected(state: ClaimState) -> dict:
            try:
                updates = inner(state)
                _validate_result_type(updates, config.result_key, config.result_model)
                return updates
            except Exception as exc:  # noqa: BLE001
                return _exception_fallback(state, config, exc)

        _injected.__name__ = "node_clinical_consistency"
        return _injected

    return _make_node(_clinical_consistency, config)


node_clinical_consistency = make_node_clinical_consistency()


def make_node_fraud_detection(
    impl: FraudDetectionRunnable | None = None,
) -> Callable[[ClaimState], dict]:
    """Crée un nœud fraud_detection avec isolation d'exceptions.

    Args:
        impl: Implémentation injectable.  None utilise le stub NOT_EVALUATED.
    """
    config = _AgentConfig(
        agent_name="fraud_detection",
        step_name="fraud_detection",
        result_key="fraud_result",
        result_model=FraudDetectionResult,
    )
    if impl is not None:
        inner = _fraud_detection.make_node(impl)

        def _injected(state: ClaimState) -> dict:
            try:
                updates = inner(state)
                _validate_result_type(updates, config.result_key, config.result_model)
                return updates
            except Exception as exc:  # noqa: BLE001
                return _exception_fallback(state, config, exc)

        _injected.__name__ = "node_fraud_detection"
        return _injected

    return _make_node(_fraud_detection, config)


node_fraud_detection = make_node_fraud_detection()


def make_node_case_reviewer(
    impl: CaseReviewerRunnable | None = None,
) -> Callable[[ClaimState], dict]:
    """Crée un nœud case_reviewer avec isolation d'exceptions.

    Args:
        impl: Implémentation injectable.  None utilise le stub PENDING.
    """
    config = _AgentConfig(
        agent_name="case_reviewer",
        step_name="case_reviewer",
        result_key="review_result",
        result_model=CaseReviewerResult,
    )
    if impl is not None:
        inner = _case_reviewer.make_node(impl)

        def _injected(state: ClaimState) -> dict:
            try:
                updates = inner(state)
                _validate_result_type(updates, config.result_key, config.result_model)
                return updates
            except Exception as exc:  # noqa: BLE001
                return _exception_fallback(state, config, exc)

        _injected.__name__ = "node_case_reviewer"
        return _injected

    return _make_node(_case_reviewer, config)


node_case_reviewer = make_node_case_reviewer()


def make_node_audit(
    impl: AuditAgentRunnable | None = None,
) -> Callable[[ClaimState], dict]:
    """Crée un nœud audit avec isolation d'exceptions.

    Args:
        impl: Implémentation injectable.  None utilise le stub NOT_EVALUATED.
    """
    config = _AgentConfig(
        agent_name="audit",
        step_name="audit",
        result_key="audit_result",
        result_model=AuditResult,
    )
    if impl is not None:
        inner = _audit.make_node(impl)

        def _injected(state: ClaimState) -> dict:
            try:
                updates = inner(state)
                _validate_result_type(updates, config.result_key, config.result_model)
                return updates
            except Exception as exc:  # noqa: BLE001
                return _exception_fallback(state, config, exc)

        _injected.__name__ = "node_audit"
        return _injected

    return _make_node(_audit, config)


node_audit = make_node_audit()


# ── Registre ───────────────────────────────────────────────────────────────────

NODE_REGISTRY: dict[str, Any] = {
    # Agents avec implémentation réelle
    "claim_intake": node_claim_intake,
    "security_gate": node_security_gate,
    "privacy": node_privacy,
    "fhir_validator": node_fhir_validator,
    "medical_coding": node_medical_coding,
    "document_ocr": node_document_ocr,
    "identity_coverage": node_identity_coverage,
    # Agents avec interface injectable (stub par défaut)
    "clinical_consistency": node_clinical_consistency,
    "fraud_detection": node_fraud_detection,
    "case_reviewer": node_case_reviewer,
    "audit": node_audit,
}
