"""Contrats d'appel d'un agent — ClaimShield Santé.

Modélise l'invocation d'un agent indépendamment du moteur d'exécution :
``AgentCallRequest`` (ce qu'un orchestrateur doit fournir pour appeler un
agent) et ``AgentCallOutcome`` (résultat typé — succès, erreur, tentative,
métadonnées). Cette couche ne dépend d'aucun module ``graph/`` (LangGraph)
pour rester légère et réutilisable par un futur point d'entrée non-LangGraph
(ex. API directe) — voir CLAUDE.md, rôle prévu de ``orchestrator/``.

Réutilise sans dupliquer :
- ``StructuredError`` (``schemas.results``) pour l'erreur du résultat.
- Les 11 schémas de résultat par agent (``schemas.results``) pour valider
  ``AgentCallOutcome.result_payload`` selon l'agent appelé.
- ``ClaimState`` (``state.claim_state``) pour borner ``authorized_context``
  aux champs réellement déclarés dans l'état partagé.
- ``graph.edges.RELAUNCH_RESULT_FIELDS`` (pur, sans LangGraph) pour 7 des 11
  entrées de ``AGENT_RESULT_FIELD``.

``validate_agent_result`` est le point de passage obligé de toute sortie
d'agent avant qu'elle ne devienne un ``AgentCallOutcome`` réussi : jamais un
dictionnaire brut ou du texte libre accepté tel quel, toujours revalidé
contre ``AGENT_RESULT_MODELS[agent_name]``, toute anomalie transformée en
``AgentResultValidationError`` structurée et attribuée à l'agent — jamais la
valeur brute fautive dans le message (potentiellement sensible). Distinct de
``graph/nodes.py::_validate_result_type`` : celui-ci tolère ``None`` (un
nœud LangGraph peut légitimement ne rien produire à une étape) et ne
retourne pas de payload normalisé — ``validate_agent_result`` sert un besoin
différent (contrat d'appel direct, hors LangGraph) et n'importe donc pas
``graph/nodes.py``.

Registre et logique de routage/dispatch : hors périmètre de ce module (voir
``orchestrator/policies.py``, ``orchestrator/routing.py`` et
``orchestrator/executor.py``).
"""
from __future__ import annotations

import enum
from typing import Any, Sequence

from pydantic import Field, ValidationError, model_validator

from graph.edges import RELAUNCH_RESULT_FIELDS
from schemas.domain import StrictModel
from schemas.results import (
    AuditEvent,
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
from state.claim_state import ClaimState

CASE_ID_PATTERN = r"^CLM-\d{4,}$"


class AgentName(str, enum.Enum):
    """Noms d'agents stables invocables par l'orchestrateur.

    Alignés sur les clés de ``graph.nodes.NODE_REGISTRY`` (non importé ici
    afin de ne pas coupler ce contrat au moteur LangGraph — voir le
    docstring du module).
    """

    CLAIM_INTAKE = "claim_intake"
    SECURITY_GATE = "security_gate"
    PRIVACY = "privacy"
    FHIR_VALIDATOR = "fhir_validator"
    MEDICAL_CODING = "medical_coding"
    DOCUMENT_OCR = "document_ocr"
    IDENTITY_COVERAGE = "identity_coverage"
    CLINICAL_CONSISTENCY = "clinical_consistency"
    FRAUD_DETECTION = "fraud_detection"
    CASE_REVIEWER = "case_reviewer"
    AUDIT = "audit"


AGENT_RESULT_MODELS: dict[AgentName, type[StrictModel]] = {
    AgentName.CLAIM_INTAKE: ClaimIntakeResult,
    AgentName.SECURITY_GATE: SecurityGateResult,
    AgentName.PRIVACY: PrivacyResult,
    AgentName.FHIR_VALIDATOR: FhirValidatorResult,
    AgentName.MEDICAL_CODING: MedicalCodingResult,
    AgentName.DOCUMENT_OCR: DocumentOcrResult,
    AgentName.IDENTITY_COVERAGE: IdentityCoverageResult,
    AgentName.CLINICAL_CONSISTENCY: ClinicalConsistencyResult,
    AgentName.FRAUD_DETECTION: FraudDetectionResult,
    AgentName.CASE_REVIEWER: CaseReviewerResult,
    AgentName.AUDIT: AuditResult,
}
"""Modèle de résultat attendu par agent — source unique de vérité utilisée
par ``AgentCallRequest`` (cohérence de ``requested_model``) et
``AgentCallOutcome`` (validation de ``result_payload``)."""

_ADDITIONAL_AGENT_RESULT_FIELDS: dict[AgentName, str] = {
    AgentName.AUDIT: "audit_result",
}
"""Champ ``ClaimState`` de l'agent absent de
``graph.edges.RELAUNCH_RESULT_FIELDS`` (``audit_agent`` reste un stub jamais
relançable par décision humaine — voir la docstring de
``graph.edges.RELAUNCH_TARGETS``), donc ajouté ici pour compléter
``AGENT_RESULT_FIELD``."""

AGENT_RESULT_FIELD: dict[AgentName, str] = {
    AgentName(name): field for name, field in RELAUNCH_RESULT_FIELDS.items()
} | _ADDITIONAL_AGENT_RESULT_FIELDS
"""Champ ``ClaimState`` prouvant qu'un agent a déjà produit un résultat pour
le dossier — une entrée par agent (11), dont 10 réutilisées directement de
``graph.edges.RELAUNCH_RESULT_FIELDS``. Source unique de vérité pour
``orchestrator/routing.py`` (préconditions) et ``orchestrator/executor.py``
(extraction du résultat après exécution)."""


# ── Validation de la sortie d'un agent ────────────────────────────────────────


def _sanitized_validation_error_fields(exc: ValidationError) -> list[str]:
    """Chemins de champs en erreur (``err['loc']``) uniquement — jamais
    ``err['input']`` (la valeur fautive) ni ``str(exc)`` (qui l'inclut par
    défaut) : un résultat d'agent peut contenir des données sensibles, une
    erreur de validation ne doit jamais les faire fuiter dans un message
    structuré potentiellement journalisé."""
    return sorted(
        {".".join(str(part) for part in err["loc"]) or "<racine>" for err in exc.errors()}
    )


class AgentResultValidationError(ValueError):
    """Levée quand la sortie brute d'un agent ne correspond pas au modèle
    Pydantic attendu — toujours structurée (``StructuredError``, même
    patron que ``ModelRegistryError``/``ToolAccessError``), toujours
    attribuée à l'agent concerné, jamais accompagnée de la valeur brute
    fautive (potentiellement sensible)."""

    def __init__(self, structured: StructuredError) -> None:
        self.structured = structured
        super().__init__(structured.message)


def without_computed_fields(model: type[StrictModel], payload: dict[str, Any]) -> dict[str, Any]:
    """Retire du ``payload`` les clés correspondant à un ``@computed_field``
    de ``model`` (ex. ``PrivacyResult.decision``, dérivé de ``status``).

    Un ``@computed_field`` apparaît dans ``model_dump()`` mais n'est jamais
    accepté par ``model_validate()`` (``extra='forbid'`` — ce n'est pas un
    champ assignable). Sans ce filtre, revalider un dict déjà issu de
    ``model_dump()`` (round-trip dump → validate, ex. dans
    ``AgentCallOutcome``) échouerait à tort sur ces champs en lecture seule
    alors que le résultat est parfaitement valide."""
    computed = model.model_computed_fields
    if not computed:
        return payload
    return {key: value for key, value in payload.items() if key not in computed}


def validate_agent_result(agent_name: AgentName, raw_result: Any) -> dict[str, Any]:
    """Valide la sortie brute d'un agent contre ``AGENT_RESULT_MODELS[agent_name]``
    et retourne son ``model_dump()`` — jamais le dictionnaire brut fourni par
    l'agent tel quel : même s'il ressemble déjà au bon format, il est
    toujours revalidé via ``model_validate()`` pour produire une véritable
    instance du modèle attendu avant d'être accepté comme résultat final.

    ``raw_result`` est typiquement ``updates.get(AGENT_RESULT_FIELD[agent_name])``,
    c'est-à-dire la valeur retournée par l'agent lui-même (``agents.<nom>.agent.node``).

    Rejette explicitement, sans jamais accepter un résultat partiel :
      - ``None`` — l'agent n'a produit aucun résultat (code
        ``AGENT_RESULT_MISSING``) ;
      - toute valeur qui n'est ni une instance du modèle attendu ni un
        ``dict`` — texte libre, liste, nombre, etc. (code
        ``AGENT_RESULT_UNSTRUCTURED``) — jamais tentée en validation Pydantic,
        catégoriquement rejetée ;
      - un ``dict`` qui ne valide pas contre le modèle attendu — champ
        manquant, mauvais type... (code ``AGENT_RESULT_INVALID``).

    Lève ``AgentResultValidationError`` dans les trois cas — jamais
    silencieux, jamais la valeur brute (potentiellement sensible) dans le
    message : seuls les chemins de champs en erreur y figurent.
    """
    model = AGENT_RESULT_MODELS[agent_name]

    if raw_result is None:
        raise AgentResultValidationError(
            StructuredError(
                code="AGENT_RESULT_MISSING",
                message=f"Aucun résultat produit par l'agent {agent_name.value!r}.",
                field="result",
            )
        )

    if isinstance(raw_result, model):
        return raw_result.model_dump()

    if not isinstance(raw_result, dict):
        raise AgentResultValidationError(
            StructuredError(
                code="AGENT_RESULT_UNSTRUCTURED",
                message=(
                    f"Résultat de l'agent {agent_name.value!r} rejeté : type "
                    f"{type(raw_result).__name__!r} inattendu — un dictionnaire "
                    "brut ou du texte libre n'est jamais accepté comme résultat "
                    f"final, seule une instance validée de {model.__name__!r} l'est."
                ),
                field="result",
            )
        )

    try:
        validated = model.model_validate(without_computed_fields(model, raw_result))
    except ValidationError as exc:
        raise AgentResultValidationError(
            StructuredError(
                code="AGENT_RESULT_INVALID",
                message=(
                    f"Résultat de l'agent {agent_name.value!r} invalide pour "
                    f"{model.__name__!r} — champs en erreur : "
                    f"{_sanitized_validation_error_fields(exc)}."
                ),
                field="result",
            )
        ) from exc

    return validated.model_dump()


_CLAIM_STATE_FIELDS: frozenset[str] = frozenset(ClaimState.__annotations__)
"""Champs connus de l'état partagé — borne ``AgentCallRequest.authorized_context``
à des champs qui existent réellement dans ``ClaimState``."""


class AgentCallRequest(StrictModel):
    """Contrat d'appel d'un agent.

    Ce qu'un orchestrateur doit fournir pour invoquer un agent : quel agent,
    pour quel dossier, à quelle étape, avec quel modèle de résultat attendu
    et quel sous-ensemble de l'état partagé il est autorisé à lire.
    """

    agent_name: AgentName
    case_id: str = Field(..., pattern=CASE_ID_PATTERN)
    current_step: str = Field(..., min_length=1, max_length=100)
    requested_model: str = Field(
        ...,
        min_length=1,
        description=(
            "Nom de classe du modèle Pydantic attendu en résultat — doit "
            "correspondre à AGENT_RESULT_MODELS[agent_name]."
        ),
    )
    authorized_context: frozenset[str] = Field(
        default_factory=frozenset,
        description="Champs de ClaimState que cet appel est autorisé à lire.",
    )
    attempt: int = Field(1, ge=1)

    @model_validator(mode="after")
    def _requested_model_matches_agent(self) -> "AgentCallRequest":
        expected = AGENT_RESULT_MODELS[self.agent_name].__name__
        if self.requested_model != expected:
            raise ValueError(
                f"requested_model {self.requested_model!r} incohérent avec "
                f"agent_name {self.agent_name.value!r} — attendu {expected!r}"
            )
        return self

    @model_validator(mode="after")
    def _authorized_context_is_known(self) -> "AgentCallRequest":
        unknown = self.authorized_context - _CLAIM_STATE_FIELDS
        if unknown:
            raise ValueError(
                "authorized_context contient des champs inconnus de "
                f"ClaimState : {sorted(unknown)}"
            )
        return self


class AgentCallOutcome(StrictModel):
    """Résultat typé d'un appel d'agent.

    Contient le statut (succès/échec), la tentative concernée et des
    métadonnées — jamais de contenu brut (document, texte OCR complet,
    secret) : ``result_payload`` est le ``model_dump()`` du résultat
    Pydantic de l'agent, déjà minimisé par construction (mêmes garanties que
    ``ClaimState``).

    ``audit_events`` reprend l'interface append-only existante
    (``AuditEvent``, déjà utilisée par ``agents/privacy_agent`` pour
    alimenter ``ClaimState.audit_trail``) — jamais un nouveau schéma
    d'événement concurrent. N'implémente pas l'Audit Agent (étape 12,
    toujours stub) : ``Orchestrator`` ne fait qu'émettre ces événements et
    les exposer ici, prêts à être ajoutés (append) par l'appelant à
    ``state["audit_trail"]``, exactement comme le fait déjà chaque nœud
    d'agent — jamais mutés ou réordonnés une fois émis.

    ``state_updates`` porte, uniquement quand ``success=True``, la mise à
    jour ``ClaimState`` brute déjà retournée par l'agent invoqué (bookkeeping
    de l'étape 10 : ``current_step``, ``completed_steps``, ``errors``/
    ``alerts`` conditionnels, champ d'entrée consommé, propre ``audit_trail``
    métier de l'agent) — distincte de ``result_payload`` (strictement
    validé) : ce champ n'est jamais revalidé en profondeur ici, il reste la
    responsabilité de l'agent qui l'a produit (inchangée depuis l'étape 10).
    Sert exclusivement à ce qu'un appelant intégré à LangGraph
    (``graph/nodes.py``) puisse reconstruire la même mise à jour partielle de
    ``ClaimState`` qu'avant l'introduction de l'orchestrateur, sans que
    celui-ci n'ait à connaître la moindre règle métier. Vide (``{}``) en cas
    d'échec.
    """

    agent_name: AgentName
    case_id: str = Field(..., pattern=CASE_ID_PATTERN)
    current_step: str = Field(..., min_length=1, max_length=100)
    attempt: int = Field(..., ge=1)
    success: bool
    result_payload: dict[str, Any] | None = None
    error: StructuredError | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    audit_events: tuple[AuditEvent, ...] = Field(default_factory=tuple)
    state_updates: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _success_xor_error(self) -> "AgentCallOutcome":
        if self.success:
            if self.error is not None:
                raise ValueError("error doit être None quand success=True")
            if self.result_payload is None:
                raise ValueError("result_payload obligatoire quand success=True")
        else:
            if self.error is None:
                raise ValueError("error obligatoire quand success=False")
            if self.result_payload is not None:
                raise ValueError("result_payload doit être None quand success=False")
        return self

    @model_validator(mode="after")
    def _result_payload_matches_agent_model(self) -> "AgentCallOutcome":
        if self.result_payload is not None:
            model = AGENT_RESULT_MODELS[self.agent_name]
            try:
                model.model_validate(without_computed_fields(model, self.result_payload))
            except ValidationError as exc:
                # Jamais {exc} brut : str(ValidationError) inclut les valeurs
                # fautives, potentiellement sensibles. Seuls les chemins de
                # champs en erreur sont exposés (_sanitized_validation_error_fields).
                raise ValueError(
                    f"result_payload invalide pour {self.agent_name.value!r} "
                    f"(attendu {model.__name__!r}) — champs en erreur : "
                    f"{_sanitized_validation_error_fields(exc)}"
                ) from exc
        return self

    @classmethod
    def from_request(
        cls,
        request: AgentCallRequest,
        *,
        success: bool,
        result_payload: dict[str, Any] | None = None,
        error: StructuredError | None = None,
        metadata: dict[str, str] | None = None,
        audit_events: Sequence[AuditEvent] | None = None,
        state_updates: dict[str, Any] | None = None,
    ) -> "AgentCallOutcome":
        """Construit le résultat d'un appel en reprenant l'identité de la
        requête (agent, dossier, étape, tentative) — évite toute
        incohérence entre une requête et son résultat."""
        return cls(
            agent_name=request.agent_name,
            case_id=request.case_id,
            current_step=request.current_step,
            attempt=request.attempt,
            success=success,
            result_payload=result_payload,
            error=error,
            metadata=metadata or {},
            audit_events=tuple(audit_events) if audit_events is not None else (),
            state_updates=dict(state_updates) if state_updates is not None else {},
        )
