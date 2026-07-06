"""Politiques d'autorisation — orchestrator/policies.py.

Politiques pures et testables : aucune E/S, aucun état mutable partagé,
aucun appel LLM. Trois allowlists indexées par agent (``AgentName``) —
agents, modèles, outils — chacune évaluée par une fonction pure qui
retourne une ``PolicyDecision`` (``ALLOW``/``DENY`` + motif structuré).

``build_authorized_tools`` et ``get_authorized_tool`` transforment
l'allowlist d'outils en contrôle d'accès exécutable : seuls les outils
listés sont physiquement exposés à un agent (ex. via
``create_react_agent(llm, tools=build_authorized_tools(agent_name))``), et
toute résolution dynamique par nom repasse par ``evaluate_tool_authorization``
— le prompt système n'est jamais considéré comme un mécanisme de permission
suffisant.

Réutilise sans dupliquer :
- ``AgentName`` (``orchestrator.orchestrator``) pour l'identité des agents —
  aucune liste concurrente de noms d'agents.
- ``ModelRegistry`` / ``AGENT_REQUIRED_CAPABILITIES``
  (``orchestrator.model_registry``) : l'autorisation « modèle » délègue
  entièrement à ``ModelRegistry.select_for_agent`` — aucune allowlist de
  ``model_id`` concurrente de celle du registre.
- Les outils ``@tool`` réels de chaque ``agents/<nom>/tools.py`` —
  ``ALLOWED_TOOLS_PER_AGENT`` est dérivé par introspection
  (``BaseTool.name``) plutôt que recopié à la main : ne peut pas diverger
  du code réel des agents.
- ``StructuredError`` (``schemas.results``) pour le motif de chaque décision.

Distinct de ``security/policies.py::ToolPolicy``, qui couvre un tout autre
périmètre : l'allowlist système bas niveau appliquée par le Security Gate
(``compute_sha256``, ``detect_mime_type``, ``scan_claim_fields``, ...), pas
les outils LLM propres à chaque agent.
"""
from __future__ import annotations

import enum
from types import ModuleType

from langchain_core.tools import BaseTool

from agents.claim_intake_agent import tools as _claim_intake_tools
from agents.clinical_consistency_agent import tools as _clinical_consistency_tools
from agents.document_ocr_agent import tools as _document_ocr_tools
from agents.fhir_validator_agent import tools as _fhir_validator_tools
from agents.fraud_detection_agent import tools as _fraud_detection_tools
from agents.identity_coverage_agent import tools as _identity_coverage_tools
from agents.medical_coding_agent import tools as _medical_coding_tools
from agents.privacy_agent import tools as _privacy_tools
from agents.security_gate_agent import tools as _security_gate_tools
from orchestrator.model_registry import ModelRegistry, ModelRegistryError
from orchestrator.orchestrator import AgentName
from schemas.domain import StrictModel
from schemas.results import StructuredError

# ── Décision de politique ─────────────────────────────────────────────────────


class PolicyEffect(str, enum.Enum):
    """Effet d'une décision de politique d'autorisation.

    Distinct de ``SecurityDecision`` (ALLOW/BLOCK/QUARANTINE, jugement de
    menace) et de ``PrivacyDecision`` (ALLOW/BLOCK, RBAC) : ``PolicyEffect``
    couvre exclusivement les autorisations agent/modèle/outil de cette
    couche orchestrateur.
    """

    ALLOW = "ALLOW"
    DENY = "DENY"


class PolicyDecision(StrictModel):
    """Décision d'autorisation, toujours accompagnée d'un motif structuré —
    aussi bien pour un ALLOW (traçabilité) que pour un DENY (refus)."""

    effect: PolicyEffect
    reason: StructuredError

    @property
    def allowed(self) -> bool:
        return self.effect is PolicyEffect.ALLOW


# ── Allowlist des agents ──────────────────────────────────────────────────────
# AgentName (orchestrator.orchestrator) est la seule source de vérité pour
# l'identité des agents — aucune liste concurrente ici.


def evaluate_agent_authorization(agent_name: str) -> PolicyDecision:
    """ALLOW si ``agent_name`` correspond à un ``AgentName`` connu, DENY sinon."""
    try:
        resolved = AgentName(agent_name)
    except ValueError:
        return PolicyDecision(
            effect=PolicyEffect.DENY,
            reason=StructuredError(
                code="AGENT_UNKNOWN",
                message=f"Agent inconnu de l'orchestrateur : {agent_name!r}",
                field="agent_name",
            ),
        )
    return PolicyDecision(
        effect=PolicyEffect.ALLOW,
        reason=StructuredError(
            code="AGENT_KNOWN",
            message=f"Agent reconnu : {resolved.value!r}",
            field="agent_name",
        ),
    )


# ── Allowlist des outils par agent ────────────────────────────────────────────

_TOOL_MODULE_BY_AGENT: dict[AgentName, ModuleType] = {
    AgentName.CLAIM_INTAKE: _claim_intake_tools,
    AgentName.SECURITY_GATE: _security_gate_tools,
    AgentName.PRIVACY: _privacy_tools,
    AgentName.FHIR_VALIDATOR: _fhir_validator_tools,
    AgentName.MEDICAL_CODING: _medical_coding_tools,
    AgentName.DOCUMENT_OCR: _document_ocr_tools,
    AgentName.IDENTITY_COVERAGE: _identity_coverage_tools,
    AgentName.CLINICAL_CONSISTENCY: _clinical_consistency_tools,
    AgentName.FRAUD_DETECTION: _fraud_detection_tools,
    # Agents sans module tools.py — absents de ce dict, traités via .get(..., ()) ci-dessous.
}


def _index_tools_by_name(module: ModuleType) -> dict[str, BaseTool]:
    """Introspecte un module ``agents/<nom>/tools.py`` : retourne
    ``{nom_reel: objet_outil}`` pour chaque ``@tool`` qu'il expose
    (``BaseTool.name``), jamais une liste recopiée à la main."""
    return {
        value.name: value
        for attr_name in dir(module)
        if not attr_name.startswith("_")
        for value in (getattr(module, attr_name),)
        if isinstance(value, BaseTool)
    }


_TOOL_OBJECTS_BY_AGENT: dict[AgentName, dict[str, BaseTool]] = {
    agent_name: _index_tools_by_name(module)
    for agent_name, module in _TOOL_MODULE_BY_AGENT.items()
}
"""Objets ``BaseTool`` réels, indexés par nom, pour chaque agent disposant
d'un ``tools.py``. Source unique pour ``ALLOWED_TOOLS_PER_AGENT`` (les noms)
et ``build_authorized_tools``/``get_authorized_tool`` (les objets) — les
deux ne peuvent donc jamais diverger l'un de l'autre."""

ALLOWED_TOOLS_PER_AGENT: dict[AgentName, frozenset[str]] = {
    agent_name: frozenset(tools_by_name)
    for agent_name, tools_by_name in _TOOL_OBJECTS_BY_AGENT.items()
}
"""Outils ``@tool`` autorisés par agent, dérivés par introspection des
modules ``agents/<nom>/tools.py`` réels. Les agents sans ``tools.py``
(clinical_consistency, fraud_detection, case_reviewer, audit) n'ont pas d'entrée :
``evaluate_tool_authorization`` retombe alors sur un frozenset vide — aucun
outil n'est autorisé pour un agent qui n'a pas encore de tools.py."""


def evaluate_tool_authorization(agent_name: AgentName, tool_name: str) -> PolicyDecision:
    """ALLOW si ``tool_name`` fait partie des outils réels exposés par
    ``agents/<agent_name>/tools.py``, DENY sinon (y compris pour les 4
    agents sans ``tools.py``, qui n'ont aucun outil autorisé)."""
    allowed = ALLOWED_TOOLS_PER_AGENT.get(agent_name, frozenset())
    if tool_name in allowed:
        return PolicyDecision(
            effect=PolicyEffect.ALLOW,
            reason=StructuredError(
                code="TOOL_AUTHORIZED",
                message=f"Outil {tool_name!r} autorisé pour l'agent {agent_name.value!r}.",
                field="tool_name",
            ),
        )
    return PolicyDecision(
        effect=PolicyEffect.DENY,
        reason=StructuredError(
            code="TOOL_NOT_AUTHORIZED_FOR_AGENT",
            message=(
                f"Outil {tool_name!r} non autorisé pour l'agent "
                f"{agent_name.value!r} — outils autorisés : {sorted(allowed)}"
            ),
            field="tool_name",
        ),
    )


class ToolAccessError(ValueError):
    """Levée quand un outil demandé n'est pas autorisé pour un agent.

    Toujours structurée (``StructuredError``, réutilisé depuis la
    ``PolicyDecision`` de ``evaluate_tool_authorization`` — même patron que
    ``ModelRegistryError``). Le refus n'est jamais contournable via le
    prompt système : c'est un contrôle de code, déclenché avant tout appel
    LLM ou exécution d'outil."""

    def __init__(self, structured: StructuredError) -> None:
        self.structured = structured
        super().__init__(structured.message)


def build_authorized_tools(agent_name: AgentName) -> tuple[BaseTool, ...]:
    """Construit la liste des outils réellement exposables à ``agent_name``.

    Ne retourne **que** les outils de son allowlist explicite
    (``ALLOWED_TOOLS_PER_AGENT`` / ``_TOOL_OBJECTS_BY_AGENT``) — jamais
    l'ensemble des outils du système. Destinée à alimenter directement
    ``create_react_agent(llm, tools=build_authorized_tools(agent_name))`` :
    un outil absent de cette liste n'est pas seulement « déconseillé » par
    un prompt, il est physiquement injoignable par le LLM, qui ne reçoit
    même pas sa description. Le prompt système n'est jamais le mécanisme de
    permission — seule cette liste l'est.

    Un agent sans ``tools.py`` reçoit un tuple vide.
    """
    return tuple(_TOOL_OBJECTS_BY_AGENT.get(agent_name, {}).values())


def get_authorized_tool(agent_name: AgentName, tool_name: str) -> BaseTool:
    """Résout un outil demandé dynamiquement (par son nom) pour un agent.

    Rejoue systématiquement ``evaluate_tool_authorization`` avant toute
    résolution — bloque explicitement :
      - un nom d'outil inconnu du système ;
      - un outil réel mais appartenant à un autre agent (tentative de
        contournement : demander par son nom un outil absent de la propre
        allowlist de l'agent, en espérant qu'aucun contrôle ne le refasse) ;
      - un agent sans allowlist d'outils.

    Lève ``ToolAccessError`` dans tous ces cas — jamais d'accès « par
    défaut » à un outil non listé, quel que soit le contenu du prompt
    système de l'agent appelant.
    """
    decision = evaluate_tool_authorization(agent_name, tool_name)
    if not decision.allowed:
        raise ToolAccessError(decision.reason)
    return _TOOL_OBJECTS_BY_AGENT[agent_name][tool_name]


# ── Allowlist des modèles par agent ───────────────────────────────────────────
# Délègue entièrement à ModelRegistry.select_for_agent (orchestrator.model_
# registry), qui applique déjà AGENT_REQUIRED_CAPABILITIES — pas de second
# calcul concurrent des capacités requises ici.


def evaluate_model_authorization(
    registry: ModelRegistry, agent_name: AgentName, model_id: str
) -> PolicyDecision:
    """ALLOW si ``registry`` expose ``model_id`` comme disponible et
    compatible avec ``agent_name`` (voir ``ModelRegistry.select_for_agent``),
    DENY sinon — le motif de refus est le ``StructuredError`` porté par
    l'exception du registre (absent / désactivé / incompatible), jamais
    reconstruit ici."""
    try:
        spec = registry.select_for_agent(agent_name, model_id)
    except ModelRegistryError as exc:
        return PolicyDecision(effect=PolicyEffect.DENY, reason=exc.structured)
    return PolicyDecision(
        effect=PolicyEffect.ALLOW,
        reason=StructuredError(
            code="MODEL_AUTHORIZED",
            message=(
                f"Modèle {spec.model_id!r} autorisé pour l'agent "
                f"{agent_name.value!r}."
            ),
            field="model_id",
        ),
    )
