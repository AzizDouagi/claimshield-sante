"""Registre injectable de modèles LLM — ClaimShield Santé.

Associe chaque modèle autorisé à un identifiant stable, une fabrique de
client différée (``client_factory``) et les capacités qu'il expose. Le
registre ne stocke jamais de clé API, d'URL ni d'instance de client déjà
construite — seule une fabrique sans argument (``Callable[[], Any]``) est
conservée, résolue à la demande (voir ``ModelSpec.client_factory`` et
``ModelRegistry.get_client``).

Aucun registre global caché : ``ModelRegistry`` est une classe injectable,
instanciée explicitement par l'appelant (aucun singleton créé à l'import de
ce module). ``build_default_registry()`` est une fabrique explicite qui
enregistre le modèle Ollama actuellement configuré (``config.settings``) —
elle doit être appelée par le code appelant, jamais implicitement.

Réutilise sans dupliquer :
- ``StructuredError`` (``schemas.results``) pour toutes les erreurs du
  registre, selon le même patron que ``services.storage.StorageError``.
- ``AgentName`` (``orchestrator.orchestrator``) pour l'identité des agents.
- ``llm.factory.get_llm`` comme fabrique du modèle Ollama par défaut.

``ModelRegistry.find_fallback`` propose, sans jamais l'imposer, un modèle de
remplacement — enregistré, activé et compatible avec l'agent — quand le
modèle demandé est refusé. La décision d'utiliser ce candidat (et sa
journalisation) reste à la charge de l'appelant (voir
``orchestrator.executor.Orchestrator._resolve_model``) : ce module ne fait
jamais de fallback implicite.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, fields
from typing import Any, Callable

from config.settings import get_settings
from llm.factory import get_llm
from orchestrator.orchestrator import AgentName
from schemas.results import StructuredError

# ── Capacités ──────────────────────────────────────────────────────────────


class ModelCapability(str, enum.Enum):
    """Capacité qu'un modèle doit exposer pour servir un agent donné.

    STRUCTURED_OUTPUT — ``llm.with_structured_output(...)`` (claim_intake,
    security_gate, privacy, fhir_validator, case_reviewer).
    TOOL_CALLING — ``create_react_agent(llm, tools=...)`` (medical_coding,
    document_ocr).
    """

    STRUCTURED_OUTPUT = "structured_output"
    TOOL_CALLING = "tool_calling"


AGENT_REQUIRED_CAPABILITIES: dict[AgentName, frozenset[ModelCapability]] = {
    AgentName.CLAIM_INTAKE: frozenset({ModelCapability.STRUCTURED_OUTPUT}),
    AgentName.SECURITY_GATE: frozenset({ModelCapability.STRUCTURED_OUTPUT}),
    AgentName.PRIVACY: frozenset({ModelCapability.STRUCTURED_OUTPUT}),
    AgentName.FHIR_VALIDATOR: frozenset({ModelCapability.STRUCTURED_OUTPUT}),
    AgentName.MEDICAL_CODING: frozenset({ModelCapability.TOOL_CALLING}),
    AgentName.DOCUMENT_OCR: frozenset({ModelCapability.TOOL_CALLING}),
    AgentName.CASE_REVIEWER: frozenset({ModelCapability.STRUCTURED_OUTPUT}),
    AgentName.CLINICAL_CONSISTENCY: frozenset({ModelCapability.TOOL_CALLING}),
    AgentName.FRAUD_DETECTION: frozenset({ModelCapability.TOOL_CALLING}),
    # Agents sans exigence déclarée à l'orchestrateur pour l'instant.
    AgentName.IDENTITY_COVERAGE: frozenset(),
    AgentName.AUDIT: frozenset(),
}
"""Capacités minimales requises par agent — source unique de vérité pour
``ModelRegistry.select_for_agent``."""

_FORBIDDEN_SPEC_FIELD_NAMES: frozenset[str] = frozenset({
    "api_key", "apikey", "secret", "token", "password", "credential",
    "base_url", "url", "endpoint", "client",
})
"""Noms de champs interdits sur ModelSpec — un modèle ne stocke jamais de
secret, d'URL ni de client déjà construit ; seule une fabrique différée
(``client_factory``) est autorisée. Vérifié structurellement à l'import
(``_assert_no_forbidden_fields``) plutôt que par une regex sur des valeurs
dynamiques : le registre ne contient que des identifiants et des fabriques,
jamais de valeur secrète à filtrer."""


# ── Spécification d'un modèle ────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelSpec:
    """Modèle autorisé : identifiant, fabrique de client différée, capacités.

    ``client_factory`` est un callable sans argument résolu à la demande
    (ex. ``llm.factory.get_llm``) — jamais un client déjà instancié, jamais
    une clé API ou une URL stockée directement.
    """

    model_id: str
    provider: str
    capabilities: frozenset[ModelCapability]
    client_factory: Callable[[], Any]
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id ne peut pas être vide")
        if not self.provider.strip():
            raise ValueError("provider ne peut pas être vide")
        if not isinstance(self.capabilities, frozenset):
            raise TypeError("capabilities doit être un frozenset[ModelCapability]")
        if not callable(self.client_factory):
            raise TypeError("client_factory doit être un callable sans argument")


def _assert_no_forbidden_fields() -> None:
    """Garde-fou exécuté à l'import : ModelSpec ne doit jamais gagner un
    champ secret/URL/client — lève RuntimeError sinon (échec explicite,
    pas un test silencieusement obsolète)."""
    names = {f.name for f in fields(ModelSpec)}
    leaked = names & _FORBIDDEN_SPEC_FIELD_NAMES
    if leaked:
        raise RuntimeError(
            f"ModelSpec contient des champs interdits : {sorted(leaked)} — "
            "aucune clé API, URL ou instance de client ne doit être stockée "
            "dans le registre."
        )


_assert_no_forbidden_fields()


# ── Erreurs structurées ───────────────────────────────────────────────────────


class ModelRegistryError(ValueError):
    """Erreur du registre de modèles — toujours structurée (StructuredError),
    même patron que ``services.storage.StorageError``."""

    def __init__(self, structured: StructuredError) -> None:
        self.structured = structured
        super().__init__(structured.message)


class ModelNotFoundError(ModelRegistryError):
    """Modèle absent du registre."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(
            StructuredError(
                code="MODEL_NOT_FOUND",
                message=f"Modèle inconnu du registre : {model_id!r}",
                field="model_id",
            )
        )


class ModelDisabledError(ModelRegistryError):
    """Modèle enregistré mais désactivé (``enabled=False``)."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(
            StructuredError(
                code="MODEL_DISABLED",
                message=f"Modèle désactivé : {model_id!r}",
                field="model_id",
            )
        )


class ModelIncompatibleError(ModelRegistryError):
    """Modèle disponible mais dépourvu d'une capacité requise par l'agent."""

    def __init__(
        self, model_id: str, agent_name: AgentName, missing: frozenset[ModelCapability]
    ) -> None:
        self.model_id = model_id
        self.agent_name = agent_name
        self.missing_capabilities = missing
        missing_labels = sorted(c.value for c in missing)
        super().__init__(
            StructuredError(
                code="MODEL_INCOMPATIBLE",
                message=(
                    f"Modèle {model_id!r} incompatible avec l'agent "
                    f"{agent_name.value!r} — capacité(s) manquante(s) : {missing_labels}"
                ),
                field="capabilities",
            )
        )


class ModelUnavailableError(ModelRegistryError):
    """Modèle valide et compatible, mais sa fabrique a échoué à l'appel
    (ex. Ollama injoignable) — distinct d'un refus de sélection."""

    def __init__(self, model_id: str, cause: Exception) -> None:
        self.model_id = model_id
        super().__init__(
            StructuredError(
                code="MODEL_UNAVAILABLE",
                message=(
                    f"Modèle {model_id!r} indisponible : "
                    f"{type(cause).__name__} : {cause}"
                ),
                field="model_id",
            )
        )


# ── Registre ───────────────────────────────────────────────────────────────────


class ModelRegistry:
    """Registre injectable de modèles autorisés.

    Chaque appelant construit et détient sa propre instance — aucun état
    global partagé entre deux ``ModelRegistry()``.
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelSpec] = {}

    def register(self, spec: ModelSpec) -> None:
        """Enregistre un modèle. Refuse tout écrasement silencieux d'un
        ``model_id`` déjà présent (même politique que ``StorageService`` :
        pas d'overwrite implicite)."""
        if spec.model_id in self._models:
            raise ValueError(f"Modèle déjà enregistré : {spec.model_id!r}")
        self._models[spec.model_id] = spec

    def is_registered(self, model_id: str) -> bool:
        return model_id in self._models

    def list_models(self) -> tuple[ModelSpec, ...]:
        return tuple(self._models.values())

    def get(self, model_id: str) -> ModelSpec:
        """Retourne la spécification d'un modèle disponible.

        Lève ``ModelNotFoundError`` si absent, ``ModelDisabledError`` si
        désactivé. Ne construit jamais de client (voir ``get_client``).
        """
        spec = self._models.get(model_id)
        if spec is None:
            raise ModelNotFoundError(model_id)
        if not spec.enabled:
            raise ModelDisabledError(model_id)
        return spec

    def select_for_agent(self, agent_name: AgentName, model_id: str) -> ModelSpec:
        """Valide qu'un modèle est disponible ET compatible avec un agent.

        Lève ``ModelNotFoundError``, ``ModelDisabledError`` (via ``get``)
        ou ``ModelIncompatibleError`` si une capacité requise manque.
        """
        spec = self.get(model_id)
        required = AGENT_REQUIRED_CAPABILITIES[agent_name]
        missing = required - spec.capabilities
        if missing:
            raise ModelIncompatibleError(model_id, agent_name, missing)
        return spec

    def get_client(self, model_id: str) -> Any:
        """Résout et retourne un client de modèle disponible.

        La fabrique n'est appelée qu'ici, jamais avant — aucune instance de
        client n'est conservée dans le registre. Toute exception levée par
        ``client_factory`` est enveloppée dans ``ModelUnavailableError``
        (distincte d'un refus de sélection).
        """
        spec = self.get(model_id)
        try:
            return spec.client_factory()
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailableError(model_id, exc) from exc

    def get_client_for_agent(self, agent_name: AgentName, model_id: str) -> Any:
        """Combine ``select_for_agent`` (refus) et l'appel différé de la
        fabrique (indisponibilité) — point d'entrée destiné aux agents."""
        spec = self.select_for_agent(agent_name, model_id)
        try:
            return spec.client_factory()
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailableError(model_id, exc) from exc

    def find_fallback(self, agent_name: AgentName, *, exclude_model_id: str) -> ModelSpec | None:
        """Retourne le premier modèle du registre — autre que
        ``exclude_model_id``, activé et compatible avec ``agent_name``
        (mêmes capacités requises que ``select_for_agent``) — ou ``None`` si
        aucun n'est disponible.

        Ordre déterministe : ordre d'enregistrement (``dict`` insertion
        order). Ne lève jamais — l'absence de fallback est une réponse
        valide, pas une erreur (voir ``Orchestrator._resolve_model``, qui
        conserve alors l'erreur d'origine sans la masquer)."""
        required = AGENT_REQUIRED_CAPABILITIES[agent_name]
        for spec in self._models.values():
            if spec.model_id == exclude_model_id or not spec.enabled:
                continue
            if required - spec.capabilities:
                continue
            return spec
        return None


# ── Registre de référence (construction explicite, jamais implicite) ────────


def build_default_registry() -> ModelRegistry:
    """Construit un ``ModelRegistry`` enregistrant le modèle Ollama
    actuellement configuré (``config.settings``).

    N'est jamais appelée à l'import de ce module : chaque appelant décide
    explicitement de construire (et de détenir) ce registre — pas de
    singleton caché.
    """
    settings = get_settings()
    registry = ModelRegistry()
    registry.register(
        ModelSpec(
            model_id=settings.claimshield_llm_model,
            provider=settings.claimshield_llm_provider,
            capabilities=frozenset(
                {ModelCapability.STRUCTURED_OUTPUT, ModelCapability.TOOL_CALLING}
            ),
            client_factory=get_llm,
            enabled=True,
        )
    )
    return registry
