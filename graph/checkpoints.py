"""Configuration centralisée des checkpoints LangGraph.

Ce qui est sauvegardé en checkpoint :
- le `ClaimState` minimisé : identifiants, progression, statuts, résultats
  Pydantic structurés des agents, erreurs/alertes, événements d'audit et
  références d'artefacts (`artifact_id`, chemins relatifs, hashes, provenance).

Ce qui reste hors checkpoint, dans le stockage sécurisé :
- documents PDF/images originaux ;
- texte OCR complet et pages brutes ;
- réponses brutes de modèle, prompts complets, secrets, clés API ou tokens.

Le checkpoint sert uniquement à reprendre le workflow. Les artefacts lourds
restent dans `storage/` et sont référencés par hash, chemin relatif sécurisé
ou identifiant d'artefact.
"""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from langgraph.checkpoint.memory import InMemorySaver

from config.settings import Settings, get_settings
from state.claim_state import validate_claim_state

THREAD_ID_CONFIG_KEY = "thread_id"


class CheckpointBackend(StrEnum):
    """Backends supportés par la factory de checkpointer."""

    MEMORY = "memory"
    SQLITE = "sqlite"
    POSTGRES = "postgres"


def _jsonable(value: Any) -> Any:
    """Convertit un state en structure JSON sans perdre les modèles Pydantic."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_computed_fields=True)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def serialize_checkpoint_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Retourne le state sous forme JSON-compatible canonique."""
    return _jsonable(dict(state))


def validate_checkpoint_state(state: Mapping[str, Any]) -> None:
    """Vérifie qu'un state peut être sérialisé sans contenu interdit.

    La validation réutilise `validate_state_update` pour refuser bytes, chemins
    absolus, documents bruts, texte OCR complet, prompts, réponses LLM brutes
    et secrets. Le JSON final représente ce que le checkpointer peut persister.
    """
    validate_claim_state(state)
    json.dumps(serialize_checkpoint_state(state), ensure_ascii=False)


def make_thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    """Construit la configuration LangGraph obligatoire pour la reprise."""
    if not thread_id or not thread_id.strip():
        raise ValueError("thread_id obligatoire pour les checkpoints LangGraph")
    return {"configurable": {THREAD_ID_CONFIG_KEY: thread_id.strip(), "checkpoint_ns": ""}}


def get_thread_id(config: Mapping[str, Any]) -> str:
    """Extrait le thread_id depuis une configuration LangGraph."""
    configurable = config.get("configurable")
    if not isinstance(configurable, Mapping):
        raise ValueError("Configuration LangGraph invalide : clé configurable absente")
    thread_id = configurable.get(THREAD_ID_CONFIG_KEY)
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError("Configuration LangGraph invalide : thread_id absent")
    return thread_id


def assert_same_thread_id(initial_config: Mapping[str, Any], resume_config: Mapping[str, Any]) -> None:
    """Garantit qu'une reprise utilise exactement le même thread_id."""
    initial_thread_id = get_thread_id(initial_config)
    resume_thread_id = get_thread_id(resume_config)
    if initial_thread_id != resume_thread_id:
        raise ValueError(
            "Reprise checkpoint refusée : thread_id différent "
            f"({initial_thread_id!r} != {resume_thread_id!r})"
        )


def get_checkpointer(
    *,
    backend: str | CheckpointBackend | None = None,
    settings: Settings | None = None,
):
    """Retourne le checkpointer LangGraph configuré pour l'environnement.

    Tests : `memory` avec `InMemorySaver`.
    Développement local : `sqlite` possible si `langgraph.checkpoint.sqlite`
    est disponible.
    Futur production : `postgres`, prévu via `LANGGRAPH_CHECKPOINT_POSTGRES_URL`
    et le paquet `langgraph-checkpoint-postgres`.
    """
    resolved_settings = settings or get_settings()
    backend_name = CheckpointBackend(
        str(backend or resolved_settings.langgraph_checkpoint_backend).lower()
    )

    if backend_name is CheckpointBackend.MEMORY:
        return InMemorySaver()

    if backend_name is CheckpointBackend.SQLITE:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        # Construction directe (pas `SqliteSaver.from_conn_string`, qui est un
        # context manager fermant la connexion à la sortie du `with` — ici le
        # checkpointer doit rester ouvert pour toute la durée de vie du
        # process, exactement comme `InMemorySaver`). `check_same_thread=False`
        # : même précaution que `from_conn_string` (connexion partagée entre
        # threads/requêtes, cf. commentaire dans la lib langgraph).
        db_path = Path(resolved_settings.langgraph_checkpoint_db)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        return SqliteSaver(conn)

    if backend_name is CheckpointBackend.POSTGRES:
        postgres_url = resolved_settings.langgraph_checkpoint_postgres_url
        if not postgres_url:
            raise ValueError(
                "LANGGRAPH_CHECKPOINT_POSTGRES_URL requis pour le backend postgres"
            )
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Backend checkpoint postgres configuré, mais "
                "langgraph-checkpoint-postgres n'est pas installé"
            ) from exc
        return PostgresSaver.from_conn_string(postgres_url)

    raise ValueError(f"Backend checkpoint non supporté : {backend_name!r}")


# ── Factory injectable ────────────────────────────────────────────────────────


class CheckpointerFactory:
    """Factory injectable pour la compilation du graphe LangGraph.

    Permet d'injecter un checkpointer existant (tests, intégration)
    ou d'en créer un depuis les paramètres d'environnement (production).

    Usage au point de compilation du workflow ::

        # Production — backend issu de .env
        app = workflow.compile(checkpointer=CheckpointerFactory.from_settings().build())

        # Tests — InMemorySaver sans I/O
        app = workflow.compile(checkpointer=CheckpointerFactory.for_tests().build())

        # Injection explicite
        app = workflow.compile(checkpointer=CheckpointerFactory(my_saver).build())

    Le backend reste remplaçable sans modifier le code de workflow.
    """

    def __init__(
        self,
        checkpointer: Any = None,
        *,
        backend: str | CheckpointBackend | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._instance = checkpointer
        self._backend = backend
        self._settings = settings

    def build(self) -> Any:
        """Retourne le checkpointer — instance injectée ou nouvellement créée."""
        if self._instance is not None:
            return self._instance
        return get_checkpointer(backend=self._backend, settings=self._settings)

    @classmethod
    def for_tests(cls) -> CheckpointerFactory:
        """Factory préconfigurée avec ``InMemorySaver`` — aucune dépendance I/O."""
        return cls(InMemorySaver())

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> CheckpointerFactory:
        """Factory utilisant les paramètres d'environnement (``LANGGRAPH_CHECKPOINT_BACKEND``)."""
        return cls(settings=settings)


# ── Session de checkpoint par dossier ────────────────────────────────────────


class CheckpointSession:
    """Gère la persistance d'un dossier avec un thread_id stable et immuable.

    Le thread_id est dérivé de ``case_id`` et ne change jamais.
    Toute tentative de reprise avec un thread_id différent lève ``ValueError``
    immédiatement — garantie d'intégrité des dossiers.

    ``save`` valide le ClaimState avant de le persister.
    ``load`` retourne les ``channel_values`` du dernier checkpoint ou ``None``.

    Usage ::

        session = CheckpointSession("CLM-0001")
        saved_config = session.save(checkpointer, state, step=1)
        restored   = session.load(checkpointer)   # dict | None

        # À la reprise :
        session.assert_resume(resume_config)       # lève si thread_id différent
    """

    def __init__(self, case_id: str) -> None:
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("case_id obligatoire et non vide pour CheckpointSession")
        self._case_id = case_id.strip()
        self._config = make_thread_config(self._case_id)

    # ── Propriétés ──────────────────────────────────────────────────────────

    @property
    def case_id(self) -> str:
        return self._case_id

    @property
    def config(self) -> dict[str, Any]:
        """Configuration LangGraph canonique avec thread_id stable."""
        return self._config

    @property
    def thread_id(self) -> str:
        return get_thread_id(self._config)

    # ── Contrainte de reprise ────────────────────────────────────────────────

    def assert_resume(self, resume_config: Mapping[str, Any]) -> None:
        """Lève ``ValueError`` si ``resume_config`` utilise un thread_id différent.

        Interdit la création d'un nouveau thread_id pour un dossier existant.
        """
        assert_same_thread_id(self._config, resume_config)

    # ── Persistance ─────────────────────────────────────────────────────────

    def save(
        self,
        checkpointer: Any,
        state: Mapping[str, Any],
        *,
        step: int,
    ) -> dict[str, Any]:
        """Valide ``state`` puis l'écrit dans le checkpointer.

        Args:
            checkpointer: Instance retournée par ``CheckpointerFactory.build()``.
            state: ClaimState complet ou partiel à sauvegarder.
            step: Numéro d'étape — détermine les versions de canaux.

        Returns:
            Configuration LangGraph mise à jour (inclut ``checkpoint_id``).

        Raises:
            ValueError: Si ``state`` contient du contenu interdit
                (bytes, chemins absolus, texte OCR brut, secrets…).
        """
        validate_checkpoint_state(state)
        state_dict = dict(state)
        channel_versions: dict[str, int] = {k: step for k in state_dict}
        checkpoint: dict[str, Any] = {
            "v": 1,
            "id": str(uuid.uuid4()),
            "ts": datetime.now(UTC).isoformat(),
            "channel_values": state_dict,
            "channel_versions": channel_versions,
            "versions_seen": {},
            "pending_sends": [],
        }
        metadata: dict[str, Any] = {
            "source": "update",
            "step": step,
            "writes": {},
            "parents": {},
        }
        return checkpointer.put(self._config, checkpoint, metadata, channel_versions)

    def load(self, checkpointer: Any) -> dict[str, Any] | None:
        """Charge les données du dernier checkpoint ou ``None`` si inexistant.

        Retourne uniquement les ``channel_values`` — pas les métadonnées internes
        de LangGraph (versions, identifiants de checkpoint, horodatage).
        """
        checkpoint = checkpointer.get(self._config)
        if checkpoint is None:
            return None
        return checkpoint.get("channel_values")
