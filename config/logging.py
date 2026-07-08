"""Configuration structlog centralisée — ClaimShield Santé (P3-2).

Un seul point de configuration pour toute l'application (API, scripts qui le
souhaitent) : JSON structuré en production, rendu lisible en développement
(``Settings.claimshield_debug``). ``case_id``/``agent_name`` sont liés au
contexte courant via ``structlog.contextvars`` (``bind_case_context``) —
tout log émis depuis un point de décision agent porte automatiquement ces
deux champs, sans avoir à les répéter à chaque appel.

Ne remplace jamais ``logging`` stdlib : structlog s'appuie dessus
(``structlog.stdlib.LoggerFactory``) — un module qui utilise encore
``logging.getLogger(__name__).warning(...)`` continue de fonctionner et
traverse le même pipeline de rendu, seul le format de sortie change.

Ne configure jamais le logging à l'import d'un module métier : appelée
explicitement une fois au démarrage (``api/main.py::create_app``), jamais
implicitement — évite qu'importer un module quelconque n'ait l'effet de bord
de reconfigurer le logging de tout le processus (y compris en test, où
``pytest`` gère déjà la capture des logs).
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from config.settings import get_settings

_CONFIGURED = False
_BASELINE_CONFIGURED = False


def _ensure_baseline_configured() -> None:
    """Configuration structlog minimale, appliquée automatiquement au
    premier appel de ``get_logger`` — garantit que structlog achemine
    toujours ses logs à travers le module ``logging`` stdlib (capturable
    par ``caplog``, tout handler déjà en place), même si
    ``configure_logging()`` n'a jamais été appelée explicitement (ex. tests
    unitaires, scripts, imports isolés). Sans cette garantie, structlog
    utilise par défaut ``PrintLoggerFactory`` (écriture directe sur stdout,
    hors du pipeline ``logging``) — un module qui migre de
    ``logging.getLogger`` vers ``structlog.get_logger`` verrait alors ses
    logs disparaître silencieusement de tout outil basé sur ``logging``.
    ``configure_logging()`` (appelée par le point d'entrée applicatif, ex.
    ``api/main.py::create_app``) reste responsable du format de sortie
    final (JSON/console) et remplace cette configuration minimale sans
    conflit — ``structlog.configure()`` peut être appelée plusieurs fois,
    la dernière l'emporte."""
    global _BASELINE_CONFIGURED
    if _CONFIGURED or _BASELINE_CONFIGURED:
        return
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    _BASELINE_CONFIGURED = True


def get_logger(name: str) -> Any:
    """Retourne un logger structlog nommé — même contrat que
    ``logging.getLogger(__name__)``, mais avec un contexte structuré
    (``bind_case_context``) et un acheminement stdlib garanti (voir
    ``_ensure_baseline_configured``). Point d'entrée recommandé pour tout
    module métier — préférer à ``structlog.get_logger`` directement."""
    _ensure_baseline_configured()
    return structlog.get_logger(name)


def configure_logging(*, level: str | None = None, json_output: bool | None = None) -> None:
    """Configure structlog (+ stdlib logging sous-jacent) pour tout le processus.

    Idempotente : les appels suivants sont des no-op silencieux — évite une
    reconfiguration cumulative (ex. rechargement ``uvicorn --reload``,
    imports multiples). ``json_output=None`` déduit le format de
    ``Settings.claimshield_debug`` (console lisible en dev, JSON sinon) ;
    forcer explicitement ``True``/``False`` reste possible (ex. tests qui
    veulent un rendu prévisible).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    resolved_level = (level or settings.claimshield_log_level).upper()
    use_json = json_output if json_output is not None else not settings.claimshield_debug

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    renderer = structlog.processors.JSONRenderer() if use_json else structlog.dev.ConsoleRenderer()
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(resolved_level)

    _CONFIGURED = True


def bind_case_context(*, case_id: str | None = None, agent_name: str | None = None, **extra: object) -> None:
    """Lie ``case_id``/``agent_name`` (et tout contexte additionnel fourni)
    à tous les logs émis depuis le contexte courant (thread/tâche async),
    jusqu'au prochain ``clear_case_context()``. Ne journalise jamais de
    donnée personnelle brute, de secret ou de contenu de document — seuls
    des identifiants et noms d'agent sont attendus ici."""
    values = {"case_id": case_id, "agent_name": agent_name, **extra}
    structlog.contextvars.bind_contextvars(**{k: v for k, v in values.items() if v is not None})


def clear_case_context() -> None:
    """Efface le contexte lié par ``bind_case_context`` — à appeler en fin de
    requête/tâche pour ne jamais faire fuiter le contexte d'un dossier vers
    le traitement du suivant (ex. middleware FastAPI, fin de nœud LangGraph)."""
    structlog.contextvars.clear_contextvars()


__all__ = ["bind_case_context", "clear_case_context", "configure_logging", "get_logger"]
