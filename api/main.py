"""API HTTP minimale exposant le graphe LangGraph compilé — ClaimShield Santé.

Périmètre volontairement minimal (P3-1 du plan de remédiation) : quatre
endpoints, aucun accès direct à un agent individuel (le seul point d'entrée
métier est le graphe compilé via ``Orchestrator.execute_agent()``, exactement
comme ``graph/workflow.py`` — cohérent avec ce que
``tests/graph/test_architecture.py`` vérifie déjà statiquement). Persistance
``InMemorySaver`` par défaut (backend configurable via
``LANGGRAPH_CHECKPOINT_BACKEND`` — voir ``graph.checkpoints``) ; la
persistance multi-worker réelle (Postgres partagé) reste hors périmètre de
cette première étape.

Endpoints :
  - ``POST /claims``                          — soumet un nouveau dossier.
  - ``GET  /claims/{case_id}``                 — état courant minimisé.
  - ``POST /claims/{case_id}/human-decision``  — reprend après interruption HITL.
  - ``GET  /healthz``                          — liveness, sans authentification.

Lancer en développement ::

    uvicorn api.main:app --reload --host $API_HOST --port $API_PORT
"""
from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Path, status
from langgraph.types import Command

from api.dependencies import require_api_key
from api.schemas import (
    ClaimStatusResponse,
    ClaimSubmissionRequest,
    HealthResponse,
    HumanDecisionRequest,
)
from api.v2 import v2_router
from config.logging import bind_case_context, clear_case_context, configure_logging, get_logger
from graph.checkpoints import CheckpointerFactory, make_thread_config
from graph.workflow import compile_workflow
from human_review.service import HumanDecisionValidationError, build_human_review_payload

logger = get_logger(__name__)

_CASE_ID_PATH = Path(..., pattern=r"^CLM-\d{4,}$")


def _build_status_response(case_id: str, values: dict[str, Any], next_nodes: tuple) -> ClaimStatusResponse:
    """Construit la réponse minimisée à partir d'un ``ClaimState`` réel.

    Jamais de document brut, de texte OCR complet ni de secret : ``values``
    provient de LangGraph mais tout ce qui est exposé passe par des champs
    déjà minimisés ailleurs dans le projet (``final_recommendation``,
    ``errors``/``alerts`` déjà agrégés, ``pending_review`` réutilisant
    ``human_review.service.build_human_review_payload`` — même garantie de
    minimisation que le payload d'interruption HITL).
    """
    interrupted = "await_human_review" in next_nodes
    final_recommendation = values.get("final_recommendation")
    return ClaimStatusResponse(
        case_id=case_id,
        current_step=values.get("current_step"),
        completed_steps=list(values.get("completed_steps") or []),
        final_recommendation=getattr(final_recommendation, "value", final_recommendation),
        errors=list(values.get("errors") or []),
        alerts=list(values.get("alerts") or []),
        interrupted=interrupted,
        pending_review=build_human_review_payload(values) if interrupted else None,
    )


def create_app(checkpointer: Any | None = None, *, compiled_graph: Any | None = None) -> FastAPI:
    """Construit l'application FastAPI, graphe compilé une seule fois.

    ``checkpointer`` injectable (tests : ``InMemorySaver`` dédié par test) ;
    ``None`` construit le backend configuré par l'environnement
    (``CheckpointerFactory.from_settings()`` — ``memory`` par défaut).

    ``compiled_graph`` (mot-clé uniquement) permet de fournir un graphe déjà
    compilé (ex. via ``graph.workflow.compile_workflow(..., case_reviewer_impl=...)``
    dans les tests qui ont besoin d'un agent injecté) — dans ce cas il est
    utilisé tel quel et ``checkpointer``/``CheckpointerFactory`` ne sont
    jamais consultés. Fournir les deux à la fois est une erreur d'appelant
    (ambiguïté sur la source de vérité), refusée explicitement. Même principe
    que ``compile_workflow(graph=None, ...)`` : un paramètre optionnel qui
    court-circuite la construction interne quand l'appelant fournit déjà
    l'artefact construit.
    """
    if compiled_graph is not None and checkpointer is not None:
        raise ValueError(
            "create_app() : fournir soit 'checkpointer' soit 'compiled_graph', jamais les deux."
        )

    configure_logging()

    if compiled_graph is not None:
        resolved_graph = compiled_graph
    else:
        resolved_checkpointer = (
            checkpointer if checkpointer is not None else CheckpointerFactory.from_settings().build()
        )
        # interrupt_before=[] : désactive l'interruption statique par défaut
        # (DEFAULT_INTERRUPT_BEFORE=["needs_review"], une pause technique sans
        # payload avant le nœud "needs_review") au profit exclusif de
        # l'interruption dynamique réelle sur "await_human_review"
        # (langgraph.types.interrupt(), payload de revue exploitable par le
        # client API — voir _build_status_response/pending_review). Même
        # convention que l'ensemble de tests/graph/test_workflow*.py.
        resolved_graph = compile_workflow(resolved_checkpointer, interrupt_before=[])

    app = FastAPI(
        title="ClaimShield Santé API",
        description="API minimale exposant le pipeline multi-agents de traitement des réclamations.",
        version="0.1.0",
    )
    app.state.compiled_graph = resolved_graph

    # Point d'intégration V2 (§0 du plan de refonte) — une seule ligne
    # additive, aucune modification des endpoints V1 ci-dessous.
    app.include_router(v2_router, prefix="/v2")

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse()

    @app.post(
        "/claims",
        response_model=ClaimStatusResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_api_key)],
    )
    def submit_claim(payload: ClaimSubmissionRequest) -> ClaimStatusResponse:
        bind_case_context(case_id=payload.case_id, agent_name="api.submit_claim")
        try:
            logger.info("claim_submission_received")
            config = make_thread_config(payload.case_id)
            initial_state: dict[str, Any] = {
                "case_id": payload.case_id,
                "schema_version": "1.0.0",
                "current_step": "initial",
                "completed_steps": [],
                "errors": [],
                "alerts": [],
                "final_justification": [],
                "intake_input": {
                    "case_id": payload.case_id,
                    "source_path": payload.source_path,
                    "required_documents": payload.required_documents,
                    "uploaded_files": [f.model_dump(mode="json") for f in payload.uploaded_files],
                },
                "privacy_input": {
                    "case_id": payload.case_id,
                    "role": payload.role.value,
                },
            }
            app.state.compiled_graph.invoke(initial_state, config=config)
            snapshot = app.state.compiled_graph.get_state(config)
            response = _build_status_response(payload.case_id, dict(snapshot.values), snapshot.next)
            logger.info(
                "claim_submission_processed",
                current_step=response.current_step,
                interrupted=response.interrupted,
            )
            return response
        finally:
            clear_case_context()

    @app.get("/claims/{case_id}", response_model=ClaimStatusResponse)
    def get_claim_status(case_id: str = _CASE_ID_PATH) -> ClaimStatusResponse:
        config = make_thread_config(case_id)
        snapshot = app.state.compiled_graph.get_state(config)
        if not snapshot.values:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dossier {case_id!r} introuvable — jamais soumis ou thread expiré.",
            )
        return _build_status_response(case_id, dict(snapshot.values), snapshot.next)

    @app.post(
        "/claims/{case_id}/human-decision",
        response_model=ClaimStatusResponse,
        dependencies=[Depends(require_api_key)],
    )
    def submit_human_decision(
        decision: HumanDecisionRequest, case_id: str = _CASE_ID_PATH
    ) -> ClaimStatusResponse:
        bind_case_context(case_id=case_id, agent_name="api.submit_human_decision")
        try:
            logger.info("human_decision_received", action=decision.action.value)
            config = make_thread_config(case_id)
            existing = app.state.compiled_graph.get_state(config)
            if not existing.values:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Dossier {case_id!r} introuvable — jamais soumis ou thread expiré.",
                )
            if "await_human_review" not in existing.next:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Dossier {case_id!r} n'est pas en attente de revue humaine.",
                )

            resume_payload = decision.model_dump(mode="json", exclude_none=True)
            resume_payload.pop("case_id", None)  # complété automatiquement par le nœud, jamais redemandé

            try:
                app.state.compiled_graph.invoke(Command(resume=resume_payload), config=config)
            except HumanDecisionValidationError as exc:
                logger.warning("human_decision_rejected", codes=[err.code for err in exc.errors])
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=[err.model_dump(mode="json") for err in exc.errors],
                ) from exc

            snapshot = app.state.compiled_graph.get_state(config)
            response = _build_status_response(case_id, dict(snapshot.values), snapshot.next)
            logger.info(
                "human_decision_processed",
                current_step=response.current_step,
                final_recommendation=response.final_recommendation,
            )
            return response
        finally:
            clear_case_context()

    return app


app = create_app()

__all__ = ["app", "create_app"]
