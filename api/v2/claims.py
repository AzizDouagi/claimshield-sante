"""Endpoints ``/v2/claims`` — pipeline autonome, jamais de revue humaine bloquante.

Aucun accès direct à un agent individuel : le seul point d'entrée métier est
le graphe compilé (`graph.workflow_v2.compile_workflow_v2`) — même garantie
que `api/main.py` (V1, non modifié) pour son propre graphe.

``REOPEN`` (voir `services/override_store.py::OverrideAction`) n'est **jamais**
traité spécialement ici : cet endpoint ne fait qu'enregistrer l'intention
humaine (voir `human_review.override_service.validate_and_record_override`).
Une reprise réelle du dossier passe par un nouveau ``POST /v2/claims`` avec le
même ``case_id`` — LangGraph, en l'absence de tout `interrupt()` en attente
sur ce thread (le graphe V2 n'en pose jamais), redémarre alors naturellement
une exécution complète depuis START sur le même ``thread_id`` : aucune
reprise partielle n'est jamais possible, conforme au plan (Phase V2-8).
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, status

from api.v2.dependencies import require_api_key
from api.v2.schemas import (
    ClaimStatusResponseV2,
    ClaimSubmissionRequestV2,
    OverrideRecordResponseV2,
    OverrideRequestBodyV2,
)
from config.logging import bind_case_context, clear_case_context, get_logger
from graph.checkpoints import CheckpointerFactory, make_thread_config
from graph.workflow_v2 import compile_workflow_v2
from human_review.override_service import OverrideValidationError, validate_and_record_override
from services.override_store import OverrideStore

__all__ = ["build_v2_router"]

logger = get_logger(__name__)

_CASE_ID_PATH = Path(..., pattern=r"^CLM-\d{4,}$")


def _build_status_response(case_id: str, values: dict[str, Any]) -> ClaimStatusResponseV2:
    """Construit la réponse minimisée à partir d'un `ClaimStateV2` réel.

    ``decision_summary``/``bounded_by`` proviennent de
    ``decision_result.justification``/``.bounded_by`` — déjà des champs
    structurés et rédigés par `agents/autonomous_decision_agent`, jamais un
    nouveau résumé recalculé ici. Les champs d'explicabilité additifs
    (``missing_information``/``assumptions``/``decisive_factors``/
    ``counterfactuals``/``recommended_action``/``evidence_completeness``,
    Phase 7) sont repris de la même façon — une simple exposition de champs
    déjà calculés, jamais un nouveau calcul.
    """
    decision_result = values.get("decision_result")
    final_decision = values.get("final_decision")
    return ClaimStatusResponseV2(
        case_id=case_id,
        current_step=values.get("current_step"),
        completed_steps=list(values.get("completed_steps") or []),
        final_decision=getattr(final_decision, "value", final_decision),
        decision_summary=list(decision_result.justification) if decision_result is not None else [],
        bounded_by=list(decision_result.bounded_by) if decision_result is not None else [],
        errors=list(values.get("errors") or []),
        alerts=list(values.get("alerts") or []),
        missing_information=list(decision_result.missing_information) if decision_result is not None else [],
        assumptions=list(decision_result.assumptions) if decision_result is not None else [],
        decisive_factors=list(decision_result.decisive_factors) if decision_result is not None else [],
        counterfactuals=list(decision_result.counterfactuals) if decision_result is not None else [],
        recommended_action=decision_result.recommended_action if decision_result is not None else "",
        evidence_completeness=decision_result.evidence_completeness if decision_result is not None else None,
    )


def build_v2_router(
    *,
    compiled_graph: Any | None = None,
    override_store: OverrideStore | None = None,
) -> APIRouter:
    """Construit le routeur ``/v2/claims`` — graphe compilé une seule fois.

    ``compiled_graph``/``override_store`` injectables (tests) ; ``None``
    construit les instances par défaut (`compile_workflow_v2` avec le
    backend de checkpoint configuré par l'environnement — même
    `CheckpointerFactory.from_settings()` que V1 —, `OverrideStore()` neuf)
    — jamais un singleton caché, même convention que
    `graph.workflow.compile_workflow`/`orchestrator.executor.Orchestrator`.
    """
    resolved_graph = (
        compiled_graph
        if compiled_graph is not None
        else compile_workflow_v2(CheckpointerFactory.from_settings().build())
    )
    resolved_store = override_store if override_store is not None else OverrideStore()

    router = APIRouter(tags=["v2-claims"])

    @router.post(
        "/claims",
        response_model=ClaimStatusResponseV2,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_api_key)],
    )
    def submit_claim_v2(payload: ClaimSubmissionRequestV2) -> ClaimStatusResponseV2:
        bind_case_context(case_id=payload.case_id, agent_name="api.v2.submit_claim")
        try:
            logger.info("claim_submission_v2_received")
            config = make_thread_config(payload.case_id)
            initial_state: dict[str, Any] = {
                "case_id": payload.case_id,
                "schema_version": "2.0.0",
                "current_step": "initial",
                "completed_steps": [],
                "errors": [],
                "alerts": [],
                "intake_input": {
                    "source_path": payload.source_path,
                    "required_documents": payload.required_documents,
                    "revision_of_case_id": payload.revision_of_case_id,
                },
                "reader_role": payload.role.value,
            }
            state = resolved_graph.invoke(initial_state, config=config)
            response = _build_status_response(payload.case_id, dict(state))
            logger.info(
                "claim_submission_v2_processed",
                current_step=response.current_step,
                final_decision=response.final_decision,
            )
            return response
        finally:
            clear_case_context()

    @router.get("/claims/{case_id}", response_model=ClaimStatusResponseV2)
    def get_claim_status_v2(case_id: str = _CASE_ID_PATH) -> ClaimStatusResponseV2:
        config = make_thread_config(case_id)
        snapshot = resolved_graph.get_state(config)
        if not snapshot.values:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dossier {case_id!r} introuvable — jamais soumis ou thread expiré.",
            )
        return _build_status_response(case_id, dict(snapshot.values))

    @router.post(
        "/claims/{case_id}/override",
        response_model=OverrideRecordResponseV2,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_api_key)],
    )
    def submit_override_v2(
        body: OverrideRequestBodyV2, case_id: str = _CASE_ID_PATH
    ) -> OverrideRecordResponseV2:
        bind_case_context(case_id=case_id, agent_name="api.v2.submit_override")
        try:
            logger.info("override_v2_received", action=body.action.value)
            config = make_thread_config(case_id)
            snapshot = resolved_graph.get_state(config)
            if not snapshot.values:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Dossier {case_id!r} introuvable — jamais soumis ou thread expiré.",
                )
            final_decision = snapshot.values.get("final_decision")
            raw = {
                "case_id": case_id,
                "actor": body.actor,
                "action": body.action.value,
                "justification": body.justification,
            }
            try:
                record = validate_and_record_override(
                    raw,
                    store=resolved_store,
                    original_decision=getattr(final_decision, "value", final_decision),
                )
            except OverrideValidationError as exc:
                logger.warning("override_v2_rejected", codes=[err.code for err in exc.errors])
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=[err.model_dump(mode="json") for err in exc.errors],
                ) from exc
            logger.info("override_v2_recorded", action=record.action.value)
            return OverrideRecordResponseV2.model_validate(record.model_dump())
        finally:
            clear_case_context()

    return router
