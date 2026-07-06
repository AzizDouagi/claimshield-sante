"""Contrat de traçabilité LLM obligatoire pour les 11 agents.

La règle projet documentée dans ``CLAUDE.md`` impose qu'aucun agent métier ne
soit purement déterministe : chaque exécution effective doit produire une trace
LLM. Le contrat de sortie doit donc rendre cette trace obligatoire dans
``llm_metadata`` ou dans un champ explicitement équivalent.
"""
from __future__ import annotations

from types import UnionType
from typing import Union, get_args, get_origin

import pytest

from orchestrator.orchestrator import AGENT_RESULT_MODELS, AgentName
from schemas.results import LlmMetadata

_LLM_TRACE_FIELD_NAMES = frozenset(
    {
        "llm_metadata",
        "llm_trace",
        "model_metadata",
    }
)


def _is_llm_metadata_annotation(annotation: object) -> bool:
    if annotation is LlmMetadata:
        return True
    return LlmMetadata in get_args(annotation)


def _allows_none(annotation: object) -> bool:
    origin = get_origin(annotation)
    return (origin in (UnionType, Union) or isinstance(annotation, UnionType)) and (
        type(None) in get_args(annotation)
    )


def _llm_trace_fields(result_model: type) -> list[str]:
    return [
        field_name
        for field_name, field_info in result_model.model_fields.items()
        if field_name in _LLM_TRACE_FIELD_NAMES
        and _is_llm_metadata_annotation(field_info.annotation)
    ]


def test_contrat_trace_llm_couvre_les_11_agents():
    """Le contrôle doit couvrir tout le registre, stubs inclus."""
    assert set(AGENT_RESULT_MODELS) == set(AgentName)
    assert len(AGENT_RESULT_MODELS) == 11
    assert AgentName.CASE_REVIEWER in AGENT_RESULT_MODELS
    assert AgentName.AUDIT in AGENT_RESULT_MODELS


@pytest.mark.parametrize(
    ("agent_name", "result_model"),
    sorted(AGENT_RESULT_MODELS.items(), key=lambda item: item[0].value),
    ids=[
        agent_name.value
        for agent_name in sorted(AGENT_RESULT_MODELS, key=lambda item: item.value)
    ],
)
def test_chaque_resultat_agent_exige_une_trace_llm_non_nulle(
    agent_name: AgentName,
    result_model: type,
):
    """Un résultat sans trace LLM ne doit pas pouvoir représenter une exécution valide."""
    trace_fields = _llm_trace_fields(result_model)
    assert trace_fields, (
        f"{agent_name.value} doit exposer une trace LLM dans "
        "`llm_metadata` ou un champ équivalent typé LlmMetadata."
    )

    trace_field = result_model.model_fields[trace_fields[0]]
    assert trace_field.is_required(), (
        f"{agent_name.value}.{trace_fields[0]} doit être obligatoire : "
        "une exécution d'agent sans trace LLM ne doit pas valider."
    )
    assert not _allows_none(trace_field.annotation), (
        f"{agent_name.value}.{trace_fields[0]} ne doit pas accepter None : "
        "un fallback déterministe sans appel LLM réussi doit échouer fail-closed."
    )
