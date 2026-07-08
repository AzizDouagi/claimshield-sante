"""Validation des préconditions d'appel d'un agent — orchestrator/routing.py.

Répond à une seule question, avant tout dispatch : *est-il structurellement
valide d'invoquer cet agent maintenant, compte tenu du ``ClaimState``
réel ?* Trois vérifications, pures et sans effet de bord :

1. Le dossier ciblé par la requête (``case_id``) est bien celui de l'état.
2. L'étape déclarée par la requête (``current_step``) correspond à l'état
   réel — protection contre une requête construite sur une vue périmée.
3. L'agent demandé correspond à l'étape courante du pipeline (son
   prédécesseur nominal vient de s'exécuter, ou l'agent lui-même est rejoué
   — retry) ET le résultat de ce prédécesseur est bien présent dans l'état.

Ne décide jamais **quoi faire** du résultat d'un agent (ALLOW/QUARANTINE/
NEEDS_REVIEW/FAILURE) : ces branches métier restent exclusivement définies
par ``graph/edges.py`` (``route_intake``, ``route_security``, etc.), jamais
reproduites ici. Ce module réutilise au contraire ``AGENT_RESULT_FIELD``
(``orchestrator.orchestrator`` — lui-même dérivé de
``graph.edges.RELAUNCH_RESULT_FIELDS`` pour 7 des 11 agents) plutôt que de
redéfinir la correspondance agent → champ résultat en concurrence.

Réutilise aussi ``PolicyDecision``/``PolicyEffect`` (``orchestrator.policies``)
comme forme de résultat — même contrat ALLOW/DENY + motif structuré que les
autres évaluateurs de l'orchestrateur, pas une quatrième forme concurrente.

Ce contrôle N'EST PAS celui exécuté par le pipeline LangGraph de production
--------------------------------------------------------------------------
``graph/nodes.py::build_orchestrator()`` construit l'``Orchestrator`` par
défaut avec ``preconditions_check=_graph_preconditions_check`` (défini dans
``graph/nodes.py``, pas ici) — une version allégée qui ne revérifie que
``case_id`` et laisse la topologie du graphe garantir l'ordre d'exécution.
``evaluate_call_preconditions`` (ce module) est donc le contrat de référence
pour un futur appelant *hors* LangGraph, pas le contrôle réellement traversé
en production. Voir la section dédiée du docstring de ``graph/nodes.py`` pour
la justification complète.

``scripts/run_agent_manual.py`` n'est PAS aujourd'hui un tel appelant : il
n'utilise ni ``Orchestrator`` ni ``ClaimState`` accumulé — chaque probe
appelle ``agent.run(...)`` isolément et affiche le résultat sans le fusionner
(diagnostic complet : ``docs/debug_manual_runner.md``). Le brancher sur ce
contrôle suppose d'abord de lui faire maintenir un ``ClaimState`` réel — une
des deux approches déjà documentées dans ce diagnostic, hors périmètre de ce
module.

Limite connue de ce contrôle riche — décision tranchée en Phase 2 (P2-1,
parallélisation) : la précondition 3 suppose un **unique prédécesseur
nominal par agent** (``AGENT_PIPELINE_ORDER`` est une séquence linéaire).
Cette hypothèse est désormais **fausse** pour 4 agents du graphe de
production réel : ``document_ocr``/``fhir_validator`` (fanned-out depuis
``privacy``) et ``clinical_consistency``/``fraud_detection`` (fanned-out
depuis ``medical_coding``) — voir ``graph/workflow.py``,
``graph/edges.py::route_privacy_fan_out``/``route_coding_fan_out``. Décision
retenue : **ne pas adapter** ``AGENT_PIPELINE_ORDER`` à une structure de
prédécesseurs multiples — cette fonction reste un contrat de référence pour
un futur appelant hors LangGraph (jamais exercée par le graphe de production,
qui utilise exclusivement ``graph/nodes.py::_graph_preconditions_check``,
lequel ne dépend d'aucune notion de prédécesseur unique). Adapter une
fonction à zéro appelant de production pour un cas qu'elle ne rencontre
jamais aurait été un investissement sans bénéfice mesurable ; si un appelant
hors LangGraph voit un jour le jour, cette limite devra être retraitée à ce
moment, avec la connaissance précise de ses propres besoins de
parallélisme (potentiellement différents de ceux du graphe de production).
"""
from __future__ import annotations

from orchestrator.orchestrator import AGENT_RESULT_FIELD, AgentCallRequest, AgentName
from orchestrator.policies import PolicyDecision, PolicyEffect
from schemas.results import StructuredError
from state.claim_state import ClaimState

# ── Ordre nominal du pipeline ──────────────────────────────────────────────────
#
# Backbone linéaire uniquement — jamais les branches conditionnelles
# (QUARANTINE / NEEDS_REVIEW / FAILURE) qui restent la responsabilité
# exclusive de graph/workflow.py et graph/edges.py. Sert uniquement à
# déterminer, pour un agent donné, quel est son prédécesseur nominal.

AGENT_PIPELINE_ORDER: tuple[AgentName, ...] = (
    AgentName.CLAIM_INTAKE,
    AgentName.SECURITY_GATE,
    AgentName.PRIVACY,
    AgentName.DOCUMENT_OCR,
    AgentName.FHIR_VALIDATOR,
    AgentName.IDENTITY_COVERAGE,
    AgentName.MEDICAL_CODING,
    AgentName.CLINICAL_CONSISTENCY,
    AgentName.FRAUD_DETECTION,
    AgentName.CASE_REVIEWER,
    AgentName.AUDIT,
)

PIPELINE_START = "initial"
"""Valeur conventionnelle de ``ClaimState.current_step`` avant toute
exécution — même convention que les fixtures de ``tests/graph/test_workflow*.py``."""


def _predecessor(agent_name: AgentName) -> AgentName | None:
    """Agent nominal précédent dans ``AGENT_PIPELINE_ORDER``. ``None`` pour
    ``claim_intake`` (premier agent, aucune précondition de résultat)."""
    index = AGENT_PIPELINE_ORDER.index(agent_name)
    return AGENT_PIPELINE_ORDER[index - 1] if index > 0 else None


# ── Évaluation des préconditions ──────────────────────────────────────────────
#
# AGENT_RESULT_FIELD (le champ ClaimState prouvant qu'un agent a produit un
# résultat) est importé de orchestrator.orchestrator — réutilisé tel quel,
# pas redéfini ici (voir le docstring du module).


def evaluate_call_preconditions(state: ClaimState, request: AgentCallRequest) -> PolicyDecision:
    """Valide qu'il est structurellement possible d'exécuter
    ``request.agent_name`` compte tenu de l'état réel du dossier.

    Ordre des vérifications (la première anomalie rencontrée est retournée) :
      1. ``state["case_id"] == request.case_id`` — sinon incohérence.
      2. ``state["current_step"] == request.current_step`` — sinon la
         requête a été construite sur une vue périmée de l'état.
      3. ``state["current_step"]`` correspond au prédécesseur nominal de
         ``request.agent_name`` (progression normale) ou à
         ``request.agent_name`` lui-même (nouvelle tentative) — sinon
         l'agent demandé ne correspond pas à l'étape courante.
      4. Le résultat du prédécesseur nominal (``AGENT_RESULT_FIELD``) est
         présent dans l'état — sinon précondition absente. Sans objet pour
         ``claim_intake`` (aucun prédécesseur).
    """
    if state.get("case_id") != request.case_id:
        return PolicyDecision(
            effect=PolicyEffect.DENY,
            reason=StructuredError(
                code="CASE_ID_MISMATCH",
                message=(
                    f"Requête pour {request.case_id!r} appliquée à un état "
                    f"portant case_id={state.get('case_id')!r}."
                ),
                field="case_id",
            ),
        )

    actual_step = state.get("current_step")
    if actual_step != request.current_step:
        return PolicyDecision(
            effect=PolicyEffect.DENY,
            reason=StructuredError(
                code="STEP_DECLARATION_MISMATCH",
                message=(
                    f"current_step déclaré {request.current_step!r} incohérent "
                    f"avec l'état réel ({actual_step!r}) — vue périmée."
                ),
                field="current_step",
            ),
        )

    predecessor = _predecessor(request.agent_name)
    expected_steps = (
        {PIPELINE_START, request.agent_name.value}
        if predecessor is None
        else {predecessor.value, request.agent_name.value}
    )
    if actual_step not in expected_steps:
        return PolicyDecision(
            effect=PolicyEffect.DENY,
            reason=StructuredError(
                code="STEP_MISMATCH",
                message=(
                    f"Agent {request.agent_name.value!r} ne correspond pas à "
                    f"l'étape courante {actual_step!r} — attendu l'une de "
                    f"{sorted(expected_steps)!r}."
                ),
                field="current_step",
            ),
        )

    if predecessor is not None:
        result_field = AGENT_RESULT_FIELD[predecessor]
        if state.get(result_field) is None:
            return PolicyDecision(
                effect=PolicyEffect.DENY,
                reason=StructuredError(
                    code="PRECONDITION_RESULT_MISSING",
                    message=(
                        f"Résultat requis absent : {result_field!r} "
                        f"(produit par {predecessor.value!r}) avant d'appeler "
                        f"{request.agent_name.value!r}."
                    ),
                    field=result_field,
                ),
            )

    return PolicyDecision(
        effect=PolicyEffect.ALLOW,
        reason=StructuredError(
            code="PRECONDITIONS_SATISFIED",
            message=(
                f"Préconditions satisfaites pour {request.agent_name.value!r} "
                f"au dossier {request.case_id!r}."
            ),
            field="agent_name",
        ),
    )
