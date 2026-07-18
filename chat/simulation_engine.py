"""Sandbox de simulation — chat/simulation_engine.py (plan V2 §6, Phase V2-11b).

Simulation **réelle** (décision AZIZ, plan V2 §0 : « simulate_changes =
sandbox réelle sur copie en mémoire du state, jamais une estimation
heuristique ») — réinvoque le graphe compilé sur une copie du dossier réel,
jamais une approximation.

Exception documentée à la convention de `chat/tools.py` (V2-11a : « jamais
un accès direct à `graph.*`/`agents.*` ») : la simulation doit
structurellement réinvoquer le graphe compilé — impossible via un simple
appel HTTP à `/v2/claims/*`, qui n'expose jamais l'état interne complet ni
les documents sources. **Seul module de `chat/` à importer `graph.*`
directement**, pour cette raison précise et documentée.

Garanties de non-mutation (bloquantes, jamais contournables) :
  1. Les documents du dossier réel (`storage/incoming/{case_id}/...`) ne
     sont jamais modifiés — uniquement copiés (jamais déplacés) vers un
     répertoire temporaire dédié à la simulation, supprimé après usage.
  2. La simulation s'exécute sous un **identifiant de dossier synthétique**
     (`CLM-9xxxxxxxx`, jamais celui du dossier réel) — indispensable :
     `intake_safety_agent` (réutilisé tel quel) écrit réellement des
     fichiers sous `storage/incoming/{case_id}/`, keyé par `case_id`, pas
     par thread_id. Utiliser le vrai `case_id` ferait donc réellement
     collisionner ou polluer le stockage réel du dossier, quel que soit le
     thread_id utilisé pour le checkpoint. Ce détail invalide une lecture
     trop littérale du plan V2 (« thread_id éphémère SIM-{uuid4()} » à lui
     seul insuffisant) — voir `tests/v2/chat/test_simulation_engine.py::
     test_simulate_never_mutates_real_case`, qui vérifie cette garantie
     directement sur le système de fichiers, pas seulement sur le state.
  3. Le thread_id de checkpoint est lui aussi éphémère (`SIM-{uuid4()}`),
     jamais celui du dossier réel.
  4. Les artefacts de stockage de la simulation elle-même (dossier
     synthétique) sont nettoyés après l'appel — jamais persistés au-delà de
     la requête (même mécanisme que
     `scripts/evaluate_recommendations_v2.py::_reset_storage_for_case`).

Limite MVP assumée (chat texte seul, aucun mécanisme d'upload de fichier) :
`SimulationChangeRequest` ne permet jamais d'introduire un nouveau contenu
de document inconnu du système — seul un document déjà accepté peut être
retiré hypothétiquement (`remove_document`), ou le rôle de lecture changé
(`reader_role`). Un scénario « et si j'ajoutais un nouveau document » n'est
pas réalisable via ce canal (aucun upload possible depuis un message texte)
— hors périmètre de V2-11b, documenté ici plutôt que simulé silencieusement.

**Simulation ciblée** (`run_targeted_simulation`, Phase 9, plan de
remédiation « autonomie décisionnelle V2 », §7) — **second module
d'exception** documenté explicitement : outre `graph.*` (ci-dessus), ce
fichier importe aussi directement `agents.autonomous_decision_agent.agent`,
pour la raison précise que décrit le plan : « exécution ciblée via
ré-invocation directe de `autonomous_decision_agent.run()` », jamais le
graphe entier. Contrairement à `run_simulation` (ci-dessus), aucun fichier
n'est copié, aucun `case_id` synthétique n'est créé, aucune écriture disque
n'a lieu : l'état réel déjà calculé du dossier (`eligibility_result`) est lu
une seule fois (`compiled_graph.get_state()`, lecture seule), patché sur une
**copie** (`model_copy`, jamais une mutation de l'objet réel), puis
`autonomous_decision_agent.run(case_id, patched_state)` est rejoué —
équivalent au patron « appel direct d'une fonction pure hors LangGraph »
déjà utilisé partout ailleurs dans le projet pour les tests/scripts. Bornée
à une liste blanche fermée de champs d'éligibilité déjà calculés
(`chat.schemas.SimulationPatchField`) — jamais un champ arbitraire, jamais
un acte/médicament (nécessiterait de retraiter l'OCR/FHIR, hors périmètre de
cette phase, resterait à `remove_document`/simulation complète).

**Contrainte opérationnelle réelle, non résolue dans cette phase** (§0 du
plan : « fichiers existants touchés : aucun » pour V2-11b — corrigerait
sinon `api/v2/claims.py`/`api/v2/__init__.py`, hors périmètre) : en
production, `run_simulation(..., compiled_graph=None)` (chemin réel de
`chat.tools.simulate_changes`, jamais injecté hors tests) construit son
**propre** graphe compilé via `CheckpointerFactory.from_settings().build()`
— une instance de checkpointer **distincte** de celle utilisée par
`api/v2/claims.py::build_v2_router()` (construite une seule fois à l'import
de `api.v2`, jamais partagée). Avec le backend par défaut `memory`
(`InMemorySaver`, `Settings.langgraph_checkpoint_backend`), chaque instance
vit uniquement en mémoire du processus qui l'a créée : la simulation ne
verra donc **jamais** un dossier réellement soumis, systématiquement
`applied=False` (« Dossier introuvable »), même s'il existe bel et bien
côté `/v2/claims/{case_id}`. Avec un backend persistant partagé (`sqlite`/
`postgres` — déjà la configuration retenue pour le déploiement Docker,
voir CLAUDE.md « Item E »), les deux instances se reconnectent au même
support physique et la simulation fonctionne correctement. Les tests de ce
module contournent cette limitation en injectant explicitement le même
`compiled_graph` (`compiled_graph=graph`) — jamais le chemin par défaut.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from agents.autonomous_decision_agent.agent import run as run_autonomous_decision
from config.settings import get_settings
from graph.checkpoints import CheckpointerFactory, make_thread_config
from graph.workflow_v2 import compile_workflow_v2
from schemas.domain import FileStatus, ReaderRole, VerificationStatus
from schemas.results import ClaimManifest
from schemas.v2_results import EligibilityResult
from services.storage import StorageService

from chat.schemas import SimulationChangeRequest, SimulationPatch, SimulationPatchField, SimulationResult

__all__ = ["run_simulation", "run_targeted_simulation"]

_STATUS_PATCH_FIELDS = frozenset(
    {SimulationPatchField.IDENTITY_STATUS, SimulationPatchField.COVERAGE_STATUS}
)
_BOOL_PATCH_FIELDS = frozenset(
    {SimulationPatchField.CEILING_EXCEEDED, SimulationPatchField.PREAUTHORIZATION_REQUIRED}
)


def _stringify(value: Any) -> str | None:
    return getattr(value, "value", value)


def _new_synthetic_case_id() -> str:
    """Toujours un `case_id` synthétique distinct du dossier réel, au
    format `CLM-\\d{4,}` (contrainte Pydantic partagée par tous les schémas
    de résultat V2) — jamais le vrai `case_id`, jamais le thread_id
    éphémère (formats incompatibles : `SIM-...` ne matche pas ce pattern)."""
    return f"CLM-9{uuid4().int % 10**9:09d}"


def _load_real_case(
    compiled_graph: Any, case_id: str
) -> tuple[ClaimManifest | None, str | None, str | None]:
    """Retourne (manifeste des documents déjà acceptés, décision réelle
    actuelle, rôle de lecture réel) — lecture seule (`get_state`), jamais
    une mutation du state réel."""
    config = make_thread_config(case_id)
    snapshot = compiled_graph.get_state(config)
    if not snapshot.values:
        return None, None, None

    intake_result = snapshot.values.get("intake_safety_result")
    manifest = getattr(intake_result, "manifest", None) if intake_result is not None else None
    if isinstance(intake_result, dict):
        manifest = intake_result.get("manifest")

    final_decision = _stringify(snapshot.values.get("final_decision"))
    reader_role = snapshot.values.get("reader_role")
    return manifest, final_decision, reader_role


def _stage_simulation_documents(
    manifest: ClaimManifest, *, remove_document: str | None, storage_root: Path
) -> Path:
    """Copie (jamais ne déplace) les documents déjà acceptés du dossier réel
    vers un répertoire temporaire — la seule écriture réalisée directement
    sous `storage/incoming/` est celle que `intake_safety_agent` effectuera
    lui-même, plus tard, sous le `case_id` synthétique, jamais sous celui du
    dossier réel."""
    temp_dir = Path(tempfile.mkdtemp(prefix="claimshield-sim-"))
    keyword = remove_document.lower() if remove_document else None

    for f in manifest.files:
        if f.status is not FileStatus.ACCEPTED or not f.relative_storage_path:
            continue
        if keyword and keyword in f.original_name.lower():
            continue
        source = storage_root / f.relative_storage_path
        if not source.is_file():
            continue
        shutil.copyfile(source, temp_dir / f.original_name)

    return temp_dir


def _cleanup_synthetic_case_storage(sim_case_id: str, settings) -> None:
    """Retire les artefacts laissés par `intake_safety_agent` sous le
    `case_id` synthétique — même mécanisme que
    `scripts.evaluate_recommendations_v2._reset_storage_for_case` (dupliqué
    volontairement, pas un import cross-script)."""
    svc = StorageService(settings=settings)
    for base_dir in (svc.incoming_dir, svc.quarantine_dir):
        case_dir = base_dir / sim_case_id
        if case_dir.exists():
            shutil.rmtree(case_dir, ignore_errors=True)
    manifest_path = svc.manifests_dir / f"{sim_case_id}.json"
    manifest_path.unlink(missing_ok=True)


def run_simulation(
    case_id: str,
    changes: SimulationChangeRequest,
    *,
    compiled_graph: Any | None = None,
) -> SimulationResult:
    """Exécute une simulation réelle — jamais une estimation heuristique
    (décision AZIZ, plan V2 §0). `compiled_graph` injectable (tests) ;
    `None` construit une instance depuis les paramètres d'environnement,
    même convention que `graph.workflow_v2.compile_workflow_v2` partout
    ailleurs dans le projet.

    `changes.field_patches` non vide (Phase 9) délègue entièrement à
    `run_targeted_simulation` — simulation ciblée, jamais le graphe entier
    (mutuellement exclusif avec `remove_document`/`reader_role`, déjà validé
    par `chat.schemas.SimulationChangeRequest`)."""
    if changes.field_patches:
        return run_targeted_simulation(case_id, changes.field_patches, compiled_graph=compiled_graph)

    graph = (
        compiled_graph
        if compiled_graph is not None
        else compile_workflow_v2(CheckpointerFactory.from_settings().build())
    )
    settings = get_settings()

    manifest, original_decision, real_reader_role = _load_real_case(graph, case_id)
    if manifest is None:
        return SimulationResult(
            case_id=case_id,
            applied=False,
            original_decision=None,
            simulated_decision=None,
            decision_changed=False,
            error="Dossier introuvable — jamais soumis, thread expiré, ou aucun document accepté.",
        )

    reader_role = _stringify(changes.reader_role) or real_reader_role or ReaderRole.ADMINISTRATIVE_MANAGER.value

    temp_dir = _stage_simulation_documents(
        manifest, remove_document=changes.remove_document, storage_root=settings.storage_dir
    )
    sim_case_id = _new_synthetic_case_id()
    try:
        sim_config = make_thread_config(f"SIM-{uuid4()}")
        initial_state = {
            "case_id": sim_case_id,
            "schema_version": "2.0.0",
            "current_step": "initial",
            "completed_steps": [],
            "errors": [],
            "alerts": [],
            "intake_input": {"source_path": str(temp_dir), "required_documents": []},
            "reader_role": reader_role,
        }
        try:
            final_state = graph.invoke(initial_state, config=sim_config)
        except Exception as exc:  # noqa: BLE001 — une simulation en échec ne doit jamais lever
            return SimulationResult(
                case_id=case_id,
                applied=False,
                original_decision=original_decision,
                simulated_decision=None,
                decision_changed=False,
                error=f"{type(exc).__name__}: {exc}",
            )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        _cleanup_synthetic_case_storage(sim_case_id, settings)

    simulated_decision = _stringify(final_state.get("final_decision"))
    decision_result = final_state.get("decision_result")
    reasons = list(getattr(decision_result, "justification", None) or [])

    return SimulationResult(
        case_id=case_id,
        applied=True,
        original_decision=original_decision,
        simulated_decision=simulated_decision,
        decision_changed=simulated_decision != original_decision,
        simulated_reasons=reasons,
    )


# ── Simulation ciblée (Phase 9) ────────────────────────────────────────────────


def _apply_patches_to_eligibility(
    eligibility_result: EligibilityResult, patches: list[SimulationPatch]
) -> EligibilityResult:
    """Applique les patches sur une **copie** (`model_copy`) — l'objet
    `eligibility_result` réel (lu depuis `compiled_graph.get_state()`) n'est
    jamais mutée. Bornée à la liste blanche fermée `SimulationPatchField` —
    la validation de cohérence valeur/champ est déjà garantie par
    `SimulationPatch` (schéma), jamais revalidée en doublon ici."""
    identity_updates: dict[str, object] = {}
    coverage_updates: dict[str, object] = {}
    for patch in patches:
        if patch.field is SimulationPatchField.IDENTITY_STATUS:
            identity_updates["status"] = VerificationStatus(patch.value)
        elif patch.field is SimulationPatchField.COVERAGE_STATUS:
            coverage_updates["status"] = VerificationStatus(patch.value)
        elif patch.field is SimulationPatchField.CEILING_EXCEEDED:
            coverage_updates["ceiling_exceeded"] = patch.value.lower() == "true"
        elif patch.field is SimulationPatchField.PREAUTHORIZATION_REQUIRED:
            coverage_updates["preauthorization_required"] = patch.value.lower() == "true"

    identity = (
        eligibility_result.identity.model_copy(update=identity_updates)
        if identity_updates
        else eligibility_result.identity
    )
    coverage = (
        eligibility_result.coverage.model_copy(update=coverage_updates)
        if coverage_updates
        else eligibility_result.coverage
    )
    return eligibility_result.model_copy(update={"identity": identity, "coverage": coverage})


def run_targeted_simulation(
    case_id: str,
    patches: list[SimulationPatch],
    *,
    compiled_graph: Any | None = None,
) -> SimulationResult:
    """Simulation **ciblée** (Phase 9) — jamais le graphe entier réinvoqué,
    jamais de fichier copié/modifié, jamais de `case_id` synthétique : lit
    l'état réel déjà calculé (`eligibility_result`, lecture seule via
    `compiled_graph.get_state()`), le patche sur une copie, puis rejoue
    directement `agents.autonomous_decision_agent.agent.run()` — le seul
    résultat qui dépend de `eligibility_result` dans la matrice de décision.

    `compiled_graph` injectable (tests) ; `None` construit une instance
    depuis les paramètres d'environnement, même convention que
    `run_simulation`."""
    graph = (
        compiled_graph
        if compiled_graph is not None
        else compile_workflow_v2(CheckpointerFactory.from_settings().build())
    )
    config = make_thread_config(case_id)
    snapshot = graph.get_state(config)
    if not snapshot.values:
        return SimulationResult(
            case_id=case_id,
            applied=False,
            error="Dossier introuvable — jamais soumis, thread expiré, ou aucun document accepté.",
        )

    values = snapshot.values
    original_decision = _stringify(values.get("final_decision"))
    eligibility_result = values.get("eligibility_result")
    if eligibility_result is None:
        return SimulationResult(
            case_id=case_id,
            applied=False,
            original_decision=original_decision,
            error="Simulation ciblée impossible : résultat d'éligibilité non disponible pour ce dossier.",
        )
    if isinstance(eligibility_result, dict):
        eligibility_result = EligibilityResult.model_validate(eligibility_result)

    try:
        patched_eligibility = _apply_patches_to_eligibility(eligibility_result, patches)
        patched_state = dict(values)
        patched_state["eligibility_result"] = patched_eligibility
        result = run_autonomous_decision(case_id, patched_state)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 — une simulation en échec ne doit jamais lever
        return SimulationResult(
            case_id=case_id,
            applied=False,
            original_decision=original_decision,
            error=f"{type(exc).__name__}: {exc}",
        )

    simulated_decision = result.decision.value
    return SimulationResult(
        case_id=case_id,
        applied=True,
        original_decision=original_decision,
        simulated_decision=simulated_decision,
        decision_changed=simulated_decision != original_decision,
        simulated_reasons=list(result.justification),
    )
