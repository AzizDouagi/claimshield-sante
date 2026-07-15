"""Évaluation qualité du pipeline V2 complet contre l'oracle — ClaimShield Santé.

Plan de refonte V2, Phase V2-10 (jalon de stabilité). Même patron que
``scripts/evaluate_recommendations.py`` (V1, non modifié) adapté au graphe
V2 : hors CI, nécessite un vrai Ollama joignable (``OLLAMA_BASE_URL``).
Invoque le graphe V2 compilé (``graph.workflow_v2.compile_workflow_v2``)
avec le **vrai** LLM sur chaque dossier de ``datasets/fixtures/valid/``, et
compare ``state["final_decision"]`` (``schemas.domain.ClaimDecisionV2``,
6 issues — jamais une pré-recommandation intermédiaire : le graphe V2 ne
s'interrompt jamais, ``final_decision`` est toujours la décision terminale
réellement produite) à ``expected_recommendation`` de l'oracle
(``oracle/ground_truth.json``).

Différence assumée avec le script V1 : l'oracle (``expected_recommendation``)
est un champ à 3 valeurs hérité de V1 (`APPROVE`/`REJECT`/`PENDING`, voir
``schemas.domain.Recommendation``) — jamais réécrit pour la V2 (aucun nouvel
oracle à 6 valeurs n'existe). La comparaison utilise donc une table de
compatibilité documentée (``_COMPATIBLE_V2_DECISIONS``) plutôt qu'une égalité
stricte de chaîne : un oracle `APPROVE` est jugé `MATCH` si `final_decision`
est `APPROVE` ou `PARTIAL_APPROVE` (approbation totale ou partielle, jamais un
rejet/gel silencieux) ; `REJECT` n'est `MATCH` que pour `REJECT` ; `PENDING`
(V1 : recommandation indéterminée) est `MATCH` pour `REQUEST_MORE_INFO`,
`QUARANTINE` ou `PARTIAL_APPROVE` (tous des issues de prudence, cohérentes
avec une indétermination V1). Cette table n'invente aucune correspondance
métier nouvelle — elle documente une équivalence de prudence, jamais un
critère de qualité inventé après coup.

Usage ::

    python scripts/evaluate_recommendations_v2.py
    python scripts/evaluate_recommendations_v2.py --cases CLM-0001,CLM-0007 --format table
    python scripts/evaluate_recommendations_v2.py --limit 5 --output logs/evals/run_v2.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIXTURES_ROOT = PROJECT_ROOT / "datasets" / "fixtures" / "valid"
DEFAULT_ROLE = "ADMINISTRATIVE_MANAGER"

_COMPATIBLE_V2_DECISIONS: dict[str, frozenset[str]] = {
    "APPROVE": frozenset({"APPROVE", "PARTIAL_APPROVE"}),
    "REJECT": frozenset({"REJECT"}),
    "PENDING": frozenset({"REQUEST_MORE_INFO", "QUARANTINE", "PARTIAL_APPROVE"}),
}
"""Table de compatibilité oracle V1 (3 valeurs) -> décision V2 (6 valeurs) —
voir docstring du module. Un oracle absent de cette table (valeur inconnue)
n'est jamais comparé (verdict ERROR, jamais un MATCH par défaut silencieux)."""


@dataclass
class CaseEvaluation:
    case_id: str
    expected: str | None
    actual: str | None
    match: bool | None
    verdict: str  # MATCH | MISMATCH | ERROR
    current_step: str | None = None
    bounded_by: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    note: str = ""
    errors: list[str] = field(default_factory=list)


def _discover_case_ids(selection: str, limit: int | None) -> list[str]:
    if selection != "all":
        case_ids = [c.strip() for c in selection.split(",") if c.strip()]
    else:
        case_ids = sorted(
            p.name for p in FIXTURES_ROOT.iterdir() if p.is_dir() and p.name.startswith("CLM-")
        )
    if limit is not None:
        case_ids = case_ids[:limit]
    return case_ids


def _load_expected_recommendation(case_id: str) -> str | None:
    gt_path = FIXTURES_ROOT / case_id / "oracle" / "ground_truth.json"
    if not gt_path.is_file():
        return None
    with gt_path.open(encoding="utf-8") as f:
        return json.load(f).get("expected_recommendation")


def _build_initial_state(case_id: str, source_path: Path) -> dict[str, Any]:
    """Même forme que ``api.v2.claims.submit_claim_v2`` — ne duplique pas
    ``compile_workflow_v2``/``graph.workflow_v2`` (trop volumineux pour être
    factorisé utilement ici), seulement les sept champs d'état initial."""
    return {
        "case_id": case_id,
        "schema_version": "2.0.0",
        "current_step": "initial",
        "completed_steps": [],
        "errors": [],
        "alerts": [],
        "intake_input": {
            "source_path": str(source_path),
            "required_documents": [],
        },
        "reader_role": DEFAULT_ROLE,
    }


def _reset_storage_for_case(case_id: str) -> None:
    """Supprime les artefacts de stockage d'une exécution antérieure pour ce
    `case_id` avant de le rejouer — corrige la collision `NO_OVERWRITE`
    (`services/storage.py::commit_file`, nommage physique déterministe de
    `tools/file_inspection.py::build_storage_name`) qui a produit un faux
    `TECHNICAL_FAILURE` sur CLM-0001 lors de la mesure V2-10 (artefacts
    périmés d'un smoke test antérieur). Idempotent : ne fait rien si aucun
    artefact n'existe. N'agit jamais sur `datasets/fixtures/` (entrée en
    lecture seule) ni sur l'oracle."""
    import shutil

    from services.storage import StorageService

    svc = StorageService()
    for base_dir in (svc.incoming_dir, svc.quarantine_dir):
        case_dir = base_dir / case_id
        if case_dir.exists():
            shutil.rmtree(case_dir)
    manifest_path = svc.manifests_dir / f"{case_id}.json"
    if manifest_path.exists():
        manifest_path.unlink()


def _evaluate_case(app: Any, case_id: str, *, reset_storage: bool = True) -> CaseEvaluation:
    from graph.checkpoints import make_thread_config

    if reset_storage:
        _reset_storage_for_case(case_id)

    expected = _load_expected_recommendation(case_id)
    source_path = FIXTURES_ROOT / case_id / "input"

    started = time.monotonic()
    try:
        config = make_thread_config(case_id)
        result = app.invoke(_build_initial_state(case_id, source_path), config=config)
    except Exception as exc:  # noqa: BLE001 — un dossier en échec ne doit jamais arrêter les autres
        duration = time.monotonic() - started
        return CaseEvaluation(
            case_id=case_id,
            expected=expected,
            actual=None,
            match=None,
            verdict="ERROR",
            duration_seconds=duration,
            note=f"{type(exc).__name__}: {exc}",
            errors=[traceback.format_exc(limit=3)],
        )
    duration = time.monotonic() - started

    final_decision = result.get("final_decision")
    current_step = result.get("current_step")
    errors = list(result.get("errors") or [])
    decision_result = result.get("decision_result")
    bounded_by = list(getattr(decision_result, "bounded_by", None) or [])

    actual = getattr(final_decision, "value", final_decision)
    compatible = _COMPATIBLE_V2_DECISIONS.get(str(expected)) if expected is not None else None

    if actual is None or compatible is None:
        return CaseEvaluation(
            case_id=case_id,
            expected=expected,
            actual=actual,
            match=None,
            verdict="ERROR",
            current_step=current_step,
            bounded_by=bounded_by,
            duration_seconds=duration,
            note="Décision finale absente ou oracle hors table de compatibilité.",
            errors=errors,
        )

    match = actual in compatible
    return CaseEvaluation(
        case_id=case_id,
        expected=expected,
        actual=actual,
        match=match,
        verdict="MATCH" if match else "MISMATCH",
        current_step=current_step,
        bounded_by=bounded_by,
        duration_seconds=duration,
        errors=errors,
    )


def _print_table(evaluations: list[CaseEvaluation]) -> None:
    header = f"{'case_id':<12} {'expected':<10} {'actual':<18} {'verdict':<10} note"
    print(header)
    print("-" * len(header))
    for ev in evaluations:
        print(
            f"{ev.case_id:<12} {str(ev.expected):<10} {str(ev.actual):<18} "
            f"{ev.verdict:<10} {ev.note}"
        )


def _summarize(evaluations: list[CaseEvaluation]) -> dict[str, Any]:
    comparable = [ev for ev in evaluations if ev.verdict in {"MATCH", "MISMATCH"}]
    matches = [ev for ev in comparable if ev.match]
    confusion: dict[str, dict[str, int]] = {}
    for ev in comparable:
        confusion.setdefault(str(ev.expected), {}).setdefault(str(ev.actual), 0)
        confusion[str(ev.expected)][str(ev.actual)] += 1

    counts = {"MATCH": 0, "MISMATCH": 0, "ERROR": 0}
    for ev in evaluations:
        counts[ev.verdict] = counts.get(ev.verdict, 0) + 1

    return {
        "total_cases": len(evaluations),
        "comparable_cases": len(comparable),
        "match_rate": (len(matches) / len(comparable)) if comparable else None,
        "counts": counts,
        "confusion_matrix": confusion,
    }


def run_evaluation(
    case_ids: list[str], *, reset_storage: bool = True
) -> tuple[list[CaseEvaluation], dict[str, Any]]:
    from config.settings import get_settings
    from graph.workflow_v2 import compile_workflow_v2
    from langgraph.checkpoint.memory import InMemorySaver

    app = compile_workflow_v2(InMemorySaver())
    evaluations = [_evaluate_case(app, case_id, reset_storage=reset_storage) for case_id in case_ids]
    summary = _summarize(evaluations)
    settings = get_settings()
    summary["run_metadata"] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "llm_provider": settings.claimshield_llm_provider,
        "llm_model": settings.claimshield_llm_model,
        "cases_requested": len(case_ids),
    }
    return evaluations, summary


def _write_output(path: Path, evaluations: list[CaseEvaluation], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(evaluations[0]).keys()) if evaluations else [])
            writer.writeheader()
            for ev in evaluations:
                writer.writerow(asdict(ev))
    else:
        payload = {"summary": summary, "cases": [asdict(ev) for ev in evaluations]}
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", default="all", help="'all' ou liste séparée par virgules (ex. CLM-0001,CLM-0007)")
    parser.add_argument("--limit", type=int, default=None, help="Limite le nombre de dossiers évalués")
    parser.add_argument("--output", type=Path, default=None, help="Chemin de sortie JSON/CSV (défaut : logs/evals/<horodatage>.json)")
    parser.add_argument("--format", choices=["table", "json", "csv"], default="table", help="Format d'affichage console")
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help=(
            "Désactive le nettoyage automatique de storage/{incoming,quarantine,manifests}/<case_id> "
            "avant chaque dossier (actif par défaut — évite une fausse collision NO_OVERWRITE/"
            "TECHNICAL_FAILURE sur un case_id déjà traité par un run antérieur)."
        ),
    )
    args = parser.parse_args()

    case_ids = _discover_case_ids(args.cases, args.limit)
    if not case_ids:
        print("Aucun dossier à évaluer.", file=sys.stderr)
        sys.exit(1)

    evaluations, summary = run_evaluation(case_ids, reset_storage=not args.no_reset)

    if args.format == "table":
        _print_table(evaluations)
        print()
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    else:
        payload = {"summary": summary, "cases": [asdict(ev) for ev in evaluations]}
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))

    output_path = args.output or (
        PROJECT_ROOT / "logs" / "evals" / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_recommendations_v2.json"
    )
    _write_output(output_path, evaluations, summary)
    print(f"\nRapport écrit : {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
