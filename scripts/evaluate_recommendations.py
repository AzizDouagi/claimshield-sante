"""Évaluation qualité du pipeline complet contre l'oracle — ClaimShield Santé.

Hors CI, nécessite un vrai Ollama joignable (``OLLAMA_BASE_URL``) : contrairement
à la suite pytest (LLM systématiquement stubbé via ``tests.conftest.deterministic_agent_llm``),
ce script invoque le graphe compilé avec le **vrai** LLM sur chacun des dossiers
de ``datasets/fixtures/valid/``, et compare la pré-recommandation de
``case_reviewer_agent`` (``review_result.result_payload.recommendation`` — la
seule recommandation que le pipeline puisse produire sans intervention
humaine, cf. verrouillage de ``CaseReviewerResult`` à l'étape 13 : le LLM n'a
jamais d'autorité finale) à ``expected_recommendation`` de l'oracle
(``oracle/ground_truth.json``).

Limite connue et actuellement bloquante (voir CLAUDE.md, section « Finalisation
post-Phase 4 ») : rien en production ne construit ``ocr_input``/``fhir_input``
à partir du manifest de ``claim_intake_agent`` — toute soumission avec de vrais
documents échoue donc aujourd'hui sur ``document_ocr``/``fhir_validator`` avant
d'atteindre ``case_reviewer``. Tant que ce câblage n'existe pas, ce script
classera systématiquement chaque dossier en ``EARLY_EXIT`` plutôt que de
produire une comparaison exploitable — il reste néanmoins l'infrastructure
prête pour le jour où ce câblage existera, et sert déjà à vérifier qu'aucun
dossier ne fait planter le pipeline (panne technique distincte de l'absence de
recommandation).

Usage ::

    python scripts/evaluate_recommendations.py
    python scripts/evaluate_recommendations.py --cases CLM-0001,CLM-0007 --format table
    python scripts/evaluate_recommendations.py --limit 5 --output logs/evals/run1.json
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


@dataclass
class CaseEvaluation:
    case_id: str
    expected: str | None
    actual: str | None
    match: bool | None
    verdict: str  # MATCH | MISMATCH | EARLY_EXIT | ERROR
    current_step: str | None = None
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
    """Même forme que ``api.main.submit_claim`` — ne duplique pas
    ``compile_workflow``/``graph.workflow`` (trop volumineux pour être
    factorisé utilement ici), seulement les six champs d'état initial."""
    return {
        "case_id": case_id,
        "schema_version": "1.0.0",
        "current_step": "initial",
        "completed_steps": [],
        "errors": [],
        "alerts": [],
        "final_justification": [],
        "intake_input": {
            "case_id": case_id,
            "source_path": str(source_path),
            "required_documents": [],
            "uploaded_files": [],
        },
        "privacy_input": {"case_id": case_id, "role": DEFAULT_ROLE},
    }


def _evaluate_case(app: Any, case_id: str) -> CaseEvaluation:
    from graph.checkpoints import make_thread_config

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

    review_result = result.get("review_result")
    current_step = result.get("current_step")
    errors = list(result.get("errors") or [])

    if review_result is None:
        return CaseEvaluation(
            case_id=case_id,
            expected=expected,
            actual=None,
            match=None,
            verdict="EARLY_EXIT",
            current_step=current_step,
            duration_seconds=duration,
            note="Pipeline interrompu avant case_reviewer (voir errors).",
            errors=errors,
        )

    actual = review_result.result_payload.recommendation.value
    match = expected is not None and actual == expected
    return CaseEvaluation(
        case_id=case_id,
        expected=expected,
        actual=actual,
        match=match,
        verdict="MATCH" if match else "MISMATCH",
        current_step=current_step,
        duration_seconds=duration,
        errors=errors,
    )


def _print_table(evaluations: list[CaseEvaluation]) -> None:
    header = f"{'case_id':<12} {'expected':<10} {'actual':<10} {'verdict':<12} note"
    print(header)
    print("-" * len(header))
    for ev in evaluations:
        print(
            f"{ev.case_id:<12} {str(ev.expected):<10} {str(ev.actual):<10} "
            f"{ev.verdict:<12} {ev.note}"
        )


def _summarize(evaluations: list[CaseEvaluation]) -> dict[str, Any]:
    comparable = [ev for ev in evaluations if ev.verdict in {"MATCH", "MISMATCH"}]
    matches = [ev for ev in comparable if ev.match]
    confusion: dict[str, dict[str, int]] = {}
    for ev in comparable:
        confusion.setdefault(str(ev.expected), {}).setdefault(str(ev.actual), 0)
        confusion[str(ev.expected)][str(ev.actual)] += 1

    counts = {"MATCH": 0, "MISMATCH": 0, "EARLY_EXIT": 0, "ERROR": 0}
    for ev in evaluations:
        counts[ev.verdict] = counts.get(ev.verdict, 0) + 1

    return {
        "total_cases": len(evaluations),
        "comparable_cases": len(comparable),
        "match_rate": (len(matches) / len(comparable)) if comparable else None,
        "counts": counts,
        "confusion_matrix": confusion,
    }


def run_evaluation(case_ids: list[str]) -> tuple[list[CaseEvaluation], dict[str, Any]]:
    from config.settings import get_settings
    from graph.workflow import compile_workflow
    from langgraph.checkpoint.memory import InMemorySaver

    app = compile_workflow(InMemorySaver(), interrupt_before=[])
    evaluations = [_evaluate_case(app, case_id) for case_id in case_ids]
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
    args = parser.parse_args()

    case_ids = _discover_case_ids(args.cases, args.limit)
    if not case_ids:
        print("Aucun dossier à évaluer.", file=sys.stderr)
        sys.exit(1)

    evaluations, summary = run_evaluation(case_ids)

    if args.format == "table":
        _print_table(evaluations)
        print()
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    else:
        payload = {"summary": summary, "cases": [asdict(ev) for ev in evaluations]}
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))

    output_path = args.output or (
        PROJECT_ROOT / "logs" / "evals" / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_recommendations.json"
    )
    _write_output(output_path, evaluations, summary)
    print(f"\nRapport écrit : {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
