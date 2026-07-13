"""Mesure du critère (a)/(c) de la Phase V2-5 (plan de refonte V2).

Compare, sur les fixtures réelles de `datasets/fixtures/valid/`, le statut
combiné de `medical_coding_agent` + `clinical_consistency_agent` +
`fraud_detection_agent` (V1, appelés **directement**, 3 appels LLM séparés)
au statut produit par `agents.medical_risk_agent.run()` (V2, fusion en un
seul appel LLM) — sur les **mêmes** entrées synthétiques, dérivées
directement de l'oracle (`oracle/case_data.json` + `oracle/ground_truth.json`)
plutôt que du graphe V1 complet.

Écart assumé par rapport à une première version de ce script (décision
AZIZ, option 2) : le graphe V1 complet (`graph.workflow.compile_workflow`)
s'est révélé instable avec un vrai LLM dans cet environnement — la
validation FHIR/OCR en amont ne complète pas de façon fiable jusqu'à
`fraud_detection` (non-déterminisme du LLM réel d'un appel à l'autre,
observé et non lié au code V2). Ce script s'affranchit donc entièrement du
graphe et des étapes amont (intake/sécurité/privacy/OCR/FHIR/identité) —
il compare **uniquement** la Phase codification+clinique+fraude,
directement, avec des entrées déterministes déjà connues (mêmes données
pour les deux côtés de la comparaison) : c'est exactement le périmètre du
critère (a)/(c), qui ne porte que sur cette fusion.

Hors CI, nécessite un vrai Ollama joignable (`OLLAMA_BASE_URL`). Coût :
4 appels LLM par dossier (3 pour V1 séparé + 1 pour V2 fusionné) —
nettement plus rapide et fiable que la version graphe complet.

Critères de déclenchement de la Phase V2-5-bis (plan V2 §4) :
  (a) taux de divergence de statut global (PASS/NEEDS_REVIEW/FAIL) entre les
      deux approches <= 10% des dossiers comparables ;
  (c) pas de hausse du taux de codes proposés hors référentiel (invalides)
      par rapport à V1.
(b) — aucun signal sans preuve — est une garantie de schéma déjà vérifiée
      par les tests (`tests/v2/agents/test_medical_risk_agent.py`), non
      mesurée ici.

Usage ::

    python scripts/evaluate_medical_risk_fusion.py --cases CLM-0001,CLM-0010 --format table
    python scripts/evaluate_medical_risk_fusion.py --limit 5
    python scripts/evaluate_medical_risk_fusion.py --output logs/evals/medical_risk_fusion.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIXTURES_ROOT = PROJECT_ROOT / "datasets" / "fixtures" / "valid"
_STATUS_RANK = {"PASS": 0, "NEEDS_REVIEW": 1, "FAIL": 2}


@dataclass
class CaseComparison:
    case_id: str
    v1_coding_status: str | None = None
    v1_clinical_status: str | None = None
    v1_fraud_status: str | None = None
    v1_overall_status: str | None = None
    v1_risk_score: float | None = None
    v1_invalid_codes: int = 0
    v2_status: str | None = None
    v2_risk_level: str | None = None
    v2_risk_score: float | None = None
    v2_invalid_codes: int = 0
    status_diverges: bool | None = None
    verdict: str = "ERROR"  # COMPARABLE | SKIPPED | ERROR
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


def _load_oracle(case_id: str) -> tuple[dict, dict] | None:
    """Charge case_data.json + ground_truth.json — None si absents."""
    oracle_dir = FIXTURES_ROOT / case_id / "oracle"
    case_data_path = oracle_dir / "case_data.json"
    ground_truth_path = oracle_dir / "ground_truth.json"
    if not case_data_path.is_file() or not ground_truth_path.is_file():
        return None
    with case_data_path.open(encoding="utf-8") as f:
        case_data = json.load(f)
    with ground_truth_path.open(encoding="utf-8") as f:
        ground_truth = json.load(f)
    return case_data, ground_truth


def _build_synthetic_inputs(case_data: dict, ground_truth: dict) -> dict[str, Any]:
    """Dérive des entrées synthétiques directement de l'oracle — jamais du
    graphe V1. `extracted_fields` reprend `expected_extraction` (déjà les
    valeurs réellement présentes dans les documents Synthea sources) ;
    `identity`/`coverage` reprennent `expected_identity`/`expected_coverage`.
    """
    from schemas.domain import VerificationStatus

    extraction = ground_truth.get("expected_extraction", {}) or {}
    identity_expected = ground_truth.get("expected_identity", {}) or {}
    coverage_expected = ground_truth.get("expected_coverage", {}) or {}
    deterministic_rules = ground_truth.get("deterministic_rules", {}) or {}

    procedures = [p.get("DESCRIPTION", "") for p in case_data.get("procedures", []) if p.get("DESCRIPTION")]
    medications = [m.get("DESCRIPTION", "") for m in case_data.get("medications", []) if m.get("DESCRIPTION")]

    extracted_fields = {
        "patient_id": extraction.get("patient_id"),
        "patient_name": extraction.get("patient_name"),
        "medication_count": str(extraction.get("medication_count", len(medications))),
        "procedure_count": str(extraction.get("procedure_count", len(procedures))),
        "prescription_number": extraction.get("prescription_number"),
        "service_date": extraction.get("service_date"),
    }
    ocr_result_like = SimpleNamespace(
        extracted_fields=extracted_fields,
        confidence_score=1.0,
        document_type=None,
        sha256=None,
    )

    def _status(raw: str | None) -> VerificationStatus:
        try:
            return VerificationStatus(raw or "PASS")
        except ValueError:
            return VerificationStatus.NEEDS_REVIEW

    identity_coverage_like = SimpleNamespace(
        identity=SimpleNamespace(status=_status(identity_expected.get("status"))),
        coverage=SimpleNamespace(
            status=_status(coverage_expected.get("status")),
            ceiling_exceeded=False,  # non présent dans l'oracle — jamais inventé, False par défaut
            preauthorization_required=bool(deterministic_rules.get("authorization_required", False)),
            preauthorization_status=str(deterministic_rules.get("authorization_status", "")),
        ),
    )

    return {
        "procedures": procedures,
        "medications": medications,
        "ocr_result": ocr_result_like,
        "identity_coverage_result": identity_coverage_like,
    }


def _worst_status(*statuses: str | None) -> str | None:
    known = [s for s in statuses if s]
    if not known:
        return None
    return max(known, key=lambda s: _STATUS_RANK.get(s, 1))


def _count_invalid_codes(codings: list) -> int:
    from tools.medical_coding import code_exists_in_reference

    invalid = 0
    for c in codings:
        if not c.proposed_code:
            continue
        if not (
            code_exists_in_reference(c.proposed_code, "procedures")
            or code_exists_in_reference(c.proposed_code, "medications")
        ):
            invalid += 1
    return invalid


def _evaluate_case(case_id: str) -> CaseComparison:
    from agents.clinical_consistency_agent.agent import run as run_clinical
    from agents.fraud_detection_agent.agent import run as run_fraud
    from agents.medical_coding_agent.agent import run as run_coding
    from agents.medical_risk_agent.agent import run as run_medical_risk

    started = time.monotonic()

    oracle = _load_oracle(case_id)
    if oracle is None:
        return CaseComparison(
            case_id=case_id,
            verdict="SKIPPED",
            duration_seconds=time.monotonic() - started,
            note="oracle/case_data.json ou oracle/ground_truth.json absent.",
        )
    case_data, ground_truth = oracle

    try:
        inputs = _build_synthetic_inputs(case_data, ground_truth)

        # ── V1 : 3 agents séparés, 3 appels LLM ──────────────────────────────
        coding_result = run_coding(
            case_id=case_id, procedures=inputs["procedures"], medications=inputs["medications"]
        )
        clinical_result = run_clinical(
            case_id=case_id, ocr_result=inputs["ocr_result"], coding_result=coding_result
        )
        fraud_result = run_fraud(
            case_id=case_id,
            identity_coverage_result=inputs["identity_coverage_result"],
            coding_result=coding_result,
            ocr_result=inputs["ocr_result"],
        )

        v1_coding_status = coding_result.status.value
        v1_clinical_status = clinical_result.status.value
        v1_fraud_status = fraud_result.status.value
        v1_overall = _worst_status(v1_coding_status, v1_clinical_status, v1_fraud_status)
        v1_invalid = _count_invalid_codes(coding_result.codings)

        # ── V2 : un seul agent fusionné, un seul appel LLM ───────────────────
        v2_result = run_medical_risk(
            case_id=case_id,
            procedures=inputs["procedures"],
            medications=inputs["medications"],
            ocr_result=inputs["ocr_result"],
            identity_coverage_result=inputs["identity_coverage_result"],
        )
        v2_invalid = _count_invalid_codes(v2_result.result_payload.codings)

    except Exception as exc:  # noqa: BLE001 — un dossier en échec ne doit jamais arrêter les autres
        return CaseComparison(
            case_id=case_id,
            verdict="ERROR",
            duration_seconds=time.monotonic() - started,
            note=f"{type(exc).__name__}: {exc}",
            errors=[traceback.format_exc(limit=3)],
        )

    diverges = v1_overall is not None and v1_overall != v2_result.status.value

    return CaseComparison(
        case_id=case_id,
        v1_coding_status=v1_coding_status,
        v1_clinical_status=v1_clinical_status,
        v1_fraud_status=v1_fraud_status,
        v1_overall_status=v1_overall,
        v1_risk_score=fraud_result.result_payload.risk_score,
        v1_invalid_codes=v1_invalid,
        v2_status=v2_result.status.value,
        v2_risk_level=v2_result.result_payload.risk_level.value,
        v2_risk_score=v2_result.result_payload.risk_score,
        v2_invalid_codes=v2_invalid,
        status_diverges=diverges,
        verdict="COMPARABLE",
        duration_seconds=time.monotonic() - started,
    )


def _print_table(comparisons: list[CaseComparison]) -> None:
    header = f"{'case_id':<12} {'v1_overall':<14} {'v2_status':<14} {'diverges':<9} {'verdict':<12} note"
    print(header)
    print("-" * len(header))
    for c in comparisons:
        print(
            f"{c.case_id:<12} {str(c.v1_overall_status):<14} {str(c.v2_status):<14} "
            f"{str(c.status_diverges):<9} {c.verdict:<12} {c.note}"
        )


def _summarize(comparisons: list[CaseComparison]) -> dict[str, Any]:
    comparable = [c for c in comparisons if c.verdict == "COMPARABLE"]
    diverging = [c for c in comparable if c.status_diverges]
    v1_invalid_total = sum(c.v1_invalid_codes for c in comparable)
    v2_invalid_total = sum(c.v2_invalid_codes for c in comparable)

    counts = {"COMPARABLE": 0, "SKIPPED": 0, "ERROR": 0}
    for c in comparisons:
        counts[c.verdict] = counts.get(c.verdict, 0) + 1

    divergence_rate = (len(diverging) / len(comparable)) if comparable else None
    criterion_a_passed = divergence_rate is not None and divergence_rate <= 0.10
    criterion_c_passed = v2_invalid_total <= v1_invalid_total

    return {
        "total_cases": len(comparisons),
        "comparable_cases": len(comparable),
        "counts": counts,
        "divergence_rate": divergence_rate,
        "diverging_case_ids": [c.case_id for c in diverging],
        "v1_invalid_codes_total": v1_invalid_total,
        "v2_invalid_codes_total": v2_invalid_total,
        "criterion_a_divergence_le_10pct": criterion_a_passed,
        "criterion_c_no_increase_invalid_codes": criterion_c_passed,
        "v2_5_bis_recommended": bool(comparable) and not (criterion_a_passed and criterion_c_passed),
    }


def run_evaluation(
    case_ids: list[str], *, progress_path: Path | None = None
) -> tuple[list[CaseComparison], dict[str, Any]]:
    from config.settings import get_settings

    comparisons: list[CaseComparison] = []
    total = len(case_ids)
    for index, case_id in enumerate(case_ids, start=1):
        comparison = _evaluate_case(case_id)
        comparisons.append(comparison)
        print(
            f"[{index}/{total}] {case_id} -> verdict={comparison.verdict} "
            f"v1={comparison.v1_overall_status} v2={comparison.v2_status} "
            f"diverges={comparison.status_diverges} "
            f"({comparison.duration_seconds:.1f}s){' — ' + comparison.note if comparison.note else ''}",
            file=sys.stderr,
            flush=True,
        )
        if progress_path is not None:
            partial_summary = _summarize(comparisons)
            _write_output(progress_path, comparisons, partial_summary)

    summary = _summarize(comparisons)
    settings = get_settings()
    summary["run_metadata"] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "llm_provider": settings.claimshield_llm_provider,
        "llm_model": settings.claimshield_llm_model,
        "cases_requested": len(case_ids),
    }
    return comparisons, summary


def _write_output(path: Path, comparisons: list[CaseComparison], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "cases": [asdict(c) for c in comparisons]}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", default="all", help="'all' ou liste séparée par virgules")
    parser.add_argument("--limit", type=int, default=None, help="Limite le nombre de dossiers évalués")
    parser.add_argument("--output", type=Path, default=None, help="Chemin de sortie JSON")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    args = parser.parse_args()

    case_ids = _discover_case_ids(args.cases, args.limit)
    if not case_ids:
        print("Aucun dossier à évaluer.", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or (
        PROJECT_ROOT / "logs" / "evals" / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_medical_risk_fusion.json"
    )
    print(f"Progression écrite au fur et à mesure dans : {output_path}", file=sys.stderr)

    comparisons, summary = run_evaluation(case_ids, progress_path=output_path)

    if args.format == "table":
        _print_table(comparisons)
        print()
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    else:
        print(json.dumps({"summary": summary, "cases": [asdict(c) for c in comparisons]}, ensure_ascii=False, indent=2, default=str))

    _write_output(output_path, comparisons, summary)
    print(f"\nRapport final écrit : {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
