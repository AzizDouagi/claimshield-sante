from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings  # noqa: E402


CASE_PATTERN = re.compile(r"^CLM-\d{4}$")
IGNORE_NAMES = {".DS_Store", "generation.log"}
PDF_RENAMES = {
    "claim_request.pdf": "demande_remboursement_{case_id}.pdf",
    "medical_invoice.pdf": "facture_{case_id}.pdf",
    "prescription.pdf": "ordonnance_{case_id}.pdf",
    "encounter_summary.pdf": "compte_rendu_{case_id}.pdf",
}


def sha256_bytes(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_patient_ids_from_fhir(payload: Any) -> set[str]:
    patient_ids: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("resourceType") == "Patient" and value.get("id"):
                patient_ids.add(str(value["id"]))
            reference = value.get("reference")
            if isinstance(reference, str) and reference.startswith("Patient/"):
                patient_ids.add(reference.split("/", 1)[1])
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    return patient_ids


@dataclass
class PlannedFile:
    relative_path: Path
    source_path: Path


@dataclass
class ValidationResult:
    ok: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class ImportStats:
    found_cases: int = 0
    copied_cases: int = 0
    unchanged_cases: int = 0
    backups_created: int = 0
    errors: int = 0
    warnings: list[str] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)
    validations: list[dict[str, Any]] = field(default_factory=list)


class ClaimshieldImporter:
    def __init__(self, project_root: Path) -> None:
        settings = get_settings()
        self.project_root = project_root
        self.source_root = settings.claimshield_source_root
        self.source_generated = self.source_root / "generated"
        self.fixtures_root = settings.datasets_dir / "fixtures"
        self.valid_root = self.fixtures_root / "valid"
        self.backups_root = self.fixtures_root / "backups"
        self.metadata_root = self.fixtures_root / "metadata"
        self.report_path = self.metadata_root / "import_report.json"
        self.logger = self._build_logger()

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger("claimshield_importer")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.addHandler(logging.StreamHandler())
        logger.propagate = False
        return logger

    def ensure_directories(self) -> None:
        self.valid_root.mkdir(parents=True, exist_ok=True)
        self.backups_root.mkdir(parents=True, exist_ok=True)
        self.metadata_root.mkdir(parents=True, exist_ok=True)

    def list_source_cases(self, case_id: str | None = None) -> list[Path]:
        if not self.source_generated.exists():
            raise FileNotFoundError(f"Source introuvable : {self.source_generated}")
        cases = [path for path in sorted(self.source_generated.iterdir()) if path.is_dir() and CASE_PATTERN.match(path.name)]
        if case_id:
            cases = [path for path in cases if path.name == case_id]
        return cases

    def plan_case_files(self, source_case_dir: Path) -> list[PlannedFile]:
        case_id = source_case_dir.name
        planned: list[PlannedFile] = []

        for source_name, target_pattern in PDF_RENAMES.items():
            source_path = source_case_dir / source_name
            if source_path.exists():
                planned.append(
                    PlannedFile(
                        relative_path=Path("input") / target_pattern.format(case_id=case_id),
                        source_path=source_path,
                    )
                )

        for filename in ("patient.json", "claim.json", "patient_fhir_bundle.json"):
            source_path = source_case_dir / filename
            if source_path.exists():
                planned.append(PlannedFile(Path("input") / filename, source_path))

        fhir_dir = source_case_dir / "fhir"
        if fhir_dir.exists():
            for source_path in sorted(fhir_dir.rglob("*")):
                if not source_path.is_file() or source_path.name.startswith(".") or source_path.name in IGNORE_NAMES:
                    continue
                planned.append(
                    PlannedFile(
                        relative_path=Path("input") / "fhir" / source_path.relative_to(fhir_dir),
                        source_path=source_path,
                    )
                )

        case_data_path = source_case_dir / "case_data.json"
        if case_data_path.exists():
            planned.append(PlannedFile(Path("oracle") / "case_data.json", case_data_path))

        ground_truth_path = source_case_dir / "ground_truth.json"
        if ground_truth_path.exists():
            planned.append(PlannedFile(Path("oracle") / "ground_truth.json", ground_truth_path))

        manifest_path = source_case_dir / "manifest.json"
        if manifest_path.exists():
            planned.append(PlannedFile(Path("audit") / "manifest.json", manifest_path))

        return planned

    def compute_source_signature(self, planned_files: list[PlannedFile]) -> dict[str, str]:
        signature: dict[str, str] = {}
        for item in planned_files:
            signature[str(item.relative_path)] = sha256_file(item.source_path)
        return signature

    def compute_destination_signature(self, destination_case_dir: Path) -> dict[str, str]:
        signature: dict[str, str] = {}
        if not destination_case_dir.exists():
            return signature
        for path in sorted(destination_case_dir.rglob("*")):
            if not path.is_file() or path.name.startswith(".") or path.name in IGNORE_NAMES:
                continue
            signature[str(path.relative_to(destination_case_dir))] = sha256_file(path)
        return signature

    def backup_case(self, destination_case_dir: Path) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.backups_root / f"{destination_case_dir.name}_{timestamp}"
        shutil.copytree(destination_case_dir, backup_dir)
        return backup_dir

    def copy_case(self, source_case_dir: Path, destination_case_dir: Path, planned_files: list[PlannedFile]) -> None:
        with tempfile.TemporaryDirectory(dir=self.valid_root) as temp_dir_name:
            temp_dir = Path(temp_dir_name) / destination_case_dir.name
            temp_dir.mkdir(parents=True, exist_ok=True)
            (temp_dir / "input").mkdir(parents=True, exist_ok=True)
            (temp_dir / "oracle").mkdir(parents=True, exist_ok=True)
            (temp_dir / "audit").mkdir(parents=True, exist_ok=True)

            for item in planned_files:
                target_path = temp_dir / item.relative_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item.source_path, target_path)

            if destination_case_dir.exists():
                shutil.rmtree(destination_case_dir)
            temp_dir.replace(destination_case_dir)

    def copy_metadata_files(self, dry_run: bool, operations: list[str]) -> None:
        for filename in ("index.json", "generation_report.json", "generation.log"):
            source_path = self.source_root / filename
            if not source_path.exists():
                continue
            destination_path = self.metadata_root / filename
            operations.append(f"metadata copy {source_path} -> {destination_path}")
            if dry_run:
                continue
            shutil.copy2(source_path, destination_path)

    def validate_case(self, destination_case_dir: Path) -> ValidationResult:
        warnings: list[str] = []
        errors: list[str] = []
        case_id = destination_case_dir.name

        input_dir = destination_case_dir / "input"
        if not input_dir.exists():
            errors.append("input/ absent")
            return ValidationResult(False, warnings, errors)

        pdfs = list(input_dir.glob("*.pdf"))
        if not pdfs:
            errors.append("Aucun PDF dans input/")
        for pdf_path in pdfs:
            if case_id not in pdf_path.name:
                errors.append(f"Identifiant CLM absent du nom de fichier : {pdf_path.name}")
            if pdf_path.stat().st_size == 0:
                errors.append(f"Fichier vide : {pdf_path.relative_to(destination_case_dir)}")
            else:
                try:
                    reader = PdfReader(str(pdf_path))
                    if len(reader.pages) < 1:
                        errors.append(f"PDF sans page : {pdf_path.name}")
                except Exception as exc:  # pragma: no cover
                    errors.append(f"PDF illisible : {pdf_path.name} ({exc})")

        fhir_dir = input_dir / "fhir"
        if not fhir_dir.exists():
            warnings.append("Dossier FHIR absent")

        manifest_path = destination_case_dir / "audit" / "manifest.json"
        manifest = None
        if not manifest_path.exists():
            errors.append("audit/manifest.json absent")
        else:
            if manifest_path.stat().st_size == 0:
                errors.append("audit/manifest.json vide")
            try:
                manifest = read_json(manifest_path)
            except json.JSONDecodeError:
                errors.append("audit/manifest.json invalide")

        patient_data = None
        case_data = None

        patient_json_path = input_dir / "patient.json"
        claim_json_path = input_dir / "claim.json"
        case_data_path = destination_case_dir / "oracle" / "case_data.json"

        if patient_json_path.exists():
            patient_data = read_json(patient_json_path)
        elif case_data_path.exists():
            case_data = read_json(case_data_path)
            patient_data = case_data.get("patient")

        if not claim_json_path.exists() and case_data_path.exists():
            if case_data is None:
                case_data = read_json(case_data_path)

        if case_data_path.exists() and case_data is None:
            case_data = read_json(case_data_path)

        if case_data is not None:
            if case_data.get("synthetic") is not True and case_data.get("source", {}).get("contains_real_patient_data") is not False:
                errors.append("Case data ne prouve pas clairement le caractere synthetique")
            encounter = case_data.get("encounter") or {}
            patient = case_data.get("patient") or {}
            claim = case_data.get("claim") or {}
            if encounter.get("PATIENT") and patient.get("Id") and encounter.get("PATIENT") != patient.get("Id"):
                errors.append("Incoherence patient/encounter")
            if claim.get("PATIENTID") and patient.get("Id") and claim.get("PATIENTID") != patient.get("Id"):
                errors.append("Incoherence patient/claim")
            if encounter.get("Id") and claim.get("APPOINTMENTID") and encounter.get("Id") != claim.get("APPOINTMENTID"):
                errors.append("Incoherence encounter/claim")

        if manifest is not None:
            dataset = manifest.get("dataset", {})
            if dataset.get("contains_real_personal_data") is not False:
                errors.append("Le manifest ne marque pas clairement les donnees comme synthetiques")

        fhir_patient_ids: set[str] = set()
        fhir_json_files = [input_dir / "patient_fhir_bundle.json"]
        if fhir_dir.exists():
            fhir_json_files.extend(sorted(fhir_dir.glob("*.json")))
        for fhir_json in fhir_json_files:
            if not fhir_json.exists():
                continue
            if fhir_json.stat().st_size == 0:
                errors.append(f"Fichier vide : {fhir_json.relative_to(destination_case_dir)}")
                continue
            try:
                fhir_patient_ids |= extract_patient_ids_from_fhir(read_json(fhir_json))
            except json.JSONDecodeError:
                errors.append(f"FHIR JSON invalide : {fhir_json.relative_to(destination_case_dir)}")

        if patient_data and patient_data.get("Id") and fhir_patient_ids and patient_data.get("Id") not in fhir_patient_ids:
            errors.append("Les ressources FHIR ne correspondent pas au patient")

        for path in sorted(destination_case_dir.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            if path.stat().st_size == 0:
                errors.append(f"Fichier vide : {path.relative_to(destination_case_dir)}")
            if path.suffix.lower() == ".json":
                content = path.read_text(encoding="utf-8")
                if str(self.source_root) in content:
                    errors.append(f"Chemin source Synthea encore present dans {path.relative_to(destination_case_dir)}")

        return ValidationResult(not errors, warnings, errors)

    def run(self, case_id: str | None, dry_run: bool, force: bool, validate_only: bool) -> ImportStats:
        self.ensure_directories()
        stats = ImportStats()

        if validate_only:
            destination_cases = [
                path for path in sorted(self.valid_root.iterdir())
                if path.is_dir() and CASE_PATTERN.match(path.name) and (case_id is None or path.name == case_id)
            ]
            stats.found_cases = len(destination_cases)
            for destination_case_dir in destination_cases:
                validation = self.validate_case(destination_case_dir)
                stats.validations.append(
                    {
                        "case_id": destination_case_dir.name,
                        "ok": validation.ok,
                        "warnings": validation.warnings,
                        "errors": validation.errors,
                    }
                )
                stats.warnings.extend(validation.warnings)
                if validation.ok:
                    stats.unchanged_cases += 1
                else:
                    stats.errors += 1
            self.write_report(stats, dry_run, validate_only)
            return stats

        source_cases = self.list_source_cases(case_id)
        stats.found_cases = len(source_cases)

        for source_case_dir in source_cases:
            case_name = source_case_dir.name
            destination_case_dir = self.valid_root / case_name
            planned_files = self.plan_case_files(source_case_dir)
            source_signature = self.compute_source_signature(planned_files)
            destination_signature = self.compute_destination_signature(destination_case_dir)

            if not planned_files:
                stats.errors += 1
                stats.validations.append({"case_id": case_name, "ok": False, "warnings": [], "errors": ["Aucun fichier a importer"]})
                continue

            if destination_case_dir.exists() and destination_signature == source_signature:
                stats.unchanged_cases += 1
                stats.operations.append(f"skip unchanged {case_name}")
                validation = self.validate_case(destination_case_dir)
                stats.validations.append({"case_id": case_name, "ok": validation.ok, "warnings": validation.warnings, "errors": validation.errors})
                if not validation.ok:
                    stats.errors += 1
                continue

            if destination_case_dir.exists() and destination_signature != source_signature and not force:
                warning = f"{case_name}: destination differente, utiliser --force pour sauvegarder puis remplacer"
                stats.warnings.append(warning)
                stats.operations.append(f"skip differing {case_name}")
                validation = self.validate_case(destination_case_dir)
                stats.validations.append({"case_id": case_name, "ok": validation.ok, "warnings": validation.warnings, "errors": validation.errors})
                continue

            if destination_case_dir.exists() and destination_signature != source_signature:
                backup_dir = self.backups_root / f"{case_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                stats.operations.append(f"backup {destination_case_dir} -> {backup_dir}")
                if not dry_run:
                    shutil.copytree(destination_case_dir, backup_dir)
                stats.backups_created += 1

            stats.operations.append(f"copy {source_case_dir} -> {destination_case_dir}")
            if not dry_run:
                self.copy_case(source_case_dir, destination_case_dir, planned_files)
                validation = self.validate_case(destination_case_dir)
            else:
                validation = ValidationResult(True, [], [])
            stats.validations.append({"case_id": case_name, "ok": validation.ok, "warnings": validation.warnings, "errors": validation.errors})
            stats.warnings.extend(validation.warnings)
            if validation.ok:
                stats.copied_cases += 1
            else:
                stats.errors += 1

        self.copy_metadata_files(dry_run, stats.operations)
        self.write_report(stats, dry_run, validate_only)
        return stats

    def write_report(self, stats: ImportStats, dry_run: bool, validate_only: bool) -> None:
        report = {
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "source_directory": str(self.source_root),
            "source_generated_directory": str(self.source_generated),
            "destination_directory": str(self.valid_root),
            "dry_run": dry_run,
            "validate_only": validate_only,
            "cases_found": stats.found_cases,
            "cases_copied": stats.copied_cases,
            "cases_unchanged": stats.unchanged_cases,
            "backups_created": stats.backups_created,
            "errors_count": stats.errors,
            "warnings": stats.warnings,
            "operations": stats.operations,
            "validations": stats.validations,
        }
        atomic_write_json(self.report_path, report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Importer les dossiers ClaimShield synthetiques dans claimshield-sante.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--case", dest="case_id", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    importer = ClaimshieldImporter(PROJECT_ROOT)
    stats = importer.run(
        case_id=args.case_id,
        dry_run=args.dry_run,
        force=args.force,
        validate_only=args.validate_only,
    )
    print(f"Dossiers trouves : {stats.found_cases}")
    print(f"Dossiers copies : {stats.copied_cases}")
    print(f"Dossiers inchanges : {stats.unchanged_cases}")
    print(f"Sauvegardes creees : {stats.backups_created}")
    print(f"Erreurs : {stats.errors}")
    if stats.operations:
        print("Operations :")
        for operation in stats.operations:
            print(f"- {operation}")
    return 0 if stats.errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
