import csv
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


CSV_DIR = Path("output_claimshield/csv")
FHIR_DIR = Path("output_claimshield/fhir")
OUTPUT_DIR = Path("claimshield_cases/generated/CLM-0001")


def read_csv(filename: str) -> list[dict[str, str]]:
    """Lire un fichier CSV Synthea."""
    path = CSV_DIR / filename

    if not path.exists():
        print(f"Fichier absent : {path}")
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def parse_datetime(value: str | None) -> datetime:
    """Convertir une date Synthea pour permettre le tri."""
    if not value:
        return datetime.min

    value = value.replace("Z", "+00:00")

    try:
        return datetime.fromisoformat(value).replace(tzinfo=None)
    except ValueError:
        return datetime.min


def index_by(
    rows: list[dict[str, str]],
    column: str,
) -> dict[str, dict[str, str]]:
    """Indexer une liste de lignes par une colonne unique."""
    return {
        row[column]: row
        for row in rows
        if row.get(column)
    }


def group_by(
    rows: list[dict[str, str]],
    column: str,
) -> dict[str, list[dict[str, str]]]:
    """Regrouper plusieurs lignes par une même valeur."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        value = row.get(column)

        if value:
            grouped[value].append(row)

    return dict(grouped)


patients = read_csv("patients.csv")
claims = read_csv("claims.csv")
encounters = read_csv("encounters.csv")
medications = read_csv("medications.csv")
procedures = read_csv("procedures.csv")
payers = read_csv("payers.csv")
providers = read_csv("providers.csv")
organizations = read_csv("organizations.csv")
payer_transitions = read_csv("payer_transitions.csv")

patients_by_id = index_by(patients, "Id")
encounters_by_id = index_by(encounters, "Id")
payers_by_id = index_by(payers, "Id")
providers_by_id = index_by(providers, "Id")
organizations_by_id = index_by(organizations, "Id")

medications_by_encounter = group_by(medications, "ENCOUNTER")
procedures_by_encounter = group_by(procedures, "ENCOUNTER")
coverages_by_patient = group_by(payer_transitions, "PATIENT")

valid_candidates: list[dict[str, Any]] = []

for claim in claims:
    patient_id = claim.get("PATIENTID")
    encounter_id = claim.get("APPOINTMENTID")

    patient = patients_by_id.get(patient_id or "")
    encounter = encounters_by_id.get(encounter_id or "")

    if patient is None or encounter is None:
        continue

    # Ne garder que les patients vivants.
    if patient.get("DEATHDATE"):
        continue

    claim_medications = medications_by_encounter.get(encounter_id or "", [])
    claim_procedures = procedures_by_encounter.get(encounter_id or "", [])

    # Pour le premier dossier, on veut au moins un acte ou un médicament.
    if not claim_medications or not claim_procedures:
        continue

    valid_candidates.append(
        {
            "claim": claim,
            "patient": patient,
            "encounter": encounter,
            "medications": claim_medications,
            "procedures": claim_procedures,
        }
    )

if not valid_candidates:
    raise RuntimeError(
        "Aucune demande cohérente avec consultation, acte ou médicament."
    )

# Choisir la demande cohérente la plus récente.
valid_candidates.sort(
    key=lambda item: parse_datetime(item["claim"].get("SERVICEDATE")),
    reverse=True,
)

selected = valid_candidates[0]

claim = selected["claim"]
patient = selected["patient"]
encounter = selected["encounter"]

patient_id = patient["Id"]
payer_id = (
    claim.get("PRIMARYPATIENTINSURANCEID")
    or encounter.get("PAYER")
    or ""
)
provider_id = (
    claim.get("PROVIDERID")
    or encounter.get("PROVIDER")
    or ""
)
organization_id = encounter.get("ORGANIZATION") or ""

payer = payers_by_id.get(payer_id)
provider = providers_by_id.get(provider_id)
organization = organizations_by_id.get(organization_id)

case_data = {
    "case_id": "CLM-0001",
    "source": {
        "generator": "Synthea",
        "data_type": "fully_synthetic",
        "contains_real_patient_data": False,
    },
    "patient": patient,
    "claim": claim,
    "encounter": encounter,
    "payer": payer,
    "provider": provider,
    "organization": organization,
    "coverages": coverages_by_patient.get(patient_id, []),
    "medications": selected["medications"],
    "procedures": selected["procedures"],
    "ground_truth": {
        "expected_status": "valid",
        "expected_anomalies": [],
        "human_review_required": False,
    },
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

case_data_path = OUTPUT_DIR / "case_data.json"

with case_data_path.open("w", encoding="utf-8") as file:
    json.dump(
        case_data,
        file,
        ensure_ascii=False,
        indent=2,
    )

# Rechercher le bundle FHIR correspondant au patient.
matching_fhir_files = list(FHIR_DIR.glob(f"*{patient_id}.json"))

if matching_fhir_files:
    destination = OUTPUT_DIR / "patient_fhir_bundle.json"
    shutil.copy2(matching_fhir_files[0], destination)
    fhir_status = str(destination)
else:
    fhir_status = "Bundle FHIR correspondant introuvable"

print("=" * 70)
print("DOSSIER CLAIMSHIELD CRÉÉ")
print("=" * 70)

print("Identifiant du dossier : CLM-0001")
print(f"Demande Synthea        : {claim.get('Id')}")
print(
    "Patient                : "
    f"{patient.get('FIRST')} {patient.get('LAST')}"
)
print(f"Identifiant patient     : {patient_id}")
print(f"Date du soin            : {claim.get('SERVICEDATE')}")
print(f"Consultation            : {encounter.get('DESCRIPTION')}")
print(f"Nombre d'actes          : {len(selected['procedures'])}")
print(f"Nombre de médicaments   : {len(selected['medications'])}")
print(f"Données JSON            : {case_data_path}")
print(f"Bundle FHIR             : {fhir_status}")

