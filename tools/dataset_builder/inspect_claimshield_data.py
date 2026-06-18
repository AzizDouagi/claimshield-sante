import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

BASE_DIR = Path("output_claimshield/csv")


def read_csv(filename: str) -> list[dict[str, str]]:
    path = BASE_DIR / filename

    if not path.exists():
        print(f"Fichier absent : {path}")
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def parse_date(value: str):
    if not value:
        return None

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            return datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            return None


patients = read_csv("patients.csv")
claims = read_csv("claims.csv")
medications = read_csv("medications.csv")
procedures = read_csv("procedures.csv")
encounters = read_csv("encounters.csv")

alive_patients = {
    patient["Id"]: patient
    for patient in patients
    if not patient.get("DEATHDATE")
}

claims_for_alive_patients = [
    claim
    for claim in claims
    if claim.get("PATIENTID") in alive_patients
]

claims_by_patient = Counter(
    claim.get("PATIENTID")
    for claim in claims_for_alive_patients
)

medications_by_patient = Counter(
    medication.get("PATIENT")
    for medication in medications
)

usable_patients = [
    patient_id
    for patient_id in alive_patients
    if claims_by_patient[patient_id] > 0
]

print("=" * 60)
print("RÉSUMÉ DES DONNÉES CLAIMSHIELD")
print("=" * 60)

print(f"Patients totaux             : {len(patients)}")
print(f"Patients vivants            : {len(alive_patients)}")
print(f"Demandes totales            : {len(claims)}")
print(f"Demandes de patients vivants: {len(claims_for_alive_patients)}")
print(f"Consultations               : {len(encounters)}")
print(f"Actes médicaux              : {len(procedures)}")
print(f"Médicaments                 : {len(medications)}")
print(f"Patients exploitables       : {len(usable_patients)}")

print("\nPremiers patients exploitables :")

for patient_id in usable_patients[:10]:
    patient = alive_patients[patient_id]

    print(
        f"- {patient_id} | "
        f"{patient.get('FIRST')} {patient.get('LAST')} | "
        f"{claims_by_patient[patient_id]} demandes | "
        f"{medications_by_patient[patient_id]} médicaments"
    )
