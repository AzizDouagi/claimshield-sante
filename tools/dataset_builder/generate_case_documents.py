from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
)


CASE_ID = "CLM-0001"
CASE_DIR = Path("claimshield_cases/generated") / CASE_ID
CASE_DATA_PATH = CASE_DIR / "case_data.json"

INVOICE_NUMBER = f"INV-{CASE_ID}"
PRESCRIPTION_NUMBER = f"RX-{CASE_ID}"
AUTHORIZATION_NUMBER = f"AUTH-{CASE_ID}"

COVERAGE_RATE = Decimal("0.80")


def decimal_value(value: Any) -> Decimal:
    """Convertir une valeur en Decimal sans provoquer d'erreur."""
    if value in (None, ""):
        return Decimal("0.00")

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal("0.00")


def amount(value: Decimal) -> Decimal:
    """Arrondir un montant à deux décimales."""
    return value.quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )


def money(value: Decimal) -> str:
    """Afficher un montant synthétique en dollars."""
    return f"{amount(value):,.2f} USD"


def safe(value: Any, default: str = "Non renseigné") -> str:
    """Retourner une chaîne exploitable pour les documents."""
    if value is None:
        return default

    text = str(value).strip()
    return text if text else default


def format_date(value: str | None) -> str:
    """Convertir une date ISO en format lisible."""
    if not value:
        return "Non renseignée"

    try:
        parsed = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
        return parsed.strftime("%d/%m/%Y")
    except ValueError:
        return value


def full_name(person: dict[str, Any] | None) -> str:
    """Construire le nom complet d'une personne."""
    if not person:
        return "Non renseigné"

    parts = [
        person.get("PREFIX"),
        person.get("FIRST"),
        person.get("MIDDLE"),
        person.get("LAST"),
        person.get("SUFFIX"),
    ]

    return " ".join(
        str(part).strip()
        for part in parts
        if part and str(part).strip()
    )


def sha256_file(path: Path) -> str:
    """Calculer l'empreinte SHA-256 d'un fichier."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_case_data() -> dict[str, Any]:
    if not CASE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {CASE_DATA_PATH}"
        )

    return json.loads(
        CASE_DATA_PATH.read_text(encoding="utf-8")
    )


STYLES = getSampleStyleSheet()

STYLES.add(
    ParagraphStyle(
        name="DocumentTitle",
        parent=STYLES["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=12,
    )
)

STYLES.add(
    ParagraphStyle(
        name="SectionTitle",
        parent=STYLES["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        spaceBefore=10,
        spaceAfter=6,
    )
)

STYLES.add(
    ParagraphStyle(
        name="SmallText",
        parent=STYLES["BodyText"],
        fontSize=8,
        leading=10,
    )
)

STYLES.add(
    ParagraphStyle(
        name="RightText",
        parent=STYLES["BodyText"],
        alignment=TA_RIGHT,
    )
)

STYLES.add(
    ParagraphStyle(
        name="Notice",
        parent=STYLES["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#8B0000"),
        spaceAfter=8,
    )
)


def paragraph(value: Any, style: str = "BodyText") -> Paragraph:
    text = safe(value)
    text = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return Paragraph(text, STYLES[style])


def footer(canvas, document) -> None:
    """Ajouter le pied de page à chaque document."""
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.grey)

    canvas.drawString(
        18 * mm,
        10 * mm,
        "DONNEES ENTIEREMENT SYNTHETIQUES - USAGE DE TEST UNIQUEMENT",
    )

    canvas.drawRightString(
        192 * mm,
        10 * mm,
        f"Page {document.page}",
    )

    canvas.restoreState()


def create_document(path: Path, story: list[Any]) -> None:
    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title=path.stem,
        author="ClaimShield Santé",
    )

    document.build(
        story,
        onFirstPage=footer,
        onLaterPages=footer,
    )


def common_header(
    title: str,
    reference: str,
) -> list[Any]:
    return [
        Paragraph(title, STYLES["DocumentTitle"]),
        Paragraph(
            "DOCUMENT SYNTHETIQUE - AUCUNE PERSONNE REELLE",
            STYLES["Notice"],
        ),
        Table(
            [
                [
                    paragraph("Référence", "SmallText"),
                    paragraph(reference, "SmallText"),
                    paragraph("Dossier", "SmallText"),
                    paragraph(CASE_ID, "SmallText"),
                ]
            ],
            colWidths=[
                28 * mm,
                57 * mm,
                25 * mm,
                57 * mm,
            ],
            style=[
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F3F5")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ],
        ),
        Spacer(1, 8),
    ]


def information_table(rows: list[list[Any]]) -> Table:
    formatted_rows = []

    for label, value in rows:
        formatted_rows.append(
            [
                paragraph(label, "SmallText"),
                paragraph(value),
            ]
        )

    return Table(
        formatted_rows,
        colWidths=[48 * mm, 119 * mm],
        repeatRows=0,
        style=[
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E9ECEF")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ],
    )


def calculate_amounts(
    data: dict[str, Any],
) -> dict[str, Decimal]:
    encounter = data.get("encounter") or {}
    procedures = data.get("procedures") or []
    medications = data.get("medications") or []

    consultation_cost = decimal_value(
        encounter.get("BASE_ENCOUNTER_COST")
    )

    procedure_total = sum(
        (
            decimal_value(item.get("BASE_COST"))
            for item in procedures
        ),
        Decimal("0.00"),
    )

    medication_total = sum(
        (
            decimal_value(item.get("BASE_COST"))
            for item in medications
        ),
        Decimal("0.00"),
    )

    total_billed = amount(
        consultation_cost
        + procedure_total
        + medication_total
    )

    amount_covered = amount(
        total_billed * COVERAGE_RATE
    )

    patient_share = amount(
        total_billed - amount_covered
    )

    return {
        "consultation_cost": amount(consultation_cost),
        "procedure_total": amount(procedure_total),
        "medication_total": amount(medication_total),
        "total_billed": total_billed,
        "amount_covered": amount_covered,
        "amount_requested": amount_covered,
        "patient_share": patient_share,
    }


def create_invoice(
    data: dict[str, Any],
    totals: dict[str, Decimal],
) -> Path:
    patient = data["patient"]
    provider = data.get("provider") or {}
    organization = data.get("organization") or {}
    encounter = data.get("encounter") or {}
    procedures = data.get("procedures") or []
    medications = data.get("medications") or []

    path = CASE_DIR / "medical_invoice.pdf"

    story = common_header(
        "FACTURE MEDICALE SYNTHETIQUE",
        INVOICE_NUMBER,
    )

    story.append(
        information_table(
            [
                ["Patient", full_name(patient)],
                ["Identifiant patient", patient.get("Id")],
                ["Date de naissance", format_date(patient.get("BIRTHDATE"))],
                ["Date des soins", format_date(encounter.get("START"))],
                ["Professionnel", safe(provider.get("NAME"))],
                ["Spécialité", safe(provider.get("SPECIALITY"))],
                ["Établissement", safe(organization.get("NAME"))],
                [
                    "Adresse établissement",
                    " ".join(
                        part
                        for part in [
                            safe(
                                organization.get("ADDRESS"),
                                "",
                            ),
                            safe(
                                organization.get("CITY"),
                                "",
                            ),
                            safe(
                                organization.get("STATE"),
                                "",
                            ),
                            safe(
                                organization.get("ZIP"),
                                "",
                            ),
                        ]
                        if part
                    ),
                ],
            ]
        )
    )

    story.append(
        Paragraph("Détail des prestations", STYLES["SectionTitle"])
    )

    lines: list[list[Any]] = [
        [
            paragraph("N°", "SmallText"),
            paragraph("Description", "SmallText"),
            paragraph("Code", "SmallText"),
            paragraph("Montant", "SmallText"),
        ]
    ]

    line_number = 1

    if totals["consultation_cost"] > 0:
        lines.append(
            [
                paragraph(str(line_number)),
                paragraph(
                    encounter.get(
                        "DESCRIPTION",
                        "Consultation médicale",
                    )
                ),
                paragraph(encounter.get("CODE")),
                paragraph(money(totals["consultation_cost"])),
            ]
        )
        line_number += 1

    for procedure in procedures:
        lines.append(
            [
                paragraph(str(line_number)),
                paragraph(procedure.get("DESCRIPTION")),
                paragraph(procedure.get("CODE")),
                paragraph(
                    money(decimal_value(procedure.get("BASE_COST")))
                ),
            ]
        )
        line_number += 1

    for medication in medications:
        lines.append(
            [
                paragraph(str(line_number)),
                paragraph(
                    f"Médicament : {safe(medication.get('DESCRIPTION'))}"
                ),
                paragraph(medication.get("CODE")),
                paragraph(
                    money(decimal_value(medication.get("BASE_COST")))
                ),
            ]
        )
        line_number += 1

    service_table = Table(
        lines,
        colWidths=[
            12 * mm,
            100 * mm,
            25 * mm,
            30 * mm,
        ],
        repeatRows=1,
        style=[
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (-1, 1), (-1, -1), "RIGHT"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ],
    )

    story.append(service_table)
    story.append(Spacer(1, 10))

    totals_table = Table(
        [
            [
                paragraph("Total facturé"),
                paragraph(
                    money(totals["total_billed"]),
                    "RightText",
                ),
            ],
            [
                paragraph("Prise en charge synthétique (80 %)"),
                paragraph(
                    money(totals["amount_covered"]),
                    "RightText",
                ),
            ],
            [
                paragraph("Reste à charge"),
                paragraph(
                    money(totals["patient_share"]),
                    "RightText",
                ),
            ],
        ],
        colWidths=[117 * mm, 50 * mm],
        style=[
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8F9FA")),
            ("BOX", (0, 0), (-1, -1), 0.7, colors.grey),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ],
    )

    story.append(totals_table)
    story.append(Spacer(1, 12))

    story.append(
        paragraph(
            "Cette facture a été générée à partir de données entièrement "
            "synthétiques. Les montants ont été normalisés pour les tests "
            "déterministes de ClaimShield Santé.",
            "SmallText",
        )
    )

    create_document(path, story)
    return path


def create_prescription(
    data: dict[str, Any],
) -> Path:
    patient = data["patient"]
    provider = data.get("provider") or {}
    organization = data.get("organization") or {}
    encounter = data.get("encounter") or {}
    medications = data.get("medications") or []

    path = CASE_DIR / "prescription.pdf"

    story = common_header(
        "ORDONNANCE SYNTHETIQUE",
        PRESCRIPTION_NUMBER,
    )

    story.append(
        information_table(
            [
                ["Patient", full_name(patient)],
                ["Date de naissance", format_date(patient.get("BIRTHDATE"))],
                ["Date de prescription", format_date(encounter.get("START"))],
                ["Prescripteur", safe(provider.get("NAME"))],
                ["Spécialité", safe(provider.get("SPECIALITY"))],
                ["Établissement", safe(organization.get("NAME"))],
            ]
        )
    )

    story.append(
        Paragraph("Médicaments prescrits", STYLES["SectionTitle"])
    )

    medication_rows: list[list[Any]] = [
        [
            paragraph("N°", "SmallText"),
            paragraph("Médicament", "SmallText"),
            paragraph("Code", "SmallText"),
            paragraph("Instructions", "SmallText"),
        ]
    ]

    for index, medication in enumerate(medications, start=1):
        stop_date = format_date(medication.get("STOP"))

        instructions = (
            "Utilisation conformément aux instructions du professionnel. "
            f"Date de fin : {stop_date}."
        )

        medication_rows.append(
            [
                paragraph(str(index)),
                paragraph(medication.get("DESCRIPTION")),
                paragraph(medication.get("CODE")),
                paragraph(instructions),
            ]
        )

    medication_table = Table(
        medication_rows,
        colWidths=[
            12 * mm,
            72 * mm,
            25 * mm,
            58 * mm,
        ],
        repeatRows=1,
        style=[
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2F0D9")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ],
    )

    story.append(medication_table)
    story.append(Spacer(1, 22))

    signature_table = Table(
        [
            [
                paragraph("Signature synthétique du prescripteur"),
                paragraph(
                    f"Dr {safe(provider.get('NAME'))}",
                    "RightText",
                ),
            ]
        ],
        colWidths=[85 * mm, 82 * mm],
        style=[
            ("LINEABOVE", (1, 0), (1, 0), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
        ],
    )

    story.append(signature_table)

    create_document(path, story)
    return path


def create_claim_request(
    data: dict[str, Any],
    totals: dict[str, Decimal],
) -> Path:
    patient = data["patient"]
    claim = data["claim"]
    payer = data.get("payer") or {}
    provider = data.get("provider") or {}
    organization = data.get("organization") or {}
    procedures = data.get("procedures") or []
    medications = data.get("medications") or []

    patient_id = safe(patient.get("Id"))
    policy_number = f"POL-{patient_id[:8].upper()}"

    authorization_required = (
        totals["total_billed"] >= Decimal("3000.00")
        or len(procedures) > 5
    )

    authorization_status = (
        "APPROUVEE - DONNEE SYNTHETIQUE"
        if authorization_required
        else "NON REQUISE"
    )

    path = CASE_DIR / "claim_request.pdf"

    story = common_header(
        "DEMANDE DE REMBOURSEMENT SYNTHETIQUE",
        CASE_ID,
    )

    story.append(
        Paragraph("Demandeur", STYLES["SectionTitle"])
    )

    story.append(
        information_table(
            [
                ["Patient", full_name(patient)],
                ["Identifiant patient", patient_id],
                ["Date de naissance", format_date(patient.get("BIRTHDATE"))],
                [
                    "Adresse",
                    " ".join(
                        part
                        for part in [
                            safe(patient.get("ADDRESS"), ""),
                            safe(patient.get("CITY"), ""),
                            safe(patient.get("STATE"), ""),
                            safe(patient.get("ZIP"), ""),
                        ]
                        if part
                    ),
                ],
                ["Assureur", safe(payer.get("NAME"))],
                ["Numéro de couverture", policy_number],
            ]
        )
    )

    story.append(
        Paragraph("Informations sur les soins", STYLES["SectionTitle"])
    )

    story.append(
        information_table(
            [
                ["Référence ClaimShield", CASE_ID],
                ["Référence Synthea", safe(claim.get("Id"))],
                ["Date des soins", format_date(claim.get("SERVICEDATE"))],
                ["Professionnel", safe(provider.get("NAME"))],
                ["Établissement", safe(organization.get("NAME"))],
                ["Nombre d'actes", str(len(procedures))],
                ["Nombre de médicaments", str(len(medications))],
                ["Facture associée", INVOICE_NUMBER],
                ["Ordonnance associée", PRESCRIPTION_NUMBER],
            ]
        )
    )

    story.append(
        Paragraph("Montants demandés", STYLES["SectionTitle"])
    )

    story.append(
        information_table(
            [
                ["Montant total facturé", money(totals["total_billed"])],
                [
                    "Taux de couverture synthétique",
                    f"{int(COVERAGE_RATE * 100)} %",
                ],
                [
                    "Montant demandé à l'assureur",
                    money(totals["amount_requested"]),
                ],
                ["Reste à charge", money(totals["patient_share"])],
            ]
        )
    )

    story.append(
        Paragraph("Autorisation préalable", STYLES["SectionTitle"])
    )

    story.append(
        information_table(
            [
                [
                    "Autorisation requise",
                    "Oui" if authorization_required else "Non",
                ],
                ["Statut", authorization_status],
                [
                    "Numéro d'autorisation",
                    (
                        AUTHORIZATION_NUMBER
                        if authorization_required
                        else "Non applicable"
                    ),
                ],
                [
                    "Règle appliquée",
                    (
                        "Montant supérieur ou égal à 3 000 USD, "
                        "ou plus de cinq actes."
                    ),
                ],
            ]
        )
    )

    story.append(Spacer(1, 12))

    story.append(
        paragraph(
            "Je certifie que les informations de ce formulaire sont "
            "entièrement synthétiques et destinées exclusivement à "
            "l'évaluation fonctionnelle et sécuritaire de ClaimShield Santé.",
            "SmallText",
        )
    )

    create_document(path, story)
    return path


def create_ground_truth(
    data: dict[str, Any],
    totals: dict[str, Decimal],
) -> Path:
    patient = data["patient"]
    payer = data.get("payer") or {}
    procedures = data.get("procedures") or []
    medications = data.get("medications") or []

    authorization_required = (
        totals["total_billed"] >= Decimal("3000.00")
        or len(procedures) > 5
    )

    ground_truth = {
        "case_id": CASE_ID,
        "data_classification": "SYNTHETIC_TEST_DATA",
        "contains_real_personal_data": False,
        "expected_recommendation": "APPROVE",
        "human_review_required": True,
        "expected_anomalies": [],
        "required_documents": {
            "claim_request.pdf": True,
            "medical_invoice.pdf": True,
            "prescription.pdf": len(medications) > 0,
            "patient_fhir_bundle.json": False,
        },
        "expected_extraction": {
            "patient_name": full_name(patient),
            "patient_id": patient.get("Id"),
            "payer_name": payer.get("NAME"),
            "service_date": format_date(
                data["claim"].get("SERVICEDATE")
            ),
            "claim_reference": CASE_ID,
            "invoice_number": INVOICE_NUMBER,
            "prescription_number": PRESCRIPTION_NUMBER,
            "procedure_count": len(procedures),
            "medication_count": len(medications),
            "total_billed": f"{totals['total_billed']:.2f}",
            "amount_requested": f"{totals['amount_requested']:.2f}",
            "patient_share": f"{totals['patient_share']:.2f}",
            "currency": "USD",
        },
        "deterministic_rules": {
            "coverage_rate": "0.80",
            "authorization_required": authorization_required,
            "authorization_status": (
                "approved" if authorization_required else "not_required"
            ),
            "duplicate_invoice": False,
            "prompt_injection_detected": False,
        },
        "expected_explanation": [
            "Les documents obligatoires sont présents.",
            "L'identité du patient est cohérente entre les documents.",
            "Les montants sont identiques dans la facture et la demande.",
            "L'ordonnance correspond au médicament facturé.",
            "L'autorisation préalable synthétique est valide.",
            "Aucun doublon ni prompt injection n'est attendu.",
        ],
    }

    path = CASE_DIR / "ground_truth.json"
    path.write_text(
        json.dumps(
            ground_truth,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


def create_manifest(
    data: dict[str, Any],
    totals: dict[str, Decimal],
) -> Path:
    files = []

    for path in sorted(CASE_DIR.iterdir()):
        if not path.is_file():
            continue

        if path.name == "manifest.json":
            continue

        files.append(
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "required": path.name
                in {
                    "claim_request.pdf",
                    "medical_invoice.pdf",
                    "prescription.pdf",
                    "case_data.json",
                },
            }
        )

    manifest = {
        "case_id": CASE_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "source_generator": "Synthea",
            "source_software_license": "Apache-2.0",
            "contains_real_personal_data": False,
            "purpose": "ClaimShield Santé MVP testing",
        },
        "patient_id": data["patient"].get("Id"),
        "source_claim_id": data["claim"].get("Id"),
        "financial_summary": {
            "currency": "USD",
            "total_billed": f"{totals['total_billed']:.2f}",
            "amount_requested": f"{totals['amount_requested']:.2f}",
            "patient_share": f"{totals['patient_share']:.2f}",
        },
        "files": files,
    }

    path = CASE_DIR / "manifest.json"
    path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


def main() -> None:
    CASE_DIR.mkdir(parents=True, exist_ok=True)

    data = load_case_data()
    totals = calculate_amounts(data)

    created_files = [
        create_invoice(data, totals),
        create_prescription(data),
        create_claim_request(data, totals),
        create_ground_truth(data, totals),
    ]

    manifest_path = create_manifest(data, totals)
    created_files.append(manifest_path)

    print("=" * 72)
    print("DOCUMENTS CLAIMSHIELD GENERES")
    print("=" * 72)

    print(f"Dossier             : {CASE_ID}")
    print(f"Total facturé       : {money(totals['total_billed'])}")
    print(f"Montant demandé     : {money(totals['amount_requested'])}")
    print(f"Reste à charge      : {money(totals['patient_share'])}")

    print("\nFichiers créés :")

    for path in created_files:
        print(f"- {path}")


if __name__ == "__main__":
    main()
