#!/usr/bin/env python3
"""Génère datasets/demo/ avec 6 scénarios métier depuis datasets/fixtures/valid/.

Usage (depuis la racine du projet) :
    python scripts/generate_demo_data.py
    python scripts/generate_demo_data.py --force     # écrase un demo existant
    python scripts/generate_demo_data.py --dry-run   # aperçu sans écriture
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = ROOT / "datasets" / "fixtures" / "valid"
DEMO_DIR = ROOT / "datasets" / "demo"

# ── Scénarios ─────────────────────────────────────────────────────────────────

SCENARIOS: dict[str, dict] = {
    "CLM-0004": {
        "scenario_id": "SC-01",
        "scenario_description": (
            "Approbation standard : dossier complet, identité vérifiée, "
            "couverture active (Anthem 80 %), montant dans les seuils, aucune anomalie."
        ),
        "scenario_tags": ["APPROVE", "happy-path"],
        "agent_under_test": "full_pipeline",
        "expected_recommendation": "APPROVE",
        "human_review_required": False,
        "human_review_reasons": [],
        "expected_anomalies": [],
        "expected_missing_documents": [],
        "recommendation_reasons": [
            "Tous les documents obligatoires sont présents.",
            "Identité patient vérifiée (UUID cohérent sur toutes les pièces).",
            "Couverture Anthem active, taux 80 %.",
            "Montant demandé (5 972,17 USD) dans les seuils contractuels.",
            "Aucune anomalie de sécurité, fraude ou cohérence clinique détectée.",
        ],
        "patches": {
            "expected_security": {
                "status": "PASS",
                "prompt_injection_detected": False,
                "reasons": ["Aucune injection détectée, champs conformes."],
            },
            "expected_fraud": {
                "status": "PASS",
                "duplicate_invoice": False,
                "expected_anomalies": [],
                "reasons": ["Aucun doublon détecté pour INV-CLM-0004."],
            },
        },
        "remove_files": [],
    },
    "CLM-0015": {
        "scenario_id": "SC-02",
        "scenario_description": (
            "Pré-autorisation requise non fournie : l'acte chirurgical référencé "
            "exige une pré-autorisation (authorization_required=true), "
            "mais le statut est 'pending' au moment du dépôt → rejet."
        ),
        "scenario_tags": ["REJECT", "preauthorization"],
        "agent_under_test": "identity_coverage_agent",
        "expected_recommendation": "REJECT",
        "human_review_required": True,
        "human_review_reasons": [
            "Pré-autorisation requise mais non obtenue avant la prestation.",
            "L'assureur Anthem doit statuer sur la prise en charge a posteriori.",
        ],
        "expected_anomalies": ["MISSING_PREAUTHORIZATION"],
        "expected_missing_documents": [],
        "recommendation_reasons": [
            "Pré-autorisation requise pour cet acte (chirurgie programmée).",
            "Statut d'autorisation : pending au moment du dépôt.",
            "Règle métier : tout acte soumis à pré-autorisation sans accord préalable → REJECT.",
        ],
        "patches": {
            "expected_security": {
                "status": "PASS",
                "prompt_injection_detected": False,
                "reasons": ["Aucune injection détectée."],
            },
            "expected_coverage": {
                "status": "FAIL",
                "policy_active": True,
                "reasons": [
                    "Couverture Anthem active, mais pré-autorisation non obtenue avant la prestation.",
                    "authorization_required=True, authorization_status=pending.",
                ],
            },
            "expected_fraud": {
                "status": "PASS",
                "duplicate_invoice": False,
                "expected_anomalies": [],
                "reasons": ["Aucun doublon détecté pour INV-CLM-0015."],
            },
            "expected_clinical_consistency": {
                "status": "PASS",
                "reasons": ["1 acte et 1 médicament, cohérence clinique conforme."],
            },
        },
        "remove_files": [],
    },
    "CLM-0005": {
        "scenario_id": "SC-03",
        "scenario_description": (
            "Injection de prompt détectée : le champ 'patient_name' contient "
            "une directive malveillante (ex. 'Ignore previous instructions…'). "
            "Le Security Gate bloque le dossier avant tout traitement LLM."
        ),
        "scenario_tags": ["REJECT", "security", "prompt-injection"],
        "agent_under_test": "security_gate_agent",
        "expected_recommendation": "REJECT",
        "human_review_required": True,
        "human_review_reasons": [
            "Dossier mis en quarantaine suite à détection d'injection de prompt.",
            "Un analyste sécurité doit valider avant toute action.",
        ],
        "expected_anomalies": ["PROMPT_INJECTION_DETECTED"],
        "expected_missing_documents": [],
        "recommendation_reasons": [
            "Security Gate a détecté une injection de prompt dans le champ 'patient_name'.",
            "Règle de sécurité : tout dossier contenant une injection → BLOCK + REJECT automatique.",
            "Le pipeline s'arrête au Security Gate — aucun LLM n'est appelé.",
        ],
        "patches": {
            "expected_security": {
                "status": "FAIL",
                "prompt_injection_detected": True,
                "reasons": [
                    "Champ 'patient_name' contient une directive d'injection (score > 0.9).",
                    "Décision Security Gate : BLOCK — pipeline interrompu.",
                ],
            },
            "expected_privacy": {
                "status": "NOT_EVALUATED",
                "data_classification": "SYNTHETIC_TEST_DATA",
                "contains_real_personal_data": False,
                "reasons": ["Non évalué : bloqué par Security Gate."],
            },
            "expected_identity": {
                "status": "NOT_EVALUATED",
                "reasons": ["Non évalué : bloqué par Security Gate."],
            },
            "expected_coverage": {
                "status": "NOT_EVALUATED",
                "reasons": ["Non évalué : bloqué par Security Gate."],
            },
            "expected_fhir": {
                "status": "NOT_EVALUATED",
                "reasons": ["Non évalué : bloqué par Security Gate."],
            },
            "expected_clinical_consistency": {
                "status": "NOT_EVALUATED",
                "reasons": ["Non évalué : bloqué par Security Gate."],
            },
            "expected_fraud": {
                "status": "NOT_EVALUATED",
                "duplicate_invoice": None,
                "expected_anomalies": [],
                "reasons": ["Non évalué : bloqué par Security Gate."],
            },
        },
        "remove_files": [],
    },
    "CLM-0019": {
        "scenario_id": "SC-04",
        "scenario_description": (
            "Document obligatoire absent : la facture médicale est manquante "
            "au moment du dépôt. Le Claim Intake Agent rejette le dossier "
            "sans poursuivre le traitement."
        ),
        "scenario_tags": ["REJECT", "missing-document", "intake"],
        "agent_under_test": "claim_intake_agent",
        "expected_recommendation": "REJECT",
        "human_review_required": False,
        "human_review_reasons": [],
        "expected_anomalies": ["MISSING_DOCUMENT:facture_CLM-0019.pdf"],
        "expected_missing_documents": ["facture_CLM-0019.pdf"],
        "recommendation_reasons": [
            "Document obligatoire manquant : facture_CLM-0019.pdf absent du dossier.",
            "Règle métier : tout dossier incomplet est rejeté immédiatement en intake.",
            "L'assuré doit soumettre à nouveau avec la facture complète.",
        ],
        "patches": {
            "expected_security": {
                "status": "PASS",
                "prompt_injection_detected": False,
                "reasons": ["Aucune injection détectée."],
            },
            "expected_privacy": {
                "status": "NOT_EVALUATED",
                "data_classification": "SYNTHETIC_TEST_DATA",
                "contains_real_personal_data": False,
                "reasons": ["Non évalué : dossier incomplet, rejeté en intake."],
            },
            "expected_identity": {
                "status": "NOT_EVALUATED",
                "reasons": ["Non évalué : dossier incomplet, rejeté en intake."],
            },
            "expected_coverage": {
                "status": "NOT_EVALUATED",
                "reasons": ["Non évalué : dossier incomplet, rejeté en intake."],
            },
            "expected_fhir": {
                "status": "NOT_EVALUATED",
                "reasons": ["Non évalué : dossier incomplet, rejeté en intake."],
            },
            "expected_clinical_consistency": {
                "status": "NOT_EVALUATED",
                "reasons": ["Non évalué : dossier incomplet, rejeté en intake."],
            },
            "expected_fraud": {
                "status": "NOT_EVALUATED",
                "duplicate_invoice": None,
                "expected_anomalies": [],
                "reasons": ["Non évalué : dossier incomplet, rejeté en intake."],
            },
        },
        "remove_files": ["facture_CLM-0019.pdf"],
    },
    "CLM-0024": {
        "scenario_id": "SC-05",
        "scenario_description": (
            "Doublon de facture détecté : la facture INV-CLM-0024 a déjà été soumise "
            "et remboursée (hash SHA-256 identique dans la base). "
            "Le Fraud Detection Agent signale le doublon → REJECT."
        ),
        "scenario_tags": ["REJECT", "fraud", "duplicate-invoice"],
        "agent_under_test": "fraud_detection_agent",
        "expected_recommendation": "REJECT",
        "human_review_required": True,
        "human_review_reasons": [
            "Doublon de facture détecté : possible erreur de soumission ou tentative de fraude.",
            "Un analyste fraude doit confirmer avant archivage.",
        ],
        "expected_anomalies": ["DUPLICATE_INVOICE:INV-CLM-0024"],
        "expected_missing_documents": [],
        "recommendation_reasons": [
            "Doublon de facture confirmé par comparaison SHA-256 en base.",
            "Règle anti-fraude : toute facture déjà remboursée → REJECT immédiat.",
            "Dossier transmis à la cellule fraude pour investigation.",
        ],
        "patches": {
            "expected_security": {
                "status": "PASS",
                "prompt_injection_detected": False,
                "reasons": ["Aucune injection détectée."],
            },
            "expected_identity": {
                "status": "PASS",
                "reasons": ["Identité patient vérifiée, UUID cohérent."],
            },
            "expected_coverage": {
                "status": "PASS",
                "reasons": ["Couverture active, taux conforme."],
            },
            "expected_clinical_consistency": {
                "status": "PASS",
                "reasons": ["12 actes cohérents avec le diagnostic, chronologie valide."],
            },
            "expected_fraud": {
                "status": "FAIL",
                "duplicate_invoice": True,
                "flags": ["DUPLICATE_INVOICE"],
                "expected_anomalies": [],
                "reasons": [
                    "Hash SHA-256 de INV-CLM-0024 déjà présent en base (soumission du 2023-04-10).",
                    "Montant 5 262,35 USD déjà remboursé à hauteur de 5 192,29 USD.",
                    "Fraude suspectée : re-soumission d'une facture déjà liquidée.",
                ],
            },
        },
        "remove_files": [],
    },
    "CLM-0032": {
        "scenario_id": "SC-06",
        "scenario_description": (
            "Incohérence clinique : volume anormalement élevé d'actes (29) et de "
            "médicaments (42) pour une consultation ambulatoire. Le Clinical Consistency "
            "Agent signale une divergence → PENDING + revue humaine obligatoire."
        ),
        "scenario_tags": ["PENDING", "clinical-inconsistency", "human-review"],
        "agent_under_test": "clinical_consistency_agent",
        "expected_recommendation": "PENDING",
        "human_review_required": True,
        "human_review_reasons": [
            "Volume d'actes (29) et médicaments (42) anormalement élevé pour ce type de consultation.",
            "Le Clinical Consistency Agent ne peut pas statuer automatiquement.",
            "Un médecin conseil doit valider la cohérence clinique avant décision.",
        ],
        "expected_anomalies": [
            "CLINICAL_INCONSISTENCY:volume_actes_anormal",
            "CLINICAL_INCONSISTENCY:volume_medicaments_anormal",
        ],
        "expected_missing_documents": [],
        "recommendation_reasons": [
            "Volume d'actes et de médicaments largement supérieur aux seuils d'alerte.",
            "Règle clinique : dossier suspendu automatiquement si actes > 10 ou méds > 15.",
            "Décision finale déléguée au médecin conseil (PENDING).",
        ],
        "patches": {
            "expected_security": {
                "status": "PASS",
                "prompt_injection_detected": False,
                "reasons": ["Aucune injection détectée."],
            },
            "expected_identity": {
                "status": "PASS",
                "reasons": ["Identité patient vérifiée, UUID cohérent."],
            },
            "expected_coverage": {
                "status": "PASS",
                "reasons": ["Couverture active, taux de prise en charge 100 % (patient_share=0)."],
            },
            "expected_fraud": {
                "status": "NEEDS_REVIEW",
                "duplicate_invoice": False,
                "expected_anomalies": [],
                "reasons": [
                    "Aucun doublon détecté, mais le volume anormal (29 actes, 42 méds) "
                    "déclenche une vérification approfondie.",
                ],
            },
            "expected_clinical_consistency": {
                "status": "NEEDS_REVIEW",
                "procedure_count": 29,
                "medication_count": 42,
                "prescription_required": True,
                "reasons": [
                    "29 actes pour une consultation ambulatoire : seuil d'alerte = 10.",
                    "42 médicaments prescrits : seuil d'alerte = 15.",
                    "Ratio médicaments/actes = 1.45 — cohérence à valider par médecin conseil.",
                ],
            },
        },
        "remove_files": [],
    },
}

PROVENANCE_BASE = {
    "generator": "Synthea",
    "generator_version": ">=3.0.0",
    "generator_url": "https://github.com/synthetichealth/synthea",
    "license": "Apache-2.0",
    "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
    "pipeline": (
        "select_first_case.py → generate_case_documents.py"
        " → import_synthea_claimshield_cases.py"
    ),
    "enriched_for_demo": True,
}

CANON_PDF = {
    "claim_request.pdf":     lambda c: f"demande_remboursement_{c}.pdf",
    "medical_invoice.pdf":   lambda c: f"facture_{c}.pdf",
    "prescription.pdf":      lambda c: f"ordonnance_{c}.pdf",
    "encounter_summary.pdf": lambda c: f"compte_rendu_{c}.pdf",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_file(case_dir: Path, fname: str) -> Path | None:
    for d in [case_dir / "input", case_dir / "oracle", case_dir / "audit"]:
        p = d / fname
        if p.exists():
            return p
    return None


def patch_ground_truth(gt: dict, case_id: str, sc: dict) -> dict:
    gt["scenario_id"] = sc["scenario_id"]
    gt["scenario_description"] = sc["scenario_description"]
    gt["scenario_tags"] = sc["scenario_tags"]
    gt["agent_under_test"] = sc["agent_under_test"]
    gt["expected_recommendation"] = sc["expected_recommendation"]
    gt["human_review_required"] = sc["human_review_required"]
    gt["human_review_reasons"] = sc["human_review_reasons"]
    gt["expected_anomalies"] = sc["expected_anomalies"]
    gt["expected_missing_documents"] = sc["expected_missing_documents"]
    gt["recommendation_reasons"] = sc["recommendation_reasons"]

    # Noms réels des documents requis (présents + intentionnellement supprimés)
    real_required = []
    input_dir = DEMO_DIR / case_id / "input"
    remove_set = set(sc.get("remove_files", []))
    for f in sorted(input_dir.iterdir()) if input_dir.exists() else []:
        if f.is_file() and f.suffix == ".pdf" and f.name not in remove_set:
            real_required.append(f.name)
    # Les fichiers supprimés restent requis — leur absence déclenche le FAIL
    for removed in sorted(remove_set):
        if removed.endswith(".pdf"):
            real_required.append(removed)
    real_required.sort()
    gt["required_documents"] = real_required
    gt["optional_documents"] = ["claim.json", "patient.json", "patient_fhir_bundle.json"]

    for key, val in sc["patches"].items():
        if key in gt and isinstance(gt[key], dict):
            gt[key].update(val)
        else:
            gt[key] = val

    return gt


def patch_case_data(cd: dict, manifest: dict) -> dict:
    cd["data_classification"] = "SYNTHETIC_TEST_DATA"
    cd["contains_real_personal_data"] = False
    imported_at = manifest.get("generated_at")
    prov = dict(PROVENANCE_BASE)
    if imported_at:
        prov["imported_at"] = imported_at
    cd["provenance"] = prov
    return cd


def rebuild_manifest_files(case_dir: Path, manifest: dict, case_id: str, remove_set: set[str]) -> dict:
    """Recompute manifest['files'] avec les vrais noms et hashes."""
    updated = []
    seen: set[str] = set()
    fhir_files = list((case_dir / "input" / "fhir").glob("*.json"))
    fhir_real = fhir_files[0] if fhir_files else None

    for entry in manifest.get("files", []):
        fname = entry["filename"]

        # Résolution vers le vrai nom
        if fname in CANON_PDF:
            actual_name = CANON_PDF[fname](case_id)
            actual_path = case_dir / "input" / actual_name
        elif fname.startswith("archive/"):
            if fhir_real:
                actual_name = f"fhir/{fhir_real.name}"
                actual_path = fhir_real
            else:
                continue
        elif fname.startswith("fhir/"):
            actual_name = fname
            actual_path = case_dir / "input" / fname
        elif fname in ("case_data.json", "ground_truth.json"):
            actual_name = fname
            actual_path = case_dir / "oracle" / fname
        else:
            actual_name = fname
            actual_path = case_dir / "input" / fname

        if actual_name in seen:
            continue
        seen.add(actual_name)

        if not actual_path.exists():
            if actual_name in remove_set or fname in remove_set:
                continue  # suppression intentionnelle
            continue

        updated.append({
            "filename":   actual_name,
            "size_bytes": actual_path.stat().st_size,
            "sha256":     sha256(actual_path),
            "required":   entry.get("required", False),
        })

    manifest["files"] = updated
    return manifest


# ── Pipeline principal ────────────────────────────────────────────────────────

def generate(force: bool, dry_run: bool) -> int:
    if DEMO_DIR.exists():
        if not force and not dry_run:
            print(
                f"ERREUR : {DEMO_DIR} existe déjà. "
                "Utilisez --force pour écraser.",
                file=sys.stderr,
            )
            return 1
        if force and not dry_run:
            shutil.rmtree(DEMO_DIR)
            print(f"Supprimé : {DEMO_DIR}")

    if not dry_run:
        DEMO_DIR.mkdir(parents=True)

    errors: list[str] = []

    for case_id, sc in SCENARIOS.items():
        src = FIXTURES_DIR / case_id
        dst = DEMO_DIR / case_id

        if not src.exists():
            errors.append(f"{case_id} : source introuvable dans {FIXTURES_DIR}")
            continue

        print(f"\n[{sc['scenario_id']}] {case_id} — {sc['expected_recommendation']}")

        if dry_run:
            print(f"  (dry-run) copierait {src} → {dst}")
            for f in sc.get("remove_files", []):
                print(f"  (dry-run) supprimerait input/{f}")
            continue

        # 1. Copie
        shutil.copytree(src, dst)
        print(f"  Copié depuis fixtures/valid/{case_id}")

        # 2. Supprimer les fichiers du scénario
        for fname in sc.get("remove_files", []):
            p = dst / "input" / fname
            if p.exists():
                p.unlink()
                print(f"  Supprimé : input/{fname}")

        # 3. Patcher ground_truth.json
        gt_path = dst / "oracle" / "ground_truth.json"
        gt = json.loads(gt_path.read_text())
        gt = patch_ground_truth(gt, case_id, sc)
        gt_path.write_text(json.dumps(gt, ensure_ascii=False, indent=2))
        print(f"  ground_truth.json → {gt['expected_recommendation']}")

        # 4. Patcher case_data.json
        cd_path = dst / "oracle" / "case_data.json"
        manifest_path = dst / "audit" / "manifest.json"
        cd = json.loads(cd_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        cd = patch_case_data(cd, manifest)
        cd_path.write_text(json.dumps(cd, ensure_ascii=False, indent=2))
        print("  case_data.json → provenance + data_classification ajoutés")

        # 5. Reconstruire manifest avec hashes corrects
        remove_set = set(sc.get("remove_files", []))
        manifest = rebuild_manifest_files(dst, manifest, case_id, remove_set)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        print(f"  manifest.json → {len(manifest['files'])} fichiers, hashes recomputés")

    if errors:
        for e in errors:
            print(f"ERREUR : {e}", file=sys.stderr)
        return 1

    if not dry_run:
        print(f"\ndatasets/demo/ généré avec {len(SCENARIOS)} scénarios.")
    else:
        print("\n(dry-run terminé — aucun fichier écrit)")
    return 0


# ── Entrée ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force",   action="store_true", help="Écraser datasets/demo/ si existant")
    parser.add_argument("--dry-run", action="store_true", help="Aperçu sans écriture")
    args = parser.parse_args()
    sys.exit(generate(force=args.force, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
