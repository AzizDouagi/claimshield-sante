# Jeux de données — ClaimShield Santé

Ce dossier contient l'ensemble des dossiers de remboursement utilisés pour
les tests d'intégration, la démonstration et la validation du pipeline multi-agents.

Toutes les données sont **entièrement synthétiques** (générées par Synthea™, licence Apache 2.0).
Aucune donnée de patient réel n'est présente.

---

## Structure

```
datasets/
├── fixtures/
│   ├── valid/          # 37 dossiers CLM-0001 … CLM-0037 (tests d'intégration)
│   ├── backups/        # sauvegardes avant écrasement (--force)
│   └── metadata/
│       ├── index.json          # index de tous les cas importés
│       ├── import_report.json  # rapport du dernier import Synthea
│       └── generation_report.json
└── demo/
    ├── README.md       # description des 6 scénarios métier
    ├── PROVENANCE.md   # source, licence Apache 2.0, pipeline de génération
    └── CLM-XXXX/       # 6 dossiers couvrant les scénarios clés
```

---

## `fixtures/valid/` — 37 cas d'intégration

Importés depuis Synthea via `scripts/import_synthea_claimshield_cases.py`.

| Indicateur | Valeur |
|---|---|
| Nombre de cas | 37 (CLM-0001 → CLM-0037) |
| Montant facturé | 578 USD – 113 495 USD (moy. 6 324 USD) |
| Actes médicaux | 1 – 29 par dossier (moy. 7,1) |
| Médicaments | 1 – 42 par dossier (moy. 2,7) |
| Recommandation oracle | `APPROVE` (tous les cas — base nominale) |

### Structure d'un cas

```
CLM-XXXX/
  input/
    demande_remboursement_CLM-XXXX.pdf   # formulaire de demande
    facture_CLM-XXXX.pdf                 # facture médicale
    ordonnance_CLM-XXXX.pdf              # prescription
    compte_rendu_CLM-XXXX.pdf            # compte rendu de consultation
    patient_fhir_bundle.json             # bundle FHIR R4 (optionnel)
    claim.json                           # données Synthea brutes (claim)
    patient.json                         # données Synthea brutes (patient)
  oracle/
    case_data.json      # données Synthea structurées + provenance
    ground_truth.json   # oracle complet : expected_recommendation,
                        # expected_anomalies, expected_security,
                        # expected_privacy, expected_identity,
                        # expected_coverage, expected_fhir,
                        # expected_clinical_consistency, expected_fraud
  audit/
    manifest.json       # inventaire SHA-256 + résumé financier
```

### Importer ou mettre à jour les fixtures

```bash
# Importer tous les cas depuis Synthea
python scripts/import_synthea_claimshield_cases.py

# Prévisualiser sans écrire
python scripts/import_synthea_claimshield_cases.py --dry-run

# Forcer l'écrasement (sauvegarde automatique dans fixtures/backups/)
python scripts/import_synthea_claimshield_cases.py --force

# Importer un cas spécifique
python scripts/import_synthea_claimshield_cases.py --case CLM-0001
```

---

## `demo/` — 6 scénarios métier

Sous-ensemble de 6 cas sélectionnés et enrichis pour couvrir les situations
représentatives du pipeline. Les `ground_truth.json` ont été adaptés manuellement.

| Dossier  | Scénario | Agent concerné | Résultat | Description courte |
|----------|----------|----------------|----------|--------------------|
| CLM-0004 | SC-01 | `full_pipeline` | `APPROVE`  | Approbation standard — dossier parfait |
| CLM-0015 | SC-02 | `identity_coverage_agent` | `REJECT`   | Pré-autorisation requise non fournie |
| CLM-0005 | SC-03 | `security_gate_agent` | `REJECT`   | Injection de prompt détectée |
| CLM-0019 | SC-04 | `claim_intake_agent` | `REJECT`   | Document obligatoire manquant |
| CLM-0024 | SC-05 | `fraud_detection_agent` | `REJECT`   | Facture en doublon |
| CLM-0032 | SC-06 | `clinical_consistency_agent` | `PENDING`  | Incohérence clinique — revue médecin requise |

Voir [`demo/README.md`](demo/README.md) pour le détail de chaque scénario
et [`demo/PROVENANCE.md`](demo/PROVENANCE.md) pour la source et la licence.

Chaque `ground_truth.json` de démonstration contient `scenario_id`,
`agent_under_test`, `expected_recommendation`, `expected_anomalies`,
les documents requis/manquants et les résultats attendus par agent.

### Reproduire `datasets/demo/`

```bash
python scripts/generate_demo_data.py --dry-run
python scripts/generate_demo_data.py --force
pytest tests/unit/test_demo_dataset.py -q
```

---

## Provenance et licences

| Élément | Valeur |
|---|---|
| Générateur | Synthea™ — MITRE Corporation |
| Type de données | Entièrement synthétique |
| Licence des données | Apache License 2.0 |
| Données réelles | Aucune (`contains_real_personal_data: false`) |
| Classification | `SYNTHETIC_TEST_DATA` |

Chaque `case_data.json` contient un champ `provenance` avec le générateur,
la licence et l'horodatage d'import. Voir [`demo/PROVENANCE.md`](demo/PROVENANCE.md).
