# Démonstration — six scénarios synthétiques

`datasets/demo/` contient exactement six dossiers synthétiques destinés aux
démonstrations et aux tests de non-régression du pipeline ClaimShield Santé.
Ils sont dérivés de `datasets/fixtures/valid/` puis enrichis par
`scripts/generate_demo_data.py` pour couvrir les décisions métier principales.

| Dossier | Scénario | Agent concerné | Résultat attendu | Cas couvert |
|---|---|---|---|---|
| `CLM-0004` | `SC-01` | `full_pipeline` | `APPROVE` | Dossier complet, aucune anomalie |
| `CLM-0015` | `SC-02` | `identity_coverage_agent` | `REJECT` | Pré-autorisation requise non fournie |
| `CLM-0005` | `SC-03` | `security_gate_agent` | `REJECT` | Injection de prompt détectée |
| `CLM-0019` | `SC-04` | `claim_intake_agent` | `REJECT` | Facture obligatoire absente |
| `CLM-0024` | `SC-05` | `fraud_detection_agent` | `REJECT` | Facture en doublon |
| `CLM-0032` | `SC-06` | `clinical_consistency_agent` | `PENDING` | Volume clinique anormal, revue humaine |

## Structure d'un dossier

```text
CLM-XXXX/
  input/
    demande_remboursement_CLM-XXXX.pdf
    facture_CLM-XXXX.pdf
    ordonnance_CLM-XXXX.pdf
    compte_rendu_CLM-XXXX.pdf
    claim.json
    patient.json
    patient_fhir_bundle.json
    fhir/*.json
  oracle/
    case_data.json
    ground_truth.json
  audit/
    manifest.json
```

`CLM-0019` est volontairement incomplet : `facture_CLM-0019.pdf` est listé
comme document requis dans `ground_truth.json`, mais absent de `input/`.

## Vérité Terrain

Chaque `oracle/ground_truth.json` contient :

- `case_id`, `scenario_id` et `agent_under_test` ;
- `expected_recommendation` ;
- `expected_anomalies` ;
- les documents requis, optionnels et manquants attendus ;
- les résultats attendus par agent (`expected_security`,
  `expected_identity`, `expected_coverage`, `expected_fhir`,
  `expected_clinical_consistency`, `expected_fraud`) ;
- les raisons de recommandation et de revue humaine.

## Reproduction

Depuis la racine du dépôt :

```bash
python scripts/generate_demo_data.py --dry-run
python scripts/generate_demo_data.py --force
pytest tests/unit/test_demo_dataset.py -q
```

La génération est déterministe parce qu'elle copie des fixtures versionnées,
applique des patchs déclaratifs dans `scripts/generate_demo_data.py`, supprime
seulement les fichiers explicitement listés par scénario, puis recalcule les
tailles et SHA-256 dans `audit/manifest.json`.

