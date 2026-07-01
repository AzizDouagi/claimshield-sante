# Provenance et Licence

Les six dossiers de `datasets/demo/` sont entièrement synthétiques.
Ils proviennent de cas générés par Synthea, sélectionnés dans
`datasets/fixtures/valid/`, puis enrichis pour les scénarios de démonstration
ClaimShield Santé.

## Source

| Élément | Valeur |
|---|---|
| Générateur | Synthea |
| Projet source | `https://github.com/synthetichealth/synthea` |
| Type de données | Données patient et parcours de soins synthétiques |
| Licence | Apache License 2.0 |
| Données réelles | Aucune |
| Classification | `SYNTHETIC_TEST_DATA` |

Les noms de patients et prestataires, identifiants, adresses, événements,
actes, médicaments, montants et documents PDF sont des artefacts synthétiques
issus du pipeline de génération. Les fichiers ne doivent pas être mélangés à
des données réelles ni utilisés comme preuves médicales.

## Méthode de génération

Le flux reproductible est :

```text
Synthea output
  -> tools/dataset_builder/select_first_case.py
  -> tools/dataset_builder/generate_case_documents.py
  -> scripts/import_synthea_claimshield_cases.py
  -> scripts/generate_demo_data.py
```

`scripts/generate_demo_data.py` sélectionne six dossiers stables, applique les
patchs métier des scénarios, renseigne `agent_under_test`, met à jour la
provenance dans `case_data.json`, puis reconstruit les entrées de manifest avec
les tailles et SHA-256 des fichiers présents.

## Contrôles

Les garanties sont vérifiées par `pytest tests/unit/test_demo_dataset.py -q` :

- exactement six dossiers `CLM-*` ;
- aucun dossier marqué comme contenant des données personnelles réelles ;
- présence des documents attendus et absence intentionnelle de la facture
  `CLM-0019` ;
- vérité terrain contenant scénario, anomalie, résultat attendu et agent
  concerné ;
- JSON conformes aux schémas Pydantic du projet ;
- hashes SHA-256 du manifest alignés avec les fichiers réels ;
- génération reproductible par comparaison des signatures de fichiers.

