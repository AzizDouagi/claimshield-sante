# Évaluation qualité des recommandations — `scripts/evaluate_recommendations.py`

Script manuel, **hors CI**, distinct de la suite pytest (`tests/e2e/`, LLM
toujours stubbé). Nécessite un vrai Ollama joignable via `OLLAMA_BASE_URL`.
Mesure la qualité réelle du jugement de `case_reviewer_agent` sur les 37
dossiers de `datasets/fixtures/valid/`.

## Métrique

Compare `state["review_result"].result_payload.recommendation` (la
**pré-recommandation** de `case_reviewer_agent`, présente dans le state au
moment de l'interruption HITL) à `ground_truth.json["expected_recommendation"]`
de chaque dossier — **jamais** une décision humaine simulée. C'est la seule
comparaison cohérente avec le verrouillage de l'étape 13 : `CaseReviewerResult`
ne peut structurellement jamais représenter une décision finale automatique
(`status`/`human_review_required` verrouillés), donc il n'existe aucune
« recommandation finale » à comparer sans un humain réel dans la boucle.

## Limite actuelle — le script rapportera 0 dossier comparable

Voir `CLAUDE.md` (section « Finalisation post-Phase 4 ») : rien en production
ne construit aujourd'hui `ocr_input`/`fhir_input` à partir du manifest de
`claim_intake_agent`. Toute soumission avec de vrais documents échoue donc
sur `document_ocr`/`fhir_validator` avant d'atteindre `case_reviewer` — chaque
dossier sera classé `EARLY_EXIT`, jamais `MATCH`/`MISMATCH`. Le script reste
l'infrastructure prête pour le jour où ce câblage existera ; en attendant, il
sert déjà à vérifier qu'aucun dossier ne fait planter le pipeline (`ERROR`,
distinct d'`EARLY_EXIT`).

## Verdicts par dossier

| Verdict | Signification |
|---|---|
| `MATCH` | `review_result` présent, recommandation = attendu |
| `MISMATCH` | `review_result` présent, recommandation ≠ attendu |
| `EARLY_EXIT` | pipeline interrompu avant `case_reviewer` (exclu du taux) |
| `ERROR` | exception technique pendant l'invocation (exclu du taux, dossier suivant non affecté) |

## Usage

```bash
python scripts/evaluate_recommendations.py                                   # 37 dossiers
python scripts/evaluate_recommendations.py --cases CLM-0001,CLM-0007         # sous-ensemble
python scripts/evaluate_recommendations.py --limit 5 --format table          # test rapide
python scripts/evaluate_recommendations.py --output logs/evals/run1.json     # sortie explicite
```

Sortie par défaut : `logs/evals/<horodatage>_recommendations.json` (`run_metadata`,
`summary` — taux, matrice de confusion, compteurs par verdict —, `cases` — détail
par dossier). `--format csv` en sortie fichier si `--output *.csv`.
