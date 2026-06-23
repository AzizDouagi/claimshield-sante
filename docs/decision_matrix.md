# Matrice de décision du MVP

| ID | Règle | Condition | Résultat | Preuve attendue |
|---|---|---|---|---|
| R01 | Documents obligatoires | Demande, facture et ordonnance présentes | Continuer | Liste des documents |
| R02 | Format autorisé | Fichier PDF, PNG, JPEG ou JSON et non vide | Continuer ou quarantaine | MIME, taille, nom |
| R03 | Identité cohérente | Même patient dans les documents et données synthétiques | Continuer ou revue humaine | Identifiants comparés |
| R04 | Couverture active | Date du soin comprise dans la période de couverture | Continuer ou revue humaine | Dates du contrat |
| R05 | FHIR optionnel valide | Si un bundle est présent, JSON et références cohérentes | Continuer ou revue humaine | Erreurs FHIR localisées |
| R06 | Montants cohérents | Montants positifs et montant demandé inférieur ou égal au total facturé | Continuer ou revue humaine | Calcul détaillé |
| R07 | Facture non dupliquée | Aucun même hash ou même numéro, fournisseur, date et montant | Continuer ou alerte antifraude | Critères du doublon |
| R08 | Sécurité et validation | Toute injection, sortie invalide ou ambiguïté impose quarantaine ou revue humaine | Bloquer ou revue humaine | Motif et événement d’audit |

## Priorité des résultats

1. Une alerte de sécurité critique entraîne `QUARANTINED`.
2. Un document obligatoire absent entraîne `INCOMPLETE`.
3. Une incohérence métier entraîne `NEEDS_REVIEW`.
4. Un dossier sans blocage entraîne `READY_FOR_HUMAN_REVIEW`.
5. Seul un humain peut produire `APPROVED_BY_HUMAN` ou `REJECTED_BY_HUMAN`.

## Principe de décision

Aucune règle ne déclenche directement un paiement.

La recommandation du système est une aide à la décision et doit toujours
être validée par une personne autorisée.

## Test manuel avec CLM-0001

- Dossier complet : oui
- Documents obligatoires présents : oui
- Bundle FHIR présent : oui
- Couverture attendue : active
- Facture attendue comme doublon : non
- Prompt injection attendue : non
- Recommandation attendue : prêt pour revue humaine
- Décision finale automatique : interdite