# Politique des données du MVP

## 1. Nature des données

ClaimShield Santé utilise uniquement :

- des données synthétiques générées avec Synthea ;
- des documents PDF ou images générés artificiellement ;
- des données publiques ou sous licence compatible ;
- aucun document médical réel.

## 2. Données interdites

Il est interdit d’utiliser :

- des dossiers médicaux réels ;
- des noms ou coordonnées de vraies personnes ;
- des identifiants nationaux réels ;
- des comptes bancaires réels ;
- des secrets ou clés API dans les datasets ;
- des documents dont la licence est inconnue.

## 3. Formats autorisés

Pour le MVP :

- PDF ;
- PNG ;
- JPEG ;
- JSON.

Les formats suivants sont exclus :

- DICOM ;
- fichiers exécutables ;
- archives non contrôlées ;
- scripts ;
- documents contenant du contenu actif.

## 4. Zones de stockage prévues

- `storage/incoming/` : dépôt temporaire ;
- `storage/quarantine/` : fichiers bloqués ;
- `storage/sanitized/` : copies autorisées pour analyse ;
- `storage/artifacts/` : données extraites et résultats ;
- `logs/` : événements techniques minimisés.

Ces dossiers décrivent la politique cible. Leur comportement sera
implémenté dans les étapes suivantes.

## 5. Minimisation

Les logs ne doivent pas contenir :

- le texte OCR complet ;
- les documents originaux ;
- les prompts système ;
- les clés ou tokens ;
- les identifiants médicaux complets.

## 6. Audit

Chaque événement d’audit devra contenir au minimum :

- `claim_id` ;
- acteur ou composant ;
- action ;
- horodatage ;
- version de règle ;
- résultat ;
- motif ;
- identifiant de corrélation.

## 7. Décisions humaines

La personne autorisée doit pouvoir :

- accepter ;
- modifier ;
- refuser ;
- demander une nouvelle analyse.

La décision et son motif doivent être enregistrés dans l’audit.