# Périmètre du MVP ClaimShield Santé

## 1. Scénario unique

Le MVP analyse une demande synthétique de remboursement liée à une
consultation médicale.

Le dossier contient obligatoirement :

- une demande de remboursement ;
- une facture médicale ;
- une ordonnance.

Il peut également contenir :

- un petit bundle FHIR synthétique au format JSON.

## 2. Acteurs

### Gestionnaire

- dépose le dossier ;
- consulte son statut ;
- consulte la recommandation ;
- transmet le dossier à la validation humaine.

### Médecin-conseil

- examine les incohérences médicales simples ;
- valide, modifie ou refuse la recommandation.

### Analyste antifraude

- consulte les doublons et anomalies détectées ;
- ne déclare jamais automatiquement une fraude.

### Responsable sécurité

- consulte les alertes de sécurité ;
- peut bloquer ou placer un dossier en quarantaine.

### Auditeur

- consulte les événements d’audit minimisés.

### Administrateur

- gère l’application ;
- ne possède pas automatiquement l’accès aux données médicales.

## 3. Entrées

- PDF ;
- PNG ;
- JPEG ;
- JSON ;
- données exclusivement synthétiques.

## 4. Sorties

Le MVP produit :

- le statut du dossier ;
- la liste des documents présents et manquants ;
- les données extraites ;
- le résultat de vérification de l’identité ;
- le résultat de vérification de la couverture ;
- le résultat FHIR si un bundle est présent ;
- les alertes de doublon ;
- les alertes de sécurité ;
- une recommandation explicable ;
- une décision humaine ;
- un journal d’audit.

## 5. Statuts possibles

- RECEIVED
- QUARANTINED
- INCOMPLETE
- NEEDS_REVIEW
- READY_FOR_HUMAN_REVIEW
- APPROVED_BY_HUMAN
- REJECTED_BY_HUMAN

## 6. Décision humaine obligatoire

Le système ne prend jamais seul une décision finale.

Un humain doit obligatoirement :

- accepter la recommandation ;
- la modifier ;
- ou la refuser.

## 7. Hors périmètre

Le MVP ne traite pas :

- les données réelles de patients ;
- DICOM ou DICOMweb ;
- les paiements automatiques ;
- les diagnostics médicaux ;
- les accusations automatiques de fraude ;
- les modèles avancés de fraude ;
- les communications A2A distribuées ;
- les terminologies médicales distantes ;
- les décisions finales prises par un LLM.