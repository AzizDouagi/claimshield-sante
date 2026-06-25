# Privacy Agent

## Rôle

Applique des vues minimisées par rôle de lecteur sur les données d'un dossier ClaimShield Santé.  
L'agent est **entièrement déterministe** — aucun appel LLM.

## Pré-condition

Le Security Gate Agent **doit avoir retourné `ALLOW`** avant que ce nœud soit exécuté.  
Si `security_result` est absent ou que sa décision n'est pas `ALLOW`, le résultat est `FAIL` et le pipeline s'arrête.

## Pipeline

```
1. Vérification du Security Gate (decision == ALLOW)
2. Validation Pydantic de PrivacyInput
3. Calcul des champs à masquer  (rôle × classification × données réelles)
4. Pseudonymisation des valeurs personnelles si fournies
5. Détermination du statut (PASS / NEEDS_REVIEW / FAIL)
6. Retour de PrivacyResult
```

## Point d'entrée

```python
from agents.privacy_agent.agent import run, node
from agents.privacy_agent.schemas import PrivacyInput
from schemas.domain import DataClassification, ReaderRole

# Sans LangGraph
result = run(
    PrivacyInput(
        case_id="CLM-0001",
        role=ReaderRole.GESTIONNAIRE,
        data_classification=DataClassification.SYNTHETIC_TEST_DATA,
        contains_real_personal_data=False,
    ),
    security_result=gate_result,  # SecurityGateResult avec decision=ALLOW
)

# Nœud LangGraph (lit state["privacy_input"], écrit state["privacy_result"])
updates = node(state)
```

## Rôles et champs masqués

| Rôle              | Base masquée                      | Sur CONFIDENTIAL            | Sur données réelles |
|-------------------|-----------------------------------|-----------------------------|---------------------|
| `SYSTEME`         | —                                 | —                           | —                   |
| `GESTIONNAIRE`    | —                                 | Champs personnels           | Champs personnels   |
| `MEDECIN_CONSEIL` | Champs financiers                 | Perso + financiers          | Champs personnels   |
| `AUDITEUR`        | Perso + médicaux                  | Perso + médicaux + financiers | Champs personnels |
| `EXTERNE`         | Tous les champs sensibles         | Tous les champs sensibles   | Tous               |

**Champs personnels** : `patient_name`, `patient_id`, `birth_date`, `gender`  
**Champs financiers** : `total_billed`, `amount_requested`, `patient_share`, `coverage_rate`, `payer_name`, `invoice_number`, `prescription_number`  
**Champs médicaux** : `procedures`, `prescriptions`, `diagnosis_codes`, `encounter_class`, `provider_id`, `organization_id`

## Statuts de sortie

| Statut          | Condition                                                        |
|-----------------|------------------------------------------------------------------|
| `PASS`          | Classification SYNTHETIC_TEST_DATA ou ANONYMIZED, pas de données réelles |
| `NEEDS_REVIEW`  | Données CONFIDENTIAL ou présence de données personnelles réelles |
| `FAIL`          | Security Gate non ALLOW, ou entrée invalide                      |

## Schéma d'entrée (`PrivacyInput`)

| Champ                       | Type              | Défaut                    | Description                            |
|-----------------------------|-------------------|---------------------------|----------------------------------------|
| `case_id`                   | `str`             | requis                    | Format `CLM-XXXX`                      |
| `role`                      | `ReaderRole`      | `GESTIONNAIRE`            | Rôle du lecteur                        |
| `data_classification`       | `DataClassification` | `SYNTHETIC_TEST_DATA`  | Classification du dossier              |
| `contains_real_personal_data` | `bool`          | `False`                   | Présence de données réelles            |
| `fields_to_evaluate`        | `list[str]`       | `[]`                      | Champs à évaluer (tous si vide)        |
| `patient_name`              | `str \| None`     | `None`                    | Pour pseudonymisation                  |
| `patient_id`                | `str \| None`     | `None`                    | Pour pseudonymisation                  |
| `payer_name`                | `str \| None`     | `None`                    | Pour pseudonymisation                  |
| `invoice_number`            | `str \| None`     | `None`                    | Pour masquage partiel                  |
| `prescription_number`       | `str \| None`     | `None`                    | Pour masquage partiel                  |

## Schéma de sortie (`PrivacyResult`)

Défini dans `schemas/results.py` :

```python
class PrivacyResult(StrictModel):
    case_id: str
    status: VerificationStatus          # PASS | NEEDS_REVIEW | FAIL
    data_classification: DataClassification
    contains_real_personal_data: bool
    masked_fields: list[str]            # champs masqués pour ce rôle
    reasons: list[str]                  # motifs lisibles
```

## Dépendances

- `security/access_policies.py` — politiques RBAC par rôle
- `tools/pseudonymize.py` — masquage et pseudonymisation des valeurs
- `schemas/domain.py` — `ReaderRole`, `DataClassification`, `VerificationStatus`
- `schemas/results.py` — `PrivacyResult`, `SecurityGateResult`
- `state/claim_state.py` — `ClaimState`, `validate_state_update`

## Invariants

- Jamais de décision de remboursement.
- Jamais de contenu brut (OCR, PDF) dans le résultat.
- Jamais de secret, token ou chemin absolu dans `PrivacyResult`.
- Le résultat est JSON-sérialisable (StrictModel Pydantic).
