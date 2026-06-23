# Claim Intake Agent

## Rôle

Premier agent du pipeline ClaimShield Santé. Il reçoit un dossier de remboursement déposé dans `storage/incoming/` et vérifie sa complétude documentaire **sans appel LLM** (agent purement déterministe).

## Responsabilités

1. **Inventaire** — lister tous les fichiers présents dans le répertoire source.
2. **Complétude** — vérifier que chaque document obligatoire est présent.
3. **Intégrité** — calculer le hash SHA-256 et détecter le type MIME de chaque fichier.
4. **Décision** — retourner `PASS` si le dossier est complet, `FAIL` sinon.

Le dossier rejeté (`FAIL`) est arrêté immédiatement : aucun agent suivant n'est invoqué.

## Permissions

| Ressource | Accès |
|---|---|
| `storage/incoming/{case_id}/` | Lecture seule |
| `storage/quarantine/{case_id}/` | Écriture (si rejet) |
| LLM Ollama | **Aucun** |
| Base de données | **Aucun** |
| Réseau externe | **Aucun** |

## Entrée (`ClaimIntakeInput`)

```python
ClaimIntakeInput(
    case_id="CLM-0004",
    source_path=Path("storage/incoming/CLM-0004"),
    required_documents=[
        "demande_remboursement_CLM-0004.pdf",
        "facture_CLM-0004.pdf",
        "ordonnance_CLM-0004.pdf",
        "compte_rendu_CLM-0004.pdf",
    ],
)
```

## Sortie (`ClaimIntakeResult`)

```python
ClaimIntakeResult(
    case_id="CLM-0004",
    status=VerificationStatus.PASS,          # ou FAIL
    ingestion_path="storage/incoming/CLM-0004",
    documents=[DocumentEntry(...)],           # un par fichier présent
    missing_documents=[],                     # vide si PASS
    reasons=["4 document(s) reçu(s) et vérifiés (SHA-256 calculé)."],
)
```

## Scénarios couverts

| Scénario | Entrée | Résultat |
|---|---|---|
| SC-01 — Dossier complet | 4 PDFs présents | `PASS` |
| SC-04 — Facture manquante | 3 PDFs sur 4 | `FAIL`, `missing_documents=["facture_CLM-0019.pdf"]` |

## Fichiers

| Fichier | Rôle |
|---|---|
| `agent.py` | `run()` (logique pure) + `node()` (nœud LangGraph) |
| `schemas.py` | `ClaimIntakeInput` + réexport de `ClaimIntakeResult` |
| `README.md` | Ce fichier |
