# Claim Intake Agent

Premier agent du pipeline ClaimShield Santé. Reçoit un dossier de remboursement, inspecte chaque fichier, calcule les hashes SHA-256, effectue un stockage sécurisé et retourne un manifeste complet.

**Agent hybride : contrôles déterministes obligatoires + synthèse LLM structurée.**

---

## Rôle

Le `claim_intake_agent` est la porte d'entrée du pipeline. Il ne prend aucune décision médicale ni financière : il se limite à vérifier qu'un dossier est exploitable avant de le transmettre aux agents suivants.

Pipeline d'exécution (dans l'ordre) :

1. Validation des métadonnées du dossier (non-vide, quota de fichiers)
2. Pour chaque fichier, dans l'ordre alphabétique :
   - a. Écriture dans la zone `temporary/` (`StorageService.stage_file`)
   - b. Vérification du quota destination (taille + nombre)
   - c. Inspection complète : nom · extension · taille · MIME · SHA-256
   - d. Détection de doublons par SHA-256
   - e. Déplacement atomique vers `incoming/` ou `quarantine/`
3. Vérification des documents obligatoires
4. Construction du `ClaimManifest`
5. Appel LLM avec prompt système versionné pour produire une décision Pydantic
   (`LlmIntakeDecision`) alignée sur les contrôles déterministes
6. Retour du `ClaimIntakeResult` + persistance du manifest sur disque

---

## Entrées

### Via `run()` (sans LangGraph)

```python
from agents.claim_intake_agent.agent import run
from pathlib import Path

result = run(
    case_id="CLM-0004",                          # obligatoire — pattern CLM-XXXX
    source_path=Path("datasets/demo/CLM-0004/input"),  # répertoire source
    required_documents=[                          # optionnel
        "demande_remboursement_CLM-0004.pdf",
        "facture_CLM-0004.pdf",
        "ordonnance_CLM-0004.pdf",
    ],
    depositor_id="user-42",                      # optionnel — pour le manifest
)
```

### Via `node()` (nœud LangGraph)

Le nœud lit les clés suivantes dans le `ClaimState` :

| Clé dans le state | Type | Obligatoire | Description |
|---|---|---|---|
| `case_id` | `str` | Oui | Identifiant du dossier (`CLM-XXXX`) |
| `intake_input.source_path` | `str` | Oui | Chemin absolu du répertoire source |
| `intake_input.required_documents` | `list[str]` | Non | Noms des fichiers obligatoires |
| `intake_input.depositor_id` | `str` | Non | Identifiant du déposant |

### Schéma Pydantic (`ClaimIntakeInput`)

```python
ClaimIntakeInput(
    case_id="CLM-0004",               # pattern : ^CLM-\d{4,}$
    source_path=Path("..."),
    required_documents=["facture.pdf"],
    uploaded_files=[],                # métadonnées annoncées avant inspection
)
```

---

## Sorties

### `ClaimIntakeResult`

```python
ClaimIntakeResult(
    claim_id="CLM-0004",
    status=IntakeStatus.ACCEPTED,     # accepted | quarantined | blocked | error
    manifest=ClaimManifest(...),
    accepted_count=4,
    quarantined_count=0,
    duplicate_count=0,
    error_count=0,
    reasons=["4 fichier(s) accepté(s) et stockés — dossier prêt pour traitement"],
    errors=[],
)
```

### `ClaimManifest` (contenu du manifest)

```python
ClaimManifest(
    claim_id="CLM-0004",
    received_at=datetime(..., tzinfo=UTC),
    depositor_id="user-42",           # None si non fourni
    file_count=4,
    total_size_bytes=481200,
    files=[InspectedFile(...)],       # un par fichier traité
    status=IntakeStatus.ACCEPTED,
    alerts=[],                        # messages lisibles sur documents manquants
    schema_version="1.0.0",
)
```

### `InspectedFile` (un par fichier)

```python
InspectedFile(
    original_name="facture_CLM-0004.pdf",
    storage_name="CLM-0004_0000_a3f8...pdf",  # nom physique sans le nom original
    normalized_extension=".pdf",
    detected_mime_type="application/pdf",
    actual_size=120300,
    sha256="e3b0c44298fc1c149afb...",           # 64 caractères hexadécimaux
    status=FileStatus.ACCEPTED,                 # accepted | quarantined | blocked | duplicate | error
    reasons=[],                                 # liste de StructuredError si problème
    relative_storage_path="incoming/CLM-0004/CLM-0004_0000_a3f8...pdf",
)
```

### Mises à jour du `ClaimState` par `node()`

| Clé | Valeur |
|---|---|
| `intake_result` | `ClaimIntakeResult` complet |
| `intake_status` | `IntakeStatus` promu pour le routage |
| `intake_input` | `None` — source_path absolu supprimé |
| `current_step` | `"claim_intake"` |
| `completed_steps` | `["claim_intake"]` |
| `errors` | Raisons si statut ≠ `ACCEPTED` |
| `alerts` | Documents manquants (si présents) |

---

## Permissions

| Ressource | Accès |
|---|---|
| `source_path/` | Lecture seule (fichiers source) |
| `storage/temporary/{case_id}/` | Écriture (staging atomique avant inspection) |
| `storage/incoming/{case_id}/` | Écriture (fichiers acceptés) |
| `storage/quarantine/{case_id}/` | Écriture (fichiers suspects ou doublons) |
| `storage/manifests/{case_id}.json` | Écriture (manifest de traçabilité) |
| LLM Ollama | Lecture du résumé minimisé, sortie Pydantic stricte |
| Base de données | **Aucun** |
| Réseau externe | **Aucun** |

---

## Interdictions strictes

- Aucune analyse médicale ou clinique.
- Aucun OCR (réservé à `document_ocr_agent`).
- Aucune décision de remboursement.
- Aucun contenu brut (octets, base64, texte OCR) dans le `ClaimState`.
- Aucun chemin absolu dans le manifest ou le `ClaimState`.
- Aucun écrasement silencieux de fichier déjà stocké (`NO_OVERWRITE`).
- La détection de doublon SHA-256 ne conclut **pas** à une fraude (réservé à `fraud_detection_agent`).

---

## Extensions acceptées

Configurables via `CLAIMSHIELD_ALLOWED_EXTENSIONS` (`.env`). Valeurs par défaut :

| Extension | MIME associé |
|---|---|
| `.pdf` | `application/pdf` |
| `.png` | `image/png` |
| `.jpeg` / `.jpg` | `image/jpeg` |
| `.json` | `application/json` |

---

## Types MIME acceptés

Configurables via `CLAIMSHIELD_ALLOWED_MIME_TYPES` (`.env`). Valeurs par défaut :

```
application/pdf
image/png
image/jpeg
application/json
```

La détection MIME est effectuée par `python-magic` (lecture des magic bytes). En l'absence de `libmagic`, repli sur `mimetypes` (détection par extension uniquement — la vérification de cohérence MIME/extension est alors désactivée).

---

## Limites de taille

| Limite | Valeur par défaut | Variable d'environnement |
|---|---|---|
| Taille max par fichier | 20 Mo | `CLAIMSHIELD_MAX_FILE_SIZE_MB` |
| Taille cumulée max par dossier | 200 Mo | `CLAIMSHIELD_MAX_FOLDER_SIZE_MB` |
| Nombre max de fichiers par dossier | 50 | `CLAIMSHIELD_MAX_FILES_PER_FOLDER` |

---

## Logique de quarantaine

Un fichier est dirigé vers `storage/quarantine/{case_id}/` dans les cas suivants :

| Condition | Code | Statut fichier |
|---|---|---|
| MIME détecté ≠ MIME attendu pour l'extension | `MIME_EXTENSION_MISMATCH` | `QUARANTINED` |
| Type MIME non autorisé (detecté par magic bytes) | `UNSUPPORTED_MIME_TYPE` | `QUARANTINED` |
| SHA-256 identique à un fichier déjà accepté dans le dossier | `DUPLICATE_FILE` | `DUPLICATE` |

Un fichier est **bloqué** (non stocké) dans les cas suivants :

| Condition | Code | Statut fichier |
|---|---|---|
| Fichier de 0 octet | `EMPTY_FILE` | `BLOCKED` |
| Extension non autorisée | `UNSUPPORTED_EXTENSION` | `BLOCKED` |
| Taille dépasse la limite individuelle | `FILE_TOO_LARGE` | `BLOCKED` |
| Nom de fichier contient `../` ou `..\` | `PATH_TRAVERSAL_ATTEMPT` | `BLOCKED` |
| Nom de fichier invalide | `INVALID_FILENAME` | `BLOCKED` |
| Quota dossier destination dépassé | `FOLDER_QUOTA_EXCEEDED` | `BLOCKED` |

### Priorité du statut global

```
ERROR > BLOCKED > QUARANTINED/DUPLICATE > ACCEPTED
```

| Statut global | Condition |
|---|---|
| `ERROR` | Au moins un fichier en erreur technique I/O |
| `BLOCKED` | Au moins un fichier bloqué (aucune erreur technique) |
| `QUARANTINED` | Au moins un fichier en quarantaine, doublon, ou document obligatoire manquant |
| `ACCEPTED` | Tous les fichiers acceptés, aucune alerte |

---

## Codes d'erreur

Tous les codes sont des valeurs stables de `IntakeReasonCode`. Ils ne changent jamais entre versions.

| Code | Description |
|---|---|
| `EMPTY_CLAIM` | Dossier vide — aucun fichier soumis |
| `EMPTY_FILE` | Fichier vide (0 octet) |
| `UNSUPPORTED_EXTENSION` | Extension non autorisée |
| `UNSUPPORTED_MIME_TYPE` | Type MIME non autorisé (détecté dans le contenu réel) |
| `MIME_EXTENSION_MISMATCH` | Contenu réel ≠ extension déclarée (ex. PNG dans un .pdf) |
| `FILE_TOO_LARGE` | Fichier dépasse la limite individuelle |
| `CLAIM_TOO_LARGE` | Quota cumulé du dossier dépassé |
| `PATH_TRAVERSAL_ATTEMPT` | Nom de fichier dangereux (`../`, `..\`) |
| `DUPLICATE_FILE` | SHA-256 identique à un fichier déjà accepté dans ce dossier |
| `STORAGE_ERROR` | Échec technique I/O (écriture ou déplacement impossible) |
| `TOO_MANY_FILES` | Nombre de fichiers dépasse la limite configurée |
| `FOLDER_QUOTA_EXCEEDED` | Taille cumulée dépasse le quota du dossier |
| `INVALID_FILENAME` | Nom de fichier invalide (caractères ou structure non autorisés) |
| `LLM_OUTPUT_INVALID` | Sortie LLM invalide ou indisponible |

---

## Exemples de dossiers

### Dossier valide — 3 PDFs acceptés

```
source/
  facture.pdf          → ACCEPTED → incoming/CLM-8001/CLM-8001_0000_xxx.pdf
  identite.pdf         → ACCEPTED → incoming/CLM-8001/CLM-8001_0001_yyy.pdf
  ordonnance.pdf       → ACCEPTED → incoming/CLM-8001/CLM-8001_0002_zzz.pdf

ClaimIntakeResult(
    status=ACCEPTED,
    accepted_count=3,
    quarantined_count=0,
    duplicate_count=0,
    error_count=0,
)
```

### Dossier invalide — dossier vide

```
source/   (vide)

ClaimIntakeResult(
    status=BLOCKED,
    accepted_count=0,
    errors=[StructuredError(code="EMPTY_CLAIM", ...)],
)
```

### Dossier invalide — extension refusée

```
source/
  programme.exe        → BLOCKED (UNSUPPORTED_EXTENSION)

ClaimIntakeResult(
    status=BLOCKED,
    accepted_count=0,
)
```

### Dossier avec doublon SHA-256

```
source/
  invoice.pdf          → ACCEPTED → incoming/CLM-9998/...
  invoice_copy.pdf     → DUPLICATE (même SHA-256) → quarantine/CLM-9998/...

ClaimIntakeResult(
    status=QUARANTINED,
    accepted_count=1,
    duplicate_count=1,
)
```

### Dossier avec document obligatoire manquant

```
source/
  ordonnance.pdf       → ACCEPTED
  # facture.pdf absent

required_documents=["facture.pdf", "ordonnance.pdf"]

ClaimIntakeResult(
    status=QUARANTINED,
    manifest.alerts=["Document obligatoire manquant : facture.pdf"],
)
```

### Dossier avec incohérence MIME/extension

```
source/
  image.pdf            → contenu PNG dans un fichier .pdf
                       → QUARANTINED (MIME_EXTENSION_MISMATCH)
                       → quarantine/CLM-8034/...

ClaimIntakeResult(
    status=QUARANTINED,
    quarantined_count=1,
)
```

---

## Commande de test

```bash
# Suite complète de l'agent (273 lignes, aucun mock)
pytest tests/agents/test_claim_intake.py -v

# Avec couverture
pytest tests/agents/test_claim_intake.py --cov=agents --cov-report=term-missing

# Un scénario précis
pytest tests/agents/test_claim_intake.py::test_sc01_dossier_complet -v
pytest tests/agents/test_claim_intake.py::test_dossier_vide -v
pytest tests/agents/test_claim_intake.py::test_doublon_detecte_meme_noms_differents -v
```

---

## Fichiers

| Fichier | Rôle |
|---|---|
| [agent.py](agent.py) | `run()` (logique pure, testable sans LangGraph) + `node()` (nœud LangGraph) |
| [prompt.py](prompt.py) | Chargement du prompt système versionné depuis `prompts/claim_intake_agent.yaml` |
| [schemas.py](schemas.py) | `ClaimIntakeInput` + réexports de `ClaimIntakeResult`, `ClaimManifest`, `InspectedFile`, `StructuredError` |
| [README.md](README.md) | Ce fichier |

## Dépendances internes

| Module | Usage |
|---|---|
| [config/settings.py](../../config/settings.py) | Limites et listes d'autorisation (extensions, MIME, tailles) |
| [schemas/domain.py](../../schemas/domain.py) | Enums `IntakeStatus`, `FileStatus`, `IntakeReasonCode` |
| [schemas/results.py](../../schemas/results.py) | `ClaimIntakeResult`, `ClaimManifest`, `InspectedFile`, `StructuredError` |
| [services/storage.py](../../services/storage.py) | Zones de stockage, écriture atomique, anti-path-traversal |
| [state/claim_state.py](../../state/claim_state.py) | `ClaimState`, `validate_state_update` |
| [tools/file_inspection.py](../../tools/file_inspection.py) | `inspect_file`, `validate_filename`, `compute_sha256`, `check_folder_limits` |
