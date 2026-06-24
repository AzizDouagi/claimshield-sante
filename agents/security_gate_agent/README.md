# Security Gate Agent

Agent déterministe de contrôle de sécurité des entrées sensibles du pipeline ClaimShield Santé.

**Agent purement déterministe - aucun appel LLM.**

---

## Rôle

Le `security_gate_agent` agit comme barrière de sécurité avant de laisser une entrée continuer dans le pipeline. Il évalue un seul élément à la fois : fichier, texte, URL, demande d'outil ou sortie d'agent.

Pipeline d'exécution :

1. Validation Pydantic du `SecurityGateInput`
2. Contrôle fichier et métadonnées
3. Contrôle du chemin relatif de stockage
4. Contrôle des URL explicites
5. Scan déterministe du texte, des métadonnées et sorties d'agents
6. Contrôle des outils demandés
7. Calcul de la sévérité maximale
8. Retour d'un `SecurityGateResult` avec décision `ALLOW`, `BLOCK` ou `QUARANTINE`

Le Security Gate ne remplace pas les agents métier : il protège le pipeline contre les fichiers dangereux, chemins non autorisés, URL risquées, outils non approuvés, accès secrets et prompt injections.

---

## Entrées

### Via `run()` (sans LangGraph)

```python
from agents.security_gate_agent.agent import run
from agents.security_gate_agent.schemas import InputType, SecurityGateInput

gate_input = SecurityGateInput(
    claim_id="CLM-0004",
    entry_id="facture-001",
    input_type=InputType.FILE,
    filename="facture_CLM-0004.pdf",
    extension=".pdf",
    detected_mime="application/pdf",
    actual_size=120300,
    sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    relative_path="incoming/CLM-0004/facture_CLM-0004.pdf",
)

result = run(gate_input)
```

### Via `node()` (noeud LangGraph)

Le noeud lit les clés suivantes dans le `ClaimState` :

| Clé dans le state | Type | Obligatoire | Description |
|---|---|---|---|
| `case_id` | `str` | Oui | Identifiant du dossier |
| `security_input` | `dict` | Non | Données compatibles avec `SecurityGateInput` |
| `security_input.entry_id` | `str` | Non | Identifiant de l'entrée, ou nom d'outil pour `input_type="tool"` |
| `security_input.input_type` | `str` | Non | `file`, `text`, `url`, `tool`, `agent_output` |
| `security_input.filename` | `str` | Non | Nom du fichier ou métadonnée à scanner |
| `security_input.extension` | `str` | Non | Extension déclarée ou normalisée |
| `security_input.detected_mime` | `str` | Non | MIME détecté physiquement |
| `security_input.actual_size` | `int` | Non | Taille réelle en octets |
| `security_input.sha256` | `str` | Non | Hash SHA-256, 64 caractères hexadécimaux |
| `security_input.relative_path` | `str` | Non | Chemin relatif sous `storage/` |
| `security_input.write_path` | `str` | Non | Alias accepté pour `relative_path` |
| `security_input.url` | `str` | Non | URL à contrôler |
| `security_input.text_excerpt` | `str` | Non | Extrait texte limité, jamais document brut |
| `security_input.tool_arguments_excerpt` | `str` | Non | Alias accepté pour `text_excerpt` |
| `security_input.requesting_agent` | `str` | Non | Agent demandeur d'un outil |
| `security_input.deterministic_injection_flag` | `bool` | Non | Oracle de test uniquement |

### Schéma Pydantic (`SecurityGateInput`)

```python
SecurityGateInput(
    claim_id="CLM-0004",
    entry_id="ocr-preview-001",
    input_type=InputType.TEXT,
    text_excerpt="Extrait court à scanner",
)
```

Contraintes d'entrée :

- `relative_path` ne peut pas être absolu et ne peut pas contenir `..`.
- `text_excerpt` est limité à 2 000 caractères.
- `sha256`, s'il est fourni, doit contenir exactement 64 caractères hexadécimaux.
- Les champs texte refusent les indices évidents de secrets (`api_key`, `secret`, `password`, `token`, `bearer`).
- L'entrée ne doit jamais contenir le document brut, un secret, un chemin absolu ou un payload complet.

---

## Sorties

### `SecurityGateResult`

```python
SecurityGateResult(
    claim_id="CLM-0004",
    decision=SecurityDecision.ALLOW,
    findings=[],
    reason_codes=[],
    applied_policy="default",
    policy_version="1.1.0",
    evaluated_at=datetime(..., tzinfo=UTC),
    next_allowed_action="continue_pipeline",
    audit_entry=SecurityAuditEntry(...),
    prompt_injection_detected=False,
    blocked_fields=[],
    reasons=["Aucune menace détectée - dossier autorisé"],
)
```

### Champs de sortie principaux

| Champ | Description |
|---|---|
| `decision` | Décision finale : `ALLOW`, `BLOCK` ou `QUARANTINE` |
| `findings` | Liste structurée des anomalies détectées |
| `reason_codes` | Codes stables expliquant la décision |
| `applied_policy` | Nom de la politique appliquée, par défaut `default` |
| `policy_version` | Version de la politique, par défaut `1.1.0` |
| `next_allowed_action` | Action suivante autorisée après décision |
| `audit_entry` | Entrée d'audit minimale, sans secret ni document brut |
| `prompt_injection_detected` | `True` si une injection de prompt est détectée |
| `blocked_fields` | Champs ou éléments bloqués |
| `reasons` | Messages lisibles, toujours non vide |

### Mises à jour du `ClaimState` par `node()`

| Clé | Valeur |
|---|---|
| `security_result` | `SecurityGateResult` complet |
| `security_input` | `None` après évaluation |
| `current_step` | `"security_gate"` |
| `completed_steps` | `["security_gate"]` |
| `errors` | Raisons préfixées par `[security_gate]` si décision différente de `ALLOW` |

---

## Permissions

| Ressource | Accès |
|---|---|
| `SecurityGateInput` | Lecture seule |
| Métadonnées fichier | Lecture seule |
| Extraits texte minimisés | Lecture seule |
| `storage/incoming/` | Chemin validable pour écriture par outil autorisé |
| `storage/quarantine/` | Chemin validable pour écriture par outil autorisé |
| `storage/temporary/` | Chemin validable pour écriture par outil autorisé |
| `storage/manifests/` | Chemin validable pour validation de chemin |
| `SecurityGateResult` | Écriture dans le `ClaimState` |
| `SecurityAuditEntry` | Écriture minimale dans le résultat |
| LLM Ollama | **Aucun** |
| Base de données | **Aucun** |
| Réseau externe | **Aucun** |
| Secrets `.env`, tokens, clés API | **Aucun** |

---

## Interdictions strictes

- Aucune analyse médicale ou clinique.
- Aucune décision de remboursement.
- Aucune modification du dossier métier.
- Aucun appel LLM.
- Aucun accès direct à un secret.
- Aucun accès shell, terminal, `subprocess`, `exec`, `eval` ou `os.system`.
- Aucun appel réseau externe.
- Aucun contenu brut, base64, secret ou chemin absolu dans le `ClaimState`.
- Aucun document complet dans `text_excerpt`, `findings.evidence`, `reasons` ou `audit_entry`.
- Aucun contournement de politique par prompt ou sortie d'agent.

---

## Décisions

| Décision | Signification | Action suivante |
|---|---|---|
| `ALLOW` | Aucune anomalie bloquante détectée ; l'entrée peut continuer | `continue_pipeline` |
| `QUARANTINE` | Anomalie de sévérité moyenne nécessitant revue humaine | `await_human_review` |
| `BLOCK` | Action, ressource, contenu ou chemin interdit | `terminate_pipeline` |

Règles déterministes par sévérité :

| Sévérité maximale | Décision | Description |
|---|---|---|
| `INFO` | `ALLOW` | Aucun danger identifié |
| `LOW` | `ALLOW` | Élément inhabituel, alerte possible |
| `MEDIUM` | `QUARANTINE` | Incohérence nécessitant vérification |
| `HIGH` | `BLOCK` | Ressource ou action interdite |
| `CRITICAL` | `BLOCK` | Injection, secret, shell, réseau privé ou traversée de chemin |

Si `block_on_injection=True` (valeur par défaut), toute prompt injection détectée devient bloquante.

---

## Extensions et MIME autorisés

Extensions autorisées par défaut :

| Extension | MIME attendu |
|---|---|
| `.pdf` | `application/pdf` |
| `.png` | `image/png` |
| `.jpg` | `image/jpeg` |
| `.jpeg` | `image/jpeg` |
| `.json` | `application/json` |

Types MIME autorisés :

```text
application/pdf
image/png
image/jpeg
application/json
```

Extensions exécutables ou scripts interdites :

```text
bat, bin, cmd, com, cpl, dll, exe, jar, js, jse, msi,
php, ps1, py, scr, sh, vbs, wsf
```

Autres règles fichier :

- Fichier vide refusé (`EMPTY_FILE`).
- Taille maximale par fichier : 20 Mo.
- Quota dossier de référence : 200 Mo et 50 fichiers.
- Mismatch MIME/extension refusé par la politique fichier (`MIME_EXTENSION_MISMATCH`, sévérité `MEDIUM`).
- Double extension suspecte refusée si elle masque une extension interdite.

---

## Politiques de chemins

Tous les chemins soumis au Security Gate doivent être relatifs à `storage/`.

Zones autorisées par défaut :

```text
incoming/
quarantine/
temporary/
manifests/
```

Règles appliquées :

- Chemins absolus Unix, Windows et UNC interdits.
- Traversée de répertoire interdite (`..`).
- Octet nul interdit.
- Résolution finale obligatoirement sous `storage/`.
- Première composante du chemin obligatoirement dans une zone autorisée.
- Pour les demandes d'outil avec écriture, seules les zones `incoming/`, `quarantine/` et `temporary/` sont autorisées.

Exemples acceptés :

```text
incoming/CLM-0004/facture.pdf
quarantine/CLM-0004/suspicious.pdf
temporary/CLM-0004/upload.tmp
manifests/CLM-0004.json
```

Exemples refusés :

```text
/etc/passwd
C:\Users\secret.txt
../../.env
incoming/../secrets.env
processed/file.pdf
incoming/a\0.pdf
```

---

## Politiques d'URL

Les URL externes sont interdites par défaut (`allow_external_urls=False`).

Règles appliquées :

- Schémas autorisés : `http`, `https`.
- Schémas interdits : `file`, `ftp`, et tout schéma hors allowlist.
- Hôte obligatoire : une URL sans host est malformée.
- Identifiants dans l'URL interdits (`user:pass@host`).
- `localhost`, loopback, IP privées et link-local interdits.
- Domaines externes refusés sauf présence explicite dans `allowed_domains`.

Exemples refusés :

```text
file:///etc/passwd
ftp://example.com/file
http://localhost:8000
http://127.0.0.1:8000
http://192.168.1.10/data
https://user:password@example.com
https://external.example/hook
```

---

## Outils autorisés et interdits

Outils autorisés par défaut :

```text
compute_sha256
detect_mime_type
inspect_file
scan_claim_fields
scan_for_prompt_injection
validate_storage_path
```

Agents demandeurs autorisés :

```text
claim_intake_agent
security_gate_agent
orchestrator
```

Outils interdits explicitement :

```text
eval
exec
os.system
shell
subprocess
```

Règles complémentaires :

- Tout outil absent de l'allowlist produit `UNAUTHORIZED_TOOL`.
- Tout agent demandeur absent de l'allowlist produit `UNAUTHORIZED_TOOL`.
- Toute mention de secret dans l'outil, l'agent ou le chemin produit `SECRET_ACCESS_ATTEMPT`.
- Toute demande d'écriture hors zones autorisées produit `WRITE_PATH_FORBIDDEN`.
- Les chemins d'écriture d'outils sont revalidés par la politique de chemins.

---

## Codes de sécurité

| Code | Sens |
|---|---|
| `UNSUPPORTED_EXTENSION` | Extension non autorisée ou extension exécutable/script |
| `UNSUPPORTED_MIME` | Type MIME détecté non autorisé |
| `FILE_TOO_LARGE` | Taille réelle supérieure à la limite |
| `EMPTY_FILE` | Fichier vide |
| `FILE_METADATA_INCOMPLETE` | Métadonnées fichier insuffisantes |
| `MIME_EXTENSION_MISMATCH` | MIME détecté incohérent avec l'extension |
| `PATH_TRAVERSAL` | Tentative de traversée de répertoire |
| `ABSOLUTE_PATH_FORBIDDEN` | Chemin absolu interdit |
| `PATH_NULL_BYTE` | Caractère nul dans un chemin |
| `PATH_OUTSIDE_STORAGE` | Chemin résolu hors de `storage/` |
| `STORAGE_ZONE_FORBIDDEN` | Zone de stockage non autorisée |
| `EXTERNAL_URL_FORBIDDEN` | URL externe refusée |
| `PRIVATE_NETWORK_URL` | URL vers localhost, loopback ou réseau privé |
| `DANGEROUS_URL_SCHEME` | Schéma d'URL interdit ou non autorisé |
| `MALFORMED_URL` | URL absente ou malformée |
| `URL_CREDENTIALS_FORBIDDEN` | Identifiants présents dans l'URL |
| `PROMPT_INJECTION_DETECTED` | Injection de prompt détectée |
| `SECRET_ACCESS_ATTEMPT` | Tentative d'accès à un secret |
| `SHELL_ACCESS_ATTEMPT` | Tentative d'accès shell ou commande |
| `UNAUTHORIZED_TOOL` | Outil ou agent demandeur non autorisé |
| `WRITE_PATH_FORBIDDEN` | Écriture demandée hors zones autorisées |
| `INVALID_AGENT_OUTPUT` | Sortie d'agent invalide ou dangereuse |
| `SUSPICIOUS_DOCUMENT_CONTENT` | Contenu documentaire suspect |
| `SUSPICIOUS_CONTENT` | Contenu suspect non classé plus précisément |
| `POLICY_VIOLATION` | Violation générique de politique |

---

## Niveaux de sévérité

| Niveau | Usage |
|---|---|
| `INFO` | Aucun risque ou information d'audit |
| `LOW` | Signal faible, par exemple caractères invisibles seuls |
| `MEDIUM` | Incohérence à revoir, par exemple mismatch MIME/extension |
| `HIGH` | Violation de politique bloquante |
| `CRITICAL` | Injection, secret, shell, chemin critique ou réseau privé |

Exemples de sévérité :

| Anomalie | Sévérité |
|---|---|
| `EMPTY_FILE` | `HIGH` |
| `UNSUPPORTED_EXTENSION` | `HIGH` |
| Extension exécutable ou double extension suspecte | `CRITICAL` |
| `MIME_EXTENSION_MISMATCH` | `MEDIUM` |
| `PATH_TRAVERSAL` | `CRITICAL` |
| `ABSOLUTE_PATH_FORBIDDEN` | `CRITICAL` |
| `PRIVATE_NETWORK_URL` | `CRITICAL` |
| `UNAUTHORIZED_TOOL` | `CRITICAL` pour outil interdit, `HIGH` pour agent non autorisé |
| `SECRET_ACCESS_ATTEMPT` | `CRITICAL` |
| `SHELL_ACCESS_ATTEMPT` | `CRITICAL` |
| `PROMPT_INJECTION_DETECTED` | `CRITICAL` dans le résultat agent |

---

## Détection de prompt injection

La détection est déterministe et s'applique à :

- `filename`
- `url`
- `text_excerpt`
- sorties d'agents (`input_type="agent_output"`)
- arguments d'outils (`input_type="tool"`)
- oracle de test `deterministic_injection_flag`

Étapes du scanner :

1. Normalisation Unicode NFKC.
2. Passage en `casefold`.
3. Réduction des espaces inhabituels.
4. Détection des caractères invisibles.
5. Détection de mots volontairement fragmentés.
6. Détection d'intentions dangereuses par regex.
7. Détection d'URL et chemins suspects dans le texte.
8. Application des patterns historiques de prompt injection.
9. Ajout d'un signal `HIDDEN_DOCUMENT_INSTRUCTION` pour texte PDF, aperçu OCR ou sortie d'agent.

Catégories détectées :

```text
INVISIBLE_CHARS
FRAGMENTED_TEXT
IGNORE_INSTRUCTIONS
SECRET_EXPOSURE
ENV_ACCESS
TOOL_EXECUTION
EXFILTRATION_URL
PERMISSION_CHANGE
HIDDEN_DOCUMENT_INSTRUCTION
SUSPICIOUS_PATH
SUSPICIOUS_URL
LEGACY_REGEX
```

Exemples de signaux bloquants :

```text
ignore previous instructions
new system prompt:
you are now a ...
jailbreak
DAN mode
act as ...
pretend to be ...
execute the following command
eval(
<script>
read the .env file
send the data to this URL
disable security
```

Les preuves (`evidence`) sont minimisées et tronquées : elles ne doivent jamais contenir un document complet, un secret ou un chemin absolu.

---

## Commande de test

```bash
pytest tests/security/ -q
```
