# ClaimShield Santé

Système multi-agent sécurisé pour le traitement de dossiers synthétiques de remboursement médical.

Construit avec **LangGraph**, **Ollama (gemma4)** et **Pydantic v2**. Toutes les décisions financières et médicales restent déterministes ou humaines — le LLM explique et classe, jamais n'autorise seul.

---

## Prérequis

| Outil | Version minimale | Obligatoire |
|---|---|---|
| Python | 3.11+ | Oui |
| Ollama | 0.4+ avec `gemma4:latest` tiré | Oui |
| Git | tout récent | Oui |
| `libmagic` | 5.x | Non — repli automatique sur `mimetypes` |

Vérifier que le modèle local est disponible :

```bash
ollama list | grep gemma4
```

> **Note `libmagic`** — utilisée par `python-magic` pour détecter le type MIME réel d'un fichier à partir de son contenu (magic bytes), indépendamment de son extension.
> Sans `libmagic`, le code se replie automatiquement sur `mimetypes` (détection par extension uniquement) : les fichiers restent traités, mais les incohérences MIME/extension ne sont plus détectées.
>
> macOS : `brew install libmagic`
> Linux : `apt-get install libmagic1` ou `dnf install file-libs`

---

## Installation

```bash
# 1. Cloner le dépôt
git clone <url-du-dépôt>
cd claimshield-sante

# 2. Créer et activer l'environnement virtuel
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Installer la dépendance système libmagic (recommandé)
# macOS :
brew install libmagic
# Ubuntu/Debian :
# sudo apt-get install libmagic1
# Fedora/RHEL :
# sudo dnf install file-libs

# 4. Installer le projet et les dépendances de développement
pip install -e ".[dev]"

# 5. Configurer l'environnement
cp .env.example .env
# Éditer .env si nécessaire (les chemins Synthea et Ollama sont préconfigurés)
```

---

## Vérification de l'installation

```bash
# Vérifier que Python et les imports de base fonctionnent
python -V

# Vérifier que les tests sont collectables
pytest --collect-only -q

# Vérifier la qualité du code
ruff check .
```

Résultat attendu : `All checks passed!` pour ruff, et la liste des tests pour pytest.

---

## Lancer les tests

```bash
# Tous les tests
pytest -q

# Avec rapport de couverture
pytest -q --cov=. --cov-report=term-missing

# Un dossier de tests spécifique
pytest tests/unit/ -q
pytest tests/agents/ -q
pytest tests/security/ -q
```

---

## Import des fixtures Synthea

Les 37 dossiers synthétiques (`CLM-0001` … `CLM-0037`) sont importés depuis Synthea vers `datasets/fixtures/valid/`.

```bash
# Prévisualiser sans écrire
python scripts/import_synthea_claimshield_cases.py --dry-run

# Importer tous les cas
python scripts/import_synthea_claimshield_cases.py

# Importer un seul cas
python scripts/import_synthea_claimshield_cases.py --case CLM-0001

# Écraser les cas modifiés (crée un backup avant)
python scripts/import_synthea_claimshield_cases.py --force

# Vérifier les fixtures existantes sans importer
python scripts/import_synthea_claimshield_cases.py --validate-only
```

---

## Lancer l'API et l'interface Chainlit

L'API expose le graphe compilé (`POST /claims`, `GET /claims/{case_id}`,
`POST /claims/{case_id}/human-decision`, `GET /healthz`). L'UI Chainlit
(`ui/`) est un client HTTP séparé de cette API — les deux se lancent
indépendamment (ports différents), l'UI a besoin de l'API déjà démarrée.

```bash
# 1. Démarrer l'API (nécessite Ollama lancé séparément — gemma4:latest)
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

# 2. Dans un autre terminal, démarrer l'UI Chainlit
CLAIMSHIELD_API_BASE_URL=http://127.0.0.1:8000 chainlit run ui/app.py -w
```

L'UI s'ouvre sur `http://localhost:8000` par défaut (Chainlit — port distinct
de l'API si les deux tournent en local, ajuster avec `--port`). Voir
`CLAIMSHIELD_API_BASE_URL`/`CLAIMSHIELD_API_KEY`/`CLAIMSHIELD_UI_UPLOAD_DIR`
dans `.env.example`.

---

## Déploiement Docker

Deux services (`api`, `ui`), même image. **Ollama reste sur l'hôte** —
aucun Ollama n'est embarqué dans un conteneur ; il doit déjà tourner
(`gemma4:latest` tiré) avant de démarrer les conteneurs.

```bash
# 1. Copier et adapter la config Docker (chemins /app/..., secrets à changer)
cp .env.docker.example .env.docker

# 2. Construire et démarrer
docker compose up --build
```

- API : `http://localhost:8000` (healthcheck sur `/healthz`)
- UI Chainlit : `http://localhost:8001`
- Persistance : backend de checkpoint `sqlite` (fichier `storage/checkpoints.db`,
  volume partagé avec l'hôte) — les dossiers en attente de revue humaine
  survivent à un redémarrage de conteneur (`docker compose restart api`),
  vérifié manuellement.
- Fixtures de démo (`datasets/fixtures/`) montées en lecture seule dans le
  conteneur `api`.
- `api` et `ui` partagent le **même** volume `./storage` — un dossier soumis
  via l'UI (upload) doit être lisible par le conteneur `api` (voir
  `ui/uploads.py`).

---

## Structure du projet

```
claimshield-sante/
├── agents/                  # 11 agents spécialisés (stubs → implémentations)
│   ├── claim_intake_agent/
│   ├── security_gate_agent/
│   ├── document_ocr_agent/
│   ├── identity_coverage_agent/
│   ├── fhir_validator_agent/
│   ├── medical_coding_agent/
│   ├── clinical_consistency_agent/
│   ├── fraud_detection_agent/
│   ├── privacy_agent/
│   ├── case_reviewer_agent/
│   └── audit_agent/
├── graph/                   # Workflow LangGraph (nodes, edges, checkpoints)
├── orchestrator/            # Routage et politiques d'appel
├── state/                   # ClaimState partagé
├── schemas/                 # Modèles Pydantic communs
├── security/                # Allowlists, scanners, politiques
├── services/                # Stockage, audit, notifications
├── tools/                   # Fonctions déterministes (hash, OCR, parsing)
├── datasets/fixtures/       # 37 dossiers CLM-0001 … CLM-0037
├── tests/                   # Unitaires, agents, graph, sécurité, E2E
├── .env.example             # Modèle de configuration
├── requirements.txt         # Dépendances Python
└── pyproject.toml           # Métadonnées et configuration ruff/pytest
```

---

## Variables d'environnement clés

| Variable | Valeur par défaut | Rôle |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Serveur Ollama local |
| `CLAIMSHIELD_LLM_MODEL` | `gemma4:latest` | Modèle LLM utilisé |
| `DATABASE_URL` | `sqlite+aiosqlite:///./storage/claimshield.db` | Base de données (SQLite en dev) |
| `CLAIMSHIELD_MAX_FILE_SIZE_MB` | `20` | Taille max des fichiers déposés |

Voir `.env.example` pour la liste complète.

---

## Règles non négociables

- Aucune donnée réelle de patient.
- Aucun paiement automatique.
- Aucune accusation automatique de fraude.
- Toute recommandation finale requiert une validation humaine.

---

## Codes de sécurité stables

Les codes ci-dessous sont des identifiants techniques stables. Ils ne contiennent
aucune donnée médicale, aucun secret et aucun contenu de document. Seules les
descriptions peuvent évoluer.

### Audit minimal du Security Gate

Chaque décision du Security Gate embarque un `SecurityAuditEntry` minimal dans
le résultat Pydantic. Cet événement contient uniquement :

- `claim_id`
- horodatage `evaluated_at`
- acteur ou agent demandeur `actor`
- type d'entrée contrôlée `input_type`
- politique appliquée `policy_applied`
- version de politique `policy_version`
- décision `decision` / `outcome`
- codes stables `reason_codes`
- hash `file_sha256` si un fichier est concerné

L'audit minimal ne stocke jamais de document brut, texte OCR complet, prompt
système, token, clé API ou mot de passe.

### Seuils de sévérité

Les seuils sont déterministes, versionnés dans `SeverityPolicy` et appliqués par
du code Python. Aucun LLM ne choisit librement le niveau ou la décision.

| Niveau | Exemple | Décision |
|---|---|---|
| `LOW` | Élément inhabituel sans danger immédiat | `ALLOW` avec alerte |
| `MEDIUM` | Incohérence nécessitant vérification | `QUARANTINE` |
| `HIGH` | URL externe ou outil interdit | `BLOCK` |
| `CRITICAL` | Injection, accès secret, shell ou path traversal | `BLOCK` |

| Code | Sévérité | Description |
|---|---|---|
| `UNSUPPORTED_EXTENSION` | HIGH | Extension de fichier non autorisée. |
| `UNSUPPORTED_MIME` | HIGH | Type MIME détecté non autorisé. |
| `FILE_TOO_LARGE` | HIGH | Taille réelle du fichier supérieure à la limite. |
| `EMPTY_FILE` | HIGH | Fichier vide refusé. |
| `FILE_METADATA_INCOMPLETE` | HIGH | Métadonnées fichier insuffisantes. |
| `MIME_EXTENSION_MISMATCH` | MEDIUM | Incohérence entre extension et MIME détecté. |
| `PATH_TRAVERSAL` | CRITICAL | Tentative de traversée de répertoire. |
| `ABSOLUTE_PATH_FORBIDDEN` | CRITICAL | Chemin absolu interdit. |
| `PATH_NULL_BYTE` | CRITICAL | Caractère nul interdit dans un chemin. |
| `PATH_OUTSIDE_STORAGE` | CRITICAL | Chemin résolu hors de la racine storage. |
| `STORAGE_ZONE_FORBIDDEN` | HIGH | Zone de stockage non autorisée. |
| `EXTERNAL_URL_FORBIDDEN` | HIGH | URL externe refusée par défaut. |
| `PRIVATE_NETWORK_URL` | CRITICAL | URL vers localhost, loopback ou réseau privé. |
| `DANGEROUS_URL_SCHEME` | HIGH | Schéma d'URL dangereux ou non autorisé. |
| `MALFORMED_URL` | HIGH | URL absente ou malformée. |
| `URL_CREDENTIALS_FORBIDDEN` | HIGH | Identifiants présents dans l'URL. |
| `PROMPT_INJECTION_DETECTED` | CRITICAL | Tentative d'injection de prompt détectée. |
| `SECRET_ACCESS_ATTEMPT` | CRITICAL | Tentative d'accès à un secret. |
| `SHELL_ACCESS_ATTEMPT` | CRITICAL | Tentative d'accès shell, terminal ou commande. |
| `UNAUTHORIZED_TOOL` | CRITICAL | Outil ou agent demandeur non autorisé. |
| `WRITE_PATH_FORBIDDEN` | HIGH | Écriture demandée hors des zones autorisées. |
| `INVALID_AGENT_OUTPUT` | HIGH | Sortie d'agent invalide ou dangereuse. |
| `SUSPICIOUS_DOCUMENT_CONTENT` | HIGH | Contenu documentaire suspect. |
| `SUSPICIOUS_CONTENT` | MEDIUM | Contenu suspect non classé plus précisément. |
| `POLICY_VIOLATION` | MEDIUM | Violation générique de politique de sécurité. |
