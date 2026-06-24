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

# 4. Installer les dépendances Python
pip install -r requirements.txt

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
