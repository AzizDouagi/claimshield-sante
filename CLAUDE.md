# CLAUDE.md

dans chaque réponse appele moi toujours par mon prènom: AZIZ

**Commits git** : ne jamais ajouter de ligne `Co-Authored-By` dans les messages de commit. Seul le nom d'Aziz Douagi doit apparaître comme auteur sur GitHub.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT — keep this file up to date.** Whenever a file gains a real implementation (replacing an empty stub), update the corresponding entry below.

## Project overview

ClaimShield Santé is a multi-agent health insurance claims processing system built in Python. All code, comments, and user-facing text are in **French**. The language is LangGraph. The LLM local utilisé est **Ollama gemma4:latest** (aucune clé API requise).

Le projet avance par étapes validées (feuille de route 18 étapes). État actuel : **Étape 2 en cours**.

Couches par état d'avancement :

- **Implémenté** — outillage dataset/fixtures, configuration, schémas Pydantic.
- **Stubs vides** — agents, graphe LangGraph, orchestrateur, state, API, app.

---

## Répertoire des fichiers et dossiers

### Racine

| Fichier | Statut | Rôle |
|---|---|---|
| `.env` | implémenté (gitignored) | Variables locales de développement — ne jamais committer |
| `.env.example` | implémenté | Modèle public : Ollama, API, DB, stockage, audit |
| `requirements.txt` | implémenté | Toutes les dépendances Python (voir section Dépendances) |
| `pyproject.toml` | implémenté | Métadonnées, Python ≥ 3.11, ruff, pytest |
| `README.md` | implémenté | Installation, tests, import Synthea, structure du projet |
| `docker-compose.yml` | **stub vide** | À compléter pour lancer les services (DB, API, etc.) |

### `config/`

| Fichier | Statut | Rôle |
|---|---|---|
| `config/settings.py` | implémenté | `Settings(BaseSettings)` via pydantic-settings — lit `.env`, expose toutes les variables typées. Singleton via `@lru_cache`. Pas de chemin absolu personnel. |

Usage :
```python
from config.settings import get_settings
s = get_settings()
s.claimshield_llm_model   # "gemma4:latest"
s.ollama_base_url          # "http://localhost:11434"
s.datasets_dir             # Path("datasets")
```

### `schemas/` — Modèles Pydantic partagés

| Fichier | Statut | Rôle |
|---|---|---|
| `schemas/domain.py` | implémenté | Modèles métier : `ClaimSubmission`, `PatientInfo`, `CoverageInfo`, `ExtractedData`, `MedicalProcedure`, `Prescription`, `DeterministicRules` + enums (`Recommendation`, `VerificationStatus`, `SecurityDecision`, `DataClassification`, `AuthorizationStatus`). `extra="forbid"` sur tous. |
| `schemas/results.py` | implémenté | Un schéma de sortie par agent (11 au total) : `ClaimIntakeResult`, `SecurityGateResult`, `PrivacyResult`, `IdentityCoverageResult`, `FhirValidatorResult`, `DocumentOcrResult`, `MedicalCodingResult`, `ClinicalConsistencyResult`, `FraudDetectionResult`, `CaseReviewerResult`, `AuditEvent`. |

Règles communes à tous les schémas :
- `extra="forbid"` — tout champ inconnu lève une `ValidationError`
- `validate_assignment=True` — mutations invalides détectées immédiatement
- Montants en `Decimal` (jamais `float`)

### `tools/dataset_builder/` — Outils de génération (Stage 1 Synthea)

Ces scripts s'exécutent **depuis le répertoire Synthea**.

| Fichier | Rôle |
|---|---|
| `inspect_claimshield_data.py` | Lit les CSV Synthea et affiche un résumé : patients vivants, demandes valides, actes, médicaments. |
| `select_first_case.py` | Sélectionne la demande la plus récente (patient vivant, ≥ 1 acte ET 1 médicament). Écrit `case_data.json` et copie le bundle FHIR. |
| `generate_case_documents.py` | Génère `medical_invoice.pdf`, `prescription.pdf`, `claim_request.pdf`, `ground_truth.json`, `manifest.json`. Couverture 80 %. Pré-autorisation si total ≥ 3 000 USD ou > 5 actes. |

### `scripts/` — Import dans les fixtures (Stage 2)

Ces scripts s'exécutent **depuis la racine du projet**.

| Fichier | Rôle |
|---|---|
| `import_synthea_claimshield_cases.py` | Importe depuis `CLAIMSHIELD_SOURCE_ROOT` vers `datasets/fixtures/valid/`. Renomme les PDFs, calcule SHA-256, saute les cas inchangés. Rapport dans `datasets/fixtures/metadata/import_report.json`. |

```bash
python scripts/import_synthea_claimshield_cases.py               # importer tous les cas
python scripts/import_synthea_claimshield_cases.py --dry-run      # prévisualiser sans écrire
python scripts/import_synthea_claimshield_cases.py --case CLM-0001
python scripts/import_synthea_claimshield_cases.py --force        # écraser (backup avant)
python scripts/import_synthea_claimshield_cases.py --validate-only
```

### `datasets/` — Données de test

```
datasets/fixtures/
  valid/              # 37 dossiers CLM-0001 … CLM-0037
    CLM-XXXX/
      input/
        demande_remboursement_CLM-XXXX.pdf
        facture_CLM-XXXX.pdf
        ordonnance_CLM-XXXX.pdf
        patient_fhir_bundle.json
      oracle/
        case_data.json      # données Synthea brutes
        ground_truth.json   # oracle : expected_recommendation, expected_anomalies,
                            #          expected_extraction, deterministic_rules,
                            #          expected_security, expected_privacy,
                            #          expected_identity, expected_coverage,
                            #          expected_fhir, expected_clinical_consistency,
                            #          expected_fraud
      audit/
        manifest.json       # inventaire SHA-256 + résumé financier
  backups/
  metadata/
    import_report.json
    index.json
    generation_report.json
    generation.log
```

`ground_truth.json` est l'oracle principal. Les schémas de `schemas/results.py` sont calqués sur sa structure.

### `agents/` — Agents LLM (stubs vides)

Chaque sous-dossier contient `__init__.py`, `agent.py`, `schemas.py`, `README.md` — tous vides.

| Agent | Responsabilité | Schéma de sortie |
|---|---|---|
| `security_gate_agent` | Prompt injection, entrées malveillantes | `SecurityGateResult` |
| `claim_intake_agent` | Complétude documentaire, hashes | `ClaimIntakeResult` |
| `document_ocr_agent` | Extraction PDF/image avec provenance | `DocumentOcrResult` |
| `fhir_validator_agent` | Validation bundle FHIR R4 | `FhirValidatorResult` |
| `identity_coverage_agent` | Identité patient + couverture | `IdentityCoverageResult` |
| `medical_coding_agent` | Codes ICD/actes médicaux | `MedicalCodingResult` |
| `clinical_consistency_agent` | Cohérence clinique | `ClinicalConsistencyResult` |
| `fraud_detection_agent` | Doublons, anomalies, fraude | `FraudDetectionResult` |
| `privacy_agent` | Vues minimisées par rôle | `PrivacyResult` |
| `case_reviewer_agent` | Synthèse + recommandation APPROVE/REJECT | `CaseReviewerResult` |
| `audit_agent` | Journal append-only | `AuditEvent` |

### `graph/` — Graphe LangGraph (stubs vides)

| Fichier | Rôle prévu |
|---|---|
| `workflow.py` | `StateGraph` LangGraph — point d'entrée du pipeline |
| `nodes.py` | Fonctions nœuds wrappant les appels agents |
| `edges.py` | Routage conditionnel |
| `checkpoints.py` | Persistance SQLite → PostgreSQL |

### `orchestrator/` — Orchestrateur (stubs vides)

| Fichier | Rôle prévu |
|---|---|
| `orchestrator.py` | Registre agents, validation sorties, gestion erreurs |
| `policies.py` | Politiques de routage et règles métier |
| `routing.py` | Sélection agents selon contexte |

### `state/claim_state.py` — État partagé (stub vide)

Contiendra le `ClaimState` TypedDict passé à travers tout le graphe LangGraph. Les champs seront alignés sur `schemas/results.py`.

### Dossiers planifiés (vides)

| Dossier | Rôle prévu |
|---|---|
| `api/` | API REST FastAPI |
| `app/` | Point d'entrée applicatif |
| `human_review/` | Interface HITL |
| `mcp_servers/` | Adaptateurs MCP |
| `prompts/` | Templates de prompts |
| `services/` | Stockage, audit, notifications |
| `database/` | Modèles SQLAlchemy + migrations Alembic |
| `security/` | Allowlists, scanners, politiques RBAC/ABAC |

---

## Dépendances installées (`requirements.txt`)

| Groupe | Packages principaux |
|---|---|
| LangGraph | `langgraph>=1.0.0`, `langgraph-checkpoint>=4.1.0`, `langchain-core>=1.4.0`, `langchain-ollama>=0.3.0` |
| LLM local | `ollama>=0.4.0` |
| Pydantic | `pydantic>=2.7.4`, `pydantic-settings>=2.5.0` |
| API | `fastapi>=0.115.0`, `uvicorn[standard]`, `python-multipart` |
| Documents | `pypdf>=5.0`, `reportlab>=4.0`, `Pillow>=11.0` |
| Sécurité | `python-magic>=0.4.27`, `cryptography>=43.0` |
| FHIR + règles | `fhir.resources>=8.0.0`, `PyYAML>=6.0` |
| Fraude | `rapidfuzz>=3.10.0` |
| Persistance | `sqlalchemy>=2.0`, `aiosqlite`, `asyncpg`, `alembic>=1.14.0` |
| Observabilité | `structlog>=24.4.0`, `tenacity>=9.0.0` |
| MCP | `mcp>=1.0.0` |
| Tests | `pytest>=8.0`, `pytest-cov`, `pytest-asyncio`, `httpx`, `Faker` |
| Qualité | `ruff>=0.8.0` |

## Environnement

Variables principales (voir `.env.example` pour la liste complète) :

```
OLLAMA_BASE_URL=http://localhost:11434
CLAIMSHIELD_LLM_MODEL=gemma4:latest
CLAIMSHIELD_LLM_PROVIDER=ollama
DATABASE_URL=sqlite+aiosqlite:///./storage/claimshield.db
LANGGRAPH_CHECKPOINT_DB=./storage/checkpoints.db
SYNTHEA_ROOT=/path/to/synthea
CLAIMSHIELD_SOURCE_ROOT=/path/to/synthea/claimshield_cases
```

La configuration est lue exclusivement via `config/settings.py` — aucun `os.getenv` direct dans le code des agents.
