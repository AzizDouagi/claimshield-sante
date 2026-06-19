# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

ClaimShield Santé is a multi-agent health insurance claims processing system built in Python. The codebase is in early development: the **dataset tooling is the only implemented layer**; every other module (`agents/`, `graph/`, `orchestrator/`, `state/`, `api/`, `app/`, etc.) exists as an empty scaffold awaiting implementation. All code, comments, and user-facing text are in **French**.

## Dataset pipeline

The synthetic data pipeline flows from [Synthea](https://github.com/synthetichealth/synthea) through two stages:

**Stage 1 — Case extraction** (run from the Synthea output directory):
```bash
# Inspect what Synthea produced
python tools/dataset_builder/inspect_claimshield_data.py

# Extract the first matching patient/claim into CLM-0001 case_data.json
python tools/dataset_builder/select_first_case.py

# Generate PDFs (demande_remboursement, facture, ordonnance) + ground_truth.json + manifest.json
python tools/dataset_builder/generate_case_documents.py
```

**Stage 2 — Import into the project fixtures** (run from the project root):
```bash
python scripts/import_synthea_claimshield_cases.py               # import all cases
python scripts/import_synthea_claimshield_cases.py --dry-run      # preview without writing
python scripts/import_synthea_claimshield_cases.py --case CLM-0001  # single case
python scripts/import_synthea_claimshield_cases.py --force        # overwrite changed cases (creates backup first)
python scripts/import_synthea_claimshield_cases.py --validate-only # check existing fixtures
```

The importer reads from an external Synthea directory (hardcoded in `ClaimshieldImporter.__init__`) and writes into `datasets/fixtures/valid/`. It computes SHA-256 content signatures to skip unchanged cases and writes a structured report to `datasets/fixtures/metadata/import_report.json`.

## Fixture layout

Each case under `datasets/fixtures/valid/CLM-XXXX/` follows this layout:

```
CLM-XXXX/
  input/
    demande_remboursement_CLM-XXXX.pdf   # claim request
    facture_CLM-XXXX.pdf                 # medical invoice
    ordonnance_CLM-XXXX.pdf              # prescription
    patient_fhir_bundle.json             # FHIR R4 patient record
  oracle/
    case_data.json     # raw Synthea data used to generate documents
    ground_truth.json  # expected agent outputs (recommendation, anomalies, extracted fields)
  audit/
    manifest.json      # file inventory with SHA-256 hashes + financial summary
```

`ground_truth.json` is the test oracle: it specifies `expected_recommendation` (APPROVE/REJECT), `expected_anomalies`, `required_documents`, `deterministic_rules`, and `expected_extraction` field values. New agent implementations must produce output that matches these.

## Planned architecture (to be implemented)

The intended processing pipeline (all files currently empty stubs):

1. **`state/claim_state.py`** — shared `ClaimState` TypedDict passed through the LangGraph graph
2. **`graph/workflow.py`** — LangGraph `StateGraph` definition; `nodes.py` wraps agent calls; `edges.py` defines conditional routing; `checkpoints.py` handles persistence
3. **`orchestrator/`** — higher-level routing policies and orchestrator entry point
4. **`agents/<name>/agent.py`** — one agent per subdirectory; `schemas.py` holds Pydantic input/output models

The 11 planned agents and their roles:

| Agent | Responsibility |
|---|---|
| `security_gate_agent` | Prompt injection / malicious input detection |
| `claim_intake_agent` | Document completeness check |
| `document_ocr_agent` | Extract text from PDFs |
| `fhir_validator_agent` | Validate FHIR bundle structure |
| `identity_coverage_agent` | Patient identity + insurance coverage verification |
| `medical_coding_agent` | ICD/procedure code validation |
| `clinical_consistency_agent` | Cross-document clinical coherence |
| `fraud_detection_agent` | Duplicate, anomaly, and fraud signals |
| `privacy_agent` | PHI handling and anonymisation |
| `case_reviewer_agent` | Final recommendation synthesis |
| `audit_agent` | Audit trail and logging |

Additional planned layers: `api/` (REST/async API), `app/` (application entrypoint), `human_review/` (human-in-the-loop UI), `mcp_servers/` (MCP tool integrations), `prompts/` (prompt templates), `schemas/` (shared Pydantic models), `services/` (business logic), `database/` (persistence), `security/` (auth/authz).

## Dependencies

`reportlab` and `pypdf` are used by the dataset builder. The `pyproject.toml` is currently empty; dependencies are not yet declared in the project.

## Environment

Copy `.env.example` to `.env` and set the Synthea paths before running the pipeline tools:
```
SYNTHEA_ROOT=<path to synthea repo>
SYNTHEA_OUTPUT_DIR=<path to synthea output>
CLAIMSHIELD_DATASETS_DIR=./datasets
CLAIMSHIELD_STORAGE_DIR=./storage
CLAIMSHIELD_INBOX_DIR=./storage/inbox
CLAIMSHIELD_QUARANTINE_DIR=./storage/quarantine
```

The `storage/` directory (inbox, processed, quarantine, rejected) is gitignored and represents live claim document storage, separate from the test fixtures in `datasets/`.
