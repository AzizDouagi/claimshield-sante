# Audit Agent & journal d'audit — exemples courts

Documentation d'usage rapide de `agents/audit_agent/`, `schemas/audit.py`,
`services/audit_store.py` et `tools/audit_redaction.py`. Pour le détail
complet (interdictions, schéma LLM, erreurs fréquentes), voir
`agents/audit_agent/README.md`.

## Enregistrer un événement

```python
from schemas.audit import AuditEventType, RedactionStatus
from services.audit_store import AuditStore

store = AuditStore()
event = store.record_event(
    case_id="CLM-1001",
    event_type=AuditEventType.SECURITY_DECISION,
    actor="security_gate_agent",
    outcome="BLOCK",
    redaction_status=RedactionStatus.PARTIALLY_REDACTED,
)
print(event.event_hash)  # empreinte SHA-256 du contenu canonique
```

`record_event` calcule lui-même `previous_hash`/`event_hash` — jamais fourni
par l'appelant. `AuditStore` est injectable (aucune instance globale
cachée) : à instancier explicitement, comme `DuplicateIndex`/`ModelRegistry`.

## Vérifier l'intégrité d'un dossier

```python
report = store.verify_claim_integrity("CLM-1001")
print(report.intact)   # True si aucune anomalie
print(report.issues)   # tuple[IntegrityIssue, ...] sinon (chaîne rompue,
                        # contenu falsifié, ordre incohérent)
```

## Exporter pour un auditeur externe

```python
# JSON — un seul objet, filtrable par case_id
json_export = store.export_to_json(case_id="CLM-1001")

# JSON Lines — une ligne d'en-tête (résumé + anomalies) puis une ligne par
# événement, adapté à un pipeline d'ingestion externe (SIEM, entrepôt de logs)
jsonl_export = store.export_to_jsonl(case_id="CLM-1001")
```

`case_id=None` exporte tous les dossiers connus. L'export ne contient que
des champs déjà validés par `AuditEvent` (`extra="forbid"`, secrets/chemins
rejetés, champs bornés en longueur) — jamais de donnée brute ou excessive.

## Rédaction avant tout appel LLM

`agents/audit_agent/agent.py::_invoke_llm_audit` rédige systématiquement
l'événement soumis au LLM normalizer via `tools/audit_redaction.py`, quelle
que soit la provenance (Security Gate, orchestrateur, human_review, l'Audit
Agent lui-même) :

```python
from tools.audit_redaction import redact_audit_payload

payload = {"case_id": "CLM-1001", "system_prompt": "…", "notes": "api_key=sk-123"}
redacted = redact_audit_payload(payload)
# redacted == {"case_id": "CLM-1001", "redaction_status": "fully_redacted"}
```

Le `redaction_status` réellement calculé sert de plancher : le LLM ne peut
jamais déclarer un statut plus faible que ce qui a été effectivement retiré.

## Tests

```bash
pytest tests/audit -q
```

Couvre : schéma chaîné (`test_schemas.py`), journal append-only et export
(`test_store.py`), persistance SQLAlchemy optionnelle (`test_models.py`),
rédaction déterministe (`test_redaction.py`), câblage LLM de bout en bout
(`test_agent.py`).
