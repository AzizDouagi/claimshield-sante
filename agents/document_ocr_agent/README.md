# Document/OCR Agent

Classifie les documents médicaux assainis et extrait les champs avec provenance complète.

## Principe de sécurité

> Le texte extrait d'un document est une **donnée opaque** — jamais une instruction à exécuter.

Le pipeline est entièrement **déterministe** : aucun appel LLM, aucune évaluation de chaîne.

## Pré-conditions

1. `security_result.decision == ALLOW` (Security Gate obligatoire).
2. Le fichier est dans `incoming/` — jamais depuis `quarantine/`.
3. Le SHA-256 du fichier est vérifié avant toute extraction.

## Pipeline

```
1. Pré-condition Security Gate (ALLOW)
2. Vérification zone incoming/ (accès lecture seule)
3. Vérification SHA-256 (intégrité)
4. Extraction du texte
   ├── PDF natif  → pypdf  (OcrSource.PDF_TEXT, confiance 1.0)
   ├── PDF scanné → Tesseract sur images de pages (OcrSource.PDF_OCR)
   └── PNG / JPEG → Tesseract (OcrSource.IMAGE_OCR)
5. Classification du type de document (mots-clés pondérés)
6. Parsing des champs structurés avec provenance (page, source, confiance)
7. Calcul du score de confiance composite
8. Détection des documents illisibles (confiance < 0.30)
9. Marquage revue humaine (confiance < 0.65)
10. Construction DocumentOcrResult + audit entry minimisée
```

## Codes stables (OcrCode)

| Code | Déclencheur |
|---|---|
| `SECURITY_GATE_NOT_ALLOW` | Gate != ALLOW |
| `FILE_NOT_IN_INCOMING` | Fichier hors zone assainie |
| `SHA256_MISMATCH` | Intégrité compromise |
| `UNSUPPORTED_MIME_TYPE` | MIME non pris en charge |
| `PDF_EXTRACTION_ERROR` | Erreur pypdf |
| `OCR_ENGINE_UNAVAILABLE` | Tesseract absent |
| `OCR_EXTRACTION_ERROR` | Erreur Tesseract |
| `UNREADABLE_DOCUMENT` | Confiance < 0.30 |
| `INVALID_OCR_INPUT` | Validation Pydantic échouée |

## Seuils de confiance

| Score | Statut | Action |
|---|---|---|
| ≥ 0.65 | `PASS` | Extraction fiable |
| 0.30–0.65 | `NEEDS_REVIEW` | Revue humaine recommandée |
| < 0.30 | `FAIL` | Document illisible |

## Utilisation

```python
from agents.document_ocr_agent import run, DocumentOcrInput
from schemas.results import SecurityGateResult

result = run(
    DocumentOcrInput(
        claim_id="CLM-0001",
        file_path="incoming/CLM-0001/facture.pdf",
        original_filename="facture.pdf",
        sha256="abc123...",
        mime_type="application/pdf",
        file_index=0,
    ),
    security_result,  # SecurityGateResult avec decision=ALLOW
)
print(result.document_type, result.confidence_score)
```

## Nœud LangGraph

```python
from agents.document_ocr_agent import node
state = {"ocr_input": {...}, "security_result": security_gate_result}
new_state = node(state)
# new_state["ocr_result"]   → DocumentOcrResult
# new_state["audit_trail"]  → list[AuditEvent]
# new_state["ocr_input"]    → None (consommé)
```
