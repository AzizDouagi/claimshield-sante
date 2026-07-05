# Représentation Mermaid du workflow — graph/workflow.py

Exemple généré via `graph.workflow.get_workflow_mermaid()`, qui appelle
`CompiledStateGraph.get_graph().draw_mermaid()` sur un workflow compilé avec
`interrupt_before=[]` et les agents futurs (`clinical_consistency`,
`fraud_detection`, `case_reviewer`, `audit`) en stub par défaut.

Aucune donnée sensible : uniquement des noms de nœuds et de routes
(`continue`, `end`, etc.), jamais de contenu de dossier, de secret ou de
donnée patient.

## Régénérer cet exemple

```python
from graph.workflow import get_workflow_mermaid

print(get_workflow_mermaid())
```

## Exemple

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	claim_intake(claim_intake)
	security_gate(security_gate)
	privacy(privacy)
	document_ocr(document_ocr)
	fhir_validator(fhir_validator)
	identity_coverage(identity_coverage)
	medical_coding(medical_coding)
	clinical_consistency(clinical_consistency)
	fraud_detection(fraud_detection)
	case_reviewer(case_reviewer)
	audit(audit)
	quarantine(quarantine)
	needs_review(needs_review)
	await_human_review(await_human_review)
	failure(failure)
	finalize(finalize)
	__end__([<p>__end__</p>]):::last
	__start__ --> claim_intake;
	audit --> finalize;
	await_human_review -. &nbsp;end&nbsp; .-> __end__;
	await_human_review -.-> claim_intake;
	await_human_review -.-> document_ocr;
	await_human_review -.-> failure;
	await_human_review -.-> fhir_validator;
	await_human_review -.-> identity_coverage;
	await_human_review -.-> medical_coding;
	await_human_review -.-> privacy;
	await_human_review -.-> security_gate;
	case_reviewer -. &nbsp;end&nbsp; .-> audit;
	case_reviewer -.-> failure;
	case_reviewer -.-> needs_review;
	claim_intake -.-> failure;
	claim_intake -.-> quarantine;
	claim_intake -. &nbsp;continue&nbsp; .-> security_gate;
	clinical_consistency --> fraud_detection;
	document_ocr -.-> failure;
	document_ocr -. &nbsp;continue&nbsp; .-> fhir_validator;
	document_ocr -.-> needs_review;
	fhir_validator -.-> failure;
	fhir_validator -. &nbsp;continue&nbsp; .-> identity_coverage;
	fhir_validator -.-> needs_review;
	fraud_detection --> case_reviewer;
	identity_coverage -.-> failure;
	identity_coverage -. &nbsp;continue&nbsp; .-> medical_coding;
	identity_coverage -.-> needs_review;
	medical_coding -. &nbsp;continue&nbsp; .-> clinical_consistency;
	medical_coding -.-> failure;
	medical_coding -.-> needs_review;
	needs_review --> await_human_review;
	privacy -. &nbsp;continue&nbsp; .-> document_ocr;
	privacy -.-> failure;
	security_gate -.-> failure;
	security_gate -. &nbsp;continue&nbsp; .-> privacy;
	security_gate -.-> quarantine;
	failure --> __end__;
	finalize --> __end__;
	quarantine --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

## Lecture rapide

- Les flèches en pointillés (`-.->`) sont des arêtes conditionnelles (routage
  selon le résultat d'un agent) ; les flèches pleines (`-->`) sont des arêtes
  normales.
- `await_human_review` pointe vers les 7 nœuds agents amont
  (`RELAUNCH_TARGETS`, dans `graph/edges.py`) : c'est la route de relance
  (« relancer ») déclenchée par une décision humaine `NEEDS_MORE_INFO`.
- Tout chemin se termine par `__end__`, directement ou via `failure`.
