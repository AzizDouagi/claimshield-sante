# Audit Agent

L'Audit Agent transforme un evenement structure deja minimise en une trace
d'audit normalisee. Il passe obligatoirement par un LLM pour produire une
sortie JSON conforme au schema Pydantic attendu, puis delegue la persistance
au `AuditStore`.

Il ne prend aucune decision metier: il ne valide pas un remboursement, ne
juge pas la coherence clinique, ne conclut pas a une fraude et ne modifie pas
la recommandation finale.

## Role

- Normaliser un evenement de workflow (`agent_called`, `tool_called`,
  `human_decision`, etc.).
- Produire un resume minimal, factuel et non excessif.
- Identifier les anomalies explicitement visibles dans l'entree.
- Decrire les redactions ou minimisations appliquees.
- Classifier le niveau de donnees (`SYNTHETIC_TEST_DATA`, `ANONYMIZED`,
  `CONFIDENTIAL`).
- Retourner un `AuditResult` valide pour le `ClaimState`.

## Entrees

L'agent lit uniquement le `ClaimState` courant:

- `case_id`
- `current_step`
- `completed_steps`
- compteurs `errors` et `alerts`
- `final_recommendation`, si deja presente
- `human_decision`, si deja validee
- dernier evenement de `audit_trail`, si present
- synthese minimale des resultats agents deja disponibles

L'entree envoyee au LLM est compacte. Elle ne doit jamais contenir de document
brut, OCR complet, prompt complet, reponse brute LLM, secret, chemin absolu ou
donnee personnelle excessive.

## Sorties

Le LLM retourne une sortie stricte `LlmAuditNormalizedEvent`:

- `event_type`
- `actor`
- `outcome`
- `summary`
- `redaction_status`
- `classification`
- `anomalies`
- `redactions`
- `agent_name`, `tool_calls`, `evidence_ids`, `reasons`, `confidence_score`

L'agent retourne ensuite un `AuditResult` Pydantic valide:

- `status=PASS` si la normalisation LLM et la persistance reussissent
- `status=FAIL` si le LLM est indisponible, invalide, ou si la persistance est refusee
- `events_count` derive du `AuditStore`
- `events` contient seulement une trace legere autorisee pour `audit_trail`
- `llm_metadata` contient le modele, la version de prompt et la confiance

## Permissions

L'Audit Agent peut:

- appeler le LLM configure via `llm.factory.get_llm()`
- normaliser un evenement structure deja minimise
- appeler `AuditStore.record_event()`
- ajouter une trace legere au `ClaimState.audit_trail`
- produire un `AuditResult`

L'Audit Agent ne peut pas:

- ecrire directement dans une base ou un fichier d'audit
- calculer ou fournir lui-meme `previous_hash` ou `event_hash`
- modifier ou supprimer un evenement deja journalise
- enrichir l'evenement avec des donnees absentes
- prendre une decision metier

## Interdictions

Le prompt et les schemas interdisent:

- invention d'information absente de l'entree
- jugement metier ou conclusion clinique/fraude/remboursement
- prompts complets, messages LLM complets ou reponses brutes
- OCR complet, documents bruts, PDF/images/base64
- secrets, tokens, mots de passe, cles API
- chemins absolus ou traversal (`..`)
- donnees personnelles, medicales ou administratives excessives

## Append-Only

La persistance est append-only. L'agent ne dispose d'aucune methode
`update`, `delete`, `remove` ou `clear`.

Le point d'ecriture est `AuditStore.record_event()`. Cette methode construit
un nouvel evenement et l'ajoute au journal. Si une tentative reutilise un
`event_id` existant ou casse la chaine, l'ajout est refuse.

## Hash-Chain

Chaque evenement persistant `schemas.audit.AuditEvent` contient:

- `previous_hash`: hash de l'evenement precedent du meme dossier
- `event_hash`: SHA-256 du contenu canonique de l'evenement courant

Le premier evenement d'un dossier a `previous_hash=None`. Chaque evenement
suivant doit prolonger exactement le dernier `event_hash` connu. Cette chaine
permet de detecter une suppression, une insertion incorrecte ou une mutation
de contenu.

Le LLM ne calcule jamais ces hashes. `AuditStore` determine `previous_hash`,
`schemas.audit.build_audit_event()` calcule `event_hash`, puis `AuditStore`
verifie la continuite avant stockage.

## Minimisation

L'Audit Agent ne persiste pas de contenu brut. Il conserve seulement:

- un outcome court
- un resume minimal
- une classification
- des anomalies courtes
- des redactions courtes
- des identifiants de preuves, si disponibles
- des noms d'outils, si disponibles

La trace ajoutee a `ClaimState.audit_trail` est volontairement legere. Le
journal enrichi et chaine reste dans `AuditStore`.

## Export Auditeur

`AuditStore.export_for_auditor(case_id)` produit un instantane en lecture
seule pour un dossier. `case_id=None` exporte tous les dossiers connus.

L'export indique:

- `event_count`
- `chain_intact`
- `broken_at_event_id`, si la chaine est rompue
- la liste des evenements copies defensivement

L'export ne permet aucune modification du journal.

## Exemples

### agent_called

```json
{
  "event_type": "agent_called",
  "actor": "case_reviewer_agent",
  "outcome": "review_completed",
  "summary": "Le Case Reviewer a produit une pre-recommandation non finale.",
  "redaction_status": "fully_redacted",
  "classification": "CONFIDENTIAL",
  "anomalies": [],
  "redactions": ["Aucun contenu medical brut repris."],
  "agent_name": "case_reviewer_agent",
  "tool_calls": [],
  "evidence_ids": ["review_result"],
  "reasons": ["Trace d'appel agent normalisee."]
}
```

### tool_called

```json
{
  "event_type": "tool_called",
  "actor": "medical_coding_agent",
  "outcome": "tool_completed",
  "summary": "Un outil autorise de recherche de code a ete appele.",
  "redaction_status": "fully_redacted",
  "classification": "CONFIDENTIAL",
  "anomalies": [],
  "redactions": ["Description clinique detaillee non reprise."],
  "agent_name": "medical_coding_agent",
  "tool_calls": ["rechercher_code"],
  "evidence_ids": ["coding_result"],
  "reasons": ["Trace d'appel outil normalisee."]
}
```

### human_decision

```json
{
  "event_type": "human_decision",
  "actor": "reviewer@example.com",
  "outcome": "human_review_recorded",
  "summary": "Une decision humaine validee a ete rattachee au dossier.",
  "redaction_status": "partially_redacted",
  "classification": "CONFIDENTIAL",
  "anomalies": [],
  "redactions": ["Justification detaillee minimisee."],
  "agent_name": "audit_agent",
  "tool_calls": [],
  "evidence_ids": ["human_decision"],
  "reasons": ["Decision humaine journalisee sans jugement metier additionnel."]
}
```
