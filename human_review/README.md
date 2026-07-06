# `human_review/` — Service HITL (Human-In-The-Loop)

## Rôle

Package **indépendant de LangGraph** qui prépare la couche de revue humaine :
modèles de décision (`models.py`), service de payload/validation/audit
(`service.py`), vue formulaire prête à afficher (`views.py`).

**Câblé dans le graphe de production.** `graph/technical_nodes.py::node_await_human_review`
délègue désormais entièrement sa validation à
`human_review.service.validate_and_audit_human_decision` — même modèle
Pydantic strict (`HumanDecision`, `extra="forbid"`, `justification`
obligatoire), mêmes 4 actions (`APPROVE`/`MODIFY`/`REJECT`/`RETRY` —
`NEEDS_MORE_INFO` a été renommé en `RETRY` pour cet alignement), même audit
(`AuditEvent` ajouté à `state["audit_trail"]`). `case_id` est complété
automatiquement depuis le state si absent du payload de reprise (déjà connu
du graphe). `views.py` (formulaire, adaptateurs FastAPI/Chainlit) reste la
seule partie de ce module pas encore câblée — voir « Limites explicites ».

## Fichiers

| Fichier | Rôle |
|---|---|
| `models.py` | `ReviewAction` (enum) et `HumanDecision` (`StrictModel`, `extra="forbid"`) — le contrat de décision humaine. |
| `service.py` | `build_human_review_payload()` (payload minimal), `validate_human_decision()` (validation stricte), `build_human_decision_audit_event()`/`validate_and_audit_human_decision()` (audit minimal). |
| `views.py` | `HumanReviewFormView`/`EditableField` (formulaire prêt à afficher), `build_human_review_form()`, `submit_human_review_decision()`, adaptateurs `render_for_fastapi()`/`render_for_chainlit_actions()`. |
| `__init__.py` | vide, aucune logique. |

## Actions humaines (`ReviewAction`)

| Action | Libellé (FR) | `target_node` | Effet prévu |
|---|---|---|---|
| `APPROVE` | Valider | interdit | Pré-recommandation acceptée telle quelle — chemin terminal (audit puis clôture). |
| `MODIFY` | Modifier | interdit | Pré-recommandation acceptée avec une correction humaine (ex. montant) — la modification elle-même n'est **jamais** un champ de `HumanDecision` (schéma volontairement minimal et générique) : elle voyage à côté de la décision dans le payload de reprise brut, extraite par l'appelant *avant* validation, puis appliquée seulement *après* acceptation de la décision (justification obligatoire déjà garantie par le schéma). |
| `REJECT` | Refuser | interdit | Pré-recommandation rejetée — chemin terminal (audit puis échec contrôlé). |
| `RETRY` | Relancer | **obligatoire** | Reprise d'un nœud amont — pas un chemin terminal, boucle dans le pipeline. |

`HumanDecision.justification` est **obligatoire pour les quatre actions**
(`min_length=1`, `max_length=1000` — jamais un document entier) : aucune
action ne peut jamais être appliquée sans un motif humain explicite.
`target_node` est interdit pour `APPROVE`/`MODIFY`/`REJECT` et obligatoire
pour `RETRY` (`model_validator` de `HumanDecision`).

## Cycle interruption / reprise

Ce module ne dépend pas de LangGraph mais son usage prévu suit exactement le
même cycle que `graph/technical_nodes.py::node_await_human_review` :

```
build_human_review_payload(state) → interrupt(payload)
                                          │
                        (suspend le graphe, persiste via checkpointer)
                                          │
                       Command(resume=raw_decision), même thread_id
                                          │
                validate_human_decision(raw) / validate_and_audit_human_decision(raw, evidence_ids=...)
```

**Le `thread_id` de la reprise doit impérativement être identique** à celui
de l'invocation initiale — sinon LangGraph ne retrouve aucun checkpoint en
attente et redémarre une exécution indépendante depuis `START` (voir
`graph/checkpoints.py::assert_same_thread_id`,
`tests/human_review/test_interrupt_resume.py::TestResumeThreadId`).

`tests/human_review/test_interrupt_resume.py` construit un graphe LangGraph
**de test, minimal et autonome** (distinct du graphe de production) qui
exerce ce cycle complet avec les 4 `ReviewAction` — il prouve que le contrat
est prêt pour un câblage réel, sans modifier `graph/technical_nodes.py`.

## Exemples de payload

### Payload d'interruption (`HumanReviewPayload`, retourné par `build_human_review_payload`)

```json
{
  "case_id": "CLM-0001",
  "summary": ["Score de risque de fraude élevé (0.75).", "1 incohérence clinique détectée."],
  "evidence": {
    "identity_coverage_result": "PASS",
    "clinical_result": "NEEDS_REVIEW",
    "fraud_result": "NEEDS_REVIEW",
    "review_result": "NEEDS_REVIEW"
  },
  "options": ["APPROVE", "MODIFY", "REJECT", "RETRY"]
}
```

Ne contient jamais de document brut, de texte OCR complet, de secret ou de
prompt — uniquement des motifs déjà agrégés (`alerts`/`errors`) et des
statuts déjà publics des résultats d'agents (voir
`tests/human_review/test_service.py`/`test_views.py` : preuve explicite
qu'un `DocumentOcrResult.full_text` réel n'apparaît jamais dans ce payload).

### `HumanDecision` — APPROVE

```json
{
  "case_id": "CLM-0001",
  "actor": "reviewer@example.com",
  "action": "APPROVE",
  "justification": "Dossier conforme, toutes les pièces sont cohérentes.",
  "decided_at": "2026-07-06T14:32:10+00:00"
}
```

### `HumanDecision` — MODIFY (avec modification de montant hors schéma)

Le payload de reprise brut peut porter une clé `modification` **à côté** de
la décision — jamais validée par `HumanDecision` elle-même (`extra="forbid"`
la rejetterait), extraite par l'appelant avant d'appeler
`validate_human_decision()`/`validate_and_audit_human_decision()` :

```json
{
  "case_id": "CLM-0001",
  "actor": "reviewer@example.com",
  "action": "MODIFY",
  "justification": "Montant corrigé après vérification de la facture jointe.",
  "modification": {"amount_requested": "450.00"}
}
```

### `HumanDecision` — REJECT

```json
{
  "case_id": "CLM-0001",
  "actor": "reviewer@example.com",
  "action": "REJECT",
  "justification": "Preuves insuffisantes pour valider la couverture déclarée."
}
```

### `HumanDecision` — RETRY

```json
{
  "case_id": "CLM-0001",
  "actor": "reviewer@example.com",
  "action": "RETRY",
  "justification": "Pièce manquante — l'ordonnance doit être redemandée.",
  "target_node": "document_ocr"
}
```

### Rejet — justification absente (toutes actions)

```json
{"case_id": "CLM-0001", "actor": "reviewer@example.com", "action": "APPROVE"}
```

Lève `HumanDecisionValidationError` (code `HUMAN_DECISION_INVALID`) — aucune
action, y compris `APPROVE`, ne peut jamais être acceptée sans justification.

## Erreurs

| Exception / code | Déclencheur |
|---|---|
| `HumanDecisionValidationError` (code `HUMAN_DECISION_UNSTRUCTURED`) | Le payload de reprise n'est pas un mapping (texte libre, liste, `None`...) — jamais tenté en validation Pydantic. |
| `HumanDecisionValidationError` (code `HUMAN_DECISION_INVALID`) | Mapping invalide contre `HumanDecision` : action inconnue, justification absente/vide/trop longue, `target_node` manquant pour `RETRY` ou fourni hors `RETRY`, `case_id` hors pattern, champ inconnu. |
| Message d'erreur | Ne contient **jamais** la valeur brute fautive (`err['input']` ni `str(exc)` bruts) — uniquement les chemins de champs en erreur (`_sanitized_validation_error_fields`), une décision peut porter un commentaire libre potentiellement sensible. |

## Limites explicites

- **`views.py` pas encore câblé.** Le formulaire (`HumanReviewFormView`) et les adaptateurs FastAPI/Chainlit restent utilisables indépendamment (ex. future API), mais `node_await_human_review` ne les consomme pas — il construit son propre payload d'interruption (`graph/technical_nodes.py::_build_human_review_payload`).
- **`state.claim_state.HumanDecision`** (`TypedDict`) reflète désormais exactement `model_dump(mode="json")` du modèle Pydantic — ce n'est plus qu'une projection typée, jamais une validation indépendante.
- **`RETRY` n'est pas borné par `RELAUNCH_TARGETS`** ici : `HumanDecision` accepte n'importe quelle chaîne non vide comme `target_node`. Le contrôle métier (nœud réellement relançable et déjà exécuté) reste la responsabilité de `graph/edges.py::route_human_review`, appliqué uniquement une fois câblé.
- **Aucune persistance.** `build_human_decision_audit_event`/`validate_and_audit_human_decision` construisent un événement d'audit mais ne l'ajoutent jamais eux-mêmes à `state["audit_trail"]` ni à un stockage durable — à l'appelant de le faire (même convention que `Orchestrator.execute_agent()`).
- **`justification` bornée à 1000 caractères** au niveau du schéma, et tronquée à 500 caractères supplémentaires dans l'événement d'audit — jamais un document brut, un texte OCR complet ou un prompt entier n'y est jamais conservé.
- **`views.py` ne génère aucun HTML.** Uniquement des structures Pydantic/JSON — le rendu visuel reste hors périmètre.

## Point d'intégration audit — câblé

`graph/technical_nodes.py::node_await_human_review` appelle désormais
réellement `validate_and_audit_human_decision()` à chaque reprise :

```python
evidence_ids = _collect_decision_evidence_ids(state)  # depuis review_result.evidence_ids
decision, event = validate_and_audit_human_decision(raw_decision, evidence_ids=evidence_ids)
updates["human_decision"] = decision.model_dump(mode="json")
updates["audit_trail"] = [event]  # append, jamais un remplacement
```

L'événement (`schemas.results.AuditEvent`, interface append-only déjà
utilisée par tous les agents) trace exactement cinq éléments — `action`,
`justification` (tronquée), `actor`, `timestamp` (= `decision.decided_at`),
`evidence_ids` — jamais un document brut, un prompt complet ou un texte OCR
complet (voir `tests/human_review/test_interrupt_resume.py::TestMinimalStateNoRawContent`
et `tests/graph/test_technical_nodes.py::test_resume_produces_an_audit_event`).
Reste hors périmètre : la **persistance durable** de l'audit au-delà du
`ClaimState`/checkpoint (base de données dédiée, export) — non implémentée.

## Tests

| Fichier | Couverture |
|---|---|
| `tests/human_review/test_service.py` (39 tests) | Payload minimal, validation stricte, audit minimal (action/justification/auteur/horodatage/preuves), troncature, non-fuite de contenu brut. |
| `tests/human_review/test_views.py` (20 tests) | Formulaire (recommandation/preuves/alertes/risques/contradictions/champs), verrouillage `justification_required`, soumission, adaptateurs FastAPI/Chainlit. |
| `tests/human_review/test_interrupt_resume.py` (20 tests) | Cycle `interrupt()`/`Command(resume=...)` complet (APPROVE/MODIFY/REJECT/RETRY) sur un graphe de test autonome, reprise avec le même `thread_id`, auto-approbation interdite, state minimal sans contenu brut. |
