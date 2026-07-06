# Case Reviewer Agent

## Rôle

Synthétise les résultats déjà produits par les 9 agents amont (`claim_intake`,
`security_gate`, `privacy`, `identity_coverage`, `fhir_validator`,
`document_ocr`, `medical_coding`, `clinical_consistency`, `fraud_detection`)
en une pré-recommandation explicable — jamais une décision finale. Ne
recalcule jamais l'identité, la couverture, la codification, la cohérence
clinique ou le risque de fraude : il les combine.

**Ce que cet agent ne fait jamais** : il n'approuve ni ne rejette jamais un
dossier de façon définitive, n'autorise jamais un paiement, ne pose jamais
de diagnostic médical et ne déclare jamais une fraude avérée. Il produit une
synthèse — recommandation, preuves citées, risques reconnus, contradictions
reconnues et questions pour l'humain — toujours soumise à revue humaine
(HITL) avant toute conséquence.

## Pipeline

1. **Phase A (déterministe)** :
   - `_build_agent_snapshot()` — résumé sûr des 9 résultats amont (statuts,
     compteurs, indicateurs), jamais de contenu métier brut ;
   - `_collect_disagreements()` — désaccords de statut entre agents
     (`tools.consistency.detect_result_disagreements`) ;
   - `_collect_risks()` — risques dérivés du snapshot (plafond dépassé,
     pré-autorisation requise, score de fraude ≥ 0.7, doublon, incohérences
     cliniques) — jamais une affirmation inventée ;
   - `_collect_evidence_ids()` — agrège les identifiants de preuves déjà
     validées par `clinical_result`/`fraud_result` ;
   - `_deterministic_pre_recommendation()` — borne une pré-recommandation
     (REJECT si blocage/erreur, PENDING si désaccord/résultat manquant/revue
     demandée en amont, APPROVE seulement si tout est compatible) — **ne
     remplace jamais le LLM, seulement encadre sa sortie.**
2. **Phase B (LLM, appel obligatoire à chaque exécution)** —
   `llm.with_structured_output(LlmCaseReviewDecision, method="json_schema")`.
   Reçoit le snapshot, les désaccords, les risques et les identifiants de
   preuves déjà calculés ; produit une synthèse citant uniquement ces
   éléments (jamais une invention). Ne produit jamais de verdict — voir
   « Interdictions » ci-dessous.
3. **Phase C** — `_merge_llm_decision()` fusionne la sortie LLM dans la
   justification et les motifs de revue humaine (jamais dans `status` ni
   `human_review_required`, verrouillés au niveau du schéma) ;
   `_merge_recommendation()` ne permet au LLM que de durcir une
   pré-recommandation déterministe, jamais de l'assouplir (un REJECT
   déterministe reste REJECT quoi que dise le LLM).

## Outils autorisés

Aucun. Cet agent n'a pas de `tools.py` — il ne fait qu'appeler le LLM en
`with_structured_output` sur un résumé déjà minimisé
(`orchestrator.policies.ALLOWED_TOOLS_PER_AGENT` ne contient aucune entrée
pour `AgentName.CASE_REVIEWER`, même convention que `claim_intake_agent`/
`security_gate_agent`/`privacy_agent`/`fhir_validator_agent`).

## Permissions (accès au state)

| Ressource | Accès |
|---|---|
| `state["intake_result"]`, `["security_result"]`, `["privacy_result"]`, `["identity_coverage_result"]`, `["fhir_result"]`, `["ocr_result"]`, `["coding_result"]`, `["clinical_result"]`, `["fraud_result"]` | lecture seule — uniquement statuts/compteurs/indicateurs déjà publics des schémas Pydantic, jamais un champ brut |
| `state["errors"]` / `state["alerts"]` | lecture seule — comptés, jamais reproduits verbatim au LLM |
| LLM (`gemma4:latest` via Ollama) | appel obligatoire à chaque exécution effective ; ne reçoit qu'un résumé déjà minimisé (statuts, compteurs, risques, désaccords, identifiants de preuve), jamais un document brut ni une donnée patient |
| `ClaimState` | écrit uniquement `review_result`, `final_recommendation`, `final_justification`, `current_step`, `completed_steps`, `audit_trail`, et conditionnellement `errors`/`alerts` |

### Traçabilité d'audit (`AuditEvent`, `state["audit_trail"]`)

Chaque exécution ajoute un `AuditEvent` (`action="case_review"`) dont
`details` porte : `status` (toujours `"NEEDS_REVIEW"` — confirme visiblement
qu'aucune décision finale automatique n'a eu lieu), `human_review_required`,
`disagreement_count`/`justification_count`/`risk_count`, `evidence_ids`
(preuves amont réellement citées, jointes par une virgule), `llm_call_id`
(UUID unique par exécution — jamais réutilisé), `model_name`/`prompt_version`
(depuis `result.llm_trace`), `errors` (codes, vide si aucun). Voir
`tests/agents/test_case_reviewer_llm.py::test_audit_event_carries_llm_traceability_fields_and_status`.

## Preuves, risques et contradictions attendus

| Élément | Champ (`CaseReviewerResultPayload`) | Origine |
|---|---|---|
| Preuves | `evidence_ids` (enveloppe) | Agrégées depuis `clinical_result.evidence_ids`/`fraud_result.evidence_ids` — jamais un identifiant inventé |
| Risques | `result_payload.risks` | Dérivés déterministiquement du snapshot (`_collect_risks`) |
| Contradictions | `result_payload.disagreements` | `DisagreementPoint` — désaccords de statut entre agents déjà validés |
| Questions humain | `result_payload.human_review_reasons` | Jamais vide (`Field(..., min_length=1)`) — au moins un motif toujours présent |

Le LLM ne peut que *citer* des sous-ensembles de ces éléments
(`referenced_evidence_ids`/`acknowledged_risks`/`acknowledged_disagreements`
dans `LlmCaseReviewDecision`) — toute citation d'un identifiant inexistant
est silencieusement ignorée (`agent.py::_merge_llm_decision`), jamais
acceptée comme une preuve nouvelle.

## Erreurs fréquentes

| Situation | Comportement observé | Cause |
|---|---|---|
| `result.errors == [StructuredError(code="LLM_UNAVAILABLE", ...)]` | Recommandation `PENDING`, justification LLM absente | LLM injoignable ou réponse non conforme (`_invoke_llm_case_review` renvoie `None`) — fail-closed, jamais un succès fabriqué |
| `ValidationError` sur `LlmCaseReviewDecision.summary`/`reasons` | Rejeté avant d'atteindre l'agent | Formulation de paiement, diagnostic, accusation ou validation finale détectée — voir `schemas.py::_reject_prohibited_assertions` |
| `ValidationError` sur `CaseReviewerResult.status`/`human_review_required` | Rejeté à la construction | Toute valeur autre que `NEEDS_REVIEW`/`True` — verrouillage structurel, aucune décision finale automatique possible |
| Justification mentionne « références ignorées » | Recommandation normale, motif ajouté | LLM a cité une preuve/un risque/une contradiction inexistant — ignoré, jamais accepté |
| Reprise HITL routée vers `failure` sans exécuter le nœud demandé | `route_human_review` a refusé la relance | `target_node` hors `RELAUNCH_TARGETS`, jamais exécuté pour ce dossier, ou `correction_attempts` au-delà de la limite configurée |
| Décision de reprise invalide lève `HumanDecisionValidationError`, aucun `human_decision` fixé | `node_await_human_review` → `human_review.service.validate_and_audit_human_decision` a rejeté le payload | Acteur manquant, action inconnue, `target_node` manquant/superflu, **justification absente ou vide** — jamais une progression silencieuse vers `END` |

## Revue humaine (HITL) — de la pré-recommandation à la décision

`review_result` ne déclenche jamais lui-même une fin de pipeline : le nœud
`case_reviewer` route toujours vers `needs_review` (`graph/edges.py::route_review`,
puisque `human_review_required` est verrouillé à `True`), qui enchaîne sur
`await_human_review` (`graph/technical_nodes.py::node_await_human_review`).
Ce nœud suspend le graphe via `langgraph.types.interrupt()` avec un payload
minimal incluant un résumé de `review_result`
(`_collect_review_result_summary` : `recommendation`/`justification`/`risks`,
jamais l'instance Pydantic complète) — voir exemple ci-dessous.

**Après la décision humaine, aucun chemin ne contourne l'audit** :
`APPROVE`/`MODIFY`/`REJECT` routent tous vers `audit` avant toute clôture
(`graph/edges.py::route_human_review`/`route_after_audit`) — un rejet passe
par un « échec contrôlé » (audité), jamais un court-circuit direct.

La décision humaine (`APPROVE`/`MODIFY`/`REJECT`/`RETRY`,
`human_review.models.ReviewAction`) est validée par
`graph/technical_nodes.py::node_await_human_review` via
`human_review.service.validate_and_audit_human_decision` — modèle Pydantic
strict (`HumanDecision`, `extra="forbid"`, `justification` obligatoire),
jamais une validation maison. Chaque décision produit un `AuditEvent` ajouté
à `state["audit_trail"]` (action, justification tronquée, auteur,
horodatage, preuves déjà validées par ce même agent — voir
`human_review/README.md`). Seule la vue formulaire (`human_review/views.py`)
reste hors périmètre du graphe de production.

### Exemple de payload d'interruption (`node_await_human_review`)

```json
{
  "case_id": "CLM-0001",
  "thread_id": "CLM-0001",
  "motifs": ["Score de risque de fraude élevé (0.75).", "1 incohérence clinique détectée."],
  "preuves_minimisees": {"clinical_result": "NEEDS_REVIEW", "fraud_result": "NEEDS_REVIEW"},
  "review_result": {
    "recommendation": "PENDING",
    "justification": ["Score de risque de fraude élevé et incohérence clinique détectée..."],
    "risks": ["Score de risque de fraude élevé (0.75)."]
  },
  "actions_autorisees": ["APPROVE", "MODIFY", "REJECT", "RETRY"]
}
```

### Exemple de décision de reprise (`Command(resume=...)`)

`case_id` peut être omis (complété automatiquement depuis le state) —
`justification` est en revanche toujours obligatoire :

```json
{"actor": "reviewer@example.com", "action": "APPROVE", "justification": "Dossier conforme."}
```

### Exemple de `HumanDecision` validé (`human_review.models.HumanDecision`)

```json
{
  "case_id": "CLM-0001",
  "actor": "reviewer@example.com",
  "action": "REJECT",
  "justification": "Preuves insuffisantes pour valider la couverture déclarée.",
  "decided_at": "2026-07-06T14:32:10+00:00"
}
```

## Limites explicites

- **Aucune décision finale, jamais.** Ni la Phase A déterministe ni le LLM
  ne peuvent produire un `APPROVE`/`REJECT` définitif — `status` reste
  toujours `NEEDS_REVIEW` et `human_review_required` toujours `True`,
  verrouillés au niveau du schéma (`schemas/results.py::CaseReviewerResult`),
  pas seulement au niveau du nœud LangGraph.
- **Aucun recalcul métier.** Identité, couverture, codification, cohérence
  clinique et risque de fraude restent les résultats exclusifs de leurs
  agents respectifs — cet agent ne fait que les synthétiser.
- **Les risques sont des signaux, pas des conclusions.** Un risque reconnu
  (ex. score de fraude élevé) n'implique jamais un verdict — seule une
  revue humaine peut trancher.
- **`human_review/` n'est pas encore câblé ici.** Le résumé transmis à
  `interrupt()` reste construit localement par `graph/technical_nodes.py`
  (`_collect_review_result_summary`), pas par `human_review.service`.

## Interdictions strictes

- **Aucune décision finale.** Ni approbation, ni rejet définitif — ce rôle
  appartient exclusivement à l'humain (HITL, `graph/technical_nodes.py::node_await_human_review`).
- **Aucun paiement ni remboursement autorisé.** `LlmCaseReviewDecision`
  rejette structurellement toute formulation de type « remboursement
  validé », « paiement autorisé » — voir `_PAYMENT_DECISION_RE`.
- **Aucun diagnostic médical.** Ce n'est jamais le rôle de cet agent (voir
  `clinical_consistency_agent`) — voir `_DIAGNOSIS_RE`.
- **Aucune accusation.** `LlmCaseReviewDecision` rejette toute formulation
  de type « fraude confirmée/avérée/prouvée/établie », ou toute
  qualification de la personne (« coupable », « fraudeur ») — voir
  `_ACCUSATORY_RE`. Une négation explicite (« fraude non confirmée »)
  reste autorisée : elle exprime une incertitude légitime.
- **Aucune validation finale.** Toute formulation de type « décision
  finale », « validé définitivement », « sans revue humaine » est rejetée
  — voir `_FINAL_DECISION_RE`.
- **Aucune preuve, risque ou contradiction inventé.** Toute citation ne
  correspondant pas à un identifiant réellement calculé est silencieusement
  ignorée (`agent.py::_merge_llm_decision`).
- **Aucune autorité sur le résultat final.** Le LLM ne peut jamais changer
  `status` ni `human_review_required` — verrouillés au niveau du schéma.
- **Aucun contenu brut.** Ni document brut, ni donnée personnelle non
  pseudonymisée, ni secret, ni chemin de fichier, ni texte OCR complet.

## Fichiers

- `agent.py` — `run()`/`node()` ; `_build_agent_snapshot()` (Phase A,
  résumé multi-agent), `_collect_disagreements()`/`_collect_risks()`/
  `_collect_evidence_ids()` (Phase A, contradictions/risques/preuves),
  `_deterministic_pre_recommendation()` (borne la pré-recommandation),
  `_merge_llm_decision()` (fusion Phase B → Phase C, anti-hallucination),
  `_merge_recommendation()` (le LLM peut durcir, jamais assouplir un rejet),
  `_force_human_review()` (défense en profondeur côté nœud).
- `prompt.py` — `load_case_reviewer_prompt()`, charge et versionne le
  prompt système depuis `prompts/case_reviewer_agent.yaml`
  (`PROMPT_VERSION` — garde-fou contre un YAML désynchronisé).
- `schemas.py` — `LlmCaseReviewDecision` (détaillé ci-dessous) ; re-export
  de `CaseReviewerResult`/`DisagreementPoint` (définis dans
  `schemas/results.py`, source unique de vérité).
- `__init__.py` — vide, aucune logique.

## Schéma de sortie LLM (`LlmCaseReviewDecision`)

Sortie JSON stricte forcée par `with_structured_output(LlmCaseReviewDecision,
method="json_schema")` — tout champ inconnu ou tout écart au schéma est
rejeté par Pydantic avant même d'atteindre l'agent.

| Champ | Type | Rôle |
|---|---|---|
| `recommendation` | `Recommendation` | APPROVE / REJECT / PENDING — non finale, toujours révisable |
| `summary` | `str` | Synthèse en français — rejette paiement/diagnostic/accusation/validation finale |
| `reasons` | `list[str]` | Motifs courts — mêmes rejets que `summary` |
| `referenced_evidence_ids` | `list[str]` | Preuves citées, revérifiées contre `evidence_ids` réels |
| `acknowledged_risks` | `list[str]` | Risques reconnus, revérifiés contre `risks` réels |
| `acknowledged_disagreements` | `list[str]` | Contradictions reconnues, revérifiées contre `disagreement_ids` réels |
| `human_review_reasons` | `list[str]` | Questions pour l'humain — mêmes rejets que `summary` |

### Exemple de sortie LLM conforme

```json
{
  "recommendation": "PENDING",
  "summary": "Score de risque de fraude élevé et incohérence clinique détectée ; dossier à examiner avant toute décision.",
  "reasons": ["Score de risque de fraude élevé (0.75).", "1 incohérence clinique détectée."],
  "referenced_evidence_ids": ["EVID-9f8e7d6c5b"],
  "acknowledged_risks": ["Score de risque de fraude élevé (0.75)."],
  "acknowledged_disagreements": [],
  "human_review_reasons": ["Vérifier la cohérence entre l'ordonnance et l'acte facturé avant toute suite."]
}
```

### Exemple de sortie rejetée par le schéma

```json
{
  "recommendation": "APPROVE",
  "summary": "Remboursement validé, le dossier est définitivement clos.",
  "reasons": ["Fraude confirmée mais remboursement accordé quand même."]
}
```

Rejetée par `ValidationError` : `"Remboursement validé"` (décision de
paiement), `"définitivement clos"` (validation finale) et `"Fraude
confirmée"` (accusation) sont chacun interdits par
`schemas.py::_reject_prohibited_assertions`.

### Exemple de `review_result` (`CaseReviewerResult`, `state["review_result"]`)

```json
{
  "case_id": "CLM-0001",
  "status": "NEEDS_REVIEW",
  "llm_trace": {"model_name": "gemma4:latest", "prompt_version": "1.1.0", "confidence": 0.55},
  "confidence": 0.55,
  "errors": [],
  "evidence_ids": ["EVID-9f8e7d6c5b"],
  "human_review_required": true,
  "result_payload": {
    "recommendation": "PENDING",
    "justification": [
      "Score de risque de fraude élevé et incohérence clinique détectée ; dossier à examiner avant toute décision.",
      "Score de risque de fraude élevé (0.75).",
      "1 incohérence clinique détectée."
    ],
    "disagreements": [],
    "risks": ["Score de risque de fraude élevé (0.75).", "1 incohérence(s) clinique(s) détectée(s)."],
    "human_review_reasons": [
      "Validation humaine obligatoire avant toute décision finale.",
      "Vérifier la cohérence entre l'ordonnance et l'acte facturé avant toute suite."
    ]
  }
}
```

Aucun champ `is_final`/`payment_amount`/`diagnosis`/`accusation` — uniquement
une recommandation révisable, des motifs, des risques et des contradictions.

## Cas de test

Couverts par `tests/agents/test_case_reviewer_llm.py` et
`tests/agents/test_case_reviewer_schemas.py` (voir aussi
`tests/orchestrator/test_case_reviewer_orchestration.py` — orchestration
bout-en-bout — et `human_review/README.md`/`tests/human_review/` pour le
cycle HITL interruption/reprise en préparation de l'étape 14) :

| Cas | Résultat attendu |
|---|---|
| Tous les résultats amont disponibles et compatibles, LLM disponible | `APPROVE`/`REJECT`/`PENDING` selon la Phase A + LLM, `human_review_required=True` |
| Résultats amont manquants | `PENDING`, motif « manquants » |
| Fraude/clinique en `FAIL` | Pré-recommandation déterministe `REJECT`, jamais assouplie par le LLM |
| LLM indisponible | `PENDING` fail-closed, `errors=[StructuredError(code="LLM_UNAVAILABLE")]` |
| LLM cite une preuve/un risque/une contradiction inexistant | Référence ignorée, motif explicite (« références ignorées ») |
| LLM emploie une formulation de paiement/diagnostic/accusation/validation finale | `ValidationError` — rejeté avant toute construction du résultat |
| Construction avec `status` ≠ `NEEDS_REVIEW` ou `human_review_required=False` | `ValidationError` — verrouillage structurel, aucune décision finale automatique |

## Interface injectable

`CaseReviewerRunnable` (Protocol) et `make_node(impl)` restent disponibles
pour l'injection de tests
(`graph.nodes.build_orchestrator(case_reviewer_impl=...)`).
L'implémentation par défaut (`node`) exécute la synthèse réelle ci-dessus.
