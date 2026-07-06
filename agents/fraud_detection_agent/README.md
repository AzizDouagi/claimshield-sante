# Fraud Detection Agent

## Rôle

Combine des preuves déjà validées par d'autres agents (`identity_coverage_result`,
`coding_result`, `ocr_result`) et l'historique pseudonymisé des dossiers déjà
soumis (`services.duplicate_index`, via la vue antifraude minimisée déjà
pseudonymisée par `privacy_agent` — `privacy_result.view`, rôle
`FRAUD_ANALYST`) en un score de risque explicable. Ne recalcule jamais
l'identité, la couverture, la codification ni l'identité pseudonymisée du
patient — il les combine.

**Ce que cet agent ne fait jamais** : il ne déclare jamais une fraude
avérée, ne bloque jamais définitivement un dossier et ne remplace jamais la
revue humaine. Il signale un risque et fournit des preuves ; la décision
finale reste exclusivement humaine (via `case_reviewer_agent`/HITL).

## Pipeline

1. **Phase A (déterministe)** :
   - combine les signaux déjà disponibles (identité NEEDS_REVIEW/FAIL,
     couverture inactive/expirée, plafond dépassé, préautorisation
     manquante, codification non résolue, confiance d'extraction OCR
     faible) ;
   - recherche un doublon exact ou un quasi-doublon (`_check_duplicate()`)
     via `services.duplicate_index`/`tools.statistics`, à partir de la vue
     antifraude minimisée uniquement (jamais de donnée brute) ;
   - combine tout en un score de risque pondéré `risk_score` (0.0 à 1.0,
     explicable — chaque signal porte sa propre contribution) et un statut
     PASS / NEEDS_REVIEW / FAIL (seuils `1.0.0` : < 0.3 PASS, < 0.7
     NEEDS_REVIEW, ≥ 0.7 FAIL). **Seule cette phase décide du statut.**
2. **Phase B (agent ReAct LLM, appel obligatoire à chaque exécution)** —
   `create_react_agent(model=llm, tools=[verifier_doublon],
   response_format=LlmFraudDecision)`. Interprète les doublons, les montants
   atypiques et les signaux déjà calculés ; distingue explicitement dans sa
   justification le **signal** invoqué, le **risque perçu**, l'**incertitude**
   de son analyse et le **besoin d'investigation** complémentaire. Ne
   produit jamais de verdict — voir « Interdictions » ci-dessous.
3. **Phase C** — `_merge_llm_decision()` fusionne la sortie LLM dans les
   motifs (jamais dans le score, le statut ou le besoin de revue), puis
   construction de `FraudDetectionResult` (validée Pydantic, `extra='forbid'`
   à tous les niveaux).

## Outils autorisés

| Outil | Portée |
|---|---|
| `verifier_doublon` | **Seul** outil physiquement joignable (`orchestrator.policies.ALLOWED_TOOLS_PER_AGENT[AgentName.FRAUD_DETECTION] == frozenset({"verifier_doublon"})`). Wrapper fin sur `services.duplicate_index.DuplicateIndex.check()` — ne recalcule qu'un score de similarité déjà défini par la politique versionnée (`DuplicateDetectionPolicy`), ne consulte et ne mute que l'index partagé `_DEFAULT_DUPLICATE_INDEX`, jamais une donnée brute. |

Aucun autre outil n'est enregistré pour cet agent — toute tentative
d'appeler un outil d'un autre agent est bloquée par
`orchestrator.policies.evaluate_tool_authorization` (`ToolAccessError`),
indépendamment de ce que le prompt système autorise ou non.

## Permissions (accès au state)

| Ressource | Accès |
|---|---|
| `state["identity_coverage_result"]` | lecture — statuts identité/couverture déjà validés |
| `state["coding_result"]` | lecture — statut de codification déjà résolu |
| `state["ocr_result"]` | lecture — `confidence_score` et `sha256` (repli si la vue antifraude ne porte pas de hash) uniquement |
| `state["privacy_result"].view` | lecture — uniquement si `view_role == "FRAUD_ANALYST"`, revalidé en `FraudView` ; **jamais reconstruit à partir de données brutes** |
| `services.duplicate_index._DEFAULT_DUPLICATE_INDEX` | lecture (`check()`) et écriture (`register()`) — index partagé par défaut, injectable (`run(duplicate_index=...)`) |
| LLM (`gemma4:latest` via Ollama) | appel obligatoire à chaque exécution effective ; ne reçoit que des résumés déjà minimisés (statut, score, types de signaux, indicateurs de doublon, montant demandé), jamais un document brut |
| `ClaimState` | écrit uniquement `fraud_result`, `current_step`, `completed_steps`, `audit_trail`, et conditionnellement `errors`/`alerts` |

## Seuils

| Seuil | Valeur | Rôle |
|---|---|---|
| `threshold_version` | `"1.0.0"` | Version des seuils ci-dessous — versionnée, jamais changée silencieusement |
| Statut final | `risk_score < 0.3` → `PASS` ; `< 0.7` → `NEEDS_REVIEW` ; `≥ 0.7` → `FAIL` | `_NEEDS_REVIEW_THRESHOLD`/`_FAIL_THRESHOLD`, `agent.py` |
| `_LOW_CONFIDENCE_THRESHOLD` | `0.5` | En dessous, `LOW_EXTRACTION_CONFIDENCE` (+0.15) |
| Contributions de risque | `IDENTITY_MISMATCH` 0.4 · `COVERAGE_INACTIVE_OR_EXPIRED` 0.35 · `PREAUTHORIZATION_MISSING` 0.3 · `CEILING_EXCEEDED` 0.25 · `IDENTITY_AMBIGUOUS` 0.2 · `UNRESOLVED_CODING`/`LOW_EXTRACTION_CONFIDENCE` 0.15 · `EXACT_DUPLICATE_INVOICE` 0.5 · `NEAR_DUPLICATE_INVOICE` 0.25 | Sommées et plafonnées à 1.0 — `risk_score` toujours explicable comme la somme des contributions |
| `DuplicateDetectionPolicy` (`services/duplicate_index.py`) | version `"1.0.0"`, `amount_tolerance_ratio=0.02`, `date_window_days=3`, `near_duplicate_score_threshold=0.85`, poids `weight_amount/text/date=0.4/0.4/0.2` | Versionnée, configurable par instance (`DuplicateIndex(policy=...)`), jamais codée en dur ailleurs |

## Preuves attendues

Chaque `FraudSignal` porte `evidence: list[FraudEvidence]` — **obligatoire**
(`Field(min_length=1)`) : un signal de fraude ne peut jamais être une
affirmation non appuyée, il combine uniquement des preuves déjà validées
par d'autres agents. Chaque `FraudEvidence` porte :

| Champ | Contrainte |
|---|---|
| `evidence_id` | généré automatiquement (`EVID-…`), jamais fourni par le LLM |
| `source` | `FraudEvidenceSource.OCR_EXTRACTION`/`MEDICAL_CODING`/`IDENTITY_COVERAGE`/`DUPLICATE_INDEX` — jamais une valeur flottante non attribuée |
| `field` | nom du champ source, 1 à 100 caractères |
| `document_reference` | identifiant du résultat source (ex. `"identity_coverage_result"`, `"duplicate_index"`), jamais un chemin de fichier |
| `value` | 1 à 500 caractères, **rejette** chemin absolu, secret et contenu multi-lignes (> 2 retours à la ligne) — jamais un document brut |

`evidence_ids` (au niveau de l'enveloppe) est validé pour ne jamais
référencer un identifiant absent des preuves réellement présentes dans
`result_payload` — un identifiant inventé lève une `ValidationError`.

## Erreurs fréquentes

| Situation | Comportement observé | Cause |
|---|---|---|
| `result.errors == [StructuredError(code="LLM_UNAVAILABLE", ...)]` | Score/statut Phase A conservés, justification LLM absente des motifs | LLM injoignable ou réponse non conforme (`_invoke_llm_fraud` renvoie `None`) — fail-closed, jamais un succès fabriqué |
| `ValidationError` sur `LlmFraudDecision.rationale`/`reasons` | Rejeté avant d'atteindre l'agent | Formulation accusatoire détectée (« fraude confirmée », « coupable »...) — voir `_reject_accusatory_language` |
| `ValidationError` sur `FraudSignal` | Rejeté à la construction | `evidence` vide ou absent — un signal ne peut jamais être une affirmation non appuyée |
| `ValidationError` sur `FraudEvidence.value` | Rejeté à la construction | Chemin absolu, secret détecté, ou plus de 2 retours à la ligne |
| `ValidationError` sur `evidence_ids` | Rejeté à la construction | Un identifiant ne correspond à aucune preuve de `result_payload` |
| `duplicate_invoice is None` de façon inattendue | Vérification de doublon non menée | Vue antifraude absente (rôle `!= FRAUD_ANALYST`), ou hash/montant manquant/invalide dans `FraudView` |
| `ToolAccessError` | Levée par `orchestrator.policies.get_authorized_tool` | Tentative d'appeler un outil hors de `ALLOWED_TOOLS_PER_AGENT[AgentName.FRAUD_DETECTION]` |
| `NO_AUTHORIZED_TOOLS` (`AgentCallOutcome.error`) | Agent jamais appelé | Aucun outil autorisé disponible pour un agent exigeant `TOOL_CALLING` — refus fail-closed avant tout appel |

## Limites explicites

- **Pas de base de données de réclamations passées à grande échelle** :
  l'historique de doublons se limite à ce qui a été effectivement soumis à
  l'index en mémoire (`DuplicateIndex`) pendant la durée de vie du
  processus — aucune persistance entre redémarrages (`database/` reste un
  stub). `duplicate_invoice` reste `None` tant que la vérification n'a pas
  pu être menée (vue antifraude absente, hash ou montant manquant/invalide)
  — jamais une valeur inventée.
- **Aucun jugement de fraude, jamais.** Ni la Phase A déterministe ni le
  LLM ne concluent à une fraude avérée — uniquement des signaux et des
  scores de risque, toujours soumis à revue humaine avant toute conséquence.
- **La similarité n'est pas une preuve d'intention.** Un quasi-doublon
  signalé peut avoir une explication légitime (ressaisie administrative,
  consultation de suivi proche) — l'agent ne tranche jamais cette question.

## Interdictions strictes

- **Aucun diagnostic médical.** Ce n'est pas le rôle de cet agent (voir
  `clinical_consistency_agent`) : ni la Phase A ni le LLM ne se prononcent
  jamais sur l'état de santé du patient, uniquement sur le risque de fraude.
- **Aucune accusation.** `LlmFraudDecision` rejette structurellement toute
  formulation de type « fraude confirmée/avérée/certaine/prouvée/établie »,
  « confirmed/proven fraud », ou toute qualification de la personne
  (« coupable », « escroc », « fraudeur ») — voir
  `schemas.py::_reject_accusatory_language`. Une négation explicite
  (« fraude non confirmée », « aucune fraude avérée ») reste autorisée :
  elle exprime une incertitude légitime, jamais une accusation.
- **Aucune décision finale.** Ni remboursement, ni validation ou rejet du
  dossier — ce rôle appartient à `case_reviewer_agent`/HITL.
- **Aucun blocage définitif.** Un statut `FAIL` implique toujours
  `human_review_required=True` (dérivé uniquement du statut) — jamais
  contournable par le LLM.
- **Aucune décision sans humain.** Ce rôle appartient exclusivement à
  `case_reviewer_agent`/HITL.
- **Aucune autorité sur le résultat final.** Le LLM ne peut jamais changer
  `risk_score`, `status` ni `human_review_required` — `llm_risk_perception`
  et `suggests_human_review` sont strictement informatifs.
- **Aucune preuve ou document inventé.** `referenced_signal_types` est
  revalidé contre les signaux réellement calculés — toute référence
  inconnue est silencieusement ignorée (`agent.py::_merge_llm_decision`).
- **Aucun contenu brut.** Ni document brut, ni donnée personnelle non
  pseudonymisée, ni secret, ni chemin de fichier.

## Fichiers

- `agent.py` — `run()`/`node()` ; `_collect_signals()` (Phase A, signaux
  identité/couverture/codification/OCR), `_check_duplicate()`/
  `_extract_fraud_view()` (doublons via historique pseudonymisé),
  `_merge_llm_decision()` (fusion Phase B → Phase C).
- `tools.py` — `verifier_doublon` (unique `@tool` autorisé) et
  `_DEFAULT_DUPLICATE_INDEX` (index partagé, visible, injectable).
- `prompt.py` — `load_fraud_detection_prompt()`, charge et versionne le
  prompt système depuis `prompts/fraud_detection_agent.yaml`
  (`PROMPT_VERSION` — garde-fou contre un YAML désynchronisé).
- `schemas.py` — `LlmFraudDecision` (détaillé ci-dessous) ; re-export de
  `FraudDetectionResult`, `FraudResultPayload`, `FraudSignal`,
  `FraudEvidence`, `FraudEvidenceSource` (définis dans `schemas/results.py`,
  source unique de vérité).
- `__init__.py` — vide, aucune logique.

## Schéma de sortie LLM (`LlmFraudDecision`)

Sortie JSON stricte forcée par `response_format=LlmFraudDecision`
(`create_react_agent`) — tout champ inconnu ou tout écart au schéma est
rejeté par Pydantic avant même d'atteindre l'agent.

| Champ | Type | Rôle |
|---|---|---|
| `rationale` | `str` | Justification en français — rejette toute formulation accusatoire |
| `referenced_signal_types` | `list[str]` | Signaux cités, revérifiés contre les signaux réels |
| `llm_risk_perception` | `float \| None` (0–1) | Risque perçu — explicable, informatif uniquement |
| `suggests_human_review` | `bool` | Besoin d'investigation indicatif — informatif uniquement |
| `reasons` | `list[str]` | Motifs courts, rattachables à un signal — rejette aussi toute formulation accusatoire |

### Exemple de `fraud_result` (`FraudDetectionResult`, `state["fraud_result"]`)

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
    "duplicate_invoice": false,
    "risk_score": 0.45,
    "signals": [
      {
        "signal_type": "PREAUTHORIZATION_MISSING",
        "description": "Préautorisation requise mais absente ou non approuvée.",
        "risk_contribution": 0.3,
        "severity": "MEDIUM",
        "evidence": [
          {
            "evidence_id": "EVID-9f8e7d6c5b",
            "source": "identity_coverage",
            "field": "coverage.preauthorization_status",
            "document_reference": "identity_coverage_result",
            "value": "missing"
          }
        ]
      }
    ],
    "threshold_version": "1.0.0",
    "reasons": [
      "1 signal(aux) de risque combiné(s) — score 0.30.",
      "Aucun doublon détecté dans l'historique pseudonymisé disponible.",
      "Préautorisation manquante à vérifier avant tout remboursement."
    ]
  }
}
```

Aucun champ `is_fraud`/`accusation`/`decision` — uniquement des signaux
attribués, un score explicable et un statut, jamais un verdict.

## Cas de test

Couverts par `tests/agents/test_fraud_detection_agent.py` et
`tests/agents/test_fraud_detection_schemas.py` :

| Cas | Résultat attendu |
|---|---|
| Aucune preuve amont disponible | `NEEDS_REVIEW`, `risk_score=0.0` |
| Identité/couverture/codification/OCR toutes `PASS` | `PASS`, aucun signal |
| Identité `FAIL` | Signal `IDENTITY_MISMATCH` |
| Sans vue antifraude (`fraud_view=None`) | `duplicate_invoice=None` (vérification non menée) |
| Vue antifraude disponible, index vide | `duplicate_invoice=False` (vérifié, rien trouvé) |
| Même `document_hash` qu'un dossier déjà indexé | `duplicate_invoice=True`, signal `EXACT_DUPLICATE_INVOICE` (`CRITICAL`) |
| Montant/date/description proches, même patient | `duplicate_invoice=True`, signal `NEAR_DUPLICATE_INVOICE` (`MEDIUM`) |
| Même montant, patients différents | Jamais un quasi-doublon (filtré par pseudonyme patient) |
| LLM indisponible | Score/statut déterministes conservés, motif explicite ajouté |
| LLM cite un signal inexistant | Référence ignorée, motif explicite (« références ignorées ») |
| LLM emploie une formulation accusatoire (« fraude confirmée ») | `ValidationError` — rejeté avant toute construction du résultat |
| Statut `FAIL` | `human_review_required=True`, toujours |

## Interface injectable

`FraudDetectionRunnable` (Protocol) et `make_node(impl)` restent
disponibles pour l'injection de tests
(`graph.nodes.build_orchestrator(fraud_detection_impl=...)`).
L'implémentation par défaut (`node`) exécute l'évaluation réelle
ci-dessus — elle n'est plus un stub `NOT_EVALUATED`.
