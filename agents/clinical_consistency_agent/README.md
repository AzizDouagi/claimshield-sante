# Clinical Consistency Agent

## Rôle

Vérifie la cohérence clinique d'un dossier ClaimShield Santé en croisant les
résultats déjà produits par les agents amont : extraction OCR (`ocr_result`),
codification médicale (`coding_result`), validation FHIR (`fhir_result`) et
vue médicale minimisée déjà pseudonymisée par le Privacy Agent
(`privacy_result.view`, si un rôle `MEDICAL_REVIEWER` a été demandé). Il
s'exécute après `identity_coverage` et avant `fraud_detection` dans le
pipeline nominal.

Il ne recalcule jamais l'identité, la couverture, l'OCR ou la codification —
il compare des résultats déjà validés et signale les incohérences.

## Pipeline

1. **Phase A (déterministe)** — `_collect_signals()` :
   - compare le nombre d'actes/médicaments extraits au nombre de codes
     SNOMED-CT/RxNorm résolus ;
   - vérifie la présence d'une ordonnance si des médicaments sont facturés ;
   - vérifie la présence d'une date de service ;
   - vérifie la chronologie ordonnance/soin et l'absence d'acte codifié via
     `tools.date_checks.run_date_checks()` (mêmes signaux que l'outil exposé
     en Phase B — jamais deux implémentations divergentes).
   - Produit des `ClinicalSignal`/`ClinicalInconsistency` (toujours
     attribués : `evidence_id`, `source`, `field`) et le statut
     PASS / NEEDS_REVIEW / FAIL. **Seule cette phase décide du statut.**
2. **Phase B (agent ReAct LLM, appel obligatoire à chaque exécution)** —
   `create_react_agent(model=llm, tools=[verifier_chronologie],
   response_format=LlmClinicalDecision)`. Analyse la chronologie,
   l'ordonnance, l'acte, la codification et le résumé FHIR minimisé déjà
   calculés, avec pour seul outil autorisé `verifier_chronologie` et, si
   disponible, la vue médicale minimisée. Ne produit qu'un contexte
   explicatif, jamais une décision — voir « Interdictions » ci-dessous.
3. **Phase C** — `_merge_llm_decision()` fusionne la sortie LLM dans les
   motifs (jamais dans le statut, la confiance ou le besoin de revue), puis
   construction de `ClinicalConsistencyResult` (validé Pydantic,
   `extra='forbid'` à tous les niveaux).

## Outils autorisés

| Outil | Portée |
|---|---|
| `verifier_chronologie` | **Seul** outil physiquement joignable (`orchestrator.policies.ALLOWED_TOOLS_PER_AGENT[AgentName.CLINICAL_CONSISTENCY] == frozenset({"verifier_chronologie"})`). Wrapper fin sur `tools/date_checks.run_date_checks` — ne recalcule que les mêmes signaux déterministes que la Phase A, aucune autre lecture/écriture, aucune autorité supplémentaire. |

Aucun autre outil n'est enregistré pour cet agent — toute tentative
d'appeler un outil d'un autre agent (ex. `verifier_doublon` de
`fraud_detection_agent`) est bloquée par
`orchestrator.policies.evaluate_tool_authorization` (`ToolAccessError`),
indépendamment de ce que le prompt système autorise ou non.

## Permissions (accès au state)

| Ressource | Accès |
|---|---|
| `state["ocr_result"]` | lecture — champs déjà extraits (`procedure_count`, `medication_count`, `prescription_number`, `service_date`, `care_date`, `prescription_date`) |
| `state["coding_result"]` | lecture — codes déjà résolus et leur statut |
| `state["fhir_result"]` | lecture — résumé minimisé uniquement (`status`, `resource_count`, `resource_types`), **jamais le bundle FHIR brut** |
| `state["privacy_result"].view` | lecture — uniquement si `view_role == "MEDICAL_REVIEWER"`, revalidé en `MedicalView` ; **jamais reconstruit à partir de données brutes** |
| Outil `verifier_chronologie` | voir ci-dessus |
| LLM (`gemma4:latest` via Ollama) | appel obligatoire à chaque exécution effective ; ne reçoit que les résumés déjà minimisés ci-dessus, jamais un document brut, un texte OCR complet ou un bundle FHIR complet |
| `ClaimState` | écrit uniquement `clinical_result`, `current_step`, `completed_steps`, `audit_trail`, et conditionnellement `errors`/`alerts` |

## Seuils

| Seuil | Valeur | Rôle |
|---|---|---|
| `MAX_PRESCRIPTION_AFTER_CARE_DAYS` (`tools/date_checks.py`) | 30 jours | Au-delà, `PRESCRIPTION_TOO_FAR_AFTER_CARE` (`MEDIUM`) — configurable par appel |
| Statut final | signal `CRITICAL` présent → `FAIL` ; sinon signal présent → `NEEDS_REVIEW` ; aucun signal → `PASS` | Calculé une seule fois en Phase A, jamais réévalué par le LLM |
| `confidence` | `max(0.4, 1.0 - 0.15 × nombre_de_signaux)` | Score explicable, plancher à 0.4, jamais un nombre arbitraire |
| Sévérité des signaux | `IMPOSSIBLE_DATE`/`PRESCRIPTION_BEFORE_CARE`/`MISSING_PRESCRIPTION_REFERENCE` = `CRITICAL` ; `PRESCRIPTION_TOO_FAR_AFTER_CARE`/`MISSING_PROCEDURE_EVIDENCE`/`MISSING_SERVICE_DATE`/`UPSTREAM_CODING_UNRESOLVED` = `MEDIUM` (ou `CRITICAL` pour `PROCEDURE_CODING_COUNT_MISMATCH` si `coded_count == 0`) | Détermine si le dossier peut atteindre `FAIL` |

## Preuves attendues

Chaque `ClinicalSignal`/`ClinicalInconsistency` doit référencer au moins un
champ comparé (`fields_compared`) ou un document (`documents_compared`), et
porte une liste `evidence: list[ClinicalEvidence]` — pour
`ClinicalInconsistency`, au moins une preuve est **obligatoire**
(`Field(min_length=1)`). Chaque `ClinicalEvidence` porte :

| Champ | Contrainte |
|---|---|
| `evidence_id` | généré automatiquement (`EVID-…`), jamais fourni par le LLM |
| `source` | `ClinicalEvidenceSource.OCR_EXTRACTION` ou `MEDICAL_CODING` — jamais une valeur flottante non attribuée |
| `field` | nom du champ source, 1 à 100 caractères |
| `document_reference` | type de document ou nom du résultat source (ex. `"INVOICE"`, `"coding_result"`), jamais un chemin de fichier |
| `value` | 1 à 500 caractères, **rejette** chemin absolu, secret (`api_key`, `password=`...) et contenu multi-lignes (> 2 retours à la ligne) — jamais un document brut ou un texte OCR complet |

`evidence_ids` (au niveau de l'enveloppe) est validé pour ne jamais
référencer un identifiant absent des preuves réellement présentes dans
`result_payload` — un identifiant inventé lève une `ValidationError`.

## Erreurs fréquentes

| Situation | Comportement observé | Cause |
|---|---|---|
| `result.errors == [StructuredError(code="LLM_UNAVAILABLE", ...)]` | Statut Phase A conservé, contexte LLM absent des motifs | LLM injoignable ou réponse non conforme (`_invoke_llm_clinical` renvoie `None`) — fail-closed, jamais un succès fabriqué |
| `ValidationError` sur `ClinicalSignal` | Rejeté à la construction | Ni `fields_compared` ni `documents_compared` fourni — sortie libre non structurée interdite |
| `ValidationError` sur `ClinicalInconsistency` | Rejeté à la construction | `evidence` vide ou absent — une incohérence ne peut jamais être affirmée sans preuve |
| `ValidationError` sur `ClinicalEvidence.value` | Rejeté à la construction | Chemin absolu, secret détecté, ou plus de 2 retours à la ligne (contenu assimilable à un document brut) |
| `ValidationError` sur `evidence_ids` | Rejeté à la construction | Un identifiant ne correspond à aucune preuve de `result_payload` |
| `ValidationError` sur `LlmClinicalDecision` | Rejeté avant d'atteindre l'agent | Champ inconnu (`extra='forbid'`), `reasons` en texte libre au lieu d'une liste, ou secret/chemin détecté |
| `ToolAccessError` | Levée par `orchestrator.policies.get_authorized_tool` | Tentative d'appeler un outil hors de `ALLOWED_TOOLS_PER_AGENT[AgentName.CLINICAL_CONSISTENCY]` |
| `NO_AUTHORIZED_TOOLS` (`AgentCallOutcome.error`) | Agent jamais appelé | Aucun outil autorisé disponible pour un agent exigeant `TOOL_CALLING` — refus fail-closed avant tout appel |

## Limites et interdictions strictes

- **Aucun diagnostic médical.** Le LLM ne pose jamais de diagnostic sur
  l'état de santé du patient.
- **Aucune accusation.** Ni la Phase A ni le LLM ne qualifient un dossier de
  frauduleux ou un acte d'abusif — ce rôle appartient exclusivement à
  `fraud_detection_agent` (signalement de risque, jamais une accusation) et
  reste de toute façon soumis à revue humaine.
- **Aucune décision finale.** Ni remboursement, ni fraude, ni validation ou
  rejet du dossier — ce rôle appartient à `fraud_detection_agent` et
  `case_reviewer_agent`, eux-mêmes jamais définitifs sans revue humaine.
- **Aucun traitement recommandé.** Le LLM ne suggère jamais de médicament,
  de posologie ou de conduite à tenir médicale.
- **Aucun document inventé.** `LlmClinicalDecision` ne porte aucun champ
  libre de nom de document : le LLM ne peut que citer des `evidence_ids`/
  `inconsistency_types` déjà calculés par la Phase A — toute référence à un
  identifiant inexistant est silencieusement ignorée
  (`agent.py::_merge_llm_decision`), jamais acceptée comme preuve.
- **Aucune affirmation non prouvée.** Chaque motif du contexte explicatif
  doit être rattachable à un signal ou une donnée réellement fournie.
- **Aucune autorité sur le résultat final.** Le LLM ne peut jamais changer
  le statut déterministe, la confiance (`confidence`) ni le besoin de revue
  (`human_review_required`, dérivé uniquement du statut) — `llm_confidence`
  et `suggests_human_review` sont strictement informatifs, ajoutés aux
  motifs sans jamais écraser les valeurs déterministes.
- **Aucun contenu brut.** Ni document brut, ni texte OCR complet, ni bundle
  FHIR complet, ni donnée personnelle brute, ni secret, ni chemin de fichier
  — dans le résultat comme dans les données envoyées au LLM.

## Fichiers

- `agent.py` — `run()` (testable sans LangGraph) et `node()` (adaptateur
  LangGraph) ; `_collect_signals()` (Phase A), `_merge_llm_decision()`
  (fusion Phase B → Phase C), `_fhir_summary()`/`_medical_view_summary()`/
  `_extract_medical_view()` (résumés minimisés transmis au LLM).
- `tools.py` — `verifier_chronologie` (unique `@tool` autorisé, wrapper fin
  sur `tools/date_checks.run_date_checks`).
- `prompt.py` — `load_clinical_consistency_prompt()`, charge et versionne le
  prompt système depuis `prompts/clinical_consistency_agent.yaml`
  (`PROMPT_VERSION` — garde-fou contre un YAML désynchronisé).
- `schemas.py` — `LlmClinicalDecision` (schéma de sortie LLM intermédiaire,
  détaillé ci-dessous) ; re-export de `ClinicalConsistencyResult`,
  `ClinicalResultPayload`, `ClinicalSignal`, `ClinicalInconsistency`,
  `ClinicalEvidence`, `ClinicalEvidenceSource` (définis dans
  `schemas/results.py`, source unique de vérité).
- `__init__.py` — vide, aucune logique.

## Schéma de sortie LLM (`LlmClinicalDecision`)

Sortie JSON stricte forcée par `response_format=LlmClinicalDecision`
(`create_react_agent`) — tout champ inconnu ou tout écart au schéma est
rejeté par Pydantic avant même d'atteindre l'agent.

| Champ | Type | Rôle |
|---|---|---|
| `clinical_context` | `str` | Contexte explicatif en français |
| `referenced_evidence_ids` | `list[str]` | Preuves citées, revérifiées contre les `evidence_ids` réels |
| `acknowledged_inconsistencies` | `list[str]` | Types d'incohérence reconnus, revérifiés contre les types réels |
| `llm_confidence` | `float \| None` (0–1) | Confiance perçue — informative uniquement |
| `suggests_human_review` | `bool` | Signal indicatif de revue complémentaire — informatif uniquement |
| `reasons` | `list[str]` | Motifs courts, rattachables à un signal |

## Exemples

### Entrée transmise au LLM (Phase B, déjà minimisée)

```json
{
  "case_id": "CLM-0001",
  "status": "NEEDS_REVIEW",
  "chronologie": [
    {"signal_type": "PRESCRIPTION_TOO_FAR_AFTER_CARE", "severity": "MEDIUM"}
  ],
  "ordonnance": {"medication_count": 2, "prescription_required": true},
  "acte": {"procedure_count": 3},
  "code": {"coded_count": 3, "status": "PASS"},
  "fhir_minimise": {"status": "PASS", "resource_count": 4, "resource_types": ["Patient", "Claim"]},
  "vue_medicale_minimisee": {
    "patient_pseudonym": "PAT-4f2a9c1e0b3d",
    "service_date": "2024-01-15",
    "procedures": ["consultation"],
    "prescription_names": ["amoxicilline"],
    "diagnosis_codes": ["J06.9"],
    "encounter_class": "ambulatory"
  },
  "signals": [{"signal_type": "PRESCRIPTION_TOO_FAR_AFTER_CARE", "severity": "MEDIUM"}],
  "evidence_ids": ["EVID-a1b2c3d4e5"],
  "inconsistency_types": []
}
```

### Sortie LLM conforme (`LlmClinicalDecision`)

```json
{
  "clinical_context": "L'ordonnance est datée 45 jours après le soin facturé, au-delà de la tolérance habituelle.",
  "referenced_evidence_ids": ["EVID-a1b2c3d4e5"],
  "acknowledged_inconsistencies": [],
  "llm_confidence": 0.7,
  "suggests_human_review": true,
  "reasons": ["Écart de délai ordonnance/soin à vérifier manuellement."]
}
```

### Sortie LLM rejetée par le schéma (exemple d'interdiction)

```json
{
  "clinical_context": "Le patient présente probablement une infection bactérienne, prescrire de l'amoxicilline 1g x3/j.",
  "diagnosis": "infection bactérienne"
}
```

Rejetée avant toute exécution : `diagnosis` n'existe pas dans
`LlmClinicalDecision` (`extra='forbid'`) et le contexte recommande un
traitement, ce que le prompt système interdit explicitement.

### Exemple de `clinical_result` (`ClinicalConsistencyResult`, `state["clinical_result"]`)

```json
{
  "case_id": "CLM-0001",
  "status": "NEEDS_REVIEW",
  "llm_trace": {"model_name": "gemma4:latest", "prompt_version": "1.2.0", "confidence": 0.85},
  "confidence": 0.85,
  "errors": [],
  "evidence_ids": ["EVID-a1b2c3d4e5"],
  "human_review_required": true,
  "result_payload": {
    "procedure_count": 3,
    "medication_count": 2,
    "prescription_required": true,
    "signals": [
      {
        "signal_type": "PRESCRIPTION_TOO_FAR_AFTER_CARE",
        "description": "Ordonnance datée 45 jour(s) après le soin correspondant (tolérance 30 jours).",
        "fields_compared": ["prescription_date", "care_date"],
        "documents_compared": [],
        "evidence": [
          {
            "evidence_id": "EVID-a1b2c3d4e5",
            "source": "ocr_extraction",
            "field": "prescription_date",
            "document_reference": null,
            "value": "2024-02-29"
          }
        ],
        "severity": "MEDIUM"
      }
    ],
    "inconsistencies": [],
    "reasons": [
      "Incohérence(s) clinique(s) mineure(s) détectée(s) — revue recommandée.",
      "L'ordonnance est datée 45 jours après le soin facturé, au-delà de la tolérance habituelle.",
      "Écart de délai ordonnance/soin à vérifier manuellement."
    ]
  }
}
```

Aucun champ `recommendation`/`diagnosis`/`decision` — uniquement des
signaux attribués, un statut et une confiance explicables.

## Interface injectable

`ClinicalConsistencyRunnable` (Protocol) et `make_node(impl)` restent
disponibles pour l'injection de tests
(`graph.nodes.build_orchestrator(clinical_consistency_impl=...)`).
L'implémentation par défaut (`node`) exécute l'évaluation réelle
ci-dessus — elle n'est plus un stub `NOT_EVALUATED`.
