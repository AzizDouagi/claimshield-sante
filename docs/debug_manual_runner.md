# Diagnostic `scripts/run_agent_manual.py`

## Objet

Audit statique de `scripts/run_agent_manual.py` pour comprendre pourquoi
`--agent all` affiche des resultats d'agents sans les injecter correctement
dans `ClaimState`.

Je n'ai pas execute `python scripts/run_agent_manual.py --agent all` pendant
cet audit: la probe OCR copie un fichier sous `storage/incoming/...`, ce qui
ajouterait un effet de bord inutile pour un diagnostic.

## Cause racine

`--agent all` n'exécute pas le workflow LangGraph et ne maintient aucun
`ClaimState` partagé.

Dans `main()`, le chemin `--agent all` itere simplement sur `PROBES` et appelle
chaque fonction `probe(...)` de facon independante:

- `scripts/run_agent_manual.py:318-325` boucle sur `PROBES.items()`.
- Chaque probe imprime son propre resume via `_print(...)`.
- La valeur retournee par la probe n'est pas conservee dans un `state`.
- Aucun merge LangGraph n'est applique.
- Aucun champ `*_result`, `current_step`, `completed_steps`, `errors`,
  `alerts` ou `audit_trail` n'est alimente globalement.

Le runner appelle majoritairement les fonctions testables `run(...)` des
agents (`run_claim_intake`, `run_security`, `run_privacy`, etc.), pas les
fonctions `node(state)` ni le graphe compile. Or les fonctions `run(...)`
retournent un modele de resultat, alors que les noeuds LangGraph retournent
une mise a jour partielle du `ClaimState`.

Exemple du comportement attendu cote graphe:

- `graph/nodes.py:267-301` traduit un `AgentCallOutcome` en dict de mise a jour
  `ClaimState`.
- `graph/nodes.py:282-289` reinjecte le resultat valide sous
  `config.result_key`.
- `graph/nodes.py:291-300` ajoute aussi `current_step`, `completed_steps`,
  l'input consomme et l'audit en cas d'echec.
- `graph/workflow.py:427-451` cable ces noeuds dans un `StateGraph(ClaimState)`.

Le runner manuel contourne donc la couche qui fait l'injection state.
Il produit des objets valides et les affiche, mais ne les transforme pas en
updates `ClaimState`.

## Symptomes dans le runner

### Probes independantes

Les probes ne partagent pas un dictionnaire `ClaimState`.

Quelques exemples:

- `probe_claim_intake()` appelle `run_claim_intake(...)` puis imprime le resultat
  (`scripts/run_agent_manual.py:95-105`), sans ecrire `intake_result`.
- `probe_security_safe()` appelle `run_security(...)` puis imprime le resultat
  (`scripts/run_agent_manual.py:108-125`), sans ecrire `security_result`.
- `probe_privacy()` rappelle `probe_security_safe()` localement
  (`scripts/run_agent_manual.py:144-169`), utilise le resultat `gate` comme
  argument direct, puis n'ecrit ni `security_result` ni `privacy_result` dans un
  state commun.
- `probe_ocr()` fabrique son propre `gate` local (`scripts/run_agent_manual.py:226-255`),
  puis imprime `document_ocr`, sans ecrire `ocr_result`.
- `probe_fraud()` rappelle `probe_identity()` localement
  (`scripts/run_agent_manual.py:268-271`), ce qui imprime un resultat
  supplementaire pendant la probe fraud, mais n'alimente toujours pas
  `identity_coverage_result` dans un `ClaimState`.

### Pseudo-state incomplet pour case_reviewer

`probe_case_reviewer()` est le seul endroit qui construit un dict appele
`state`, mais c'est un pseudo-state local et incomplet:

```python
state={
    "case_id": case_id,
    "coding_result": coding,
    "clinical_result": clinical,
    "fraud_result": fraud,
}
```

Voir `scripts/run_agent_manual.py:274-289`.

Ce dict n'est pas un `ClaimState` accumule depuis les probes precedentes. Il ne
contient pas les resultats amont attendus par `case_reviewer_agent`:
`intake_result`, `security_result`, `privacy_result`,
`identity_coverage_result`, `fhir_result`, `ocr_result`.

L'agent reviewer sait lire ces champs s'ils sont presents
(`agents/case_reviewer_agent/agent.py:88-99`), mais le runner ne les lui fournit
pas.

### Noms de probes != noms d'agents

`PROBES` expose des noms pratiques de CLI:

- `security_safe`
- `security_injection`
- `ocr`
- `identity`
- `coding`
- `clinical`
- `fraud`

Ces noms ne correspondent pas toujours aux noms d'agents/states canoniques:

- `security_safe` et `security_injection` sont deux scenarios pour l'agent
  canonique `security_gate`.
- `ocr` correspond a `document_ocr`.
- `identity` correspond a `identity_coverage`.
- `coding` correspond a `medical_coding`.
- `clinical` correspond a `clinical_consistency`.
- `fraud` correspond a `fraud_detection`.

Sans table explicite, le runner ne peut pas savoir quel champ `ClaimState`
alimenter pour chaque probe.

### Identifiant de dossier divergent

`probe_claim_intake()` ignore le `case_id` fourni pour le resultat agent et
cree un identifiant temporel:

```python
manual_case_id = f"CLM-{int(time.time()) % 10000:04d}"
```

Voir `scripts/run_agent_manual.py:95-104`.

Les autres probes utilisent `args.case`. Donc meme si le runner accumulait un
state, `intake_result.case_id` pourrait ne pas correspondre au `case_id` global
du dossier manuel.

## Mappings agent -> champ `ClaimState`

### Mapping canonique utilise par le graphe

Source principale: `_AGENT_CONFIGS` dans `graph/nodes.py:326-410`.

| Agent canonique | Champ resultat `ClaimState` | Champ input `ClaimState` | `step_name` ecrit |
| --- | --- | --- | --- |
| `claim_intake` | `intake_result` | `intake_input` | `claim_intake` |
| `security_gate` | `security_result` | `security_input` | `security_gate` |
| `privacy` | `privacy_result` | `privacy_input` | `privacy` |
| `fhir_validator` | `fhir_result` | `fhir_input` | `fhir_validation` |
| `medical_coding` | `coding_result` | `coding_input` | `medical_coding` |
| `document_ocr` | `ocr_result` | `ocr_input` | `document_ocr_agent` |
| `identity_coverage` | `identity_coverage_result` | `identity_coverage_input` | `identity_coverage` |
| `clinical_consistency` | `clinical_result` | aucun | `clinical_consistency` |
| `fraud_detection` | `fraud_result` | aucun | `fraud_detection` |
| `case_reviewer` | `review_result` | aucun | `case_reviewer` |
| `audit` | `audit_result` | aucun | `audit` |

### Mapping utilise par le routage/reprise humaine

Source: `RELAUNCH_RESULT_FIELDS` dans `graph/edges.py:96-107`.

| Agent relancable | Champ resultat `ClaimState` |
| --- | --- |
| `claim_intake` | `intake_result` |
| `security_gate` | `security_result` |
| `privacy` | `privacy_result` |
| `document_ocr` | `ocr_result` |
| `fhir_validator` | `fhir_result` |
| `identity_coverage` | `identity_coverage_result` |
| `medical_coding` | `coding_result` |
| `clinical_consistency` | `clinical_result` |
| `fraud_detection` | `fraud_result` |
| `case_reviewer` | `review_result` |

`audit` n'est pas relancable ici, mais il est ajoute au mapping orchestrateur
via `_ADDITIONAL_AGENT_RESULT_FIELDS` (`orchestrator/orchestrator.py:102-113`).

### Mapping actuellement implicite dans `scripts/run_agent_manual.py`

Le runner n'a pas de mapping central agent -> champ `ClaimState`.
Il fait seulement ces associations implicites par convention ou par arguments
directs:

| Probe CLI | Agent canonique vise | Champ `ClaimState` attendu | Etat actuel dans le runner |
| --- | --- | --- | --- |
| `claim_intake` | `claim_intake` | `intake_result` | resultat imprime uniquement |
| `security_safe` | `security_gate` | `security_result` | resultat imprime uniquement |
| `security_injection` | `security_gate` | `security_result` | resultat imprime uniquement |
| `privacy` | `privacy` | `privacy_result` | utilise un `gate` local, imprime uniquement |
| `ocr` | `document_ocr` | `ocr_result` | utilise un `gate` local, imprime uniquement |
| `fhir` | `fhir_validator` | `fhir_result` | resultat imprime uniquement |
| `identity` | `identity_coverage` | `identity_coverage_result` | resultat imprime uniquement |
| `coding` | `medical_coding` | `coding_result` | resultat imprime uniquement |
| `clinical` | `clinical_consistency` | `clinical_result` | utilise un `coding` local, imprime uniquement |
| `fraud` | `fraud_detection` | `fraud_result` | utilise `identity`/`coding` locaux, imprime uniquement |
| `case_reviewer` | `case_reviewer` | `review_result` | recoit un pseudo-state local avec `coding_result`, `clinical_result`, `fraud_result`; resultat imprime uniquement |

## Conclusion

`--agent all` donne l'impression d'une execution pipeline, mais c'est en realite
une suite de probes unitaires. Les resultats existent bien en memoire au moment
de chaque appel, puis sont perdus apres affichage parce qu'aucune structure
`ClaimState` n'est creee, fusionnee, validee et transmise a la probe suivante.

Pour corriger plus tard, il faudra choisir explicitement entre deux approches:

1. Faire de `--agent all` un vrai appel au workflow compile (`StateGraph`), avec
   un `ClaimState` initial.
2. Garder le runner manuel, mais ajouter un accumulateur `ClaimState` explicite
   qui mappe chaque probe vers son champ `*_result`, applique les champs de
   progression (`current_step`, `completed_steps`) et transmet ce state aux
   agents dependants.
