# ClaimState — Documentation technique

`ClaimState` est le TypedDict unique qui traverse tous les nœuds du graphe
LangGraph.  Il est sérialisé à chaque checkpoint et restauré lors d'une reprise.

---

## Groupes de champs

| Groupe | Champs | Rôle |
|---|---|---|
| **Routage** | `case_id`, `schema_version`, `intake_status`, `current_step`, `completed_steps` | Décision du prochain nœud, reprise depuis checkpoint. |
| **Entrées consommées** | `intake_input`, `security_input`, `privacy_input`, `ocr_input`, `identity_coverage_input`, `fhir_input`, `coding_input` | Fournis au nœud entrant, remis à `None` après traitement. |
| **Résultats d'agents** | `intake_result`, `security_result`, `privacy_result`, `ocr_result`, `fhir_result`, `coding_result`, `clinical_result`, `fraud_result`, `review_result`, `audit_result` | Un objet Pydantic par agent, écrasable. |
| **Historiques** | `completed_steps`, `errors`, `alerts`, `audit_trail`, `final_justification` | Append-only via reducer `operator.add`. |
| **HITL** | `human_decision` | Écrit uniquement par le point d'interruption LangGraph. |
| **Pré-recommandation / décision finale** | `final_recommendation`, `final_justification` | Pré-recommandation produite par `case_reviewer_agent`, issue finale après HITL/nœuds terminaux. |

---

## Reducers

Cinq champs sont annotés `Annotated[list, operator.add]` :

```python
completed_steps: Annotated[list[str], operator.add]
errors:          Annotated[list[str], operator.add]
alerts:          Annotated[list[str], operator.add]
audit_trail:     Annotated[list[AuditEvent], operator.add]
final_justification: Annotated[list[str], operator.add]
```

LangGraph appelle `operator.add(existant, nouveau)` lors de la fusion.  Le
nœud retourne **uniquement les nouveaux éléments** — jamais la liste complète.

```python
# Correct — retourner seulement les nouveaux éléments
return {"completed_steps": ["security_gate"], "errors": []}

# Incorrect — écraser toute la liste (sans reducer, ça marcherait — mais
# avec operator.add ça duplique les éléments existants si le nœud est repris)
return {"completed_steps": state["completed_steps"] + ["security_gate"]}
```

---

## Pourquoi les documents bruts sont exclus

Le state est persisté à chaque checkpoint.  Stocker PDF, octets d'image ou
texte OCR complet :

- gonfle les checkpoints (plusieurs Mo par dossier) ;
- expose des données médicales en clair dans la base de données ;
- complique la désérialisation (bytes non JSON-compatible).

**Règle** : après OCR, seuls les champs structurés entrent dans le state :
codes, scores, métadonnées, hashes SHA-256, références d'artefacts.  Le texte
complet est écrit dans un artefact externe et référencé par `artifact_id` /
`artifact_path`.

`validate_state_update()` (appelé par chaque nœud) lève `ValueError` si ces
contenus sont détectés.

---

## errors vs alerts

| | `errors` | `alerts` |
|---|---|---|
| Sévérité | Bloquante | Non bloquante |
| Effet sur le workflow | Dossier ne peut pas recevoir `APPROVE` | Workflow continue |
| Exemples | Injection détectée, fichier corrompu, rôle inconnu | Document optionnel absent, OCR à confiance limite |
| Format recommandé | `"[nom_agent] description"` | `"[nom_agent] description"` |

---

## Comment un agent retourne une mise à jour partielle

Un nœud retourne **un dict des seules clés modifiées**.  LangGraph fusionne
ce dict avec le state existant grâce aux reducers.

```python
from state.claim_state import ClaimState, validate_state_update

def node(state: ClaimState) -> dict:
    # --- traitement ---
    result = run(state["security_input"])

    updates = {
        "security_result": result,
        "security_input": None,            # champ consommé → toujours remettre à None
        "current_step": "privacy",
        "completed_steps": ["security_gate"],  # reducer : append automatique
        "errors": [],                      # liste vide = rien ajouté
        "alerts": ["Préauth recommandée"] if result.preauth_required else [],
    }
    validate_state_update(updates)         # obligatoire avant return
    return updates
```

Invariants :
1. Appeler `validate_state_update(updates)` avant `return`.
2. Remettre à `None` le champ `*_input` consommé.
3. N'inclure dans les listes append-only que les **nouveaux éléments**.
4. Ne jamais inclure `bytes`, chemins absolus, texte OCR complet ou secrets.

---

## Exemple de state minimal valide

```python
from state.claim_state import ClaimState, validate_claim_state

state: ClaimState = {
    "case_id": "CLM-0001",
    "schema_version": "1.0.0",
    "current_step": "claim_intake",
    "completed_steps": [],
}

validate_claim_state(state)  # passe sans erreur
```

Ce state est suffisant pour initialiser le graphe.  Tous les autres champs
sont optionnels (`total=False`) et sont ajoutés au fur et à mesure par les
nœuds successifs.

### State complet en fin de workflow (exemple)

```python
{
    # Routage
    "case_id": "CLM-0001",
    "schema_version": "1.0.0",
    "intake_status": "ACCEPTED",
    "current_step": "case_review",
    "completed_steps": [
        "claim_intake", "security_gate", "privacy",
        "document_ocr", "fhir_validator", "medical_coding",
        "clinical_consistency", "fraud_detection", "case_review",
    ],

    # Entrées consommées (toutes à None en fin de workflow)
    "intake_input": None,
    "security_input": None,
    "privacy_input": None,
    "ocr_input": None,
    "fhir_input": None,
    "coding_input": None,

    # Résultats des agents (un objet Pydantic par agent)
    "intake_result": ClaimIntakeResult(...),
    "security_result": SecurityGateResult(decision="ALLOW", ...),
    "privacy_result": PrivacyResult(status="PASS", ...),
    "ocr_result": DocumentOcrResult(artifact_id="ocr-42", ...),
    # … etc.

    # Historiques append-only
    "errors": [],
    "alerts": ["Préautorisation recommandée pour acte 99213"],
    "audit_trail": [AuditEvent(...), AuditEvent(...), ...],
    "final_justification": ["Tous les contrôles sont passés."],

    # Pré-recommandation / décision finale
    "final_recommendation": "APPROVE",
}
```

---

## Validation

```python
from state.claim_state import validate_claim_state, validate_state_update

# Avant checkpoint ou reprise : validation complète
validate_claim_state(state)

# Avant chaque return de nœud : validation des mises à jour
validate_state_update({"security_result": result, "security_input": None})
```
