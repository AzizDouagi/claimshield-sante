# Orchestrator

## Rôle

L'orchestrateur (`Orchestrator`, `orchestrator/executor.py`) est le point
d'entrée unique pour appeler réellement un agent, que ce soit depuis le
graphe LangGraph de production (`graph/nodes.py::build_orchestrator()`) ou,
demain, depuis un futur appelant hors LangGraph (API directe). Il enchaîne,
dans un ordre strict et jamais réordonné, les contrôles définis ailleurs dans
ce paquet — préconditions, modèle, outils — puis délègue l'exécution à
l'agent injecté et valide sa sortie.

**L'orchestrateur est un coordinateur, pas un agent LLM.** Il ne possède :

- ni prompt, ni appel LLM, ni logique de raisonnement ;
- ni règle clinique, financière ou anti-fraude ;
- ni interprétation du *sens* d'un résultat d'agent (seulement de sa
  *forme* — voir `validate_agent_result`).

Il ne fait **qu'orchestrer** : décider si un appel est autorisé, l'exécuter
avec retry si permis, valider ce qui revient, et journaliser ce qui s'est
passé.

## Limites explicites

- **Ne choisit jamais quel résultat métier est correct.** Un désaccord entre
  deux résultats déjà validés (ex. un agent dit PASS, un autre dit FAIL sur
  le même dossier) n'est jamais arbitré ici — voir
  [Désaccords entre résultats](#désaccords-entre-résultats-hors-périmètre-direct).
  Trancher reste le rôle d'un humain (route `needs_review`) ou, plus tard,
  de `case_reviewer_agent`.
- **Ne mute jamais `ClaimState`.** `execute_agent()` retourne un
  `AgentCallOutcome` (résultat validé + bookkeeping + événements d'audit) ;
  c'est à l'appelant (aujourd'hui `graph/nodes.py`) d'en faire une mise à
  jour de state.
- **Ne journalise jamais de contenu brut.** Ni secret, ni prompt complet, ni
  document, ni texte OCR complet — dans les erreurs (`_sanitized_validation_error_fields`)
  comme dans les événements d'audit (`details` limité à des identifiants et
  des codes).
- **N'est pas encore branché à une API HTTP.** `api/` et `app/` restent des
  stubs vides ; le seul appelant en production aujourd'hui est
  `graph/nodes.py` (voir `tests/graph/test_architecture.py`, qui vérifie
  qu'aucun autre chemin ne contourne l'orchestrateur pour appeler un agent).

## Modules

| Fichier | Rôle |
|---|---|
| `orchestrator.py` | Contrats indépendants de LangGraph : `AgentName` (11 valeurs), `AgentCallRequest`/`AgentCallOutcome`, `validate_agent_result` (point de passage obligé de toute sortie d'agent), `without_computed_fields`. |
| `model_registry.py` | `ModelRegistry` injectable : `ModelSpec`, `ModelCapability`, sélection et fallback de modèle. |
| `policies.py` | Politiques d'autorisation pures : allowlists agent/outil/modèle, `PolicyDecision` ALLOW/DENY + motif structuré. |
| `routing.py` | Préconditions d'appel : cohérence `case_id`/étape avec le `ClaimState` réel et le pipeline nominal. |
| `executor.py` | `Orchestrator` — assemble tout ce qui précède, plus `RetryPolicy` et l'émission d'événements d'audit. |

## Ordre des contrôles (`execute_agent`)

```
1. préconditions   (routing.evaluate_call_preconditions, ou un contrôle
                     substitué — ex. graph.nodes._graph_preconditions_check,
                     allégé pour l'intégration LangGraph)
2. modèle          (policies.evaluate_model_authorization, avec fallback —
                     voir plus bas)
3. outils          (policies.build_authorized_tools + capacité TOOL_CALLING)
4. agent           (agent_registry[agent_name](state), avec retry — voir
                     plus bas)
5. validation       (orchestrator.validate_agent_result sur la sortie brute)
```

Le **premier refus** (ALLOW/DENY à n'importe laquelle des trois premières
étapes) retourne immédiatement un `AgentCallOutcome` en échec — aucune étape
suivante ne s'exécute, l'agent n'est jamais appelé. Il n'existe aucun chemin
de code qui atteigne l'agent sans être passé par les trois contrôles dans cet
ordre.

## Allowlists

Trois listes fermées, jamais recopiées en double :

| Autorisation | Source de vérité | Résolution |
|---|---|---|
| **Agent** | `AgentName` (enum fermé, 11 valeurs) | Un nom hors de l'enum est rejeté par le typage lui-même, avant tout appel. |
| **Outil** | `ALLOWED_TOOLS_PER_AGENT` (`policies.py`) | Dérivé par **introspection** des objets `@tool` réels de chaque `agents/<nom>/tools.py` — jamais une liste recopiée à la main. Les 4 agents stubs (sans `tools.py`) en sont absents : toujours refusés. |
| **Modèle** | `ModelRegistry` (`model_registry.py`) | Un modèle doit être enregistré, activé, et posséder les capacités requises par l'agent (`AGENT_REQUIRED_CAPABILITIES`). |

## Retries

`RetryPolicy` (injectée dans `Orchestrator.retry_policy`) ne s'applique
**qu'à l'appel de l'agent** (étape 4) — jamais aux trois contrôles amont, qui
sont déterministes et donc évalués une seule fois quel que soit
`max_attempts`.

- `max_attempts` : défaut `1` (aucun retry — comportement conservateur par
  défaut).
- Rejoué seulement si :
  - l'agent a levé une **exception transitoire** (`transient_exceptions`,
    défaut `httpx.ConnectError`, `httpx.TimeoutException`, `ConnectionError`) ;
  - ou la sortie est **explicitement réparable** (`retryable_error_codes`,
    défaut `AGENT_RESULT_INVALID`/`AGENT_RESULT_UNSTRUCTURED` — une sortie
    malformée peut différer d'un nouvel appel LLM).
- **Jamais rejoué** : un refus de permission/précondition (déterministe,
  rejouer ne changerait rien), une exception non catégorisée (bug), ou
  `AGENT_RESULT_MISSING` (absence de résultat, pas une panne transitoire).
- Entre deux tentatives : même `state`, même requête sauf `attempt`
  (incrémenté) — jamais de mutation de la requête de l'appelant.

## Fallback modèle

Si le modèle demandé est refusé, `Orchestrator._resolve_model()` tente **un
seul** candidat de repli — `ModelRegistry.find_fallback()` (premier autre
modèle enregistré, activé, compatible avec l'agent) — rejoué à travers le
**même** contrôle de modèle. Le fallback n'est retenu que s'il est lui aussi
explicitement autorisé : jamais un contournement de la politique. Sans
candidat, ou si le candidat est aussi refusé, l'erreur d'origine est
retournée inchangée — jamais masquée par un code générique.

## Désaccords entre résultats (hors périmètre direct)

L'orchestrateur ne détecte ni n'arbitre jamais lui-même un désaccord entre
deux résultats d'agents déjà produits — ce mécanisme générique vit dans
`tools/consistency.py` (`detect_result_disagreements`,
`classify_disagreement_severity`) et `graph/edges.py`
(`route_result_consistency`). Un désaccord **critique** (ex. PASS vs FAIL sur
le même dossier) route vers `needs_review`, jamais vers une décision
automatique. L'orchestrateur reste en dehors de cette boucle : il fournit les
résultats validés, il ne les compare jamais entre eux.

## Audit

Chaque appel `execute_agent()` accumule des `AuditEvent` (schéma existant,
`schemas.results.AuditEvent`) dans `AgentCallOutcome.audit_events` — six
natures possibles (`action`) :

| Action | Signification |
|---|---|
| `authorization` | Un contrôle (précondition/modèle/outils) a été franchi. |
| `refusal` | Un contrôle a refusé — l'exécution s'arrête ici. |
| `call` | Tentative d'appel de l'agent. |
| `retry` | Un nouvel essai va être rejoué. |
| `fallback` | Bascule de modèle appliquée ou rejetée. |
| `result` | Issue finale d'`execute_agent` — succès ou échec. |

Chaque événement porte `case_id`/`actor` (champs natifs) puis, dans
`details`, uniquement `model_id`, `tools` (noms joints par une virgule),
`policy` (code du motif), `attempt`, `final_status` — jamais de secret, de
prompt complet, de document brut ni de texte OCR complet.

L'orchestrateur ne les ajoute jamais lui-même à `ClaimState` (il ne mute
jamais le state) : à l'appelant de les ajouter (append) à
`state["audit_trail"]`, exactement comme le ferait un nœud LangGraph.

### Normalisation d'audit batchée (option C d'AZIZ)

Chaque `AuditEvent` doit être normalisé par LLM (`agents.audit_agent`) avant
persistance dans `AuditStore` — c'est la garantie de conformité de l'étape
14, non négociable. Historiquement, `_persist_audit_events` normalisait **un
événement à la fois** (3 à 9 appels LLM par nœud, en plus de l'appel de
décision propre à l'agent) — un mesure réelle avec Ollama a montré que ce
coût dominait très largement le temps d'exécution d'un nœud.

`Orchestrator` accepte désormais un second injectable,
`audit_batch_normalizer` (type `AuditEventBatchNormalizer`,
`Sequence[Mapping] -> Sequence[Any]`), **prioritaire** sur `audit_normalizer`
s'il est fourni (`audit_normalizer` n'est alors jamais appelé) : un seul
appel normalise tous les événements d'audit produits par le nœud en une
fois (`agents.audit_agent.agent.normalize_events_batch`, câblé par défaut
dans `graph.nodes.build_orchestrator()`). Le gain attendu — un seul appel de
normalisation au lieu de N — reste **à confirmer par une mesure réelle avec
Ollama**, jamais un chiffre garanti avant benchmark.

Garanties conservées à l'identique par rapport au chemin single-event
(`_persist_audit_events`, toujours utilisé si aucun `audit_batch_normalizer`
n'est fourni — non-régression par construction) :

- **Jamais de persistance non normalisée.** La réponse du normalizer est
  explicitement réalignée sur exactement `len(events)` éléments avant toute
  itération (jamais un `zip(events, normalized_list)` direct, qui
  tronquerait silencieusement une réponse trop courte) — un événement sans
  normalisation correspondante (réponse trop courte, ou normalisation
  individuellement `None`) est journalisé (`orchestrator_audit_not_persisted`)
  et jamais passé à `AuditStore.record_event`, jamais un crash de
  l'exécution de l'agent.
- **Réassociation par index explicite, jamais par position.** Le lot envoyé
  au LLM (`agents.audit_agent.agent._invoke_llm_audit_batch`) porte un index
  par événement ; la réponse structurée (`LlmAuditNormalizedEventBatch`)
  doit recopier cet index — un index absent, dupliqué (les deux occurrences
  sont invalidées, jamais un « premier gagnant ») ou hors bornes déclenche
  un repli individuel ciblé (`_invoke_llm_audit`, chemin single-event
  inchangé) pour les seuls événements concernés.
- **Aucun mélange entre événements**, même partageant `case_id`/`actor` —
  chaque normalisation ne reflète que son événement d'origine (voir
  `tests/audit/test_agent.py::TestInvokeLlmAuditBatchNeverMixesSameCaseIdOrActor`).
- **Rédaction et plancher appliqués par élément**, jamais partagés entre
  événements du même lot (`tools.audit_redaction.redact_audit_payload`,
  inchangée).

## Exemple minimal d'invocation

Aucune donnée réelle, aucun secret, aucun contenu médical — uniquement des
identifiants synthétiques et un agent factice déterministe.

```python
from orchestrator.executor import Orchestrator
from orchestrator.model_registry import ModelCapability, ModelRegistry, ModelSpec
from orchestrator.orchestrator import AgentCallRequest, AgentName
from schemas.results import SecurityGateResult

# Faux agent déterministe — aucun appel LLM réel.
def fake_security_gate(state: dict) -> dict:
    return {
        "security_result": SecurityGateResult(
            claim_id=state["case_id"], decision="ALLOW", reasons=["exemple synthétique"]
        ),
        "current_step": "security_gate",
        "completed_steps": ["security_gate"],
    }

registry = ModelRegistry()
registry.register(
    ModelSpec(
        model_id="demo-model",
        provider="demo",
        capabilities=frozenset({ModelCapability.STRUCTURED_OUTPUT}),
        client_factory=object,  # jamais instancié tant que l'appel n'est pas autorisé
    )
)

orchestrator = Orchestrator(
    model_registry=registry,
    agent_registry={AgentName.SECURITY_GATE: fake_security_gate},
)

request = AgentCallRequest(
    agent_name=AgentName.SECURITY_GATE,
    case_id="CLM-0001",
    current_step="claim_intake",
    requested_model="SecurityGateResult",
)
state = {"case_id": "CLM-0001", "current_step": "claim_intake", "intake_result": object()}

outcome = orchestrator.execute_agent(request, state, model_id="demo-model")

assert outcome.success is True
assert outcome.result_payload["decision"] == "ALLOW"
```

## Enregistrer un nouvel agent

Aucun agent n'est appelable sans être déclaré à chacune de ces étapes —
c'est délibéré, chaque source de vérité a une seule responsabilité :

1. **Identité** — ajouter la valeur à `AgentName` (`orchestrator.py`).
2. **Schéma de résultat** — ajouter l'entrée à `AGENT_RESULT_MODELS`
   (`orchestrator.py`), le modèle Pydantic attendu en sortie.
3. **Champ ClaimState** — ajouter l'entrée à `AGENT_RESULT_FIELD`
   (`orchestrator.py` — dérivé de `graph.edges.RELAUNCH_RESULT_FIELDS` si
   l'agent est relançable par décision humaine, sinon ajouté directement).
4. **Ordre du pipeline** (si l'agent fait partie du backbone nominal) —
   ajouter l'entrée à `AGENT_PIPELINE_ORDER` (`routing.py`), à la bonne
   position (détermine son prédécesseur pour les préconditions).
5. **Capacités modèle requises** — ajouter l'entrée à
   `AGENT_REQUIRED_CAPABILITIES` (`model_registry.py`) : `frozenset()` si
   l'agent ne nécessite aucun LLM, sinon `STRUCTURED_OUTPUT` et/ou
   `TOOL_CALLING`.
6. **Outils** (si l'agent en utilise) — créer `agents/<nom>/tools.py` avec
   des objets `@tool` réels : `ALLOWED_TOOLS_PER_AGENT` (`policies.py`) les
   découvre automatiquement par introspection, rien à déclarer à la main.
7. **Câblage réel** — enregistrer l'exécuteur dans `agent_registry` au sein
   de `graph.nodes.build_orchestrator()` — **la seule fonction du projet
   autorisée à référencer `agent_module.node`/`.run` directement** (vérifié
   statiquement par `tests/graph/test_architecture.py` — voir
   `graph.nodes._ORCHESTRATOR_REGISTRATION_FUNCTION`). Toute autre façon
   d'appeler l'agent depuis `graph/nodes.py` fait échouer ce test.

## Enregistrer un nouveau modèle

```python
from orchestrator.model_registry import ModelCapability, ModelRegistry, ModelSpec

registry = ModelRegistry()  # ou celui déjà injecté dans l'Orchestrator
registry.register(
    ModelSpec(
        model_id="mon-modele-v2",
        provider="ollama",
        capabilities=frozenset({ModelCapability.STRUCTURED_OUTPUT, ModelCapability.TOOL_CALLING}),
        client_factory=get_llm,  # callable sans argument, résolu à la demande
        enabled=True,
    )
)
```

Contraintes vérifiées structurellement (`ModelSpec.__post_init__` et
`_assert_no_forbidden_fields`, exécutée à l'import du module) :

- `model_id`/`provider` non vides.
- `capabilities` est un `frozenset[ModelCapability]`.
- `client_factory` est un **callable sans argument** — jamais un client déjà
  instancié.
- `ModelSpec` ne peut **jamais** gagner de champ `api_key`/`secret`/`token`/
  `base_url`/`url`/`client` — aucune clé, URL ou instance de client n'est
  stockée dans le registre ; seule une fabrique différée l'est.
- Un `model_id` déjà enregistré ne peut pas être écrasé silencieusement
  (`register()` lève `ValueError`).

Pour qu'un modèle serve un agent donné, ses `capabilities` doivent couvrir
`AGENT_REQUIRED_CAPABILITIES[agent_name]` (`model_registry.py`) — sinon
`evaluate_model_authorization` le refuse (`MODEL_INCOMPATIBLE`).
