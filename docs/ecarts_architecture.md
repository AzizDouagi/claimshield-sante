# Écarts d'architecture assumés — par rapport à la spec initiale

Trois écarts identifiés lors de l'audit du projet par rapport au PDF de
spécification initial (feuille de route 22 étapes). Vérifiés sur le code
réel, tous acceptés tels quels — aucun n'a d'impact fonctionnel, uniquement
organisationnel ou de choix de conception documenté.

## 1. Organisation des schémas Pydantic (étape 2 de la spec)

**Écart** : la spec envisage un fichier de schémas par agent. Le projet
regroupe tous les schémas de résultats dans `schemas/results.py` (~1400
lignes) et les modèles de domaine partagés dans `schemas/domain.py`
(~800 lignes), plus `schemas/audit.py` séparé pour le journal d'audit.

**Justification** : un schéma de résultat référence souvent des types
partagés avec d'autres agents (`VerificationStatus`, `SeverityLevel`,
`StructuredError`, `LlmMetadata`...) — les regrouper évite les imports
circulaires entre modules d'agents et centralise les règles communes
(`extra="forbid"`, validateurs anti-secret/anti-chemin absolu réutilisés).

**Décision** : accepté tel quel. Un éclatement en fichiers par agent
resterait un pur confort de lecture, sans bénéfice fonctionnel — non
prioritaire.

## 2. Autorité de décision du Security Gate (étape 5 de la spec)

**Écart** : `security_gate_agent` donne au LLM l'autorité de la décision
finale (`ALLOW`/`BLOCK`/`QUARANTINE`), la Phase A déterministe ne faisant
que préparer les scans et une recommandation. Une lecture stricte de la
spec pourrait attendre une décision purement déterministe pour un composant
de sécurité.

**Justification** : cohérent avec la règle projet non négociable (voir
`CLAUDE.md`, section « Architecture LLM migrée ») — chaque agent métier doit
appeler un LLM à chaque exécution effective. Le risque est borné : repli
conservateur `BLOCK` si le LLM est indisponible ou renvoie une décision
invalide (fail-closed), jamais un `ALLOW` par défaut.

**Décision** : accepté tel quel — choix de sécurité défendable (fail-closed
en cas de doute), cohérent avec le reste du projet.

## 3. Nommage et périmètre de l'orchestrateur (étape 11 de la spec)

**Écart** : nommage différent du PDF de spec (`orchestrator.Orchestrator`,
`AgentCallRequest`/`AgentCallOutcome` plutôt que la terminologie originale).

**Justification** : `orchestrator/` est volontairement découplé de LangGraph
— une seule exception documentée dans le code
(`orchestrator/orchestrator.py` importe `RELAUNCH_RESULT_FIELDS` depuis
`graph.edges`, une constante pure, jamais un couplage au moteur LangGraph
lui-même). Le nommage reflète ce périmètre (contrats d'appel d'agent
génériques, indépendants du graphe) plutôt que la terminologie du PDF.

**Décision** : accepté tel quel — cosmétique, le découplage réel (la
propriété qui compte) est respecté et vérifié.
