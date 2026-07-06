# Audit complet ClaimShield Santé - Étapes 0 à 12

## Mise à jour post-audit — Case Reviewer

- Résolu après audit : `case_reviewer_agent` n'est plus un stub. Il appelle désormais un LLM structuré à chaque exécution effective, synthétise les résultats des agents amont, produit une pré-recommandation non finale et force `human_review_required=True`.
- Ajouts associés : `agents/case_reviewer_agent/prompt.py`, `prompts/case_reviewer_agent.yaml`, `LlmCaseReviewDecision`, tests `tests/agents/test_case_reviewer_llm.py`.
- Dette restante dans cette section : `audit_agent` reste à migrer vers un vrai agent LLM ; les constats historiques ci-dessous sont conservés comme référence d'audit initiale.

## 1. Résumé exécutif

- Score global : 68/100
- Statut global : Non conforme aux règles non négociables, avec une base technique partiellement conforme.
- 5 forces principales :
  - Architecture multi-dossiers claire avec 11 dossiers agents présents : `agents/*`.
  - Schémas Pydantic stricts et état partagé minimisé : `schemas/results.py`, `state/claim_state.py`.
  - Sécurité documentaire et prompt injection largement prise en compte : `security/scanners.py`, `security/policies.py`, `services/storage.py`.
  - Workflow LangGraph complet avec checkpoints, interruptions humaines et branches techniques : `graph/workflow.py`, `graph/checkpoints.py`, `graph/technical_nodes.py`.
  - Suite de tests conséquente : 2432 tests passés, 2 ignorés.
- 5 risques principaux :
  - CRITIQUE : `Case Reviewer Agent` reste un stub non LLM : `agents/case_reviewer_agent/agent.py:1`, `agents/case_reviewer_agent/agent.py:44`.
  - CRITIQUE : `Audit Agent` reste un stub non LLM : `agents/audit_agent/agent.py:1`, `agents/audit_agent/agent.py:45`.
  - CRITIQUE : une route permettrait de terminer sans validation humaine si le reviewer produisait `APPROVE` sans revue humaine : `graph/edges.py:331`, `graph/workflow.py:519`.
  - MAJEUR : le registre modèle et les politiques sont en retard sur l'implémentation réelle de certains agents : `orchestrator/model_registry.py:56`, `orchestrator/policies.py:119`.
  - MAJEUR : l'audit n'est pas une preuve append-only persistante avec hash-chain, export robuste et trace explicite de décision humaine : `agents/audit_agent/agent.py:45`, `schemas/results.py:1042`, `graph/technical_nodes.py:220`.
- Verdict court : le projet est bien structuré et bien testé, mais il ne peut pas être déclaré conforme aux étapes 0-12 car 2 des 11 agents ne sont pas de vrais agents LLM et la validation humaine finale n'est pas rendue impossible à contourner au niveau du routage.

## 2. Méthodologie

- Fichiers inspectés :
  - Documentation : `README.md`, `docs/mvp_scope.md`, `docs/decision_matrix.md`, `docs/data_policy.md`, `datasets/README.md`, `datasets/demo/README.md`, `datasets/demo/PROVENANCE.md`.
  - Agents : tous les dossiers sous `agents/`.
  - Orchestration : `orchestrator/orchestrator.py`, `orchestrator/executor.py`, `orchestrator/model_registry.py`, `orchestrator/policies.py`.
  - LangGraph : `graph/workflow.py`, `graph/edges.py`, `graph/nodes.py`, `graph/technical_nodes.py`, `graph/checkpoints.py`.
  - Schémas et état : `schemas/results.py`, `state/claim_state.py`.
  - Sécurité, stockage, outils : `security/scanners.py`, `security/policies.py`, `services/storage.py`, `tools/*`, `agents/*/tools.py`.
  - Configuration et tests : `pyproject.toml`, `.env.example`, `.gitignore`, `config/settings.py`, `docker-compose.yml`, `tests/*`.
- Commandes exécutées :
  - `git status --short` : plusieurs modifications et fichiers non suivis étaient déjà présents avant l'audit; aucun fichier n'a été modifié sauf `audit.md`.
  - `find . -maxdepth 3 -type f | sort` : arborescence inspectée.
  - `python -V` : échec, `python` absent du shell.
  - `python3 -V` : Python 3.13.13.
  - `.venv/bin/python -V` : Python 3.13.13.
  - `pytest -q` : 2432 passed, 2 skipped.
  - `ruff check .` : échec, `ruff` absent du shell.
  - `.venv/bin/ruff check .` : All checks passed.
  - `.venv/bin/python -m pytest --collect-only -q` : 2434 tests collectés.
  - Recherches `rg` sur `TODO`, `FIXME`, `pass`, `NotImplementedError`, `mock`, `fake`, `deterministic`, `LLM`, `pydantic`, `interrupt`, `Command`, `checkpoint`, `human_review`, `quarantine`, `audit`.
- Limites de l'audit :
  - Les appels LLM réels via Ollama n'ont pas été exécutés en bout en bout contre un serveur modèle réel; les tests passent principalement par doubles/mocks.
  - La feuille de route détaillée des étapes 0 à 12 n'est pas présente comme fichier unique vérifiable. Les noms d'étapes ci-dessous sont reconstruits depuis `README.md`, `docs/mvp_scope.md`, les agents et le workflow.
  - L'état Git est sale avant intervention; les changements existants ne sont pas attribués à l'audit.
- Éléments non vérifiables :
  - Déploiement réel en production.
  - Présence d'un RBAC/ABAC applicatif complet au-delà des politiques de vue privacy.
  - Robustesse réelle des réponses LLM hors tests mockés.
  - Exhaustivité d'un journal append-only persistant externe.

## 3. Score global par catégorie

| Catégorie | Score /5 | Statut | Commentaire |
|---|---:|---|---|
| MVP | 4 | Partiel | Cadrage clair et exclusions documentées dans `docs/mvp_scope.md`, mais conformité finale bloquée par agents stubs. |
| Architecture | 4 | Partiel | Séparation agents/outils/orchestrateur correcte; registre et commentaires obsolètes dans `orchestrator/model_registry.py`. |
| Structure projet | 4 | Partiel | Dossiers principaux présents sauf `database/`; `docker-compose.yml` est vide. |
| Schémas Pydantic | 4 | Partiel | Schémas stricts dans `schemas/results.py`; pas de hash-chain audit. |
| ClaimState | 4 | Conforme | État minimisé, validé et compatible LangGraph : `state/claim_state.py`. |
| Agents LLM | 3 | Non conforme | 9/11 agents ont une intégration LLM; `case_reviewer_agent` et `audit_agent` sont des stubs. |
| Intelligence agents | 3 | Partiel | Bons garde-fous sur plusieurs agents, mais reviewer/audit absents et fraud incomplet. |
| Orchestrateur | 3 | Partiel | Validation forte, mais capacités/politiques non alignées et stubs encore déclarés. |
| LangGraph | 3 | Partiel | Topologie riche, mais routage revue humaine et relance incomplets. |
| Sécurité | 4 | Partiel | Bonne couverture prompt/path/MIME/quarantaine; audit sécurité persistant incomplet. |
| Privacy | 4 | Partiel | Minimisation et vues présentes; RBAC/ABAC complet non vérifiable. |
| Document/OCR | 4 | Partiel | Lecture et OCR sécurisés, provenance présente; dépend de preuves textuelles heuristiques. |
| FHIR | 4 | Partiel | Validation structurelle localisée; LLM indisponible peut faire échouer un cas déterministe valide. |
| Medical Coding | 3 | Partiel | Référentiel versionné et codes non inventés; LLM seulement conditionnel. |
| Clinical Consistency | 4 | Partiel | LLM structuré, statut déterministe; routage LangGraph aval incomplet. |
| Fraud Detection | 3 | Partiel | LLM structuré, pas d'accusation; absence d'historique pour duplicats réels. |
| Case Reviewer | 1 | Non conforme | Stub non LLM : `agents/case_reviewer_agent/agent.py`. |
| Human-in-the-loop | 3 | Partiel | Interruptions présentes; chemin de bypass théorique et trace audit humaine insuffisante. |
| Audit | 2 | Non conforme | `AuditEvent` structuré, mais agent stub, pas de hash-chain ni store append-only vérifiable. |
| Tests | 4 | Partiel | Très nombreux tests; manque E2E LLM réel et tests reviewer/audit réels. |
| Documentation | 4 | Partiel | Docs MVP/data/decision solides; documentation des limites reviewer/audit incomplète. |
| Configuration | 3 | Partiel | `pyproject.toml`, `.env.example`, settings présents; `docker-compose.yml` vide. |
| Qualité code | 4 | Partiel | Ruff OK, typage Pydantic fort; commentaires et registres obsolètes. |

## 4. Audit étape par étape de 0 à 12

### Étape 0 - Cadrage, MVP et règles non négociables

- Statut : Partiel
- Score : 4/5
- Ce qui est présent :
  - Scénario métier synthétique documenté : `docs/mvp_scope.md:3`.
  - Exclusions explicites : données réelles, DICOM, paiement automatique, diagnostic automatique, décisions finales LLM : `docs/mvp_scope.md:95`.
  - Décision humaine obligatoire documentée : `docs/mvp_scope.md:85`, `docs/decision_matrix.md:14`.
  - Politique données synthétiques : `docs/data_policy.md:3`, `datasets/README.md:6`, `datasets/demo/PROVENANCE.md:3`.
- Ce qui manque :
  - Garantie technique absolue que toute issue finale passe par une décision humaine : `graph/edges.py:331`.
  - Validation finale par un vrai `Case Reviewer Agent` LLM : `agents/case_reviewer_agent/agent.py:44`.
- Fichiers concernés : `docs/mvp_scope.md`, `docs/decision_matrix.md`, `docs/data_policy.md`, `graph/edges.py`, `agents/case_reviewer_agent/agent.py`.
- Tests associés : `tests/graph/test_workflow_interrupt_resume.py`, `tests/graph/test_human_review_decision.py`.
- Tests manquants : test interdisant explicitement `END` sans `human_decision` pour tous les chemins finaux.
- Risques :
  - CRITIQUE : conformité réglementaire et métier fragilisée par un chemin théorique sans humain.
- Recommandations :
  - Bloquer tout routage final tant que `human_decision` est absent dans `graph/edges.py`.

### Étape 1 - Structure projet

- Statut : Partiel
- Score : 4/5
- Ce qui est présent :
  - Dossiers `agents`, `graph`, `orchestrator`, `state`, `schemas`, `tools`, `security`, `services`, `tests`, `docs`, `config`.
  - 11 dossiers agents exactement sous `agents/`.
  - Pas de fichier géant `agents.py`; séparation lisible agents/outils/orchestration.
- Ce qui manque :
  - Dossier `database/` attendu par la demande : absent.
  - `docker-compose.yml` présent mais vide.
  - Dossier `storage/` non versionné comme code applicatif, seulement runtime/ignoré.
- Fichiers concernés : `README.md:119`, `docker-compose.yml`, `.gitignore:23`.
- Tests associés : Non vérifiable pour structure.
- Tests manquants : test de structure minimal vérifiant les 11 agents et fichiers requis.
- Risques :
  - MAJEUR : packaging/déploiement incomplet.
- Recommandations :
  - Documenter explicitement l'absence de `database/` ou ajouter la structure attendue lors d'une étape ultérieure.
  - Remplir ou supprimer `docker-compose.yml` selon le choix de déploiement.

### Étape 2 - Schémas Pydantic et ClaimState

- Statut : Conforme
- Score : 4/5
- Ce qui est présent :
  - Modèles de résultats pour les 11 agents : `schemas/results.py:1`.
  - `ClaimState` Pydantic avec champs append-only et données minimisées : `state/claim_state.py:102`.
  - Validation des mises à jour et refus de données brutes/secrets/prompts : `state/claim_state.py:280`, `state/claim_state.py:430`.
  - Sérialisation checkpoint compatible LangGraph : `graph/checkpoints.py:53`.
- Ce qui manque :
  - Versionnement formel de chaque schéma au-delà de champs `schema_version`.
  - Hash-chain d'audit non présent : `schemas/results.py:1042`.
- Fichiers concernés : `schemas/results.py`, `state/claim_state.py`, `graph/checkpoints.py`.
- Tests associés : `tests/unit/test_state_contracts.py`, `tests/unit/test_schemas_results.py`, `tests/unit/test_claim_state_agent_slots.py`.
- Tests manquants : compatibilité ascendante entre versions de schémas.
- Risques :
  - MOYEN : évolution future des schémas sans migration explicite.
- Recommandations :
  - Ajouter une politique de migration/compatibilité des schémas.

### Étape 3 - Claim Intake Agent

- Statut : Partiel
- Score : 4/5
- Ce qui est présent :
  - Dossier dédié complet avec `agent.py`, `prompt.py`, `schemas.py`, `README.md`.
  - Appel LLM structuré : `agents/claim_intake_agent/agent.py:54`.
  - Sortie Pydantic et validation state update : `agents/claim_intake_agent/agent.py:422`.
  - Tests LLM : `tests/agents/test_claim_intake_llm.py`.
- Ce qui manque :
  - L'indisponibilité LLM fait échouer l'intake, ce qui peut bloquer une étape pourtant partiellement déterministe : `agents/claim_intake_agent/agent.py:377`.
- Fichiers concernés : `agents/claim_intake_agent/agent.py`, `agents/claim_intake_agent/prompt.py`, `agents/claim_intake_agent/schemas.py`.
- Tests associés : `tests/agents/test_claim_intake.py`, `tests/agents/test_claim_intake_llm.py`.
- Tests manquants : simulation bout en bout avec LLM réel indisponible et politique métier attendue.
- Risques :
  - MOYEN : fragilité opérationnelle en cas de panne modèle.
- Recommandations :
  - Documenter si l'échec LLM est un choix fail-closed volontaire pour l'étape intake.

### Étape 4 - Security Gate Agent

- Statut : Conforme
- Score : 4/5
- Ce qui est présent :
  - Dossier complet avec prompt dédié : `agents/security_gate_agent/prompt.py`.
  - Scan déterministe plus LLM structuré : `agents/security_gate_agent/agent.py:324`.
  - Le LLM ne peut pas affaiblir une décision déterministe : `agents/security_gate_agent/agent.py:577`.
  - Détection prompt injection, chemins, URL et outils : `security/scanners.py`, `security/policies.py`.
- Ce qui manque :
  - Audit sécurité persistant append-only externe non vérifiable.
- Fichiers concernés : `agents/security_gate_agent/agent.py`, `security/scanners.py`, `security/policies.py`.
- Tests associés : `tests/agents/test_security_gate_llm.py`, `tests/security/test_prompt_injection.py`, `tests/security/test_file_security_policy.py`.
- Tests manquants : test de journalisation persistante des événements sécurité.
- Risques :
  - MOYEN : preuve d'incident limitée à l'état courant.
- Recommandations :
  - Relier les événements sécurité à un store d'audit append-only.

### Étape 5 - Privacy Agent

- Statut : Partiel
- Score : 4/5
- Ce qui est présent :
  - Agent LLM structuré avec décision déterministe fail-closed : `agents/privacy_agent/agent.py:67`, `agents/privacy_agent/agent.py:377`.
  - Minimisation et pseudonymisation : `agents/privacy_agent/agent.py:591`, `config/settings.py:91`.
  - Politique de données : `docs/data_policy.md:3`.
- Ce qui manque :
  - Pas de `prompt.py` dédié dans le dossier agent; prompt YAML chargé via `llm/prompts.py`.
  - RBAC/ABAC applicatif complet non vérifiable.
- Fichiers concernés : `agents/privacy_agent/agent.py`, `prompts/privacy_agent.yaml`, `llm/prompts.py`, `docs/data_policy.md`.
- Tests associés : `tests/agents/test_privacy_agent_llm.py`, `tests/privacy/test_privacy_agent.py`.
- Tests manquants : test d'intégration RBAC/ABAC si une API est ajoutée.
- Risques :
  - MOYEN : contrôle d'accès réel dépendant d'un contexte d'appel non prouvé par une couche applicative.
- Recommandations :
  - Documenter le modèle d'autorisation et ajouter un `prompt.py` ou accepter officiellement le standard YAML.

### Étape 6 - Document and OCR Agent

- Statut : Partiel
- Score : 4/5
- Ce qui est présent :
  - Agent LLM ReAct avec outils autorisés : `agents/document_ocr_agent/agent.py:683`.
  - Texte extrait traité comme donnée non fiable : `agents/document_ocr_agent/agent.py:922`.
  - Lecture PDF sécurisée : `tools/pdf_reader.py:76`.
  - OCR images avec limites : `tools/ocr.py:232`.
  - Provenance et vérification de présence dans le texte source avant acceptation : `agents/document_ocr_agent/agent.py:976`.
  - Raw OCR externalisé/minimisé avant entrée dans l'état : `agents/document_ocr_agent/agent.py:1218`, `state/claim_state.py:333`.
- Ce qui manque :
  - Pas de `prompt.py` dédié.
  - Persistance sécurisée des artefacts non pleinement auditée dans un store externe immuable.
- Fichiers concernés : `agents/document_ocr_agent/agent.py`, `tools/pdf_reader.py`, `tools/ocr.py`, `services/storage.py`.
- Tests associés : `tests/agents/test_document_ocr.py`, `tests/agents/test_document_ocr_llm.py`, `tests/tools/test_pdf_reader.py`, `tests/tools/test_ocr.py`.
- Tests manquants : tests avec vrais PDF/image malformés supplémentaires et moteur OCR réel si disponible.
- Risques :
  - MOYEN : OCR réel et modèles peuvent varier hors mocks.
- Recommandations :
  - Ajouter tests de corpus documentaire hostile en intégration.

### Étape 7 - FHIR Validator Agent

- Statut : Partiel
- Score : 4/5
- Ce qui est présent :
  - Validation bundle JSON et erreurs localisées : `tools/fhir_validation.py:154`, `tools/fhir_validation.py:202`.
  - Bundle optionnel accepté : `agents/fhir_validator_agent/agent.py:249`.
  - Séparation conformité FHIR / validité clinique documentée dans le code : `agents/fhir_validator_agent/agent.py:1`.
  - LLM ReAct avec outils limités : `agents/fhir_validator_agent/agent.py:165`.
- Ce qui manque :
  - Pas de `prompt.py` dédié.
  - LLM indisponible peut faire échouer un bundle déterministiquement valide : `agents/fhir_validator_agent/agent.py:322`.
- Fichiers concernés : `agents/fhir_validator_agent/agent.py`, `tools/fhir_validation.py`, `prompts/fhir_validator_agent.yaml`.
- Tests associés : `tests/agents/test_fhir_validator_agent.py`, `tests/agents/test_fhir_validator_llm.py`, `tests/tools/test_fhir_validation.py`.
- Tests manquants : tests de référence FHIR plus complets avec ressources interconnectées.
- Risques :
  - MOYEN : disponibilité LLM couplée à une validation structurelle.
- Recommandations :
  - Clarifier la politique fail-open/fail-closed pour validation FHIR déterministe.

### Étape 8 - Identity and Coverage Agent

- Statut : Partiel
- Score : 4/5
- Ce qui est présent :
  - Agent LLM avec outils limités contrat/identité/couverture : `agents/identity_coverage_agent/agent.py:514`.
  - Décisions déterministes, LLM consultatif : `agents/identity_coverage_agent/agent.py:542`.
  - Sortie Pydantic et audit event : `agents/identity_coverage_agent/agent.py:792`.
- Ce qui manque :
  - Pas de `prompt.py` dédié.
  - Registre modèle le marque encore sans capacités requises : `orchestrator/model_registry.py:64`.
- Fichiers concernés : `agents/identity_coverage_agent/agent.py`, `agents/identity_coverage_agent/tools.py`, `orchestrator/model_registry.py`.
- Tests associés : `tests/agents/test_identity_coverage_agent.py`, `tests/agents/test_identity_coverage_llm.py`.
- Tests manquants : test de cohérence entre capacités déclarées et appels LLM réels.
- Risques :
  - MAJEUR : un modèle non conforme pourrait être sélectionné par le registre.
- Recommandations :
  - Aligner `orchestrator/model_registry.py` et `orchestrator/policies.py` sur l'agent réel.

### Étape 9 - Medical Coding Agent

- Statut : Partiel
- Score : 3/5
- Ce qui est présent :
  - Référentiel versionné local : `tools/medical_coding.py:40`.
  - Interdiction d'inventer un code : `agents/medical_coding_agent/agent.py:83`.
  - LLM uniquement pour cas à revue et uniquement si le code existe dans le référentiel : `agents/medical_coding_agent/agent.py:51`.
- Ce qui manque :
  - L'agent ne sollicite pas toujours le LLM; si tous les codes sont déterminés, `llm_metadata=None` : `agents/medical_coding_agent/agent.py:127`.
  - Pas de `prompt.py` dédié.
- Fichiers concernés : `agents/medical_coding_agent/agent.py`, `tools/medical_coding.py`, `prompts/medical_coding_agent.yaml`.
- Tests associés : `tests/agents/test_medical_coding_agent.py`, `tests/agents/test_medical_coding_llm.py`, `tests/rules/test_medical_codes.py`.
- Tests manquants : test imposant une trace LLM si la règle projet exige "les 11 agents utilisent un LLM" à chaque exécution.
- Risques :
  - MAJEUR : non-conformité possible à la règle non négociable "les 11 agents doivent utiliser un LLM".
- Recommandations :
  - Décider si "utiliser un LLM" signifie capacité disponible ou appel obligatoire, puis aligner l'implémentation.

### Étape 10 - LangGraph, orchestration, checkpoints et humain

- Statut : Partiel
- Score : 3/5
- Ce qui est présent :
  - 11 nœuds agents plus nœuds techniques : `graph/workflow.py:400`.
  - Branches `quarantine`, `needs_review`, `await_human_review`, `failure`, `finalize`, `END` : `graph/workflow.py:560`.
  - Checkpoints mémoire/sqlite/postgres et reprise par `thread_id` : `graph/checkpoints.py:98`, `graph/checkpoints.py:198`.
  - Interruptions humaines : `graph/technical_nodes.py:220`.
- Ce qui manque :
  - `route_review` peut router vers `END` sans humain si `human_review_required=False` : `graph/edges.py:331`.
  - Décision humaine non inscrite comme `AuditEvent` structuré dans `node_await_human_review` : `graph/technical_nodes.py:220`.
  - Relance limitée, excluant clinical/fraud/case/audit : `graph/edges.py:64`.
  - Clinical/fraud ont des edges inconditionnels : `graph/workflow.py:549`.
- Fichiers concernés : `graph/workflow.py`, `graph/edges.py`, `graph/technical_nodes.py`, `graph/checkpoints.py`.
- Tests associés : `tests/graph/test_workflow_topology.py`, `tests/graph/test_workflow_interrupt_resume.py`, `tests/graph/test_checkpoint_threading.py`.
- Tests manquants : test de propriété "aucun chemin final sans humain".
- Risques :
  - CRITIQUE : contournement futur de la validation humaine.
  - MAJEUR : relance humaine incomplète.
- Recommandations :
  - Forcer `await_human_review` avant toute finalisation métier.

### Étape 11 - Clinical Consistency et Fraud Detection

- Statut : Partiel
- Score : 3/5
- Ce qui est présent :
  - Clinical LLM structuré, statut déterministe et preuves : `agents/clinical_consistency_agent/agent.py:287`, `agents/clinical_consistency_agent/agent.py:332`.
  - Fraud LLM structuré, statut déterministe, pas d'accusation automatique : `agents/fraud_detection_agent/agent.py:190`, `agents/fraud_detection_agent/agent.py:215`.
  - Prompts dédiés présents pour ces deux agents : `agents/clinical_consistency_agent/prompt.py`, `agents/fraud_detection_agent/prompt.py`.
- Ce qui manque :
  - Fraud ne dispose pas d'historique; `duplicate_invoice` reste non évalué : `agents/fraud_detection_agent/agent.py:20`, `agents/fraud_detection_agent/agent.py:235`.
  - Registre modèle/politiques encore obsolètes pour ces agents : `orchestrator/model_registry.py:64`, `orchestrator/policies.py:119`.
  - Routage LangGraph ne traite pas directement leurs statuts : `graph/workflow.py:549`.
- Fichiers concernés : `agents/clinical_consistency_agent/agent.py`, `agents/fraud_detection_agent/agent.py`, `graph/workflow.py`, `orchestrator/model_registry.py`.
- Tests associés : `tests/agents/test_clinical_consistency_agent.py`, `tests/agents/test_fraud_detection_agent.py`.
- Tests manquants : test de fraude avec historique de factures et test de routage en cas de status problématique.
- Risques :
  - MAJEUR : détection fraude incomplète pour duplicats.
  - MAJEUR : résultats ignorés ou seulement synthétisés par un reviewer encore stub.
- Recommandations :
  - Ajouter un mécanisme d'historique synthétique et router les statuts critiques.

### Étape 12 - Case Reviewer et Audit Agent

- Statut : Non conforme
- Score : 1/5
- Ce qui est présent :
  - Dossiers et schémas existent : `agents/case_reviewer_agent/schemas.py`, `agents/audit_agent/schemas.py`, `schemas/results.py:1027`, `schemas/results.py:1042`.
  - Stubs minimaux connectés au workflow : `agents/case_reviewer_agent/agent.py:65`, `agents/audit_agent/agent.py:70`.
- Ce qui manque :
  - Pas d'appel LLM pour Case Reviewer : `agents/case_reviewer_agent/agent.py:44`.
  - Pas d'appel LLM pour Audit Agent : `agents/audit_agent/agent.py:45`.
  - Pas de `prompt.py` ni prompt YAML pour case reviewer et audit.
  - Pas de synthèse réelle multi-agent explicable par le reviewer.
  - Pas d'audit append-only persistant, hash-chain ou export robuste.
- Fichiers concernés : `agents/case_reviewer_agent/agent.py`, `agents/audit_agent/agent.py`, `schemas/results.py`, `graph/workflow.py`.
- Tests associés : `tests/graph/test_stub_agents.py`.
- Tests manquants : tests LLM reviewer, tests LLM audit, tests hash-chain, tests export, tests décision humaine tracée.
- Risques :
  - CRITIQUE : violation directe de la règle "11 agents doivent utiliser un LLM".
  - CRITIQUE : auditabilité finale insuffisante.
- Recommandations :
  - Implémenter ces deux agents avant toute étape suivante.

## 5. Audit détaillé des 11 agents

| Agent | LLM réel | Prompt | Outils limités | Sortie Pydantic | Tests | Score intelligence /5 | Statut |
|---|---|---|---|---|---|---:|---|
| Claim Intake Agent | Oui | Oui | Partiel | Oui | Oui | 4 | Partiel |
| Privacy Agent | Oui | Partiel | Oui | Oui | Oui | 4 | Partiel |
| Identity and Coverage Agent | Oui | Partiel | Oui | Oui | Oui | 4 | Partiel |
| FHIR Validator Agent | Oui | Partiel | Oui | Oui | Oui | 4 | Partiel |
| Document and OCR Agent | Oui | Partiel | Oui | Oui | Oui | 4 | Partiel |
| Medical Coding Agent | Conditionnel | Partiel | Oui | Oui | Oui | 3 | Partiel |
| Clinical Consistency Agent | Oui | Oui | Oui | Oui | Oui | 4 | Partiel |
| Fraud Detection Agent | Oui | Oui | Oui | Oui | Oui | 3 | Partiel |
| Case Reviewer Agent | Non | Non | Non vérifiable | Oui | Stub | 1 | Non conforme |
| Security Gate Agent | Oui | Oui | Oui | Oui | Oui | 4 | Conforme |
| Audit Agent | Non | Non | Non vérifiable | Oui | Stub | 1 | Non conforme |

### Claim Intake Agent

- Rôle attendu : normaliser et valider l'entrée du dossier sans décision médicale ou financière.
- Implémentation observée : LLM structuré via `with_structured_output` dans `agents/claim_intake_agent/agent.py:54`, résultat validé et injecté dans l'état dans `agents/claim_intake_agent/agent.py:422`.
- Niveau d'intelligence réel : 4/5, raisonnement LLM réel mais fortement borné.
- Risques d'hallucination : MOYEN, car le LLM participe au statut intake; limité par Pydantic et validations.
- Garde-fous présents : prompt dédié, sortie structurée, refus de raw docs dans l'état.
- Garde-fous manquants : stratégie alternative documentée si le LLM est indisponible.
- Recommandations : clarifier la politique fail-closed et tester un modèle réel.

### Privacy Agent

- Rôle attendu : minimiser, pseudonymiser et contrôler la vue selon le contexte.
- Implémentation observée : décision déterministe avec LLM d'explication/audit dans `agents/privacy_agent/agent.py:377`.
- Niveau d'intelligence réel : 4/5.
- Risques d'hallucination : FAIBLE à MOYEN, car les décisions principales restent déterministes.
- Garde-fous présents : pseudonymisation, vues limitées, audit event, SecretStr pour clé.
- Garde-fous manquants : `prompt.py` dédié et RBAC/ABAC complet non vérifiable.
- Recommandations : formaliser le contrôle d'accès hors agent.

### Identity and Coverage Agent

- Rôle attendu : vérifier identité, contrat et couverture sans décision finale.
- Implémentation observée : ReAct LLM avec outils contrat explicitement autorisés : `agents/identity_coverage_agent/agent.py:514`.
- Niveau d'intelligence réel : 4/5.
- Risques d'hallucination : FAIBLE, car les statuts sont déterministes et les outils bornés.
- Garde-fous présents : outils limités, sortie Pydantic, audit event.
- Garde-fous manquants : capacités LLM non déclarées dans `orchestrator/model_registry.py:64`.
- Recommandations : mettre à jour registre et tests de politique.

### FHIR Validator Agent

- Rôle attendu : valider la conformité FHIR structurelle sans juger la validité clinique.
- Implémentation observée : validation déterministe plus LLM ReAct avec outils FHIR : `agents/fhir_validator_agent/agent.py:165`.
- Niveau d'intelligence réel : 4/5.
- Risques d'hallucination : FAIBLE, car erreurs structurelles localisées.
- Garde-fous présents : bundle optionnel, erreurs localisées, séparation clinique/FHIR.
- Garde-fous manquants : fallback robuste si LLM indisponible.
- Recommandations : éviter qu'une panne LLM transforme un résultat déterministe valide en échec.

### Document and OCR Agent

- Rôle attendu : extraire documents/OCR comme données non fiables avec provenance.
- Implémentation observée : ReAct LLM avec outils PDF/OCR et scanner injection : `agents/document_ocr_agent/agent.py:683`.
- Niveau d'intelligence réel : 4/5.
- Risques d'hallucination : MOYEN, réduit par vérification de présence textuelle et provenance : `agents/document_ocr_agent/agent.py:976`.
- Garde-fous présents : limites MIME/taille, quarantaine, source excerpt, confiance.
- Garde-fous manquants : tests OCR réels plus larges.
- Recommandations : enrichir corpus hostile PDF/image.

### Medical Coding Agent

- Rôle attendu : proposer des codes médicaux depuis référentiel versionné, sans inventer.
- Implémentation observée : LLM seulement pour cas ambigus; exact match déterministe sans LLM : `agents/medical_coding_agent/agent.py:127`.
- Niveau d'intelligence réel : 3/5.
- Risques d'hallucination : FAIBLE pour codes acceptés, car vérification référentiel : `agents/medical_coding_agent/agent.py:83`.
- Garde-fous présents : référentiel local, rejet des codes inexistants.
- Garde-fous manquants : appel LLM systématique si exigé par la règle projet.
- Recommandations : aligner la règle "11 agents utilisent un LLM" avec le comportement réel.

### Clinical Consistency Agent

- Rôle attendu : évaluer cohérence clinique sans diagnostic.
- Implémentation observée : signaux déterministes plus LLM structuré qui n'altère pas le statut : `agents/clinical_consistency_agent/agent.py:287`, `agents/clinical_consistency_agent/agent.py:332`.
- Niveau d'intelligence réel : 4/5.
- Risques d'hallucination : FAIBLE à MOYEN, LLM limité à justification.
- Garde-fous présents : preuves, scores, refus de diagnostic.
- Garde-fous manquants : routage LangGraph conditionnel sur statut.
- Recommandations : router les statuts critiques et mettre à jour registre modèle.

### Fraud Detection Agent

- Rôle attendu : signaler des risques de fraude sans accusation automatique.
- Implémentation observée : signaux déterministes plus LLM explicatif; pas de verdict par LLM : `agents/fraud_detection_agent/agent.py:190`, `agents/fraud_detection_agent/agent.py:215`.
- Niveau d'intelligence réel : 3/5.
- Risques d'hallucination : FAIBLE pour statut, MOYEN pour explication.
- Garde-fous présents : pas d'accusation, preuves, score, données insuffisantes.
- Garde-fous manquants : historique synthétique; duplicats non évalués : `agents/fraud_detection_agent/agent.py:20`.
- Recommandations : connecter une source d'historique synthétique ou marquer explicitement le risque comme non vérifiable.

### Case Reviewer Agent

- Rôle attendu : synthèse multi-agent, recommandation non finale et obligation de revue humaine.
- Implémentation observée : stub `_NotImplementedStub` renvoyant `PENDING` : `agents/case_reviewer_agent/agent.py:44`.
- Niveau d'intelligence réel : 1/5.
- Risques d'hallucination : non applicable, car pas de LLM.
- Garde-fous présents : `human_review_required=True` dans le stub.
- Garde-fous manquants : LLM, prompt, outils, synthèse, gestion de désaccords, tests dédiés.
- Recommandations : implémenter un agent LLM explicable avant continuation.

### Security Gate Agent

- Rôle attendu : bloquer/quarantainer entrées dangereuses et injections.
- Implémentation observée : scans déterministes puis LLM structuré; le LLM ne peut pas affaiblir le déterministe : `agents/security_gate_agent/agent.py:577`.
- Niveau d'intelligence réel : 4/5.
- Risques d'hallucination : FAIBLE, fail-closed.
- Garde-fous présents : prompt dédié, scanners, politiques chemin/MIME/outils.
- Garde-fous manquants : persistance audit sécurité externe.
- Recommandations : relier à audit immuable.

### Audit Agent

- Rôle attendu : produire audit structuré, append-only, exportable et sans secrets.
- Implémentation observée : stub comptant les événements : `agents/audit_agent/agent.py:45`.
- Niveau d'intelligence réel : 1/5.
- Risques d'hallucination : non applicable, car pas de LLM.
- Garde-fous présents : schéma `AuditEvent` et `AuditResult` : `schemas/results.py:1042`.
- Garde-fous manquants : LLM, prompt, hash-chain, export, persistence append-only, tests réels.
- Recommandations : implémenter l'agent et un journal append-only vérifiable.

## 6. Analyse de l’intelligence agentique

- Statut : Partiel
- Score global Intelligence agentique : 62/100
- Raisonnement LLM : présent dans 9 agents, absent dans `case_reviewer_agent` et `audit_agent`.
- Utilisation des outils : bonne pour FHIR, OCR, identity, coding; limitée ou non applicable pour stubs.
- Capacité à gérer l'ambiguïté : présente dans coding, clinical, fraud et reviewer stub `PENDING`; insuffisante car reviewer réel absent.
- Refus et non-détermination : bien représentés par statuts `NEEDS_REVIEW`, `PENDING`, `NOT_EVALUATED`, données insuffisantes.
- Explicabilité : preuves, justifications et scores présents dans plusieurs résultats : `schemas/results.py`.
- Provenance des preuves : forte côté OCR/FHIR; plus faible côté fraude faute d'historique.
- Robustesse aux injections : forte côté document/security; dépend d'un audit persistant absent.
- Validation structurée : bonne grâce à Pydantic et `validate_state_update`.
- Coordination multi-agent : workflow complet mais synthèse finale non intelligente car reviewer stub.
- Qualité des prompts : bonne pour agents avec `prompt.py` ou YAML; incomplète pour case/audit et incohérente avec l'exigence `prompt.py` par agent.
- Problèmes avec gravité :
  - CRITIQUE : deux agents non LLM : `agents/case_reviewer_agent/agent.py`, `agents/audit_agent/agent.py`.
  - MAJEUR : coding n'appelle pas le LLM dans tous les cas : `agents/medical_coding_agent/agent.py:127`.
  - MAJEUR : fraud ne peut pas vérifier les duplicats historiques : `agents/fraud_detection_agent/agent.py:20`.
  - MAJEUR : coordination aval dépend d'un reviewer stub : `graph/workflow.py:519`.

## 7. Audit sécurité complet

- Statut : Partiel
- Vulnérabilités détectées :
  - CRITIQUE : route finale possible sans humain si `human_review_required=False` : `graph/edges.py:331`.
  - MAJEUR : décision humaine non tracée comme `AuditEvent` structuré : `graph/technical_nodes.py:220`.
  - MAJEUR : audit agent stub sans hash-chain ni export fiable : `agents/audit_agent/agent.py:45`.
  - MAJEUR : registre/politiques d'outils obsolètes pour agents devenus LLM : `orchestrator/model_registry.py:56`, `orchestrator/policies.py:119`.
  - MOYEN : exceptions orchestrateur peuvent inclure le message brut d'exception : `orchestrator/executor.py:560`.
- Risques prompt injection :
  - Détection directe/indirecte présente : `security/scanners.py:60`, `security/scanners.py:310`.
  - OCR traité comme donnée non fiable : `agents/document_ocr_agent/agent.py:922`.
  - Statut : Partiel, car preuve d'audit persistante non vérifiable.
- Risques tool misuse :
  - Allowlist et outils autorisés présents : `orchestrator/policies.py:198`.
  - Statut : Partiel, car les commentaires et capacités sont obsolètes pour plusieurs agents.
- Risques fuite données :
  - État refuse raw docs/secrets/prompts : `state/claim_state.py:280`.
  - Politique de logs interdit OCR complet : `docs/data_policy.md:51`.
  - Statut : Partiel, car RBAC applicatif complet non vérifiable.
- Risques path traversal :
  - Rejet chemins absolus/traversal/outside storage : `security/policies.py:381`, `services/storage.py:89`.
  - Statut : Conforme.
- Risques secrets :
  - `.env` ignoré : `.gitignore:23`.
  - `.env.example` versionné avec placeholders : `.env.example`.
  - Aucun secret réel versionné vérifié par `git ls-files .env .env.example`.
  - Statut : Conforme avec réserve mineure sur placeholders faibles.
- Risques audit incomplet :
  - Pas de hash-chain, pas de store immuable, audit agent stub : `agents/audit_agent/agent.py`.
  - Statut : Non conforme.
- Recommandations classées par priorité :
  - Priorité 0 : imposer humain avant finalisation dans `graph/edges.py`.
  - Priorité 0 : implémenter `Audit Agent` réel avec journal append-only.
  - Priorité 1 : tracer `human_decision` comme `AuditEvent`.
  - Priorité 1 : aligner `orchestrator/model_registry.py` et `orchestrator/policies.py`.
  - Priorité 2 : durcir la sanitization des exceptions dans `orchestrator/executor.py`.

## 8. Audit LangGraph et orchestration

- Statut : Partiel
- Topologie :
  - Workflow avec 11 agents et nœuds techniques : `graph/workflow.py:400`.
  - Branches nominales et d'erreur : `graph/workflow.py:445`, `graph/workflow.py:560`.
- Nœuds :
  - Agents : claim intake, security gate, privacy, document OCR, FHIR, identity, medical coding, clinical, fraud, case reviewer, audit.
  - Techniques : quarantine, needs_review, await_human_review, failure, finalize.
- Edges :
  - Conditionnels sur intake/security/privacy/OCR/FHIR/identity/coding : `graph/workflow.py:445`.
  - Clinical/fraud inconditionnels : `graph/workflow.py:549`.
  - Case reviewer vers audit/needs_review/failure : `graph/workflow.py:519`.
- Checkpoints :
  - Backends mémoire/sqlite/postgres : `graph/checkpoints.py:98`.
  - Reprise avec `thread_id` identique : `graph/checkpoints.py:198`.
- Interruptions :
  - `interrupt` humain dans `node_await_human_review` : `graph/technical_nodes.py:220`.
- Reprise :
  - Tests dédiés présents : `tests/graph/test_workflow_interrupt_resume.py`, `tests/graph/test_checkpoint_threading.py`.
- Erreurs :
  - Quarantine/failure présents.
  - `node_failure` pose une recommandation `REJECT` automatique : `graph/technical_nodes.py:256`, risque de confusion entre échec technique et rejet métier.
- Cohérence avec les 11 agents :
  - Non conforme pour case/audit stubs.
  - Partiel pour model registry et policies obsolètes.
- Problèmes avec gravité :
  - CRITIQUE : chemin final théorique sans humain : `graph/edges.py:331`.
  - MAJEUR : relance humaine limitée : `graph/edges.py:64`.
  - MAJEUR : clinical/fraud non routés conditionnellement : `graph/workflow.py:549`.
  - MAJEUR : orchestrateur dit encore que plusieurs agents sont stubs : `orchestrator/model_registry.py:64`.

## 9. Audit tests

- Statut : Partiel
- Tests existants :
  - Unitaires et intégration agents sous `tests/agents/`.
  - Graph et workflow sous `tests/graph/`.
  - Sécurité sous `tests/security/`.
  - Outils sous `tests/tools/`.
  - Pydantic/state sous `tests/unit/`.
  - Fixtures/demo et 6 dossiers synthétiques couverts : `tests/unit/test_demo_dataset.py`, `datasets/demo/README.md`.
- Résultats :
  - `pytest -q` : 2432 passed, 2 skipped.
  - `.venv/bin/python -m pytest --collect-only -q` : 2434 tests collectés.
  - `.venv/bin/ruff check .` : All checks passed.
  - `python -V` : échec, commande absente.
  - `ruff check .` : échec, commande absente hors venv.
- Tests manquants :
  - CRITIQUE : tests LLM réels pour `case_reviewer_agent` et `audit_agent`.
  - CRITIQUE : test de propriété empêchant tout `END` sans `human_decision`.
  - MAJEUR : test hash-chain/export audit.
  - MAJEUR : test registre capacités pour identity/clinical/fraud.
  - MAJEUR : test fraude avec historique synthétique de duplicats.
  - MOYEN : test E2E avec Ollama réel ou contrat d'intégration LLM.
- Tests critiques à ajouter :
  - `tests/graph/test_no_final_without_human.py`.
  - `tests/agents/test_case_reviewer_llm.py`.
  - `tests/agents/test_audit_agent_llm.py`.
  - `tests/audit/test_hash_chain.py`.
  - `tests/orchestrator/test_agent_capabilities_are_current.py`.
- Risque de régression :
  - MOYEN : la suite est large, mais certains tests valident encore des stubs et peuvent masquer l'incomplétude fonctionnelle.

## 10. Non-conformités critiques

| Gravité | Problème | Fichier | Impact | Correction recommandée |
|---|---|---|---|---|
| CRITIQUE | Case Reviewer Agent non LLM, stub `PENDING` | `agents/case_reviewer_agent/agent.py:44` | Violation directe des 11 agents LLM; pas de synthèse finale intelligente | Implémenter agent LLM structuré avec prompt, schéma, tests et décision non finale |
| CRITIQUE | Audit Agent non LLM, stub `NOT_EVALUATED` | `agents/audit_agent/agent.py:45` | Audit final non fiable et non conforme | Implémenter agent LLM + journal append-only + export |
| CRITIQUE | Finalisation possible sans humain si reviewer approuve sans revue | `graph/edges.py:331` | Violation de validation humaine obligatoire | Forcer `await_human_review` avant toute issue finale métier |
| MAJEUR | Décision humaine non tracée comme AuditEvent structuré | `graph/technical_nodes.py:220` | Auditabilité de la décision finale insuffisante | Ajouter événement audit minimal sans données sensibles |
| MAJEUR | Registre modèle obsolète pour agents LLM actuels | `orchestrator/model_registry.py:56` | Mauvaise sélection modèle/capacités | Déclarer capacités réelles pour identity/clinical/fraud/case/audit |
| MAJEUR | Politiques d'outils obsolètes | `orchestrator/policies.py:119` | Allowlist et documentation incohérentes | Mettre à jour politiques et tests |
| MAJEUR | Clinical/fraud non routés conditionnellement | `graph/workflow.py:549` | Alertes critiques potentiellement seulement synthétisées par stub | Ajouter routes selon statut ou reviewer réel robuste |
| MAJEUR | Relance humaine exclut clinical/fraud/case/audit | `graph/edges.py:64` | Revue humaine moins opérable | Étendre `RELAUNCH_TARGETS` selon politique |
| MAJEUR | Duplicats fraude non évaluables faute d'historique | `agents/fraud_detection_agent/agent.py:20` | Scénarios fraude incomplets | Ajouter historique synthétique ou statut explicite non vérifiable |
| MAJEUR | `docker-compose.yml` vide | `docker-compose.yml` | Déploiement non documenté | Compléter ou retirer du périmètre attendu |
| MOYEN | `prompt.py` manquant pour 7 agents | `agents/privacy_agent`, `agents/identity_coverage_agent`, `agents/fhir_validator_agent`, `agents/document_ocr_agent`, `agents/medical_coding_agent`, `agents/case_reviewer_agent`, `agents/audit_agent` | Non respect du format attendu par agent | Standardiser prompt YAML ou ajouter `prompt.py` |
| MOYEN | Prompts YAML absents pour case/audit | `prompts/` | Agents non câblables au LLM | Ajouter prompts versionnés |
| MOYEN | FHIR échoue si LLM indisponible | `agents/fhir_validator_agent/agent.py:322` | Fragilité opérationnelle | Séparer validation déterministe et enrichissement LLM |
| MOYEN | Audit sans hash-chain | `schemas/results.py:1042` | Intégrité non prouvable | Ajouter hash précédent/hash courant et tests |
| MINEUR | `python` et `ruff` absents du PATH hors venv | environnement shell | Onboarding moins fluide | Documenter l'activation `.venv` |

## 11. Plan d’action priorisé

### Priorité 0 - Bloquant

- Implémenter `Case Reviewer Agent` LLM dans `agents/case_reviewer_agent/agent.py`, avec `prompt.py`, prompt YAML, schéma strict et tests.
- Implémenter `Audit Agent` LLM dans `agents/audit_agent/agent.py`, avec `prompt.py`, prompt YAML, export et tests.
- Modifier le routage de `graph/edges.py` pour rendre impossible toute finalisation métier sans `human_decision`.
- Ajouter un test bloquant couvrant tous les chemins `END` sans validation humaine.

### Priorité 1 - Important

- Tracer la décision humaine comme `AuditEvent` structuré dans `graph/technical_nodes.py`.
- Mettre à jour `orchestrator/model_registry.py` pour les capacités réelles de identity, clinical, fraud, case reviewer et audit.
- Mettre à jour `orchestrator/policies.py` pour les outils et commentaires réels.
- Ajouter un historique synthétique minimal pour la détection des duplicats dans `agents/fraud_detection_agent/agent.py`.
- Ajouter des routes conditionnelles ou une synthèse reviewer réelle pour clinical/fraud dans `graph/workflow.py`.
- Étendre ou documenter `RELAUNCH_TARGETS` dans `graph/edges.py`.

### Priorité 2 - Amélioration

- Standardiser les prompts : soit `prompt.py` dans chaque agent, soit règle documentée autour de `prompts/*.yaml`.
- Compléter `docker-compose.yml` ou documenter son absence de contenu.
- Ajouter hash-chain dans `schemas/results.py` et tests d'intégrité audit.
- Ajouter tests E2E LLM réels optionnels marqués.
- Documenter les limites de disponibilité LLM pour claim intake, FHIR et medical coding.

## 12. Verdict final

- Le projet ne peut pas être considéré conforme aux étapes 0-12.
- Les agents ne sont pas tous réellement LLM : 9/11 ont une intégration LLM réelle ou conditionnelle, mais `Case Reviewer Agent` et `Audit Agent` sont seulement simulés/stubs.
- L'intelligence agentique est prometteuse mais insuffisante pour le niveau attendu : score 62/100.
- Le projet n'est pas prêt pour la suite si la règle "11 agents LLM, validation humaine obligatoire, auditabilité robuste" est bloquante.
- Corrections obligatoires avant de continuer :
  - implémenter `case_reviewer_agent` et `audit_agent` comme vrais agents LLM;
  - verrouiller techniquement la validation humaine obligatoire;
  - tracer la décision humaine dans l'audit;
  - aligner registre, politiques et workflow avec les agents réellement LLM;
  - compléter l'audit append-only et les tests critiques associés.
