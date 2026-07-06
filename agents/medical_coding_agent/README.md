# Medical Coding Agent

Agent LLM de codification médicale, borné par un référentiel local.

## Rôle

- Lit les descriptions d'actes et de médicaments depuis `coding_input`.
- Applique la table locale `config/rules/medical_codes.yaml`.
- Appelle le LLM à chaque exécution pour valider/justifier les codes, y compris
  lorsque la correspondance est déjà trouvée déterministiquement.
- Produit un `MedicalCodingResult` sérialisable dans le `ClaimState`.

## Statuts

- `PASS` : toutes les descriptions ont une correspondance exacte.
- `NEEDS_REVIEW` : au moins une description nécessite une revue humaine.
- `FAIL` : l'entrée du state est absente ou invalide.

## Entrée attendue

```json
{
  "case_id": "CLM-0001",
  "procedures": ["Office Visit"],
  "medications": ["Acetaminophen 325 MG Oral Tablet"]
}
```

## Garanties

- Appel LLM obligatoire à chaque exécution effective.
- Le LLM ne peut jamais inventer un code : toute proposition est acceptée
  uniquement si le code existe et est actif dans le référentiel local, dans la
  bonne section SNOMED-CT/RxNorm.
- Si le LLM est indisponible ou retourne une réponse invalide, l'agent échoue
  en fail-closed (`FAIL`) au lieu d'accepter un résultat purement déterministe.
- Aucun contenu brut médical ajouté au state hors descriptions déjà structurées.
- Table de codes versionnée via `medical_codes.yaml`.
