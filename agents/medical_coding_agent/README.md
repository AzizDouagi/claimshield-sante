# Medical Coding Agent

Agent déterministe de codification médicale.

## Rôle

- Lit les descriptions d'actes et de médicaments depuis `coding_input`.
- Applique la table locale `config/rules/medical_codes.yaml`.
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

- Aucun appel LLM.
- Aucun contenu brut médical ajouté au state hors descriptions déjà structurées.
- Table de codes versionnée via `medical_codes.yaml`.
