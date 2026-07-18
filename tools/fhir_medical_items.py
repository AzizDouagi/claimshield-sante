"""Extraction conservatrice de ressources cliniques FHIR — plan de remédiation
« autonomie décisionnelle V2 », phase 3.

Fonctions pures, aucun effet de bord, aucune règle métier inventée : lit
strictement les champs standards FHIR R4 (`Procedure.code`,
`MedicationRequest`/`MedicationStatement.medicationCodeableConcept`,
`Coverage.payer`/`.identifier`) déjà chargés/validés structurellement par
`tools.fhir_validation.load_fhir_bundle` (jamais un bundle brut, jamais un
nouveau parseur JSON). Réutilisé exclusivement par `graph/recovery_node_v2.py`
(boucle de récupération, phase 6 du plan) — le pipeline nominal continue de
n'exploiter le bundle FHIR qu'en validation structurelle
(`agents/document_understanding_agent/agent.py`, comportement inchangé).

Le type de ressource FHIR (`Procedure` vs `MedicationRequest`/
`MedicationStatement`) sert lui-même de classification — plus fiable qu'une
correspondance approximative contre le référentiel local, d'où
`confidence=1.0`/`classification_method="fhir_resource_type"`.
"""
from __future__ import annotations

from typing import Any

from schemas.domain import VerificationStatus
from schemas.v2_results import ClassifiedMedicalItem, MedicalItemType
from tools.fhir_validation import _iter_indexed_resources

__all__ = ["extract_medical_items_from_bundle", "extract_payer_hint_from_coverage"]

_MEDICATION_RESOURCE_TYPES = frozenset({"MedicationRequest", "MedicationStatement"})


def _coding_display_or_text(codeable_concept: Any) -> str | None:
    """Extrait un libellé exploitable d'un `CodeableConcept` FHIR standard —
    `.text` en priorité, sinon le premier `.coding[].display` non vide.
    Jamais une valeur inventée : `None` si aucun champ standard n'est
    exploitable."""
    if not isinstance(codeable_concept, dict):
        return None
    text = codeable_concept.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    codings = codeable_concept.get("coding")
    if isinstance(codings, list):
        for coding in codings:
            if isinstance(coding, dict):
                display = coding.get("display")
                if isinstance(display, str) and display.strip():
                    return display.strip()
    return None


def extract_medical_items_from_bundle(bundle: dict) -> list[ClassifiedMedicalItem]:
    """Extrait les actes (`Procedure.code`) et médicaments
    (`MedicationRequest`/`MedicationStatement.medicationCodeableConcept`)
    d'un bundle FHIR déjà chargé — restreint à ces deux types de ressources
    standards, jamais une extraction de contenu clinique arbitraire.

    Chaque élément reste soumis à revue (`resolution_status=NEEDS_REVIEW`,
    même convention que `agents.medical_risk_agent.agent._classify_medical_item`)
    — le type de ressource FHIR confirme la catégorie, jamais la décision
    finale.
    """
    items: list[ClassifiedMedicalItem] = []
    for position, resource in _iter_indexed_resources(bundle):
        resource_type = resource.get("resourceType")
        source_document_id = resource.get("id") or f"fhir-entry-{position}"

        if resource_type == "Procedure":
            description = _coding_display_or_text(resource.get("code"))
            if description:
                items.append(
                    ClassifiedMedicalItem(
                        description=description,
                        item_type=MedicalItemType.PROCEDURE,
                        source_document_id=source_document_id,
                        confidence=1.0,
                        classification_method="fhir_resource_type",
                        resolution_status=VerificationStatus.NEEDS_REVIEW,
                    )
                )
        elif resource_type in _MEDICATION_RESOURCE_TYPES:
            description = _coding_display_or_text(resource.get("medicationCodeableConcept"))
            if description:
                items.append(
                    ClassifiedMedicalItem(
                        description=description,
                        item_type=MedicalItemType.MEDICATION,
                        source_document_id=source_document_id,
                        confidence=1.0,
                        classification_method="fhir_resource_type",
                        resolution_status=VerificationStatus.NEEDS_REVIEW,
                    )
                )

    return items


def extract_payer_hint_from_coverage(bundle: dict) -> dict[str, str | None] | None:
    """Extrait un indice payeur/police d'une ressource `Coverage` FHIR
    standard (`.payer[0].display`/`.identifier[].value`) — `None` si aucune
    ressource `Coverage` exploitable n'existe, jamais une valeur inventée."""
    for _position, resource in _iter_indexed_resources(bundle):
        if resource.get("resourceType") != "Coverage":
            continue

        payer_name: str | None = None
        payer_list = resource.get("payer")
        if isinstance(payer_list, list) and payer_list:
            first_payer = payer_list[0]
            if isinstance(first_payer, dict):
                display = first_payer.get("display")
                if isinstance(display, str) and display.strip():
                    payer_name = display.strip()

        policy_number: str | None = None
        identifiers = resource.get("identifier")
        if isinstance(identifiers, list):
            for identifier in identifiers:
                if isinstance(identifier, dict) and isinstance(identifier.get("value"), str):
                    value = identifier["value"].strip()
                    if value:
                        policy_number = value
                        break

        if payer_name or policy_number:
            return {"payer_name": payer_name, "policy_number": policy_number}

    return None
