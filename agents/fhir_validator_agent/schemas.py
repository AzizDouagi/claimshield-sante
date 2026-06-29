"""Schémas d'entrée du FHIR Validator Agent — ClaimShield Santé."""
from __future__ import annotations

from pydantic import Field, field_validator

from schemas.domain import StrictModel
from schemas.results import FhirValidatorResult


class FhirValidatorInput(StrictModel):
    """Entrée du nœud FHIR Validator Agent dans le workflow LangGraph.

    Contient les informations nécessaires pour localiser et valider
    le bundle FHIR R4 associé à un dossier de remboursement.

    Contraintes de sécurité :
      - fhir_bundle_path doit être un chemin relatif (jamais absolu)
      - Aucun secret ni donnée personnelle brute dans ce schéma
    """

    case_id: str = Field(..., min_length=1, description="Identifiant du dossier")
    fhir_bundle_path: str | None = Field(
        default=None,
        description="Chemin relatif vers le bundle FHIR sous storage/incoming/",
    )
    bundle_expected: bool = Field(
        default=True,
        description="Indique si un bundle FHIR est attendu pour ce dossier",
    )

    @field_validator("fhir_bundle_path")
    @classmethod
    def no_absolute_path(cls, v: str | None) -> str | None:
        """Refuse les chemins absolus pour éviter toute fuite de contexte système."""
        if v is None:
            return v
        import re
        if re.match(r"^(?:/|[A-Za-z]:[/\\]|\\\\)", v):
            raise ValueError(
                f"Chemin absolu interdit dans fhir_bundle_path : {v!r}. "
                "Utilisez un chemin relatif sous storage/incoming/."
            )
        return v


__all__ = ["FhirValidatorInput", "FhirValidatorResult"]
