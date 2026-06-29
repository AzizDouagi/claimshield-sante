"""Schémas d'entrée du Medical Coding Agent — ClaimShield Santé."""
from __future__ import annotations

from pydantic import Field

from schemas.domain import StrictModel
from schemas.results import MedicalCodingResult, ProcedureCoding


class MedicalCodingInput(StrictModel):
    """Données d'entrée pour la codification médicale.

    Contient l'identifiant du dossier et les listes de descriptions
    d'actes médicaux et de médicaments à coder.
    """

    case_id: str
    procedures: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)


__all__ = ["MedicalCodingInput", "MedicalCodingResult", "ProcedureCoding"]
