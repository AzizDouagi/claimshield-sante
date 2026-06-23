"""Schémas d'entrée/sortie du Claim Intake Agent.

Schémas d'entrée définis ici ; schémas de sortie importés depuis schemas/results.py.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field

from schemas.domain import StrictModel
from schemas.results import (  # noqa: F401  — réexportés pour usage agent
    ClaimIntakeResult,
    ClaimManifest,
    InspectedFile,
    StructuredError,
    UploadedFileInfo,
)

# Documents obligatoires par défaut (noms génériques — remplacés par les noms
# réels du cas lors de l'appel depuis le workflow).
DEFAULT_REQUIRED_DOCUMENTS: list[str] = [
    "demande_remboursement.pdf",
    "facture.pdf",
    "ordonnance.pdf",
    "compte_rendu.pdf",
]


class ClaimIntakeInput(StrictModel):
    """Paramètres d'entrée du Claim Intake Agent."""

    case_id: str = Field(..., pattern=r"^CLM-\d{4,}$")
    source_path: Path = Field(..., description="Répertoire contenant les fichiers du dossier")
    required_documents: list[str] = Field(
        default_factory=list,
        description="Noms de fichiers obligatoires pour ce dossier",
    )
    uploaded_files: list[UploadedFileInfo] = Field(
        default_factory=list,
        description="Métadonnées annoncées par le déposant avant inspection",
    )
