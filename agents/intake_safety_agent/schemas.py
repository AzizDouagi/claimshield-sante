"""Schémas d'entrée/décision de intake_safety_agent (V2, plan Phase V2-2).

Fusionne `agents/claim_intake_agent/schemas.py::ClaimIntakeInput` +
`agents/security_gate_agent/schemas.py::LlmSecurityDecision` en un seul
agent, un seul appel LLM. Le résultat structuré final est
`schemas.v2_results.IntakeSafetyResult` (réutilisé, jamais dupliqué) —
ce module ne définit que l'entrée et le schéma de décision LLM
intermédiaire (jamais persisté dans `ClaimStateV2`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from schemas.domain import StrictModel
from schemas.results import UploadedFileInfo

__all__ = ["IntakeSafetyInput", "LlmIntakeSafetyDecision"]


class IntakeSafetyInput(StrictModel):
    """Paramètres d'entrée de `intake_safety_agent`.

    Porte de `agents.claim_intake_agent.schemas.ClaimIntakeInput` (V1),
    sans `role` — déplacé vers `ClaimStateV2.reader_role` (posé une fois à
    la soumission, lu plus tard par `document_understanding_agent`, jamais
    par cet agent).
    """

    case_id: str = Field(..., pattern=r"^CLM-\d{4,}$")
    source_path: Path = Field(..., description="Répertoire contenant les fichiers du dossier")
    required_documents: list[str] = Field(
        default_factory=list,
        description="Noms de fichiers obligatoires pour ce dossier",
    )
    depositor_id: str | None = None
    uploaded_files: list[UploadedFileInfo] = Field(
        default_factory=list,
        description="Métadonnées annoncées par le déposant avant inspection",
    )


class LlmIntakeSafetyDecision(StrictModel):
    """Décision LLM combinée complétude + sécurité.

    Ne peut jamais adoucir un statut déterministe plus restrictif que le
    sien — voir `agent.py::_merge_status` (règle de sécurité non
    négociable, plan V2 §7 : « le LLM ne peut plus jamais adoucir un
    BLOCK »). ``TECHNICAL_FAILURE`` reste réservé aux pannes
    d'infrastructure détectées en Phase A, jamais choisi librement ici.
    """

    status: Literal["ACCEPTED", "QUARANTINED", "BLOCKED", "TECHNICAL_FAILURE"]
    reasons: list[str] = Field(default_factory=list)
    explanation: str = Field(default="", max_length=500)
