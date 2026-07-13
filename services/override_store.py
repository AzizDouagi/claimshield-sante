"""Journal d'overrides asynchrones — services/override_store.py (V2).

Corrections humaines post-décision, indépendantes de `ClaimStateV2` — même
patron que `services.audit_store.AuditStore` (injectable, en mémoire,
append-only, aucune instance globale cachée). Une correction n'entre
jamais dans le state du graphe V2 : elle est uniquement annotée ici,
jamais superposée à la décision d'origine (`decision_result` reste
immuable — voir le plan V2, Phase V2-8).

`OverrideAction`/`OverrideRecord` sont le schéma canonique du journal —
`human_review/override_models.py` (Phase V2-8) définira séparément le
contrat de *requête* humaine brute (`OverrideRequest`, à valider avant
construction d'un `OverrideRecord`), distinction identique à celle déjà
en place entre `schemas.audit.AuditEvent` (stocké) et
`human_review/models.py::HumanDecision` (requête V1).
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

from pydantic import Field

from schemas.domain import StrictModel

__all__ = ["OverrideAction", "OverrideRecord", "OverrideStore"]


class OverrideAction(str, Enum):
    """Actions humaines possibles sur un dossier déjà décidé par le pipeline V2.

    Distinct de `human_review.models.ReviewAction` (V1, HITL bloquant) —
    ici, l'action s'applique toujours *après* une décision autonome déjà
    rendue, jamais avant.
    """

    CONFIRM = "CONFIRM"
    OVERRIDE_APPROVE = "OVERRIDE_APPROVE"
    OVERRIDE_REJECT = "OVERRIDE_REJECT"
    REOPEN = "REOPEN"


class OverrideRecord(StrictModel):
    """Une correction humaine post-décision — jamais une mutation de `decision_result`.

    `original_decision` est purement informatif (la valeur de
    `ClaimDecisionV2` observée au moment de l'override) — ce champ ne fait
    jamais autorité, `decision_result` reste la seule source de vérité de
    la décision autonome d'origine.
    """

    override_id: str = Field(default_factory=lambda: str(uuid4()))
    case_id: str = Field(..., pattern=r"^CLM-\d{4,}$")
    actor: str = Field(..., min_length=1, max_length=255)
    action: OverrideAction
    justification: str = Field(..., min_length=1, max_length=1000)
    original_decision: str | None = Field(default=None, max_length=64)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class OverrideStore:
    """Journal append-only des overrides, en mémoire, injectable.

    Seule méthode de mutation : `record_override`. Aucune méthode de
    modification/suppression — même garantie structurelle qu'`AuditStore` :
    rien dans cette classe ne permet d'altérer ou de retirer un override
    déjà enregistré.
    """

    def __init__(self) -> None:
        self._overrides_by_case: dict[str, list[OverrideRecord]] = {}

    def __len__(self) -> int:
        return sum(len(records) for records in self._overrides_by_case.values())

    def record_override(self, record: OverrideRecord) -> None:
        """Ajoute un override déjà validé au journal — copie défensive à l'entrée."""
        chain = self._overrides_by_case.setdefault(record.case_id, [])
        chain.append(record.model_copy(deep=True))

    def read_by_case_id(self, case_id: str) -> tuple[OverrideRecord, ...]:
        """Retourne les overrides d'un dossier, dans l'ordre d'ajout.

        Dossier inconnu → tuple vide, jamais une erreur. Toujours des
        copies — muter la valeur retournée n'affecte jamais le journal.
        """
        return tuple(
            record.model_copy(deep=True) for record in self._overrides_by_case.get(case_id, ())
        )
