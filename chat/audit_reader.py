"""Lecteur d'audit — chat/audit_reader.py (plan V2 §6, Phase V2-11c).

Exception documentée à la convention de `chat/tools.py` (« jamais un accès
direct à `graph.*`/`agents.*`/`services.*` métier »), même patron que
`chat/simulation_engine.py` (V2-11b) : accède directement à
`services.audit_service.AuditService`, jamais via HTTP — aucun endpoint
`/v2/*` n'expose l'audit à ce stade, et le plan V2-11c n'autorise aucune
modification de fichier existant (`api/v2/claims.py` compris), donc pas de
nouvel endpoint créé pour cette phase.

`build_audit_summary` ne retourne **jamais** le contenu brut d'un
`outcome` d'événement (`schemas.audit.AuditEvent.outcome`) — uniquement des
compteurs, types d'événements et acteurs déjà structurés. C'est le critère
d'acceptation explicite de V2-11c.

**Contrainte opérationnelle réelle, non résolue dans cette phase** (même
nature que celle documentée dans `chat/simulation_engine.py` pour les
checkpoints) : en production, `build_audit_summary(..., audit_service=None)`
(chemin réel de `chat.tools.get_audit_summary`) construit son **propre**
`AuditService()` — une instance **distincte** de celle utilisée par
`graph/nodes_v2.py::make_audit_service_node` à l'intérieur du graphe compilé
par `api/v2/claims.py::build_v2_router()` (jamais partagée, jamais exposée
hors de sa fermeture). Le résumé serait donc systématiquement vide
(`event_count=0`) pour un dossier réellement soumis, sauf injection
explicite de la même instance — corrigerait `api/v2/claims.py`/
`graph/workflow_v2.py` pour partager une seule instance applicative, hors
périmètre de cette phase (« fichiers existants touchés : aucun »). Les
tests de ce module injectent explicitement le même `AuditService`.
"""
from __future__ import annotations

from collections import Counter

from chat.schemas import AuditSummary
from services.audit_service import AuditService

__all__ = ["build_audit_summary"]


def build_audit_summary(case_id: str, *, audit_service: AuditService | None = None) -> AuditSummary:
    """Résumé minimisé de l'historique d'audit d'un dossier — `event_count=0`,
    `chain_intact=True` pour un dossier inconnu (absence d'historique n'est
    jamais une anomalie, même convention que
    `services.audit_store.AuditStore.verify_claim_integrity`)."""
    service = audit_service if audit_service is not None else AuditService()
    export = service.export_for_auditor(case_id=case_id)

    event_type_counts = Counter(event.event_type.value for event in export.events)
    actors = sorted({event.actor for event in export.events})

    return AuditSummary(
        case_id=case_id,
        event_count=export.event_count,
        chain_intact=export.chain_intact,
        event_type_counts=dict(event_type_counts),
        actors=actors,
        issues_count=len(export.issues),
    )
