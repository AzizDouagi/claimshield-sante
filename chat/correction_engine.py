"""Moteur de correction — chat/correction_engine.py (plan V2 §6, Phase V2-11a).

Fonction pure : associe les motifs `errors`/`alerts` déjà présents dans une
réponse API v2 minimisée (`dict` JSON) à une action corrective, via une
table déterministe fixe — jamais une action inventée par un LLM (voir
`chat/response_composer.py`, qui ne fait que reformuler ces recommandations
déjà calculées, jamais en proposer de nouvelles).

La table associe des motifs (sous-chaînes, recherche insensible à la casse)
aux textes déjà présents dans `errors`/`alerts` de
`api.v2.schemas.ClaimStatusResponseV2` — ces motifs proviennent des raisons
déjà produites par les agents V2 (`agents/eligibility_agent`,
`agents/medical_risk_agent`, `agents/document_understanding_agent`), jamais
recalculés ici.
"""
from __future__ import annotations

from chat.schemas import CorrectionRecommendation

__all__ = ["build_corrections"]

_CORRECTION_TABLE: tuple[tuple[str, str], ...] = (
    (
        "assureur",
        "Fournir un document mentionnant clairement le nom de l'assureur "
        "(ex. demande de remboursement).",
    ),
    (
        "payer_name",
        "Fournir un document mentionnant clairement le nom de l'assureur "
        "(ex. demande de remboursement).",
    ),
    (
        "nom patient",
        "Vérifier que le nom du patient apparaît lisiblement sur les "
        "documents transmis.",
    ),
    (
        "montant demandé",
        "Fournir le montant demandé, visible sur la facture ou la demande "
        "de remboursement.",
    ),
    (
        "amount_requested",
        "Fournir le montant demandé, visible sur la facture ou la demande "
        "de remboursement.",
    ),
    (
        "codification",
        "Vérifier la description de l'acte ou du médicament (ex. sur "
        "l'ordonnance) pour permettre une codification exacte.",
    ),
    (
        "unresolved_coding",
        "Vérifier la description de l'acte ou du médicament (ex. sur "
        "l'ordonnance) pour permettre une codification exacte.",
    ),
    (
        "date",
        "Vérifier la cohérence des dates entre les documents fournis "
        "(ordonnance, facture, demande de remboursement).",
    ),
    (
        "identité",
        "Confirmer l'identité du patient (pièce d'identité, numéro de dossier).",
    ),
    (
        "confiance d'extraction",
        "Fournir un document plus lisible (meilleure qualité de scan ou de photo).",
    ),
    (
        "préautorisation",
        "Fournir le document de préautorisation requis.",
    ),
    (
        "plafond",
        "Vérifier le plafond de couverture restant auprès de l'assureur.",
    ),
    (
        "doublon",
        "Vérifier qu'il ne s'agit pas d'une facture déjà soumise précédemment.",
    ),
)


def build_corrections(context: dict) -> list[CorrectionRecommendation]:
    """Une seule recommandation par action distincte — un même motif ne
    produit jamais deux fois la même action, même s'il apparaît plusieurs
    fois dans `errors`/`alerts`."""
    texts = [*(context.get("errors") or []), *(context.get("alerts") or [])]
    recommendations: list[CorrectionRecommendation] = []
    seen_actions: set[str] = set()

    for text in texts:
        lowered = text.lower()
        for keyword, action in _CORRECTION_TABLE:
            if keyword in lowered and action not in seen_actions:
                recommendations.append(CorrectionRecommendation(trigger=text, action=action))
                seen_actions.add(action)

    return recommendations
