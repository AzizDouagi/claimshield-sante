"""Fonctions de routage conditionnel du workflow LangGraph — ClaimShield Santé.

Chaque fonction est pure : elle lit le state et retourne un nom de route connu.
Aucun effet de bord, aucune logique métier, aucune importation d'agent.

Routes disponibles
------------------
``continue``     — nœud suivant dans le pipeline nominal.
``quarantine``   — fichier/dossier mis en quarantaine ; revue humaine bloquante.
``needs_review`` — pipeline poursuit, une vérification humaine est signalée.
``retry``        — erreur transitoire sur un nœud ; peut être relancé.
``failure``      — erreur irrécupérable ; fin du pipeline sans approbation.
``end``          — pipeline terminé ; recommandation finale disponible.

Utilisation dans workflow.py
----------------------------
    from graph.edges import route_intake, CONTINUE, QUARANTINE, FAILURE, RETRY

    graph.add_conditional_edges(
        "claim_intake",
        route_intake,
        {CONTINUE: "security_gate", QUARANTINE: "human_review",
         FAILURE: END, RETRY: "claim_intake"},
    )
"""
from __future__ import annotations

from typing import Literal

from schemas.domain import (
    IntakeStatus,
    PrivacyDecision,
    Recommendation,
    SecurityDecision,
    VerificationStatus,
)
from state.claim_state import ClaimState

# ── Noms de routes stables ─────────────────────────────────────────────────────

Route = Literal["continue", "quarantine", "needs_review", "retry", "failure", "end"]

CONTINUE: Route = "continue"
QUARANTINE: Route = "quarantine"
NEEDS_REVIEW: Route = "needs_review"
RETRY: Route = "retry"
FAILURE: Route = "failure"
END: Route = "end"

ALL_ROUTES: frozenset[Route] = frozenset({
    CONTINUE, QUARANTINE, NEEDS_REVIEW, RETRY, FAILURE, END,
})


# ── Helper interne ────────────────────────────────────────────────────────────


def _route_by_verification_status(raw_status: object) -> Route:
    """Traduit un VerificationStatus brut en route.

    PASS → continue · NEEDS_REVIEW → needs_review · FAIL → failure
    PENDING / NOT_EVALUATED → retry (état intermédiaire inattendu en sortie)
    """
    try:
        status = VerificationStatus(raw_status)
    except ValueError:
        return FAILURE

    if status is VerificationStatus.PASS:
        return CONTINUE
    if status is VerificationStatus.NEEDS_REVIEW:
        return NEEDS_REVIEW
    if status is VerificationStatus.FAIL:
        return FAILURE
    # PENDING ou NOT_EVALUATED : le nœud n'a pas encore produit de décision finale
    return RETRY


# ── Fonctions de routage ──────────────────────────────────────────────────────


def route_intake(state: ClaimState) -> Route:
    """Route après claim_intake_agent.

    Lit ``intake_status`` en priorité (promu par le nœud), puis se rabat sur
    ``intake_result.status`` si le champ promu est absent.

    ACCEPTED    → continue    (dossier complet, prêt pour le Security Gate)
    QUARANTINED → quarantine  (fichier suspect, revue humaine bloquante)
    BLOCKED     → failure     (dossier rejeté définitivement)
    ERROR       → retry       (erreur de stockage ou transitoire)
    """
    raw = state.get("intake_status")
    if raw is None:
        result = state.get("intake_result")
        raw = result.status if result is not None else None
    if raw is None:
        return FAILURE

    try:
        status = IntakeStatus(raw)
    except ValueError:
        return FAILURE

    if status is IntakeStatus.ACCEPTED:
        return CONTINUE
    if status is IntakeStatus.QUARANTINED:
        return QUARANTINE
    if status is IntakeStatus.BLOCKED:
        return FAILURE
    if status is IntakeStatus.ERROR:
        return RETRY
    return FAILURE


def route_security(state: ClaimState) -> Route:
    """Route après security_gate_agent.

    Lit ``security_result.decision``.

    ALLOW     → continue    (contenu sûr, pipeline autorisé)
    QUARANTINE → quarantine (contenu suspect, isolement requis)
    BLOCK     → failure     (menace détectée, pipeline arrêté)
    """
    result = state.get("security_result")
    if result is None:
        return FAILURE

    try:
        decision = SecurityDecision(result.decision)
    except ValueError:
        return FAILURE

    if decision is SecurityDecision.ALLOW:
        return CONTINUE
    if decision is SecurityDecision.QUARANTINE:
        return QUARANTINE
    return FAILURE  # BLOCK ou valeur inconnue


def route_privacy(state: ClaimState) -> Route:
    """Route après privacy_agent.

    Lit ``privacy_result.decision`` (champ calculé @computed_field).

    ALLOW → continue  (vue minimisée produite, accès accordé)
    BLOCK → failure   (rôle absent/inconnu, clé manquante, violation RBAC)
    """
    result = state.get("privacy_result")
    if result is None:
        return FAILURE

    try:
        decision = PrivacyDecision(result.decision)
    except ValueError:
        return FAILURE

    if decision is PrivacyDecision.ALLOW:
        return CONTINUE
    return FAILURE  # BLOCK ou valeur inconnue


def route_ocr(state: ClaimState) -> Route:
    """Route après document_ocr_agent selon ``ocr_result.status``."""
    result = state.get("ocr_result")
    if result is None:
        return FAILURE
    return _route_by_verification_status(result.status)


def route_fhir(state: ClaimState) -> Route:
    """Route après fhir_validator_agent selon ``fhir_result.status``.

    NOT_EVALUATED est retourné quand aucun bundle FHIR n'est attendu
    (bundle_expected=False) ; le helper le traduit en retry, que les arêtes
    du graphe peuvent rediriger selon le contexte.
    """
    result = state.get("fhir_result")
    if result is None:
        return FAILURE
    return _route_by_verification_status(result.status)


def route_identity_coverage(state: ClaimState) -> Route:
    """Route après identity_coverage_agent.

    Consolide ``identity.status`` et ``coverage.status`` :
    FAIL sur l'un       → failure
    NEEDS_REVIEW sur l'un → needs_review
    Les deux PASS       → continue
    Autre combinaison   → retry
    """
    result = state.get("identity_coverage_result")
    if result is None:
        return FAILURE

    try:
        id_status = VerificationStatus(result.identity.status)
        cov_status = VerificationStatus(result.coverage.status)
    except ValueError:
        return FAILURE

    if id_status is VerificationStatus.FAIL or cov_status is VerificationStatus.FAIL:
        return FAILURE
    if id_status is VerificationStatus.NEEDS_REVIEW or cov_status is VerificationStatus.NEEDS_REVIEW:
        return NEEDS_REVIEW
    if id_status is VerificationStatus.PASS and cov_status is VerificationStatus.PASS:
        return CONTINUE
    # PENDING ou NOT_EVALUATED sur l'un ou l'autre
    return RETRY


def route_coding(state: ClaimState) -> Route:
    """Route après medical_coding_agent selon ``coding_result.status``."""
    result = state.get("coding_result")
    if result is None:
        return FAILURE
    return _route_by_verification_status(result.status)


def route_verification_fan_in(state: ClaimState) -> Route:
    """Route après la phase de vérification parallèle (OCR, FHIR, coding, identité).

    Consolide tous les résultats disponibles dans le state :
    - Un seul FAIL suffit → failure.
    - Au moins un NEEDS_REVIEW (sans FAIL) → needs_review.
    - Aucun résultat exploitable (tous None ou NOT_EVALUATED) → failure.
    - Tous PASS ou mélange PASS/NOT_EVALUATED → continue.

    NOT_EVALUATED est ignoré dans la consolidation (non applicable pour ce
    dossier) mais ne déclenche pas de failure à lui seul.
    """
    statuses: list[VerificationStatus] = []

    for result in (
        state.get("ocr_result"),
        state.get("fhir_result"),
        state.get("coding_result"),
    ):
        if result is not None:
            try:
                s = VerificationStatus(result.status)
                if s is not VerificationStatus.NOT_EVALUATED:
                    statuses.append(s)
            except ValueError:
                return FAILURE

    id_cov = state.get("identity_coverage_result")
    if id_cov is not None:
        try:
            for raw in (id_cov.identity.status, id_cov.coverage.status):
                s = VerificationStatus(raw)
                if s is not VerificationStatus.NOT_EVALUATED:
                    statuses.append(s)
        except ValueError:
            return FAILURE

    if not statuses:
        return FAILURE

    if VerificationStatus.FAIL in statuses:
        return FAILURE
    if VerificationStatus.NEEDS_REVIEW in statuses:
        return NEEDS_REVIEW
    return CONTINUE


def route_review(state: ClaimState) -> Route:
    """Route après case_reviewer_agent.

    APPROVE + human_review_required=False → end          (approbation définitive)
    APPROVE + human_review_required=True  → needs_review (validation HITL requise)
    PENDING                               → needs_review (en attente d'information)
    REJECT                                → end          (rejet définitif)
    """
    result = state.get("review_result")
    if result is None:
        return FAILURE

    try:
        rec = Recommendation(result.recommendation)
    except ValueError:
        return FAILURE

    if rec is Recommendation.APPROVE:
        return NEEDS_REVIEW if result.human_review_required else END
    if rec is Recommendation.REJECT:
        return END
    if rec is Recommendation.PENDING:
        return NEEDS_REVIEW
    return FAILURE
