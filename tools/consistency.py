"""Détection générique de désaccords entre résultats déjà validés — tools/consistency.py.

Compare uniquement un champ structurel commun à plusieurs schémas de
résultat (``schemas.results``) : ``status: VerificationStatus``. Ne compare
jamais un champ métier propre à un domaine (actes cliniques, prescriptions,
doublons de facture, score de risque...) — cette détection reste
volontairement générique et n'anticipe ni ne duplique la logique de
``clinical_consistency_agent`` / ``fraud_detection_agent`` (étape 12,
toujours stubs), qui comparent des champs métier spécifiques.

Réutilise sans dupliquer ``DisagreementPoint`` (``schemas.results``, déjà
consommé par ``CaseReviewerResult.disagreements`` — étape 13, stub) : cette
détection alimente le même schéma en amont, avant qu'un humain ou un futur
``case_reviewer_agent`` n'ait à trancher.

Ne décide jamais quel résultat est correct : les fonctions de ce module se
contentent de signaler l'existence et la sévérité d'une divergence — à
charge de l'humain (revue HITL) ou d'un futur agent métier de choisir.
Aucun état n'est mutable, aucune E/S, aucun appel LLM.
"""
from __future__ import annotations

from typing import Literal, Sequence

from schemas.domain import VerificationStatus
from schemas.results import DisagreementPoint
from state.claim_state import ClaimState

Severity = Literal["minor", "critical"]

GENERIC_STATUS_FIELDS: tuple[str, ...] = (
    "privacy_result",
    "ocr_result",
    "fhir_result",
    "coding_result",
    "clinical_result",
    "fraud_result",
    "audit_result",
)
"""Champs ``ClaimState`` dont le schéma de résultat expose un ``status``
générique de type ``VerificationStatus`` — seuls ceux-là sont comparables
sans interprétation métier. Volontairement exclus : ``intake_result``
(``IntakeStatus``), ``security_result`` (``SecurityDecision``),
``identity_coverage_result`` (statuts imbriqués ``identity``/``coverage``,
pas de champ ``status`` de premier niveau) et ``review_result``
(``Recommendation``) — normaliser ces enums entre eux exigerait une règle
d'équivalence métier, hors du périmètre générique de ce module."""

_AGENT_LABEL_BY_RESULT_FIELD: dict[str, str] = {
    "privacy_result": "privacy",
    "ocr_result": "document_ocr",
    "fhir_result": "fhir_validator",
    "coding_result": "medical_coding",
    "clinical_result": "clinical_consistency",
    "fraud_result": "fraud_detection",
    "audit_result": "audit",
}
"""Étiquette lisible par champ ``ClaimState`` — purement cosmétique (pour
``DisagreementPoint.agent``), aucune dépendance vers ``orchestrator`` ou
``graph`` (évite tout couplage inutile de ce module bas niveau)."""

_IGNORED_STATUSES: frozenset[VerificationStatus] = frozenset({
    VerificationStatus.PENDING,
    VerificationStatus.NOT_EVALUATED,
})
"""Statuts intermédiaires ignorés — un agent qui n'a pas encore tranché
n'est pas en désaccord avec les autres, il n'a simplement pas d'avis."""

_STATUS_SEVERITY_RANK: dict[str, int] = {
    VerificationStatus.PASS.value: 0,
    VerificationStatus.NEEDS_REVIEW.value: 1,
    VerificationStatus.FAIL.value: 2,
}
"""Ordre générique de sévérité entre valeurs de ``VerificationStatus`` — ne
reflète aucune règle métier, sert uniquement à mesurer l'écart entre deux
conclusions déjà produites par des agents différents sur le même dossier."""

CRITICAL_SEVERITY_GAP = 2
"""Écart de sévérité (PASS vs FAIL) à partir duquel un désaccord est
qualifié de critique — deux agents affirment des conclusions strictement
opposées sur le même dossier."""


def _collect_generic_statuses(
    state: ClaimState, fields: Sequence[str]
) -> list[tuple[str, VerificationStatus]]:
    """Relève, pour chaque champ fourni, le ``status`` générique du résultat
    déjà présent dans ``state`` (résultat absent ou statut non conforme à
    ``VerificationStatus`` ignoré silencieusement — rien à comparer)."""
    collected: list[tuple[str, VerificationStatus]] = []
    for field_name in fields:
        result = state.get(field_name)
        if result is None:
            continue
        raw_status = getattr(result, "status", None)
        if raw_status is None:
            continue
        try:
            status = VerificationStatus(raw_status)
        except ValueError:
            continue
        if status not in _IGNORED_STATUSES:
            collected.append((field_name, status))
    return collected


def detect_result_disagreements(
    state: ClaimState, *, fields: Sequence[str] = GENERIC_STATUS_FIELDS
) -> tuple[DisagreementPoint, ...]:
    """Détecte les désaccords génériques entre résultats déjà validés.

    Prend le premier statut générique rencontré (ordre de ``fields``) comme
    référence de présentation — ``expected`` — et signale, pour chaque
    autre résultat, un ``DisagreementPoint`` si son statut diffère —
    ``observed``. Ce choix de référence est purement mécanique (premier
    résultat disponible) : il ne prétend jamais désigner le résultat
    "correct", seulement fournir un point de comparaison lisible pour la
    revue humaine.

    Moins de deux statuts génériques disponibles → aucun désaccord possible
    (rien à comparer), retourne un tuple vide.
    """
    collected = _collect_generic_statuses(state, fields)
    if len(collected) < 2:
        return ()

    _reference_field, reference_status = collected[0]

    points: list[DisagreementPoint] = []
    for field_name, status in collected[1:]:
        if status == reference_status:
            continue
        points.append(
            DisagreementPoint(
                agent=_AGENT_LABEL_BY_RESULT_FIELD.get(field_name, field_name),
                field="status",
                expected=reference_status.value,
                observed=status.value,
            )
        )
    return tuple(points)


def classify_disagreement_severity(point: DisagreementPoint) -> Severity:
    """Classe un désaccord déjà détecté en ``"minor"`` ou ``"critical"``.

    Générique : ne s'appuie que sur l'écart entre les rangs de
    ``VerificationStatus`` (``_STATUS_SEVERITY_RANK``), jamais sur le sens
    métier du champ concerné. Une valeur non reconnue (hors énumération)
    est traitée par prudence comme critique — jamais silencieusement
    ignorée."""
    expected_rank = _STATUS_SEVERITY_RANK.get(point.expected)
    observed_rank = _STATUS_SEVERITY_RANK.get(point.observed)
    if expected_rank is None or observed_rank is None:
        return "critical"
    gap = abs(expected_rank - observed_rank)
    return "critical" if gap >= CRITICAL_SEVERITY_GAP else "minor"


def has_critical_disagreement(disagreements: Sequence[DisagreementPoint]) -> bool:
    """``True`` si au moins un désaccord de la liste est critique."""
    return any(classify_disagreement_severity(point) == "critical" for point in disagreements)
