"""Fraud Detection Agent — ClaimShield Santé.

Détecte des anomalies évocatrices de fraude à partir des résultats déjà
validés dans le state (``identity_coverage_result``, ``coding_result``,
``ocr_result``) et de l'historique pseudonymisé des dossiers déjà soumis
(``services.duplicate_index``, via la vue antifraude minimisée déjà
pseudonymisée par ``privacy_agent`` — ``privacy_result.view``, rôle
``FRAUD_ANALYST``) — ne recalcule jamais l'identité, la couverture, la
codification ni l'identité pseudonymisée du patient, il les combine.

Agent LLM (gemma4:latest via ChatOllama, ReAct) + calcul déterministe du score.

Pipeline :
  Phase A — déterministe : combine les signaux d'anomalie déjà disponibles
            (identité NEEDS_REVIEW/FAIL, couverture inactive/expirée,
            plafond dépassé, préautorisation manquante, codification non
            résolue, confiance d'extraction faible, doublon exact ou
            quasi-doublon via ``services.duplicate_index``/``tools.statistics``)
            en un score de risque pondéré et un statut PASS/NEEDS_REVIEW/FAIL
            provisoires — jamais le LLM.
  Phase B — agent ReAct LLM (``create_react_agent``, appel obligatoire à
            chaque exécution) : interprète les doublons détectés, les
            montants atypiques (similarité de montant avec un dossier
            rapproché) et les signaux antifraude déjà calculés, avec pour
            seul outil autorisé ``verifier_doublon``
            (``agents/fraud_detection_agent/tools.py``, introspecté par
            ``orchestrator.policies.ALLOWED_TOOLS_PER_AGENT`` — aucun autre
            outil n'est physiquement joignable). Produit une justification
            explicative et, depuis P1-1, une **autorité réelle mais bornée**
            sur la pondération : ``LlmFraudDecision.signal_assessments``
            permet de proposer, pour un signal déjà calculé et déjà attribué
            à une preuve (jamais un signal inventé), un ajustement
            DOWNGRADE/NEUTRAL/UPGRADE avec justification obligatoire dès que
            l'ajustement s'écarte de NEUTRAL — ``agent.py::
            _apply_signal_assessments`` multiplie alors ``risk_contribution``
            par un facteur fixe (0.5/1.0/1.5, jamais choisi par le LLM) et
            ``_determine_status`` recalcule le statut final avec les mêmes
            seuils fixes. Le LLM ne fixe donc jamais lui-même une valeur
            numérique de score ni un statut — il ne choisit qu'une catégorie
            bornée sur un fait déjà établi par la Phase A, jamais un nouveau
            fait. Garantie structurelle inchangée par ailleurs : aucune
            accusation de fraude, aucun blocage définitif et aucune décision
            sans revue humaine ne peut jamais être introduite par le LLM
            (``human_review_required`` reste toujours dérivé du seul statut
            recalculé, jamais du LLM directement).
  Phase C — construction de ``FraudDetectionResult`` (validée Pydantic,
            ``extra='forbid'`` à tous les niveaux).

``duplicate_invoice`` (``FraudResultPayload``) reflète désormais un vrai
résultat de recherche de doublon quand la vue antifraude minimisée est
disponible (``True``/``False``) — ``None`` uniquement quand la vérification
elle-même n'a pas pu être menée (vue absente, hash ou montant manquant),
jamais une valeur inventée.

Interdictions strictes :
  - Aucune accusation de fraude : ni la Phase A ni le LLM ne qualifient un
    dossier de frauduleux — uniquement des signaux et des scores.
  - Aucun blocage définitif : un statut FAIL implique toujours
    ``human_review_required=True`` (dérivé du statut, jamais contournable).
  - Aucune décision sans humain : ce rôle appartient exclusivement à
    ``case_reviewer_agent``/HITL, jamais à cet agent ni à son LLM.
  - Aucun contenu brut de document, aucune donnée personnelle non
    pseudonymisée dans le résultat ni dans les données envoyées au LLM.

Conserve l'interface injectable (``FraudDetectionRunnable``, ``make_node``)
utilisée par ``graph.nodes.build_orchestrator(fraud_detection_impl=...)``
pour l'injection de tests — l'implémentation par défaut (``node``) exécute
désormais une évaluation réelle.
"""
from __future__ import annotations

import uuid
from datetime import date as _date
from decimal import Decimal, InvalidOperation
from typing import Callable, Protocol, runtime_checkable

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from llm.factory import get_llm
from llm.metadata import build_llm_metadata
from schemas.domain import ReaderRole, SeverityLevel, VerificationStatus
from schemas.results import (
    AuditEvent,
    FraudDetectionResult,
    FraudEvidence,
    FraudEvidenceSource,
    FraudResultPayload,
    FraudSignal,
    StructuredError,
)
from services.duplicate_index import ClaimFingerprint, DuplicateIndex
from state.claim_state import ClaimState, validate_state_update

try:
    from langgraph.prebuilt import create_react_agent
except ModuleNotFoundError:  # pragma: no cover - dépendance optionnelle en tests locaux
    def create_react_agent(*_args, **_kwargs):
        raise RuntimeError("langgraph indisponible")

from agents.fraud_detection_agent.prompt import load_fraud_detection_prompt
from agents.fraud_detection_agent.schemas import LlmFraudDecision, SignalAssessment
from agents.fraud_detection_agent.tools import _DEFAULT_DUPLICATE_INDEX, verifier_doublon
from agents.privacy_agent.schemas import FraudView

_STEP_NAME = "fraud_detection"
_AGENT_NAME = "fraud_detection_agent"
_THRESHOLD_VERSION = "1.0.0"

_NEEDS_REVIEW_THRESHOLD = 0.3
_FAIL_THRESHOLD = 0.7

_LOW_CONFIDENCE_THRESHOLD = 0.5

_EXACT_DUPLICATE_RISK_CONTRIBUTION = 0.5
_NEAR_DUPLICATE_RISK_CONTRIBUTION = 0.25

# P1-1 : pondération bornée du LLM sur des signaux déjà calculés — jamais un
# levier d'invention de signal. Constantes Phase A, jamais choisies par le
# LLM lui-même (il ne propose qu'une catégorie DOWNGRADE/NEUTRAL/UPGRADE).
_ADJUSTMENT_MULTIPLIER: dict[str, float] = {
    "DOWNGRADE": 0.5,
    "NEUTRAL": 1.0,
    "UPGRADE": 1.5,
}


# ── Interface ──────────────────────────────────────────────────────────────────


@runtime_checkable
class FraudDetectionRunnable(Protocol):
    """Interface minimale requise par le nœud LangGraph."""

    def run(self, state: ClaimState) -> FraudDetectionResult: ...


# ── Helpers déterministes ─────────────────────────────────────────────────────


def _status_of(result: object | None) -> VerificationStatus | None:
    if result is None:
        return None
    status = getattr(result, "status", None)
    return status if isinstance(status, VerificationStatus) else None


def _collect_signals(
    identity_coverage_result: object | None,
    coding_result: object | None,
    ocr_result: object | None,
) -> tuple[list[FraudSignal], bool | None]:
    """Phase A — combine les preuves déjà validées en signaux de risque pondérés.

    Retourne (signals, duplicate_invoice). ``duplicate_invoice`` reste ``None`` :
    aucun entrepôt d'historique des dossiers n'existe encore (voir docstring
    du module).
    """
    signals: list[FraudSignal] = []

    identity = getattr(identity_coverage_result, "identity", None) if identity_coverage_result else None
    identity_status = getattr(identity, "status", None) if identity is not None else None
    if identity_status == VerificationStatus.FAIL:
        signals.append(
            FraudSignal(
                signal_type="IDENTITY_MISMATCH",
                description="Identité patient non concordante (statut FAIL).",
                risk_contribution=0.4,
                evidence=[
                    FraudEvidence(
                        source=FraudEvidenceSource.IDENTITY_COVERAGE,
                        field="identity.status",
                        document_reference="identity_coverage_result",
                        value=identity_status.value,
                    )
                ],
            )
        )
    elif identity_status == VerificationStatus.NEEDS_REVIEW:
        signals.append(
            FraudSignal(
                signal_type="IDENTITY_AMBIGUOUS",
                description="Identité patient ambiguë (statut NEEDS_REVIEW).",
                risk_contribution=0.2,
                evidence=[
                    FraudEvidence(
                        source=FraudEvidenceSource.IDENTITY_COVERAGE,
                        field="identity.status",
                        document_reference="identity_coverage_result",
                        value=identity_status.value,
                    )
                ],
            )
        )

    coverage = getattr(identity_coverage_result, "coverage", None) if identity_coverage_result else None
    coverage_status = getattr(coverage, "status", None) if coverage is not None else None
    if coverage_status == VerificationStatus.FAIL:
        signals.append(
            FraudSignal(
                signal_type="COVERAGE_INACTIVE_OR_EXPIRED",
                description="Couverture assurance inactive ou expirée à la date de service.",
                risk_contribution=0.35,
                evidence=[
                    FraudEvidence(
                        source=FraudEvidenceSource.IDENTITY_COVERAGE,
                        field="coverage.status",
                        document_reference="identity_coverage_result",
                        value=coverage_status.value,
                    )
                ],
            )
        )

    if coverage is not None and getattr(coverage, "ceiling_exceeded", False):
        signals.append(
            FraudSignal(
                signal_type="CEILING_EXCEEDED",
                description="Montant demandé supérieur au plafond de garantie restant.",
                risk_contribution=0.25,
                evidence=[
                    FraudEvidence(
                        source=FraudEvidenceSource.IDENTITY_COVERAGE,
                        field="coverage.ceiling_exceeded",
                        document_reference="identity_coverage_result",
                        value="true",
                    )
                ],
            )
        )

    if coverage is not None and getattr(coverage, "preauthorization_required", False):
        preauth_status = str(getattr(coverage, "preauthorization_status", "") or "").casefold()
        if preauth_status not in {"approved", "present"}:
            signals.append(
                FraudSignal(
                    signal_type="PREAUTHORIZATION_MISSING",
                    description="Préautorisation requise mais absente ou non approuvée.",
                    risk_contribution=0.3,
                    evidence=[
                        FraudEvidence(
                            source=FraudEvidenceSource.IDENTITY_COVERAGE,
                            field="coverage.preauthorization_status",
                            document_reference="identity_coverage_result",
                            value=preauth_status or "absent",
                        )
                    ],
                )
            )

    coding_status = _status_of(coding_result)
    if coding_status in (VerificationStatus.NEEDS_REVIEW, VerificationStatus.FAIL):
        signals.append(
            FraudSignal(
                signal_type="UNRESOLVED_CODING",
                description=(
                    "Codification médicale non résolue — impossible de confirmer "
                    "la conformité des actes facturés au référentiel."
                ),
                risk_contribution=0.15,
                evidence=[
                    FraudEvidence(
                        source=FraudEvidenceSource.MEDICAL_CODING,
                        field="status",
                        document_reference="coding_result",
                        value=coding_status.value if coding_status is not None else "UNKNOWN",
                    )
                ],
            )
        )

    confidence_score = getattr(ocr_result, "confidence_score", None) if ocr_result is not None else None
    if confidence_score is not None and confidence_score < _LOW_CONFIDENCE_THRESHOLD:
        signals.append(
            FraudSignal(
                signal_type="LOW_EXTRACTION_CONFIDENCE",
                description=(
                    f"Confiance d'extraction OCR faible ({confidence_score:.2f}) — "
                    "données potentiellement non fiables."
                ),
                risk_contribution=0.15,
                evidence=[
                    FraudEvidence(
                        source=FraudEvidenceSource.OCR_EXTRACTION,
                        field="confidence_score",
                        document_reference="ocr_result",
                        value=f"{confidence_score:.2f}",
                    )
                ],
            )
        )

    return signals, None


def _determine_status(
    signals: list[FraudSignal],
    *,
    insufficient_evidence: bool,
) -> tuple[VerificationStatus, float]:
    risk_score = min(1.0, sum(s.risk_contribution for s in signals))
    if insufficient_evidence:
        return VerificationStatus.NEEDS_REVIEW, risk_score
    if risk_score >= _FAIL_THRESHOLD:
        return VerificationStatus.FAIL, risk_score
    if risk_score >= _NEEDS_REVIEW_THRESHOLD:
        return VerificationStatus.NEEDS_REVIEW, risk_score
    return VerificationStatus.PASS, risk_score


def _apply_signal_assessments(
    signals: list[FraudSignal],
    assessments: list[SignalAssessment],
) -> tuple[list[FraudSignal], list[str]]:
    """P1-1 — applique les ajustements de pondération LLM bornés.

    Pour chaque ``FraudSignal`` déjà calculé par la Phase A, si son
    ``signal_type`` apparaît dans ``assessments`` (sinon ignoré
    silencieusement — même garantie anti-hallucination que
    ``referenced_signal_types``), son ``risk_contribution`` est multiplié
    par le facteur fixe correspondant (jamais choisi par le LLM lui-même) et
    plafonné à 1.0. Chaque signal ajusté reste ancré à ses ``evidence``
    d'origine (jamais mutées, jamais recréées) — seule la pondération
    numérique change, jamais le fait sous-jacent.

    Retourne la liste des signaux (ajustés ou inchangés, même ordre) et les
    motifs d'ajustement effectivement appliqués (vide si ``assessments`` est
    vide ou ne référence que des types NEUTRAL/inconnus).
    """
    if not assessments:
        return signals, []

    assessment_by_type = {a.signal_type: a for a in assessments}
    adjusted: list[FraudSignal] = []
    notes: list[str] = []
    for signal in signals:
        assessment = assessment_by_type.get(signal.signal_type)
        if assessment is None or assessment.severity_adjustment == "NEUTRAL":
            adjusted.append(signal)
            continue
        multiplier = _ADJUSTMENT_MULTIPLIER[assessment.severity_adjustment]
        new_contribution = min(1.0, signal.risk_contribution * multiplier)
        adjusted.append(signal.model_copy(update={"risk_contribution": new_contribution}))
        notes.append(
            f"Pondération du signal {signal.signal_type!r} ajustée par le LLM "
            f"({assessment.severity_adjustment}, {signal.risk_contribution:.2f} → "
            f"{new_contribution:.2f}) : {assessment.rationale}"
        )
    return adjusted, notes


# ── Phase A (suite) : doublons — historique pseudonymisé seulement ──────────


def _extract_fraud_view(privacy_result: object | None) -> FraudView | None:
    """Récupère la vue antifraude déjà minimisée et pseudonymisée par
    ``privacy_agent`` (``privacy_result.view``, un dict JSON-sérialisable —
    ``PrivacyResult.view: dict | None``), jamais reconstruite ici à partir de
    données brutes. ``None`` si aucune vue FRAUD_ANALYST n'a été produite
    (rôle différent, vue non construite, ou privacy_result absent)."""
    if privacy_result is None:
        return None
    if getattr(privacy_result, "view_role", None) != ReaderRole.FRAUD_ANALYST.value:
        return None
    view_dict = getattr(privacy_result, "view", None)
    if not isinstance(view_dict, dict):
        return None
    try:
        return FraudView.model_validate(view_dict)
    except ValidationError:
        return None


def _parse_decimal(raw: str | None) -> Decimal | None:
    if not raw:
        return None
    try:
        return Decimal(str(raw))
    except (InvalidOperation, ValueError):
        return None


def _parse_iso_date(raw: str | None) -> _date | None:
    if not raw:
        return None
    try:
        return _date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def _resolve_document_hash(fraud_view: FraudView, ocr_result: object | None) -> str | None:
    """Préfère le hash déjà porté par la vue antifraude (``document_hashes``,
    prévue pour la comparaison historique) ; retombe sur ``ocr_result.sha256``
    (un hash n'est pas une donnée personnelle) si la vue n'en porte aucun."""
    if fraud_view.document_hashes:
        return fraud_view.document_hashes.get("invoice") or next(
            iter(fraud_view.document_hashes.values())
        )
    sha256 = getattr(ocr_result, "sha256", None)
    return sha256 if isinstance(sha256, str) else None


def _check_duplicate(
    case_id: str,
    fraud_view: FraudView | None,
    ocr_result: object | None,
    duplicate_index: DuplicateIndex,
) -> tuple[FraudSignal | None, bool | None, str]:
    """Vérifie un doublon exact ou un quasi-doublon via l'historique
    pseudonymisé (``services.duplicate_index``) — jamais une accusation de
    fraude, uniquement un signal structurel attribué.

    Retourne ``(signal_ou_None, duplicate_invoice, reason)``.
    ``duplicate_invoice`` est ``None`` uniquement quand la vérification
    elle-même n'a pas pu être menée (vue absente, hash ou montant
    manquant/invalide) — jamais une valeur inventée.
    """
    if fraud_view is None:
        return None, None, "Vue antifraude minimisée indisponible : doublon non évaluable."

    document_hash = _resolve_document_hash(fraud_view, ocr_result)
    amount = _parse_decimal(fraud_view.amount_requested) or _parse_decimal(fraud_view.total_billed)
    if document_hash is None or amount is None:
        return None, None, "Hash de document ou montant indisponible : doublon non évaluable."

    try:
        fingerprint = ClaimFingerprint(
            case_id=case_id,
            document_hash=document_hash,
            patient_pseudonym=fraud_view.patient_pseudonym,
            amount=amount,
            service_date=_parse_iso_date(fraud_view.service_date),
            description=fraud_view.invoice_reference or "",
        )
    except ValidationError:
        return None, None, "Vue antifraude invalide : doublon non évaluable."

    check_result = duplicate_index.check(fingerprint)
    duplicate_index.register(fingerprint)

    if not check_result.matches:
        return None, False, "Aucun doublon détecté dans l'historique pseudonymisé disponible."

    best_match = max(check_result.matches, key=lambda m: m.similarity_score)
    is_exact = check_result.has_exact_duplicate
    signal = FraudSignal(
        signal_type="EXACT_DUPLICATE_INVOICE" if is_exact else "NEAR_DUPLICATE_INVOICE",
        description=(
            "Document facturé strictement identique à un autre dossier déjà soumis."
            if is_exact
            else "Dossier proche (montant, description, date) d'un autre dossier déjà soumis "
            "pour le même patient."
        ),
        risk_contribution=(
            _EXACT_DUPLICATE_RISK_CONTRIBUTION if is_exact else _NEAR_DUPLICATE_RISK_CONTRIBUTION
        ),
        severity=SeverityLevel.CRITICAL if is_exact else SeverityLevel.MEDIUM,
        evidence=[
            FraudEvidence(
                source=FraudEvidenceSource.DUPLICATE_INDEX,
                field="matched_case_id",
                document_reference="duplicate_index",
                value=best_match.matched_case_id,
            ),
            FraudEvidence(
                source=FraudEvidenceSource.DUPLICATE_INDEX,
                field="similarity_score",
                document_reference="duplicate_index",
                value=f"{best_match.similarity_score:.2f}",
            ),
        ],
    )
    return signal, True, "Doublon détecté dans l'historique pseudonymisé — voir signal."


# ── Phase B : agent ReAct LLM ─────────────────────────────────────────────────


def _invoke_llm_fraud(data: dict) -> LlmFraudDecision | None:
    """Lance l'agent ReAct LLM (appel obligatoire à chaque exécution) pour
    une justification — jamais de verdict de fraude, d'accusation, de
    blocage définitif ni de décision sans revue humaine. Seul outil
    physiquement joignable : ``verifier_doublon``
    (``ALLOWED_TOOLS_PER_AGENT[AgentName.FRAUD_DETECTION]``)."""
    try:
        prompt = load_fraud_detection_prompt()
        llm = get_llm()
        agent = create_react_agent(
            model=llm,
            tools=[verifier_doublon],
            response_format=LlmFraudDecision,
        )
        result = agent.invoke(
            {
                "messages": [
                    SystemMessage(content=prompt.system_prompt),
                    HumanMessage(content=str(data)),
                ]
            }
        )
        structured = result.get("structured_response")
        if isinstance(structured, LlmFraudDecision):
            return structured
        if isinstance(structured, dict):
            return LlmFraudDecision(**structured)
        return None
    except Exception:
        return None


# ── Fonction principale (testable sans LangGraph) ─────────────────────────────


def _merge_llm_decision(
    llm_decision: LlmFraudDecision | None,
    reasons: list[str],
    *,
    known_signal_types: set[str],
) -> list[str]:
    """Fusionne la décision LLM dans les motifs narratifs uniquement — ne
    touche jamais elle-même au score de risque, au statut ou au besoin de
    revue. L'unique canal d'influence du LLM sur le score
    (``signal_assessments``) est appliqué séparément, avant cet appel, par
    ``_apply_signal_assessments``/``_determine_status`` — cette fonction-ci
    ne fait que fusionner les motifs textuels autour du résultat déjà figé.

    Tout signal (``referenced_signal_types``) cité par le LLM mais absent des
    signaux réellement calculés est silencieusement ignoré — jamais une
    accusation non prouvée acceptée telle quelle. ``llm_risk_perception`` et
    ``suggests_human_review`` restent purement informatifs : ils n'écrasent
    jamais ``risk_score`` ni ``human_review_required``.
    """
    reasons = list(reasons)
    if llm_decision is None:
        reasons.append(
            "LLM indisponible : score de risque déterministe conservé sans justification enrichie."
        )
        return reasons

    if llm_decision.rationale:
        reasons.append(llm_decision.rationale)
    reasons.extend(llm_decision.reasons)

    unknown_signals = [
        s for s in llm_decision.referenced_signal_types if s not in known_signal_types
    ]
    if unknown_signals:
        reasons.append(
            "LLM a référencé des signaux inexistants — références ignorées "
            "(aucune accusation non prouvée acceptée)."
        )

    if llm_decision.suggests_human_review:
        reasons.append(
            "Le LLM signale un besoin de revue complémentaire (information, non "
            "contraignante — statut et besoin de revue déterministes conservés)."
        )

    if llm_decision.llm_risk_perception is not None:
        reasons.append(
            f"Risque perçu par le LLM : {llm_decision.llm_risk_perception:.2f} "
            "(n'affecte pas le score de risque déterministe)."
        )

    return reasons


def run(
    case_id: str,
    identity_coverage_result: object | None = None,
    coding_result: object | None = None,
    ocr_result: object | None = None,
    fraud_view: object | None = None,
    duplicate_index: DuplicateIndex | None = None,
) -> FraudDetectionResult:
    """Évalue le risque de fraude d'un dossier à partir des preuves déjà validées.

    Args:
        case_id: identifiant du dossier.
        identity_coverage_result: ``IdentityCoverageResult | None``.
        coding_result: ``MedicalCodingResult | None``.
        ocr_result: ``DocumentOcrResult | None`` (confiance d'extraction).
        fraud_view: ``FraudView | None`` — vue antifraude déjà minimisée et
            pseudonymisée par ``privacy_agent`` (``privacy_result.view``),
            seule source utilisée pour la recherche de doublon — jamais
            reconstruite ici à partir de données brutes.
        duplicate_index: ``DuplicateIndex | None`` — index injecté (tests) ;
            ``None`` retombe sur l'index partagé par défaut
            (``agents.fraud_detection_agent.tools._DEFAULT_DUPLICATE_INDEX``).

    Returns:
        ``FraudDetectionResult`` avec statut PASS / NEEDS_REVIEW / FAIL.
    """
    insufficient_evidence = (
        identity_coverage_result is None and coding_result is None and ocr_result is None
    )
    signals, _ = _collect_signals(identity_coverage_result, coding_result, ocr_result)

    index = duplicate_index if duplicate_index is not None else _DEFAULT_DUPLICATE_INDEX
    duplicate_signal, duplicate_invoice, duplicate_reason = _check_duplicate(
        case_id, fraud_view if isinstance(fraud_view, FraudView) else None, ocr_result, index
    )
    if duplicate_signal is not None:
        signals.append(duplicate_signal)

    status, risk_score = _determine_status(signals, insufficient_evidence=insufficient_evidence)

    reasons: list[str] = []
    if insufficient_evidence:
        reasons.append(
            "Aucune preuve d'identité, de couverture, de codification ou d'extraction "
            "disponible : évaluation anti-fraude non fiable."
        )
    elif not signals:
        reasons.append("Aucun signal de fraude détecté dans les preuves disponibles.")
    else:
        reasons.append(f"{len(signals)} signal(aux) de risque combiné(s) — score {risk_score:.2f}.")
    reasons.append(duplicate_reason)

    evidence_ids = [evidence.evidence_id for signal in signals for evidence in signal.evidence]
    known_signal_types = {s.signal_type for s in signals}

    llm_decision = _invoke_llm_fraud(
        {
            "case_id": case_id,
            "status": status.value,
            "risk_score": risk_score,
            "signal_types": sorted(known_signal_types),
            "signaux_detailles": [
                {"signal_type": s.signal_type, "risk_contribution": round(s.risk_contribution, 2)}
                for s in signals
            ],
            "doublons": {
                "duplicate_invoice": duplicate_invoice,
                "has_exact_duplicate": duplicate_signal is not None
                and duplicate_signal.signal_type == "EXACT_DUPLICATE_INVOICE",
                "has_near_duplicate": duplicate_signal is not None
                and duplicate_signal.signal_type == "NEAR_DUPLICATE_INVOICE",
            },
            "montant": {
                "amount_requested": str(fraud_view.amount_requested)
                if isinstance(fraud_view, FraudView) and fraud_view.amount_requested
                else None,
            },
            "signal_evidence_ids": evidence_ids,
            "instruction": (
                "Interprète les doublons, les montants atypiques et les signaux "
                "antifraude déjà calculés. Tu ne fixes jamais toi-même le score de "
                "risque ni le statut : ton seul levier est signal_assessments, un "
                "ajustement borné (DOWNGRADE/NEUTRAL/UPGRADE) sur un signal déjà "
                "calculé, avec justification obligatoire dès que tu t'écartes de "
                "NEUTRAL — jamais accuser toi-même de fraude, jamais bloquer "
                "définitivement un dossier, et jamais décider sans revue humaine. Ne "
                "cite que des signal_types déjà présents ci-dessus."
            ),
        }
    )

    # P1-1 : ajustement borné de pondération — recalcul déterministe du
    # statut/score si le LLM a proposé un ajustement sur un signal réel.
    signals, adjustment_notes = _apply_signal_assessments(
        signals, llm_decision.signal_assessments if llm_decision is not None else []
    )
    if adjustment_notes:
        status, risk_score = _determine_status(signals, insufficient_evidence=insufficient_evidence)
        reasons.extend(adjustment_notes)
        evidence_ids = [evidence.evidence_id for signal in signals for evidence in signal.evidence]

    reasons = _merge_llm_decision(llm_decision, reasons, known_signal_types=known_signal_types)
    errors = (
        [
            StructuredError(
                code="LLM_UNAVAILABLE",
                message="LLM indisponible ou réponse invalide : score déterministe conservé.",
                field="llm_trace",
            )
        ]
        if llm_decision is None
        else []
    )

    confidence = 0.5 if insufficient_evidence else max(0.4, 1.0 - 0.1 * len(signals))

    return FraudDetectionResult(
        case_id=case_id,
        status=status,
        llm_trace=build_llm_metadata(_AGENT_NAME, confidence=confidence),
        confidence=confidence,
        errors=errors,
        evidence_ids=evidence_ids,
        human_review_required=status is not VerificationStatus.PASS,
        result_payload=FraudResultPayload(
            duplicate_invoice=duplicate_invoice,
            risk_score=risk_score,
            signals=signals,
            threshold_version=_THRESHOLD_VERSION,
            reasons=reasons,
        ),
    )


# ── Implémentation par défaut (réelle) ────────────────────────────────────────


class _RealImplementation:
    """Adapte ``run()`` à l'interface ``FraudDetectionRunnable``."""

    def run(self, state: ClaimState) -> FraudDetectionResult:
        case_id = str(state.get("case_id", "UNKNOWN"))
        return run(
            case_id=case_id,
            identity_coverage_result=state.get("identity_coverage_result"),
            coding_result=state.get("coding_result"),
            ocr_result=state.get("ocr_result"),
            fraud_view=_extract_fraud_view(state.get("privacy_result")),
        )


_DEFAULT_IMPL: FraudDetectionRunnable = _RealImplementation()


# ── Factory et nœud LangGraph ─────────────────────────────────────────────────


def make_node(
    impl: FraudDetectionRunnable = _DEFAULT_IMPL,
) -> Callable[[ClaimState], dict]:
    """Crée un nœud LangGraph avec l'implémentation injectable fournie."""
    def _node(state: ClaimState) -> dict:
        result = impl.run(state)
        case_id = str(state.get("case_id", result.case_id))
        llm_call_id = str(uuid.uuid4())
        audit = AuditEvent(
            event_id=str(uuid.uuid4()),
            case_id=case_id,
            actor=_AGENT_NAME,
            action="fraud_detection_check",
            outcome=result.status.value,
            details={
                "risk_score": f"{result.result_payload.risk_score:.2f}",
                "signal_count": str(len(result.result_payload.signals)),
                "threshold_version": result.result_payload.threshold_version,
                "llm_call_id": llm_call_id,
                "model_name": result.llm_trace.model_name,
                "prompt_version": result.llm_trace.prompt_version,
                "tools": verifier_doublon.name,
                "errors": ",".join(e.code for e in result.errors),
            },
        )
        updates: dict = {
            "fraud_result": result,
            "current_step": _STEP_NAME,
            "completed_steps": [_STEP_NAME],
            "audit_trail": [audit],
        }
        if result.status is VerificationStatus.FAIL:
            updates["errors"] = [
                f"[{_AGENT_NAME}] {r}" for r in result.result_payload.reasons
            ]
        elif result.status in (VerificationStatus.NEEDS_REVIEW, VerificationStatus.NOT_EVALUATED):
            updates["alerts"] = [
                f"Détection fraude : {result.status.value} — "
                f"score={result.result_payload.risk_score:.2f}"
            ]
        validate_state_update(updates)
        return updates

    _node.__name__ = f"node_{_STEP_NAME}"
    return _node


# Nœud stable — nom utilisé comme clé dans le StateGraph.
node = make_node()
