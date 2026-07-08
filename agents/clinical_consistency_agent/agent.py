"""Clinical Consistency Agent — ClaimShield Santé.

Vérifie la cohérence clinique entre les documents extraits par l'OCR
(``ocr_result``), la codification médicale (``coding_result``), la
validation FHIR (``fhir_result``) et la vue médicale minimisée déjà produite
par ``privacy_agent`` (``privacy_result.view``), tous déjà présents dans le
state à ce stade du pipeline.

Agent LLM (gemma4:latest via ChatOllama, ReAct) + vérifications déterministes.

Pipeline :
  Phase A — déterministe : compte les actes/médicaments extraits, compare au
            nombre de codes SNOMED-CT/RxNorm résolus, vérifie la présence
            d'une ordonnance si des médicaments sont facturés, vérifie la
            date de service et la chronologie ordonnance/soin
            (``tools.date_checks``). Produit les ``ClinicalSignal`` et un
            statut provisoire — jamais le LLM.
  Phase B — agent ReAct LLM (``create_react_agent``, appel obligatoire à
            chaque exécution) : analyse la chronologie, l'ordonnance, l'acte,
            la codification et le résumé FHIR minimisé déjà calculés en
            Phase A, avec pour seul outil autorisé
            ``verifier_chronologie`` (``agents/clinical_consistency_agent/tools.py``,
            introspecté par ``orchestrator.policies.ALLOWED_TOOLS_PER_AGENT`` —
            aucun autre outil n'est physiquement joignable) et, si
            disponible, la vue médicale minimisée (``MedicalView`` —
            pseudonymisée par ``privacy_agent``, jamais de donnée brute).
            Produit un contexte explicatif et, depuis P1-2, une **autorité
            réelle mais bornée** sur la sévérité : ``LlmClinicalDecision.
            severity_assessments`` permet de proposer, pour un signal déjà
            calculé et déjà attribué à une preuve (jamais un signal
            inventé), un ajustement de sévérité borné à un cran maximum sur
            le lattice ``SeverityLevel`` (CRITICAL > HIGH > MEDIUM > LOW >
            INFO), avec justification obligatoire — ``agent.py::
            _apply_signal_assessments`` applique l'ajustement seulement s'il
            reste dans la borne, ``_status_from_signals`` recalcule alors le
            statut final avec la même règle fixe (``any(CRITICAL) → FAIL``
            sinon ``NEEDS_REVIEW``). Le LLM ne fixe donc jamais lui-même un
            statut, un diagnostic ni une recommandation — il ne choisit
            qu'une sévérité bornée sur un fait déjà établi par la Phase A,
            jamais un nouveau fait. Garantie structurelle inchangée par
            ailleurs : aucune décision médicale, décision finale ou
            affirmation non prouvée par les signaux Phase A ne peut jamais
            être introduite par le LLM.
  Phase C — construction de ``ClinicalConsistencyResult`` (validée Pydantic,
            ``extra='forbid'`` à tous les niveaux).

Interdictions strictes :
  - Aucun diagnostic médical, aucune décision finale : le LLM ne fait
    qu'expliquer/contextualiser des signaux déjà calculés, jamais décider.
  - Aucune affirmation non prouvée : toute anomalie mentionnée provient d'un
    ``ClinicalSignal`` déjà attribué (``evidence_id``/``source``/``field``).
  - Aucune décision de remboursement ou de fraude (rôle de fraud_detection_agent
    et case_reviewer_agent).
  - Aucun contenu brut de document, aucun texte OCR complet, aucun bundle
    FHIR brut dans le résultat ni dans les données envoyées au LLM.

Conserve l'interface injectable (``ClinicalConsistencyRunnable``,
``make_node``) utilisée par ``graph.nodes.build_orchestrator(
clinical_consistency_impl=...)`` pour l'injection de tests — l'implémentation
par défaut (``node``) exécute désormais une évaluation réelle.
"""
from __future__ import annotations

import json
import uuid
from typing import Callable, Protocol, runtime_checkable

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from config.logging import get_logger
from llm.factory import get_llm
from llm.metadata import build_llm_metadata
from schemas.domain import ReaderRole, SeverityLevel, VerificationStatus
from schemas.results import (
    AuditEvent,
    ClinicalConsistencyResult,
    ClinicalEvidence,
    ClinicalEvidenceSource,
    ClinicalInconsistency,
    ClinicalResultPayload,
    ClinicalSignal,
    StructuredError,
)
from state.claim_state import ClaimState, validate_state_update
from tools.date_checks import run_date_checks

try:
    from langgraph.prebuilt import create_react_agent
except ModuleNotFoundError:  # pragma: no cover - dépendance optionnelle en tests locaux
    def create_react_agent(*_args, **_kwargs):
        raise RuntimeError("langgraph indisponible")

from agents.clinical_consistency_agent.prompt import load_clinical_consistency_prompt
from agents.clinical_consistency_agent.schemas import ClinicalSignalAssessment, LlmClinicalDecision
from agents.clinical_consistency_agent.tools import verifier_chronologie
from agents.privacy_agent.schemas import MedicalView

_STEP_NAME = "clinical_consistency"
_AGENT_NAME = "clinical_consistency_agent"

logger = get_logger(__name__)

# P1-2 : lattice de sévérité — le LLM ne peut jamais proposer un écart de
# plus d'un cran par rapport à la sévérité déjà calculée par la Phase A.
_SEVERITY_ORDER: tuple[SeverityLevel, ...] = (
    SeverityLevel.CRITICAL,
    SeverityLevel.HIGH,
    SeverityLevel.MEDIUM,
    SeverityLevel.LOW,
    SeverityLevel.INFO,
)


# ── Interface ──────────────────────────────────────────────────────────────────


@runtime_checkable
class ClinicalConsistencyRunnable(Protocol):
    """Interface minimale requise par le nœud LangGraph.

    Lit l'état partagé (résultats OCR, codification déjà présents) et
    retourne un ``ClinicalConsistencyResult`` structuré.
    """

    def run(self, state: ClaimState) -> ClinicalConsistencyResult: ...


# ── Helpers déterministes ─────────────────────────────────────────────────────


def _extract_field(fields: dict, field_name: str) -> str | None:
    """Extrait la valeur d'un champ depuis extracted_fields (objet ou dict brut)."""
    field = fields.get(field_name)
    if field is None:
        return None
    if hasattr(field, "value"):
        val = field.value
    elif isinstance(field, dict):
        val = field.get("value")
    elif isinstance(field, str):
        val = field
    else:
        return None
    return val if isinstance(val, str) and val.strip() else None


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _document_type_of(ocr_result: object | None) -> str | None:
    """Type de document OCR (ex. 'INVOICE') — jamais un chemin ni un contenu."""
    if ocr_result is None:
        return None
    doc_type = getattr(ocr_result, "document_type", None)
    if doc_type is None:
        return None
    return str(getattr(doc_type, "value", doc_type))


def _collect_signals(
    ocr_result: object | None,
    coding_result: object | None,
) -> tuple[
    list[ClinicalSignal],
    list[ClinicalInconsistency],
    int | None,
    int | None,
    bool | None,
    VerificationStatus,
    list[str],
]:
    """Phase A — calcule les signaux/incohérences de cohérence clinique et le statut.

    Retourne (signals, inconsistencies, procedure_count, medication_count,
    prescription_required, status, reasons).
    """
    if ocr_result is None and coding_result is None:
        return (
            [],
            [],
            None,
            None,
            None,
            VerificationStatus.NEEDS_REVIEW,
            [
                "Aucune donnée d'extraction OCR ni de codification médicale disponible : "
                "cohérence clinique non vérifiable."
            ],
        )

    fields = getattr(ocr_result, "extracted_fields", {}) or {} if ocr_result is not None else {}
    procedure_count = _to_int(_extract_field(fields, "procedure_count"))
    medication_count = _to_int(_extract_field(fields, "medication_count"))
    prescription_number = _extract_field(fields, "prescription_number")
    service_date = _extract_field(fields, "service_date")
    care_date_raw = _extract_field(fields, "care_date") or service_date
    prescription_date_raw = _extract_field(fields, "prescription_date")
    prescription_required = medication_count > 0 if medication_count is not None else None
    document_type = _document_type_of(ocr_result)

    signals: list[ClinicalSignal] = []
    inconsistencies: list[ClinicalInconsistency] = []

    if medication_count and medication_count > 0 and not prescription_number:
        evidence = [
            ClinicalEvidence(
                source=ClinicalEvidenceSource.OCR_EXTRACTION,
                field="medication_count",
                document_reference=document_type,
                value=str(medication_count),
            ),
            ClinicalEvidence(
                source=ClinicalEvidenceSource.OCR_EXTRACTION,
                field="prescription_number",
                document_reference=document_type,
                value="absent",
            ),
        ]
        signals.append(
            ClinicalSignal(
                signal_type="MISSING_PRESCRIPTION_REFERENCE",
                description=(
                    f"{medication_count} médicament(s) facturé(s) sans numéro "
                    "d'ordonnance identifié dans les documents."
                ),
                fields_compared=["medication_count", "prescription_number"],
                documents_compared=[document_type] if document_type else [],
                evidence=evidence,
                severity=SeverityLevel.CRITICAL,
            )
        )
        inconsistencies.append(
            ClinicalInconsistency(
                inconsistency_type="MISSING_PRESCRIPTION_REFERENCE",
                expected="prescription_number renseigné",
                observed="prescription_number absent",
                severity=SeverityLevel.CRITICAL,
                evidence=evidence,
            )
        )

    codings = getattr(coding_result, "codings", None) if coding_result is not None else None
    coded_count = len(codings) if codings is not None else None
    if procedure_count is not None and coded_count is not None and procedure_count != coded_count:
        evidence = [
            ClinicalEvidence(
                source=ClinicalEvidenceSource.OCR_EXTRACTION,
                field="procedure_count",
                document_reference=document_type,
                value=str(procedure_count),
            ),
            ClinicalEvidence(
                source=ClinicalEvidenceSource.MEDICAL_CODING,
                field="codings",
                document_reference="medical_coding_agent",
                value=str(coded_count),
            ),
        ]
        mismatch_severity = SeverityLevel.CRITICAL if coded_count == 0 else SeverityLevel.MEDIUM
        signals.append(
            ClinicalSignal(
                signal_type="PROCEDURE_CODING_COUNT_MISMATCH",
                description=(
                    f"{procedure_count} acte(s) facturé(s) contre {coded_count} code(s) "
                    "SNOMED-CT/RxNorm résolus par la codification médicale."
                ),
                fields_compared=["procedure_count", "coding_result.codings"],
                documents_compared=[document_type] if document_type else [],
                evidence=evidence,
                severity=mismatch_severity,
            )
        )
        inconsistencies.append(
            ClinicalInconsistency(
                inconsistency_type="PROCEDURE_CODING_COUNT_MISMATCH",
                expected=str(procedure_count),
                observed=str(coded_count),
                severity=mismatch_severity,
                evidence=evidence,
            )
        )

    coding_status = getattr(coding_result, "status", None) if coding_result is not None else None
    if coding_status in (VerificationStatus.NEEDS_REVIEW, VerificationStatus.FAIL):
        signals.append(
            ClinicalSignal(
                signal_type="UPSTREAM_CODING_UNRESOLVED",
                description=f"Codification médicale non résolue : statut {coding_status.value}.",
                fields_compared=["coding_result.status"],
                evidence=[
                    ClinicalEvidence(
                        source=ClinicalEvidenceSource.MEDICAL_CODING,
                        field="status",
                        document_reference="medical_coding_agent",
                        value=coding_status.value,
                    )
                ],
                severity=SeverityLevel.MEDIUM,
            )
        )

    if (procedure_count or medication_count) and not service_date:
        signals.append(
            ClinicalSignal(
                signal_type="MISSING_SERVICE_DATE",
                description=(
                    "Date de service absente des documents malgré des actes ou "
                    "médicaments facturés."
                ),
                fields_compared=["service_date"],
                documents_compared=[document_type] if document_type else [],
                evidence=[
                    ClinicalEvidence(
                        source=ClinicalEvidenceSource.OCR_EXTRACTION,
                        field="service_date",
                        document_reference=document_type,
                        value="absent",
                    )
                ],
                severity=SeverityLevel.MEDIUM,
            )
        )

    # Chronologie ordonnance/soin et acte absent — même outil déterministe
    # (``tools.date_checks``) que celui exposé au LLM en Phase B
    # (``verifier_chronologie``) : jamais deux implémentations divergentes.
    signals.extend(
        run_date_checks(
            prescription_date_raw=prescription_date_raw,
            care_date_raw=care_date_raw,
            coded_count=coded_count,
            document_reference=document_type,
        )
    )

    status, reasons = _status_from_signals(signals)

    return signals, inconsistencies, procedure_count, medication_count, prescription_required, status, reasons


def _status_from_signals(signals: list[ClinicalSignal]) -> tuple[VerificationStatus, list[str]]:
    """Dérive le statut global à partir des signaux (sévérités effectives —
    Phase A seule, ou déjà ajustées par ``_apply_signal_assessments``).
    Factorisée pour être réutilisée par ``_collect_signals`` (Phase A) et par
    le recalcul post-ajustement LLM dans ``run()`` (P1-2) — un seul et même
    calcul de statut, jamais deux implémentations divergentes."""
    if not signals:
        return VerificationStatus.PASS, [
            "Aucune incohérence clinique détectée entre les documents et la codification."
        ]
    if any(s.severity == SeverityLevel.CRITICAL for s in signals):
        return VerificationStatus.FAIL, ["Incohérence clinique critique détectée — voir signaux."]
    return VerificationStatus.NEEDS_REVIEW, [
        "Incohérence(s) clinique(s) mineure(s) détectée(s) — revue recommandée."
    ]


def _bounded_severity(current: SeverityLevel, requested: SeverityLevel) -> SeverityLevel | None:
    """Retourne ``requested`` si son écart avec ``current`` est au plus d'un
    cran sur le lattice ``SeverityLevel`` (CRITICAL > HIGH > MEDIUM > LOW >
    INFO), sinon ``None`` — un ajustement hors borne est toujours ignoré,
    jamais partiellement appliqué."""
    current_index = _SEVERITY_ORDER.index(current)
    requested_index = _SEVERITY_ORDER.index(requested)
    if abs(current_index - requested_index) <= 1:
        return requested
    return None


def _apply_signal_assessments(
    signals: list[ClinicalSignal],
    assessments: list[ClinicalSignalAssessment],
) -> tuple[list[ClinicalSignal], list[str], bool]:
    """P1-2 — applique les ajustements de sévérité LLM bornés.

    Pour chaque ``ClinicalSignal`` déjà calculé par la Phase A, si son
    ``signal_type`` apparaît dans ``assessments`` (sinon ignoré
    silencieusement — même garantie anti-hallucination que
    ``referenced_evidence_ids``), sa sévérité n'est remplacée que si l'écart
    proposé reste borné à un cran (``_bounded_severity``). Chaque signal
    ajusté reste ancré à ses ``evidence`` d'origine (jamais mutées, jamais
    recréées) — seule la sévérité change, jamais le fait sous-jacent.

    Retourne la liste des signaux (ajustés ou inchangés, même ordre), les
    motifs (ajustements appliqués ou explicitement rejetés hors borne) et un
    booléen indiquant si au moins une sévérité a réellement changé (pour ne
    déclencher un recalcul de statut que si nécessaire).
    """
    if not assessments:
        return signals, [], False

    assessment_by_type = {a.signal_type: a for a in assessments}
    adjusted: list[ClinicalSignal] = []
    notes: list[str] = []
    changed = False
    for signal in signals:
        assessment = assessment_by_type.get(signal.signal_type)
        if assessment is None:
            adjusted.append(signal)
            continue
        bounded = _bounded_severity(signal.severity, assessment.severity_override)
        if bounded is None:
            notes.append(
                f"Ajustement de sévérité hors borne autorisée pour le signal "
                f"{signal.signal_type!r} (proposé {assessment.severity_override.value}, "
                f"actuel {signal.severity.value}, écart maximal autorisé : un cran) — ignoré."
            )
            adjusted.append(signal)
            continue
        if bounded == signal.severity:
            adjusted.append(signal)
            continue
        adjusted.append(signal.model_copy(update={"severity": bounded}))
        changed = True
        notes.append(
            f"Sévérité du signal {signal.signal_type!r} ajustée par le LLM "
            f"({signal.severity.value} → {bounded.value}) : {assessment.rationale}"
        )
    return adjusted, notes, changed


# ── Phase B : LLM ─────────────────────────────────────────────────────────────


def _invoke_llm_clinical(data: dict) -> LlmClinicalDecision | None:
    """Lance l'agent ReAct LLM (appel obligatoire à chaque exécution) pour un
    contexte explicatif — jamais de statut, de diagnostic ni de décision
    médicale. Seul outil physiquement joignable : ``verifier_chronologie``
    (``ALLOWED_TOOLS_PER_AGENT[AgentName.CLINICAL_CONSISTENCY]``) — les
    données transmises (``data``) ne contiennent jamais de contenu brut de
    document, de texte OCR complet ni de bundle FHIR brut, uniquement des
    compteurs, statuts, signaux déjà attribués et la vue médicale minimisée
    éventuellement disponible."""
    try:
        prompt = load_clinical_consistency_prompt()
        llm = get_llm()
        agent = create_react_agent(
            model=llm,
            tools=[verifier_chronologie],
            response_format=LlmClinicalDecision,
        )
        result = agent.invoke(
            {
                "messages": [
                    SystemMessage(content=prompt.system_prompt),
                    HumanMessage(content=json.dumps(data, ensure_ascii=False, default=str)),
                ]
            }
        )
        structured = result.get("structured_response")
        if isinstance(structured, LlmClinicalDecision):
            return structured
        if isinstance(structured, dict):
            return LlmClinicalDecision(**structured)
        return None
    except Exception:
        return None


# ── Fonction principale (testable sans LangGraph) ─────────────────────────────


def _fhir_summary(fhir_result: object | None) -> dict | None:
    """Résumé FHIR minimisé — jamais le bundle brut ni les ressources complètes."""
    if fhir_result is None:
        return None
    status = getattr(fhir_result, "status", None)
    return {
        "status": getattr(status, "value", status),
        "resource_count": getattr(fhir_result, "resource_count", None),
        "resource_types": list(getattr(fhir_result, "resource_types", []) or []),
    }


def _extract_medical_view(privacy_result: object | None) -> MedicalView | None:
    """Récupère la vue médicale déjà minimisée et pseudonymisée par
    ``privacy_agent`` (``privacy_result.view``, un dict JSON-sérialisable —
    ``PrivacyResult.view: dict | None``), jamais reconstruite ici à partir de
    données brutes. ``None`` si aucune vue MEDICAL_REVIEWER n'a été produite
    (rôle différent, vue non construite, ou privacy_result absent)."""
    if privacy_result is None:
        return None
    if getattr(privacy_result, "view_role", None) != ReaderRole.MEDICAL_REVIEWER.value:
        return None
    view_dict = getattr(privacy_result, "view", None)
    if not isinstance(view_dict, dict):
        return None
    try:
        return MedicalView.model_validate(view_dict)
    except ValidationError:
        return None


def _medical_view_summary(medical_view: object | None) -> dict | None:
    """Vue médicale minimisée déjà pseudonymisée par ``privacy_agent`` —
    jamais de nom de patient, jamais un accès direct aux documents bruts."""
    if not isinstance(medical_view, MedicalView):
        return None
    return {
        "patient_pseudonym": medical_view.patient_pseudonym,
        "service_date": medical_view.service_date,
        "procedures": medical_view.procedures,
        "prescription_names": medical_view.prescription_names,
        "diagnosis_codes": medical_view.diagnosis_codes,
        "encounter_class": medical_view.encounter_class,
    }


def _merge_llm_decision(
    llm_decision: LlmClinicalDecision | None,
    reasons: list[str],
    *,
    known_evidence_ids: set[str],
    known_inconsistency_types: set[str],
) -> list[str]:
    """Fusionne la décision LLM dans les motifs narratifs uniquement — ne
    touche jamais elle-même au statut, à la confiance ou au besoin de revue.
    L'unique canal d'influence du LLM sur le statut
    (``severity_assessments``) est appliqué séparément, avant cet appel, par
    ``_apply_signal_assessments``/``_status_from_signals`` — cette
    fonction-ci ne fait que fusionner les motifs textuels autour du résultat
    déjà figé.

    Toute preuve (``referenced_evidence_ids``) ou incohérence
    (``acknowledged_inconsistencies``) citée par le LLM mais absente des
    identifiants réellement calculés est silencieusement ignorée — jamais
    une affirmation non prouvée acceptée telle quelle. ``llm_confidence`` et
    ``suggests_human_review`` restent purement informatifs : ils n'écrasent
    jamais ``confidence`` ni ``human_review_required``, qui ne dépendent que
    du statut déterministe.
    """
    reasons = list(reasons)
    if llm_decision is None:
        reasons.append(
            "LLM indisponible : statut déterministe conservé sans contexte clinique enrichi."
        )
        return reasons

    if llm_decision.clinical_context:
        reasons.append(llm_decision.clinical_context)
    reasons.extend(llm_decision.reasons)

    unknown_evidence = [
        e for e in llm_decision.referenced_evidence_ids if e not in known_evidence_ids
    ]
    unknown_inconsistencies = [
        i for i in llm_decision.acknowledged_inconsistencies if i not in known_inconsistency_types
    ]
    if unknown_evidence or unknown_inconsistencies:
        reasons.append(
            "LLM a référencé des preuves ou incohérences inexistantes — références "
            "ignorées (aucune affirmation non prouvée acceptée)."
        )

    if llm_decision.suggests_human_review:
        reasons.append(
            "Le LLM signale un besoin de revue complémentaire (information, non "
            "contraignante — statut et besoin de revue déterministes conservés)."
        )

    if llm_decision.llm_confidence is not None:
        reasons.append(
            f"Confiance perçue par le LLM : {llm_decision.llm_confidence:.2f} "
            "(n'affecte pas la confiance déterministe)."
        )

    return reasons


def run(
    case_id: str,
    ocr_result: object | None = None,
    coding_result: object | None = None,
    fhir_result: object | None = None,
    medical_view: object | None = None,
) -> ClinicalConsistencyResult:
    """Évalue la cohérence clinique d'un dossier.

    Args:
        case_id: identifiant du dossier.
        ocr_result: ``DocumentOcrResult | None`` — champs extraits.
        coding_result: ``MedicalCodingResult | None`` — codes résolus.
        fhir_result: ``FhirValidatorResult | None`` — résumé FHIR minimisé
            transmis au LLM (jamais le bundle brut).
        medical_view: ``MedicalView | None`` — vue médicale déjà minimisée et
            pseudonymisée par ``privacy_agent`` (``privacy_result.view``),
            transmise telle quelle si disponible, jamais reconstruite ici à
            partir de données brutes.

    Returns:
        ``ClinicalConsistencyResult`` avec statut PASS / NEEDS_REVIEW / FAIL.
    """
    signals, inconsistencies, procedure_count, medication_count, prescription_required, status, reasons = (
        _collect_signals(ocr_result, coding_result)
    )
    confidence = max(0.4, 1.0 - 0.15 * len(signals))

    evidence_ids = [
        evidence.evidence_id for signal in signals for evidence in signal.evidence
    ] + [
        evidence.evidence_id
        for inconsistency in inconsistencies
        for evidence in inconsistency.evidence
    ]
    inconsistency_types = [i.inconsistency_type for i in inconsistencies]

    chronology_signal_types = {
        "IMPOSSIBLE_DATE",
        "PRESCRIPTION_BEFORE_CARE",
        "PRESCRIPTION_TOO_FAR_AFTER_CARE",
        "MISSING_PROCEDURE_EVIDENCE",
    }
    coding_status = getattr(coding_result, "status", None) if coding_result is not None else None

    llm_decision = _invoke_llm_clinical(
        {
            "case_id": case_id,
            "status": status.value,
            "chronologie": [
                {"signal_type": s.signal_type, "severity": s.severity.value}
                for s in signals
                if s.signal_type in chronology_signal_types
            ],
            "ordonnance": {
                "medication_count": medication_count,
                "prescription_required": prescription_required,
            },
            "acte": {"procedure_count": procedure_count},
            "code": {
                "coded_count": len(getattr(coding_result, "codings", []) or [])
                if coding_result is not None
                else None,
                "status": getattr(coding_status, "value", coding_status),
            },
            "fhir_minimise": _fhir_summary(fhir_result),
            "vue_medicale_minimisee": _medical_view_summary(medical_view),
            "signals": [
                {"signal_type": s.signal_type, "severity": s.severity.value} for s in signals
            ],
            "evidence_ids": evidence_ids,
            "inconsistency_types": inconsistency_types,
            "instruction": (
                "Analyse la chronologie, l'ordonnance, l'acte, la codification et le "
                "résumé FHIR minimisé fournis. Tu ne fixes jamais toi-même le statut : "
                "ton seul levier est severity_assessments, un ajustement de sévérité "
                "borné à un cran maximum (échelle CRITICAL > HIGH > MEDIUM > LOW > INFO) "
                "sur un signal déjà calculé, avec justification obligatoire. Jamais de "
                "diagnostic, de traitement recommandé, de décision finale, de document "
                "inventé, ni d'affirmation qui ne soit pas déjà appuyée par les signaux "
                "fournis. Ne cite que des evidence_ids, des inconsistency_types et des "
                "signal_type déjà présents ci-dessus."
            ),
        }
    )

    # P1-2 : ajustement borné de sévérité — recalcul déterministe du statut
    # si le LLM a proposé un ajustement effectif (dans la borne) sur un
    # signal réel.
    signals, adjustment_notes, severity_changed = _apply_signal_assessments(
        signals, llm_decision.severity_assessments if llm_decision is not None else []
    )
    if severity_changed:
        previous_status = status
        status, status_reasons = _status_from_signals(signals)
        reasons.extend(status_reasons)
        # P1-2/P3-2 : ajustement de sévérité LLM borné effectivement
        # appliqué — point de décision autonome, journalisé pour
        # traçabilité opérationnelle.
        logger.info(
            "clinical_consistency_severity_adjusted",
            case_id=case_id,
            adjustment_count=len(adjustment_notes),
            status_before=previous_status.value,
            status_after=status.value,
        )
    reasons.extend(adjustment_notes)

    reasons = _merge_llm_decision(
        llm_decision,
        reasons,
        known_evidence_ids=set(evidence_ids),
        known_inconsistency_types=set(inconsistency_types),
    )
    errors = (
        [
            StructuredError(
                code="LLM_UNAVAILABLE",
                message="LLM indisponible ou réponse invalide : statut déterministe conservé.",
                field="llm_trace",
            )
        ]
        if llm_decision is None
        else []
    )

    return ClinicalConsistencyResult(
        case_id=case_id,
        status=status,
        llm_trace=build_llm_metadata(_AGENT_NAME, confidence=confidence),
        confidence=confidence,
        errors=errors,
        evidence_ids=evidence_ids,
        human_review_required=status is not VerificationStatus.PASS,
        result_payload=ClinicalResultPayload(
            procedure_count=procedure_count,
            medication_count=medication_count,
            prescription_required=prescription_required,
            signals=signals,
            inconsistencies=inconsistencies,
            reasons=reasons,
        ),
    )


# ── Implémentation par défaut (réelle) ────────────────────────────────────────


class _RealImplementation:
    """Adapte ``run()`` à l'interface ``ClinicalConsistencyRunnable``."""

    def run(self, state: ClaimState) -> ClinicalConsistencyResult:
        case_id = str(state.get("case_id", "UNKNOWN"))
        return run(
            case_id=case_id,
            ocr_result=state.get("ocr_result"),
            coding_result=state.get("coding_result"),
            fhir_result=state.get("fhir_result"),
            medical_view=_extract_medical_view(state.get("privacy_result")),
        )


_DEFAULT_IMPL: ClinicalConsistencyRunnable = _RealImplementation()


# ── Factory et nœud LangGraph ─────────────────────────────────────────────────


def make_node(
    impl: ClinicalConsistencyRunnable = _DEFAULT_IMPL,
) -> Callable[[ClaimState], dict]:
    """Crée un nœud LangGraph avec l'implémentation injectable fournie.

    Args:
        impl: Toute classe satisfaisant ``ClinicalConsistencyRunnable``.
              Par défaut : ``_RealImplementation`` (évaluation réelle).

    Returns:
        Fonction ``(state) -> dict`` compatible LangGraph.
    """
    def _node(state: ClaimState) -> dict:
        result = impl.run(state)
        case_id = str(state.get("case_id", result.case_id))
        llm_call_id = str(uuid.uuid4())
        audit = AuditEvent(
            event_id=str(uuid.uuid4()),
            case_id=case_id,
            actor=_AGENT_NAME,
            action="clinical_consistency_check",
            outcome=result.status.value,
            details={
                "procedure_count": str(result.result_payload.procedure_count),
                "medication_count": str(result.result_payload.medication_count),
                "signal_count": str(len(result.result_payload.signals)),
                "inconsistency_count": str(len(result.result_payload.inconsistencies)),
                "llm_call_id": llm_call_id,
                "model_name": result.llm_trace.model_name,
                "prompt_version": result.llm_trace.prompt_version,
                "tools": verifier_chronologie.name,
                "errors": ",".join(e.code for e in result.errors),
            },
        )
        updates: dict = {
            "clinical_result": result,
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
                f"Cohérence clinique : {result.status.value} — "
                f"{'; '.join(result.result_payload.reasons)}"
            ]
        validate_state_update(updates)
        return updates

    _node.__name__ = f"node_{_STEP_NAME}"
    return _node


# Nœud stable — nom utilisé comme clé dans le StateGraph.
node = make_node()
