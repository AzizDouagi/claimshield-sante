"""Eligibility Agent (V2) — porte quasi 1:1 `identity_coverage_agent` (V1).

Réutilise directement `agents.identity_coverage_agent.agent.run()` (import,
jamais dupliqué ni modifié — §0 du plan) : la logique métier (matching
identité, vérification couverture/plafond/préautorisation) est déjà
entièrement déterministe côté V1 — le LLM n'a jamais eu autorité sur
`identity.status`/`coverage.status` (déjà vrai en V1, inchangé ici), il
n'ajoute qu'une synthèse consultative (`warnings`), cohérent avec le plan
V2 §5 : « 1 appel optionnel (explication, jamais la décision) ».

Ce module ne fait que traduire `schemas.results.IdentityCoverageResult`
(V1) vers `schemas.v2_results.EligibilityResult` (V2) — même contenu,
enveloppe renommée, `status` global dérivé du pire des deux statuts
(identité, couverture), jamais recalculé indépendamment.
"""
from __future__ import annotations

from agents.identity_coverage_agent import agent as v1_agent
from schemas.domain import VerificationStatus
from schemas.results import StructuredError
from schemas.v2_results import EligibilityResult
from state.claim_state_v2 import ClaimStateV2, validate_state_update_v2
from tools.text_normalizer import normalize_date_value

_STATUS_RANK: dict[VerificationStatus, int] = {
    VerificationStatus.PASS: 0,
    VerificationStatus.NEEDS_REVIEW: 1,
    VerificationStatus.FAIL: 2,
}


def _worse(a: VerificationStatus, b: VerificationStatus) -> VerificationStatus:
    return a if _STATUS_RANK[a] >= _STATUS_RANK[b] else b


def _normalize_extracted_fields(extracted_fields: dict | None) -> dict | None:
    """Normalise `service_date` (seul champ de type `date` consommé par
    `IdentityCoverageInput`, V1) avant délégation à `v1_agent.run()`.

    Bug réel découvert en Phase V2-10 (E2E sur documents réels) : contrairement
    au pipeline V1 — où `identity_coverage_agent` n'est jamais atteint avec des
    documents réels faute de câblage complet (court-circuit `NEEDS_REVIEW` sur
    `document_ocr`/`fhir_validator` avant, voir CLAUDE.md « câblage minimal »)
    — le graphe V2 est strictement séquentiel et atteint réellement
    `eligibility_agent` avec la valeur brute extraite par l'OCR (ex.
    `"19/05/1976"`, format `JJ/MM/AAAA`), jamais normalisée par
    `tools.document_parser`/`tools.text_normalizer` en amont pour ce champ
    précis. `v1_agent.run()` (non modifié — §0 du plan) transmet cette valeur
    brute telle quelle à `IdentityCoverageInput.service_date: date`, qui la
    rejette (`ValidationError`) — jamais observé en V1 car le chemin n'a
    jamais été exercé bout en bout. Corrigé ici, côté V2 uniquement : normalise
    via `tools.text_normalizer.normalize_date_value` (source unique de vérité
    du projet pour le parsing de dates OCR) vers ISO ; une date ambiguë ou
    invalide est retirée du dict plutôt que transmise — `service_date` devient
    alors `None` côté V1 (déjà un cas géré, jamais une exception), pas une
    valeur inventée.
    """
    if not extracted_fields or "service_date" not in extracted_fields:
        return extracted_fields
    field = extracted_fields.get("service_date")
    # Même duck-typing que `identity_coverage_agent.agent._extract_field_value`
    # (V1, non modifié) : `field` peut être un `ExtractedField` Pydantic
    # (`.value`), un dict brut (`["value"]`) ou une chaîne directe.
    if hasattr(field, "value"):
        raw = field.value
    elif isinstance(field, dict):
        raw = field.get("value")
    elif isinstance(field, str):
        raw = field
    else:
        raw = None
    if not isinstance(raw, str) or not raw.strip():
        return extracted_fields
    normalized = normalize_date_value(raw)
    result = dict(extracted_fields)
    if normalized.normalized_value is not None:
        result["service_date"] = normalized.normalized_value.isoformat()
    else:
        result.pop("service_date", None)
    return result


def _sanitize_reason(text: str) -> str:
    """Coupe tout contenu multi-lignes accidentel avant qu'il n'atteigne le
    validateur anti-fuite d'`EligibilityResult.reasons` — les motifs V1 sont
    déjà courts, ce filtre est une simple défense en profondeur."""
    return text.replace("\n", " ").strip()


# ── Fonction principale (testable sans LangGraph) ─────────────────────────────


def run(
    case_id: str,
    ocr_result=None,  # DocumentOcrResult | None (V1) — non utilisé côté V2, voir extracted_fields
    fhir_bundle_path: str | None = None,
    *,
    extracted_fields: dict | None = None,
    dossier_patient_id: str | None = None,
    contract: dict | None = None,
    policy_number: str | None = None,
    patient_pseudonym: str | None = None,
    service_date: object | None = None,
    requested_amount: object | None = None,
    total_amount: object | None = None,
    procedure_codes: list[str] | None = None,
    preauthorization_reference: str | None = None,
    extraction_confidence: float | None = None,
    provenance: dict[str, str] | None = None,
) -> EligibilityResult:
    """Exécute la vérification identité + couverture pour un dossier.

    Mêmes paramètres que `agents.identity_coverage_agent.agent.run()` (V1) —
    délégation directe, aucune divergence de comportement (hors normalisation
    défensive de `service_date`, voir `_normalize_extracted_fields`). Voir la
    docstring de ce module pour la traduction de schéma appliquée en sortie.
    """
    extracted_fields = _normalize_extracted_fields(extracted_fields)
    if isinstance(service_date, str):
        normalized_service_date = normalize_date_value(service_date)
        service_date = normalized_service_date.normalized_value

    v1_result = v1_agent.run(
        case_id,
        ocr_result,
        fhir_bundle_path,
        extracted_fields=extracted_fields,
        dossier_patient_id=dossier_patient_id,
        contract=contract,
        policy_number=policy_number,
        patient_pseudonym=patient_pseudonym,
        service_date=service_date,
        requested_amount=requested_amount,
        total_amount=total_amount,
        procedure_codes=procedure_codes,
        preauthorization_reference=preauthorization_reference,
        extraction_confidence=extraction_confidence,
        provenance=provenance,
    )

    status = _worse(v1_result.identity.status, v1_result.coverage.status)
    reasons = [
        _sanitize_reason(r)
        for r in (*v1_result.identity.reasons, *v1_result.coverage.reasons, *v1_result.warnings)
        if r
    ]
    errors = [
        StructuredError(code=e["code"], message=e["message"], field=e.get("field") or None)
        for e in v1_result.structured_errors
    ]

    return EligibilityResult(
        case_id=v1_result.case_id,
        status=status,
        identity=v1_result.identity,
        coverage=v1_result.coverage,
        rule_version=v1_result.rule_version,
        reasons=reasons or ["Vérification identité/couverture terminée."],
        errors=errors,
        llm_trace=v1_result.llm_metadata,
    )


# ── Nœud LangGraph V2 ──────────────────────────────────────────────────────────


def _find_fhir_bundle_path(state: ClaimStateV2) -> str | None:
    """Localise le chemin (sans préfixe `incoming/`) du bundle FHIR déjà
    identifié par `intake_safety_agent`, réutilisant les mêmes helpers que
    `document_understanding_agent` (module V2 propre, pas V1)."""
    from agents.document_understanding_agent.agent import (
        _select_fhir_bundle_candidate,
        _strip_incoming_prefix,
    )

    intake_safety_result = state.get("intake_safety_result")
    manifest = None
    if intake_safety_result is not None and not isinstance(intake_safety_result, dict):
        manifest = intake_safety_result.manifest
    elif isinstance(intake_safety_result, dict):
        manifest = intake_safety_result.get("manifest")

    if manifest is None:
        return None
    files = manifest.files if hasattr(manifest, "files") else manifest.get("files", [])
    candidate = _select_fhir_bundle_candidate(files)
    if candidate is None:
        return None
    return _strip_incoming_prefix(candidate.relative_storage_path or "")


def node(state: ClaimStateV2) -> dict:
    """Nœud du graphe V2 — délègue à `run()` et met à jour `ClaimStateV2`.

    Attend dans le state :
        case_id                          : identifiant du dossier
        document_understanding_result    : DocumentUnderstandingResult (champs extraits)
        intake_safety_result             : IntakeSafetyResult (localisation du bundle FHIR)
    """
    case_id: str = state.get("case_id", "")  # type: ignore[assignment]
    document_understanding_result = state.get("document_understanding_result")

    extracted_fields: dict | None = None
    if document_understanding_result is not None:
        extraction = (
            document_understanding_result.extraction
            if not isinstance(document_understanding_result, dict)
            else document_understanding_result.get("extraction")
        )
        if extraction is not None:
            extracted_fields = (
                extraction.fields if hasattr(extraction, "fields") else extraction.get("fields", {})
            )

    fhir_bundle_path = _find_fhir_bundle_path(state)

    result = run(case_id=case_id, extracted_fields=extracted_fields, fhir_bundle_path=fhir_bundle_path)

    updates: dict = {
        "eligibility_result": result,
        "current_step": "eligibility",
        "completed_steps": ["eligibility"],
    }
    if result.status is VerificationStatus.FAIL:
        updates["errors"] = [f"[eligibility] {r}" for r in result.reasons]
    elif result.status is VerificationStatus.NEEDS_REVIEW:
        updates["alerts"] = [f"[eligibility] {r}" for r in result.reasons[:5]]

    validate_state_update_v2(updates)
    return updates
