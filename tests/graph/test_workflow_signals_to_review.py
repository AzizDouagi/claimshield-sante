"""Signaux clinical_consistency/fraud_detection → Case Reviewer — ClaimShield Santé.

Complète ``test_workflow_paths.py`` (chemin nominal) et
``test_workflow_blocking_paths.py`` (branches de blocage amont) par quatre
angles spécifiques au pipeline étape 12, sur un graphe compilé réel :

  A. un signal critique de ``clinical_consistency`` ou ``fraud_detection``
     atteint bien ``case_reviewer`` (jamais court-circuité) et route vers
     ``needs_review``, jamais vers une fin silencieuse ;
  B. aucun agent ne décide seul APPROVE/REJECT final — ``final_recommendation``
     n'existe qu'après le passage de ``case_reviewer`` et correspond
     toujours exactement à ``review_result.recommendation`` ;
  C. fail-closed si le LLM de ``clinical_consistency``/``fraud_detection``
     est absent (injoignable) ou renvoie une sortie non conforme — jamais un
     succès fabriqué, jamais une exception qui remonte ;
  D. le state final ne contient jamais de texte OCR complet, de secret ni de
     prompt système complet — uniquement des données déjà minimisées.

``claim_intake``/``security_gate``/``privacy`` restent des faux agents
nominaux (aucun appel LLM). ``document_ocr``/``fhir_validator``/
``identity_coverage``/``medical_coding`` sont de faux agents dont la sortie
est *réellement typée* (vraies instances Pydantic, pas de simple
``SimpleNamespace``) afin que ``clinical_consistency_agent``/
``fraud_detection_agent`` — qui tournent en implémentation réelle par
défaut, LLM mocké par ``tests/conftest.py::deterministic_agent_llm`` —
lisent des champs réellement exploitables (``extracted_fields``,
``confidence_score``, ``ceiling_exceeded``...). ``case_reviewer`` tourne
aussi en implémentation réelle par défaut (jamais injecté ici).
"""
from __future__ import annotations

from typing import Any

import pytest

import graph.workflow as wf
from agents.clinical_consistency_agent.agent import _invoke_llm_clinical as _REAL_INVOKE_LLM_CLINICAL
from agents.fraud_detection_agent.agent import _invoke_llm_fraud as _REAL_INVOKE_LLM_FRAUD
from graph.workflow import compile_workflow
from schemas.domain import (
    DocumentType,
    ExtractionStatus,
    IntakeStatus,
    OcrSource,
    PrivacyDecision,
    Recommendation,
    SecurityDecision,
    VerificationStatus,
)
from schemas.results import (
    CoverageResult,
    DocumentOcrResult,
    ExtractedField,
    IdentityCoverageResult,
    IdentityResult,
    MedicalCodingResult,
)

# ── Faux agents amont nominaux (aucun appel LLM) ─────────────────────────────


def _nominal_claim_intake(state: dict) -> dict:
    return {
        "intake_status": IntakeStatus.ACCEPTED,
        "intake_input": None,
        "current_step": "claim_intake",
        "completed_steps": ["claim_intake"],
    }


def _nominal_security_gate(state: dict) -> dict:
    from dataclasses import dataclass

    @dataclass
    class _Sec:
        decision: Any = SecurityDecision.ALLOW

    return {
        "security_result": _Sec(),
        "security_input": None,
        "current_step": "security_gate",
        "completed_steps": ["security_gate"],
    }


def _nominal_privacy(state: dict) -> dict:
    from dataclasses import dataclass

    @dataclass
    class _Priv:
        decision: Any = PrivacyDecision.ALLOW

    return {
        "privacy_result": _Priv(),
        "privacy_input": None,
        "current_step": "privacy",
        "completed_steps": ["privacy"],
    }


def _nominal_fhir_validator(state: dict) -> dict:
    from dataclasses import dataclass

    @dataclass
    class _Fhir:
        status: Any = VerificationStatus.PASS

    return {
        "fhir_result": _Fhir(),
        "fhir_input": None,
        "current_step": "fhir_validator",
        "completed_steps": ["fhir_validator"],
    }


# ── Fabriques de sorties réellement typées (contrôlent les signaux) ─────────


def _ocr_result(*, procedure_count: str | None = None, medication_count: str | None = None,
                prescription_number: str | None = None, service_date: str | None = "2024-01-15",
                confidence_score: float = 0.9) -> DocumentOcrResult:
    fields = {
        "procedure_count": procedure_count,
        "medication_count": medication_count,
        "prescription_number": prescription_number,
        "service_date": service_date,
    }
    extracted = {
        name: ExtractedField(field_name=name, value=value, confidence=0.9)
        for name, value in fields.items()
        if value is not None
    }
    return DocumentOcrResult(
        claim_id="CLM-0001",
        file_path="facture.pdf",
        sha256="a" * 64,
        mime_type="application/pdf",
        extraction_status=ExtractionStatus.SUCCESS,
        status=VerificationStatus.PASS,
        document_type=DocumentType.INVOICE,
        ocr_source=OcrSource.PDF_TEXT,
        extracted_fields=extracted,
        confidence_score=confidence_score,
    )


def _medical_coding_result(count: int) -> MedicalCodingResult:
    from schemas.results import LlmMetadata, ProcedureCoding

    codings = [
        ProcedureCoding(original_description=f"acte {i}", proposed_code=f"C{i}", status=VerificationStatus.PASS)
        for i in range(count)
    ]
    return MedicalCodingResult(
        case_id="CLM-0001",
        status=VerificationStatus.PASS,
        codings=codings,
        llm_metadata=LlmMetadata(model_name="test-llm", prompt_version="test"),
    )


def _identity_coverage_result(
    *, ceiling_exceeded: bool = False, preauthorization_required: bool = False,
    preauthorization_status: str | None = None,
) -> IdentityCoverageResult:
    return IdentityCoverageResult(
        case_id="CLM-0001",
        identity=IdentityResult(status=VerificationStatus.PASS),
        coverage=CoverageResult(
            status=VerificationStatus.PASS,
            ceiling_exceeded=ceiling_exceeded,
            preauthorization_required=preauthorization_required,
            preauthorization_status=preauthorization_status,
        ),
    )


def _build_app(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ocr_result: DocumentOcrResult,
    coding_result: MedicalCodingResult,
    identity_coverage_result: IdentityCoverageResult,
):
    """Compile un workflow où seuls document_ocr/identity_coverage/medical_coding
    portent des données contrôlées ; clinical_consistency, fraud_detection et
    case_reviewer tournent tous en implémentation réelle (LLM mocké par
    conftest)."""

    def _document_ocr(state: dict) -> dict:
        return {
            "ocr_result": ocr_result,
            "ocr_input": None,
            "current_step": "document_ocr",
            "completed_steps": ["document_ocr"],
        }

    def _identity_coverage(state: dict) -> dict:
        return {
            "identity_coverage_result": identity_coverage_result,
            "identity_coverage_input": None,
            "current_step": "identity_coverage",
            "completed_steps": ["identity_coverage"],
        }

    def _medical_coding(state: dict) -> dict:
        return {
            "coding_result": coding_result,
            "coding_input": None,
            "current_step": "medical_coding",
            "completed_steps": ["medical_coding"],
        }

    monkeypatch.setattr(wf, "node_claim_intake", _nominal_claim_intake)
    monkeypatch.setattr(wf, "node_security_gate", _nominal_security_gate)
    monkeypatch.setattr(wf, "node_privacy", _nominal_privacy)
    monkeypatch.setattr(wf, "node_document_ocr", _document_ocr)
    monkeypatch.setattr(wf, "node_fhir_validator", _nominal_fhir_validator)
    monkeypatch.setattr(wf, "node_identity_coverage", _identity_coverage)
    monkeypatch.setattr(wf, "node_medical_coding", _medical_coding)

    return compile_workflow(interrupt_before=[])


def _initial_state() -> dict:
    return {
        "case_id": "CLM-0001",
        "schema_version": "1.0.0",
        "current_step": "initial",
        "completed_steps": [],
        "errors": [],
        "alerts": [],
        "final_justification": [],
    }


# Configurations réutilisées entre les scénarios.


def _clean_upstream_kwargs() -> dict:
    return {
        "ocr_result": _ocr_result(procedure_count="1"),
        "coding_result": _medical_coding_result(1),
        "identity_coverage_result": _identity_coverage_result(),
    }


def _clinical_critical_kwargs() -> dict:
    """Procédures facturées sans aucun code résolu : PROCEDURE_CODING_COUNT_MISMATCH
    CRITICAL (status FAIL) — sans jamais faire dévier le routage amont
    (medical_coding.status reste PASS, seul coding_result.codings est vide)."""
    return {
        "ocr_result": _ocr_result(procedure_count="3"),
        "coding_result": _medical_coding_result(0),
        "identity_coverage_result": _identity_coverage_result(),
    }


def _fraud_critical_kwargs() -> dict:
    """Plafond dépassé + préautorisation manquante + confiance OCR faible :
    risk_score = 0.25 + 0.30 + 0.15 = 0.70 → FAIL — sans jamais faire dévier
    le routage amont (identity/coverage .status restent PASS)."""
    return {
        "ocr_result": _ocr_result(procedure_count="1", confidence_score=0.2),
        "coding_result": _medical_coding_result(1),
        "identity_coverage_result": _identity_coverage_result(
            ceiling_exceeded=True,
            preauthorization_required=True,
            preauthorization_status="missing",
        ),
    }


# ── A. Signaux critiques → Case Reviewer, jamais court-circuités ────────────


class TestCriticalSignalsReachCaseReviewer:
    def test_clinical_critical_signal_reaches_case_reviewer(self, monkeypatch):
        app = _build_app(monkeypatch, **_clinical_critical_kwargs())
        result = app.invoke(_initial_state())

        assert result["clinical_result"].status is VerificationStatus.FAIL
        assert "clinical_consistency" in result["completed_steps"]
        assert "fraud_detection" in result["completed_steps"], (
            "fraud_detection doit s'exécuter même après un signal critique de "
            "clinical_consistency — ce n'est jamais à cet agent de router seul"
        )
        assert "case_reviewer" in result["completed_steps"]
        assert result.get("review_result") is not None
        assert "__interrupt__" in result, "jamais une fin silencieuse sans revue humaine"
        assert result.get("current_step") == "needs_review"

    def test_fraud_critical_signal_reaches_case_reviewer(self, monkeypatch):
        app = _build_app(monkeypatch, **_fraud_critical_kwargs())
        result = app.invoke(_initial_state())

        assert result["fraud_result"].status is VerificationStatus.FAIL
        assert "fraud_detection" in result["completed_steps"]
        assert "case_reviewer" in result["completed_steps"]
        assert result.get("review_result") is not None
        assert "__interrupt__" in result
        assert result.get("current_step") == "needs_review"

    def test_case_reviewer_recommendation_reflects_clinical_fail(self, monkeypatch):
        app = _build_app(monkeypatch, **_clinical_critical_kwargs())
        result = app.invoke(_initial_state())
        assert result["review_result"].recommendation is Recommendation.REJECT

    def test_case_reviewer_recommendation_reflects_fraud_fail(self, monkeypatch):
        app = _build_app(monkeypatch, **_fraud_critical_kwargs())
        result = app.invoke(_initial_state())
        assert result["review_result"].recommendation is Recommendation.REJECT

    def test_clean_case_still_reaches_case_reviewer_and_requires_review(self, monkeypatch):
        """Même sans signal critique, la pré-recommandation reste non finale :
        needs_review est toujours atteint (human_review_required forcé)."""
        app = _build_app(monkeypatch, **_clean_upstream_kwargs())
        result = app.invoke(_initial_state())

        assert result["clinical_result"].status is VerificationStatus.PASS
        assert result["fraud_result"].status is VerificationStatus.PASS
        assert "case_reviewer" in result["completed_steps"]
        assert "__interrupt__" in result


# ── B. Aucun agent ne décide seul APPROVE/REJECT final ──────────────────────


class TestNoAgentDecidesAlone:
    def test_clinical_and_fraud_schemas_never_carry_a_recommendation_field(self):
        from schemas.results import ClinicalConsistencyResult, FraudDetectionResult

        forbidden = {"recommendation", "final_recommendation", "decision"}
        assert forbidden.isdisjoint(ClinicalConsistencyResult.model_fields.keys())
        assert forbidden.isdisjoint(FraudDetectionResult.model_fields.keys())

    @pytest.mark.parametrize(
        "kwargs_factory",
        [_clean_upstream_kwargs, _clinical_critical_kwargs, _fraud_critical_kwargs],
        ids=["clean", "clinical_fail", "fraud_fail"],
    )
    def test_final_recommendation_always_matches_case_reviewer(self, monkeypatch, kwargs_factory):
        app = _build_app(monkeypatch, **kwargs_factory())
        result = app.invoke(_initial_state())

        assert result.get("final_recommendation") == result["review_result"].recommendation

    def test_final_recommendation_absent_before_case_reviewer_runs(self, monkeypatch):
        """Parcourt les états intermédiaires (stream) : final_recommendation
        n'apparaît dans aucun état strictement antérieur à celui où
        case_reviewer vient de tourner."""
        app = _build_app(monkeypatch, **_clean_upstream_kwargs())

        seen_case_reviewer = False
        checked_at_least_one_prior_state = False
        for step_state in app.stream(_initial_state(), stream_mode="values"):
            if "case_reviewer" in step_state.get("completed_steps", []):
                seen_case_reviewer = True
                continue
            checked_at_least_one_prior_state = True
            assert step_state.get("final_recommendation") is None, (
                "final_recommendation ne doit jamais apparaître avant case_reviewer"
            )
        assert seen_case_reviewer, "case_reviewer doit avoir tourné au moins une fois"
        assert checked_at_least_one_prior_state, "au moins un état antérieur à case_reviewer attendu"


# ── C. Fail-closed si LLM absent ou sortie non conforme ──────────────────────


class _RaisingReactAgent:
    """Simule un LLM injoignable : lève une exception à l'invocation."""

    def invoke(self, *_args, **_kwargs):
        raise ConnectionError("Ollama indisponible")


class _NonConformingReactAgent:
    """Simule une réponse structurée non conforme (ni dict, ni instance
    valide) — déclenche le repli interne sans passer par une exception."""

    def invoke(self, *_args, **_kwargs):
        return {"structured_response": "réponse non structurée inattendue"}


class TestFailClosedOnLlmFailure:
    """``tests/conftest.py::deterministic_agent_llm`` (autouse) remplace déjà
    ``_invoke_llm_clinical``/``_invoke_llm_fraud`` par un décideur canné pour
    que le reste de la suite reste déterministe sans Ollama. Ici, on
    restaure explicitement l'implémentation réelle (capturée à l'import de
    ce module, avant tout monkeypatch) pour exercer son vrai repli interne
    face à ``create_react_agent`` — sinon patcher ``create_react_agent``
    n'aurait aucun effet observable."""

    def test_clinical_llm_absent_is_fail_closed(self, monkeypatch):
        import agents.clinical_consistency_agent.agent as clinical_module

        monkeypatch.setattr(clinical_module, "_invoke_llm_clinical", _REAL_INVOKE_LLM_CLINICAL)
        monkeypatch.setattr(
            clinical_module, "create_react_agent", lambda **_kwargs: _RaisingReactAgent()
        )
        app = _build_app(monkeypatch, **_clean_upstream_kwargs())

        result = app.invoke(_initial_state())  # ne doit jamais lever

        assert len(result["clinical_result"].errors) == 1
        assert result["clinical_result"].errors[0].code == "LLM_UNAVAILABLE"
        assert result["clinical_result"].status is VerificationStatus.PASS  # Phase A conservée
        assert "case_reviewer" in result["completed_steps"]
        assert "__interrupt__" in result

    def test_clinical_llm_non_conforming_output_is_fail_closed(self, monkeypatch):
        import agents.clinical_consistency_agent.agent as clinical_module

        monkeypatch.setattr(clinical_module, "_invoke_llm_clinical", _REAL_INVOKE_LLM_CLINICAL)
        monkeypatch.setattr(
            clinical_module, "create_react_agent", lambda **_kwargs: _NonConformingReactAgent()
        )
        app = _build_app(monkeypatch, **_clean_upstream_kwargs())

        result = app.invoke(_initial_state())

        assert len(result["clinical_result"].errors) == 1
        assert result["clinical_result"].errors[0].code == "LLM_UNAVAILABLE"
        assert "case_reviewer" in result["completed_steps"]

    def test_fraud_llm_absent_is_fail_closed(self, monkeypatch):
        import agents.fraud_detection_agent.agent as fraud_module

        monkeypatch.setattr(fraud_module, "_invoke_llm_fraud", _REAL_INVOKE_LLM_FRAUD)
        monkeypatch.setattr(
            fraud_module, "create_react_agent", lambda **_kwargs: _RaisingReactAgent()
        )
        app = _build_app(monkeypatch, **_clean_upstream_kwargs())

        result = app.invoke(_initial_state())

        assert len(result["fraud_result"].errors) == 1
        assert result["fraud_result"].errors[0].code == "LLM_UNAVAILABLE"
        assert result["fraud_result"].status is VerificationStatus.PASS
        assert "case_reviewer" in result["completed_steps"]
        assert "__interrupt__" in result

    def test_fraud_llm_non_conforming_output_is_fail_closed(self, monkeypatch):
        import agents.fraud_detection_agent.agent as fraud_module

        monkeypatch.setattr(fraud_module, "_invoke_llm_fraud", _REAL_INVOKE_LLM_FRAUD)
        monkeypatch.setattr(
            fraud_module, "create_react_agent", lambda **_kwargs: _NonConformingReactAgent()
        )
        app = _build_app(monkeypatch, **_clean_upstream_kwargs())

        result = app.invoke(_initial_state())

        assert len(result["fraud_result"].errors) == 1
        assert result["fraud_result"].errors[0].code == "LLM_UNAVAILABLE"
        assert "case_reviewer" in result["completed_steps"]


# ── D. State minimal — pas d'OCR complet, pas de secret, pas de prompt ──────


def _flatten_state_to_text(state: dict) -> str:
    """Sérialise tout le state (résultats Pydantic compris) en un seul texte
    de recherche — jamais utilisé pour autre chose qu'un scan de motifs."""
    import json

    chunks: list[str] = []
    for value in state.values():
        items = value if isinstance(value, list) else [value]
        for item in items:
            if hasattr(item, "model_dump"):
                chunks.append(json.dumps(item.model_dump(mode="json"), ensure_ascii=False, default=str))
            else:
                chunks.append(str(item))
    return "\n".join(chunks)


class TestStateStaysMinimal:
    def test_final_state_never_contains_full_system_prompts(self, monkeypatch):
        from agents.case_reviewer_agent.prompt import load_case_reviewer_prompt
        from agents.clinical_consistency_agent.prompt import load_clinical_consistency_prompt
        from agents.fraud_detection_agent.prompt import load_fraud_detection_prompt

        app = _build_app(monkeypatch, **_clinical_critical_kwargs())
        result = app.invoke(_initial_state())
        combined = _flatten_state_to_text(result)

        for loader in (
            load_clinical_consistency_prompt,
            load_fraud_detection_prompt,
            load_case_reviewer_prompt,
        ):
            full_prompt = loader().system_prompt
            assert full_prompt not in combined

    def test_final_state_never_contains_secret_like_content(self, monkeypatch):
        app = _build_app(monkeypatch, **_fraud_critical_kwargs())
        result = app.invoke(_initial_state())
        combined = _flatten_state_to_text(result).casefold()

        for marker in ("api_key", "api-key", "password=", "bearer ", "secret:"):
            assert marker not in combined

    def test_final_state_never_contains_raw_ocr_dump_markers(self, monkeypatch):
        """``DocumentOcrResult.full_text`` existe structurellement dans le
        schéma mais doit rester vide/court ici — jamais un texte OCR complet
        copié tel quel ; les champs exploités par les agents sont uniquement
        ``extracted_fields`` (déjà minimisés)."""
        app = _build_app(monkeypatch, **_clinical_critical_kwargs())
        result = app.invoke(_initial_state())

        assert len(result["ocr_result"].full_text) <= 500

    def test_final_state_never_contains_raw_document_dump_markers(self, monkeypatch):
        app = _build_app(monkeypatch, **_clinical_critical_kwargs())
        result = app.invoke(_initial_state())
        combined = _flatten_state_to_text(result).casefold()

        for marker in ("document brut", "contenu brut du document"):
            assert marker not in combined

    def test_audit_trail_details_values_are_short_and_bounded(self, monkeypatch):
        """Chaque valeur de ``AuditEvent.details`` reste courte — jamais un
        contenu de document ou un prompt complet copié dans l'audit."""
        app = _build_app(monkeypatch, **_clinical_critical_kwargs())
        result = app.invoke(_initial_state())

        for event in result.get("audit_trail", []):
            for value in event.details.values():
                assert len(value) <= 500, f"valeur d'audit anormalement longue : {value[:80]}..."
