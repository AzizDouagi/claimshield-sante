"""Tests unitaires purs de `graph/input_builders.py` — aucun orchestrateur,
aucun LangGraph, aucun appel LLM. Un `ClaimState` synthétique est construit à
la main pour chaque cas.

**Limite MVP couverte par cette suite** (voir docstring de module de
`graph/input_builders.py`) : ces builders ne sélectionnent qu'**un seul**
document par dossier pour l'OCR — jamais un vrai fan-out multi-documents
(`langgraph.types.Send`, hors périmètre, décision AZIZ). Les tests
vérifient explicitement cette limite (un seul candidat retenu même quand
plusieurs sont éligibles) plutôt que de la contourner.
"""
from __future__ import annotations

from graph.input_builders import (
    build_coding_input,
    build_fhir_input,
    build_identity_coverage_input,
    build_ocr_input,
)
from schemas.domain import FileStatus, IntakeStatus, SecurityDecision
from schemas.results import (
    ClaimIntakeResult,
    ClaimManifest,
    InspectedFile,
    LlmMetadata,
    SecurityGateResult,
)

_CASE_ID = "CLM-0001"


def _file(
    name: str,
    *,
    mime: str = "application/pdf",
    status: FileStatus = FileStatus.ACCEPTED,
    sha256: str | None = "a" * 64,
    path: str | None = None,
) -> InspectedFile:
    return InspectedFile(
        original_name=name,
        storage_name=f"{_CASE_ID}_{name}",
        normalized_extension=name.rsplit(".", 1)[-1],
        detected_mime_type=mime,
        actual_size=1234,
        sha256=sha256,
        status=status,
        relative_storage_path=path if path is not None else f"incoming/{_CASE_ID}/{name}",
    )


def _intake_result(files: list[InspectedFile]) -> ClaimIntakeResult:
    manifest = ClaimManifest(
        claim_id=_CASE_ID,
        file_count=len(files),
        total_size_bytes=sum(f.actual_size for f in files),
        files=files,
        status=IntakeStatus.ACCEPTED,
    )
    return ClaimIntakeResult(
        claim_id=_CASE_ID,
        status=IntakeStatus.ACCEPTED,
        manifest=manifest,
        accepted_count=len(files),
        quarantined_count=0,
        llm_metadata=LlmMetadata(model_name="test-llm", prompt_version="test"),
    )


def _security_result(decision: SecurityDecision = SecurityDecision.ALLOW) -> SecurityGateResult:
    return SecurityGateResult(claim_id=_CASE_ID, decision=decision, reasons=["motif de test"])


def _base_state(files: list[InspectedFile], **extra) -> dict:
    state = {
        "case_id": _CASE_ID,
        "intake_result": _intake_result(files),
        "security_result": _security_result(),
    }
    state.update(extra)
    return state


class TestBuildOcrInput:
    def test_selects_facture_by_filename_keyword(self):
        files = [_file("ordonnance_CLM-0001.pdf"), _file("facture_CLM-0001.pdf")]
        result = build_ocr_input(_base_state(files))

        assert result["filename"] == "facture_CLM-0001.pdf"
        assert result["file_index"] == 1
        assert result["document_id"] == f"{_CASE_ID}-doc-1"

    def test_falls_back_to_first_candidate_by_position_when_no_facture(self):
        files = [_file("ordonnance_CLM-0001.pdf"), _file("demande_CLM-0001.pdf")]
        result = build_ocr_input(_base_state(files))

        assert result["filename"] == "ordonnance_CLM-0001.pdf"
        assert result["file_index"] == 0

    def test_excludes_fhir_bundle_from_candidates_even_at_position_zero(self):
        files = [
            _file("patient_fhir_bundle.json", mime="application/json"),
            _file("ordonnance_CLM-0001.pdf"),
        ]
        result = build_ocr_input(_base_state(files))

        assert result["filename"] == "ordonnance_CLM-0001.pdf"
        assert result["file_index"] == 1

    def test_ignores_quarantined_and_incomplete_files(self):
        files = [
            _file("suspect.pdf", status=FileStatus.QUARANTINED),
            _file("no_path.pdf", path=None),
            _file("no_hash.pdf", sha256=None),
            _file("facture_CLM-0001.pdf"),
        ]
        result = build_ocr_input(_base_state(files))

        assert result["filename"] == "facture_CLM-0001.pdf"
        assert result["file_index"] == 3

    def test_no_candidates_returns_none(self):
        files = [_file("patient_fhir_bundle.json", mime="application/json")]
        assert build_ocr_input(_base_state(files)) is None

    def test_missing_intake_result_returns_none(self):
        state = _base_state([_file("facture.pdf")])
        del state["intake_result"]
        assert build_ocr_input(state) is None

    def test_missing_security_result_returns_none(self):
        state = _base_state([_file("facture.pdf")])
        del state["security_result"]
        assert build_ocr_input(state) is None

    def test_missing_case_id_returns_none(self):
        state = _base_state([_file("facture.pdf")])
        del state["case_id"]
        assert build_ocr_input(state) is None

    def test_security_decision_passed_through_unchanged(self):
        state = _base_state(
            [_file("facture.pdf")], security_result=_security_result(SecurityDecision.ALLOW)
        )
        result = build_ocr_input(state)
        assert result["security_decision"] == SecurityDecision.ALLOW.value

    def test_accepts_dict_serialized_intake_and_security_result(self):
        state = _base_state([_file("facture.pdf")])
        state["intake_result"] = state["intake_result"].model_dump(mode="json")
        state["security_result"] = state["security_result"].model_dump(mode="json")

        result = build_ocr_input(state)

        assert result is not None
        assert result["filename"] == "facture.pdf"

    def test_missing_key_entirely_does_not_raise(self):
        state = {"case_id": _CASE_ID}
        assert build_ocr_input(state) is None


class TestBuildFhirInput:
    def test_single_candidate_found(self):
        files = [_file("patient_fhir_bundle.json", mime="application/json")]
        result = build_fhir_input(_base_state(files))

        assert result["bundle_expected"] is True
        assert result["fhir_bundle_path"] == f"{_CASE_ID}/patient_fhir_bundle.json"

    def test_zero_candidates_not_expected_not_provided(self):
        files = [_file("facture.pdf")]
        result = build_fhir_input(_base_state(files))

        assert result["bundle_expected"] is False
        assert result["fhir_bundle_path"] is None

    def test_ambiguous_candidates_returns_none(self):
        files = [
            _file("patient_fhir_bundle.json", mime="application/json"),
            _file("encounter_fhir_extra.json", mime="application/json"),
        ]
        assert build_fhir_input(_base_state(files)) is None

    def test_missing_intake_result_returns_none(self):
        state = _base_state([_file("facture.pdf")])
        del state["intake_result"]
        assert build_fhir_input(state) is None

    def test_missing_case_id_returns_none(self):
        state = _base_state([_file("facture.pdf")])
        del state["case_id"]
        assert build_fhir_input(state) is None


class TestBuildIdentityCoverageInput:
    def test_case_id_always_populated_even_without_bundle(self):
        files = [_file("facture.pdf")]
        result = build_identity_coverage_input(_base_state(files))

        assert result["case_id"] == _CASE_ID
        assert result["fhir_bundle_path"] is None

    def test_fhir_bundle_path_matches_detection(self):
        files = [_file("patient_fhir_bundle.json", mime="application/json")]
        result = build_identity_coverage_input(_base_state(files))

        assert result["fhir_bundle_path"] == f"{_CASE_ID}/patient_fhir_bundle.json"

    def test_ambiguous_bundle_results_in_none_path_not_none_builder(self):
        files = [
            _file("patient_fhir_bundle.json", mime="application/json"),
            _file("encounter_fhir_extra.json", mime="application/json"),
        ]
        result = build_identity_coverage_input(_base_state(files))

        assert result is not None
        assert result["fhir_bundle_path"] is None

    def test_missing_case_id_returns_none(self):
        state = _base_state([_file("facture.pdf")])
        del state["case_id"]
        assert build_identity_coverage_input(state) is None


class TestBuildCodingInput:
    def test_always_empty_lists_even_with_medical_items_in_ocr_result(self):
        """Non-régression sur la limitation MVP assumée (voir docstring de
        module) : même si `ocr_result` porte des `medical_items` détaillés,
        `coding_input` ne doit jamais tenter de les répartir en
        procédures/médicaments — aucun discriminant fiable n'existe."""
        state = {
            "case_id": _CASE_ID,
            # build_coding_input ne lit jamais ocr_result — un objet
            # quelconque suffit à prouver que sa présence est sans effet.
            "ocr_result": {
                "extraction": {
                    "essential_fields": {
                        "medical_items": [{"description": "Consultation généraliste"}]
                    }
                }
            },
        }
        result = build_coding_input(state)

        assert result == {"case_id": _CASE_ID, "procedures": [], "medications": []}

    def test_missing_case_id_returns_none(self):
        assert build_coding_input({}) is None
