from schemas.domain import VerificationStatus
from schemas.results import MedicalCodingResult
from agents.medical_coding_agent.agent import node, run
from agents.medical_coding_agent.schemas import LlmCodingDecision, LlmResolvedCode
from agents.medical_coding_agent.tools import rechercher_code


def test_medical_coding_run_passes_on_exact_matches():
    result = run(
        case_id="CLM-0001",
        procedures=["Office Visit"],
        medications=["Acetaminophen 325 MG Oral Tablet"],
    )

    assert isinstance(result, MedicalCodingResult)
    assert result.status == VerificationStatus.PASS
    assert len(result.codings) == 2
    assert all(c.proposed_code for c in result.codings)
    assert result.llm_metadata is not None


def test_medical_coding_same_input_and_version_is_deterministic():
    first = run(case_id="CLM-0001", procedures=["Office Visit"])
    second = run(case_id="CLM-0001", procedures=["Office Visit"])

    assert first.table_version == second.table_version == "1.1.0"
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_medical_coding_run_needs_review_on_unknown_description():
    # P4-1 : description choisie hors seuil de similarité floue pour exercer
    # spécifiquement le palier mots-clés (voir tests/rules/test_medical_codes.py
    # TestFuzzyMatching pour le comportement flou lui-même).
    result = run(case_id="CLM-0001", procedures=["Random unclassified surgical intervention xyz123"])

    assert result.status == VerificationStatus.NEEDS_REVIEW
    assert result.codings[0].rule_applied == "keyword_match"
    assert result.llm_metadata is not None


def test_medical_coding_run_needs_review_without_items():
    result = run(case_id="CLM-0001")

    assert result.status == VerificationStatus.NEEDS_REVIEW
    assert result.codings == []
    assert result.llm_metadata is not None


def test_medical_coding_appelle_llm_meme_sur_correspondance_exacte(monkeypatch):
    calls = []

    def fake_llm(needs_review, already_coded):
        calls.append((needs_review, already_coded))
        return LlmCodingDecision(
            resolved=[],
            overall_rationale="Validation LLM des codes référentiels.",
        )

    monkeypatch.setattr("agents.medical_coding_agent.agent._invoke_llm_react", fake_llm)

    result = run(case_id="CLM-0001", procedures=["Office Visit"])

    assert len(calls) == 1
    needs_review, already_coded = calls[0]
    assert needs_review == []
    assert len(already_coded) == 1
    assert result.status == VerificationStatus.PASS
    assert "Validation LLM des codes référentiels." in result.reasons


def test_medical_coding_appelle_llm_meme_sans_item(monkeypatch):
    calls = []

    def fake_llm(needs_review, already_coded):
        calls.append((needs_review, already_coded))
        return LlmCodingDecision(resolved=[], overall_rationale="Aucun item à coder.")

    monkeypatch.setattr("agents.medical_coding_agent.agent._invoke_llm_react", fake_llm)

    result = run(case_id="CLM-0001")

    assert calls == [([], [])]
    assert result.status == VerificationStatus.NEEDS_REVIEW


class TestFuzzyMatchLlmMerge:
    """P4-1 — le LLM ne peut jamais faire passer un candidat flou en PASS, et
    ne peut choisir un code que parmi les candidats déjà proposés."""

    _FUZZY_DESCRIPTION = "Consultation ophtalmologiqe durgence"

    def test_llm_confirms_a_real_candidate_stays_needs_review(self, monkeypatch):
        def fake_llm(needs_review, already_coded):
            return LlmCodingDecision(
                resolved=[
                    LlmResolvedCode(
                        description=self._FUZZY_DESCRIPTION,
                        proposed_code="308292007",  # Consultation ophtalmologique
                        rationale="Candidat flou le plus proche confirmé.",
                    )
                ],
                overall_rationale="Un candidat flou confirmé.",
            )

        monkeypatch.setattr("agents.medical_coding_agent.agent._invoke_llm_react", fake_llm)

        result = run(case_id="CLM-0001", procedures=[self._FUZZY_DESCRIPTION])

        coding = result.codings[0]
        assert coding.rule_applied == "fuzzy_match_llm_selected"
        assert coding.proposed_code == "308292007"
        # Jamais de PASS automatique sur une correspondance approximative,
        # même confirmée par le LLM.
        assert coding.status == VerificationStatus.NEEDS_REVIEW
        assert result.status == VerificationStatus.NEEDS_REVIEW

    def test_llm_proposes_code_outside_candidates_is_rejected(self, monkeypatch):
        def fake_llm(needs_review, already_coded):
            return LlmCodingDecision(
                resolved=[
                    LlmResolvedCode(
                        description=self._FUZZY_DESCRIPTION,
                        # Code réel du référentiel (Paracétamol, section
                        # medications) mais hors des candidats flous proposés
                        # pour cette description côté procedures.
                        proposed_code="313782",
                        rationale="Code substitué hors liste bornée.",
                    )
                ],
                overall_rationale="Tentative de sortie de la liste bornée.",
            )

        monkeypatch.setattr("agents.medical_coding_agent.agent._invoke_llm_react", fake_llm)

        result = run(case_id="CLM-0001", procedures=[self._FUZZY_DESCRIPTION])

        coding = result.codings[0]
        assert coding.rule_applied == "fuzzy_match_no_selection"
        assert coding.proposed_code is None
        assert coding.status == VerificationStatus.NEEDS_REVIEW

    def test_llm_provides_no_code_for_fuzzy_item(self, monkeypatch):
        def fake_llm(needs_review, already_coded):
            return LlmCodingDecision(
                resolved=[
                    LlmResolvedCode(
                        description=self._FUZZY_DESCRIPTION,
                        proposed_code=None,
                        rationale="Aucun candidat retenu.",
                    )
                ],
                overall_rationale="Aucune confirmation.",
            )

        monkeypatch.setattr("agents.medical_coding_agent.agent._invoke_llm_react", fake_llm)

        result = run(case_id="CLM-0001", procedures=[self._FUZZY_DESCRIPTION])

        coding = result.codings[0]
        assert coding.rule_applied == "fuzzy_match_no_selection"
        assert coding.proposed_code is None


class TestRechercherCodeToolFuzzyEnrichment:
    """P4-1 — l'outil LLM enrichit sa sortie avec des candidats flous
    structurés, jamais persistés dans ProcedureCoding/ClaimState."""

    def test_fuzzy_candidates_present_when_rule_applied_is_fuzzy(self):
        payload = rechercher_code.invoke(
            {"description": "Consultation ophtalmologiqe durgence", "section": "procedures"}
        )

        assert payload["rule_applied"] == "fuzzy_candidates_found"
        assert "fuzzy_candidates" in payload
        assert payload["fuzzy_candidates"]
        for candidate in payload["fuzzy_candidates"]:
            assert set(candidate) == {"code", "label", "system", "similarity_score"}

    def test_fuzzy_candidates_absent_on_exact_match(self):
        payload = rechercher_code.invoke(
            {"description": "Office Visit", "section": "procedures"}
        )

        assert payload["rule_applied"] == "exact_match"
        assert "fuzzy_candidates" not in payload


def test_medical_coding_node_consumes_input_and_adds_audit():
    updates = node(
        {
            "case_id": "CLM-0001",
            "coding_input": {
                "case_id": "CLM-0001",
                "procedures": ["Office Visit"],
                "medications": [],
            },
        }
    )

    assert updates["coding_input"] is None
    assert updates["current_step"] == "medical_coding"
    assert updates["completed_steps"] == ["medical_coding"]
    assert updates["audit_trail"]
    assert isinstance(updates["coding_result"], MedicalCodingResult)
