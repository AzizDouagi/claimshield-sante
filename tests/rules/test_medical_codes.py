from schemas.domain import VerificationStatus
from tools.medical_coding import code_medications, code_procedures, lookup_code
from tools.rule_loader import get_rule_version, load_rules


def test_medical_codes_load_versioned_table():
    table = load_rules("medical_codes.yaml")

    assert get_rule_version("medical_codes.yaml") == "1.1.0"
    assert table["ruleset"]["status"] == "active"
    assert table["rules"][0]["id"] == "MEDICAL_CODE_EXACT_MATCH"
    # Nouveau format : liste plate de codes (plus de dicts procedures/medications)
    codes = table["codes"]
    assert any(c["code"] == "11429006" for c in codes)     # Medical consultation
    assert any(c["code"] == "313782" for c in codes)        # Paracétamol / Acetaminophen


def test_lookup_code_exact_match_passes():
    # "Office Visit" est un synonyme anglais de 11429006
    coding = lookup_code("Office Visit", "procedures")

    assert coding.status == VerificationStatus.PASS
    assert coding.proposed_code == "11429006"
    assert coding.rule_applied == "exact_match"


def test_lookup_code_keyword_match_requires_review():
    # P4-1 : description choisie pour rester hors seuil de similarité floue
    # (min_similarity_score, cf. TestFuzzyMatching) — exerce spécifiquement
    # le palier mots-clés, désormais la 3ᵉ étape après l'étape floue.
    coding = lookup_code("Random unclassified surgical intervention xyz123", "procedures")

    assert coding.status == VerificationStatus.NEEDS_REVIEW
    assert coding.proposed_code is None
    assert coding.rule_applied == "keyword_match"


class TestFuzzyMatching:
    """P4-1 — correspondance approximative (rapidfuzz), insérée entre la
    correspondance exacte et le fallback mots-clés."""

    def test_near_duplicate_description_finds_fuzzy_candidates(self):
        coding = lookup_code("Consultation ophtalmologiqe durgence", "procedures")

        assert coding.status == VerificationStatus.NEEDS_REVIEW
        assert coding.proposed_code is None
        assert coding.rule_applied == "fuzzy_candidates_found"
        assert coding.alternatives  # au moins un candidat
        assert "308292007" in coding.alternatives  # Consultation ophtalmologique

    def test_fuzzy_candidates_always_exist_in_reference(self):
        from tools.medical_coding import code_exists_in_reference, find_fuzzy_candidates

        candidates = find_fuzzy_candidates("Unknown dental procedure", "procedures")

        assert candidates  # ce cas était auparavant classé keyword_match
        for candidate in candidates:
            assert code_exists_in_reference(candidate.code, "procedures")

    def test_fuzzy_candidates_deduplicated_by_code(self):
        from tools.medical_coding import find_fuzzy_candidates

        candidates = find_fuzzy_candidates("Consultation ophtalmologique", "procedures")

        codes = [c.code for c in candidates]
        assert len(codes) == len(set(codes))

    def test_no_fuzzy_candidates_below_threshold(self):
        from tools.medical_coding import find_fuzzy_candidates

        candidates = find_fuzzy_candidates(
            "Random unclassified surgical intervention xyz123", "procedures"
        )

        assert candidates == []

    def test_min_similarity_score_read_from_referential(self):
        from tools.medical_coding import find_fuzzy_candidates, load_code_table

        table = load_code_table()
        assert table.get("min_similarity_score") == 0.80

        # Un score_cutoff explicite plus permissif élargit les candidats.
        strict = find_fuzzy_candidates("dental procedure", "procedures", score_cutoff=0.95)
        loose = find_fuzzy_candidates("dental procedure", "procedures", score_cutoff=0.5)
        assert len(loose) >= len(strict)


def test_code_helpers_preserve_count():
    # Les deux descriptions correspondent à des synonymes dans le catalogue
    codings = [
        *code_procedures(["Office Visit"]),
        *code_medications(["Acetaminophen 325 MG Oral Tablet"]),
    ]

    assert len(codings) == 2
    assert all(c.status == VerificationStatus.PASS for c in codings)
