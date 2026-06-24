"""Tests des codes stables de sécurité."""
from __future__ import annotations

from pathlib import Path

from schemas.domain import FindingCode, SECURITY_CODE_DESCRIPTIONS, SECURITY_CODE_SEVERITIES


EXPECTED_SECURITY_CODES = {
    "UNSUPPORTED_EXTENSION",
    "UNSUPPORTED_MIME",
    "FILE_TOO_LARGE",
    "MIME_EXTENSION_MISMATCH",
    "PATH_TRAVERSAL",
    "ABSOLUTE_PATH_FORBIDDEN",
    "EXTERNAL_URL_FORBIDDEN",
    "PRIVATE_NETWORK_URL",
    "DANGEROUS_URL_SCHEME",
    "PROMPT_INJECTION_DETECTED",
    "SECRET_ACCESS_ATTEMPT",
    "SHELL_ACCESS_ATTEMPT",
    "UNAUTHORIZED_TOOL",
    "INVALID_AGENT_OUTPUT",
    "SUSPICIOUS_DOCUMENT_CONTENT",
}


def test_codes_demandes_presents():
    values = {code.value for code in FindingCode}
    assert EXPECTED_SECURITY_CODES <= values


def test_chaque_code_a_description_et_severite():
    for code in FindingCode:
        assert SECURITY_CODE_DESCRIPTIONS[code]
        assert SECURITY_CODE_SEVERITIES[code]


def test_codes_ne_contiennent_pas_de_donnees_medicales():
    medical_terms = {
        "patient",
        "diagnosis",
        "diagnostic",
        "traitement",
        "ordonnance",
        "remboursement",
        "médical",
        "medical",
    }
    for code in FindingCode:
        text = f"{code.value} {SECURITY_CODE_DESCRIPTIONS[code]}".casefold()
        assert not any(term in text for term in medical_terms)


def test_codes_documentes_dans_readme():
    readme = Path("README.md").read_text(encoding="utf-8")
    for code in FindingCode:
        assert f"`{code.value}`" in readme
