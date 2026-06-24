"""Tests des politiques déterministes du Security Gate."""
from __future__ import annotations

from pathlib import Path

from schemas.domain import SecurityDecision, SeverityLevel
from security.policies import (
    DEFAULT_POLICY,
    DEFAULT_SEVERITY_POLICY,
    POLICY_EXECUTABLE_OR_SCRIPT,
    POLICY_EXTENSION_FORBIDDEN,
    POLICY_FILE_EMPTY,
    POLICY_FILE_TOO_LARGE,
    POLICY_MIME_EXTENSION_MISMATCH,
    POLICY_PATH_ABSOLUTE,
    POLICY_PATH_NULL_BYTE,
    POLICY_PATH_OUTSIDE_STORAGE,
    POLICY_PATH_TRAVERSAL,
    POLICY_PATH_ZONE_FORBIDDEN,
    POLICY_SUSPICIOUS_DOUBLE_EXTENSION,
    POLICY_TOOL_AGENT_FORBIDDEN,
    POLICY_TOOL_SECRET_ACCESS,
    POLICY_TOOL_SHELL_ACCESS,
    POLICY_TOOL_WRITE_PATH_FORBIDDEN,
    POLICY_URL_CREDENTIALS_FORBIDDEN,
    POLICY_URL_EXTERNAL_FORBIDDEN,
    POLICY_URL_LOCALHOST_FORBIDDEN,
    POLICY_URL_PRIVATE_IP_FORBIDDEN,
    POLICY_URL_SCHEME_FORBIDDEN,
    FilePolicy,
    PathPolicy,
    ToolPolicy,
    UrlPolicy,
    alert_for_severity,
    decision_for_severity,
    severity_rank,
    validate_file_policy,
    validate_storage_path,
    validate_tool_policy,
    validate_url_policy,
)


class TestFilePolicy:
    def test_fichier_pdf_valide(self):
        ok, codes = validate_file_policy(
            "invoice.pdf",
            "application/pdf",
            1024,
        )
        assert ok
        assert codes == []

    def test_fichier_vide_refuse(self):
        ok, codes = validate_file_policy("invoice.pdf", "application/pdf", 0)
        assert not ok
        assert POLICY_FILE_EMPTY in codes

    def test_fichier_trop_volumineux_refuse(self):
        policy = FilePolicy(max_file_size_bytes=10)
        ok, codes = validate_file_policy("invoice.pdf", "application/pdf", 11, policy)
        assert not ok
        assert POLICY_FILE_TOO_LARGE in codes

    def test_incoherence_extension_mime_refusee(self):
        ok, codes = validate_file_policy("invoice.pdf", "image/png", 1024)
        assert not ok
        assert POLICY_MIME_EXTENSION_MISMATCH in codes

    def test_script_refuse(self):
        ok, codes = validate_file_policy("prescription.php", "text/x-php", 1024)
        assert not ok
        assert POLICY_EXECUTABLE_OR_SCRIPT in codes
        assert POLICY_EXTENSION_FORBIDDEN in codes

    def test_double_extension_pdf_exe_refusee(self):
        ok, codes = validate_file_policy(
            "invoice.pdf.exe",
            "application/octet-stream",
            1024,
        )
        assert not ok
        assert POLICY_SUSPICIOUS_DOUBLE_EXTENSION in codes
        assert POLICY_EXECUTABLE_OR_SCRIPT in codes

    def test_double_extension_pdf_sh_refusee(self):
        ok, codes = validate_file_policy("document.pdf.sh", "text/x-shellscript", 1024)
        assert not ok
        assert POLICY_SUSPICIOUS_DOUBLE_EXTENSION in codes

    def test_double_extension_jpg_exe_refusee(self):
        ok, codes = validate_file_policy(
            "image.jpg.exe",
            "application/octet-stream",
            1024,
        )
        assert not ok
        assert POLICY_SUSPICIOUS_DOUBLE_EXTENSION in codes


class TestPathPolicy:
    def test_chemin_storage_valide(self, tmp_path: Path):
        policy = PathPolicy(storage_root=tmp_path / "storage")
        ok, codes = validate_storage_path("incoming/CLM-0001/file.pdf", policy)
        assert ok
        assert codes == []

    def test_traversee_repertoire_refusee(self, tmp_path: Path):
        policy = PathPolicy(storage_root=tmp_path / "storage")
        ok, codes = validate_storage_path("../../etc/passwd", policy)
        assert not ok
        assert POLICY_PATH_TRAVERSAL in codes
        assert POLICY_PATH_OUTSIDE_STORAGE in codes

    def test_traversee_windows_refusee(self, tmp_path: Path):
        policy = PathPolicy(storage_root=tmp_path / "storage")
        ok, codes = validate_storage_path(r"..\secrets.env", policy)
        assert not ok
        assert POLICY_PATH_TRAVERSAL in codes

    def test_chemin_absolu_unix_refuse(self, tmp_path: Path):
        policy = PathPolicy(storage_root=tmp_path / "storage")
        ok, codes = validate_storage_path("/Users/azizdouagi/.ssh/id_rsa", policy)
        assert not ok
        assert POLICY_PATH_ABSOLUTE in codes
        assert POLICY_PATH_OUTSIDE_STORAGE in codes

    def test_chemin_absolu_windows_refuse(self, tmp_path: Path):
        policy = PathPolicy(storage_root=tmp_path / "storage")
        ok, codes = validate_storage_path(r"C:\Users\secret.txt", policy)
        assert not ok
        assert POLICY_PATH_ABSOLUTE in codes

    def test_caractere_nul_refuse(self, tmp_path: Path):
        policy = PathPolicy(storage_root=tmp_path / "storage")
        ok, codes = validate_storage_path("incoming/a\x00.pdf", policy)
        assert not ok
        assert POLICY_PATH_NULL_BYTE in codes

    def test_zone_non_autorisee_refusee(self, tmp_path: Path):
        policy = PathPolicy(storage_root=tmp_path / "storage")
        ok, codes = validate_storage_path("processed/file.pdf", policy)
        assert not ok
        assert POLICY_PATH_ZONE_FORBIDDEN in codes


class TestUrlPolicy:
    def test_url_externe_refusee_par_defaut(self):
        ok, codes = validate_url_policy("https://example.com/callback")
        assert not ok
        assert POLICY_URL_EXTERNAL_FORBIDDEN in codes

    def test_url_allowlist_acceptee(self):
        policy = UrlPolicy(allowed_domains=("example.com",))
        ok, codes = validate_url_policy("https://example.com/callback", policy)
        assert ok
        assert codes == []

    def test_file_scheme_refuse(self):
        ok, codes = validate_url_policy("file:///etc/passwd")
        assert not ok
        assert POLICY_URL_SCHEME_FORBIDDEN in codes

    def test_ftp_refuse(self):
        ok, codes = validate_url_policy("ftp://example.com/file")
        assert not ok
        assert POLICY_URL_SCHEME_FORBIDDEN in codes

    def test_localhost_refuse(self):
        ok, codes = validate_url_policy("http://localhost:11434/api")
        assert not ok
        assert POLICY_URL_LOCALHOST_FORBIDDEN in codes

    def test_loopback_refuse(self):
        ok, codes = validate_url_policy("http://127.0.0.1:8000/admin")
        assert not ok
        assert POLICY_URL_LOCALHOST_FORBIDDEN in codes

    def test_ip_privee_refusee(self):
        ok, codes = validate_url_policy("http://169.254.169.254/")
        assert not ok
        assert POLICY_URL_PRIVATE_IP_FORBIDDEN in codes

    def test_identifiants_refuses(self):
        ok, codes = validate_url_policy("https://user:password@example.com")
        assert not ok
        assert POLICY_URL_CREDENTIALS_FORBIDDEN in codes


class TestToolPolicy:
    def test_outil_allowlist_accepte(self):
        ok, codes = validate_tool_policy("inspect_file", "claim_intake_agent")
        assert ok
        assert codes == []

    def test_shell_refuse(self):
        ok, codes = validate_tool_policy("shell", "security_gate_agent")
        assert not ok
        assert POLICY_TOOL_SHELL_ACCESS in codes

    def test_subprocess_refuse(self):
        ok, codes = validate_tool_policy("subprocess", "security_gate_agent")
        assert not ok
        assert POLICY_TOOL_SHELL_ACCESS in codes

    def test_os_system_refuse(self):
        ok, codes = validate_tool_policy("os.system", "security_gate_agent")
        assert not ok
        assert POLICY_TOOL_SHELL_ACCESS in codes

    def test_exec_eval_refuses(self):
        for tool_name in ("exec", "eval"):
            ok, codes = validate_tool_policy(tool_name, "security_gate_agent")
            assert not ok
            assert POLICY_TOOL_SHELL_ACCESS in codes

    def test_agent_non_autorise_refuse(self):
        ok, codes = validate_tool_policy("inspect_file", "unknown_agent")
        assert not ok
        assert POLICY_TOOL_AGENT_FORBIDDEN in codes

    def test_acces_secret_refuse(self):
        ok, codes = validate_tool_policy("inspect_file", "security_gate_agent", ".env")
        assert not ok
        assert POLICY_TOOL_SECRET_ACCESS in codes

    def test_ecriture_hors_zone_refusee(self):
        policy = ToolPolicy(writable_zones=("incoming",))
        ok, codes = validate_tool_policy(
            "inspect_file",
            "claim_intake_agent",
            "quarantine/file.pdf",
            policy=policy,
        )
        assert not ok
        assert POLICY_TOOL_WRITE_PATH_FORBIDDEN in codes


class TestSeverityPolicy:
    def test_regles_versionnees(self):
        assert DEFAULT_POLICY.version == "1.1.0"
        assert DEFAULT_SEVERITY_POLICY.version == "1.0.0"

    def test_rangs_deterministes(self):
        assert severity_rank(SeverityLevel.LOW) < severity_rank(SeverityLevel.MEDIUM)
        assert severity_rank(SeverityLevel.MEDIUM) < severity_rank(SeverityLevel.HIGH)
        assert severity_rank(SeverityLevel.HIGH) < severity_rank(SeverityLevel.CRITICAL)

    def test_decisions_par_severite(self):
        assert decision_for_severity(SeverityLevel.LOW) == SecurityDecision.ALLOW
        assert decision_for_severity(SeverityLevel.MEDIUM) == SecurityDecision.QUARANTINE
        assert decision_for_severity(SeverityLevel.HIGH) == SecurityDecision.BLOCK
        assert decision_for_severity(SeverityLevel.CRITICAL) == SecurityDecision.BLOCK

    def test_low_produit_alerte_sans_blocage(self):
        assert decision_for_severity(SeverityLevel.LOW) == SecurityDecision.ALLOW
        assert alert_for_severity(SeverityLevel.LOW) is True

    def test_seuils_documentes_dans_readme(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        assert "Seuils de sévérité" in readme
        for level in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            assert f"`{level}`" in readme
