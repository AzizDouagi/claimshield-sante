"""Politiques déterministes de sécurité.

Ces règles sont du code exécutable et testable. Elles ne reposent jamais sur un
prompt adressé au LLM : un prompt peut guider un modèle, mais il ne constitue
pas une barrière de sécurité.
"""
from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from schemas.domain import SecurityDecision, SeverityLevel


# ── Codes stables ────────────────────────────────────────────────────────────

POLICY_FILE_EMPTY = "EMPTY_FILE"
POLICY_FILE_TOO_LARGE = "FILE_TOO_LARGE"
POLICY_EXTENSION_FORBIDDEN = "UNSUPPORTED_EXTENSION"
POLICY_MIME_FORBIDDEN = "UNSUPPORTED_MIME"
POLICY_MIME_EXTENSION_MISMATCH = "MIME_EXTENSION_MISMATCH"
POLICY_EXECUTABLE_OR_SCRIPT = "UNSUPPORTED_EXTENSION"
POLICY_SUSPICIOUS_DOUBLE_EXTENSION = "SUSPICIOUS_DOUBLE_EXTENSION"

POLICY_PATH_ABSOLUTE = "ABSOLUTE_PATH_FORBIDDEN"
POLICY_PATH_TRAVERSAL = "PATH_TRAVERSAL"
POLICY_PATH_NULL_BYTE = "PATH_NULL_BYTE"
POLICY_PATH_OUTSIDE_STORAGE = "PATH_OUTSIDE_STORAGE"
POLICY_PATH_ZONE_FORBIDDEN = "STORAGE_ZONE_FORBIDDEN"

POLICY_URL_EXTERNAL_FORBIDDEN = "EXTERNAL_URL_FORBIDDEN"
POLICY_URL_SCHEME_FORBIDDEN = "DANGEROUS_URL_SCHEME"
POLICY_URL_LOCALHOST_FORBIDDEN = "PRIVATE_NETWORK_URL"
POLICY_URL_PRIVATE_IP_FORBIDDEN = "PRIVATE_NETWORK_URL"
POLICY_URL_CREDENTIALS_FORBIDDEN = "URL_CREDENTIALS_FORBIDDEN"
POLICY_URL_MALFORMED = "MALFORMED_URL"

POLICY_TOOL_FORBIDDEN = "UNAUTHORIZED_TOOL"
POLICY_TOOL_AGENT_FORBIDDEN = "UNAUTHORIZED_TOOL"
POLICY_TOOL_SECRET_ACCESS = "SECRET_ACCESS_ATTEMPT"
POLICY_TOOL_SHELL_ACCESS = "SHELL_ACCESS_ATTEMPT"
POLICY_TOOL_WRITE_PATH_FORBIDDEN = "WRITE_PATH_FORBIDDEN"


# ── Politique prompt injection ───────────────────────────────────────────────

# Patterns regex identifiant les tentatives d'injection de prompt.
# Appliqués en mode IGNORECASE sur le texte brut.
PROMPT_INJECTION_PATTERNS: tuple[str, ...] = (
    r"ignore\s+(all\s+)?(previous|prior|above|the)\s+(instructions?|prompts?|rules?|constraints?)",
    r"disregard\s+(all\s+)?(previous|prior|above|the)\s+(instructions?|prompts?|rules?)",
    r"(new\s+)?system\s+prompt[:\s]",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"jailbreak",
    r"\bDAN\s+(mode|prompt)\b",
    r"act\s+as\s+(if\s+you\s+are|a|an)\s+",
    r"ignore\s+(your|all)\s+(safety|ethical|moral)",
    r"pretend\s+(you\s+are|to\s+be)\s+",
    r"roleplay\s+as",
    r"forget\s+(you\s+are|your)\s+",
    r"override\s+(your\s+)?(previous|all|current)(\s+\w+)?\s+(instructions?|rules?|constraints?)",
    r"execute\s+the\s+following\s+(command|instruction|code)",
    r"\beval\s*\(",
    r"<\s*script[^>]*>",
    r"<!--.*inject",
)


# ── Politique fichiers ───────────────────────────────────────────────────────

FILE_MIME_BY_EXTENSION: dict[str, tuple[str, ...]] = {
    "pdf": ("application/pdf",),
    "png": ("image/png",),
    "jpg": ("image/jpeg",),
    "jpeg": ("image/jpeg",),
    "json": ("application/json",),
}

EXECUTABLE_AND_SCRIPT_EXTENSIONS: tuple[str, ...] = (
    "bat",
    "bin",
    "cmd",
    "com",
    "cpl",
    "dll",
    "exe",
    "jar",
    "js",
    "jse",
    "msi",
    "php",
    "ps1",
    "py",
    "scr",
    "sh",
    "vbs",
    "wsf",
)


@dataclass(frozen=True)
class FilePolicy:
    """Règles de validation des fichiers entrants."""

    allowed_extensions: tuple[str, ...] = ("pdf", "png", "jpg", "jpeg", "json")
    allowed_mime_types: tuple[str, ...] = (
        "application/pdf",
        "image/png",
        "image/jpeg",
        "application/json",
    )
    max_file_size_bytes: int = 20 * 1024 * 1024
    max_folder_size_bytes: int = 200 * 1024 * 1024
    max_files_per_folder: int = 50
    reject_empty_files: bool = True
    reject_mime_extension_mismatch: bool = True
    forbidden_extensions: tuple[str, ...] = EXECUTABLE_AND_SCRIPT_EXTENSIONS


# ── Politique chemins ────────────────────────────────────────────────────────

_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[/\\]")
_UNC_PATH_RE = re.compile(r"^(?:\\\\|//)")


@dataclass(frozen=True)
class PathPolicy:
    """Règles de résolution des chemins sous storage/."""

    storage_root: Path = Path("storage")
    allowed_zones: tuple[str, ...] = ("incoming", "quarantine", "temporary", "manifests")


# ── Politique URL ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class UrlPolicy:
    """Règles réseau : les URL externes sont interdites par défaut."""

    allow_external_urls: bool = False
    allowed_domains: tuple[str, ...] = ()
    allowed_schemes: tuple[str, ...] = ("http", "https")
    forbidden_schemes: tuple[str, ...] = ("file", "ftp")
    forbidden_hosts: tuple[str, ...] = ("localhost",)


# ── Politique outils ─────────────────────────────────────────────────────────

FORBIDDEN_TOOL_NAMES: tuple[str, ...] = (
    "eval",
    "exec",
    "os.system",
    "shell",
    "subprocess",
)


@dataclass(frozen=True)
class ToolPolicy:
    """Allowlist d'outils et privilèges minimaux par agent demandeur."""

    allowed_tools: tuple[str, ...] = (
        "compute_sha256",
        "detect_mime_type",
        "inspect_file",
        "scan_claim_fields",
        "scan_for_prompt_injection",
        "validate_storage_path",
    )
    forbidden_tools: tuple[str, ...] = FORBIDDEN_TOOL_NAMES
    allowed_requesting_agents: tuple[str, ...] = (
        "claim_intake_agent",
        "security_gate_agent",
        "orchestrator",
    )
    writable_zones: tuple[str, ...] = ("incoming", "quarantine", "temporary")
    secret_keywords: tuple[str, ...] = (
        ".env",
        "api_key",
        "authorization",
        "bearer",
        "password",
        "secret",
        "ssh",
        "token",
    )


@dataclass(frozen=True)
class SeverityDecisionRule:
    """Règle déterministe reliant une sévérité à une décision."""

    severity: SeverityLevel
    decision: SecurityDecision
    alert: bool
    description: str


DEFAULT_SEVERITY_RULES: tuple[SeverityDecisionRule, ...] = (
    SeverityDecisionRule(
        severity=SeverityLevel.LOW,
        decision=SecurityDecision.ALLOW,
        alert=True,
        description="Élément inhabituel sans danger immédiat.",
    ),
    SeverityDecisionRule(
        severity=SeverityLevel.MEDIUM,
        decision=SecurityDecision.QUARANTINE,
        alert=True,
        description="Incohérence nécessitant vérification.",
    ),
    SeverityDecisionRule(
        severity=SeverityLevel.HIGH,
        decision=SecurityDecision.BLOCK,
        alert=True,
        description="Action ou ressource interdite par la politique.",
    ),
    SeverityDecisionRule(
        severity=SeverityLevel.CRITICAL,
        decision=SecurityDecision.BLOCK,
        alert=True,
        description="Injection, secret, shell ou traversée de chemin.",
    ),
)


@dataclass(frozen=True)
class SeverityPolicy:
    """Seuils déterministes et versionnés de décision."""

    version: str = "1.0.0"
    rules: tuple[SeverityDecisionRule, ...] = field(
        default_factory=lambda: DEFAULT_SEVERITY_RULES
    )


@dataclass(frozen=True)
class SecurityPolicy:
    """Politique de sécurité applicable au Security Gate."""

    name: str
    version: str = "1.1.0"
    max_text_length: int = 10_000
    injection_patterns: tuple[str, ...] = field(
        default_factory=lambda: PROMPT_INJECTION_PATTERNS
    )
    block_on_injection: bool = True
    file: FilePolicy = field(default_factory=FilePolicy)
    path: PathPolicy = field(default_factory=PathPolicy)
    url: UrlPolicy = field(default_factory=UrlPolicy)
    tool: ToolPolicy = field(default_factory=ToolPolicy)
    severity: SeverityPolicy = field(default_factory=SeverityPolicy)


DEFAULT_FILE_POLICY = FilePolicy()
DEFAULT_PATH_POLICY = PathPolicy()
DEFAULT_URL_POLICY = UrlPolicy()
DEFAULT_TOOL_POLICY = ToolPolicy()
DEFAULT_SEVERITY_POLICY = SeverityPolicy()

# Politique par défaut — appliquée par le security_gate_agent.
DEFAULT_POLICY = SecurityPolicy(name="default")


_SEVERITY_RANK: dict[SeverityLevel, int] = {
    SeverityLevel.INFO: 0,
    SeverityLevel.LOW: 1,
    SeverityLevel.MEDIUM: 2,
    SeverityLevel.HIGH: 3,
    SeverityLevel.CRITICAL: 4,
}


def severity_rank(severity: SeverityLevel) -> int:
    """Rang déterministe d'une sévérité."""
    return _SEVERITY_RANK[severity]


def decision_for_severity(
    severity: SeverityLevel,
    policy: SeverityPolicy = DEFAULT_SEVERITY_POLICY,
) -> SecurityDecision:
    """Retourne la décision déterministe associée à une sévérité."""
    if severity == SeverityLevel.INFO:
        return SecurityDecision.ALLOW
    for rule in policy.rules:
        if rule.severity == severity:
            return rule.decision
    raise ValueError(f"Sévérité non couverte par la politique : {severity.value}")


def alert_for_severity(
    severity: SeverityLevel,
    policy: SeverityPolicy = DEFAULT_SEVERITY_POLICY,
) -> bool:
    """Indique si la sévérité doit produire une alerte d'audit."""
    if severity == SeverityLevel.INFO:
        return False
    for rule in policy.rules:
        if rule.severity == severity:
            return rule.alert
    raise ValueError(f"Sévérité non couverte par la politique : {severity.value}")


# ── Helpers fichiers ─────────────────────────────────────────────────────────


def _normalize_extension(ext: str) -> str:
    return ext.strip().lower().lstrip(".")


def _filename_suffixes(filename: str) -> tuple[str, ...]:
    return tuple(_normalize_extension(s) for s in Path(filename).suffixes if s)


def validate_file_policy(
    filename: str,
    detected_mime: str | None,
    size_bytes: int,
    policy: FilePolicy = DEFAULT_FILE_POLICY,
) -> tuple[bool, list[str]]:
    """Valide un fichier sans lire son contenu.

    Args:
        filename: Nom fourni pour le fichier.
        detected_mime: MIME réellement détecté par inspection du contenu.
        size_bytes: Taille réelle lue depuis le disque.
        policy: Politique fichier à appliquer.

    Returns:
        `(autorisé, codes_de_refus)`.
    """
    codes: list[str] = []
    suffixes = _filename_suffixes(filename)
    ext = suffixes[-1] if suffixes else ""
    detected = (detected_mime or "").strip().lower()

    if policy.reject_empty_files and size_bytes <= 0:
        codes.append(POLICY_FILE_EMPTY)
    if size_bytes > policy.max_file_size_bytes:
        codes.append(POLICY_FILE_TOO_LARGE)

    if ext in policy.forbidden_extensions:
        codes.append(POLICY_EXECUTABLE_OR_SCRIPT)
    if ext not in policy.allowed_extensions:
        codes.append(POLICY_EXTENSION_FORBIDDEN)

    if len(suffixes) > 1 and ext in policy.forbidden_extensions:
        codes.append(POLICY_SUSPICIOUS_DOUBLE_EXTENSION)
    if len(suffixes) > 1 and any(s in policy.allowed_extensions for s in suffixes[:-1]):
        if ext not in policy.allowed_extensions:
            if POLICY_SUSPICIOUS_DOUBLE_EXTENSION not in codes:
                codes.append(POLICY_SUSPICIOUS_DOUBLE_EXTENSION)

    if detected and detected not in policy.allowed_mime_types:
        codes.append(POLICY_MIME_FORBIDDEN)

    expected_mimes = FILE_MIME_BY_EXTENSION.get(ext, ())
    if (
        policy.reject_mime_extension_mismatch
        and detected
        and expected_mimes
        and detected not in expected_mimes
    ):
        codes.append(POLICY_MIME_EXTENSION_MISMATCH)

    return not codes, codes


# ── Helpers chemins ──────────────────────────────────────────────────────────


def _has_path_traversal(raw_path: str) -> bool:
    return any(part == ".." for part in re.split(r"[/\\]+", raw_path))


def validate_storage_path(
    path: str | Path,
    policy: PathPolicy = DEFAULT_PATH_POLICY,
) -> tuple[bool, list[str]]:
    """Valide un chemin relatif sous une zone autorisée de storage/.

    La fonction refuse les chemins absolus avant toute résolution, puis vérifie
    avec `Path.resolve()` que le chemin final reste sous `storage_root`.
    """
    codes: list[str] = []
    raw = str(path)

    if "\x00" in raw:
        codes.append(POLICY_PATH_NULL_BYTE)
    if Path(raw).is_absolute() or _WINDOWS_ABSOLUTE_RE.match(raw) or _UNC_PATH_RE.match(raw):
        codes.append(POLICY_PATH_ABSOLUTE)
    if _has_path_traversal(raw):
        codes.append(POLICY_PATH_TRAVERSAL)

    parts = [part for part in re.split(r"[/\\]+", raw) if part]
    if not parts or parts[0] not in policy.allowed_zones:
        codes.append(POLICY_PATH_ZONE_FORBIDDEN)

    if POLICY_PATH_NULL_BYTE not in codes:
        root = policy.storage_root.resolve()
        candidate = (root / raw).resolve()
        if not candidate.is_relative_to(root):
            codes.append(POLICY_PATH_OUTSIDE_STORAGE)

    return not codes, codes


# ── Helpers URL ──────────────────────────────────────────────────────────────


def _is_forbidden_host(host: str, policy: UrlPolicy) -> bool:
    normalized = host.strip().lower().rstrip(".")
    if normalized in policy.forbidden_hosts:
        return True
    try:
        ip = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


def validate_url_policy(
    url: str,
    policy: UrlPolicy = DEFAULT_URL_POLICY,
) -> tuple[bool, list[str]]:
    """Valide une URL avec urllib.parse et une allowlist optionnelle."""
    codes: list[str] = []
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()

    if not scheme or not parsed.netloc:
        codes.append(POLICY_URL_MALFORMED)
    if scheme in policy.forbidden_schemes or scheme not in policy.allowed_schemes:
        codes.append(POLICY_URL_SCHEME_FORBIDDEN)
    if parsed.username or parsed.password:
        codes.append(POLICY_URL_CREDENTIALS_FORBIDDEN)

    if host and _is_forbidden_host(host, policy):
        if host == "localhost" or host.startswith("127."):
            codes.append(POLICY_URL_LOCALHOST_FORBIDDEN)
        else:
            codes.append(POLICY_URL_PRIVATE_IP_FORBIDDEN)

    if host:
        allowed = host in policy.allowed_domains
        if not policy.allow_external_urls and not allowed:
            codes.append(POLICY_URL_EXTERNAL_FORBIDDEN)
        elif policy.allowed_domains and not allowed:
            codes.append(POLICY_URL_EXTERNAL_FORBIDDEN)

    return not codes, codes


# ── Helpers outils ───────────────────────────────────────────────────────────


def _contains_secret_hint(value: str, policy: ToolPolicy) -> bool:
    lowered = value.lower()
    return any(keyword in lowered for keyword in policy.secret_keywords)


def validate_tool_policy(
    tool_name: str,
    requesting_agent: str,
    write_path: str | Path | None = None,
    policy: ToolPolicy = DEFAULT_TOOL_POLICY,
    path_policy: PathPolicy = DEFAULT_PATH_POLICY,
) -> tuple[bool, list[str]]:
    """Valide l'usage d'un outil selon allowlist, agent et chemin d'écriture."""
    codes: list[str] = []
    normalized_tool = tool_name.strip().lower()

    if normalized_tool in policy.forbidden_tools:
        codes.append(POLICY_TOOL_SHELL_ACCESS)
    elif normalized_tool not in policy.allowed_tools:
        codes.append(POLICY_TOOL_FORBIDDEN)
    if requesting_agent not in policy.allowed_requesting_agents:
        codes.append(POLICY_TOOL_AGENT_FORBIDDEN)
    if _contains_secret_hint(tool_name, policy) or _contains_secret_hint(requesting_agent, policy):
        codes.append(POLICY_TOOL_SECRET_ACCESS)

    if write_path is not None:
        raw_path = str(write_path)
        if _contains_secret_hint(raw_path, policy):
            codes.append(POLICY_TOOL_SECRET_ACCESS)

        zone_policy = PathPolicy(
            storage_root=path_policy.storage_root,
            allowed_zones=policy.writable_zones,
        )
        path_ok, path_codes = validate_storage_path(raw_path, zone_policy)
        if not path_ok:
            codes.append(POLICY_TOOL_WRITE_PATH_FORBIDDEN)
            codes.extend(path_codes)

    return not codes, codes
