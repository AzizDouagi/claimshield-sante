"""Scanners déterministes de sécurité — aucun appel LLM.

Le scanner analyse des textes issus de messages utilisateur, métadonnées,
couches texte PDF, aperçus OCR, sorties d'agents et arguments d'outils.
Chaque fonction est pure et testable de façon isolée.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from security.policies import (
    DEFAULT_POLICY,
    PROMPT_INJECTION_PATTERNS,
    POLICY_PATH_ABSOLUTE,
    POLICY_PATH_TRAVERSAL,
    POLICY_URL_CREDENTIALS_FORBIDDEN,
    POLICY_URL_EXTERNAL_FORBIDDEN,
    POLICY_URL_LOCALHOST_FORBIDDEN,
    POLICY_URL_PRIVATE_IP_FORBIDDEN,
    POLICY_URL_SCHEME_FORBIDDEN,
    SecurityPolicy,
    validate_storage_path,
    validate_url_policy,
)


SEVERITY_NONE = "NONE"
SEVERITY_LOW = "LOW"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_HIGH = "HIGH"
SEVERITY_CRITICAL = "CRITICAL"

CATEGORY_INVISIBLE_CHARS = "INVISIBLE_CHARS"
CATEGORY_FRAGMENTED_TEXT = "FRAGMENTED_TEXT"
CATEGORY_IGNORE_INSTRUCTIONS = "IGNORE_INSTRUCTIONS"
CATEGORY_SECRET_EXPOSURE = "SECRET_EXPOSURE"
CATEGORY_ENV_ACCESS = "ENV_ACCESS"
CATEGORY_TOOL_EXECUTION = "TOOL_EXECUTION"
CATEGORY_EXFILTRATION_URL = "EXFILTRATION_URL"
CATEGORY_PERMISSION_CHANGE = "PERMISSION_CHANGE"
CATEGORY_HIDDEN_DOCUMENT_INSTRUCTION = "HIDDEN_DOCUMENT_INSTRUCTION"
CATEGORY_SUSPICIOUS_PATH = "SUSPICIOUS_PATH"
CATEGORY_SUSPICIOUS_URL = "SUSPICIOUS_URL"
CATEGORY_LEGACY_REGEX = "LEGACY_REGEX"

_MAX_TRIGGER_LENGTH = 160
_INVISIBLE_CATEGORIES = {"Cf", "Cc"}
_URL_RE = re.compile(r"\b(?:https?|ftp|file)://[^\s<>'\")]+", re.IGNORECASE)
_PATH_RE = re.compile(
    r"(?:^|[\s:'\"])(?P<path>(?:\.\.[/\\]|/|[A-Za-z]:[/\\]|\\\\)[^\s<>'\"]+)"
)

_ODD_SPACE_RE = re.compile(r"[\u00a0\u1680\u180e\u2000-\u200f\u2028-\u202f\u205f\u2060\u3000]+")
_SPACED_WORD_RE = re.compile(r"\b(?:[a-z]\s+){3,}[a-z]\b")
_WORD_FRAGMENT_RE = re.compile(r"(?<=\b[a-z])[\s._\-:/\\]+(?=[a-z]\b)")
_COMPACT_SEPARATORS_RE = re.compile(r"[^a-z0-9]+")

_IGNORE_ACTION_RE = re.compile(
    r"\b(?:ignore|ignorer|ignorez|disregard|forget|override|bypass)\b.{0,80}"
    r"\b(?:instruction|rule|règle|regle|policy|politique|constraint|contrainte|"
    r"system|système|systeme|security|sécurité|securite|previous|précédente|"
    r"precedente|prior|above)s?\b",
    re.IGNORECASE,
)
_SECRET_RE = re.compile(
    r"\b(?:reveal|show|print|dump|expose|display|read|open|access|affiche|"
    r"afficher|révèle|revele|lire|lis|ouvre)\b.{0,80}"
    r"\b(?:system\s+prompt|prompt\s+système|prompt\s+systeme|secret|secrets|"
    r"token|api[_ -]?key|password|credential)s?\b",
    re.IGNORECASE,
)
_ENV_RE = re.compile(
    r"\b(?:read|open|cat|print|dump|access|show|lire|lis|ouvre|affiche|"
    r"afficher)\b.{0,80}(?:\.env|env\s+file|environment\s+file|fichier\s+\.?env)",
    re.IGNORECASE,
)
_TOOL_EXECUTION_RE = re.compile(
    r"\b(?:execute|run|call|invoke|use)\b.{0,80}"
    r"\b(?:shell|terminal|subprocess|os\.system|command|bash|sh|exec|eval)\b",
    re.IGNORECASE,
)
_EXFILTRATION_RE = re.compile(
    r"\b(?:send|post|upload|exfiltrate|forward|leak)\b.{0,100}"
    r"\b(?:data|claim|document|file|content|result|secret|token|payload)s?\b.{0,120}"
    r"\b(?:url|http|https|webhook|endpoint|server)\b",
    re.IGNORECASE,
)
_PERMISSION_RE = re.compile(
    r"\b(?:disable|turn\s+off|bypass|override|change|grant|escalate)\b.{0,80}"
    r"\b(?:security|policy|permission|safety|guard|validation|access)\b",
    re.IGNORECASE,
)
_HIDDEN_DOC_RE = re.compile(
    r"\b(?:hidden|invisible|ocr|pdf|document)\b.{0,80}"
    r"\b(?:instruction|prompt|command|directive)s?\b",
    re.IGNORECASE,
)

_COMPACT_DANGEROUS_PATTERNS: tuple[tuple[str, str], ...] = (
    ("ignorepreviousinstructions", CATEGORY_IGNORE_INSTRUCTIONS),
    ("ignoreallrules", CATEGORY_IGNORE_INSTRUCTIONS),
    ("revealssystemprompt", CATEGORY_SECRET_EXPOSURE),
    ("revealthesystemprompt", CATEGORY_SECRET_EXPOSURE),
    ("showyoursecrets", CATEGORY_SECRET_EXPOSURE),
    ("readtheenvfile", CATEGORY_ENV_ACCESS),
    ("read.env", CATEGORY_ENV_ACCESS),
    ("executethiscommand", CATEGORY_TOOL_EXECUTION),
    ("calltheshelltool", CATEGORY_TOOL_EXECUTION),
    ("sendthedatatothisurl", CATEGORY_EXFILTRATION_URL),
    ("disablesecurity", CATEGORY_PERMISSION_CHANGE),
    ("overridepolicy", CATEGORY_PERMISSION_CHANGE),
)


@dataclass(frozen=True)
class NormalizedText:
    """Texte normalisé et signaux de normalisation."""

    original_excerpt: str
    normalized: str
    compact: str
    invisible_characters: tuple[str, ...] = ()
    fragmented_sequences: tuple[str, ...] = ()
    truncated: bool = False


@dataclass(frozen=True)
class TextFinding:
    """Anomalie textuelle minimisée détectée par le scanner."""

    category: str
    severity: str
    trigger: str
    evidence: str


@dataclass(frozen=True)
class TextScanResult:
    """Résultat structuré du scan d'un texte."""

    detected: bool
    severity: str
    findings: tuple[TextFinding, ...] = field(default_factory=tuple)
    normalized: NormalizedText | None = None

    @property
    def triggers(self) -> list[str]:
        return [finding.trigger for finding in self.findings]


def _minimize(value: str, max_length: int = _MAX_TRIGGER_LENGTH) -> str:
    text = " ".join(value.split())
    return text[:max_length]


def normalize_security_text(text: str, max_length: int = 10_000) -> NormalizedText:
    """Normalise le texte avant scan.

    Étapes :
      - normalisation Unicode NFKC ;
      - casefold ;
      - réduction des espaces inhabituels ;
      - détection des caractères invisibles ;
      - détection des mots volontairement fragmentés ;
      - troncature déterministe à `max_length`.
    """
    original_excerpt = text[:_MAX_TRIGGER_LENGTH]
    truncated = len(text) > max_length
    limited = text[:max_length]
    unicode_normalized = unicodedata.normalize("NFKC", limited)

    invisible = tuple(
        f"U+{ord(ch):04X}"
        for ch in unicode_normalized
        if unicodedata.category(ch) in _INVISIBLE_CATEGORIES and ch not in "\n\r\t"
    )
    without_odd_spaces = _ODD_SPACE_RE.sub(" ", unicode_normalized)
    normalized = re.sub(r"\s+", " ", without_odd_spaces).strip().casefold()
    fragmented = tuple(_SPACED_WORD_RE.findall(normalized))
    defragmented = _WORD_FRAGMENT_RE.sub("", normalized)
    compact = _COMPACT_SEPARATORS_RE.sub("", defragmented)

    return NormalizedText(
        original_excerpt=original_excerpt,
        normalized=normalized,
        compact=compact,
        invisible_characters=invisible,
        fragmented_sequences=fragmented,
        truncated=truncated,
    )


def _severity_rank(severity: str) -> int:
    order = {
        SEVERITY_NONE: 0,
        SEVERITY_LOW: 1,
        SEVERITY_MEDIUM: 2,
        SEVERITY_HIGH: 3,
        SEVERITY_CRITICAL: 4,
    }
    return order[severity]


def _max_severity(findings: list[TextFinding]) -> str:
    if not findings:
        return SEVERITY_NONE
    return max((f.severity for f in findings), key=_severity_rank)


def _append_finding(
    findings: list[TextFinding],
    category: str,
    severity: str,
    trigger: str,
    evidence: str,
) -> None:
    key = (category, trigger)
    if any((f.category, f.trigger) == key for f in findings):
        return
    findings.append(TextFinding(
        category=category,
        severity=severity,
        trigger=trigger,
        evidence=_minimize(evidence),
    ))


def _scan_regex_intentions(
    normalized: NormalizedText,
    findings: list[TextFinding],
) -> None:
    rules: tuple[tuple[re.Pattern[str], str, str, str], ...] = (
        (_IGNORE_ACTION_RE, CATEGORY_IGNORE_INSTRUCTIONS, SEVERITY_HIGH, "instruction_override"),
        (_SECRET_RE, CATEGORY_SECRET_EXPOSURE, SEVERITY_CRITICAL, "secret_exposure"),
        (_ENV_RE, CATEGORY_ENV_ACCESS, SEVERITY_CRITICAL, "env_file_access"),
        (_TOOL_EXECUTION_RE, CATEGORY_TOOL_EXECUTION, SEVERITY_CRITICAL, "tool_execution"),
        (_EXFILTRATION_RE, CATEGORY_EXFILTRATION_URL, SEVERITY_CRITICAL, "url_exfiltration"),
        (_PERMISSION_RE, CATEGORY_PERMISSION_CHANGE, SEVERITY_HIGH, "permission_change"),
        (_HIDDEN_DOC_RE, CATEGORY_HIDDEN_DOCUMENT_INSTRUCTION, SEVERITY_HIGH, "hidden_instruction"),
    )
    for pattern, category, severity, trigger in rules:
        match = pattern.search(normalized.normalized)
        if match:
            _append_finding(findings, category, severity, trigger, match.group(0))


def _scan_compact_fragments(
    normalized: NormalizedText,
    findings: list[TextFinding],
) -> None:
    if normalized.fragmented_sequences:
        _append_finding(
            findings,
            CATEGORY_FRAGMENTED_TEXT,
            SEVERITY_MEDIUM,
            "fragmented_text",
            normalized.fragmented_sequences[0],
        )
    for compact_phrase, category in _COMPACT_DANGEROUS_PATTERNS:
        if compact_phrase in normalized.compact:
            severity = SEVERITY_CRITICAL if category in {
                CATEGORY_SECRET_EXPOSURE,
                CATEGORY_ENV_ACCESS,
                CATEGORY_TOOL_EXECUTION,
                CATEGORY_EXFILTRATION_URL,
            } else SEVERITY_HIGH
            _append_finding(findings, category, severity, compact_phrase, compact_phrase)


def _scan_urls_and_paths(
    normalized: NormalizedText,
    policy: SecurityPolicy,
    findings: list[TextFinding],
) -> None:
    for url in _URL_RE.findall(normalized.normalized):
        ok, codes = validate_url_policy(url, policy.url)
        dangerous_codes = {
            POLICY_URL_CREDENTIALS_FORBIDDEN,
            POLICY_URL_EXTERNAL_FORBIDDEN,
            POLICY_URL_LOCALHOST_FORBIDDEN,
            POLICY_URL_PRIVATE_IP_FORBIDDEN,
            POLICY_URL_SCHEME_FORBIDDEN,
        }
        if not ok and dangerous_codes.intersection(codes):
            _append_finding(
                findings,
                CATEGORY_SUSPICIOUS_URL,
                SEVERITY_HIGH,
                ",".join(codes),
                url,
            )

    for match in _PATH_RE.finditer(normalized.normalized):
        path = match.group("path")
        ok, codes = validate_storage_path(path, policy.path)
        dangerous_codes = {POLICY_PATH_ABSOLUTE, POLICY_PATH_TRAVERSAL}
        if not ok and dangerous_codes.intersection(codes):
            severity = SEVERITY_CRITICAL if ".env" in path else SEVERITY_HIGH
            _append_finding(
                findings,
                CATEGORY_SUSPICIOUS_PATH,
                severity,
                ",".join(codes),
                path,
            )


def scan_text_security(
    text: str,
    policy: SecurityPolicy = DEFAULT_POLICY,
    source: str | None = None,
) -> TextScanResult:
    """Scanne un texte avec normalisation, intentions et contexte déterministe."""
    normalized = normalize_security_text(text, policy.max_text_length)
    findings: list[TextFinding] = []

    if normalized.invisible_characters:
        _append_finding(
            findings,
            CATEGORY_INVISIBLE_CHARS,
            SEVERITY_LOW,
            "invisible_characters",
            " ".join(normalized.invisible_characters[:8]),
        )

    _scan_regex_intentions(normalized, findings)
    _scan_compact_fragments(normalized, findings)
    _scan_urls_and_paths(normalized, policy, findings)

    for pattern in policy.injection_patterns:
        if re.search(pattern, normalized.normalized, re.IGNORECASE):
            _append_finding(
                findings,
                CATEGORY_LEGACY_REGEX,
                SEVERITY_HIGH,
                pattern,
                pattern,
            )

    if source in {"pdf_text", "ocr_preview", "agent_output"}:
        for finding in list(findings):
            if finding.category in {
                CATEGORY_IGNORE_INSTRUCTIONS,
                CATEGORY_SECRET_EXPOSURE,
                CATEGORY_ENV_ACCESS,
                CATEGORY_TOOL_EXECUTION,
                CATEGORY_EXFILTRATION_URL,
                CATEGORY_PERMISSION_CHANGE,
            }:
                _append_finding(
                    findings,
                    CATEGORY_HIDDEN_DOCUMENT_INSTRUCTION,
                    SEVERITY_HIGH,
                    f"hidden_instruction_in_{source}",
                    finding.evidence,
                )
                break

    dangerous = [
        f for f in findings
        if f.category != CATEGORY_INVISIBLE_CHARS
        and not (f.category == CATEGORY_FRAGMENTED_TEXT and len(findings) == 1)
    ]

    return TextScanResult(
        detected=bool(dangerous),
        severity=_max_severity(findings),
        findings=tuple(findings),
        normalized=normalized,
    )


def scan_for_prompt_injection(
    text: str,
    patterns: tuple[str, ...] = PROMPT_INJECTION_PATTERNS,
) -> tuple[bool, list[str]]:
    """Recherche une injection de prompt dans un texte.

    Compatibilité historique : retourne `(détecté, triggers)`.
    """
    policy = SecurityPolicy(name="scanner", injection_patterns=patterns)
    result = scan_text_security(text, policy)
    return result.detected, result.triggers


def scan_claim_fields(
    fields: dict[str, str],
    policy: SecurityPolicy = DEFAULT_POLICY,
) -> dict[str, list[str]]:
    """Scanne chaque champ texte et retourne `{nom_champ: [triggers]}`.

    Les valeurs non-string sont ignorées silencieusement.
    Les valeurs dépassant `policy.max_text_length` sont tronquées avant analyse.
    """
    results: dict[str, list[str]] = {}
    for name, value in fields.items():
        if not isinstance(value, str) or not value:
            continue
        result = scan_text_security(value, policy, source=name)
        if result.detected:
            results[name] = result.triggers
    return results
