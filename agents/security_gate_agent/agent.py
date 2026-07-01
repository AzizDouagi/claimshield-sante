"""Security Gate Agent — validation de sécurité avec décision LLM finale.

Agent LLM (gemma4:latest via ChatOllama) + analyses de sécurité déterministes.

Pipeline :
  Phase A — analyses déterministes : fichier, chemin, URL, texte, outils, oracle.
  Phase B — LLM (with_structured_output) : décision ALLOW/BLOCK/QUARANTINE finale.
  Phase C — construction SecurityGateResult + audit.

Fallback : si le LLM est indisponible → BLOCK (principe de précaution).

Interdictions strictes :
  - Aucune analyse médicale ou clinique.
  - Aucune décision de remboursement.
  - Aucun contenu brut, secret ou chemin absolu dans le ClaimState.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from agents.security_gate_agent.prompt import load_security_gate_prompt
from agents.security_gate_agent.schemas import InputType, LlmSecurityDecision, SecurityGateInput
from llm.factory import get_llm
from llm.metadata import build_llm_metadata
from schemas.domain import FindingCode, SecurityDecision, SeverityLevel
from schemas.results import SecurityAuditEntry, SecurityFinding, SecurityGateResult
from security.policies import (
    DEFAULT_POLICY,
    POLICY_EXECUTABLE_OR_SCRIPT,
    POLICY_EXTENSION_FORBIDDEN,
    POLICY_FILE_EMPTY,
    POLICY_FILE_TOO_LARGE,
    POLICY_MIME_EXTENSION_MISMATCH,
    POLICY_MIME_FORBIDDEN,
    POLICY_PATH_ABSOLUTE,
    POLICY_PATH_NULL_BYTE,
    POLICY_PATH_OUTSIDE_STORAGE,
    POLICY_PATH_TRAVERSAL,
    POLICY_PATH_ZONE_FORBIDDEN,
    POLICY_SUSPICIOUS_DOUBLE_EXTENSION,
    POLICY_TOOL_AGENT_FORBIDDEN,
    POLICY_TOOL_FORBIDDEN,
    POLICY_TOOL_SECRET_ACCESS,
    POLICY_TOOL_SHELL_ACCESS,
    POLICY_TOOL_WRITE_PATH_FORBIDDEN,
    POLICY_URL_CREDENTIALS_FORBIDDEN,
    POLICY_URL_EXTERNAL_FORBIDDEN,
    POLICY_URL_LOCALHOST_FORBIDDEN,
    POLICY_URL_MALFORMED,
    POLICY_URL_PRIVATE_IP_FORBIDDEN,
    POLICY_URL_SCHEME_FORBIDDEN,
    SecurityPolicy,
    decision_for_severity,
    severity_rank,
    validate_file_policy,
    validate_storage_path,
    validate_tool_policy,
    validate_url_policy,
)
from security.scanners import (
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    scan_text_security,
)
from state.claim_state import ClaimState, validate_state_update

_AGENT_NAME = "security_gate_agent"

# Sévérité par défaut pour chaque code d'anomalie
_POLICY_CODE_TO_FINDING: dict[str, tuple[FindingCode, SeverityLevel]] = {
    POLICY_FILE_EMPTY: (FindingCode.EMPTY_FILE, SeverityLevel.HIGH),
    "FILE_METADATA_INCOMPLETE": (
        FindingCode.FILE_METADATA_INCOMPLETE,
        SeverityLevel.HIGH,
    ),
    POLICY_FILE_TOO_LARGE: (FindingCode.FILE_TOO_LARGE, SeverityLevel.HIGH),
    POLICY_EXTENSION_FORBIDDEN: (FindingCode.UNSUPPORTED_EXTENSION, SeverityLevel.HIGH),
    POLICY_MIME_FORBIDDEN: (FindingCode.UNSUPPORTED_MIME, SeverityLevel.HIGH),
    POLICY_MIME_EXTENSION_MISMATCH: (
        FindingCode.MIME_EXTENSION_MISMATCH,
        SeverityLevel.MEDIUM,
    ),
    POLICY_EXECUTABLE_OR_SCRIPT: (
        FindingCode.UNSUPPORTED_EXTENSION,
        SeverityLevel.CRITICAL,
    ),
    POLICY_SUSPICIOUS_DOUBLE_EXTENSION: (
        FindingCode.UNSUPPORTED_EXTENSION,
        SeverityLevel.CRITICAL,
    ),
    POLICY_PATH_ABSOLUTE: (FindingCode.ABSOLUTE_PATH_FORBIDDEN, SeverityLevel.CRITICAL),
    POLICY_PATH_NULL_BYTE: (FindingCode.PATH_NULL_BYTE, SeverityLevel.CRITICAL),
    POLICY_PATH_TRAVERSAL: (FindingCode.PATH_TRAVERSAL, SeverityLevel.CRITICAL),
    POLICY_PATH_OUTSIDE_STORAGE: (FindingCode.PATH_OUTSIDE_STORAGE, SeverityLevel.CRITICAL),
    POLICY_PATH_ZONE_FORBIDDEN: (FindingCode.STORAGE_ZONE_FORBIDDEN, SeverityLevel.HIGH),
    POLICY_URL_MALFORMED: (FindingCode.MALFORMED_URL, SeverityLevel.HIGH),
    POLICY_URL_SCHEME_FORBIDDEN: (FindingCode.DANGEROUS_URL_SCHEME, SeverityLevel.HIGH),
    POLICY_URL_LOCALHOST_FORBIDDEN: (FindingCode.PRIVATE_NETWORK_URL, SeverityLevel.CRITICAL),
    POLICY_URL_PRIVATE_IP_FORBIDDEN: (FindingCode.PRIVATE_NETWORK_URL, SeverityLevel.CRITICAL),
    POLICY_URL_CREDENTIALS_FORBIDDEN: (
        FindingCode.URL_CREDENTIALS_FORBIDDEN,
        SeverityLevel.HIGH,
    ),
    POLICY_URL_EXTERNAL_FORBIDDEN: (FindingCode.EXTERNAL_URL_FORBIDDEN, SeverityLevel.HIGH),
    POLICY_TOOL_FORBIDDEN: (FindingCode.UNAUTHORIZED_TOOL, SeverityLevel.CRITICAL),
    POLICY_TOOL_AGENT_FORBIDDEN: (FindingCode.UNAUTHORIZED_TOOL, SeverityLevel.HIGH),
    POLICY_TOOL_SECRET_ACCESS: (FindingCode.SECRET_ACCESS_ATTEMPT, SeverityLevel.CRITICAL),
    POLICY_TOOL_SHELL_ACCESS: (FindingCode.SHELL_ACCESS_ATTEMPT, SeverityLevel.CRITICAL),
    POLICY_TOOL_WRITE_PATH_FORBIDDEN: (FindingCode.WRITE_PATH_FORBIDDEN, SeverityLevel.HIGH),
}

_SCANNER_SEVERITY: dict[str, SeverityLevel] = {
    SEVERITY_CRITICAL: SeverityLevel.CRITICAL,
    SEVERITY_HIGH: SeverityLevel.HIGH,
    SEVERITY_MEDIUM: SeverityLevel.MEDIUM,
    SEVERITY_LOW: SeverityLevel.LOW,
}

# Action autorisée après chaque décision
_NEXT_ACTIONS: dict[SecurityDecision, str] = {
    SecurityDecision.ALLOW: "continue_pipeline",
    SecurityDecision.BLOCK: "terminate_pipeline",
    SecurityDecision.QUARANTINE: "await_human_review",
}

_DECISION_RANK: dict[SecurityDecision, int] = {
    SecurityDecision.ALLOW: 0,
    SecurityDecision.QUARANTINE: 1,
    SecurityDecision.BLOCK: 2,
}


def _add_reason_code(reason_codes: list[FindingCode], code: FindingCode) -> None:
    if code not in reason_codes:
        reason_codes.append(code)


def _safe_evidence(value: str | None) -> str | None:
    """Retourne une preuve minimisée sans contenu brut sensible."""
    if not value:
        return None
    return value[:120]


def _deterministic_confidence(
    findings: list[SecurityFinding],
    decision: SecurityDecision,
) -> float:
    """Score stable associé à la Phase A, avant arbitrage LLM."""
    if decision == SecurityDecision.BLOCK and findings:
        return 1.0
    if decision == SecurityDecision.QUARANTINE and findings:
        return 0.9
    if decision == SecurityDecision.ALLOW:
        return 0.95
    return 0.75


def _evidence_summary(
    findings: list[SecurityFinding],
    decision: SecurityDecision,
    llm_evidence: str | None = None,
) -> str:
    """Produit une preuve courte sans propager le contenu brut analysé."""
    for finding in findings:
        if finding.evidence:
            return finding.evidence
        if finding.code:
            return finding.code.value
    if llm_evidence:
        return llm_evidence[:200]
    if decision == SecurityDecision.ALLOW:
        return "Aucun signal bloquant détecté par les politiques et scanners."
    return "Décision conservatrice sans preuve brute persistée."


def _append_policy_findings(
    findings: list[SecurityFinding],
    reason_codes: list[FindingCode],
    blocked_fields: list[str],
    policy_codes: list[str],
    source: str,
    affected_element: str,
) -> None:
    for policy_code in policy_codes:
        code, severity = _POLICY_CODE_TO_FINDING.get(
            policy_code,
            (FindingCode.POLICY_VIOLATION, SeverityLevel.MEDIUM),
        )
        _add_reason_code(reason_codes, code)
        if affected_element not in blocked_fields:
            blocked_fields.append(affected_element)
        findings.append(SecurityFinding(
            code=code,
            severity=severity,
            description=(
                f"{affected_element} refusé par la politique de sécurité : {policy_code}"
            ),
            detection_source=source,
            affected_element=affected_element,
            evidence=policy_code,
        ))


def _scan_text_field(
    findings: list[SecurityFinding],
    reason_codes: list[FindingCode],
    blocked_fields: list[str],
    field_name: str,
    value: str | None,
    policy: SecurityPolicy,
    source: str | None = None,
) -> None:
    if not value:
        return
    result = scan_text_security(value, policy, source=source or field_name)
    if not result.detected:
        return

    if field_name not in blocked_fields:
        blocked_fields.append(field_name)
    _add_reason_code(reason_codes, FindingCode.PROMPT_INJECTION)

    scanner_severity = _SCANNER_SEVERITY.get(result.severity, SeverityLevel.MEDIUM)
    severity = max(
        scanner_severity,
        SeverityLevel.CRITICAL,
        key=severity_rank,
    )
    categories = ", ".join(sorted({finding.category for finding in result.findings}))
    findings.append(SecurityFinding(
        code=FindingCode.PROMPT_INJECTION,
        severity=severity,
        description=f"Champ '{field_name}' : contenu suspect détecté ({categories})",
        detection_source="text_security_scanner",
        affected_element=field_name,
        evidence=_safe_evidence(result.triggers[0] if result.triggers else None),
    ))


def _max_finding_severity(findings: list[SecurityFinding]) -> SeverityLevel:
    if not findings:
        return SeverityLevel.INFO
    return max((finding.severity for finding in findings), key=severity_rank)


def _decide(
    findings: list[SecurityFinding],
    policy: SecurityPolicy,
) -> SecurityDecision:
    if not findings:
        return SecurityDecision.ALLOW

    blocking_findings = [
        finding for finding in findings
        if finding.code != FindingCode.PROMPT_INJECTION or policy.block_on_injection
    ]
    if not blocking_findings:
        return SecurityDecision.ALLOW

    max_severity = _max_finding_severity(blocking_findings)
    return decision_for_severity(max_severity, policy.severity)


def _validation_error_result(
    claim_id: str,
    policy: SecurityPolicy,
    findings: list[SecurityFinding] | None = None,
    reason_codes: list[FindingCode] | None = None,
    blocked_fields: list[str] | None = None,
) -> SecurityGateResult:
    """Produit un blocage minimal quand l'entrée brute ne passe pas Pydantic."""
    now = datetime.now(UTC)
    if findings is None:
        findings = [SecurityFinding(
            code=FindingCode.POLICY_VIOLATION,
            severity=SeverityLevel.HIGH,
            description="Entrée Security Gate invalide selon le schéma Pydantic",
            detection_source="pydantic_schema",
            affected_element="security_input",
            evidence="SCHEMA_VALIDATION_ERROR",
        )]
    reason_codes = reason_codes or [findings[0].code]
    blocked_fields = blocked_fields or ["security_input"]
    audit_entry = SecurityAuditEntry(
        claim_id=claim_id,
        actor="security_gate_agent",
        input_type=None,
        outcome=SecurityDecision.BLOCK.value,
        decision=SecurityDecision.BLOCK,
        evaluated_at=now,
        policy_applied=policy.name,
        policy_version=policy.version,
        reason_codes=reason_codes,
        file_sha256=None,
    )
    return SecurityGateResult(
        claim_id=claim_id,
        decision=SecurityDecision.BLOCK,
        findings=findings,
        reason_codes=reason_codes,
        applied_policy=policy.name,
        policy_version=policy.version,
        evaluated_at=now,
        confidence_score=1.0,
        evidence_summary=_evidence_summary(findings, SecurityDecision.BLOCK),
        next_allowed_action=_NEXT_ACTIONS[SecurityDecision.BLOCK],
        audit_entry=audit_entry,
        prompt_injection_detected=False,
        blocked_fields=blocked_fields,
        reasons=[finding.description for finding in findings],
    )


# ── Phase B : décision LLM ────────────────────────────────────────────────────


def _invoke_llm_security(
    *,
    case_id: str,
    input_type: str,
    findings: list[dict],
    deterministic_decision: str,
    max_severity: str,
    file_ok: bool,
    path_ok: bool,
    url_ok: bool | None,
    text_injection: bool,
    text_level: str,
    deterministic_flag: bool | None,
) -> LlmSecurityDecision | None:
    """Envoie les résultats déterministes au LLM pour la décision ALLOW/BLOCK/QUARANTINE."""
    try:
        prompt = load_security_gate_prompt()
        llm = get_llm()
        structured = llm.with_structured_output(LlmSecurityDecision, method="json_schema")
        data = {
            "case_id": case_id,
            "prompt_version": prompt.version,
            "input_type": input_type,
            "findings": findings[:20],
            "deterministic_decision": deterministic_decision,
            "max_severity": max_severity,
            "file_ok": file_ok,
            "path_ok": path_ok,
            "url_ok": url_ok,
            "text_injection": text_injection,
            "text_level": text_level,
            "deterministic_flag": deterministic_flag,
        }
        system = SystemMessage(content=prompt.system_prompt)
        human = HumanMessage(content=json.dumps(data, ensure_ascii=False, default=str))
        result = structured.invoke([system, human])
        if isinstance(result, LlmSecurityDecision):
            return result
        if isinstance(result, dict):
            return LlmSecurityDecision(**result)
        return None
    except Exception:
        return None


# ── Fonction principale (testable sans LangGraph) ─────────────────────────────


def run(
    gate_input: SecurityGateInput,
    policy: SecurityPolicy | None = None,
) -> SecurityGateResult:
    """Exécute le pipeline de sécurité pour un élément du dossier.

    Args:
        gate_input: Données structurées de l'entrée à évaluer, validées par Pydantic.
        policy:     Politique de sécurité ; DEFAULT_POLICY si None.

    Returns:
        SecurityGateResult avec decision ALLOW, BLOCK ou QUARANTINE,
        findings structurés, audit_entry et horodatage.
    """
    pol = policy or DEFAULT_POLICY
    findings: list[SecurityFinding] = []
    reason_codes: list[FindingCode] = []
    blocked_fields: list[str] = []

    # ── Étape 1 : fichier et métadonnées ─────────────────────────────────────
    if gate_input.input_type == InputType.FILE:
        file_policy_codes: list[str] = []
        filename = gate_input.filename
        actual_size = gate_input.actual_size
        detected_mime = gate_input.detected_mime
        if not filename or actual_size is None or not detected_mime:
            file_policy_codes.append("FILE_METADATA_INCOMPLETE")
        else:
            _, file_policy_codes = validate_file_policy(
                filename=filename,
                detected_mime=detected_mime,
                size_bytes=actual_size,
                policy=pol.file,
            )
        _append_policy_findings(
            findings,
            reason_codes,
            blocked_fields,
            file_policy_codes,
            source="file_policy",
            affected_element="file_metadata",
        )

    # ── Étape 2 : chemin relatif de stockage ─────────────────────────────────
    if gate_input.relative_path:
        _, path_policy_codes = validate_storage_path(gate_input.relative_path, pol.path)
        _append_policy_findings(
            findings,
            reason_codes,
            blocked_fields,
            path_policy_codes,
            source="path_policy",
            affected_element="relative_path",
        )

    # ── Étape 3 : URL explicite ──────────────────────────────────────────────
    if gate_input.url:
        _, url_policy_codes = validate_url_policy(gate_input.url, pol.url)
        _append_policy_findings(
            findings,
            reason_codes,
            blocked_fields,
            url_policy_codes,
            source="url_policy",
            affected_element="url",
        )

    # ── Étape 4 : texte, métadonnées et sorties d'agents ─────────────────────
    text_source = gate_input.text_source
    if text_source is None and gate_input.input_type == InputType.AGENT_OUTPUT:
        text_source = "agent_output"
    _scan_text_field(
        findings,
        reason_codes,
        blocked_fields,
        "filename",
        gate_input.filename,
        pol,
        source="metadata",
    )
    _scan_text_field(
        findings,
        reason_codes,
        blocked_fields,
        "url",
        gate_input.url,
        pol,
        source="url",
    )
    _scan_text_field(
        findings,
        reason_codes,
        blocked_fields,
        "text_excerpt",
        gate_input.text_excerpt,
        pol,
        source=text_source or "text_excerpt",
    )

    # ── Étape 5 : outils demandés ────────────────────────────────────────────
    if gate_input.input_type == InputType.TOOL:
        requested_tool = gate_input.entry_id
        requesting_agent = gate_input.requesting_agent or ""
        _, tool_policy_codes = validate_tool_policy(
            tool_name=requested_tool,
            requesting_agent=requesting_agent,
            write_path=gate_input.relative_path,
            policy=pol.tool,
            path_policy=pol.path,
        )
        _append_policy_findings(
            findings,
            reason_codes,
            blocked_fields,
            tool_policy_codes,
            source="tool_policy",
            affected_element="tool_request",
        )
        _scan_text_field(
            findings,
            reason_codes,
            blocked_fields,
            "tool_arguments",
            gate_input.text_excerpt,
            pol,
            source="tool_arguments",
        )

    # ── Étape 6 : flag déterministe oracle ───────────────────────────────────
    injection_by_flag = gate_input.deterministic_injection_flag is True

    if injection_by_flag:
        code = FindingCode.PROMPT_INJECTION
        _add_reason_code(reason_codes, code)
        if "oracle_flag" not in blocked_fields:
            blocked_fields.append("oracle_flag")
        findings.append(SecurityFinding(
            code=code,
            severity=SeverityLevel.CRITICAL,
            description="Injection détectée par règle déterministe (oracle)",
            detection_source="deterministic_rule",
            affected_element="oracle_flag",
        ))

    injection_detected = any(f.code == FindingCode.PROMPT_INJECTION for f in findings)

    # ── Étape 7 : sévérité déterministe puis décision LLM finale ─────────────
    deterministic_decision = _decide(findings, pol)
    deterministic_reasons = (
        [f.description for f in findings]
        if findings
        else ["Aucune menace détectée — dossier autorisé"]
    )
    deterministic_confidence = _deterministic_confidence(findings, deterministic_decision)

    findings_payload = [finding.model_dump(mode="json") for finding in findings]
    max_severity = _max_finding_severity(findings)
    url_ok = None
    if gate_input.url:
        url_ok = not any(f.affected_element == "url" for f in findings)
    injection_findings = [f for f in findings if f.code == FindingCode.PROMPT_INJECTION]
    text_level = (
        max((f.severity for f in injection_findings), key=severity_rank).value
        if injection_findings
        else "NONE"
    )

    llm_decision = _invoke_llm_security(
        case_id=gate_input.claim_id,
        input_type=gate_input.input_type.value,
        findings=findings_payload,
        deterministic_decision=deterministic_decision.value,
        max_severity=max_severity.value,
        file_ok=not any(f.affected_element == "file_metadata" for f in findings),
        path_ok=not any(f.affected_element == "relative_path" for f in findings),
        url_ok=url_ok,
        text_injection=injection_detected,
        text_level=text_level,
        deterministic_flag=gate_input.deterministic_injection_flag,
    )

    if llm_decision is None:
        decision = SecurityDecision.BLOCK
        confidence_score = max(0.6, deterministic_confidence)
        llm_evidence = None
        reasons = ["LLM indisponible — décision conservatrice BLOCK."]
        if deterministic_reasons:
            reasons.extend(deterministic_reasons[:5])
    else:
        llm_evidence = getattr(llm_decision, "evidence", None)
        try:
            requested_decision = SecurityDecision(getattr(llm_decision, "decision"))
        except (AttributeError, TypeError, ValueError):
            decision = SecurityDecision.BLOCK
            confidence_score = max(0.6, deterministic_confidence)
            reasons = ["Décision LLM invalide — décision conservatrice BLOCK."]
            reasons.extend(deterministic_reasons[:5])
        else:
            llm_confidence = max(
                0.0,
                min(
                    1.0,
                    float(getattr(llm_decision, "confidence_score", deterministic_confidence)),
                ),
            )
            if _DECISION_RANK[requested_decision] < _DECISION_RANK[deterministic_decision]:
                decision = deterministic_decision
                confidence_score = max(llm_confidence, deterministic_confidence)
                reasons = [
                    "Décision LLM abaissée refusée par la politique de sécurité.",
                    *deterministic_reasons[:5],
                ]
            else:
                decision = requested_decision
                confidence_score = llm_confidence
                reasons = list(getattr(llm_decision, "reasons", None) or deterministic_reasons)
                explanation = getattr(llm_decision, "explanation", "")
                if explanation and explanation not in reasons:
                    reasons = [*reasons, explanation]

    # ── Étape 8 : audit et résultat ──────────────────────────────────────────
    now = datetime.now(UTC)
    audit_entry = SecurityAuditEntry(
        claim_id=gate_input.claim_id,
        actor=gate_input.requesting_agent or "security_gate_agent",
        input_type=gate_input.input_type,
        outcome=decision.value,
        decision=decision,
        evaluated_at=now,
        policy_applied=pol.name,
        policy_version=pol.version,
        reason_codes=reason_codes,
        file_sha256=gate_input.sha256,
    )

    return SecurityGateResult(
        claim_id=gate_input.claim_id,
        decision=decision,
        findings=findings,
        reason_codes=reason_codes,
        applied_policy=pol.name,
        policy_version=pol.version,
        evaluated_at=now,
        confidence_score=confidence_score,
        evidence_summary=_evidence_summary(findings, decision, llm_evidence),
        next_allowed_action=_NEXT_ACTIONS[decision],
        audit_entry=audit_entry,
        prompt_injection_detected=injection_detected,
        blocked_fields=blocked_fields,
        reasons=reasons,
        llm_metadata=build_llm_metadata(_AGENT_NAME, confidence_score),
    )


# ── Nœud LangGraph ────────────────────────────────────────────────────────────


def node(state: ClaimState) -> dict:
    """Nœud LangGraph — construit SecurityGateInput depuis le state et délègue à run().

    Attend dans le state :
        case_id         : identifiant du dossier
        security_input  : dict compatible avec SecurityGateInput
                          (entry_id, input_type requis ; autres champs optionnels)
    """
    case_id: str = state.get("case_id", "")  # type: ignore[assignment]
    raw: dict = state.get("security_input", {}) or {}  # type: ignore[assignment]
    policy = DEFAULT_POLICY

    try:
        gate_input = SecurityGateInput(
            claim_id=case_id,
            entry_id=raw.get("entry_id", raw.get("tool_name", f"{case_id}-eval")),
            input_type=InputType(raw.get("input_type", InputType.TEXT.value)),
            filename=raw.get("filename"),
            extension=raw.get("extension"),
            detected_mime=raw.get("detected_mime"),
            actual_size=raw.get("actual_size"),
            sha256=raw.get("sha256"),
            relative_path=raw.get("relative_path", raw.get("write_path")),
            url=raw.get("url"),
            text_excerpt=raw.get("text_excerpt", raw.get("tool_arguments_excerpt")),
            text_source=raw.get("text_source"),
            requesting_agent=raw.get("requesting_agent"),
            deterministic_injection_flag=raw.get("deterministic_injection_flag"),
        )
    except (ValueError, ValidationError):
        findings: list[SecurityFinding] = []
        reason_codes: list[FindingCode] = []
        blocked_fields: list[str] = []
        raw_path = raw.get("relative_path", raw.get("write_path"))
        if raw_path:
            _, path_policy_codes = validate_storage_path(str(raw_path), policy.path)
            _append_policy_findings(
                findings,
                reason_codes,
                blocked_fields,
                path_policy_codes,
                source="path_policy",
                affected_element="relative_path",
            )
        result = _validation_error_result(
            case_id,
            policy,
            findings=findings or None,
            reason_codes=reason_codes or None,
            blocked_fields=blocked_fields or None,
        )
    else:
        result = run(gate_input, policy)

    updates: dict = {
        "security_result": result,
        "security_input": None,
        "current_step": "security_gate",
        "completed_steps": ["security_gate"],
    }

    if result.decision != SecurityDecision.ALLOW:
        updates["errors"] = [f"[security_gate] {r}" for r in result.reasons]
        updates["alerts"] = [
            f"Sécurité : {result.decision.value} — "
            f"{len(result.blocked_fields)} champ(s) bloqué(s)"
        ]

    validate_state_update(updates)
    return updates
