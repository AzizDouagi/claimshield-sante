"""Intake & Safety Agent (V2) — fusion de `claim_intake_agent` + `security_gate_agent` (V1).

Un seul agent, une seule Phase A déterministe (inspection fichier + scans de
sécurité réunis dans la même boucle), un seul appel LLM (plan de refonte V2,
Phase V2-2).

Pipeline :
  Phase A — I/O déterministe (staging/inspection/storage, `tools.file_inspection`,
            réutilisé tel quel) + scans de sécurité déterministes
            (`security.policies`/`security.scanners`, réutilisés tels quels)
            sur chaque fichier de la même boucle.
  Phase B — LLM unique (`with_structured_output`) : statut final + motifs.
  Phase C — construction `schemas.v2_results.IntakeSafetyResult`.

Garantie de sécurité non négociable (plan V2 §7) : le LLM ne peut jamais
adoucir un statut déterministe plus restrictif que le sien — voir
`_merge_status`. `TECHNICAL_FAILURE` (panne de stockage) ne passe jamais
par le LLM à la baisse non plus, même logique.

Interdictions strictes (héritées de V1) :
  - Aucune analyse médicale ou clinique.
  - Aucun OCR (réservé à `document_understanding_agent`).
  - Aucune décision de remboursement.
  - Aucun contenu brut (PDF, image) dans `ClaimStateV2`.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from agents.intake_safety_agent.prompt import load_intake_safety_prompt
from agents.intake_safety_agent.schemas import LlmIntakeSafetyDecision
from config.settings import Settings, get_settings
from llm.factory import get_llm
from llm.metadata import build_llm_metadata
from schemas.domain import (
    FileStatus,
    FindingCode,
    IntakeReasonCode,
    IntakeSafetyStatus,
    IntakeStatus,
    SeverityLevel,
)
from schemas.results import ClaimManifest, InspectedFile, SecurityFinding, StructuredError
from schemas.v2_results import IntakeSafetyResult
from security.policies import (
    DEFAULT_POLICY,
    POLICY_EXECUTABLE_OR_SCRIPT,
    POLICY_EXTENSION_FORBIDDEN,
    POLICY_FILE_EMPTY,
    POLICY_FILE_TOO_LARGE,
    POLICY_MIME_EXTENSION_MISMATCH,
    POLICY_MIME_FORBIDDEN,
    POLICY_SUSPICIOUS_DOUBLE_EXTENSION,
    validate_file_policy,
    validate_storage_path,
)
from security.scanners import SEVERITY_CRITICAL, SEVERITY_HIGH, SEVERITY_LOW, SEVERITY_MEDIUM, scan_text_security
from services.storage import StorageError, StorageService
from state.claim_state_v2 import ClaimStateV2, validate_state_update_v2
from tools.file_inspection import build_storage_name, check_folder_limits, compute_folder_totals, inspect_file

_AGENT_NAME = "intake_safety_agent"

# Ordre de restriction croissante — un statut ne peut jamais être rendu
# "meilleur" par une étape suivante (Phase B LLM, agrégation multi-fichiers).
_STATUS_RANK: dict[IntakeSafetyStatus, int] = {
    IntakeSafetyStatus.ACCEPTED: 0,
    IntakeSafetyStatus.QUARANTINED: 1,
    IntakeSafetyStatus.BLOCKED: 2,
    IntakeSafetyStatus.TECHNICAL_FAILURE: 3,
}

_FILE_STATUS_TO_SAFETY_STATUS: dict[FileStatus, IntakeSafetyStatus] = {
    FileStatus.ACCEPTED: IntakeSafetyStatus.ACCEPTED,
    FileStatus.QUARANTINED: IntakeSafetyStatus.QUARANTINED,
    FileStatus.DUPLICATE: IntakeSafetyStatus.QUARANTINED,
    FileStatus.BLOCKED: IntakeSafetyStatus.BLOCKED,
    FileStatus.ERROR: IntakeSafetyStatus.TECHNICAL_FAILURE,
}

# Défaillances de stockage techniques (I/O) plutôt qu'une violation de politique.
_TECHNICAL_STORAGE_ERRORS = frozenset({"WRITE_ERROR", "MOVE_ERROR", "TEMP_COLLISION"})

# Sous-ensemble de security.policies utile à l'admission de fichiers (le reste
# — URL, outils — ne s'applique pas à ce stade, aucun document n'est encore lu).
_POLICY_CODE_TO_FINDING: dict[str, tuple[FindingCode, SeverityLevel]] = {
    POLICY_FILE_EMPTY: (FindingCode.EMPTY_FILE, SeverityLevel.HIGH),
    POLICY_FILE_TOO_LARGE: (FindingCode.FILE_TOO_LARGE, SeverityLevel.HIGH),
    POLICY_EXTENSION_FORBIDDEN: (FindingCode.UNSUPPORTED_EXTENSION, SeverityLevel.HIGH),
    POLICY_MIME_FORBIDDEN: (FindingCode.UNSUPPORTED_MIME, SeverityLevel.HIGH),
    POLICY_MIME_EXTENSION_MISMATCH: (FindingCode.MIME_EXTENSION_MISMATCH, SeverityLevel.MEDIUM),
    POLICY_EXECUTABLE_OR_SCRIPT: (FindingCode.UNSUPPORTED_EXTENSION, SeverityLevel.CRITICAL),
    POLICY_SUSPICIOUS_DOUBLE_EXTENSION: (FindingCode.UNSUPPORTED_EXTENSION, SeverityLevel.CRITICAL),
}

_SCANNER_SEVERITY: dict[str, SeverityLevel] = {
    SEVERITY_CRITICAL: SeverityLevel.CRITICAL,
    SEVERITY_HIGH: SeverityLevel.HIGH,
    SEVERITY_MEDIUM: SeverityLevel.MEDIUM,
    SEVERITY_LOW: SeverityLevel.LOW,
}

_SEVERITY_RANK: dict[SeverityLevel, int] = {
    SeverityLevel.INFO: 0,
    SeverityLevel.LOW: 1,
    SeverityLevel.MEDIUM: 2,
    SeverityLevel.HIGH: 3,
    SeverityLevel.CRITICAL: 4,
}


def _severity_to_status_rank(severity: SeverityLevel) -> int:
    """CRITICAL/HIGH → BLOCKED, MEDIUM → QUARANTINED, LOW/INFO → aucune escalade."""
    if _SEVERITY_RANK[severity] >= _SEVERITY_RANK[SeverityLevel.HIGH]:
        return _STATUS_RANK[IntakeSafetyStatus.BLOCKED]
    if severity is SeverityLevel.MEDIUM:
        return _STATUS_RANK[IntakeSafetyStatus.QUARANTINED]
    return _STATUS_RANK[IntakeSafetyStatus.ACCEPTED]


def _scan_file_security(
    *, original_name: str, detected_mime: str, actual_size: int
) -> tuple[list[SecurityFinding], int]:
    """Scans de sécurité complémentaires à `tools.file_inspection.inspect_file`
    (double extension, exécutable/script déguisé, injection dans le nom de
    fichier) — retourne les findings et le rang d'escalade le plus élevé
    rencontré (0 = aucune escalade)."""
    findings: list[SecurityFinding] = []
    max_rank = 0

    _, policy_codes = validate_file_policy(
        filename=original_name,
        detected_mime=detected_mime,
        size_bytes=actual_size,
        policy=DEFAULT_POLICY.file,
    )
    for code in policy_codes:
        finding_code, severity = _POLICY_CODE_TO_FINDING.get(
            code, (FindingCode.POLICY_VIOLATION, SeverityLevel.MEDIUM)
        )
        findings.append(
            SecurityFinding(
                code=finding_code,
                severity=severity,
                description=f"Fichier '{original_name}' refusé par la politique : {code}",
                detection_source="file_policy",
                affected_element="file_metadata",
                evidence=code,
            )
        )
        max_rank = max(max_rank, _severity_to_status_rank(severity))

    scan_result = scan_text_security(original_name, DEFAULT_POLICY, source="filename")
    if scan_result.detected:
        severity = _SCANNER_SEVERITY.get(scan_result.severity, SeverityLevel.MEDIUM)
        categories = ", ".join(sorted({f.category for f in scan_result.findings}))
        findings.append(
            SecurityFinding(
                code=FindingCode.PROMPT_INJECTION_DETECTED,
                severity=severity,
                description=f"Nom de fichier '{original_name}' suspect ({categories})",
                detection_source="text_security_scanner",
                affected_element="filename",
                evidence=(scan_result.triggers[0][:120] if scan_result.triggers else None),
            )
        )
        max_rank = max(max_rank, _severity_to_status_rank(severity))

    return findings, max_rank


# ── Phase B : LLM ─────────────────────────────────────────────────────────────


def _invoke_llm_intake_safety(
    *,
    case_id: str,
    deterministic_status: str,
    file_count: int,
    accepted_count: int,
    quarantined_count: int,
    duplicate_count: int,
    error_count: int,
    alerts: list[str],
    security_findings: list[dict],
    file_summaries: list[dict],
) -> LlmIntakeSafetyDecision | None:
    """Envoie le résumé déjà calculé au LLM pour statut final et motifs."""
    try:
        prompt = load_intake_safety_prompt()
        llm = get_llm()
        structured = llm.with_structured_output(LlmIntakeSafetyDecision, method="json_schema")
        data = {
            "case_id": case_id,
            "prompt_version": prompt.version,
            "deterministic_status": deterministic_status,
            "file_count": file_count,
            "accepted_count": accepted_count,
            "quarantined_count": quarantined_count,
            "duplicate_count": duplicate_count,
            "error_count": error_count,
            "alerts": alerts,
            "security_findings": security_findings[:20],
            "file_summaries": file_summaries,
        }
        system = SystemMessage(content=prompt.system_prompt)
        human = HumanMessage(content=json.dumps(data, ensure_ascii=False))
        result = structured.invoke([system, human])
        if isinstance(result, LlmIntakeSafetyDecision):
            return result
        if isinstance(result, dict):
            return LlmIntakeSafetyDecision(**result)
        return None
    except Exception:
        return None


def _merge_status(
    deterministic_status: IntakeSafetyStatus, llm_decision: LlmIntakeSafetyDecision | None
) -> tuple[IntakeSafetyStatus, list[str]]:
    """Combine le statut déterministe et la proposition du LLM — ne peut
    jamais aboutir à un statut moins restrictif que `deterministic_status`.

    LLM indisponible ou réponse invalide → fail-closed, statut au moins
    BLOCKED (jamais silencieusement ACCEPTED/QUARANTINED faute de LLM).
    """
    deterministic_rank = _STATUS_RANK[deterministic_status]

    if llm_decision is None:
        final_rank = max(deterministic_rank, _STATUS_RANK[IntakeSafetyStatus.BLOCKED])
        reasons = ["LLM indisponible — décision conservatrice BLOCKED."]
    else:
        try:
            llm_status = IntakeSafetyStatus(llm_decision.status)
        except ValueError:
            final_rank = max(deterministic_rank, _STATUS_RANK[IntakeSafetyStatus.BLOCKED])
            reasons = ["Décision LLM invalide — décision conservatrice BLOCKED."]
        else:
            final_rank = max(deterministic_rank, _STATUS_RANK[llm_status])
            reasons = list(llm_decision.reasons) if llm_decision.reasons else []
            if llm_decision.explanation and llm_decision.explanation not in reasons:
                reasons.append(llm_decision.explanation)

    final_status = next(status for status, rank in _STATUS_RANK.items() if rank == final_rank)
    return final_status, reasons


# ── Fonction principale (testable sans LangGraph) ─────────────────────────────


def run(
    case_id: str,
    source_path: Path,
    required_documents: list[str] | None = None,
    depositor_id: str | None = None,
    storage: StorageService | None = None,
    settings: Settings | None = None,
) -> IntakeSafetyResult:
    """Exécute l'admission (ingestion + sécurité fusionnées) d'un dossier.

    Args:
        case_id: identifiant du dossier (ex. CLM-0004).
        source_path: répertoire contenant les fichiers déposés.
        required_documents: noms de fichiers dont la présence est obligatoire.
        depositor_id: identité ou ID du déposant (optionnel, pour le manifest).
        storage: instance StorageService (injectée pour les tests).
        settings: configuration (injectée pour les tests).

    Returns:
        IntakeSafetyResult — statut ACCEPTED | QUARANTINED | BLOCKED | TECHNICAL_FAILURE.
    """
    s = settings or get_settings()
    svc = storage or StorageService(settings=s)
    svc.ensure_dirs()

    required = list(required_documents or [])
    received_at = datetime.now(UTC)
    reasons: list[str] = []
    errors: list[StructuredError] = []
    security_findings: list[SecurityFinding] = []

    # ── Étape 1 : dossier absent/vide — court-circuit sans appel LLM ────────
    # Décision non ambiguë (même rationale que V1 claim_intake_agent) : un
    # appel LLM n'y apporte aucune valeur, seulement une requête réseau
    # inutile vers Ollama.
    if not source_path.exists() or not source_path.is_dir():
        err = StructuredError(
            code=IntakeReasonCode.EMPTY_CLAIM,
            message="Le répertoire source du dossier de demande est introuvable",
            field="source_path",
        )
        return _finalize_without_llm(
            case_id=case_id,
            manifest=_blocked_manifest(case_id, received_at, depositor_id, [err]),
            service=svc,
            reasons=["Dossier absent refusé"],
            errors=[err],
        )

    candidate_files = sorted(
        [f for f in source_path.iterdir() if f.is_file()], key=lambda f: f.name
    )

    if not candidate_files:
        err = StructuredError(
            code=IntakeReasonCode.EMPTY_CLAIM,
            message="Le dossier de demande ne contient aucun fichier",
            field="source_path",
        )
        return _finalize_without_llm(
            case_id=case_id,
            manifest=_blocked_manifest(case_id, received_at, depositor_id, [err]),
            service=svc,
            reasons=["Dossier vide refusé"],
            errors=[err],
        )

    if len(candidate_files) > s.max_files_per_folder:
        err = StructuredError(
            code=IntakeReasonCode.TOO_MANY_FILES,
            message=(
                f"Le dossier contient {len(candidate_files)} fichiers, "
                f"limite configurée : {s.max_files_per_folder}"
            ),
            field="source_path",
        )
        return _finalize_with_llm(
            case_id=case_id,
            manifest=_blocked_manifest(case_id, received_at, depositor_id, [err]),
            service=svc,
            accepted_count=0,
            quarantined_count=0,
            duplicate_count=0,
            error_count=0,
            reasons=[err.message],
            errors=[err],
            security_findings=[],
            deterministic_status=IntakeSafetyStatus.BLOCKED,
        )

    # ── Étape 2 : traitement fichier par fichier — inspection + sécurité ────

    inspected_files: list[InspectedFile] = []
    folder_file_count, folder_bytes = compute_folder_totals(svc.incoming_dir / case_id)
    seen_sha256: dict[str, str] = {}

    for index, fpath in enumerate(candidate_files):
        try:
            temp_path, _phys_name = svc.stage_file(case_id=case_id, original_name=fpath.name, source=fpath)
        except StorageError as exc:
            errors.append(exc.structured)
            inspected_files.append(
                InspectedFile(
                    original_name=fpath.name,
                    storage_name=build_storage_name(fpath.name, case_id, index),
                    normalized_extension="",
                    detected_mime_type="",
                    actual_size=0,
                    sha256=None,
                    status=FileStatus.ERROR,
                    reasons=[
                        StructuredError(
                            code=IntakeReasonCode.STORAGE_ERROR,
                            message=exc.structured.message,
                            field=fpath.name,
                        )
                    ],
                    relative_storage_path=None,
                )
            )
            continue

        file_size = temp_path.stat().st_size
        quota_ok, quota_reasons = check_folder_limits(
            current_file_count=folder_file_count,
            current_total_bytes=folder_bytes,
            incoming_size_bytes=file_size,
            max_total_bytes=s.max_folder_size_bytes,
            max_file_count=s.max_files_per_folder,
        )
        if not quota_ok:
            temp_path.unlink(missing_ok=True)
            errors.extend(quota_reasons)
            inspected_files.append(
                InspectedFile(
                    original_name=fpath.name,
                    storage_name=build_storage_name(fpath.name, case_id, index),
                    normalized_extension="",
                    detected_mime_type="",
                    actual_size=file_size,
                    sha256=None,
                    status=FileStatus.BLOCKED,
                    reasons=quota_reasons,
                    relative_storage_path=None,
                )
            )
            continue

        inspected = inspect_file(
            path=temp_path,
            original_name=fpath.name,
            claim_id=case_id,
            index=index,
            allowed_extensions=s.allowed_extensions,
            allowed_mime_types=s.allowed_mime_types,
            max_file_size_bytes=s.max_file_size_bytes,
        )

        # ── Scans de sécurité complémentaires (double extension, exécutable
        # déguisé, injection dans le nom) — n'escalade jamais un fichier déjà
        # BLOCKED/ERROR vers un statut moins restrictif.
        if inspected.status not in (FileStatus.BLOCKED, FileStatus.ERROR):
            file_findings, escalation_rank = _scan_file_security(
                original_name=fpath.name,
                detected_mime=inspected.detected_mime_type,
                actual_size=inspected.actual_size,
            )
            security_findings.extend(file_findings)
            current_rank = _STATUS_RANK[_FILE_STATUS_TO_SAFETY_STATUS[inspected.status]]
            if escalation_rank > current_rank:
                escalated_status = (
                    FileStatus.BLOCKED
                    if escalation_rank >= _STATUS_RANK[IntakeSafetyStatus.BLOCKED]
                    else FileStatus.QUARANTINED
                )
                inspected = inspected.model_copy(
                    update={
                        "status": escalated_status,
                        "reasons": inspected.reasons
                        + [
                            StructuredError(
                                code="SECURITY_POLICY_VIOLATION",
                                message=f"Escaladé par contrôle de sécurité complémentaire ({len(file_findings)} anomalie(s))",
                                field=fpath.name,
                            )
                        ],
                    }
                )

        # ── Détection de doublons par SHA-256 ────────────────────────────────
        if inspected.sha256 and inspected.status == FileStatus.ACCEPTED:
            if inspected.sha256 in seen_sha256:
                first_name = seen_sha256[inspected.sha256]
                dup_reason = StructuredError(
                    code=IntakeReasonCode.DUPLICATE_FILE,
                    message=(
                        f"Octets identiques (SHA-256 : {inspected.sha256}) "
                        f"au fichier déjà accepté '{first_name}' dans ce dossier."
                    ),
                    field=fpath.name,
                )
                inspected = inspected.model_copy(
                    update={"status": FileStatus.DUPLICATE, "reasons": [dup_reason]}
                )
            else:
                seen_sha256[inspected.sha256] = fpath.name

        try:
            dest = svc.commit_file(
                temp_path=temp_path,
                case_id=case_id,
                physical_name=inspected.storage_name,
                status=inspected.status,
            )
        except StorageError as exc:
            errors.append(exc.structured)
            commit_file_status = (
                FileStatus.ERROR
                if exc.structured.code in _TECHNICAL_STORAGE_ERRORS
                else FileStatus.BLOCKED
            )
            inspected = inspected.model_copy(
                update={
                    "status": commit_file_status,
                    "reasons": inspected.reasons
                    + [
                        StructuredError(
                            code=IntakeReasonCode.STORAGE_ERROR,
                            message=exc.structured.message,
                            field=fpath.name,
                        )
                    ],
                    "relative_storage_path": None,
                }
            )
            inspected_files.append(inspected)
            continue

        if dest is not None:
            zone = "incoming" if inspected.status == FileStatus.ACCEPTED else "quarantine"
            inspected = inspected.model_copy(
                update={"relative_storage_path": f"{zone}/{case_id}/{inspected.storage_name}"}
            )
            # Défense en profondeur : le chemin de stockage réel ne doit jamais
            # sortir de la racine storage (réutilise security.policies).
            _, path_codes = validate_storage_path(inspected.relative_storage_path, DEFAULT_POLICY.path)
            if path_codes:
                security_findings.append(
                    SecurityFinding(
                        code=FindingCode.PATH_OUTSIDE_STORAGE,
                        severity=SeverityLevel.CRITICAL,
                        description=f"Chemin de stockage anormal pour '{fpath.name}'",
                        detection_source="path_policy",
                        affected_element="relative_storage_path",
                        evidence=",".join(path_codes),
                    )
                )
                inspected = inspected.model_copy(update={"status": FileStatus.BLOCKED})
            if inspected.status == FileStatus.ACCEPTED:
                folder_file_count += 1
                folder_bytes += file_size

        inspected_files.append(inspected)

    # ── Étape 3 : documents obligatoires ────────────────────────────────────

    alerts: list[str] = []
    if required:
        present_originals = {f.original_name for f in inspected_files}
        for req in required:
            if req not in present_originals:
                msg = f"Document obligatoire manquant : {req}"
                alerts.append(msg)
                reasons.append(msg)

    # ── Étape 4 : statut global déterministe ─────────────────────────────────

    accepted = [f for f in inspected_files if f.status == FileStatus.ACCEPTED]
    quarantined = [f for f in inspected_files if f.status == FileStatus.QUARANTINED]
    blocked = [f for f in inspected_files if f.status == FileStatus.BLOCKED]
    duplicate = [f for f in inspected_files if f.status == FileStatus.DUPLICATE]
    errored = [f for f in inspected_files if f.status == FileStatus.ERROR]

    if errored or errors:
        global_status = IntakeSafetyStatus.TECHNICAL_FAILURE
        reasons.append(f"{len(errored)} fichier(s) en erreur technique — stockage inaccessible")
    elif blocked:
        global_status = IntakeSafetyStatus.BLOCKED
        reasons.append(f"{len(blocked)} fichier(s) bloqué(s) — dossier non traitable en l'état")
    elif quarantined or duplicate or alerts:
        global_status = IntakeSafetyStatus.QUARANTINED
        if quarantined:
            reasons.append(f"{len(quarantined)} fichier(s) en quarantaine — revue requise")
        if duplicate:
            reasons.append(f"{len(duplicate)} fichier(s) en doublon — vérification requise")
    else:
        global_status = IntakeSafetyStatus.ACCEPTED
        reasons.append(f"{len(accepted)} fichier(s) accepté(s) et stockés — dossier prêt")

    manifest = ClaimManifest(
        claim_id=case_id,
        received_at=received_at,
        depositor_id=depositor_id,
        file_count=len(inspected_files),
        total_size_bytes=sum(f.actual_size for f in inspected_files),
        files=inspected_files,
        status=_to_intake_status(global_status),
        alerts=alerts,
    )

    return _finalize_with_llm(
        case_id=case_id,
        manifest=manifest,
        service=svc,
        accepted_count=len(accepted),
        quarantined_count=len(quarantined),
        duplicate_count=len(duplicate),
        error_count=len(errored),
        reasons=reasons,
        errors=errors,
        security_findings=security_findings,
        deterministic_status=global_status,
    )


# ── Nœud LangGraph ────────────────────────────────────────────────────────────


def node(state: ClaimStateV2) -> dict:
    """Nœud du graphe V2 — délègue à `run()` et met à jour `ClaimStateV2`.

    Attend dans le state :
        case_id       : identifiant du dossier
        intake_input  : dict { source_path, required_documents?, depositor_id? }
    """
    case_id: str = state.get("case_id", "")  # type: ignore[assignment]
    intake_input: dict = state.get("intake_input") or {}
    source_path = Path(intake_input.get("source_path", ""))
    required_documents: list[str] = intake_input.get("required_documents", [])
    depositor_id: str | None = intake_input.get("depositor_id")

    result = run(
        case_id=case_id,
        source_path=source_path,
        required_documents=required_documents,
        depositor_id=depositor_id,
    )

    updates: dict = {
        "intake_safety_result": result,
        "intake_input": None,
        "current_step": "intake_safety",
        "completed_steps": ["intake_safety"],
    }

    if result.status != IntakeSafetyStatus.ACCEPTED:
        updates["errors"] = [f"[intake_safety] {r}" for r in result.reasons]

    if result.manifest is not None and result.manifest.alerts:
        updates["alerts"] = result.manifest.alerts

    validate_state_update_v2(updates)
    return updates


# ── Helpers internes ──────────────────────────────────────────────────────────


def _to_intake_status(status: IntakeSafetyStatus) -> IntakeStatus:
    """Mappe IntakeSafetyStatus (V2) vers IntakeStatus (V1, réutilisé tel
    quel par `ClaimManifest.status`, seul champ V1 encore requis ici)."""
    if status is IntakeSafetyStatus.TECHNICAL_FAILURE:
        return IntakeStatus.ERROR
    return IntakeStatus(status.value.lower())


def _blocked_manifest(
    case_id: str, received_at: datetime, depositor_id: str | None, errors: list[StructuredError]
) -> ClaimManifest:
    return ClaimManifest(
        claim_id=case_id,
        received_at=received_at,
        depositor_id=depositor_id,
        file_count=0,
        total_size_bytes=0,
        files=[],
        status=IntakeStatus.BLOCKED,
        alerts=[e.message for e in errors],
    )


def _finalize_without_llm(
    *,
    case_id: str,
    manifest: ClaimManifest,
    service: StorageService,
    reasons: list[str],
    errors: list[StructuredError],
) -> IntakeSafetyResult:
    """Construit le résultat final sans jamais invoquer le LLM — réservé au
    cas EMPTY_CLAIM (dossier sans fichier candidat), décision non ambiguë."""
    llm_trace = build_llm_metadata(_AGENT_NAME)
    service.write_intake_manifest(case_id, manifest.model_dump_json(indent=2))
    return IntakeSafetyResult(
        case_id=case_id,
        status=IntakeSafetyStatus.BLOCKED,
        manifest=manifest,
        security_findings=[],
        reasons=reasons,
        errors=errors,
        llm_trace=llm_trace,
    )


def _finalize_with_llm(
    *,
    case_id: str,
    manifest: ClaimManifest,
    service: StorageService,
    accepted_count: int,
    quarantined_count: int,
    duplicate_count: int,
    error_count: int,
    reasons: list[str],
    errors: list[StructuredError],
    security_findings: list[SecurityFinding],
    deterministic_status: IntakeSafetyStatus,
) -> IntakeSafetyResult:
    """Appelle le LLM (Phase B) puis construit le résultat final.

    `deterministic_status` reflète la conclusion réelle de la Phase A —
    toujours fourni explicitement par l'appelant, jamais redéduit d'un
    champ V1 (`manifest.status`) dont l'enum ne couvre pas exactement les
    4 valeurs V2 (notamment `TECHNICAL_FAILURE`).
    """
    det_status = deterministic_status
    llm_trace = build_llm_metadata(_AGENT_NAME)
    file_summaries = [
        {
            "nom": f.original_name,
            "statut": f.status.value,
            "mime": f.detected_mime_type,
            "taille": f.actual_size,
        }
        for f in manifest.files
    ]
    llm_decision = _invoke_llm_intake_safety(
        case_id=case_id,
        deterministic_status=det_status.value,
        file_count=manifest.file_count,
        accepted_count=accepted_count,
        quarantined_count=quarantined_count,
        duplicate_count=duplicate_count,
        error_count=error_count,
        alerts=manifest.alerts,
        security_findings=[f.model_dump(mode="json") for f in security_findings],
        file_summaries=file_summaries,
    )

    final_status, llm_reasons = _merge_status(det_status, llm_decision)
    final_reasons = llm_reasons if llm_reasons else list(reasons)
    final_errors = list(errors)

    if final_status is IntakeSafetyStatus.BLOCKED and det_status is not IntakeSafetyStatus.BLOCKED:
        # Escaladé par la Phase B (LLM indisponible/invalide, ou choix explicite) —
        # motif structuré ajouté pour ne jamais laisser un BLOCKED sans errors[].
        final_errors.append(
            StructuredError(
                code="LLM_ESCALATED_TO_BLOCKED",
                message="Statut porté à BLOCKED lors de la fusion Phase A/Phase B.",
                field="llm_output",
            )
        )

    if manifest.status != _to_intake_status(final_status):
        manifest = manifest.model_copy(update={"status": _to_intake_status(final_status)})
    service.write_intake_manifest(case_id, manifest.model_dump_json(indent=2))

    return IntakeSafetyResult(
        case_id=case_id,
        status=final_status,
        manifest=manifest,
        security_findings=security_findings,
        reasons=final_reasons or reasons or ["Aucun motif disponible."],
        errors=final_errors,
        llm_trace=llm_trace,
    )
