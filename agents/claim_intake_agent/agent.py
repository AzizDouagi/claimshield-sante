"""Claim Intake Agent — réception, validation et stockage documentaire.

Agent LLM (gemma4:latest via ChatOllama) + pipeline d'ingestion déterministe.

Pipeline :
  Phase A — I/O déterministe : staging, inspection, storage, manifest.
  Phase B — LLM (with_structured_output) : statut final + motifs enrichis.
  Phase C — construction ClaimIntakeResult final.

Interdictions strictes :
  - Aucune analyse médicale ou clinique.
  - Aucun OCR (réservé à document_ocr_agent).
  - Aucune décision de remboursement.
  - Aucun contenu brut (PDF, image) dans le ClaimState.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from agents.claim_intake_agent.prompt import load_claim_intake_prompt
from agents.claim_intake_agent.schemas import LlmIntakeDecision
from config.settings import Settings, get_settings
from langchain_core.messages import HumanMessage, SystemMessage
from llm.factory import get_llm
from llm.metadata import build_llm_metadata
from schemas.domain import FileStatus, IntakeReasonCode, IntakeStatus
from schemas.results import (
    ClaimIntakeResult,
    ClaimManifest,
    InspectedFile,
    StructuredError,
)
from services.storage import StorageError, StorageService
from state.claim_state import ClaimState, validate_state_update
from tools.file_inspection import (
    build_storage_name,
    check_folder_limits,
    compute_folder_totals,
    inspect_file,
)

_AGENT_NAME = "claim_intake_agent"

# Codes de StorageError qui signalent une défaillance technique (I/O) plutôt
# qu'une violation de politique.  NO_OVERWRITE est une règle métier → BLOCKED.
_TECHNICAL_STORAGE_ERRORS = frozenset({"WRITE_ERROR", "MOVE_ERROR", "TEMP_COLLISION"})


# ── Phase B : LLM ─────────────────────────────────────────────────────────────


def _invoke_llm_intake(
    *,
    case_id: str,
    global_status: str,
    file_count: int,
    accepted_count: int,
    quarantined_count: int,
    duplicate_count: int,
    error_count: int,
    alerts: list[str],
    file_summaries: list[dict],
) -> LlmIntakeDecision | None:
    """Envoie le résumé de l'ingestion au LLM pour statut et motifs."""
    try:
        prompt = load_claim_intake_prompt()
        llm = get_llm()
        structured = llm.with_structured_output(LlmIntakeDecision, method="json_schema")
        data = {
            "case_id": case_id,
            "prompt_version": prompt.version,
            "global_status": global_status,
            "file_count": file_count,
            "accepted_count": accepted_count,
            "quarantined_count": quarantined_count,
            "duplicate_count": duplicate_count,
            "error_count": error_count,
            "alerts": alerts,
            "file_summaries": file_summaries,
        }
        system = SystemMessage(content=prompt.system_prompt)
        human = HumanMessage(content=json.dumps(data, ensure_ascii=False))
        result = structured.invoke([system, human])
        if isinstance(result, LlmIntakeDecision):
            return result
        if isinstance(result, dict):
            return LlmIntakeDecision(**result)
        return None
    except Exception:
        return None


# ── Fonction principale (testable sans LangGraph) ─────────────────────────────


def run(
    case_id: str,
    source_path: Path,
    required_documents: list[str] | None = None,
    depositor_id: str | None = None,
    storage: StorageService | None = None,
    settings: Settings | None = None,
) -> ClaimIntakeResult:
    """Exécute le pipeline d'ingestion pour un dossier de demande de remboursement.

    Args:
        case_id: Identifiant du dossier (ex. CLM-0004).
        source_path: Répertoire contenant les fichiers déposés.
        required_documents: Noms de fichiers dont la présence est obligatoire.
        depositor_id: Identité ou ID du déposant (optionnel, pour le manifest).
        storage: Instance StorageService (injectée pour les tests ; sinon créée ici).
        settings: Configuration (injectée pour les tests ; sinon chargée via get_settings).

    Returns:
        ClaimIntakeResult avec statut accepted | quarantined | blocked | error.
    """
    s = settings or get_settings()
    svc = storage or StorageService(settings=s)
    svc.ensure_dirs()

    required = list(required_documents or [])
    received_at = datetime.now(UTC)
    reasons: list[str] = []
    errors: list[StructuredError] = []

    # ── Étape 1 : validation des métadonnées du dossier ─────────────────────

    candidate_files = sorted(
        [f for f in source_path.iterdir() if f.is_file()],
        key=lambda f: f.name,
    )

    if not candidate_files:
        err = StructuredError(
            code=IntakeReasonCode.EMPTY_CLAIM,
            message="Le dossier de demande ne contient aucun fichier",
            field="source_path",
        )
        result = _blocked_result(case_id, received_at, depositor_id, [err], ["Dossier vide refusé"])
        svc.write_intake_manifest(case_id, result.manifest.model_dump_json(indent=2))
        return result

    if len(candidate_files) > s.max_files_per_folder:
        err = StructuredError(
            code=IntakeReasonCode.TOO_MANY_FILES,
            message=(
                f"Le dossier contient {len(candidate_files)} fichiers, "
                f"limite configurée : {s.max_files_per_folder}"
            ),
            field="source_path",
        )
        result = _blocked_result(case_id, received_at, depositor_id, [err], [err.message])
        svc.write_intake_manifest(case_id, result.manifest.model_dump_json(indent=2))
        return result

    # ── Étape 2 : traitement fichier par fichier ─────────────────────────────

    inspected_files: list[InspectedFile] = []
    folder_file_count, folder_bytes = compute_folder_totals(svc.incoming_dir / case_id)
    # SHA-256 → nom original du premier fichier portant ce hash (déduplication)
    seen_sha256: dict[str, str] = {}

    for index, fpath in enumerate(candidate_files):

        # a. Écriture dans la zone temporaire ─────────────────────────────────
        try:
            temp_path, phys_name = svc.stage_file(
                case_id=case_id,
                original_name=fpath.name,
                source=fpath,
            )
        except StorageError as exc:
            errors.append(exc.structured)
            inspected_files.append(InspectedFile(
                original_name=fpath.name,
                storage_name=build_storage_name(fpath.name, case_id, index),
                normalized_extension="",
                detected_mime_type="",
                actual_size=0,
                sha256=None,
                status=FileStatus.ERROR,
                reasons=[StructuredError(
                    code=IntakeReasonCode.STORAGE_ERROR,
                    message=exc.structured.message,
                    field=fpath.name,
                )],
                relative_storage_path=None,
            ))
            continue

        # b. Quota dossier destination ─────────────────────────────────────────
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
            inspected_files.append(InspectedFile(
                original_name=fpath.name,
                storage_name=build_storage_name(fpath.name, case_id, index),
                normalized_extension="",
                detected_mime_type="",
                actual_size=file_size,
                sha256=None,
                status=FileStatus.BLOCKED,
                reasons=quota_reasons,
                relative_storage_path=None,
            ))
            continue

        # c. Inspection complète (nom · extension · taille · MIME · SHA-256) ──
        inspected = inspect_file(
            path=temp_path,
            original_name=fpath.name,
            claim_id=case_id,
            index=index,
            allowed_extensions=s.allowed_extensions,
            allowed_mime_types=s.allowed_mime_types,
            max_file_size_bytes=s.max_file_size_bytes,
        )

        # d. Détection de doublons par SHA-256 ────────────────────────────────
        # Ne concerne que les fichiers qui ont passé l'inspection (ACCEPTED).
        # Un doublon = même sha256 qu'un fichier déjà enregistré dans ce dossier.
        # On ne conclut PAS à une fraude ici : c'est le rôle de fraud_detection_agent.
        if inspected.sha256 and inspected.status == FileStatus.ACCEPTED:
            if inspected.sha256 in seen_sha256:
                first_name = seen_sha256[inspected.sha256]
                # Le message conserve explicitement la référence au hash identique
                # et le nom du premier fichier accepté — pas de décision de fraude.
                dup_reason = StructuredError(
                    code=IntakeReasonCode.DUPLICATE_FILE,
                    message=(
                        f"Octets identiques (SHA-256 : {inspected.sha256}) "
                        f"au fichier déjà accepté '{first_name}' dans ce dossier. "
                        f"Le premier fichier est conservé ; celui-ci est mis en quarantaine."
                    ),
                    field=fpath.name,
                )
                inspected = inspected.model_copy(update={
                    "status": FileStatus.DUPLICATE,
                    "reasons": [dup_reason],
                    # sha256 intentionnellement conservé : sert de preuve d'identité
                })
            else:
                seen_sha256[inspected.sha256] = fpath.name

        # e. Déplacement atomique vers incoming/ ou quarantine/ ────────────────
        try:
            dest = svc.commit_file(
                temp_path=temp_path,
                case_id=case_id,
                physical_name=inspected.storage_name,
                status=inspected.status,
            )
        except StorageError as exc:
            errors.append(exc.structured)
            # Défaillance I/O réelle (WRITE_ERROR, MOVE_ERROR…) → ERROR
            # Violation de politique (ex. NO_OVERWRITE) → BLOCKED
            commit_file_status = (
                FileStatus.ERROR
                if exc.structured.code in _TECHNICAL_STORAGE_ERRORS
                else FileStatus.BLOCKED
            )
            inspected = inspected.model_copy(update={
                "status": commit_file_status,
                "reasons": inspected.reasons + [StructuredError(
                    code=IntakeReasonCode.STORAGE_ERROR,
                    message=exc.structured.message,
                    field=fpath.name,
                )],
                "relative_storage_path": None,
            })
            inspected_files.append(inspected)
            continue

        # f. Mise à jour du chemin relatif réel (relatif à la racine du storage) ─
        if dest is not None:
            zone = "incoming" if inspected.status == FileStatus.ACCEPTED else "quarantine"
            inspected = inspected.model_copy(update={
                "relative_storage_path": f"{zone}/{case_id}/{inspected.storage_name}",
            })
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

    # ── Étape 4 : statut global ──────────────────────────────────────────────
    # Priorité : ERROR > BLOCKED > QUARANTINED/DUPLICATE > ACCEPTED

    accepted = [f for f in inspected_files if f.status == FileStatus.ACCEPTED]
    quarantined = [f for f in inspected_files if f.status == FileStatus.QUARANTINED]
    blocked = [f for f in inspected_files if f.status == FileStatus.BLOCKED]
    duplicate = [f for f in inspected_files if f.status == FileStatus.DUPLICATE]
    errored = [f for f in inspected_files if f.status == FileStatus.ERROR]

    if errored or errors:
        global_status = IntakeStatus.ERROR
        reasons.append(
            f"{len(errored)} fichier(s) en erreur technique — stockage inaccessible"
        )
    elif blocked:
        global_status = IntakeStatus.BLOCKED
        reasons.append(
            f"{len(blocked)} fichier(s) bloqué(s) — dossier non traitable en l'état"
        )
    elif quarantined or duplicate or alerts:
        global_status = IntakeStatus.QUARANTINED
        if quarantined:
            reasons.append(
                f"{len(quarantined)} fichier(s) en quarantaine — revue humaine requise"
            )
        if duplicate:
            reasons.append(
                f"{len(duplicate)} fichier(s) en doublon — vérification requise"
            )
    else:
        global_status = IntakeStatus.ACCEPTED
        reasons.append(
            f"{len(accepted)} fichier(s) accepté(s) et stockés — dossier prêt pour traitement"
        )

    # ── Étape 5 : manifest, persistance et résultat ──────────────────────────

    manifest = ClaimManifest(
        claim_id=case_id,
        received_at=received_at,
        depositor_id=depositor_id,
        file_count=len(inspected_files),
        total_size_bytes=sum(f.actual_size for f in inspected_files),
        files=inspected_files,
        status=global_status,
        alerts=alerts,
    )

    # ── Phase B : LLM — statut et motifs enrichis ────────────────────────────
    file_summaries = [
        {
            "nom": f.original_name,
            "statut": f.status.value,
            "mime": f.detected_mime_type,
            "taille": f.actual_size,
        }
        for f in inspected_files
    ]
    llm_decision = _invoke_llm_intake(
        case_id=case_id,
        global_status=global_status.value,
        file_count=len(inspected_files),
        accepted_count=len(accepted),
        quarantined_count=len(quarantined),
        duplicate_count=len(duplicate),
        error_count=len(errored),
        alerts=alerts,
        file_summaries=file_summaries,
    )

    if llm_decision is None:
        llm_error = StructuredError(
            code=IntakeReasonCode.LLM_OUTPUT_INVALID,
            message=(
                "Sortie LLM invalide ou indisponible : "
                "décision d'ingestion impossible"
            ),
            field="llm_output",
        )
        global_status = IntakeStatus.ERROR
        reasons = ["LLM indisponible : décision d'ingestion impossible."]
        errors.append(llm_error)
    else:
        llm_status_map = {
            "ACCEPTED": IntakeStatus.ACCEPTED,
            "QUARANTINED": IntakeStatus.QUARANTINED,
            "BLOCKED": IntakeStatus.BLOCKED,
            "ERROR": IntakeStatus.ERROR,
        }
        global_status = llm_status_map.get(llm_decision.status, global_status)
        if llm_decision.reasons:
            reasons = llm_decision.reasons

    # ── Phase C : persistance et résultat ────────────────────────────────────
    if manifest.status != global_status:
        manifest = manifest.model_copy(update={"status": global_status})
    svc.write_intake_manifest(case_id, manifest.model_dump_json(indent=2))

    return ClaimIntakeResult(
        claim_id=case_id,
        status=global_status,
        manifest=manifest,
        accepted_count=len(accepted),
        quarantined_count=len(quarantined),
        duplicate_count=len(duplicate),
        error_count=len(errored),
        reasons=reasons,
        errors=errors,
        llm_metadata=build_llm_metadata(_AGENT_NAME),
    )


# ── Nœud LangGraph ────────────────────────────────────────────────────────────


def node(state: ClaimState) -> dict:
    """Nœud LangGraph — délègue à run() et met à jour le state.

    Attend dans le state :
        case_id       : identifiant du dossier
        intake_input  : dict { source_path, required_documents?, depositor_id? }
    """
    case_id: str = state.get("case_id", "")  # type: ignore[assignment]
    intake_input: dict = state.get("intake_input", {})  # type: ignore[assignment]
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
        # ── Ce que le state reçoit après ingestion ────────────────────────────
        # Contenu autorisé : identifiant, statut, manifest structuré,
        # métadonnées documents (hashes + chemins relatifs).
        # Contenu interdit : octets bruts, chemins absolus, objets fichiers.
        "intake_result": result,
        "intake_status": result.status,      # promu pour le routage du graphe
        "intake_input": None,                # vidé — source_path absolu supprimé
        "current_step": "claim_intake",
        "completed_steps": ["claim_intake"],
    }

    if result.status != IntakeStatus.ACCEPTED:
        updates["errors"] = [f"[claim_intake] {r}" for r in result.reasons]

    if result.manifest.alerts:
        updates["alerts"] = result.manifest.alerts

    # Garde-fou : lève ValueError si du contenu interdit est détecté
    validate_state_update(updates)
    return updates


# ── Helpers internes ──────────────────────────────────────────────────────────


def _blocked_result(
    case_id: str,
    received_at: datetime,
    depositor_id: str | None,
    errors: list[StructuredError],
    reasons: list[str],
) -> ClaimIntakeResult:
    """Construit un ClaimIntakeResult BLOCKED lors d'un échec de validation précoce."""
    manifest = ClaimManifest(
        claim_id=case_id,
        received_at=received_at,
        depositor_id=depositor_id,
        file_count=0,
        total_size_bytes=0,
        files=[],
        status=IntakeStatus.BLOCKED,
        alerts=[e.message for e in errors],
    )
    return ClaimIntakeResult(
        claim_id=case_id,
        status=IntakeStatus.BLOCKED,
        manifest=manifest,
        accepted_count=0,
        quarantined_count=0,
        reasons=reasons,
        errors=errors,
    )
