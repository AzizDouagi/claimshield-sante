"""Claim Intake Agent — réception, validation et stockage documentaire.

Agent purement déterministe — aucun appel LLM.

Pipeline :
  1. Validation des métadonnées du dossier (non-vide, quota de fichiers)
  2. Pour chaque fichier, dans l'ordre :
       a. Écriture dans la zone temporaire  (StorageService.stage_file)
       b. Vérification du quota dossier destination
       c. Inspection complète : nom · extension · taille · MIME · SHA-256
       d. Déplacement atomique  incoming/ ou quarantine/
  3. Vérification des documents obligatoires
  4. Construction du ClaimManifest
  5. Retour ClaimIntakeResult

Interdictions strictes :
  - Aucune analyse médicale ou clinique.
  - Aucun OCR (réservé à document_ocr_agent).
  - Aucune décision de remboursement.
  - Aucun contenu brut (PDF, image) dans le ClaimState.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from config.settings import Settings, get_settings
from schemas.domain import IntakeStatus
from schemas.results import (
    ClaimIntakeResult,
    ClaimManifest,
    InspectedFile,
    StructuredError,
)
from services.storage import StorageError, StorageService
from state.claim_state import ClaimState
from tools.file_inspection import (
    build_storage_name,
    check_folder_limits,
    compute_folder_totals,
    inspect_file,
)


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
        ClaimIntakeResult avec statut accepted | quarantined | blocked.
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
            code="EMPTY_FOLDER",
            message="Le dossier ne contient aucun fichier",
            field="source_path",
        )
        return _blocked_result(case_id, received_at, depositor_id, [err], ["Dossier vide refusé"])

    if len(candidate_files) > s.max_files_per_folder:
        err = StructuredError(
            code="TOO_MANY_FILES",
            message=(
                f"Le dossier contient {len(candidate_files)} fichiers, "
                f"limite configurée : {s.max_files_per_folder}"
            ),
            field="source_path",
        )
        return _blocked_result(case_id, received_at, depositor_id, [err], [err.message])

    # ── Étape 2 : traitement fichier par fichier ─────────────────────────────

    inspected_files: list[InspectedFile] = []
    folder_file_count, folder_bytes = compute_folder_totals(svc.incoming_dir / case_id)

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
                status=IntakeStatus.BLOCKED,
                block_reasons=[exc.structured.message],
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
            err = StructuredError(
                code="FOLDER_QUOTA_EXCEEDED",
                message="; ".join(quota_reasons),
                field=fpath.name,
            )
            errors.append(err)
            inspected_files.append(InspectedFile(
                original_name=fpath.name,
                storage_name=build_storage_name(fpath.name, case_id, index),
                normalized_extension="",
                detected_mime_type="",
                actual_size=file_size,
                sha256=None,
                status=IntakeStatus.BLOCKED,
                block_reasons=quota_reasons,
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

        # d. Déplacement atomique vers incoming/ ou quarantine/ ────────────────
        try:
            dest = svc.commit_file(
                temp_path=temp_path,
                case_id=case_id,
                physical_name=inspected.storage_name,
                status=inspected.status,
            )
        except StorageError as exc:
            # Commit échoué (ex. NO_OVERWRITE) : temp déjà supprimé par commit_file
            errors.append(exc.structured)
            inspected = inspected.model_copy(update={
                "status": IntakeStatus.BLOCKED,
                "block_reasons": inspected.block_reasons + [exc.structured.message],
                "relative_storage_path": None,
            })
            inspected_files.append(inspected)
            continue

        # e. Mise à jour du chemin relatif réel ───────────────────────────────
        if dest is not None:
            zone = "incoming" if inspected.status == IntakeStatus.ACCEPTED else "quarantine"
            inspected = inspected.model_copy(update={
                "relative_storage_path": (
                    f"storage/{zone}/{case_id}/{inspected.storage_name}"
                ),
            })
            if inspected.status == IntakeStatus.ACCEPTED:
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

    accepted = [f for f in inspected_files if f.status == IntakeStatus.ACCEPTED]
    quarantined = [f for f in inspected_files if f.status == IntakeStatus.QUARANTINED]
    blocked = [f for f in inspected_files if f.status == IntakeStatus.BLOCKED]

    if blocked:
        global_status = IntakeStatus.BLOCKED
        reasons.append(
            f"{len(blocked)} fichier(s) bloqué(s) — dossier non traitable en l'état"
        )
    elif quarantined or alerts:
        global_status = IntakeStatus.QUARANTINED
        if quarantined:
            reasons.append(
                f"{len(quarantined)} fichier(s) en quarantaine — revue humaine requise"
            )
    else:
        global_status = IntakeStatus.ACCEPTED
        reasons.append(
            f"{len(accepted)} fichier(s) accepté(s) et stockés — dossier prêt pour traitement"
        )

    # ── Étape 5 : manifest et résultat ──────────────────────────────────────

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

    return ClaimIntakeResult(
        claim_id=case_id,
        status=global_status,
        manifest=manifest,
        accepted_count=len(accepted),
        quarantined_count=len(quarantined),
        reasons=reasons,
        errors=errors,
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
        "intake_result": result,
        "current_step": "claim_intake",
        "completed_steps": ["claim_intake"],
    }

    if result.status != IntakeStatus.ACCEPTED:
        updates["errors"] = [f"[claim_intake] {r}" for r in result.reasons]

    if result.manifest.alerts:
        updates["alerts"] = result.manifest.alerts

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
