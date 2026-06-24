"""Inspection bas niveau des fichiers : validation, MIME, taille, hash SHA-256.

Fonctions déterministes utilisées par claim_intake_agent et security_gate_agent.
Aucun effet de bord : chaque fonction reçoit ses paramètres explicitement —
jamais d'accès global aux settings depuis ce module.

python-magic est tenté en premier ; mimetypes sert de repli si libmagic
n'est pas installé sur la machine.
"""
from __future__ import annotations

import hashlib
import mimetypes
import re
from pathlib import Path

from schemas.domain import FileStatus, IntakeReasonCode
from schemas.results import InspectedFile, StructuredError

# ── Détection MIME (avec repli sans dépendance système) ──────────────────────

try:
    import magic as _magic

    def _detect_mime_raw(path: Path) -> str:
        return _magic.from_file(str(path), mime=True)

except (ImportError, OSError):
    def _detect_mime_raw(path: Path) -> str:  # type: ignore[misc]
        mime, _ = mimetypes.guess_type(str(path))
        return mime or "application/octet-stream"


# Correspondance canonique extension → MIME attendu (pour contrôle de cohérence)
_EXT_TO_MIME: dict[str, str] = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "json": "application/json",
}

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[/\\]")
_BLOCK_SIZE = 65_536  # 64 Kio — limite de lecture par chunk pour SHA-256


# ── Validation du nom de fichier ──────────────────────────────────────────────


def validate_filename(name: str) -> tuple[bool, list[StructuredError]]:
    """Valide le nom de fichier fourni par le déposant.

    Retourne (valide, raisons_de_rejet).
    Aucun accès disque — validation purement sur la chaîne.
    """
    reasons: list[StructuredError] = []
    name = name.strip()

    if not name:
        reasons.append(StructuredError(
            code=IntakeReasonCode.INVALID_FILENAME,
            message="Nom de fichier vide après suppression des espaces",
            field="filename",
        ))
        return False, reasons

    if "\x00" in name:
        reasons.append(StructuredError(
            code=IntakeReasonCode.INVALID_FILENAME,
            message="Caractère nul détecté dans le nom de fichier",
            field="filename",
        ))
        return False, reasons

    if all(c == "." for c in name):
        reasons.append(StructuredError(
            code=IntakeReasonCode.INVALID_FILENAME,
            message="Nom composé uniquement de points refusé",
            field="filename",
        ))
        return False, reasons

    # Chemin absolu POSIX (/etc/passwd, /home/…)
    if name.startswith("/"):
        reasons.append(StructuredError(
            code=IntakeReasonCode.PATH_TRAVERSAL_ATTEMPT,
            message="Chemin absolu POSIX refusé",
            field="filename",
        ))
        return False, reasons

    # Chemin absolu Windows (C:\, D:/, …)
    if _WINDOWS_DRIVE_RE.match(name):
        reasons.append(StructuredError(
            code=IntakeReasonCode.PATH_TRAVERSAL_ATTEMPT,
            message="Chemin absolu Windows refusé",
            field="filename",
        ))
        return False, reasons

    # Chemin UNC (\\server\ ou //server/)
    if name.startswith("\\\\") or name.startswith("//"):
        reasons.append(StructuredError(
            code=IntakeReasonCode.PATH_TRAVERSAL_ATTEMPT,
            message="Chemin UNC refusé",
            field="filename",
        ))
        return False, reasons

    # Traversée de répertoire (../ ou ..\)
    parts = re.split(r"[/\\]", name)
    if ".." in parts:
        reasons.append(StructuredError(
            code=IntakeReasonCode.PATH_TRAVERSAL_ATTEMPT,
            message="Traversée de répertoire refusée (séquence '..' détectée)",
            field="filename",
        ))
        return False, reasons

    return True, reasons


# ── Nom de stockage sécurisé ──────────────────────────────────────────────────


def build_storage_name(original_name: str, claim_id: str, index: int) -> str:
    """Génère un nom de stockage déterministe à partir du claim_id et d'un index.

    Le nom original n'est jamais utilisé directement comme nom de stockage.
    Seule l'extension normalisée est conservée.
    """
    ext = Path(original_name).suffix.lower()
    return f"{claim_id}_doc{index:02d}{ext}"


# ── Inspection de l'extension ─────────────────────────────────────────────────


def inspect_extension(
    original_name: str,
    allowed_extensions: list[str],
) -> tuple[bool, str]:
    """Extrait l'extension avec Path.suffix, normalise en minuscules, compare à l'allowlist.

    Retourne (autorisée, extension_normalisée_sans_point).
    """
    ext = Path(original_name).suffix.lstrip(".").lower()
    return ext in allowed_extensions, ext


# ── Détection MIME ────────────────────────────────────────────────────────────


def detect_mime_type(path: Path) -> str:
    """Détecte le type MIME depuis le contenu réel du fichier (pas son nom)."""
    return _detect_mime_raw(path)


def check_mime_consistency(
    detected_mime: str,
    normalized_ext: str,
    allowed_mime_types: list[str],
) -> tuple[bool, list[StructuredError], bool]:
    """Vérifie que le MIME détecté est autorisé et cohérent avec l'extension déclarée.

    Retourne (cohérent, raisons, est_quarantaine).

    - MIME non autorisé          → BLOCKED
    - MIME autorisé mais discordant avec l'extension → QUARANTINED
    """
    reasons: list[StructuredError] = []

    if detected_mime not in allowed_mime_types:
        reasons.append(StructuredError(
            code=IntakeReasonCode.UNSUPPORTED_MIME_TYPE,
            message=f"MIME détecté '{detected_mime}' non présent dans la liste autorisée",
            field="mime_type",
        ))
        return False, reasons, False

    expected_mime = _EXT_TO_MIME.get(normalized_ext)
    if expected_mime is not None and detected_mime != expected_mime:
        reasons.append(StructuredError(
            code=IntakeReasonCode.MIME_EXTENSION_MISMATCH,
            message=(
                f"Incohérence MIME/extension : "
                f"'.{normalized_ext}' attend '{expected_mime}', "
                f"contenu réel détecté '{detected_mime}'"
            ),
            field="mime_type",
        ))
        return False, reasons, True  # quarantaine

    return True, reasons, False


# ── Vérification de la taille ─────────────────────────────────────────────────


def check_file_size(path: Path, max_size_bytes: int) -> tuple[bool, list[StructuredError], int]:
    """Vérifie la taille réelle du fichier lu depuis le disque.

    Ne tient aucun compte de la taille annoncée par le client.
    Retourne (valide, raisons, taille_réelle_octets).
    """
    reasons: list[StructuredError] = []
    real_size = path.stat().st_size

    if real_size == 0:
        reasons.append(StructuredError(
            code=IntakeReasonCode.EMPTY_FILE,
            message="Fichier vide (0 octet) refusé",
            field="size_bytes",
        ))
        return False, reasons, 0

    if real_size > max_size_bytes:
        limit_mb = max_size_bytes / (1024 * 1024)
        real_mb = real_size / (1024 * 1024)
        reasons.append(StructuredError(
            code=IntakeReasonCode.FILE_TOO_LARGE,
            message=(
                f"Fichier trop volumineux : {real_mb:.2f} Mo dépasse la limite de {limit_mb:.0f} Mo"
            ),
            field="size_bytes",
        ))
        return False, reasons, real_size

    return True, reasons, real_size


def compute_folder_totals(folder_path: Path) -> tuple[int, int]:
    """Retourne (nombre_fichiers, taille_totale_octets) pour le niveau direct du dossier.

    Ne descend pas dans les sous-dossiers.
    Retourne (0, 0) si le dossier n'existe pas encore.
    """
    if not folder_path.exists():
        return 0, 0
    files = [f for f in folder_path.iterdir() if f.is_file()]
    return len(files), sum(f.stat().st_size for f in files)


def check_folder_limits(
    current_file_count: int,
    current_total_bytes: int,
    incoming_size_bytes: int,
    max_total_bytes: int,
    max_file_count: int,
) -> tuple[bool, list[StructuredError]]:
    """Vérifie si l'ajout d'un fichier dépasserait les limites configurées du dossier.

    Retourne (autorisé, raisons).
    """
    reasons: list[StructuredError] = []

    if current_file_count + 1 > max_file_count:
        reasons.append(StructuredError(
            code=IntakeReasonCode.TOO_MANY_FILES,
            message=(
                f"Quota de fichiers dépassé : {current_file_count + 1} > {max_file_count} autorisés"
            ),
            field="file_count",
        ))

    projected = current_total_bytes + incoming_size_bytes
    if projected > max_total_bytes:
        limit_mb = max_total_bytes / (1024 * 1024)
        proj_mb = projected / (1024 * 1024)
        reasons.append(StructuredError(
            code=IntakeReasonCode.FOLDER_QUOTA_EXCEEDED,
            message=(
                f"Quota de taille du dossier dépassé : {proj_mb:.2f} Mo > {limit_mb:.0f} Mo autorisés"
            ),
            field="total_size_bytes",
        ))

    return not reasons, reasons


# ── Hash SHA-256 ──────────────────────────────────────────────────────────────


def compute_sha256(path: Path) -> str:
    """Hash SHA-256 en lecture par blocs de 64 Kio — jamais chargé en mémoire entière.

    Retourne toujours le même hexdigest pour le même contenu.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_BLOCK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Pipeline complet d'inspection ─────────────────────────────────────────────


def inspect_file(
    path: Path,
    original_name: str,
    claim_id: str,
    index: int,
    allowed_extensions: list[str],
    allowed_mime_types: list[str],
    max_file_size_bytes: int,
) -> InspectedFile:
    """Pipeline complet d'inspection pour un fichier unique.

    Enchaîne validation du nom, extension, taille, MIME et hash SHA-256.
    Retourne un InspectedFile avec statut ACCEPTED, QUARANTINED ou BLOCKED.

    Le nom original n'est jamais utilisé comme nom de stockage.
    Le hash n'est calculé que si le fichier passe les contrôles de taille.
    """
    all_reasons: list[StructuredError] = []
    is_quarantine_only = False

    # ── Étape 1 : validation du nom (sans accès disque) ──────────────────────
    name_ok, name_reasons = validate_filename(original_name)
    if not name_ok:
        return InspectedFile(
            original_name=original_name,
            storage_name=f"{claim_id}_doc{index:02d}_blocked",
            normalized_extension="",
            detected_mime_type="",
            actual_size=0,
            sha256=None,
            status=FileStatus.BLOCKED,
            reasons=name_reasons,
            relative_storage_path=None,
        )

    storage_name = build_storage_name(original_name, claim_id, index)
    ext_allowed, normalized_ext = inspect_extension(original_name, allowed_extensions)

    # ── Étape 2 : extension ───────────────────────────────────────────────────
    if not ext_allowed:
        ext_label = f".{normalized_ext}" if normalized_ext else "(aucune)"
        all_reasons.append(StructuredError(
            code=IntakeReasonCode.UNSUPPORTED_EXTENSION,
            message=f"Extension {ext_label} non autorisée",
            field="filename",
        ))

    # ── Étape 3 : taille réelle (premier accès disque) ───────────────────────
    size_ok, size_reasons, actual_size = check_file_size(path, max_file_size_bytes)
    if not size_ok:
        all_reasons.extend(size_reasons)

    # ── Étape 4 : hash SHA-256 (uniquement si taille valide) ─────────────────
    sha256 = compute_sha256(path) if size_ok else None

    # ── Étape 5 : détection MIME depuis le contenu ───────────────────────────
    detected_mime = detect_mime_type(path) if actual_size > 0 else "application/octet-stream"

    # ── Étape 6 : cohérence MIME / extension ─────────────────────────────────
    mime_ok, mime_reasons, is_quarantine_signal = check_mime_consistency(
        detected_mime, normalized_ext, allowed_mime_types
    )
    if not mime_ok:
        all_reasons.extend(mime_reasons)
        if is_quarantine_signal:
            is_quarantine_only = True

    # ── Étape 7 : statut final ────────────────────────────────────────────────
    if not all_reasons:
        status = FileStatus.ACCEPTED
    elif is_quarantine_only and ext_allowed and size_ok:
        # Le seul problème est une incohérence MIME/extension : quarantaine humaine
        status = FileStatus.QUARANTINED
    else:
        status = FileStatus.BLOCKED

    # Le chemin définitif est calculé par l'agent après commit_file() ;
    # on laisse None ici pour ne pas exposer un emplacement temporaire incorrect.
    relative_storage_path = None

    return InspectedFile(
        original_name=original_name,
        storage_name=storage_name,
        normalized_extension=normalized_ext,
        detected_mime_type=detected_mime,
        actual_size=actual_size,
        sha256=sha256,
        status=status,
        reasons=all_reasons,
        relative_storage_path=relative_storage_path,
    )


# ── Compatibilité — API historique ───────────────────────────────────────────


class DocumentInfo(dict):  # type: ignore[type-arg]
    """Dict typé retourné par inspect_document (ancienne API)."""


def inspect_document(path: Path) -> DocumentInfo:
    """Retourne filename, sha256, size_bytes, mime_type pour un fichier.

    Conservé pour la compatibilité avec les tests et scripts existants.
    Préférer inspect_file() pour les nouveaux développements.
    """
    return DocumentInfo(
        filename=path.name,
        sha256=compute_sha256(path),
        size_bytes=path.stat().st_size,
        mime_type=detect_mime_type(path),
    )
