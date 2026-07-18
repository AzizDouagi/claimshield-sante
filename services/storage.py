"""Service de stockage sécurisé pour ClaimShield Santé.

Architecture des zones sous storage/ :
  temporary/   — écriture initiale avant toute inspection
  incoming/    — fichiers validés en attente de traitement
  quarantine/  — fichiers suspects en attente de revue humaine

Toute résolution de chemin est vérifiée par Path.is_relative_to() avant toute
opération disque. Les chemins ne sont jamais construits par concaténation
de chaînes.
"""
from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path

from config.settings import Settings, get_settings
from schemas.domain import FileStatus
from schemas.results import StructuredError
from tools.file_inspection import compute_sha256

_UNSAFE_CHARS = re.compile(r"[^a-zA-Z0-9._-]")


class StorageError(Exception):
    """Erreur de stockage avec contexte structuré."""

    def __init__(self, structured: StructuredError) -> None:
        super().__init__(structured.message)
        self.structured = structured


class StorageService:
    def __init__(self, settings: Settings | None = None) -> None:
        self._s = settings or get_settings()
        self._root = self._s.storage_dir.resolve()

    # ── Zones ────────────────────────────────────────────────────────────────

    @property
    def temp_dir(self) -> Path:
        return (self._root / "temporary").resolve()

    @property
    def incoming_dir(self) -> Path:
        return (self._root / "incoming").resolve()

    @property
    def quarantine_dir(self) -> Path:
        return self._s.quarantine_dir.resolve()

    @property
    def manifests_dir(self) -> Path:
        return (self._root / "manifests").resolve()

    def ensure_dirs(self) -> None:
        """Crée les quatre zones si elles n'existent pas encore."""
        for d in (self.temp_dir, self.incoming_dir, self.quarantine_dir, self.manifests_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ── Manifest d'ingestion ─────────────────────────────────────────────────

    def write_intake_manifest(self, case_id: str, content: str) -> Path:
        """Écrit le manifest d'ingestion JSON dans manifests/<case_id>.json.

        Écrase silencieusement si le fichier existe (re-ingestion idempotente).
        Retourne le chemin absolu du fichier écrit.
        """
        dest = self._safe_resolve(self.manifests_dir, f"{case_id}.json")
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        return dest

    def read_intake_manifest(self, case_id: str) -> str:
        """Lit le manifest d'ingestion JSON pour un dossier donné.

        Lève FileNotFoundError si le manifest n'existe pas encore.
        """
        path = self._safe_resolve(self.manifests_dir, f"{case_id}.json")
        if not path.exists():
            raise FileNotFoundError(
                f"Manifest introuvable pour le dossier '{case_id}' : {path}"
            )
        return path.read_text(encoding="utf-8")

    # ── Résolution sécurisée ─────────────────────────────────────────────────

    def _safe_resolve(self, root: Path, *parts: str | Path) -> Path:
        """Résout un chemin et vérifie qu'il reste strictement sous root.

        Lève StorageError si le chemin résolu sort de la racine autorisée.
        Protège contre ../, les liens symboliques hors-zone et les noms
        absolus passés dans parts.
        """
        resolved_root = root.resolve()
        candidate = resolved_root.joinpath(*[str(p) for p in parts]).resolve()
        if not candidate.is_relative_to(resolved_root):
            raise StorageError(
                StructuredError(
                    code="PATH_TRAVERSAL",
                    message=(
                        f"Chemin refusé : '{candidate}' sort de "
                        f"la racine autorisée '{resolved_root}'"
                    ),
                )
            )
        return candidate

    # ── Chemins par dossier ──────────────────────────────────────────────────

    def incoming_path(self, case_id: str) -> Path:
        return self._safe_resolve(self.incoming_dir, case_id)

    def quarantine_path(self, case_id: str) -> Path:
        return self._safe_resolve(self.quarantine_dir, case_id)

    def temporary_path(self, case_id: str) -> Path:
        return self._safe_resolve(self.temp_dir, case_id)

    # ── Nom physique unique ──────────────────────────────────────────────────

    @staticmethod
    def physical_name(original_name: str) -> str:
        """Génère <uuid_hex>_<nom_assaini>.<ext_normalisée>.

        Le nom original n'est jamais utilisé comme chemin physique.
        Deux fichiers de même nom originel ne peuvent pas entrer en collision.
        """
        p = Path(original_name)
        safe_stem = _UNSAFE_CHARS.sub("_", p.stem)[:64].strip("_") or "file"
        ext = p.suffix.lower()
        return f"{uuid.uuid4().hex}_{safe_stem}{ext}"

    # ── Écriture atomique dans la zone temporaire ────────────────────────────

    def stage_file(
        self,
        case_id: str,
        original_name: str,
        source: Path | bytes,
    ) -> tuple[Path, str]:
        """Écrit le fichier dans temporary/<case_id>/ et le ferme complètement.

        Retourne (chemin_temporaire, nom_physique_unique).
        Le nom physique est un UUID — le nom original n'est jamais utilisé
        comme nom de stockage physique.
        Un échec d'écriture supprime le fichier partiel avant de lever StorageError.
        """
        phys = self.physical_name(original_name)
        case_temp = self._safe_resolve(self._root, "temporary", case_id)
        case_temp.mkdir(parents=True, exist_ok=True)
        dest = self._safe_resolve(case_temp, phys)

        if dest.exists():
            raise StorageError(
                StructuredError(
                    code="TEMP_COLLISION",
                    message=f"Collision inattendue sur le fichier temporaire '{phys}'",
                    field="physical_name",
                )
            )

        try:
            if isinstance(source, bytes):
                dest.write_bytes(source)
            else:
                shutil.copy2(source, dest)
        except OSError as exc:
            dest.unlink(missing_ok=True)
            raise StorageError(
                StructuredError(
                    code="WRITE_ERROR",
                    message=f"Échec d'écriture dans la zone temporaire : {exc}",
                    field=original_name,
                )
            ) from exc

        return dest, phys

    # ── Déplacement atomique final ───────────────────────────────────────────

    def commit_file(
        self,
        temp_path: Path,
        case_id: str,
        physical_name: str,
        status: FileStatus,
        expected_sha256: str | None = None,
    ) -> Path | None:
        """Déplace atomiquement temp_path vers la zone définitive selon le statut.

        ACCEPTED    → incoming/<case_id>/<physical_name>
        QUARANTINED → quarantine/<case_id>/<physical_name>
        DUPLICATE   → quarantine/<case_id>/<physical_name>  (revue humaine)
        BLOCKED     → temp_path supprimé, retourne None
        ERROR       → temp_path supprimé, retourne None

        Le déplacement est atomique (Path.rename → rename(2) POSIX sur même partition).
        En cas d'erreur IO, le fichier temporaire est supprimé avant de lever
        StorageError. Aucun fichier existant n'est écrasé.

        `expected_sha256` (optionnel, additif — plan de remédiation
        « rejouabilité des dossiers », phase 1) : si fourni et que la
        destination existe déjà avec exactement ce contenu (même hash), le
        commit est traité comme un **rejeu idempotent** — `temp_path` est
        supprimé (contenu déjà en place), `dest` est retourné normalement,
        aucune exception n'est levée. Si le hash diffère, ou si
        `expected_sha256` n'est pas fourni (comportement historique,
        inchangé — jamais appelé par V1), `NO_OVERWRITE` est levée comme
        auparavant. Ne distingue jamais elle-même une révision autorisée
        d'une substitution suspecte — cette sémantique appartient à
        l'appelant (`agents/intake_safety_agent/agent.py`), pas à ce service
        générique partagé V1/V2.
        """
        if status in (FileStatus.BLOCKED, FileStatus.ERROR):
            temp_path.unlink(missing_ok=True)
            return None

        is_accepted = status == FileStatus.ACCEPTED
        zone_label = "incoming" if is_accepted else "quarantine"
        zone_root = self.incoming_dir if is_accepted else self.quarantine_dir

        dest_dir = self._safe_resolve(zone_root, case_id)
        dest = self._safe_resolve(dest_dir, physical_name)

        if dest.exists():
            if expected_sha256 is not None and compute_sha256(dest) == expected_sha256:
                temp_path.unlink(missing_ok=True)
                return dest
            temp_path.unlink(missing_ok=True)
            raise StorageError(
                StructuredError(
                    code="NO_OVERWRITE",
                    message=(
                        f"'{physical_name}' existe déjà dans "
                        f"{zone_label}/{case_id}/ avec un contenu différent — "
                        "écrasement silencieux refusé"
                    ),
                    field=physical_name,
                )
            )

        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            temp_path.rename(dest)
        except OSError as exc:
            temp_path.unlink(missing_ok=True)
            raise StorageError(
                StructuredError(
                    code="MOVE_ERROR",
                    message=f"Déplacement atomique vers {zone_label}/ échoué : {exc}",
                    field=physical_name,
                )
            ) from exc

        return dest

    # ── Nettoyage ────────────────────────────────────────────────────────────

    def cleanup_temp_file(self, path: Path) -> None:
        """Supprime un fichier temporaire après vérification qu'il est sous temporary/."""
        if not path.resolve().is_relative_to(self.temp_dir):
            raise StorageError(
                StructuredError(
                    code="CLEANUP_SECURITY",
                    message=(
                        "Suppression refusée : "
                        f"'{path.resolve()}' ne se trouve pas dans temporary/"
                    ),
                )
            )
        path.unlink(missing_ok=True)

    def cleanup_temp_case(self, case_id: str) -> None:
        """Supprime le répertoire temporaire complet d'un dossier."""
        case_temp = self._safe_resolve(self._root, "temporary", case_id)
        if case_temp.exists():
            shutil.rmtree(case_temp)

    # ── Opérations dossier — compatibilité tests et import Synthea ───────────

    def stage_to_incoming(self, case_id: str, source_dir: Path) -> Path:
        """Copie source_dir → incoming/<case_id>/.

        Lève StorageError si la destination existe déjà (pas d'écrasement silencieux).
        """
        dest = self.incoming_path(case_id)
        if dest.exists():
            raise StorageError(
                StructuredError(
                    code="NO_OVERWRITE",
                    message=(
                        f"incoming/{case_id}/ existe déjà — "
                        "supprimez-le manuellement avant de ré-ingérer"
                    ),
                    field=case_id,
                )
            )
        shutil.copytree(source_dir, dest)
        return dest

    def move_to_quarantine(self, case_id: str) -> Path:
        """Déplace incoming/<case_id> → quarantine/<case_id>.

        Lève StorageError si la destination existe déjà.
        """
        src = self.incoming_path(case_id)
        dest = self.quarantine_path(case_id)
        if dest.exists():
            raise StorageError(
                StructuredError(
                    code="NO_OVERWRITE",
                    message=f"quarantine/{case_id}/ existe déjà",
                    field=case_id,
                )
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        return dest

    def move_to_temporary(self, case_id: str) -> Path:
        """Copie incoming/<case_id> → temporary/<case_id> (laisse l'original intact).

        Lève StorageError si la destination existe déjà.
        """
        src = self.incoming_path(case_id)
        dest = self.temporary_path(case_id)
        if dest.exists():
            raise StorageError(
                StructuredError(
                    code="NO_OVERWRITE",
                    message=f"temporary/{case_id}/ existe déjà",
                    field=case_id,
                )
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(src), str(dest))
        return dest

    def cleanup_temporary(self, case_id: str) -> None:
        """Supprime temporary/<case_id>/ après traitement."""
        self.cleanup_temp_case(case_id)
