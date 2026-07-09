"""Écriture locale des fichiers uploadés via Chainlit avant soumission à
l'API — logique filesystem pure, séparée de ``ui/app.py`` pour être
testable sans runtime Chainlit. Accepte tout objet portant ``.name`` et
``.path`` ou ``.content`` (forme exacte de ``chainlit.File``, jamais importé
ici — voir ``Uploaded`` ci-dessous, un ``Protocol`` structurel).
"""
from __future__ import annotations

import shutil
from typing import Protocol

from config.settings import get_settings


class Uploaded(Protocol):
    name: str
    path: str | None
    content: bytes | str | None


def stage_uploaded_files(case_id: str, files: list[Uploaded]) -> str:
    """Écrit ``files`` sous ``{CLAIMSHIELD_UI_UPLOAD_DIR}/{case_id}/input/``
    et retourne ce répertoire — c'est le ``source_path`` à transmettre à
    ``POST /claims`` (voir ``ui/api_client.py::submit_claim``).

    Compatible Docker (voir ``docker-compose.yml``) : ``api`` et ``ui``
    montent le même volume ``./storage`` — un ``source_path`` écrit ici se
    résout donc à l'identique côté conteneur ``api``.
    """
    target_dir = get_settings().claimshield_ui_upload_dir / case_id / "input"
    target_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        dest = target_dir / f.name
        path = getattr(f, "path", None)
        content = getattr(f, "content", None)
        if path:
            shutil.copyfile(path, dest)
        elif content is not None:
            mode = "wb" if isinstance(content, bytes) else "w"
            with dest.open(mode) as out:
                out.write(content)
        else:
            raise ValueError(f"Fichier {f.name!r} sans contenu ni chemin exploitable.")

    return str(target_dir)
