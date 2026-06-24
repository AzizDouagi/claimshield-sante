"""Configuration pytest partagée pour tous les tests ClaimShield Santé.

Nettoie le répertoire storage/ partagé (zones incoming/, quarantine/, temporary/)
avant chaque test afin d'éviter les collisions liées aux runs précédents.
Les tests qui nécessitent un stockage isolé injectent leur propre StorageService
via tmp_path — ce fixture ne les affecte pas.
"""
from __future__ import annotations

import shutil

import pytest

from config.settings import get_settings


@pytest.fixture(autouse=True)
def clean_shared_storage() -> None:
    """Supprime les sous-dossiers CLM-* dans les zones de stockage partagées."""
    s = get_settings()
    zones = [
        s.storage_dir / "incoming",
        s.storage_dir / "temporary",
        s.quarantine_dir,
    ]
    for zone in zones:
        if zone.exists():
            for clm_dir in zone.glob("CLM-*"):
                shutil.rmtree(clm_dir, ignore_errors=True)
    yield
