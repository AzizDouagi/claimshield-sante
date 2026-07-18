"""Tests de `scripts/evaluate_recommendations_v2.py::_isolate_benchmark_storage`
— plan de remédiation « rejouabilité des dossiers » (phase 1).

Vérifie qu'une campagne de benchmark ne peut jamais écrire dans le stockage
réel (`storage/incoming`/`quarantine`/`manifests` partagé avec l'API en
production) — la racine de stockage du process est redirigée vers un
répertoire isolé avant tout appel à `get_settings()`/`compile_workflow_v2()`.
"""
from __future__ import annotations

import os

from config.settings import get_settings
from scripts.evaluate_recommendations_v2 import _isolate_benchmark_storage


def test_isolate_benchmark_storage_redirects_storage_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAIMSHIELD_STORAGE_DIR", raising=False)
    get_settings.cache_clear()

    benchmark_root = tmp_path / "_benchmark_runs" / "20260101T000000Z"
    _isolate_benchmark_storage(benchmark_root)

    try:
        settings = get_settings()
        assert settings.storage_dir == benchmark_root.resolve()
        assert benchmark_root.is_dir()
    finally:
        monkeypatch.delenv("CLAIMSHIELD_STORAGE_DIR", raising=False)
        get_settings.cache_clear()


def test_isolate_benchmark_storage_never_reuses_default_real_storage_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAIMSHIELD_STORAGE_DIR", raising=False)
    get_settings.cache_clear()
    default_storage_dir = get_settings().storage_dir

    benchmark_root = tmp_path / "_benchmark_runs" / "isolated"
    _isolate_benchmark_storage(benchmark_root)

    try:
        assert get_settings().storage_dir != default_storage_dir
        assert os.environ["CLAIMSHIELD_STORAGE_DIR"] == str(benchmark_root)
    finally:
        monkeypatch.delenv("CLAIMSHIELD_STORAGE_DIR", raising=False)
        get_settings.cache_clear()
