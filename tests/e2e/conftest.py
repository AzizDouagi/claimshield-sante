"""Support partagé pour la suite ``tests/e2e/`` — pipeline complet piloté via
l'API (``TestClient``), LLM stubbé (``tests.conftest.deterministic_agent_llm``,
autouse) mais **aucun nœud du graphe n'est mocké** ici, contrairement à
``tests/api/test_main.py``. Ces tests exercent donc la vraie Phase A
déterministe des 11 agents (I/O fichiers, OCR réel via ``pytesseract``,
validation FHIR, etc.) sur de vraies fixtures de démo.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from api.main import create_app

FIXTURES_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "fixtures" / "valid"


def fixture_input_dir(case_id: str) -> str:
    """Chemin absolu vers le répertoire ``input/`` d'un dossier de fixture
    réel (ex. ``datasets/fixtures/valid/CLM-0001/input``)."""
    input_dir = FIXTURES_ROOT / case_id / "input"
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Fixture introuvable : {input_dir}")
    return str(input_dir)


@pytest.fixture
def real_app_client() -> TestClient:
    """Client API sur un graphe compilé complet, sans aucun nœud mocké — seul
    le LLM reste stubbé par l'autouse ``deterministic_agent_llm``."""
    app = create_app(checkpointer=InMemorySaver())
    return TestClient(app)
