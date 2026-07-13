"""Vérification statique — ``api/v2/*.py`` n'accède jamais directement à un
agent individuel (même garantie que ``tests/graph/test_architecture.py``
pour V1). Le seul point d'entrée métier autorisé est le graphe compilé
(``graph.workflow_v2.compile_workflow_v2``)."""
from __future__ import annotations

from pathlib import Path

import api.v2.chat as chat_module
import api.v2.claims as claims_module

_V2_API_MODULES = [claims_module, chat_module]


class TestNoDirectAgentAccess:
    def test_v2_api_modules_never_import_agents_directly(self):
        for module in _V2_API_MODULES:
            source = Path(module.__file__).read_text(encoding="utf-8")
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith("import agents.") or stripped.startswith("from agents."):
                    raise AssertionError(
                        f"{module.__name__} importe directement un agent : {stripped!r} — "
                        "l'accès métier doit passer exclusivement par graph.workflow_v2."
                    )

    def test_claims_module_only_reaches_agents_through_compiled_workflow(self):
        source = Path(claims_module.__file__).read_text(encoding="utf-8")
        assert "compile_workflow_v2" in source
        assert "graph.workflow_v2" in source
