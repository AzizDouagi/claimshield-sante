"""Tests de `ui/api_client_v2.py::stream_chat_message_v2` — visibilité temps
réel des étapes/tokens du chat (demandée par AZIZ, « comme Claude Code »).

Aucune dépendance supplémentaire (`respx` absent du projet) : simulation du
flux SSE via `httpx.MockTransport`, déjà natif à `httpx` (dépendance
existante), même esprit que `ui/forms.py`/`ui/uploads.py` — logique pure,
sans jamais lancer un vrai serveur ni importer `chainlit`."""
from __future__ import annotations

import httpx
import pytest

from ui import api_client_v2

pytestmark = pytest.mark.asyncio


def _sse_body(*lines: str) -> bytes:
    return ("\n\n".join(f"data: {line}" for line in lines) + "\n\n").encode("utf-8")


_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _patch_async_client(monkeypatch, handler) -> None:
    """Remplace `httpx.AsyncClient` par une factory qui construit le vrai
    client (capturé avant monkeypatch, jamais la version patchée
    elle-même — sinon récursion infinie) avec un `MockTransport`."""
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: _REAL_ASYNC_CLIENT(*args, **kwargs, transport=httpx.MockTransport(handler)),
    )


class TestStreamChatMessageV2:
    async def test_yields_one_dict_per_sse_line_in_order(self, monkeypatch):
        body = _sse_body(
            '{"type": "step", "step_name": "comprehension", "label": "Compréhension", '
            '"status": "STARTED", "model_name": null, "input_tokens": null, '
            '"output_tokens": null, "duration_ms": null, "detail": ""}',
            '{"type": "step", "step_name": "comprehension", "label": "Compréhension", '
            '"status": "COMPLETED", "model_name": "gemma4:latest", "input_tokens": 42, '
            '"output_tokens": 7, "duration_ms": 120, "detail": ""}',
            '{"type": "final", "reply": "Voici la réponse.", "thread_id": null, '
            '"total_input_tokens": 42, "total_output_tokens": 7, "total_duration_ms": 120, '
            '"step_count": 1}',
        )

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path.endswith("/chat/stream")
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

        _patch_async_client(monkeypatch, handler)

        events = [event async for event in api_client_v2.stream_chat_message_v2("bonjour", case_id="CLM-0001")]

        assert len(events) == 3
        assert events[0]["type"] == "step"
        assert events[0]["status"] == "STARTED"
        assert events[1]["type"] == "step"
        assert events[1]["status"] == "COMPLETED"
        assert events[1]["input_tokens"] == 42
        assert events[1]["output_tokens"] == 7
        assert events[2]["type"] == "final"
        assert events[2]["reply"] == "Voici la réponse."

    async def test_blank_lines_between_events_are_skipped(self, monkeypatch):
        body = b'data: {"type": "final", "reply": "ok", "thread_id": null, "total_input_tokens": 0, "total_output_tokens": 0, "total_duration_ms": 0, "step_count": 0}\n\n\n'

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

        _patch_async_client(monkeypatch, handler)

        events = [event async for event in api_client_v2.stream_chat_message_v2("bonjour")]
        assert len(events) == 1
        assert events[0]["reply"] == "ok"

    async def test_http_error_status_raises(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"detail": "Clé API invalide"})

        _patch_async_client(monkeypatch, handler)

        with pytest.raises(httpx.HTTPStatusError):
            async for _ in api_client_v2.stream_chat_message_v2("bonjour"):
                pass
