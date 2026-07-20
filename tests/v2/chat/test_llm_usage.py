"""Tests de `chat/llm_usage.py::record_usage` — helper unique de capture des
tokens, partagé par les 4 sites LLM de `chat/` (visibilité temps réel
demandée par AZIZ, « comme Claude Code »)."""
from __future__ import annotations

from types import SimpleNamespace

from chat.llm_usage import record_usage


def _message(*, usage=None, metadata=None):
    return SimpleNamespace(usage_metadata=usage, response_metadata=metadata or {})


class TestRecordUsage:
    def test_none_sink_is_a_no_op(self):
        # Ne doit jamais lever, quel que soit le contenu de `result`.
        record_usage(_message(usage={"input_tokens": 1, "output_tokens": 2}), None)

    def test_none_result_is_a_no_op(self):
        sink: dict = {}
        record_usage(None, sink)
        assert sink == {}

    def test_captures_tokens_and_model_name(self):
        sink: dict = {}
        result = _message(
            usage={"input_tokens": 42, "output_tokens": 7, "total_tokens": 49},
            metadata={"model_name": "gemma4:latest"},
        )
        record_usage(result, sink)
        assert sink["input_tokens"] == 42
        assert sink["output_tokens"] == 7
        assert sink["model_name"] == "gemma4:latest"

    def test_falls_back_to_model_key_when_model_name_absent(self):
        sink: dict = {}
        result = _message(usage={"input_tokens": 1, "output_tokens": 1}, metadata={"model": "gemma4:latest"})
        record_usage(result, sink)
        assert sink["model_name"] == "gemma4:latest"

    def test_missing_usage_metadata_leaves_sink_without_token_keys(self):
        sink: dict = {}
        result = _message(usage=None, metadata={"model_name": "gemma4:latest"})
        record_usage(result, sink)
        assert "input_tokens" not in sink
        assert "output_tokens" not in sink
        assert sink["model_name"] == "gemma4:latest"

    def test_missing_response_metadata_leaves_sink_without_model_name(self):
        sink: dict = {}
        result = SimpleNamespace(usage_metadata={"input_tokens": 1, "output_tokens": 1})
        record_usage(result, sink)
        assert "model_name" not in sink
        assert sink["input_tokens"] == 1
