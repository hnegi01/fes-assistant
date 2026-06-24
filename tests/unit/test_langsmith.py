"""
Unit tests for LangSmith / LiteLLM observability wiring.

Tests cover:
  - _configure_langsmith_tracing() correctly sets / clears litellm callbacks
"""

import litellm
import pytest

import backend.agent._config as config_m
import backend.agent.llm_agent as m


class TestConfigureLangsmithTracing:
    def test_callbacks_set_when_tracing_enabled(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_TRACING", "true")

        config_m._configure_langsmith_tracing()

        assert "langsmith" in litellm.success_callback
        assert "langsmith" in litellm.failure_callback

    def test_callbacks_cleared_when_tracing_disabled(self, monkeypatch):
        # Pre-set to simulate a previous enable
        litellm.success_callback = ["langsmith"]
        litellm.failure_callback = ["langsmith"]

        monkeypatch.setenv("LANGSMITH_TRACING", "false")

        config_m._configure_langsmith_tracing()

        assert "langsmith" not in litellm.success_callback
        assert "langsmith" not in litellm.failure_callback

    def test_callbacks_cleared_when_tracing_unset(self, monkeypatch):
        litellm.success_callback = ["langsmith"]
        litellm.failure_callback = ["langsmith"]

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

        config_m._configure_langsmith_tracing()

        assert "langsmith" not in litellm.success_callback
        assert "langsmith" not in litellm.failure_callback

    def test_case_insensitive_true(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_TRACING", "  TRUE  ")

        config_m._configure_langsmith_tracing()

        assert "langsmith" in litellm.success_callback

    def test_langchain_tracing_v2_does_not_enable(self, monkeypatch):
        # We only read LANGSMITH_TRACING now — the old LANGCHAIN_TRACING_V2 var
        # must not activate the callbacks (avoids silent legacy behaviour).
        litellm.success_callback = []
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

        config_m._configure_langsmith_tracing()

        assert "langsmith" not in litellm.success_callback

    @pytest.fixture(autouse=True)
    def reset_litellm_callbacks(self):
        """Restore litellm callback lists after each test."""
        orig_success = list(litellm.success_callback)
        orig_failure = list(litellm.failure_callback)
        yield
        litellm.success_callback = orig_success
        litellm.failure_callback = orig_failure


class TestCallLlmWithTools:
    def test_call_llm_with_tools_is_plain_coroutine(self):
        import inspect

        # Must be a plain async def with no @traceable wrapper.
        # @traceable would expose tenant_config/migration_config tokens to LangSmith.
        fn = m.call_llm_with_tools
        assert inspect.iscoroutinefunction(fn), "call_llm_with_tools must be async"
        assert not hasattr(fn, "__wrapped__"), (
            "call_llm_with_tools must NOT be wrapped by @traceable — "
            "the decorator captures all inputs including auth tokens"
        )
